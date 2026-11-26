"""PTX, SASS, NCU, and Cold-L2 profiling harness for MXFP8 decode GEMV
(DeepSeek-V4.1-Flash dense projections on SM90).

Compares two CuTe-DSL MXFP8 GEMV implementations (plus BF16 cuBLAS in bench):

  1. Before (`Mxfp8Gemv`): SM80 `cp.async` weight+scale SMEM staging + power-of-two
     config search (`_ampere_default_config`).
  2. After (`HopperMxfp8Gemv`): SM90 zero-weight-SMEM 2-/3-stage register-ring
     streaming via side-effect-ordered `ld.global.cg.v4.u32` + Hopper PDL
     (`griddepcontrol_wait` / `griddepcontrol_launch_dependents`) + wave-aware
     config search with non-power-of-two `warps_n=5, warps_k=3`.

Subcommands (`--mode`):
  - `dump`      : Compile Before (`Mxfp8Gemv`) & After (`HopperMxfp8Gemv`) for
                  representative shapes, verify accuracy against FP64 reference,
                  dump PTX/CUBIN/SASS, and print static PTX/SASS & resource tables.
  - `ncu`       : Warm up & verify, then launch Before/After pairs inside
                  `cudaProfilerStart` / `cudaProfilerStop` for Nsight Compute.
  - `parse-ncu` : Parse an `ncu --page raw --csv` export into SOL, pipe, memory,
                  and warp-stall CPI tables (CPU-only, no CUDA needed).
  - `bench`     : Measure Cold-L2 latency (`Mxfp8Gemv` vs `HopperMxfp8Gemv` vs
                  BF16 cuBLAS) across all 6 DeepSeek-V4.1-Flash shapes x M=1..48.

Example usage:
    python experiments/ops/cutedsl/mxfp8_gemv_profile.py --mode dump --out-dir /tmp/out
    ncu --profile-from-start off --set full --cache-control all --clock-control base \\
        -f -o /tmp/mxfp8_gemv_sm90 \\
        python experiments/ops/cutedsl/mxfp8_gemv_profile.py --mode ncu
    python experiments/ops/cutedsl/mxfp8_gemv_profile.py \\
        --mode parse-ncu --csv profile/ncu/mxfp8_gemv_sm90_raw.csv
    python experiments/ops/cutedsl/mxfp8_gemv_profile.py --mode bench
"""

import argparse
import csv
import glob
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import cache

# Per-rank [N, K] at TP8 for DeepSeek-V4.1-Flash dense projections.
ALL_SHAPES: list[tuple[str, int, int]] = [
    ("qkv_a", 1792, 5120),
    ("q_b", 4096, 1280),
    ("q_b_idx", 8192, 1280),
    ("wo_b", 5120, 1024),
    ("shared_w13", 576, 5120),
    ("shared_w2", 5120, 288),
]

# Representative (shape, N, K, M) cases for static PTX/SASS dump.
DUMP_CASES: list[tuple[str, int, int, int]] = [
    ("qkv_a", 1792, 5120, 24),
    ("q_b_idx", 8192, 1280, 24),
    ("wo_b", 5120, 1024, 24),
    ("wo_b", 5120, 1024, 48),
    ("shared_w2", 5120, 288, 24),
]

# Deterministic launch sequence for --mode ncu (7 cases x 2 variants = 14 launches).
NCU_CASES: list[tuple[str, int, int, int]] = [
    ("qkv_a", 1792, 5120, 6),
    ("qkv_a", 1792, 5120, 24),
    ("q_b_idx", 8192, 1280, 6),
    ("q_b_idx", 8192, 1280, 24),
    ("wo_b", 5120, 1024, 24),
    ("wo_b", 5120, 1024, 48),
    ("shared_w2", 5120, 288, 24),
]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Static PTX / SASS parsers (CPU-only, no CUDA import needed)
# ---------------------------------------------------------------------------


def parse_ptx_stats(path: str) -> tuple[int, Counter[str]]:
    with open(path) as f:
        text = f.read()
    m = re.search(r"\.visible\s+\.entry\s+\S+.*?\{(.*)\}", text, re.S)
    body = m.group(1) if m else text
    counts: Counter[str] = Counter()
    total = 0
    for raw in body.splitlines():
        line = raw.split("//")[0].strip()
        if not line or line.startswith((".", "{", "}")) or line.endswith(":"):
            continue
        line = re.sub(r"^@!?\S+\s+", "", line)
        tok = line.split()[0].rstrip(";")
        if not tok:
            continue
        total += 1
        if tok.startswith("mma.sync"):
            counts["mma.sync"] += 1
        elif tok.startswith("cp.async.wait_group"):
            counts["cp.wait"] += 1
        elif tok.startswith("cp.async.commit_group"):
            counts["cp.commit"] += 1
        elif tok.startswith("cp.async"):
            counts["cp.async"] += 1
        elif tok.startswith("ld.global.cg"):
            counts["ld.g.cg"] += 1
        elif tok.startswith("ld.global.nc"):
            counts["ld.g.nc"] += 1
        elif tok.startswith("ld.global"):
            counts["ld.g"] += 1
        elif tok.startswith("st.global"):
            counts["st.g"] += 1
        elif tok.startswith("ld.shared"):
            counts["ld.s"] += 1
        elif tok.startswith("st.shared"):
            counts["st.s"] += 1
        elif tok.startswith(("ld.local", "st.local")):
            counts["local"] += 1
        elif tok.startswith("griddepcontrol."):
            counts["pdl"] += 1
        elif tok.startswith(("bar.sync", "barrier.")):
            counts["bar.sync"] += 1
        elif tok.startswith("fma.rn.bf16x2"):
            counts["hfma2"] += 1
        elif tok.startswith("atom."):
            counts["atom"] += 1
        elif tok.startswith("fence."):
            counts["fence"] += 1
    return total, counts


def parse_sass_stats(path: str) -> tuple[int, Counter[str]]:
    with open(path) as f:
        text = f.read()
    counts: Counter[str] = Counter()
    total = 0
    for line in text.splitlines():
        m = re.match(
            r"\s*/\*[0-9a-f]+\*/\s+(?:@!?\S+\s+)?([A-Z0-9_]+(?:\.[A-Z0-9_]+)*)",
            line,
        )
        if not m:
            continue
        base = m.group(1).split(".")[0]
        total += 1
        if base in ("HMMA", "IMMA", "QMMA"):
            counts["HMMA"] += 1
        elif base == "LDGSTS":
            counts["LDGSTS"] += 1
        elif base == "LDGDEPBAR":
            counts["LDGDEPBAR"] += 1
        elif base == "LDG":
            counts["LDG"] += 1
        elif base == "STG":
            counts["STG"] += 1
        elif base == "LDS":
            counts["LDS"] += 1
        elif base == "STS":
            counts["STS"] += 1
        elif base in ("LDL", "STL"):
            counts["LDL/STL"] += 1
        elif base == "HFMA2":
            counts["HFMA2"] += 1
        elif base in ("BAR", "DEPBAR", "MEMBAR", "ERRBAR"):
            counts["BAR/MEM"] += 1
        elif base in ("ACQBULK", "PREEXIT", "SYNCS", "ENDCOLLECTIVE", "GRIDDEPCONTROL"):
            counts["PDL"] += 1
        elif base in ("ATOMG", "REDG"):
            counts["ATOM"] += 1
    return total, counts


# ---------------------------------------------------------------------------
# NCU raw CSV parser (CPU-only)
# ---------------------------------------------------------------------------


def parse_ncu_csv(csv_path: str) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    headers, units, data_rows = rows[0], rows[1], rows[2:]
    col = {h: i for i, h in enumerate(headers)}

    def getf(r: list[str], key: str, default: float = 0.0) -> float:
        if key not in col:
            return default
        v = r[col[key]].replace(",", "").strip()
        if not v or v in ("n/a", "nan"):
            return default
        return float(v)

    def scale_dur_us(r: list[str]) -> float:
        key = "gpu__time_duration.sum"
        dur = getf(r, key)
        if key not in col:
            return 0.0
        unit = units[col[key]]
        if unit in ("nsecond", "ns"):
            return dur / 1000.0
        if unit in ("msecond", "ms"):
            return dur * 1000.0
        return dur

    def scale_dram_gbs(r: list[str]) -> float:
        key = "dram__bytes.sum.per_second"
        val = getf(r, key)
        if key not in col:
            return 0.0
        unit = units[col[key]]
        if unit in ("byte/second", "byte/s"):
            return val / 1e9
        if unit in ("Kbyte/second", "Kbyte/s"):
            return val / 1e6
        if unit in ("Mbyte/second", "Mbyte/s"):
            return val / 1e3
        if unit in ("Tbyte/second", "Tbyte/s"):
            return val * 1e3
        return val

    def scale_bytes_kb(r: list[str], key: str) -> float:
        val = getf(r, key)
        if key not in col:
            return 0.0
        unit = units[col[key]]
        if unit.startswith("byte"):
            return val / 1024.0
        if unit.startswith("Mbyte"):
            return val * 1024.0
        if unit.startswith("Gbyte"):
            return val * 1024.0 * 1024.0
        return val

    def label_for(idx: int, kname: str) -> tuple[str, str]:
        variant = "Hopper" if "Hopper" in kname or "hopper" in kname else "Ampere"
        if idx // 2 < len(NCU_CASES):
            name, n, k, m = NCU_CASES[idx // 2]
            case = f"{name} M={m}"
        else:
            case = f"launch#{idx}"
        return case, variant

    print("=== TABLE 1: NCU OVERVIEW, OCCUPANCY & COMPUTE PIPES ===")
    hdr1 = (
        f"{'Case':<16} | {'Kernel':<6} | {'Grid':<11} | {'Thr':>3} | {'Dur(us)':>7} | "
        f"{'SM%':>5} | {'Tensor%':>7} | {'LSU%':>5} | {'DRAM GB/s':>9} | "
        f"{'Reg':>3} | {'SMEM(KB)':>8} | {'ThOcc%':>6} | {'AchOcc%':>7} | {'Iss%':>5}"
    )
    print(hdr1)
    print("-" * len(hdr1))
    for idx, r in enumerate(data_rows):
        case, var = label_for(idx, r[col["Kernel Name"]])
        grid = r[col["Grid Size"]].replace(" ", "")
        bsize = r[col["Block Size"]].strip("()").split(",")[0].strip()
        tensor = getf(
            r, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        )
        lsu = getf(r, "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active")
        smem_kb = scale_bytes_kb(r, "launch__shared_mem_per_block_allocated")
        print(
            f"{case:<16} | {var:<6} | {grid:<11} | {bsize:>3} | "
            f"{scale_dur_us(r):>7.2f} | "
            f"{getf(r, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'):>5.1f} | "
            f"{tensor:>7.1f} | {lsu:>5.1f} | {scale_dram_gbs(r):>9.1f} | "
            f"{int(getf(r, 'launch__registers_per_thread')):>3} | {smem_kb:>8.1f} | "
            f"{getf(r, 'sm__maximum_warps_per_active_cycle_pct'):>6.1f} | "
            f"{getf(r, 'sm__warps_active.avg.pct_of_peak_sustained_active'):>7.1f} | "
            f"{getf(r, 'smsp__issue_active.avg.pct_of_peak_sustained_active'):>5.1f}"
        )

    print("\n=== TABLE 2: MEMORY INSTRUCTIONS & CACHE TRAFFIC ===")
    hdr2 = (
        f"{'Case':<16} | {'Kernel':<6} | {'Gld':>7} | {'Ldgsts':>7} | "
        f"{'Sld':>7} | {'Sst':>7} | {'SmemWF':>8} | {'Loc':>3} | "
        f"{'Inst(K)':>7} | {'L2Hit%':>6} | {'DRAM_R(MB)':>10}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for idx, r in enumerate(data_rows):
        case, var = label_for(idx, r[col["Kernel Name"]])
        gld = getf(r, "smsp__sass_inst_executed_op_global_ld.sum")
        ldgsts = getf(r, "smsp__inst_executed_op_ldgsts.sum")
        sld = getf(r, "smsp__sass_inst_executed_op_shared_ld.sum")
        sst = getf(r, "smsp__sass_inst_executed_op_shared_st.sum")
        smem_wf = getf(r, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")
        loc = getf(r, "smsp__sass_inst_executed_op_local_ld.sum") + getf(
            r, "smsp__sass_inst_executed_op_local_st.sum"
        )
        inst_k = getf(r, "smsp__inst_executed.sum") / 1e3
        l2_hit = getf(r, "lts__t_sector_hit_rate.pct")
        dram_r_mb = scale_bytes_kb(r, "dram__bytes_read.sum") / 1024.0
        print(
            f"{case:<16} | {var:<6} | {gld:>7.0f} | {ldgsts:>7.0f} | "
            f"{sld:>7.0f} | {sst:>7.0f} | {smem_wf:>8.0f} | {loc:>3.0f} | "
            f"{inst_k:>7.1f} | {l2_hit:>6.1f} | {dram_r_mb:>10.2f}"
        )

    stalls = [
        ("long_sb", "long_scoreboard"),
        ("short_sb", "short_scoreboard"),
        ("wait", "wait"),
        ("barrier", "barrier"),
        ("lg_thr", "lg_throttle"),
        ("mio_thr", "mio_throttle"),
        ("math_thr", "math_pipe_throttle"),
        ("not_sel", "not_selected"),
        ("selected", "selected"),
    ]
    print("\n=== TABLE 3: WARP STALL REASONS (Cycles / Issued Instruction) ===")
    hdr3 = f"{'Case':<16} | {'Kernel':<6} | " + " | ".join(
        f"{label:>8}" for label, _ in stalls
    )
    print(hdr3)
    print("-" * len(hdr3))
    for idx, r in enumerate(data_rows):
        case, var = label_for(idx, r[col["Kernel Name"]])
        vals = [
            getf(
                r,
                f"smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio",
            )
            for _, name in stalls
        ]
        print(f"{case:<16} | {var:<6} | " + " | ".join(f"{v:>8.2f}" for v in vals))


# ---------------------------------------------------------------------------
# Standalone GPU Kernels (lazy-imported so parse-ncu runs on CPU-only hosts)
# ---------------------------------------------------------------------------


def _build_gpu_module():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass import Float32, Int32, Int64, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_fake_stream
    from cutlass.cutlass_dsl import T, dsl_user_op

    _SmemAllocator = getattr(cutlass, "memory", cutlass.utils).SmemAllocator

    TILE_ROWS = 16
    GROUP = 32
    BLOCK_BYTES = TILE_ROWS * GROUP
    LANE_BYTES = BLOCK_BYTES // 32
    X_TILE = 8
    MAX_SCALE_CODE = 134
    COUNTERS = 4096
    NEGATIVE_ZERO_BF16X2 = 0x80008000
    WEIGHT_COPY = "cp.async.cg.shared.global"

    X_COST = 1.0
    SPLIT_COST = 32 << 10
    BLOCK_COST = 2 << 10
    MAX_SPLIT_READ = 96 << 10

    HOPPER_WARPS_PER_SM = 24
    HOPPER_MAX_RESIDENCY = 4
    HOPPER_X_COST = 0.25
    HOPPER_BLOCK_COST = 1 << 10
    HOPPER_SPLIT_COST = 48 << 10
    HOPPER_SPLIT_STEP_COST = 6 << 10

    def current_cu_stream() -> cuda.CUstream:
        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def _asm(result, operands, text, constraints, side_effects, loc, ip):
        return llvm.inline_asm(
            result,
            operands,
            text,
            constraints,
            has_side_effects=side_effects,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    def _words(loaded, count: int, kind, wrap, loc, ip) -> list:
        return [
            wrap(llvm.extractvalue(kind, loaded, [i], loc=loc, ip=ip))
            for i in range(count)
        ]

    @dsl_user_op
    def load_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            _asm(
                T.i32(),
                [Int64(address).ir_value(loc=loc, ip=ip)],
                "ld.global.nc.u32 $0, [$1];",
                "=r,l",
                False,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def store_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "st.global.u32 [$0], $1;",
            "l,r",
            True,
            loc,
            ip,
        )

    @dsl_user_op
    def fma_bf16x2(a: Uint32, b: Uint32, c: Uint32, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            _asm(
                T.i32(),
                [
                    Uint32(a).ir_value(loc=loc, ip=ip),
                    Uint32(b).ir_value(loc=loc, ip=ip),
                    Uint32(c).ir_value(loc=loc, ip=ip),
                ],
                "fma.rn.bf16x2 $0, $1, $2, $3;",
                "=r,r,r,r",
                False,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def fence_gpu(*, loc=None, ip=None) -> None:
        _asm(None, [], "fence.acq_rel.gpu;", "", True, loc, ip)

    @dsl_user_op
    def ticket(address: Int64, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            _asm(
                T.i32(),
                [
                    Int64(address).ir_value(loc=loc, ip=ip),
                    Uint32(1).ir_value(loc=loc, ip=ip),
                ],
                "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
                "=r,l,r",
                True,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def pack_bf16x2(low: Float32, high: Float32, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            _asm(
                T.i32(),
                [
                    Float32(high).ir_value(loc=loc, ip=ip),
                    Float32(low).ir_value(loc=loc, ip=ip),
                ],
                "cvt.rn.bf16x2.f32 $0, $1, $2;",
                "=r,f,f",
                False,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def copy_async_weights(shared: Int32, address: Int64, *, loc=None, ip=None):
        _asm(
            None,
            [
                Int32(shared).ir_value(loc=loc, ip=ip),
                Int64(address).ir_value(loc=loc, ip=ip),
            ],
            f"{WEIGHT_COPY} [$0], [$1], 16;",
            "r,l",
            True,
            loc,
            ip,
        )

    @dsl_user_op
    def copy_async_word(shared: Int32, address: Int64, *, loc=None, ip=None):
        _asm(
            None,
            [
                Int32(shared).ir_value(loc=loc, ip=ip),
                Int64(address).ir_value(loc=loc, ip=ip),
            ],
            "cp.async.ca.shared.global [$0], [$1], 4;",
            "r,l",
            True,
            loc,
            ip,
        )

    @dsl_user_op
    def load_shared_word(shared: Int32, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            _asm(
                T.i32(),
                [Int32(shared).ir_value(loc=loc, ip=ip)],
                "ld.shared.u32 $0, [$1];",
                "=r,r",
                True,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def copy_async(shared: Int32, address: Int64, size: Int32, *, loc=None, ip=None):
        _asm(
            None,
            [
                Int32(shared).ir_value(loc=loc, ip=ip),
                Int64(address).ir_value(loc=loc, ip=ip),
                Int32(size).ir_value(loc=loc, ip=ip),
            ],
            "cp.async.cg.shared.global [$0], [$1], 16, $2;",
            "r,l,r",
            True,
            loc,
            ip,
        )

    @dsl_user_op
    def load_shared(shared: Int32, *, loc=None, ip=None) -> list:
        loaded = _asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [Int32(shared).ir_value(loc=loc, ip=ip)],
            "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,r",
            True,
            loc,
            ip,
        )
        return _words(loaded, 4, T.i32(), Uint32, loc, ip)

    @dsl_user_op
    def mma_bf16(c: list, a: list, b: list, *, loc=None, ip=None) -> list:
        loaded = _asm(
            llvm.StructType.get_literal([T.f32()] * 4),
            [Uint32(v).ir_value(loc=loc, ip=ip) for v in (*a, *b)]
            + [Float32(v).ir_value(loc=loc, ip=ip) for v in c],
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
            "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
            False,
            loc,
            ip,
        )
        return _words(loaded, 4, T.f32(), Float32, loc, ip)

    @dsl_user_op
    def load_weights(address: Int64, *, loc=None, ip=None) -> list:
        loaded = _asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            True,
            loc,
            ip,
        )
        return _words(loaded, 4, T.i32(), Uint32, loc, ip)

    @dsl_user_op
    def load_partials(address: Int64, *, loc=None, ip=None) -> list:
        loaded = _asm(
            llvm.StructType.get_literal([T.f32()] * 4),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
            "=f,=f,=f,=f,l",
            True,
            loc,
            ip,
        )
        return _words(loaded, 4, T.f32(), Float32, loc, ip)

    @dsl_user_op
    def store_partials(address: Int64, values: list, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [Int64(address).ir_value(loc=loc, ip=ip)]
            + [Float32(v).ir_value(loc=loc, ip=ip) for v in values],
            "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
            "l,f,f,f,f",
            True,
            loc,
            ip,
        )

    @dsl_user_op
    def store_pair(address: Int64, low: Uint32, high: Uint32, *, loc=None, ip=None):
        _asm(
            None,
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(low).ir_value(loc=loc, ip=ip),
                Uint32(high).ir_value(loc=loc, ip=ip),
            ],
            "st.global.v2.u32 [$0], {$1, $2};",
            "l,r,r",
            True,
            loc,
            ip,
        )

    @cute.jit
    def decode_pairs(word: Uint32, scale: Uint32) -> tuple:
        even = ((word << 8) & 0x80008000) | ((word << 4) & 0x07F007F0)
        odd = (word & 0x80008000) | ((word >> 4) & 0x07F007F0)
        negative_zero = Uint32(NEGATIVE_ZERO_BF16X2)
        return (
            fma_bf16x2(even, scale, negative_zero),
            fma_bf16x2(odd, scale, negative_zero),
        )

    @dataclass(frozen=True)
    class GemvConfig:
        warps_n: int
        warps_k: int
        groups: int

        def splits(self, k: int) -> int:
            return k // GROUP // (self.warps_k * self.groups)

    class Mxfp8Gemv:
        """Before: SM80 `cp.async` weight+scale SMEM staging kernel."""

        _stage_weights = True
        _use_pdl = False

        def __init__(self, n: int, k: int, x_tiles: int, config: GemvConfig):
            self.n = n
            self.k = k
            self.x_tiles = x_tiles
            self.warps_n = config.warps_n
            self.warps_k = config.warps_k
            self.groups = config.groups
            self.cta_groups = self.warps_k * self.groups
            self.splits = config.splits(k)
            self.row_blocks = n // TILE_ROWS // self.warps_n
            self.threads = 32 * self.warps_n * self.warps_k
            self.cta_rows = TILE_ROWS * self.warps_n
            self.cta_cols = GROUP * self.cta_groups
            self.x_rows = X_TILE * x_tiles
            words = self.cta_cols // 2
            self.row_words = words if words % 32 == 16 else words + 16
            self.x_words = self.x_rows * self.row_words
            self.red_stride = self.cta_rows + 4
            if self._stage_weights:
                self.w_words = self.threads // 32 * self.groups * (BLOCK_BYTES // 4)
                self.scale_rows = self.warps_n // 2 + 1
                self.buf_words = max(
                    self.x_words + self.w_words,
                    self.warps_k * self.x_rows * self.red_stride,
                )
            else:
                self.w_words = 0
                self.scale_rows = 0
                self.buf_words = max(
                    self.x_words,
                    self.warps_k * self.x_rows * self.red_stride,
                )

        def _storage_type(self):
            @cute.struct
            class Storage:
                buf: cute.struct.Align[cute.struct.MemRange[Uint32, self.buf_words], 16]
                scale: cute.struct.Align[
                    cute.struct.MemRange[Uint32, self.scale_rows * self.cta_groups], 16
                ]
                last: cute.struct.MemRange[Int32, 1]

            return Storage

        def smem_bytes(self) -> int:
            return (self.buf_words + self.scale_rows * self.cta_groups) * 4 + 64

        @cute.jit
        def __call__(
            self,
            weight: Int64,
            scales: Int64,
            x: Int64,
            y: Int64,
            partials: Int64,
            counters: Int64,
            rows: Int32,
            x_stride: Int32,
            y_stride: Int32,
            stream: cuda.CUstream,
        ):
            Storage = self._storage_type()
            self.ampere_kernel(
                weight,
                scales,
                x,
                y,
                partials,
                counters,
                rows,
                x_stride,
                y_stride,
                Storage,
            ).launch(
                grid=(self.row_blocks, self.splits, 1),
                block=[self.threads, 1, 1],
                smem=Storage.size_in_bytes(),  # type: ignore[attr-defined]
                stream=stream,
            )

        @cute.jit
        def _load_x(
            self,
            x_base: Int32,
            x: Int64,
            rows: Int32,
            x_stride: Int32,
            split: Int32,
            tid: Int32,
        ):
            vectors = self.cta_cols // 8
            total = self.x_rows * vectors
            for it in cutlass.range_constexpr(
                (total + self.threads - 1) // self.threads
            ):
                c = tid + it * self.threads
                if c < total:
                    r = c // vectors
                    v = c % vectors
                    source = x
                    size = Int32(0)
                    if r < rows:
                        source = (
                            x
                            + (
                                Int64(r) * Int64(x_stride)
                                + Int64(split * self.cta_cols + v * 8)
                            )
                            * 2
                        )
                        size = Int32(16)
                    copy_async(x_base + (r * self.row_words + v * 4) * 4, source, size)

        @cute.jit
        def _decode_and_mma(
            self,
            raw: list,
            scale: Uint32,
            x_base: Int32,
            acc: cute.Tensor,
            column: Int32,
            g: Int32,
        ):
            a = []
            for j in cutlass.range_constexpr(4):
                even, odd = decode_pairs(raw[j], scale)
                a.append(even)
                a.append(odd)
            for t in cutlass.range_constexpr(self.x_tiles):
                b = load_shared(
                    x_base + ((t * X_TILE + g) * self.row_words + column) * 4
                )
                c = [acc[t, j] for j in range(4)]
                c = mma_bf16(c, a[0:4], b[0:2])
                c = mma_bf16(c, a[4:8], b[2:4])
                for j in cutlass.range_constexpr(4):
                    acc[t, j] = c[j]

        @cute.jit
        def _epilogue(
            self,
            red: cute.Tensor,
            last: cute.Tensor,
            acc: cute.Tensor,
            y: Int64,
            partials: Int64,
            counters: Int64,
            rows: Int32,
            y_stride: Int32,
            tid: Int32,
            block: Int32,
            split: Int32,
            wn: Int32,
            wk: Int32,
            g: Int32,
            q: Int32,
        ):
            for t in cutlass.range_constexpr(self.x_tiles):
                for j in cutlass.range_constexpr(4):
                    x_row = t * X_TILE + 2 * q + j % 2
                    red[wk, x_row, wn * TILE_ROWS + g + 8 * (j // 2)] = acc[t, j]
            cute.arch.barrier()

            quads = self.cta_rows // 4
            total = self.x_rows * quads
            trips = (total + self.threads - 1) // self.threads
            n0 = block * self.cta_rows

            if cutlass.const_expr(self._use_pdl and self.splits == 1):
                cute.arch.griddepcontrol_launch_dependents()

            for it in cutlass.range_constexpr(trips):
                c = tid + it * self.threads
                r = c // quads
                v = (c % quads) * 4
                if c < total and r < rows:
                    sums = [red[0, r, v + e] for e in range(4)]
                    for w in cutlass.range_constexpr(1, self.warps_k):
                        for e in cutlass.range_constexpr(4):
                            sums[e] = sums[e] + red[w, r, v + e]
                    if cutlass.const_expr(self.splits == 1):
                        store_pair(
                            y + (Int64(r) * Int64(y_stride) + Int64(n0 + v)) * 2,
                            pack_bf16x2(sums[0], sums[1]),
                            pack_bf16x2(sums[2], sums[3]),
                        )
                    else:
                        store_partials(
                            partials
                            + (
                                (Int64(split) * self.x_rows + r) * self.n
                                + Int64(n0 + v)
                            )
                            * 4,
                            sums,
                        )

            if cutlass.const_expr(self.splits > 1):
                fence_gpu()
                cute.arch.barrier()
                counter = counters + Int64(block) * 4
                if tid == 0:
                    last[0] = Int32(0)
                    if ticket(counter) == Uint32(self.splits - 1):
                        last[0] = Int32(1)
                cute.arch.barrier()
                if last[0] == 1:
                    if cutlass.const_expr(self._use_pdl):
                        cute.arch.griddepcontrol_launch_dependents()
                    fence_gpu()
                    for it in cutlass.range_constexpr(trips):
                        c = tid + it * self.threads
                        r = c // quads
                        v = (c % quads) * 4
                        if c < total and r < rows:
                            base = partials + (Int64(r) * self.n + Int64(n0 + v)) * 4
                            loaded = [
                                load_partials(
                                    base + Int64(s * self.x_rows * self.n * 4)
                                )
                                for s in range(self.splits)
                            ]
                            sums = loaded[0]
                            for s in cutlass.range_constexpr(1, self.splits):
                                for e in cutlass.range_constexpr(4):
                                    sums[e] = sums[e] + loaded[s][e]
                            store_pair(
                                y + (Int64(r) * Int64(y_stride) + Int64(n0 + v)) * 2,
                                pack_bf16x2(sums[0], sums[1]),
                                pack_bf16x2(sums[2], sums[3]),
                            )
                    if tid == 0:
                        store_u32(counter, Uint32(0))

        @cute.jit
        def _accumulate(
            self,
            weight: Int64,
            scales: Int64,
            x: Int64,
            x_base: Int32,
            scale_base: Int32,
            acc: cute.Tensor,
            rows: Int32,
            x_stride: Int32,
            tid: Int32,
            block: Int32,
            split: Int32,
            warp: Int32,
            lane: Int32,
            wk: Int32,
            g: Int32,
            q: Int32,
            tile: Int32,
            first: Int32,
        ):
            groups = self.groups
            k_groups = self.k // GROUP
            self._load_x(x_base, x, rows, x_stride, split, tid)
            scale_row0 = block * self.warps_n // 2
            total = self.scale_rows * self.cta_groups
            for it in cutlass.range_constexpr(
                (total + self.threads - 1) // self.threads
            ):
                c = tid + it * self.threads
                if c < total:
                    r = cutlass.min(
                        scale_row0 + c // self.cta_groups, self.n // GROUP - 1
                    )
                    column = split * self.cta_groups + c % self.cta_groups
                    copy_async_word(
                        scale_base + c * 4,
                        scales + (Int64(r) * k_groups + Int64(column)) * 4,
                    )
            cute.arch.cp_async_commit_group()

            w_lane = (
                x_base
                + (self.x_words + warp * groups * (BLOCK_BYTES // 4)) * 4
                + lane * LANE_BYTES
            )
            address = (
                weight
                + (Int64(tile) * k_groups + Int64(first)) * BLOCK_BYTES
                + Int64(lane * LANE_BYTES)
            )
            for i in cutlass.range_constexpr(groups):
                copy_async_weights(w_lane + i * BLOCK_BYTES, address + i * BLOCK_BYTES)
                cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(groups)
            cute.arch.barrier()

            acc.fill(0.0)
            scale_row = (tile // 2 - scale_row0) * self.cta_groups + wk * groups
            for i in cutlass.range_constexpr(groups):
                cute.arch.cp_async_wait_group(groups - 1 - i)
                raw = load_shared(w_lane + i * BLOCK_BYTES)
                scale = load_shared_word(scale_base + (scale_row + i) * 4)
                column = (wk * groups + i) * 16 + q * 4
                self._decode_and_mma(raw, scale, x_base, acc, column, g)

        @cute.jit
        def _run_body(
            self,
            weight: Int64,
            scales: Int64,
            x: Int64,
            y: Int64,
            partials: Int64,
            counters: Int64,
            rows: Int32,
            x_stride: Int32,
            y_stride: Int32,
            Storage: cutlass.Constexpr,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, split, _ = cute.arch.block_idx()
            warp = tid // 32
            lane = tid % 32
            wn = warp % self.warps_n
            wk = warp // self.warps_n
            g = lane // 4
            q = lane % 4
            tile = block * self.warps_n + wn
            first = split * self.cta_groups + wk * self.groups

            smem = _SmemAllocator()
            storage = smem.allocate(Storage)
            x_base = Int32(storage.buf.data_ptr().toint())

            acc = cute.make_rmem_tensor(cute.make_layout((self.x_tiles, 4)), Float32)
            scale_base = Int32(0)
            if cutlass.const_expr(self._stage_weights):
                scale_base = Int32(storage.scale.data_ptr().toint())
            self._accumulate(
                weight,
                scales,
                x,
                x_base,
                scale_base,
                acc,
                rows,
                x_stride,
                tid,
                block,
                split,
                warp,
                lane,
                wk,
                g,
                q,
                tile,
                first,
            )
            cute.arch.barrier()

            red = storage.buf.get_tensor(
                cute.make_layout(
                    (self.warps_k, self.x_rows, self.cta_rows),
                    stride=(self.x_rows * self.red_stride, self.red_stride, 1),
                ),
                dtype=Float32,
            )
            last = storage.last.get_tensor(cute.make_layout(1))
            self._epilogue(
                red,
                last,
                acc,
                y,
                partials,
                counters,
                rows,
                y_stride,
                tid,
                block,
                split,
                wn,
                wk,
                g,
                q,
            )

        @cute.kernel
        def ampere_kernel(
            self,
            weight: Int64,
            scales: Int64,
            x: Int64,
            y: Int64,
            partials: Int64,
            counters: Int64,
            rows: Int32,
            x_stride: Int32,
            y_stride: Int32,
            Storage: cutlass.Constexpr,
        ):
            self._run_body(
                weight,
                scales,
                x,
                y,
                partials,
                counters,
                rows,
                x_stride,
                y_stride,
                Storage,
            )

    class HopperMxfp8Gemv(Mxfp8Gemv):
        """After: SM90 zero-weight-SMEM register-streaming + Hopper PDL kernel."""

        _stage_weights = False
        _use_pdl = True

        def _storage_type(self):
            @cute.struct
            class Storage:
                buf: cute.struct.Align[cute.struct.MemRange[Uint32, self.buf_words], 16]
                last: cute.struct.MemRange[Int32, 1]

            return Storage

        def smem_bytes(self) -> int:
            return (self.buf_words * 4 + 4 + 15) & ~15

        @cute.jit
        def __call__(
            self,
            weight: Int64,
            scales: Int64,
            x: Int64,
            y: Int64,
            partials: Int64,
            counters: Int64,
            rows: Int32,
            x_stride: Int32,
            y_stride: Int32,
            stream: cuda.CUstream,
        ):
            Storage = self._storage_type()
            self.hopper_kernel(
                weight,
                scales,
                x,
                y,
                partials,
                counters,
                rows,
                x_stride,
                y_stride,
                Storage,
            ).launch(
                grid=(self.row_blocks, self.splits, 1),
                block=[self.threads, 1, 1],
                smem=Storage.size_in_bytes(),  # type: ignore[attr-defined]
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def hopper_kernel(
            self,
            weight: Int64,
            scales: Int64,
            x: Int64,
            y: Int64,
            partials: Int64,
            counters: Int64,
            rows: Int32,
            x_stride: Int32,
            y_stride: Int32,
            Storage: cutlass.Constexpr,
        ):
            self._run_body(
                weight,
                scales,
                x,
                y,
                partials,
                counters,
                rows,
                x_stride,
                y_stride,
                Storage,
            )

        @cute.jit
        def _accumulate(
            self,
            weight: Int64,
            scales: Int64,
            x: Int64,
            x_base: Int32,
            scale_base: Int32,
            acc: cute.Tensor,
            rows: Int32,
            x_stride: Int32,
            tid: Int32,
            block: Int32,
            split: Int32,
            warp: Int32,
            lane: Int32,
            wk: Int32,
            g: Int32,
            q: Int32,
            tile: Int32,
            first: Int32,
        ):
            groups = self.groups
            k_groups = self.k // GROUP

            weight_base = (
                weight
                + (Int64(tile) * k_groups + Int64(first)) * BLOCK_BYTES
                + Int64(lane) * LANE_BYTES
            )
            scale_row = tile // 2
            global_scale_base = (
                scales + (Int64(scale_row) * k_groups + Int64(first)) * 4
            )
            prefetch_stages = min(groups, 2 if self.x_tiles >= 4 else 3)
            weight_buffer = cute.make_rmem_tensor(
                cute.make_layout((prefetch_stages, 4)), Uint32
            )
            scale_buffer = cute.make_rmem_tensor(
                cute.make_layout(prefetch_stages), Uint32
            )
            for stage in cutlass.range_constexpr(prefetch_stages):
                scale_buffer[stage] = load_u32(global_scale_base + Int64(stage) * 4)
                loaded_weights = load_weights(weight_base + Int64(stage) * BLOCK_BYTES)
                for j in cutlass.range_constexpr(4):
                    weight_buffer[stage, j] = loaded_weights[j]

            cute.arch.griddepcontrol_wait()

            self._load_x(x_base, x, rows, x_stride, split, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            acc.fill(0.0)
            for i in cutlass.range_constexpr(groups):
                stage = i % prefetch_stages
                scale = scale_buffer[stage]
                raw = [weight_buffer[stage, j] for j in range(4)]

                if cutlass.const_expr(i + prefetch_stages < groups):
                    next_group = i + prefetch_stages
                    scale_buffer[stage] = load_u32(
                        global_scale_base + Int64(next_group) * 4
                    )
                    loaded_weights = load_weights(
                        weight_base + Int64(next_group) * BLOCK_BYTES
                    )
                    for j in cutlass.range_constexpr(4):
                        weight_buffer[stage, j] = loaded_weights[j]

                column = (wk * groups + i) * 16 + q * 4
                self._decode_and_mma(raw, scale, x_base, acc, column, g)

    def pack_weight(
        weight: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, k = weight.shape
        codes = weight.contiguous().view(torch.uint8)
        blocks = scales.view(torch.uint8)
        assert int(blocks.max()) <= MAX_SCALE_CODE
        packed = (
            codes.view(n // TILE_ROWS, 2, 8, k // GROUP, 4, 4, 2)
            .permute(0, 3, 2, 4, 5, 6, 1)
            .contiguous()
            .view(n // TILE_ROWS, k // GROUP, BLOCK_BYTES)
        )
        factors = (blocks.to(torch.int32) + 120) * 0x00800080
        return packed, factors.contiguous()

    def dequant_bf16(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        n, k = codes.shape
        w = codes.view(torch.float8_e4m3fn).float().view(n // 32, 32, k // 32, 32)
        s = torch.exp2(scales.float() - 127.0).view(n // 32, 1, k // 32, 1)
        return (w * s).view(n, k).to(torch.bfloat16).contiguous()

    def make_weights(n: int, k: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        codes = torch.randint(
            0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=g
        )
        codes[(codes & 0x7F) == 0x7F] -= 1
        scales = torch.randint(
            112, 123, (n // 32, k // 32), dtype=torch.uint8, device="cuda", generator=g
        )
        packed, factors = pack_weight(codes, scales)
        return packed, factors, dequant_bf16(codes, scales)

    def device_limits(device_index: int = 0) -> tuple[int, int]:
        props = torch.cuda.get_device_properties(device_index)
        return props.multi_processor_count, props.shared_memory_per_block_optin

    def ampere_default_config(
        n: int, k: int, rows: int, sms: int, smem_limit: int
    ) -> GemvConfig:
        x_tiles = (rows + X_TILE - 1) // X_TILE
        tiles = n // TILE_ROWS
        k_groups = k // GROUP
        candidates = []
        for warps_n in (1, 2, 4, 8, 16):
            for warps_k in (1, 2, 4, 8, 16):
                if tiles % warps_n or warps_n * warps_k > 16:
                    continue
                for groups in range(1, 17):
                    if k_groups % (warps_k * groups):
                        continue
                    cfg = GemvConfig(warps_n, warps_k, groups)
                    kernel = Mxfp8Gemv(n, k, x_tiles, cfg)
                    split_read = kernel.splits * kernel.cta_rows * kernel.x_rows * 4
                    if kernel.smem_bytes() > smem_limit or (
                        kernel.splits > 1 and split_read > MAX_SPLIT_READ
                    ):
                        continue
                    ctas = kernel.row_blocks * kernel.splits
                    busiest = -(-ctas // sms)
                    cost = busiest * (
                        kernel.cta_rows * kernel.cta_cols
                        + X_COST * 2 * rows * kernel.cta_cols
                        + BLOCK_COST * groups
                    )
                    if kernel.splits > 1:
                        cost += SPLIT_COST + kernel.splits * kernel.cta_rows * rows * 4
                    fills = sms // 2 <= ctas <= 4 * sms
                    candidates.append((not fills, cost, cfg))
        assert candidates
        return min(candidates, key=lambda c: (c[0], c[1]))[2]

    def hopper_default_config(
        n: int, k: int, x_tiles: int, sms: int, smem_limit: int
    ) -> GemvConfig:
        rows = x_tiles * X_TILE
        tiles = n // TILE_ROWS
        k_groups = k // GROUP
        min_ctas = sms if x_tiles == 1 else (3 * sms) // 4
        candidates = []
        for warps_n in (1, 2, 4, 5, 8, 16):
            for warps_k in (1, 2, 3, 4, 8, 16):
                warps = warps_n * warps_k
                if tiles % warps_n or warps > 16:
                    continue
                for groups in range(1, 17):
                    if k_groups % (warps_k * groups):
                        continue
                    cfg = GemvConfig(warps_n, warps_k, groups)
                    kernel = HopperMxfp8Gemv(n, k, x_tiles, cfg)
                    smem = kernel.smem_bytes()
                    split_read = kernel.splits * kernel.cta_rows * kernel.x_rows * 4
                    if smem > smem_limit or (
                        kernel.splits > 1 and split_read > MAX_SPLIT_READ
                    ):
                        continue
                    ctas = kernel.row_blocks * kernel.splits
                    warp_residency = max(1, HOPPER_WARPS_PER_SM // warps)
                    smem_residency = max(1, smem_limit // max(smem, 1))
                    residency = min(
                        HOPPER_MAX_RESIDENCY, warp_residency, smem_residency
                    )
                    wave_capacity = sms * residency
                    waves = -(-ctas // wave_capacity)
                    busiest = -(-ctas // sms)
                    sm_penalty = 0 if ctas >= min_ctas else (min_ctas - ctas) * 2048
                    tail_penalty = (
                        0
                        if ctas <= wave_capacity
                        else (waves * wave_capacity - ctas) * 512
                        + waves * (waves - 1) * 32768
                    )
                    cost = (
                        busiest
                        * (
                            kernel.cta_rows * kernel.cta_cols
                            + HOPPER_X_COST * 2 * rows * kernel.cta_cols
                            + HOPPER_BLOCK_COST * groups
                        )
                        + sm_penalty
                        + tail_penalty
                    )
                    if kernel.splits > 1:
                        cost += (
                            HOPPER_SPLIT_COST
                            + kernel.splits * HOPPER_SPLIT_STEP_COST
                            + kernel.splits * kernel.cta_rows * rows * 4
                        )
                    fills = min_ctas <= ctas <= HOPPER_MAX_RESIDENCY * sms
                    candidates.append((not fills, cost, cfg))
        assert candidates
        return min(candidates, key=lambda c: (c[0], c[1]))[2]

    @cache
    def compile_kernel(n: int, k: int, x_tiles: int, cfg: GemvConfig, variant: str):
        cls = HopperMxfp8Gemv if variant == "hopper" else Mxfp8Gemv
        return cute.compile(
            cls(n, k, x_tiles, cfg),
            *(Int64(0),) * 6,
            *(Int32(1),) * 3,
            make_fake_stream(),
        )

    def launch_kernel(
        compiled,
        packed_w: torch.Tensor,
        factors: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        partials: torch.Tensor | None,
        counters: torch.Tensor,
    ) -> None:
        rows = x.shape[0]
        compiled(
            Int64(packed_w.data_ptr()),
            Int64(factors.data_ptr()),
            Int64(x.data_ptr()),
            Int64(out.data_ptr()),
            Int64(partials.data_ptr() if partials is not None else 0),
            Int64(counters.data_ptr()),
            Int32(rows),
            Int32(x.stride(0)),
            Int32(out.stride(0)),
            current_cu_stream(),
        )

    return {
        "torch": torch,
        "Mxfp8Gemv": Mxfp8Gemv,
        "HopperMxfp8Gemv": HopperMxfp8Gemv,
        "GemvConfig": GemvConfig,
        "X_TILE": X_TILE,
        "COUNTERS": COUNTERS,
        "make_weights": make_weights,
        "device_limits": device_limits,
        "ampere_default_config": ampere_default_config,
        "hopper_default_config": hopper_default_config,
        "compile_kernel": compile_kernel,
        "launch_kernel": launch_kernel,
    }


def _mode_dump_single(
    shape_name: str, n: int, k: int, m: int, variant: str, out_dir: str
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CUTE_DSL_KEEP_PTX"] = "1"
        os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
        os.environ["CUTE_DSL_DUMP_DIR"] = tmpdir
        mod = _build_gpu_module()
        torch = mod["torch"]
        props = torch.cuda.get_device_properties(0)
        sm_tag = f"sm{props.major}{props.minor}"
        sms, smem_limit = mod["device_limits"](0)
        x_tiles = (m + mod["X_TILE"] - 1) // mod["X_TILE"]
        cfg = (
            mod["hopper_default_config"](n, k, x_tiles, sms, smem_limit)
            if variant == "hopper"
            else mod["ampere_default_config"](n, k, m, sms, smem_limit)
        )
        prefix = f"{variant}_{sm_tag}_{shape_name}_{n}x{k}_m{m}"
        compiled = mod["compile_kernel"](n, k, x_tiles, cfg, variant)

        # Quick numerical verification against FP64 reference
        pw, bs, w_bf16 = mod["make_weights"](n, k, seed=n + k + m)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        sp = cfg.splits(k)
        partials = (
            torch.empty(
                sp * mod["X_TILE"] * x_tiles * n, dtype=torch.float32, device="cuda"
            )
            if sp > 1
            else None
        )
        counters = torch.zeros(mod["COUNTERS"], dtype=torch.int32, device="cuda")
        mod["launch_kernel"](compiled, pw, bs, x, out, partials, counters)
        torch.cuda.synchronize()
        want = x.double() @ w_bf16.double().t()
        mag = x.double().abs() @ w_bf16.double().abs().t()
        ulp = torch.exp2(torch.floor(torch.log2(want.abs().clamp_min(2.0**-100))) - 7)
        bound = ulp + mag * 2.0**-16
        assert ((out.double() - want).abs() <= bound).all(), (
            f"accuracy failed on {prefix}"
        )

        ptxs = sorted(glob.glob(os.path.join(tmpdir, "**", "*.ptx"), recursive=True))
        cubins = sorted(
            glob.glob(os.path.join(tmpdir, "**", "*.cubin"), recursive=True)
        )
        ptx_dst = os.path.join(out_dir, f"{prefix}.ptx")
        sass_dst = os.path.join(out_dir, f"{prefix}_cuobjdump.sass")
        if ptxs:
            with open(ptxs[0]) as src, open(ptx_dst, "w") as dst:
                dst.write(src.read())
        res_str = ""
        if cubins:
            sass = subprocess.check_output(
                ["cuobjdump", "--dump-sass", cubins[0]], text=True
            )
            with open(sass_dst, "w") as dst:
                dst.write(sass)
            usage = subprocess.check_output(
                ["cuobjdump", "--dump-resource-usage", cubins[0]], text=True
            )
            rm = re.search(
                r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", usage
            )
            if rm:
                regs, stack, shared, local = map(int, rm.groups())
                res_str = (
                    f"REG={regs} LOCAL={local} STACK={stack} STATIC_SMEM={shared}B"
                )

    ptx_n, ptx_c = (
        parse_ptx_stats(ptx_dst) if os.path.exists(ptx_dst) else (0, Counter())
    )
    sass_n, sass_c = (
        parse_sass_stats(sass_dst) if os.path.exists(sass_dst) else (0, Counter())
    )
    cls = mod["HopperMxfp8Gemv"] if variant == "hopper" else mod["Mxfp8Gemv"]
    dyn_smem = cls(n, k, x_tiles, cfg).smem_bytes()
    print(
        f"[{prefix}] cfg=({cfg.warps_n},{cfg.warps_k},{cfg.groups})s{cfg.splits(k)} "
        f"{res_str} DYN_SMEM={dyn_smem}B | "
        f"PTX={ptx_n} ({dict(ptx_c)}) | SASS={sass_n} ({dict(sass_c)})"
    )


def _mode_dump(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for shape_name, n, k, m in DUMP_CASES:
        for variant in ("ampere", "hopper"):
            cmd = [
                sys.executable,
                __file__,
                "--mode",
                "dump-single",
                "--shape",
                shape_name,
                "--n",
                str(n),
                "--k",
                str(k),
                "--m",
                str(m),
                "--variant",
                variant,
                "--out-dir",
                out_dir,
            ]
            subprocess.run(cmd, check=True)


def _mode_ncu() -> None:
    mod = _build_gpu_module()
    torch = mod["torch"]
    sms, smem_limit = mod["device_limits"](0)
    counters = torch.zeros(mod["COUNTERS"], dtype=torch.int32, device="cuda")

    # Pre-compile, allocate, verify, and warm up all 14 launches outside profiler range.
    prepared = []
    for idx, (_name, n, k, m) in enumerate(NCU_CASES):
        pw, bs, w_bf16 = mod["make_weights"](n, k, seed=idx + 100)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        x_tiles = (m + mod["X_TILE"] - 1) // mod["X_TILE"]
        want = x.double() @ w_bf16.double().t()
        mag = x.double().abs() @ w_bf16.double().abs().t()
        ulp = torch.exp2(torch.floor(torch.log2(want.abs().clamp_min(2.0**-100))) - 7)
        bound = ulp + mag * 2.0**-16
        for variant in ("ampere", "hopper"):
            cfg = (
                mod["hopper_default_config"](n, k, x_tiles, sms, smem_limit)
                if variant == "hopper"
                else mod["ampere_default_config"](n, k, m, sms, smem_limit)
            )
            compiled = mod["compile_kernel"](n, k, x_tiles, cfg, variant)
            out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            sp = cfg.splits(k)
            partials = (
                torch.empty(
                    sp * mod["X_TILE"] * x_tiles * n, dtype=torch.float32, device="cuda"
                )
                if sp > 1
                else None
            )
            mod["launch_kernel"](compiled, pw, bs, x, out, partials, counters)
            torch.cuda.synchronize()
            assert ((out.double() - want).abs() <= bound).all()
            prepared.append((compiled, pw, bs, x, out, partials))

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for compiled, pw, bs, x, out, partials in prepared:
        mod["launch_kernel"](compiled, pw, bs, x, out, partials, counters)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def _mode_bench(ms: list[int]) -> None:
    mod = _build_gpu_module()
    torch = mod["torch"]
    from flashinfer.testing import bench_gpu_time_with_cupti

    sms, smem_limit = mod["device_limits"](0)
    props = torch.cuda.get_device_properties(0)
    counters = torch.zeros(mod["COUNTERS"], dtype=torch.int32, device="cuda")
    print(
        f"GPU: {props.name} (SM{props.major}{props.minor}, "
        f"{sms} SMs, smem_limit={smem_limit})"
    )

    def bench_us(fn, args=()) -> float:
        for _ in range(10):
            fn(*args)
        torch.cuda.synchronize()
        times_ms = bench_gpu_time_with_cupti(
            fn,
            input_args=args,
            cold_l2_cache=True,
            use_cuda_graph=True,
            dry_run_iters=25,
            repeat_iters=100,
        )
        return statistics.median(times_ms) * 1e3

    print(
        f"{'Shape':22s} {'M':>3s} | {'Amp(us)':>8s} {'Hop(us)':>8s} {'cuBLAS':>8s} | "
        f"{'Hop/Amp':>7s} {'Hop/cuB':>7s} | {'AmpCfg':>12s} {'HopCfg':>12s}"
    )
    print("-" * 106)
    ratios_ha_24, ratios_ha_all, ratios_hc_24 = [], [], []
    for idx, (name, n, k) in enumerate(ALL_SHAPES):
        pw, bs, w_bf16 = mod["make_weights"](n, k, seed=idx + 10)
        for m in ms:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            out_a = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            out_h = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            x_tiles = (m + mod["X_TILE"] - 1) // mod["X_TILE"]
            ca = mod["ampere_default_config"](n, k, m, sms, smem_limit)
            ch = mod["hopper_default_config"](n, k, x_tiles, sms, smem_limit)
            ka = mod["compile_kernel"](n, k, x_tiles, ca, "ampere")
            kh = mod["compile_kernel"](n, k, x_tiles, ch, "hopper")
            pa = (
                torch.empty(
                    ca.splits(k) * mod["X_TILE"] * x_tiles * n,
                    dtype=torch.float32,
                    device="cuda",
                )
                if ca.splits(k) > 1
                else None
            )
            ph = (
                torch.empty(
                    ch.splits(k) * mod["X_TILE"] * x_tiles * n,
                    dtype=torch.float32,
                    device="cuda",
                )
                if ch.splits(k) > 1
                else None
            )
            ta = bench_us(mod["launch_kernel"], (ka, pw, bs, x, out_a, pa, counters))
            th = bench_us(mod["launch_kernel"], (kh, pw, bs, x, out_h, ph, counters))
            tc = bench_us(torch.nn.functional.linear, (x, w_bf16))
            ha, hc = ta / th, tc / th
            ratios_ha_all.append(ha)
            if m <= 24:
                ratios_ha_24.append(ha)
                ratios_hc_24.append(hc)
            ca_s = f"({ca.warps_n},{ca.warps_k},{ca.groups})s{ca.splits(k)}"
            ch_s = f"({ch.warps_n},{ch.warps_k},{ch.groups})s{ch.splits(k)}"
            print(
                f"{name + f' ({n},{k})':22s} {m:3d} | {ta:8.2f} {th:8.2f} {tc:8.2f} | "
                f"{ha:6.3f}x {hc:6.3f}x | {ca_s:>12s} {ch_s:>12s}"
            )

    def gm(vals: list[float]) -> float:
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    print("-" * 106)
    print(
        f"Geomean Hop/Amp: all={gm(ratios_ha_all):.4f}x | "
        f"M<=24={gm(ratios_ha_24):.4f}x | "
        f"Hop/cuBLAS (M<=24)={gm(ratios_hc_24):.4f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dump", "dump-single", "ncu", "parse-ncu", "bench"),
        required=True,
    )
    parser.add_argument("--variant", choices=("ampere", "hopper"), default="hopper")
    parser.add_argument("--shape", type=str, default="wo_b")
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--ms", type=str, default="1,6,12,24,32,48")
    parser.add_argument("--out-dir", type=str, default="/tmp/poiesis/mxfp8_gemv")
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    if args.mode == "parse-ncu":
        if not args.csv:
            parser.error("--csv is required for --mode parse-ncu")
        parse_ncu_csv(args.csv)
    elif args.mode == "dump":
        _mode_dump(args.out_dir)
    elif args.mode == "dump-single":
        _mode_dump_single(
            args.shape, args.n, args.k, args.m, args.variant, args.out_dir
        )
    elif args.mode == "ncu":
        _mode_ncu()
    elif args.mode == "bench":
        _mode_bench(_parse_int_list(args.ms))


if __name__ == "__main__":
    main()
