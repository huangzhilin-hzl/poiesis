"""PTX, SASS, NCU, and Cold-L2 profiling harness for DeepSeek-V4 top-6 routing.

Compares three bit-exact implementations on CUDA SM80+/SM90+ (``E in {256, 384}``,
``top_k = 6``):

  1. Triton baseline (``_dsv4_topk_kernel``, 1 warp, ``BLOCK_N = next_pow2(E)``)
  2. CuTe-DSL ``single_warp_kernel`` (1 warp, 32 threads, zero SMEM, zero barrier)
  3. CuTe-DSL ``multi_warp_kernel`` (4 warps, 128 threads, two-stage SMEM merge)

Subcommands (``--mode``):
  - ``dump``      : Compile all 3 kernels, verify bit-exact parity, dump PTX/SASS,
                    and print static instruction & resource-usage tables.
  - ``ncu``       : Warm up & verify, then launch only the target kernels inside
                    ``cudaProfilerStart`` / ``cudaProfilerStop`` for Nsight Compute.
  - ``parse-ncu`` : Parse an ``ncu --page raw --csv`` export into SOL, pipe, memory,
                    and warp-stall CPI tables.
  - ``bench``     : Measure pure-GPU Cold-L2 latency via rotating-buffer CUDA Graphs.

Example usage:
    python experiments/ops/cutedsl/dsv4_topk_profile.py --mode dump --out-dir /tmp/out
    ncu --profile-from-start off --set full --cache-control all --clock-control base \\
        -f -o /tmp/dsv4_topk_e384 \\
        python experiments/ops/cutedsl/dsv4_topk_profile.py --mode ncu
    python experiments/ops/cutedsl/dsv4_topk_profile.py \\
        --mode parse-ncu --csv profile/ncu/dsv4_topk_sm90_e384_raw.csv
    python experiments/ops/cutedsl/dsv4_topk_profile.py --mode bench
"""

import argparse
import csv
import glob
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from functools import cache

_TOPK = 6
_WARP_SCHEDULERS_PER_SM = 4


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
        if tok.startswith("redux.sync"):
            counts["redux.sync"] += 1
        elif tok.startswith("shfl.sync.bfly"):
            counts["shfl.bfly"] += 1
        elif tok.startswith("shfl.sync"):
            counts["shfl.other"] += 1
        elif tok.startswith("vote.sync"):
            counts["vote.sync"] += 1
        elif tok.startswith(("bar.sync", "barrier.")):
            counts["bar.sync"] += 1
        elif tok.startswith("griddepcontrol."):
            counts["pdl"] += 1
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
        elif tok.startswith(("ex2.", "lg2.", "sqrt.", "rcp.", "div.")):
            counts["sfu/div"] += 1
        elif tok.startswith(
            ("add.f32", "mul.f32", "fma.rn.f32", "max.f32", "min.f32", "sub.f32")
        ):
            counts["fp32"] += 1
        elif tok.startswith("setp."):
            counts["setp"] += 1
        elif tok.startswith("selp."):
            counts["selp"] += 1
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
        if base in ("REDUX", "CREDUX"):
            counts["REDUX"] += 1
        elif base == "SHFL":
            counts["SHFL"] += 1
        elif base in ("VOTE", "VOTEU"):
            counts["VOTE"] += 1
        elif base in ("BAR", "BSSY", "BSYNC", "WARPSYNC"):
            counts["BAR/SYNC"] += 1
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
        elif base == "MUFU":
            counts["MUFU"] += 1
        elif base in ("FADD", "FADD2", "FMUL", "FFMA", "FMNMX", "FMNMX3"):
            counts["FP32"] += 1
        elif base in ("FSETP", "ISETP", "SEL", "FSEL"):
            counts["CMP/SEL"] += 1
        elif base in (
            "IADD3",
            "IMAD",
            "LOP3",
            "SHF",
            "LEA",
            "POPC",
            "BFE",
            "PRMT",
        ):
            counts["INT/BIT"] += 1
    return total, counts


# ---------------------------------------------------------------------------
# NCU raw CSV parser (CPU-only)
# ---------------------------------------------------------------------------


def parse_ncu_csv(csv_path: str) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    headers, units, data_rows = rows[0], rows[1], rows[2:]
    col = {h: i for i, h in enumerate(headers)}

    def short_name(kname: str) -> str:
        if "_dsv4_topk_kernel" in kname:
            return "Triton"
        if "single_warp" in kname:
            return "CuTe-1W"
        if "multi_warp" in kname:
            return "CuTe-4W"
        return kname[:16]

    def getf(r: list[str], key: str, default: float = 0.0) -> float:
        if key not in col:
            return default
        v = r[col[key]].replace(",", "").strip()
        if not v or v in ("n/a", "nan"):
            return default
        return float(v)

    def scale_dur_us(r: list[str]) -> float:
        dur = getf(r, "gpu__time_duration.sum")
        unit = units[col["gpu__time_duration.sum"]]
        if unit == "nsecond":
            return dur / 1000.0
        if unit == "msecond":
            return dur * 1000.0
        return dur

    def scale_dram_gbs(r: list[str]) -> float:
        key = "dram__bytes.sum.per_second"
        val = getf(r, key)
        if key not in col:
            return 0.0
        unit = units[col[key]]
        if unit == "byte/second":
            return val / 1e9
        if unit == "Mbyte/second":
            return val / 1e3
        if unit == "Tbyte/second":
            return val * 1e3
        return val

    print("=== TABLE 1: NCU OVERVIEW & COMPUTE PIPES ===")
    hdr1 = (
        f"{'M':>5} | {'Kernel':<8} | {'Dur(us)':>7} | {'Cycles':>7} | "
        f"{'SM%':>5} | {'ALU%':>5} | {'LSU%':>5} | {'ADU%':>5} | "
        f"{'DRAM GB/s':>9} | {'Reg':>3} | {'ThOcc%':>6} | {'AchOcc%':>7} | "
        f"{'Iss%':>5} | {'Inst/Tok':>8} | {'Thd/Inst':>8}"
    )
    print(hdr1)
    print("-" * len(hdr1))
    for r in data_rows:
        m = int(r[col["Grid Size"]].strip("()").split(",")[0].strip())
        k = short_name(r[col["Kernel Name"]])
        alu = getf(r, "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active")
        lsu = getf(r, "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active")
        adu = getf(r, "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active")
        print(
            f"{m:>5} | {k:<8} | {scale_dur_us(r):>7.2f} | "
            f"{getf(r, 'gpc__cycles_elapsed.max'):>7.0f} | "
            f"{getf(r, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'):>5.1f} | "
            f"{alu:>5.1f} | {lsu:>5.1f} | {adu:>5.1f} | "
            f"{scale_dram_gbs(r):>9.1f} | "
            f"{int(getf(r, 'launch__registers_per_thread')):>3} | "
            f"{getf(r, 'sm__maximum_warps_per_active_cycle_pct'):>6.1f} | "
            f"{getf(r, 'sm__warps_active.avg.pct_of_peak_sustained_active'):>7.1f} | "
            f"{getf(r, 'smsp__issue_active.avg.pct_of_peak_sustained_active'):>5.1f} | "
            f"{getf(r, 'smsp__inst_executed.sum') / m:>8.1f} | "
            f"{getf(r, 'smsp__thread_inst_executed_per_inst_executed.ratio'):>8.2f}"
        )

    print("\n=== TABLE 2: MEMORY OPS & SECTORS PER TOKEN ===")
    hdr2 = (
        f"{'M':>5} | {'Kernel':<8} | {'Gld/T':>5} | {'Gst/T':>5} | "
        f"{'Sld/T':>5} | {'Sst/T':>5} | {'Loc/T':>5} | "
        f"{'L1Req/T':>7} | {'L1Sec/T':>7} | {'Sec/Req':>7} | {'L2Hit%':>6}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for r in data_rows:
        m = int(r[col["Grid Size"]].strip("()").split(",")[0].strip())
        k = short_name(r[col["Kernel Name"]])
        req = getf(r, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum") / m
        sec = getf(r, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") / m
        loc = (
            getf(r, "smsp__sass_inst_executed_op_local_ld.sum")
            + getf(r, "smsp__sass_inst_executed_op_local_st.sum")
        ) / m
        print(
            f"{m:>5} | {k:<8} | "
            f"{getf(r, 'smsp__sass_inst_executed_op_global_ld.sum') / m:>5.1f} | "
            f"{getf(r, 'smsp__sass_inst_executed_op_global_st.sum') / m:>5.1f} | "
            f"{getf(r, 'smsp__sass_inst_executed_op_shared_ld.sum') / m:>5.1f} | "
            f"{getf(r, 'smsp__sass_inst_executed_op_shared_st.sum') / m:>5.1f} | "
            f"{loc:>5.1f} | {req:>7.1f} | {sec:>7.1f} | "
            f"{(sec / req if req > 0 else 0.0):>7.2f} | "
            f"{getf(r, 'lts__t_sector_hit_rate.pct'):>6.1f}"
        )

    stalls = [
        ("long_sb", "long_scoreboard"),
        ("short_sb", "short_scoreboard"),
        ("wait", "wait"),
        ("barrier", "barrier"),
        ("not_sel", "not_selected"),
        ("math_thr", "math_pipe_throttle"),
        ("mio_thr", "mio_throttle"),
        ("no_inst", "no_instruction"),
        ("selected", "selected"),
    ]
    print("\n=== TABLE 3: WARP STALL REASONS (Cycles / Issued Instruction) ===")
    hdr3 = f"{'M':>5} | {'Kernel':<8} | " + " | ".join(
        f"{label:>8}" for label, _ in stalls
    )
    print(hdr3)
    print("-" * len(hdr3))
    for r in data_rows:
        m = int(r[col["Grid Size"]].strip("()").split(",")[0].strip())
        k = short_name(r[col["Kernel Name"]])
        vals = [
            getf(
                r,
                f"smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio",
            )
            for _, name in stalls
        ]
        print(f"{m:>5} | {k:<8} | " + " | ".join(f"{v:>8.2f}" for v in vals))


# ---------------------------------------------------------------------------
# GPU Kernels (lazy-imported so parse-ncu runs on CPU-only machines)
# ---------------------------------------------------------------------------


def _build_gpu_kernels():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import torch
    import triton
    import triton.language as tl
    from cutlass import Float32, Int32, Int64, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_fake_stream
    from cutlass.cutlass_dsl import T, dsl_user_op

    _SmemAllocator = getattr(cutlass, "memory", cutlass.utils).SmemAllocator

    @triton.jit
    def _dsv4_topk_kernel(
        gating_output_ptr,
        correction_bias_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        routed_scaling_factor,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        launch_pdl: tl.constexpr,
    ):
        row = tl.program_id(0)
        expert_offsets = tl.arange(0, BLOCK_N)
        expert_mask = expert_offsets < NUM_EXPERTS
        bias = tl.load(
            correction_bias_ptr + expert_offsets, mask=expert_mask, other=0.0
        ).to(tl.float32)

        if launch_pdl:
            tl.extra.cuda.gdc_wait()

        logits = tl.load(
            gating_output_ptr + row * NUM_EXPERTS + expert_offsets,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
        weights = tl.sqrt(tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits))))
        current = tl.where(expert_mask, weights + bias, -float("inf"))
        current = tl.where(current == current, current, -1e30)

        topk_offsets = tl.arange(0, 8)
        selected_weights = tl.zeros([8], dtype=tl.float32)
        selected_ids = tl.zeros([8], dtype=tl.int32)
        available = expert_mask
        for slot in tl.static_range(6):
            max_value = tl.max(current, axis=0)
            candidate = tl.where(
                available & (current == max_value), expert_offsets, NUM_EXPERTS
            )
            expert_id = tl.min(candidate, axis=0).to(tl.int32)
            selected_weight = tl.sum(
                tl.where(expert_offsets == expert_id, weights, 0.0), axis=0
            )
            is_slot = topk_offsets == slot
            selected_weights = tl.where(is_slot, selected_weight, selected_weights)
            selected_ids = tl.where(is_slot, expert_id, selected_ids)
            available = available & (expert_offsets != expert_id)
            current = tl.where(expert_offsets == expert_id, -float("inf"), current)

        weight_sum = tl.sum(selected_weights, axis=0)
        selected_weights *= routed_scaling_factor / tl.where(
            weight_sum > 0.0, weight_sum, 1.0
        )
        output_mask = topk_offsets < 6
        output_offsets = row * 6 + topk_offsets
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(topk_weights_ptr + output_offsets, selected_weights, mask=output_mask)
        tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=output_mask)

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

    def _f32(val, loc, ip):
        return Float32(val).ir_value(loc=loc, ip=ip)

    @dsl_user_op
    def _load_f32_nc(addr: Int64, *, loc=None, ip=None) -> Float32:
        return Float32(
            _asm(
                T.f32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.global.nc.f32 $0, [$1];",
                "=f,l",
                False,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _load_f32_ordered(addr: Int64, *, loc=None, ip=None) -> Float32:
        return Float32(
            _asm(
                T.f32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.global.f32 $0, [$1];",
                "=f,l",
                True,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _store_f32(addr: Int64, val: Float32, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [Int64(addr).ir_value(loc=loc, ip=ip), _f32(val, loc, ip)],
            "st.global.f32 [$0], $1;",
            "l,f",
            True,
            loc,
            ip,
        )

    @dsl_user_op
    def _store_s32(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
            "st.global.s32 [$0], $1;",
            "l,r",
            True,
            loc,
            ip,
        )

    _SOFTPLUS_ARG_PTX = (
        "{\n"
        ".reg .f32 a;\n"
        "mul.f32 a, $1, 0f3FB8AA3B;\n"
        "ex2.approx.f32 a, a;\n"
        "add.f32 $0, a, 0f3F800000;\n"
        "}"
    )

    _SCORE_KEY_PTX = (
        "{\n"
        ".reg .pred p;\n"
        ".reg .f32 l, s, v;\n"
        ".reg .b32 b, n;\n"
        "setp.gt.f32 p, $2, 0f41A00000;\n"
        "selp.f32 l, $2, $3, p;\n"
        "sqrt.approx.ftz.f32 s, l;\n"
        "add.f32 v, s, $4;\n"
        "setp.num.f32 p, v, v;\n"
        "selp.f32 v, v, 0fF149F2CA, p;\n"
        "setp.eq.f32 p, v, 0f00000000;\n"
        "selp.f32 v, 0f00000000, v, p;\n"
        "mov.b32 b, v;\n"
        "setp.lt.s32 p, b, 0;\n"
        "not.b32 n, b;\n"
        "or.b32 b, b, 0x80000000;\n"
        "selp.b32 $1, n, b, p;\n"
        "mov.f32 $0, s;\n"
        "}"
    )

    @dsl_user_op
    def _score_and_key(logit, bias, *, loc=None, ip=None):
        arg = Float32(
            _asm(
                T.f32(),
                [_f32(logit, loc, ip)],
                _SOFTPLUS_ARG_PTX,
                "=f,f",
                False,
                loc,
                ip,
            )
        )
        log_arg = cute.math.log(arg, fastmath=False)
        res = _asm(
            llvm.StructType.get_literal([T.f32(), T.i32()]),
            [_f32(logit, loc, ip), _f32(log_arg, loc, ip), _f32(bias, loc, ip)],
            _SCORE_KEY_PTX,
            "=f,=r,f,f,f",
            False,
            loc,
            ip,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def _add_f32(a, b, *, loc=None, ip=None) -> Float32:
        return Float32(
            _asm(
                T.f32(),
                [_f32(a, loc, ip), _f32(b, loc, ip)],
                "add.f32 $0, $1, $2;",
                "=f,f,f",
                False,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _normalized_weight(scale, total, weight, *, loc=None, ip=None) -> Float32:
        return Float32(
            _asm(
                T.f32(),
                [_f32(scale, loc, ip), _f32(total, loc, ip), _f32(weight, loc, ip)],
                "{\n"
                ".reg .pred p;\n"
                ".reg .f32 d;\n"
                "setp.gt.f32 p, $2, 0f00000000;\n"
                "selp.f32 d, $2, 0f3F800000, p;\n"
                "div.full.f32 d, $1, d;\n"
                "mul.f32 $0, d, $3;\n"
                "}",
                "=f,f,f,f",
                False,
                loc,
                ip,
            )
        )

    class _Dsv4TopkCutlass:
        def __init__(self, num_experts: int, num_warps: int, use_pdl: bool) -> None:
            self.num_experts = num_experts
            self.num_warps = num_warps
            self.threads = num_warps * 32
            self.experts_per_thread = num_experts // self.threads
            self.num_merge_candidates = num_warps * _TOPK
            self.use_pdl = use_pdl

        @cute.jit
        def __call__(
            self,
            gating_output: Int64,
            correction_bias: Int64,
            topk_weights: Int64,
            topk_ids: Int64,
            num_tokens: Int32,
            routed_scaling_factor: Float32,
            stream: cuda.CUstream,
        ):
            if cutlass.const_expr(self.num_warps == 1):
                self.single_warp_kernel(
                    gating_output,
                    correction_bias,
                    topk_weights,
                    topk_ids,
                    routed_scaling_factor,
                ).launch(
                    grid=(num_tokens, 1, 1),
                    block=[32, 1, 1],
                    stream=stream,
                    use_pdl=self.use_pdl,
                )
            else:

                @cute.struct
                class Storage:
                    merge_keys: cute.struct.MemRange[Uint32, self.num_merge_candidates]
                    merge_ids: cute.struct.MemRange[Int32, self.num_merge_candidates]
                    merge_scores: cute.struct.MemRange[
                        Float32, self.num_merge_candidates
                    ]

                self.multi_warp_kernel(
                    gating_output,
                    correction_bias,
                    topk_weights,
                    topk_ids,
                    routed_scaling_factor,
                    Storage,
                ).launch(
                    grid=(num_tokens, 1, 1),
                    block=[self.threads, 1, 1],
                    stream=stream,
                    use_pdl=self.use_pdl,
                )

        @cute.jit
        def _find_best_local(self, key: cute.Tensor):
            best_key = key[0]
            best_local_idx = Int32(0)
            for j in cutlass.range_constexpr(1, self.experts_per_thread):
                if key[j] > best_key:
                    best_key = key[j]
                    best_local_idx = Int32(j)
            return best_key, best_local_idx

        @cute.jit
        def _normalize_and_store(
            self,
            slot_weight: Float32,
            slot_id: Int32,
            routed_scaling_factor: Float32,
            topk_weights: Int64,
            topk_ids: Int64,
            row: Int32,
            lane: Int32,
        ) -> None:
            total = slot_weight
            for shift in cutlass.range_constexpr(2, -1, -1):
                total = _add_f32(total, cute.arch.shuffle_sync_bfly(total, 1 << shift))
            if lane < _TOPK:
                final_w = _normalized_weight(routed_scaling_factor, total, slot_weight)
                out_idx = Int64(row) * _TOPK + lane
                _store_f32(topk_weights + out_idx * 4, final_w)
                _store_s32(topk_ids + out_idx * 4, slot_id)

        @cute.kernel
        def single_warp_kernel(
            self,
            gating_output: Int64,
            correction_bias: Int64,
            topk_weights: Int64,
            topk_ids: Int64,
            routed_scaling_factor: Float32,
        ):
            lane, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            experts_per_thread = self.experts_per_thread

            expert_bias = cute.make_rmem_tensor(experts_per_thread, Float32)
            for j in cutlass.range_constexpr(experts_per_thread):
                expert_bias[j] = _load_f32_nc(
                    correction_bias + Int64(j * 32 + lane) * 4
                )

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            score = cute.make_rmem_tensor(experts_per_thread, Float32)
            key = cute.make_rmem_tensor(experts_per_thread, Uint32)
            row_base = gating_output + Int64(row) * (self.num_experts * 4)
            for j in cutlass.range_constexpr(experts_per_thread):
                logit = _load_f32_ordered(row_base + Int64(j * 32 + lane) * 4)
                score[j], key[j] = _score_and_key(logit, expert_bias[j])

            best_key, best_local_idx = self._find_best_local(key)
            slot_weight = Float32(0.0)
            slot_id = Int32(0)
            for slot in cutlass.range_constexpr(_TOPK):
                top = cute.arch.warp_redux_sync(best_key, "umax")
                tied_expert_id = Uint32(0xFFFFFFFF)
                if best_key == top:
                    tied_expert_id = Uint32(lane + 32 * best_local_idx)
                winner = Int32(cute.arch.warp_redux_sync(tied_expert_id, "umin"))
                won = Float32(0.0)
                if winner == lane + 32 * best_local_idx:
                    for j in cutlass.range_constexpr(experts_per_thread):
                        if best_local_idx == j:
                            won = score[j]
                            key[j] = Uint32(0)
                    best_key, best_local_idx = self._find_best_local(key)
                weight = cute.arch.shuffle_sync(won, winner & 31)
                if lane == slot:
                    slot_weight = weight
                    slot_id = winner

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            self._normalize_and_store(
                slot_weight,
                slot_id,
                routed_scaling_factor,
                topk_weights,
                topk_ids,
                row,
                lane,
            )

        @cute.kernel
        def multi_warp_kernel(
            self,
            gating_output: Int64,
            correction_bias: Int64,
            topk_weights: Int64,
            topk_ids: Int64,
            routed_scaling_factor: Float32,
            Storage: cutlass.Constexpr,
        ):
            tid, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            warp = tid // 32
            lane = tid % 32
            experts_per_thread = self.experts_per_thread
            warp_base = warp * (32 * experts_per_thread)

            smem = _SmemAllocator()
            storage = smem.allocate(Storage)
            smem_keys = storage.merge_keys.get_tensor(
                cute.make_layout(self.num_merge_candidates)
            )
            smem_ids = storage.merge_ids.get_tensor(
                cute.make_layout(self.num_merge_candidates)
            )
            smem_scores = storage.merge_scores.get_tensor(
                cute.make_layout(self.num_merge_candidates)
            )

            expert_bias = cute.make_rmem_tensor(experts_per_thread, Float32)
            for j in cutlass.range_constexpr(experts_per_thread):
                e = warp_base + j * 32 + lane
                expert_bias[j] = _load_f32_nc(correction_bias + Int64(e) * 4)

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            score = cute.make_rmem_tensor(experts_per_thread, Float32)
            key = cute.make_rmem_tensor(experts_per_thread, Uint32)
            row_base = gating_output + Int64(row) * (self.num_experts * 4)
            for j in cutlass.range_constexpr(experts_per_thread):
                e = warp_base + j * 32 + lane
                logit = _load_f32_ordered(row_base + Int64(e) * 4)
                score[j], key[j] = _score_and_key(logit, expert_bias[j])

            best_key, best_local_idx = self._find_best_local(key)
            out_key = Uint32(0)
            out_id = Int32(-1)
            out_score = Float32(0.0)
            for slot in cutlass.range_constexpr(_TOPK):
                top = cute.arch.warp_redux_sync(best_key, "umax")
                tied_expert_id = Uint32(0xFFFFFFFF)
                if best_key == top:
                    tied_expert_id = Uint32(warp_base + lane + 32 * best_local_idx)
                winner = Int32(cute.arch.warp_redux_sync(tied_expert_id, "umin"))
                won = Float32(0.0)
                if winner - warp_base == lane + 32 * best_local_idx:
                    for j in cutlass.range_constexpr(experts_per_thread):
                        if best_local_idx == j:
                            won = score[j]
                            key[j] = Uint32(0)
                    best_key, best_local_idx = self._find_best_local(key)
                weight = cute.arch.shuffle_sync(won, (winner - warp_base) & 31)
                if lane == slot:
                    out_key = top
                    out_id = winner
                    out_score = weight

            if lane < _TOPK:
                idx = warp * _TOPK + lane
                smem_keys[idx] = out_key
                smem_ids[idx] = out_id
                smem_scores[idx] = out_score

            cute.arch.barrier()

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            if warp == 0:
                cand_key = Uint32(0)
                cand_id = Int32(0x7FFFFFFF)
                cand_score = Float32(0.0)
                if lane < self.num_merge_candidates:
                    cand_key = smem_keys[lane]
                    cand_id = smem_ids[lane]
                    cand_score = smem_scores[lane]

                slot_weight = Float32(0.0)
                slot_id = Int32(0)
                for slot in cutlass.range_constexpr(_TOPK):
                    top = cute.arch.warp_redux_sync(cand_key, "umax")
                    tied_expert_id = Uint32(0xFFFFFFFF)
                    if cand_key == top:
                        tied_expert_id = Uint32(cand_id)
                    chosen = Int32(cute.arch.warp_redux_sync(tied_expert_id, "umin"))
                    holders = cute.arch.vote_ballot_sync(cand_id == chosen)
                    first = cute.arch.popc((holders & (Uint32(0) - holders)) - 1)
                    weight = cute.arch.shuffle_sync(cand_score, first)
                    if cand_id == chosen:
                        cand_key = Uint32(0)
                    if lane == slot:
                        slot_weight = weight
                        slot_id = chosen

                self._normalize_and_store(
                    slot_weight,
                    slot_id,
                    routed_scaling_factor,
                    topk_weights,
                    topk_ids,
                    row,
                    lane,
                )

    @cache
    def compile_cutedsl(num_experts: int, num_warps: int, use_pdl: bool):
        return cute.compile(
            _Dsv4TopkCutlass(num_experts, num_warps, use_pdl),
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(1),
            Float32(1.0),
            make_fake_stream(),
        )

    def run_triton(logits, bias, ow, oi, scale: float, use_pdl: bool):
        num_tokens, num_experts = logits.shape
        return _dsv4_topk_kernel[(num_tokens,)](
            logits,
            bias,
            ow,
            oi,
            scale,
            NUM_EXPERTS=num_experts,
            BLOCK_N=triton.next_power_of_2(num_experts),
            num_warps=1,
            launch_pdl=use_pdl,
        )

    def run_cutedsl(compiled, logits, bias, ow, oi, scale: float):
        stream = torch.cuda.current_stream()
        compiled(
            Int64(logits.data_ptr()),
            Int64(bias.data_ptr()),
            Int64(ow.data_ptr()),
            Int64(oi.data_ptr()),
            Int32(logits.shape[0]),
            Float32(scale),
            cuda.CUstream(stream.cuda_stream),
        )

    return compile_cutedsl, run_triton, run_cutedsl


def _mode_dump_single(num_experts: int, variant: str, out_dir: str) -> None:
    import torch

    compile_cutedsl, run_triton, run_cutedsl = _build_gpu_kernels()
    major, minor = torch.cuda.get_device_capability()
    sm_tag = f"sm{major}{minor}"
    use_pdl = major >= 9
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(0)
    m = 16
    logits = torch.randn((m, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")
    tw = torch.empty((m, _TOPK), dtype=torch.float32, device="cuda")
    ti = torch.empty((m, _TOPK), dtype=torch.int32, device="cuda")
    compiled_tri = run_triton(logits, bias, tw, ti, 2.5, use_pdl)
    torch.cuda.synchronize()

    if variant == "triton":
        ptx_path = os.path.join(
            out_dir, f"triton_{sm_tag}_dsv4_topk_e{num_experts}.ptx"
        )
        cubin_path = os.path.join(
            out_dir, f"triton_{sm_tag}_dsv4_topk_e{num_experts}.cubin"
        )
        sass_path = os.path.join(
            out_dir, f"triton_{sm_tag}_dsv4_topk_e{num_experts}_cuobjdump.sass"
        )
        with open(ptx_path, "w") as f:
            f.write(compiled_tri.asm["ptx"])
        with open(cubin_path, "wb") as f:
            f.write(compiled_tri.asm["cubin"])
        sass = subprocess.check_output(
            ["cuobjdump", "--dump-sass", cubin_path], text=True
        )
        with open(sass_path, "w") as f:
            f.write(sass)
        res = subprocess.check_output(
            ["cuobjdump", "--dump-resource-usage", cubin_path], text=True
        )
        print(f"=== triton ({sm_tag}) E={num_experts} ===\n{res.strip()}")
        return

    num_warps = 1 if variant == "cutedsl_1w" else 4
    c = compile_cutedsl(num_experts, num_warps, use_pdl)
    ow = torch.empty_like(tw)
    oi = torch.empty_like(ti)
    run_cutedsl(c, logits, bias, ow, oi, 2.5)
    torch.cuda.synchronize()
    torch.testing.assert_close(oi, ti, atol=0, rtol=0)
    torch.testing.assert_close(ow, tw, atol=0, rtol=0)

    cubins = sorted(glob.glob(os.path.join(out_dir, "*.cubin")))
    ptxs = sorted(glob.glob(os.path.join(out_dir, "*.ptx")))
    if ptxs:
        dst_ptx = os.path.join(
            out_dir, f"cutedsl_{sm_tag}_dsv4_topk_{num_warps}w_e{num_experts}.ptx"
        )
        if ptxs[0] != dst_ptx:
            os.replace(ptxs[0], dst_ptx)
    if cubins:
        sass_path = os.path.join(
            out_dir,
            f"cutedsl_{sm_tag}_dsv4_topk_{num_warps}w_e{num_experts}_cuobjdump.sass",
        )
        sass = subprocess.check_output(
            ["cuobjdump", "--dump-sass", cubins[0]], text=True
        )
        with open(sass_path, "w") as f:
            f.write(sass)
        res = subprocess.check_output(
            ["cuobjdump", "--dump-resource-usage", cubins[0]], text=True
        )
        print(
            f"=== {variant} ({sm_tag}) E={num_experts} (bit-exact OK) ===\n"
            f"{res.strip()}"
        )


def _mode_dump(num_experts: int, out_dir: str) -> None:
    import torch

    major, minor = torch.cuda.get_device_capability()
    sm_tag = f"sm{major}{minor}"
    os.makedirs(out_dir, exist_ok=True)
    # Invoke isolated subprocesses so CUTE_DSL_DUMP_DIR cleanly separates 1W vs 4W.
    sub_dirs = {
        "triton": os.path.join(out_dir, "triton"),
        "cutedsl_1w": os.path.join(out_dir, "cutedsl_1w"),
        "cutedsl_4w": os.path.join(out_dir, "cutedsl_4w"),
    }
    for variant, vdir in sub_dirs.items():
        os.makedirs(vdir, exist_ok=True)
        env = os.environ.copy()
        if variant != "triton":
            env["CUTE_DSL_KEEP"] = "ptx,cubin"
            env["CUTE_DSL_DUMP_DIR"] = vdir
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--mode",
                "dump-single",
                "--variant",
                variant,
                "--num-experts",
                str(num_experts),
                "--out-dir",
                vdir,
            ],
            env=env,
            check=True,
        )

    targets = [
        (
            f"Triton_E{num_experts}",
            os.path.join(
                sub_dirs["triton"], f"triton_{sm_tag}_dsv4_topk_e{num_experts}.ptx"
            ),
            os.path.join(
                sub_dirs["triton"],
                f"triton_{sm_tag}_dsv4_topk_e{num_experts}_cuobjdump.sass",
            ),
        ),
        (
            f"CuTe_1W_E{num_experts}",
            os.path.join(
                sub_dirs["cutedsl_1w"],
                f"cutedsl_{sm_tag}_dsv4_topk_1w_e{num_experts}.ptx",
            ),
            os.path.join(
                sub_dirs["cutedsl_1w"],
                f"cutedsl_{sm_tag}_dsv4_topk_1w_e{num_experts}_cuobjdump.sass",
            ),
        ),
        (
            f"CuTe_4W_E{num_experts}",
            os.path.join(
                sub_dirs["cutedsl_4w"],
                f"cutedsl_{sm_tag}_dsv4_topk_4w_e{num_experts}.ptx",
            ),
            os.path.join(
                sub_dirs["cutedsl_4w"],
                f"cutedsl_{sm_tag}_dsv4_topk_4w_e{num_experts}_cuobjdump.sass",
            ),
        ),
    ]

    ptx_cols = [
        "redux.sync",
        "shfl.bfly",
        "shfl.other",
        "vote.sync",
        "bar.sync",
        "ld.g.nc",
        "ld.g",
        "st.g",
        "ld.s",
        "st.s",
        "local",
        "sfu/div",
    ]
    print("\n=== STATIC PTX INSTRUCTION BREAKDOWN ===")
    hdr = f"{'Kernel':<14} | {'Total':>5} | " + " | ".join(f"{c:>10}" for c in ptx_cols)
    print(hdr)
    print("-" * len(hdr))
    for name, ptx_path, _ in targets:
        tot, cnt = parse_ptx_stats(ptx_path)
        print(
            f"{name:<14} | {tot:>5} | " + " | ".join(f"{cnt[c]:>10}" for c in ptx_cols)
        )

    sass_cols = [
        "REDUX",
        "SHFL",
        "VOTE",
        "BAR/SYNC",
        "LDG",
        "STG",
        "LDS",
        "STS",
        "LDL/STL",
        "MUFU",
        "FP32",
        "CMP/SEL",
        "INT/BIT",
    ]
    print("\n=== STATIC SASS INSTRUCTION BREAKDOWN ===")
    hdr = f"{'Kernel':<14} | {'Total':>5} | " + " | ".join(f"{c:>8}" for c in sass_cols)
    print(hdr)
    print("-" * len(hdr))
    for name, _, sass_path in targets:
        tot, cnt = parse_sass_stats(sass_path)
        print(
            f"{name:<14} | {tot:>5} | " + " | ".join(f"{cnt[c]:>8}" for c in sass_cols)
        )


def _mode_ncu(num_experts: int, ms: list[int]) -> None:
    import torch

    compile_cutedsl, run_triton, run_cutedsl = _build_gpu_kernels()
    major, _ = torch.cuda.get_device_capability()
    use_pdl = major >= 9

    torch.manual_seed(42)
    bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")
    c1w = compile_cutedsl(num_experts, 1, use_pdl)
    c4w = compile_cutedsl(num_experts, 4, use_pdl)

    # Pre-allocate, compile, and verify bit-exact equality before profiling.
    data = {}
    for m in ms:
        logits = torch.randn((m, num_experts), dtype=torch.float32, device="cuda")
        tw = torch.empty((m, _TOPK), dtype=torch.float32, device="cuda")
        ti = torch.empty((m, _TOPK), dtype=torch.int32, device="cuda")
        ow1 = torch.empty_like(tw)
        oi1 = torch.empty_like(ti)
        ow4 = torch.empty_like(tw)
        oi4 = torch.empty_like(ti)
        run_triton(logits, bias, tw, ti, 2.5, use_pdl)
        run_cutedsl(c1w, logits, bias, ow1, oi1, 2.5)
        run_cutedsl(c4w, logits, bias, ow4, oi4, 2.5)
        torch.cuda.synchronize()
        torch.testing.assert_close(oi1, ti, atol=0, rtol=0)
        torch.testing.assert_close(ow1, tw, atol=0, rtol=0)
        torch.testing.assert_close(oi4, ti, atol=0, rtol=0)
        torch.testing.assert_close(ow4, tw, atol=0, rtol=0)
        data[m] = (logits, tw, ti, ow1, oi1, ow4, oi4)

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    for m in ms:
        logits, tw, ti, ow1, oi1, ow4, oi4 = data[m]
        run_triton(logits, bias, tw, ti, 2.5, use_pdl)
        run_cutedsl(c1w, logits, bias, ow1, oi1, 2.5)
        run_cutedsl(c4w, logits, bias, ow4, oi4, 2.5)
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()
    print(f"Profiled {len(ms) * 3} launches across M={ms} (E={num_experts})")


def _mode_bench(num_experts: int, ms: list[int], target_mb: int) -> None:
    import torch

    compile_cutedsl, run_triton, run_cutedsl = _build_gpu_kernels()
    props = torch.cuda.get_device_properties(0)
    use_pdl = props.major >= 9
    sm_count = props.multi_processor_count
    threshold = _WARP_SCHEDULERS_PER_SM * sm_count
    print(
        f"Device: {props.name} (SM{props.major}{props.minor}, {sm_count} SMs, "
        f"multi_warp_max_tokens={threshold}, E={num_experts})"
    )

    torch.manual_seed(42)
    bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")
    c1w = compile_cutedsl(num_experts, 1, use_pdl)
    c4w = compile_cutedsl(num_experts, 4, use_pdl)

    def bench_cold(make_call, m: int) -> float:
        row_bytes = m * num_experts * 4
        target_bytes = target_mb * 1024 * 1024
        n_bufs = max(16, min(2048, (target_bytes + row_bytes - 1) // row_bytes))
        n_rep = min(n_bufs, 200) if m >= 256 else min(n_bufs, 400)
        pool = torch.randn((n_bufs, m, num_experts), dtype=torch.float32, device="cuda")
        ow = torch.empty((m, _TOPK), dtype=torch.float32, device="cuda")
        oi = torch.empty((m, _TOPK), dtype=torch.int32, device="cuda")
        make_call(pool[0], ow, oi)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(n_rep):
                make_call(pool[i], ow, oi)
        torch.cuda.synchronize()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(25):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000.0 / n_rep)
        del pool, ow, oi, g
        torch.cuda.empty_cache()
        return statistics.median(times)

    print(
        f"{'M':>5} | {'Triton(us)':>10} | {'CuTe-1W(us)':>11} | "
        f"{'CuTe-4W(us)':>11} | {'Best/Tri':>8} | 1W vs 4W"
    )
    print("-" * 74)
    for m in ms:
        t_tri = bench_cold(
            lambda x, ow, oi: run_triton(x, bias, ow, oi, 2.5, use_pdl), m
        )
        t_1w = bench_cold(lambda x, ow, oi: run_cutedsl(c1w, x, bias, ow, oi, 2.5), m)
        t_4w = bench_cold(lambda x, ow, oi: run_cutedsl(c4w, x, bias, ow, oi, 2.5), m)
        best = min(t_1w, t_4w)
        winner = "4W" if t_4w < t_1w else "1W"
        print(
            f"{m:>5} | {t_tri:>10.2f} | {t_1w:>11.2f} | {t_4w:>11.2f} | "
            f"{t_tri / best:>7.2f}x | {winner} ({max(t_1w, t_4w) / best:.2f}x)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dump", "dump-single", "ncu", "parse-ncu", "bench"),
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=("triton", "cutedsl_1w", "cutedsl_4w"),
        default="triton",
    )
    parser.add_argument("--num-experts", type=int, default=384, choices=(256, 384))
    parser.add_argument("--ms", type=str, default="1,16,256,1024,4096")
    parser.add_argument("--out-dir", type=str, default="/tmp/poiesis/dsv4_topk")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--l2-flush-mb", type=int, default=160)
    args = parser.parse_args()

    if args.mode == "parse-ncu":
        if not args.csv:
            parser.error("--csv is required for --mode parse-ncu")
        parse_ncu_csv(args.csv)
    elif args.mode == "dump":
        _mode_dump(args.num_experts, args.out_dir)
    elif args.mode == "dump-single":
        _mode_dump_single(args.num_experts, args.variant, args.out_dir)
    elif args.mode == "ncu":
        _mode_ncu(args.num_experts, _parse_int_list(args.ms))
    elif args.mode == "bench":
        _mode_bench(args.num_experts, _parse_int_list(args.ms), args.l2_flush_mb)


if __name__ == "__main__":
    main()
