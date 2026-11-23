# DeepSeek-V4 Top-6 Router: PTX / SASS & NCU Microarchitectural Analysis

## Question

Why does the CuTe-DSL `redux.sync` + PDL implementation of DeepSeek-V4's top-6
router outperform the Triton baseline across all batch sizes, and why does the
optimal warp configuration flip from 4-warp (`multi_warp_kernel`) to 1-warp
(`single_warp_kernel`) right around `M = 4 * sm_count` (`312` on H20)?

## Environment

- **Hardware**: 1× NVIDIA H20 (96 GB HBM, SM90, 78 SMs, 60 MB L2, 1.98 GHz boost)
- **Driver / CUDA / NCU**: Driver `570.133.20`, CUDA 13.0, Nsight Compute `2025.3.0.0`
- **Python stack**: PyTorch `2.13.0+cu130`, Triton `3.7.1`, CUTLASS CuTe-DSL `4.6.2`
- **Workload**: `E = 384` (primary; `E = 256` also checked statically), `top_k = 6`,
  `routed_scaling_factor = 2.5`, `float32` logits/bias, `int32` indices, SM90 PDL on.
  All three kernels match bit-for-bit (`atol=0, rtol=0`).

## Procedure & Artifacts

- **Standalone profiling harness**: [`experiments/ops/cutedsl/dsv4_topk_profile.py`](../experiments/ops/cutedsl/dsv4_topk_profile.py)
- **End-to-end reproduction script**: [`profile/ncu/run_dsv4_topk_ncu.sh`](../profile/ncu/run_dsv4_topk_ncu.sh)
- **Raw NCU CSV (all 15 launches, `M = 1, 16, 256, 1024, 4096`)**: [`profile/ncu/dsv4_topk_sm90_e384_raw.csv`](../profile/ncu/dsv4_topk_sm90_e384_raw.csv)
- **Static PTX (`E = 384`)**:
  - [`triton_sm90_dsv4_topk_e384.ptx`](../profile/ptx/dsv4_topk/triton_sm90_dsv4_topk_e384.ptx)
  - [`cutedsl_sm90_dsv4_topk_1w_e384.ptx`](../profile/ptx/dsv4_topk/cutedsl_sm90_dsv4_topk_1w_e384.ptx)
  - [`cutedsl_sm90_dsv4_topk_4w_e384.ptx`](../profile/ptx/dsv4_topk/cutedsl_sm90_dsv4_topk_4w_e384.ptx)
- **Static `cuobjdump` SASS + NCU `--page details` & `--page source --print-source sass` (`M = 16` and `M = 4096`)**:
  - [`profile/sass/dsv4_topk/`](../profile/sass/dsv4_topk/)

Quick commands:

```bash
# 1. Re-parse the checked-in NCU CSV locally (CPU-only)
python experiments/ops/cutedsl/dsv4_topk_profile.py \
  --mode parse-ncu --csv profile/ncu/dsv4_topk_sm90_e384_raw.csv

# 2. Full GPU reproduction (PTX/SASS dump + NCU full + Cold-L2 CUDA-Graph bench)
PYTHON_BIN=/opt/sglang/bin/python3 bash profile/ncu/run_dsv4_topk_ncu.sh
```

## Evidence

### 1. Static PTX, SASS & Resource Usage (`E = 384`; parentheses show `E = 256`)

| Metric | Triton (1W, `BLOCK_N=512`) | CuTe-1W (`single_warp`) | CuTe-4W (`multi_warp`) |
| --- | ---: | ---: | ---: |
| Registers / thread (`REG`) | **76** (61) | **48** (42) | **32** (32) |
| Static Shared Memory | 0 B | 0 B | 1,024 B |
| Local / Spill (`LOCAL`/`STACK`) | 0 B | 0 B | 0 B |
| Theoretical Occupancy (limiter) | **37.5%** (`LimReg=24` blk/SM) | **50.0%** (`LimBlk=32` SM90 cap) | **100.0%** (`16 blk × 4W`) |
| PTX Instructions | **1,677** (1,006) | **1,090** (763) | **519** (410) |
| SASS Instructions | **1,664** (1,112) | **1,032** (736) | **520** (400) |
| `REDUX` (SASS) | `6` (`MIN.S32` only) | `12` (`6 MAX + 6 MIN`) | `24` (`12 MAX + 12 MIN`) |
| `SHFL` (SASS) | **`63`** (all `SHFL.BFLY`) | **`9`** (`6 IDX + 3 BFLY`) | **`15`** (`12 IDX + 3 BFLY`) |
| `VOTE` / `BAR` (SASS) | `0` / `0` | `0` / `0` (`6 BSSY/BSYNC`) | `6 VOTE.ANY` / `1 BAR.SYNC` |
| `LDG` / `STG` (SASS) | `6 LDG.E.128` / `2` | `24` (`12 CONST + 12 E`) / `2` | `6` (`3 CONST + 3 E`) / `2` |
| `LDS` / `STS` (SASS) | `0` / `0` | `0` / `0` | `3` / `3` |
| `MUFU` (`EX2`/`SQRT`/`RCP`) | `26` (PTX `33`, DCE'd masked tail) | `25` (`12 + 12 + 1`) | `7` (`3 + 3 + 1`) |
| `FP32` / `CMP+SEL` / `INT+BIT` | `471` / `667` / `303` | `237` / `489` / `97` | `66` / `192` / `118` |

### 2. NCU Overview, Pipe Utilization & Cold-L2 Latency (`E = 384`)

- `NCU Dur`: single cold launch under `--cache-control all --clock-control base`
  (SM ~1.63 GHz / GPC ~1.81 GHz, includes cold I-cache & L2).
- `Graph`: pure-GPU boost-clock (1.98 GHz) median latency over rotating input
  buffers (>160 MB > 60 MB L2) captured inside a CUDA Graph.

| M | Kernel | NCU Dur (µs) | Graph (µs) | SM% | ALU% | LSU% | ADU% | DRAM (GB/s) | AchOcc% | Iss% | Inst/Tok | Thd/Inst |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Triton | 5.41 | 2.71 | 0.1 | 8.5 | 1.1 | 0.5 | 6.0 | 1.6 | 26.7 | 1,654 | 31.97 |
| 1 | CuTe-1W | 4.38 | 1.93 | 0.1 | 6.8 | 0.8 | 0.9 | 5.1 | 1.6 | 23.9 | 1,022 | 20.14 |
| 1 | **CuTe-4W** | **3.84** | **1.77** | 0.1 | 10.2 | 2.0 | 4.9 | 3.7 | 5.1 | 13.3 | 1,607 | 25.31 |
| 16 | Triton | 5.31 | 2.75 | 1.2 | 8.8 | 1.1 | 0.5 | 10.4 | 1.6 | 26.2 | 1,654 | 31.97 |
| 16 | CuTe-1W | 4.42 | 1.96 | 0.9 | 7.4 | 0.9 | 1.0 | 10.3 | 1.6 | 25.4 | 1,022 | 20.14 |
| 16 | **CuTe-4W** | **3.94** | **1.81** | 1.3 | 11.9 | 2.3 | 5.8 | 9.4 | 4.9 | 14.2 | 1,607 | 25.31 |
| 256 | Triton | 5.89 | 3.32 | 18.6 | 29.8 | 3.8 | 1.7 | 72.0 | 5.1 | 27.2 | 1,654 | 31.97 |
| 256 | CuTe-1W | 4.67 | 2.53 | 13.6 | 24.1 | 2.8 | 3.2 | 88.6 | 5.1 | 24.6 | 1,022 | 20.14 |
| 256 | **CuTe-4W** | **4.58** | **2.49** | 17.8 | 31.3 | 6.0 | 15.5 | 88.7 | 16.4 | 33.6 | 1,607 | 25.31 |
| 312 | Triton | — | 3.37 | — | — | — | — | — | — | — | — | — |
| 312 | **CuTe-1W** | — | **2.57** | — | — | — | — | — | — | — | — | — |
| 312 | CuTe-4W | — | 2.73 | — | — | — | — | — | — | — | — | — |
| 512 | Triton | — | 4.29 | — | — | — | — | — | — | — | — | — |
| 512 | **CuTe-1W** | — | **3.12** | — | — | — | — | — | — | — | — | — |
| 512 | CuTe-4W | — | 3.41 | — | — | — | — | — | — | — | — | — |
| 1024 | Triton | 9.28 | 6.84 | 45.5 | 56.4 | 7.3 | 3.2 | 172.8 | 14.8 | 47.2 | 1,654 | 31.97 |
| 1024 | **CuTe-1W** | **6.88** | **4.75** | 34.6 | 47.0 | 5.5 | 6.1 | 231.7 | 15.2 | 45.8 | 1,022 | 20.14 |
| 1024 | CuTe-4W | 7.36 | 4.98 | 42.6 | 59.2 | 11.3 | 26.9 | 215.4 | 53.4 | 62.6 | 1,607 | 25.31 |
| 4096 | Triton | 23.04 | 19.25 | 72.3 | **80.8** | 10.4 | 4.3 | 274.4 | 31.0 | 61.1 | 1,654 | 31.97 |
| 4096 | **CuTe-1W** | **15.07** | **12.41** | 62.1 | **72.1** | **8.4** | 8.7 | **418.8** | 38.6 | 60.0 | **1,022** | 20.14 |
| 4096 | CuTe-4W | 18.62 | 15.34 | 64.9 | **73.5** | 14.1 | **32.4** | 338.5 | **82.0** | **73.7** | 1,607 | 25.31 |

Global memory traffic is identical across all three kernels:
`96.0` L1 load sectors/token (`3,072 B` = 384 bias + 384 logits), `0.0` local
memory ops, and `~6.46 MB` DRAM read at `M = 4096`.

### 3. Warp Stall Breakdown (Cycles per Issued Instruction)

| M | Kernel | `long_sb` | `short_sb` | `wait` | `barrier` | `not_sel` | `math_thr` | `mio_thr` | Sum CPI |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | Triton | 0.45 | 0.59 | 0.92 | 0.00 | 0.00 | 0.00 | 0.00 | 3.38 |
| 16 | CuTe-1W | 0.66 | 0.56 | 0.99 | 0.00 | 0.00 | 0.01 | 0.00 | 3.64 |
| 16 | CuTe-4W | 1.52 | 1.32 | 1.51 | 0.09 | 0.00 | 0.00 | 0.00 | 6.17 |
| 256 | CuTe-1W | 0.73 | 0.61 | 0.99 | 0.00 | 0.00 | 0.01 | 0.00 | 3.74 |
| 256 | CuTe-4W | 1.65 | 1.41 | 1.55 | 0.22 | 0.59 | 0.37 | 0.03 | 7.53 |
| 1024 | CuTe-1W | 0.95 | 0.56 | 1.00 | 0.00 | 1.15 | 0.72 | 0.02 | 5.65 |
| 1024 | CuTe-4W | 1.67 | 1.22 | 1.56 | **0.30** | **4.07** | **2.77** | **0.62** | 13.58 |
| 4096 | Triton | 0.44 | 0.60 | 0.93 | 0.00 | 2.76 | 2.24 | 0.01 | 8.06 |
| 4096 | CuTe-1W | 1.17 | 0.47 | 1.01 | **0.00** | 3.54 | 2.51 | 0.04 | **9.93** |
| 4096 | CuTe-4W | 1.53 | 1.18 | 1.58 | **0.39** | **6.65** | **4.63** | **0.20** | **17.33** |

## Interpretation

1. **Large-batch `dsv4_topk` is ALU/issue-bound, not HBM-bandwidth-bound.**
   At `M = 4096`, DRAM utilization stays below ~11% (`274–419 GB/s` vs 4 TB/s
   peak), while `ALU%` hits **72%–81%** and `not_sel + math_thr` dominate warp
   stalls. Runtime in the saturated regime therefore scales directly with
   `Inst/Tok / Issue_Throughput`.
2. **Why CuTe-DSL beats Triton by 1.31–1.55×.**
   Triton emits `63 × SHFL.BFLY` (30 for 6 float `tl.max` reductions, 30 for 6
   `tl.sum` scalar weight extractions, 3 for final normalization) and uses `76`
   registers/thread (capping theoretical occupancy at `37.5%`). Mapping floats
   to order-preserving `uint32` unlocks hardware `redux.sync.max.u32` and direct
   `SHFL.IDX` weight broadcasts, cutting `SHFL.BFLY` from `63 → 3`, registers
   from `76 → 48` (lifting occupancy to the SM90 1-warp ceiling of `50%`), and
   dynamic warp instructions/token from `1,654 → 1,022` (−38.2%). Even though
   CuTe-1W issues 24 scalar `LDG`s vs Triton's 6 `LDG.E.128`s, L1 load sectors
   are identical (`96/token`) and CuTe-1W's `LSU%` is actually *lower* (`8.4%`
   vs `10.4%`) because Triton's 63 butterfly shuffles consume LSU/MIO bandwidth.
3. **Why higher occupancy loses at `M = 4096` (4W vs 1W crossover at `M ≈ 312`).**
   - At `M = 1..16`, SMs are idle (`not_sel = 0`, `math_thr = 0`). Splitting a
     token across 4 parallel warps cuts critical-path depth from `1,022` to
     ~`550` instructions (`~350` Stage 1 + `~200` Stage 2), making 4W fastest
     (`1.77 µs` vs `1.93 µs` vs `2.71 µs`).
   - At `M = 4096`, 4W achieves **82.0% occupancy** and **73.7% issue active**
     vs 1W's `38.6%` / `60.0%`, yet runs **24% slower** (`18.62` vs `15.07 µs`).
     Because 4W executes **+57% more dynamic warp instructions per token**
     (`1,607` vs `1,022`) plus `barrier` (`0.39` CPI), `mio_thr` (`0.20–0.62`
     CPI), and `ADU% = 32.4%`, its effective issue cost is
     `(1607 / 0.737) / (1022 / 0.600) = 1.28×` (matching the measured `1.24×`).
   - Once `M > 4 * sm_count` (`312` on H20), 1-warp CTAs already saturate every
     SM's 4 warp schedulers, matching the empirical crossover between `M = 256`
     (4W +1%) and `M = 312` (1W +6%).
4. **Remaining 1W inefficiency exposed by `Thd/Inst`.**
   CuTe-1W averages `20.14 / 32` active threads per instruction (vs Triton's
   `31.97 / 32`) because each round's `if winner == ...` rescan activates only 1
   lane while the whole warp pays the issue cycles for `_find_best_local` (11
   compares).

## Limits

- Measurements cover the main hot path (`has_padding=False, has_vl=False,
  is_int64=False, use_pdl=True`) on H20 SM90.
- Isolated `ncu` measurements do not capture PDL overlap savings with the
  upstream router GEMM (`ll_bf16`) and downstream FP8 group-quant kernel.

## Next question

Can a 2-warp (`64`-thread) intermediate regime close any gap near `M = 256–512`,
or does `CuTe-1W` with vectorized `ld.global.v4.f32` (matching Triton's `6`
`LDG.E.128` while keeping `redux.sync`) reduce `long_sb` / `lg_throttle` further
at `M = 4096`?
