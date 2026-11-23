# DeepSeek-V4 Top-6 Router: PTX / SASS & NCU Microarchitectural Analysis (H20 SM90 & B300 SM103)

## Question

Why does the CuTe-DSL `redux.sync` + PDL implementation of DeepSeek-V4's top-6
router outperform the Triton baseline across all batch sizes, why does the
optimal warp configuration flip from 4-warp (`multi_warp_kernel`) to 1-warp
(`single_warp_kernel`) around `M = 4 * sm_count`, and how do the SASS and NCU
microarchitectural bottlenecks shift from Hopper (H20, SM90) to Blackwell Ultra
(B300, SM103)?

## Environments

| Property | H20 (SM90) | B300 (SM103) |
| --- | --- | --- |
| GPU | NVIDIA H20 (96 GB HBM) | NVIDIA B300 SXM6 AC (275 GB HBM) |
| Arch / SMs / L2 | SM90 / **78 SMs** / 60 MB L2 | SM103 / **148 SMs** / 126 MB L2 |
| `4 * sm_count` threshold | **312** | **592** |
| Max SM Boost Clock | 1.98 GHz | 2.03 GHz |
| NCU Locked Base Clock (`gpc` / `sm`) | ~1.81 / ~1.63 GHz | ~1.08 / ~1.03 GHz |
| Driver / CUDA / NCU | `570.133.20` / CUDA 13.0 / NCU `2025.3` | `580.173.02` / CUDA 13.0 / NCU `2025.3` |
| Python Stack | PyTorch `2.13.0`, Triton `3.7.1`, CUTLASS `4.6.2` | PyTorch `2.14.1`, Triton `3.8.0`, CUTLASS `4.8.0` |

Workload: `E = 384` (primary; `E = 256` also checked statically), `top_k = 6`,
`routed_scaling_factor = 2.5`, `float32` logits/bias, `int32` indices, PDL on.
All three kernels match bit-for-bit (`atol=0, rtol=0`) on both GPUs.

## Procedure & Artifacts

- **Standalone profiling harness**: [`experiments/ops/cutedsl/dsv4_topk_profile.py`](../experiments/ops/cutedsl/dsv4_topk_profile.py)
- **End-to-end reproduction script**: [`profile/ncu/run_dsv4_topk_ncu.sh`](../profile/ncu/run_dsv4_topk_ncu.sh)
- **Raw NCU CSVs (15 launches each, `M = 1, 16, 256, 1024, 4096`)**:
  - H20 SM90: [`dsv4_topk_sm90_e384_raw.csv`](../profile/ncu/dsv4_topk_sm90_e384_raw.csv)
  - B300 SM103: [`dsv4_topk_sm103_e384_raw.csv`](../profile/ncu/dsv4_topk_sm103_e384_raw.csv)
- **Static PTX (`E = 384`, SM90 & SM103)**: [`profile/ptx/dsv4_topk/`](../profile/ptx/dsv4_topk/)
- **Static `cuobjdump` SASS + NCU `--page details` & `--page source --print-source sass` (`M = 16` & `M = 4096`, SM90 & SM103)**: [`profile/sass/dsv4_topk/`](../profile/sass/dsv4_topk/)

Quick commands:

```bash
# 1. Re-parse either checked-in NCU CSV locally (CPU-only)
python experiments/ops/cutedsl/dsv4_topk_profile.py \
  --mode parse-ncu --csv profile/ncu/dsv4_topk_sm90_e384_raw.csv
python experiments/ops/cutedsl/dsv4_topk_profile.py \
  --mode parse-ncu --csv profile/ncu/dsv4_topk_sm103_e384_raw.csv

# 2. Full GPU reproduction (auto-detects sm90 / sm103 and L2 flush size)
PYTHON_BIN=/opt/sglang/bin/python3 bash profile/ncu/run_dsv4_topk_ncu.sh
```

---

## Evidence

### 1. Static PTX, SASS & Resource Usage (`E = 384`; parentheses show `E = 256`)

| Metric | Triton SM90 | Triton SM103 | CuTe-1W SM90 | CuTe-1W SM103 | CuTe-4W SM90 | CuTe-4W SM103 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Registers / thread (`REG`) | **76** (61) | **80** (74) | **48** (42) | **48** (40) | **32** (32) | **32** (32) |
| Static Shared Memory | 0 B | 0 B | 0 B | 0 B | 1,024 B | 1,024 B |
| Local / Spill | 0 B | 0 B | 0 B | 0 B | 0 B | 0 B |
| Theoretical Occupancy | 37.5% | 37.5% | 50.0% | 50.0% | 100.0% | 100.0% |
| PTX Instructions | 1,677 (1,006) | 1,677 (1,014) | 1,090 (763) | 1,090 (763) | 519 (410) | 519 (410) |
| SASS Instructions | 1,664 (1,112) | 1,608 (1,088) | 1,032 (736) | 1,032 (744) | 520 (400) | 520 (400) |
| `REDUX` / `CREDUX` (SASS) | `6 REDUX` | `6 CREDUX` | `12 REDUX` | `12 CREDUX` | `24 REDUX` | `24 CREDUX` |
| `SHFL` (SASS) | **`63 BFLY`** | **`63 BFLY`** | `6 IDX + 3 BFLY` | `6 IDX + 3 BFLY` | `12 IDX + 3 BFLY` | `12 IDX + 3 BFLY` |
| `VOTE` / `BAR` (SASS) | `0` / `0` | `0` / `0` | `0` / `0` | `0` / `0` | `6` / `1` | `6` / `1` |
| `LDG` / `STG` (SASS) | `6 LDG.E.128` / `2` | `6 LDG.E.128` / `2` | `24` / `2` | `24` / `2` | `6` / `2` | `6` / `2` |
| `MUFU` (`EX2`/`SQRT`/`RCP`) | 26 | 26 | 25 | 25 | 7 | 7 |
| New SM103 FP32 ops | — | `48 FADD2`, `24 FMNMX3` | — | 0 | — | 0 |

### 2. H20 (SM90, 78 SMs) — NCU & Pure-GPU Cold-L2 CUDA-Graph (`E = 384`)

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
| 312 | Triton / **1W** / 4W | — | 3.37 / **2.57** / 2.73 | — | — | — | — | — | — | — | — | — |
| 512 | Triton / **1W** / 4W | — | 4.29 / **3.12** / 3.41 | — | — | — | — | — | — | — | — | — |
| 1024 | Triton | 9.28 | 6.84 | 45.5 | 56.4 | 7.3 | 3.2 | 172.8 | 14.8 | 47.2 | 1,654 | 31.97 |
| 1024 | **CuTe-1W** | **6.88** | **4.75** | 34.6 | 47.0 | 5.5 | 6.1 | 231.7 | 15.2 | 45.8 | 1,022 | 20.14 |
| 1024 | CuTe-4W | 7.36 | 4.98 | 42.6 | 59.2 | 11.3 | 26.9 | 215.4 | 53.4 | 62.6 | 1,607 | 25.31 |
| 4096 | Triton | 23.04 | 19.25 | 72.3 | **80.8** | 10.4 | 4.3 | 274.4 | 31.0 | 61.1 | 1,654 | 31.97 |
| 4096 | **CuTe-1W** | **15.07** | **12.41** | 62.1 | **72.1** | **8.4** | 8.7 | **418.8** | 38.6 | 60.0 | **1,022** | 20.14 |
| 4096 | CuTe-4W | 18.62 | 15.34 | 64.9 | **73.5** | 14.1 | **32.4** | 338.5 | **82.0** | **73.7** | 1,607 | 25.31 |

### 3. B300 (SM103, 148 SMs) — NCU & Pure-GPU Cold-L2 CUDA-Graph (`E = 384`)

> Note: NCU `--clock-control base` locks B300 to ~1.08 GHz GPC / ~1.03 GHz SM
> (~0.5× of B300's 2.03 GHz boost clock, vs H20's ~1.81 / 1.98 GHz), so compare
> absolute µs across GPUs using boost-clock `Graph (µs)` and compare cycles /
> relative ratios inside NCU.

| M | Kernel | NCU Dur (µs) | Cycles | Graph (µs) | SM% | ALU% | LSU% | ADU% | DRAM (GB/s) | AchOcc% | Iss% | Inst/Tok | Thd/Inst |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Triton | 10.53 | 11,479 | 2.78 | 0.0 | 8.4 | 1.1 | 0.3 | 3.0 | 1.6 | 24.3 | 1,592 | 31.97 |
| 1 | CuTe-1W | 8.38 | 9,125 | 1.86 | 0.0 | 7.5 | 0.9 | 0.4 | 2.7 | 1.7 | 24.1 | 1,033 | 20.10 |
| 1 | **CuTe-4W** | **7.49** | **8,120** | **1.52** | 0.0 | 14.2 | 2.5 | 2.7 | 1.9 | 5.3 | 16.2 | 1,646 | 25.10 |
| 16 | Triton | 10.98 | 11,954 | 2.80 | 0.6 | 8.9 | 1.2 | 0.4 | 5.0 | 1.6 | 24.6 | 1,592 | 31.97 |
| 16 | CuTe-1W | 8.10 | 8,796 | 1.90 | 0.4 | 7.7 | 0.9 | 0.4 | 5.6 | 1.4 | 26.9 | 1,033 | 20.10 |
| 16 | **CuTe-4W** | **7.46** | **8,109** | **1.60** | 0.6 | 14.6 | 2.6 | 2.7 | 5.0 | 5.0 | 17.4 | 1,646 | 25.10 |
| 256 | Triton | 11.01 | 11,984 | 3.16 | 8.0 | 14.4 | 1.9 | 0.6 | 38.4 | 2.6 | 24.7 | 1,592 | 31.97 |
| 256 | CuTe-1W | 9.12 | 9,903 | 2.20 | 5.9 | 13.3 | 1.5 | 0.7 | 45.4 | 2.8 | 25.4 | 1,033 | 20.10 |
| 256 | **CuTe-4W** | **7.87** | **8,569** | **1.97** | 9.5 | 23.3 | 4.2 | 4.4 | 51.5 | 8.9 | 25.1 | 1,646 | 25.10 |
| 312 | Triton / 1W / **4W** | — | — | 3.49 / 2.54 / **2.31** | — | — | — | — | — | — | — | — | — |
| 512 | Triton / 1W / **4W** | — | — | 3.68 / 2.64 / **2.63** | — | — | — | — | — | — | — | — | — |
| 640 | Triton / 1W / **4W** | — | — | 4.42 / 3.12 / **2.86** | — | — | — | — | — | — | — | — | — |
| 1024 | Triton | 12.26 | 13,339 | 4.55 | 30.4 | 47.9 | 6.2 | 1.9 | 130.8 | 9.9 | 36.3 | 1,592 | 31.97 |
| 1024 | **CuTe-1W** | **9.98** | **10,818** | **3.31** | 22.1 | 42.9 | 4.9 | 2.2 | 159.6 | 9.4 | 38.4 | 1,033 | 20.10 |
| 1024 | CuTe-4W | 11.14 | 12,110 | 3.39 | 29.0 | 53.1 | 9.5 | 8.6 | 142.3 | 31.2 | 53.8 | 1,646 | 25.10 |
| 4096 | Triton | 23.84 | 25,912 | 11.16 | 60.8 | **76.0** | 9.9 | 2.5 | 265.1 | 27.5 | 55.3 | 1,592 | 31.97 |
| 4096 | **CuTe-1W** | **16.77** | **18,182** | **7.33** | 48.9 | **69.9** | **8.0** | 3.2 | **376.4** | 29.9 | 58.4 | **1,033** | 20.10 |
| 4096 | CuTe-4W | 20.64 | 22,434 | 9.05 | 55.2 | **72.8** | 13.0 | **10.1** | 305.4 | **72.9** | **70.2** | 1,646 | 25.10 |

Global L1 load sectors are identical across all kernels and both GPUs:
`96.0 sectors/token` (`3,072 B`), `0.0` local memory spills.

### 4. Warp Stall Comparison: H20 (SM90) vs B300 (SM103) (Cycles / Issued Instruction)

| M | Kernel | Arch | `long_sb` | **`short_sb`** | `wait` | `barrier` | `not_sel` | `math_thr` | `mio_thr` | Sum CPI |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | Triton | SM90 / SM103 | 0.45 / 0.35 | **0.59 / 0.72** | 0.92 / 1.05 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 3.38 / 3.89 |
| 16 | CuTe-1W | SM90 / SM103 | 0.66 / 0.67 | **0.56 / 0.21** | 0.99 / 1.05 | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 | 3.64 / 3.78 |
| 16 | CuTe-4W | SM90 / SM103 | 1.52 / 1.64 | **1.32 / 0.55** | 1.51 / 1.67 | 0.09 / 0.09 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 6.17 / 6.08 |
| 4096 | Triton | SM90 / SM103 | 0.44 / 0.53 | **0.60 / 0.71** | 0.93 / 1.06 | 0.00 / 0.00 | 2.76 / 2.42 | 2.24 / 1.89 | 0.01 / 0.08 | 8.06 / 7.99 |
| 4096 | CuTe-1W | SM90 / SM103 | 1.17 / 0.69 | **0.47 / 0.27** | 1.01 / 1.07 | 0.00 / 0.00 | 3.54 / 2.76 | 2.51 / 1.80 | 0.04 / 0.04 | 9.93 / 8.20 |
| 4096 | CuTe-4W | SM90 / SM103 | 1.53 / 1.23 | **1.18 / 0.19** | 1.58 / 1.72 | 0.39 / 0.32 | 6.65 / 6.45 | 4.63 / 4.71 | 0.20 / 0.40 | 17.33 / 17.01 |

---

## Interpretation

1. **Large-batch `dsv4_topk` is ALU/issue-bound on both Hopper and Blackwell Ultra.**
   At `M = 4096`, DRAM throughput stays under ~11% of peak on H20 (`419 GB/s`)
   and under ~5% of peak on B300 (`376 GB/s` at base clock), while `ALU%` reaches
   **70–81%** and `not_sel + math_thr` dominate warp stalls. Saturated runtime
   scales directly with `Inst/Tok / Issue_Throughput`.
2. **Blackwell Ultra SM103 upgrades `REDUX` → `CREDUX`, widening CuTe-DSL's lead at decode.**
   - On SM103, `redux.sync` compiles to `CREDUX.MAX` / `CREDUX.MIN`.
   - Compared to SM90 `REDUX`, `ADU%` in `CuTe-4W` (`M = 4096`) drops **3.2×**
     (`32.4% → 10.1%`), and `short_sb` drops **2.7–6.2×** (`0.56 → 0.21` CPI in
     `1W@M=16`; `1.18 → 0.19` CPI in `4W@M=4096`).
   - Meanwhile, Triton `3.8.0` on SM103 still emits **`63 × SHFL.BFLY`** (and
     zero FP32 `CREDUX`) while its register count climbs to **`80`** (`E=384`) /
     **`74`** (`E=256`). Even with SM103 `FADD2` (48) and `FMNMX3` (24) trimming
     Triton's SASS from `1,664 → 1,608`, Triton's `short_sb` *rises* to
     `0.71–0.84` CPI. As a result, CuTe-4W's decode speedup over Triton widens
     from **1.54× on H20 (`1.77 vs 2.71 µs`)** to **1.83× on B300 (`1.52 vs 2.78 µs`)**.
3. **Dual-GPU proof of why `multi_warp_max_tokens = 4 * sm_count` beats a hardcoded `256`.**
   - On **H20 (78 SMs, `4 * sm_count = 312`)**: 4W and 1W meet at `M = 256`
     (`2.49 vs 2.53 µs`) and 1W takes over at `M = 312` (`2.57 vs 2.73 µs`).
   - On **B300 (148 SMs, `4 * sm_count = 592`)**: at `M = 256` and `M = 312`,
     1-warp CTAs fill less than half of B300's 592 warp schedulers, so **4W
     still beats 1W by 10–12%** (`1.97 vs 2.20 µs` at `M = 256`, `2.31 vs 2.54 µs`
     at `M = 312`) and stays competitive through `M = 640` (`2.86 vs 3.12 µs`).
     Once saturated at `M = 1024–4096`, 4W's **+59% dynamic instruction overhead**
     (`1,646` vs `1,033` `Inst/Tok`) + `barrier` (`0.32` CPI) + `mio_thr`
     (`0.40` CPI) make 1W **1.23× faster** (`7.33 vs 9.05 µs`) despite 1W's
     lower occupancy (`29.9%` vs `72.9%`).
4. **SM scaling at `M = 4096`**:
   B300 (`7.33 µs`) beats H20 (`12.41 µs`) by **1.69×**, tracking the
   `148 / 78 = 1.90×` SM count increase on this ALU-bound kernel.

## Limits

- Measurements cover the main hot path (`has_padding=False, has_vl=False,
  is_int64=False, use_pdl=True`).
- Isolated `ncu` measurements do not include PDL overlap savings with upstream
  router GEMM (`ll_bf16`) and downstream FP8 quantization.

## Next question

On Blackwell SM100/SM103, does replacing the float-to-uint32 bit-flip mapping
with native hardware FP32 `redux.sync.max.f32` (`CREDUX.MAX.F32`) shave off the
remaining bit-manipulation ALU instructions and further reduce `Inst/Tok` at
`M = 4096`?
