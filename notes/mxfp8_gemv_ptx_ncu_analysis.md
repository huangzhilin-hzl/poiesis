# MXFP8 Decode GEMV (`Mxfp8Gemv` vs `HopperMxfp8Gemv`): PTX / SASS & NCU Microarchitectural Analysis (H20 SM90)

## Question

1. How does replacing SM80 `cp.async` weight+scale SMEM staging (`Mxfp8Gemv`) with SM90 zero-weight-SMEM register-ring streaming + Hopper PDL (`HopperMxfp8Gemv`) + wave-aware non-power-of-two tiling change PTX/SASS instructions, SMEM footprint, occupancy, warp stalls, and Cold-L2 latency on Hopper (SM90)?
2. Does `HopperMxfp8Gemv` extend the crossover against BF16 cuBLAS past `GEMV_MAX_ROWS = 24` to `M = 48` on H20, or does `M = 24` remain optimal?

## Environment

| Property | Value |
| --- | --- |
| GPU | NVIDIA H20 (96 GB HBM3, 4.0 TB/s peak) |
| Arch / SMs / L2 / Opt-in SMEM | SM90 / **78 SMs** / 60 MB L2 / 232,448 B |
| Driver / CUDA / NCU | CUDA 13.0 / Nsight Compute `2025.3.0.0` (`--set full --cache-control all --clock-control base`) |
| Python Stack | PyTorch `2.14.0+cu130`, `nvidia-cutlass-dsl==4.8.0`, `flashinfer-python==0.7.4`, `cupti-python==13.0.1` |

Workload: DeepSeek-V4.1-Flash TP8 per-rank dense projections (`qkv_a 1792x5120`, `q_b 4096x1280`, `q_b_idx 8192x1280`, `wo_b 5120x1024`, `shared_w13 576x5120`, `shared_w2 5120x288`), BF16 `x`/`y`, MXFP8 E4M3 weights + UE8M0 block scales (`32x32`), `M in {1, 6, 12, 24, 32, 48}`. Both kernels pass the exact FP64 + BF16-ulp accuracy bound on all shapes.

## Procedure & Artifacts

- **Standalone profiling harness (Before + After + BF16 cuBLAS)**: [`experiments/ops/cutedsl/mxfp8_gemv_profile.py`](../experiments/ops/cutedsl/mxfp8_gemv_profile.py)
- **End-to-end reproduction script**: [`profile/ncu/run_mxfp8_gemv_ncu.sh`](../profile/ncu/run_mxfp8_gemv_ncu.sh)
- **Raw NCU CSV (14 launches, 7 `(shape, M)` pairs x Before/After)**: [`profile/ncu/mxfp8_gemv_sm90_raw.csv`](../profile/ncu/mxfp8_gemv_sm90_raw.csv)
- **Static PTX dumps (Before `ampere_sm90_*` & After `hopper_sm90_*`)**: [`profile/ptx/mxfp8_gemv/`](../profile/ptx/mxfp8_gemv/)
- **Static `cuobjdump` SASS + NCU `--page details` & `--page source --print-source sass`**: [`profile/sass/mxfp8_gemv/`](../profile/sass/mxfp8_gemv/)

Quick commands:

```bash
# 1. Re-parse the checked-in NCU CSV locally (CPU-only)
python experiments/ops/cutedsl/mxfp8_gemv_profile.py \
  --mode parse-ncu --csv profile/ncu/mxfp8_gemv_sm90_raw.csv

# 2. Full GPU reproduction (dumps PTX/SASS, runs NCU --set full, exports & benchmarks)
PYTHON_BIN=/tmp/vllm-venv/bin/python bash profile/ncu/run_mxfp8_gemv_ncu.sh
```

---

## Evidence

### 1. Static PTX, SASS & Resource Usage (`cuobjdump --dump-resource-usage`)

| Case | Kernel | Config (`wn,wk,grp`)s | `REG` | `LOCAL` / `STACK` | Dyn SMEM | PTX Inst | SASS Inst | `LDGSTS` / `LDGDEPBAR` | `LDG` | `LDS` | `HMMA` | PDL SASS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qkv_a M=24` | Before (`Ampere`) | `(8,2,16)s5` | 41 | 0 / 0 | **182.5 KB** | 1,394 | 1,192 | 23 / 17 | 10 | 100 | 96 | — |
| `qkv_a M=24` | After (`Hopper`) | `(2,4,10)s4` | 55 | 0 / 0 | **63.0 KB** (`-65%`) | 1,131 | 1,080 | 15 / 1 | 24 | 50 | 60 | `ACQBULK` + `PREEXIT` |
| `q_b_idx M=24` | Before (`Ampere`) | `(8,2,10)s2` | 42 | 0 / 0 | **114.6 KB** | 956 | 880 | 15 / 11 | 4 | 79 | 60 | — |
| `q_b_idx M=24` | After (`Hopper`) | `(4,4,10)s1` | 54 | 0 / 0 | **63.0 KB** (`-45%`) | 927 | 832 | 8 / 1 | 20 | 40 | 60 | `ACQBULK` + `PREEXIT` |
| `wo_b M=24` | Before (`Ampere`) | `(8,2,16)s1` | 42 | 0 / 0 | **182.5 KB** | 1,289 | 1,072 | 23 / 17 | 0 | 87 | 96 | — |
| `wo_b M=24` | After (`Hopper`) | `(5,2,16)s1` | 54 | 0 / 0 | **50.7 KB** (`-72%`) | 1,312 | 1,128 | 10 / 1 | 32 | 58 | 96 | `ACQBULK` + `PREEXIT` |
| `wo_b M=48` | Before (`Ampere`) | `(4,4,4)s2` | 69 | 0 / 0 | 85.2 KB | 666 | 640 | 11 / 5 | 4 | 56 | 48 | — |
| `wo_b M=48` | After (`Hopper`) | `(5,2,16)s1` | 74 | 0 / 0 | 101.4 KB | 1,750 | 1,536 | 20 / 1 | 32 | 108 | 192 | `ACQBULK` + `PREEXIT` |
| `shared_w2 M=24` | Before (`Ampere`) | `(8,1,9)s1` | 40 | 0 / 0 | **50.9 KB** | 810 | 744 | 14 / 10 | 0 | 57 | 54 | — |
| `shared_w2 M=24` | After (`Hopper`) | `(5,3,3)s1` | 53 | 0 / 0 | **24.2 KB** (`-52%`) | 338 | 336 | 2 / 1 | 6 | 18 | 18 | `ACQBULK` + `PREEXIT` |

**Key SASS observations** (from [`hopper_sm90_wo_b_5120x1024_m24_cuobjdump.sass`](../profile/sass/mxfp8_gemv/hopper_sm90_wo_b_5120x1024_m24_cuobjdump.sass)):
1. **Side-effect-ordered weight prefetch stays pinned before PDL wait**: `load_weights` (`ld.global.cg.v4.u32` with `has_side_effects=True`) compiles to `LDG.E.128.STRONG.GPU` at `0x06a0`, `0x06e0`, `0x0700`, directly followed by `ACQBULK` (`griddepcontrol.wait`) at `0x0710`, and only then `LDGSTS.E.BYPASS.128` for `x` starting at `0x0770`.
2. **Scale loads (`load_u32`, `has_side_effects=False`) are sunk by `ptxas`**: `LDG.E.CONSTANT` appears at `0x0c20–0x0c40` (after `x` `LDGSTS`, still before the first `HMMA` at `0x13a0`), confirming only weight loads are strictly ordered ahead of `ACQBULK`.
3. **Zero register spill even at `x_tiles = 6`**: `LOCAL = 0, STACK = 0` across every config (`REG = 40..44` at `x_tiles=1`, `53..55` at `x_tiles=3`, `74..77` at `x_tiles=6`; even forcing `prefetch_stages = 3` at `x_tiles=6` yields `REG = 74..77, LOCAL = 0`).
4. **PDL epilogue trigger**: `griddepcontrol.launch_dependents` lowers to `PREEXIT` (`0x4190`), directly after the reduction barrier (`0x4170`) and before the BF16 `STG.E.64` output stores (`0x43e0`, `0x45c0`).

---

### 2. Nsight Compute Full Metrics (`--set full --cache-control all --clock-control base`, H20 SM90)

| Case | Kernel | Grid | Thr | Dur (µs) | SM% | Tensor% | LSU% | DRAM (GB/s) | Reg | SMEM (KB) | ThOcc% | AchOcc% | Iss% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qkv_a M=6` | Ampere | `(28,8,1)` | 512 | 8.93 | 30.2 | 7.1 | 23.6 | 1,039.8 | 32 | 53.0 | 100.0 | 60.4 | 47.0 |
| `qkv_a M=6` | Hopper | `(112,2,1)` | 256 | 9.12 | 29.7 | 6.3 | **13.2** | 1,018.5 | 44 | 42.6 | 62.5 | 32.5 | 40.0 |
| `qkv_a M=24` | Ampere | `(14,5,1)` | 512 | 13.12 | 36.4 | 13.6 | 14.3 | 722.5 | 41 | 183.6 | 25.0 | 24.8 | 25.8 |
| `qkv_a M=24` | Hopper | `(56,4,1)` | 256 | 13.28 | 36.7 | 12.2 | 13.0 | 713.6 | 55 | **64.1** | **37.5** | **29.9** | **34.2** |
| `q_b_idx M=6` | Ampere | `(128,1,1)` | 512 | 8.29 | 30.0 | 8.9 | 20.2 | 1,273.6 | 31 | 104.4 | 50.0 | 38.7 | 47.3 |
| `q_b_idx M=6` | **Hopper** | `(128,1,1)` | 512 | **7.90** | 29.2 | 8.5 | **12.5** | **1,335.4** | 40 | **22.1** | **75.0** | 39.5 | 44.2 |
| `q_b_idx M=24` | Ampere | `(64,2,1)` | 512 | 15.33 | 36.5 | 12.8 | 16.4 | 691.8 | 42 | 115.7 | 50.0 | 37.3 | 28.4 |
| `q_b_idx M=24` | **Hopper** | `(128,1,1)` | 512 | **12.90** | **45.5** | **15.6** | **13.8** | **822.3** | 54 | **64.1** | 50.0 | 36.9 | **34.8** |
| `wo_b M=24` | Ampere | `(40,1,1)` | 512 | 10.50 | 27.3 | 16.7 | 16.3 | 508.0 | 42 | 183.6 | 25.0 | 24.8 | 30.6 |
| `wo_b M=24` | **Hopper** | `(64,1,1)` | 320 | **8.83** | **33.0** | 12.7 | **10.1** | **603.9** | 54 | **51.8** | **46.9** | 15.5 | 24.2 |
| `wo_b M=48` | Ampere | `(80,2,1)` | 512 | 21.28 | 25.6 | 10.3 | 14.0 | 252.6 | 69 | 86.3 | 25.0 | 24.6 | 19.9 |
| `wo_b M=48` | **Hopper** | `(64,1,1)` | 320 | **13.50** | **42.8** | **15.2** | **9.9** | **399.1** | 74 | 102.5 | **31.2** | 15.6 | 19.9 |
| `shared_w2 M=24` | Ampere | `(40,1,1)` | 256 | 5.02 | 14.9 | 11.7 | 13.6 | 300.3 | 40 | 52.0 | 50.0 | 12.4 | 25.9 |
| `shared_w2 M=24` | **Hopper** | `(64,1,1)` | 480 | **4.96** | **17.4** | 8.5 | 12.6 | 302.9 | 53 | **25.3** | 46.9 | **23.2** | **26.4** |

#### Memory Traffic & Warp Stall Shifts

| Case | Kernel | `Gld` | `Ldgsts` | `Sld` | `SmemWF` | `Loc` | `barrier` CPI | `lg_thr` CPI | `mio_thr` CPI | `math_thr` CPI | `long_sb` CPI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_b_idx M=24` | Ampere | 3,072 | 28,672 | 110,592 | 450,021 | 0 | **3.45** | 0.18 | 0.48 | 8.52 | 1.96 |
| `q_b_idx M=24` | Hopper | 40,960 | 15,360 | **67,584** | **306,560** | 0 | **0.98** | **0.00** | 0.30 | 8.65 | 2.00 |
| `wo_b M=24` | Ampere | 0 | 14,720 | 53,120 | 192,887 | 0 | 0.84 | **0.68** | 0.06 | **7.11** | 0.21 |
| `wo_b M=24` | Hopper | 20,480 | 6,144 | **32,640** | **144,418** | 0 | 1.00 | **0.00** | 0.03 | **3.50** | 1.84 |
| `wo_b M=48` | Ampere | 3,840 | 28,160 | 99,840 | 463,875 | 0 | **3.39** | 0.11 | 0.59 | 6.68 | **2.06** |
| `wo_b M=48` | Hopper | 20,480 | 12,288 | **65,280** | **283,425** | 0 | **1.65** | **0.00** | **0.06** | 6.44 | **0.66** |

Key microarchitectural takeaways:
1. **Eliminating weight SMEM removes `lg_throttle` (`0.68 -> 0.00` CPI) and cuts SMEM wavefronts by 25%–49%**: In `Mxfp8Gemv`, issuing 10–16 back-to-back `LDGSTS` + `LDGDEPBAR` groups fills the LG queue and routes every weight byte through SMEM (`LDS` drops 35%–65% in `HopperMxfp8Gemv`).
2. **Lower SMEM eliminates unnecessary split-K**: On `q_b_idx M=24` and `wo_b M=48`, `Mxfp8Gemv` had to split K (`s=2`) to fit weight SMEM, causing `barrier` stalls of `3.45` / `3.39` CPI and extra global partial stores/loads (`3,072` / `3,840` `Gld` + ticket atomics). `HopperMxfp8Gemv` runs `s=1`, halving `barrier` CPI (`0.98` / `1.65`) and lifting SM throughput by `+9`–`+17` pp.
3. **Non-power-of-two `warps_n=5` / `warps_k=3` fixes SM underutilization on `N=5120`**: On `wo_b M=24`, power-of-two `warps_n=8` leaves 38 of 78 SMs idle (`40` CTAs); `warps_n=5` launches `64` CTAs (`82%` of SMs), cutting `math_pipe_throttle` in half (`7.11 -> 3.50` CPI).
4. **Why isolated `M=1..6` microbench has higher `long_sb` CPI (`1.97 -> 4.99`)**: Without an upstream producer kernel before `ACQBULK`, `HopperMxfp8Gemv` only issues `prefetch_stages=3` weight loads ahead of `x`, vs `Mxfp8Gemv` queuing all `10` blocks via `cp.async`. In end-to-end serving (DeepSeek-V4.1-Flash TP8), PDL overlaps those initial weight loads with the upstream producer's tail, recovering an extra ~4% (`1.089x` in-model GEMV trace vs `1.049x` isolated `M <= 24` Cold-L2 geomean).

---

### 3. Cold-L2 CUPTI Benchmark: `Mxfp8Gemv` vs `HopperMxfp8Gemv` vs BF16 cuBLAS (H20 SM90)

Per-layer dense projection total (`qkv_a + q_b_idx + wo_b + shared_w13 + shared_w2`, `cold_l2_cache=True`, CUDA Graph, median of 100 iters):

| `M` | `Mxfp8Gemv` (µs) | `HopperMxfp8Gemv` (µs) | BF16 cuBLAS (µs) | `Hopper / Ampere` | `Hopper / cuBLAS` | Optimal Dispatch |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 28.97 | **28.06** | 40.94 | `1.032x` | **`1.459x`** | `HopperMxfp8Gemv` |
| 6 | 29.99 | **29.38** | 41.27 | `1.021x` | **`1.405x`** | `HopperMxfp8Gemv` |
| 12 | 38.28 | **35.23** | 42.48 | `1.087x` | **`1.206x`** | `HopperMxfp8Gemv` |
| 24 | 47.76 | **42.31** | 44.78 | `1.129x` | **`1.058x`** | `HopperMxfp8Gemv` (beats cuBLAS; Ampere lost!) |
| 32 | 58.04 | 51.01 | **48.37** | `1.138x` | `0.948x` (`-5.5%`) | **BF16 cuBLAS** |
| 48 | 82.18 | 63.56 | **52.67** | `1.293x` | `0.829x` (`-20.7%`) | **BF16 cuBLAS** |

- **Cold-L2 geomean (production `M <= 24`, 24 cases)**: `Hopper / Ampere = 1.0486x`, `Hopper / BF16 cuBLAS = 1.2497x`.
- **Cold-L2 geomean (all 36 cases `M = 1..48`)**: `Hopper / Ampere = 1.0911x`.
- **Conclusion on `GEMV_MAX_ROWS`**: Keep `GEMV_MAX_ROWS = 24`. At `M >= 32`, Hopper WGMMA BF16 cuBLAS overtakes `mma.sync` FP8-decode GEMV (`48.37` vs `51.01 µs` at `M=32`, `52.67` vs `63.56 µs` at `M=48`); at `M = 24`, `HopperMxfp8Gemv` flips the crossover from a loss under `Mxfp8Gemv` (`47.76` vs `44.78 µs`) back to a win (`42.31` vs `44.78 µs`, `1.058x`).
