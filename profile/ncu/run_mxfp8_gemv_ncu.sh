#!/usr/bin/env bash
# Reproduce the complete MXFP8 decode GEMV PTX, SASS, NCU, and Cold-L2
# profiling pipeline (Before: SM80 cp.async Mxfp8Gemv vs After: SM90
# zero-weight-SMEM register-streaming + PDL HopperMxfp8Gemv).
#
# Usage:
#   PYTHON_BIN=/tmp/vllm-venv/bin/python bash profile/ncu/run_mxfp8_gemv_ncu.sh
#   OUT_ROOT=/tmp/poiesis/mxfp8_gemv bash profile/ncu/run_mxfp8_gemv_ncu.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

VENV_DIR="${VIRTUAL_ENV:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
MS_BENCH="${MS_BENCH:-1,6,12,24,32,48}"
SM_TAG="${SM_TAG:-$("$PYTHON_BIN" -c 'import torch; m, n = torch.cuda.get_device_capability(); print(f"sm{m}{n}")')}"
OUT_ROOT="${OUT_ROOT:-/tmp/poiesis/mxfp8_gemv_${SM_TAG}}"

TARGET_SCRIPT="$PROJECT_ROOT/experiments/ops/cutedsl/mxfp8_gemv_profile.py"

mkdir -p "$OUT_ROOT/ptx_sass" "$OUT_ROOT/ncu" "$OUT_ROOT/export"

echo "=== Target arch: ${SM_TAG} ==="
echo "=== 1. Dump PTX, CUBIN, SASS & static instruction breakdown ==="
"$PYTHON_BIN" "$TARGET_SCRIPT" \
  --mode dump \
  --out-dir "$OUT_ROOT/ptx_sass" | tee "$OUT_ROOT/dump_summary.txt"

echo "=== 2. Capture Nsight Compute full metrics (Cold-L2, base clock) ==="
REP="$OUT_ROOT/ncu/mxfp8_gemv_${SM_TAG}"
ncu \
  --profile-from-start off \
  --set full \
  --cache-control all \
  --clock-control base \
  --force-overwrite \
  -o "$REP" \
  "$PYTHON_BIN" "$TARGET_SCRIPT" \
  --mode ncu

echo "=== 3. Export NCU raw CSV, details, and annotated SASS ==="
RAW_CSV="$OUT_ROOT/ncu/mxfp8_gemv_${SM_TAG}_raw.csv"
ncu --import "${REP}.ncu-rep" --page raw --csv > "$RAW_CSV"

# Launch order from NCU_CASES (7 cases x 2 variants = 14 launches):
#   0,1   : qkv_a      (1792x5120) M=6   (ampere, hopper)
#   2,3   : qkv_a      (1792x5120) M=24  (ampere, hopper)
#   4,5   : q_b_idx    (8192x1280) M=6   (ampere, hopper)
#   6,7   : q_b_idx    (8192x1280) M=24  (ampere, hopper)
#   8,9   : wo_b       (5120x1024) M=24  (ampere, hopper)
#   10,11 : wo_b       (5120x1024) M=48  (ampere, hopper)
#   12,13 : shared_w2  (5120x288)  M=24  (ampere, hopper)
export_launch() {
  local id="$1"
  local base="$2"
  ncu --import "${REP}.ncu-rep" --launch-skip "$id" --launch-count 1 \
    --page details --print-details all > "$OUT_ROOT/export/${base}.txt"
  ncu --import "${REP}.ncu-rep" --launch-skip "$id" --launch-count 1 \
    --page source --print-source sass > "$OUT_ROOT/export/${base}_sass.txt"
}

export_launch 2  "ampere_${SM_TAG}_qkv_a_1792x5120_m24"
export_launch 3  "hopper_${SM_TAG}_qkv_a_1792x5120_m24"
export_launch 6  "ampere_${SM_TAG}_q_b_idx_8192x1280_m24"
export_launch 7  "hopper_${SM_TAG}_q_b_idx_8192x1280_m24"
export_launch 8  "ampere_${SM_TAG}_wo_b_5120x1024_m24"
export_launch 9  "hopper_${SM_TAG}_wo_b_5120x1024_m24"
export_launch 10 "ampere_${SM_TAG}_wo_b_5120x1024_m48"
export_launch 11 "hopper_${SM_TAG}_wo_b_5120x1024_m48"
export_launch 12 "ampere_${SM_TAG}_shared_w2_5120x288_m24"
export_launch 13 "hopper_${SM_TAG}_shared_w2_5120x288_m24"

echo "=== 4. Parse NCU CSV summary tables ==="
"$PYTHON_BIN" "$TARGET_SCRIPT" --mode parse-ncu --csv "$RAW_CSV" | tee "$OUT_ROOT/ncu_summary.txt"

echo "=== 5. Cold-L2 CUPTI CUDA-Graph benchmark (Ampere vs Hopper vs BF16 cuBLAS) ==="
"$PYTHON_BIN" "$TARGET_SCRIPT" --mode bench --ms "$MS_BENCH" | tee "$OUT_ROOT/bench_summary.txt"
