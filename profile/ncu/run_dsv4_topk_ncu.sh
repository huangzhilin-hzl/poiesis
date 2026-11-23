#!/usr/bin/env bash
# Reproduce the complete DeepSeek-V4 top-6 PTX, SASS, NCU, and Cold-L2
# profiling pipeline (Triton baseline vs CuTe-DSL single-warp vs multi-warp).
#
# Usage:
#   PYTHON_BIN=/opt/sglang/bin/python3 bash profile/ncu/run_dsv4_topk_ncu.sh
#   NUM_EXPERTS=384 OUT_ROOT=/tmp/poiesis/dsv4_topk bash profile/ncu/run_dsv4_topk_ncu.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

VENV_DIR="${VIRTUAL_ENV:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
NUM_EXPERTS="${NUM_EXPERTS:-384}"
MS_NCU="${MS_NCU:-1,16,256,1024,4096}"
MS_BENCH="${MS_BENCH:-1,16,256,312,512,640,1024,4096}"
SM_TAG="${SM_TAG:-$("$PYTHON_BIN" -c 'import torch; m, n = torch.cuda.get_device_capability(); print(f"sm{m}{n}")')}"
L2_FLUSH_MB="${L2_FLUSH_MB:-$("$PYTHON_BIN" -c 'import torch; p = torch.cuda.get_device_properties(0); print(max(160, int(p.L2_cache_size * 2.5 // (1024 * 1024))))')}"
OUT_ROOT="${OUT_ROOT:-/tmp/poiesis/dsv4_topk_${SM_TAG}_e${NUM_EXPERTS}}"

TARGET_SCRIPT="$PROJECT_ROOT/experiments/ops/cutedsl/dsv4_topk_profile.py"

mkdir -p "$OUT_ROOT/ptx_sass" "$OUT_ROOT/ncu" "$OUT_ROOT/export"

echo "=== Target arch: ${SM_TAG}, E=${NUM_EXPERTS}, L2_FLUSH_MB=${L2_FLUSH_MB} ==="
echo "=== 1. Dump PTX, CUBIN, SASS & static instruction breakdown ==="
"$PYTHON_BIN" "$TARGET_SCRIPT" \
  --mode dump \
  --num-experts "$NUM_EXPERTS" \
  --out-dir "$OUT_ROOT/ptx_sass"

echo "=== 2. Capture Nsight Compute full metrics (Cold-L2, base clock) ==="
REP="$OUT_ROOT/ncu/dsv4_topk_${SM_TAG}_e${NUM_EXPERTS}"
ncu \
  --profile-from-start off \
  --set full \
  --cache-control all \
  --clock-control base \
  --force-overwrite \
  -o "$REP" \
  "$PYTHON_BIN" "$TARGET_SCRIPT" \
  --mode ncu \
  --num-experts "$NUM_EXPERTS" \
  --ms "$MS_NCU"

echo "=== 3. Export NCU raw CSV, details, and annotated SASS ==="
RAW_CSV="$OUT_ROOT/ncu/dsv4_topk_${SM_TAG}_e${NUM_EXPERTS}_raw.csv"
ncu --import "${REP}.ncu-rep" --page raw --csv > "$RAW_CSV"

# Default launch order for MS_NCU=1,16,256,1024,4096 (3 kernels per M):
#   M=16:   3=Triton, 4=CuTe-1W, 5=CuTe-4W
#   M=4096: 12=Triton, 13=CuTe-1W, 14=CuTe-4W
export_launch() {
  local id="$1"
  local base="$2"
  ncu --import "${REP}.ncu-rep" --launch-skip "$id" --launch-count 1 \
    --page details --print-details all > "$OUT_ROOT/export/${base}.txt"
  ncu --import "${REP}.ncu-rep" --launch-skip "$id" --launch-count 1 \
    --page source --print-source sass > "$OUT_ROOT/export/${base}_sass.txt"
}

if [[ "$MS_NCU" == "1,16,256,1024,4096" ]]; then
  export_launch 3  "triton_${SM_TAG}_dsv4_topk_e${NUM_EXPERTS}_m16"
  export_launch 4  "cutedsl_${SM_TAG}_dsv4_topk_1w_e${NUM_EXPERTS}_m16"
  export_launch 5  "cutedsl_${SM_TAG}_dsv4_topk_4w_e${NUM_EXPERTS}_m16"
  export_launch 12 "triton_${SM_TAG}_dsv4_topk_e${NUM_EXPERTS}_m4096"
  export_launch 13 "cutedsl_${SM_TAG}_dsv4_topk_1w_e${NUM_EXPERTS}_m4096"
  export_launch 14 "cutedsl_${SM_TAG}_dsv4_topk_4w_e${NUM_EXPERTS}_m4096"
fi

echo "=== 4. Parse NCU CSV summary tables ==="
"$PYTHON_BIN" "$TARGET_SCRIPT" --mode parse-ncu --csv "$RAW_CSV"

echo "=== 5. Pure-GPU Cold-L2 CUDA-Graph latency benchmark ==="
"$PYTHON_BIN" "$TARGET_SCRIPT" \
  --mode bench \
  --num-experts "$NUM_EXPERTS" \
  --ms "$MS_BENCH" \
  --l2-flush-mb "$L2_FLUSH_MB"
