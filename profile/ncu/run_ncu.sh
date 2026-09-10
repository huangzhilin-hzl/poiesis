#! /usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

VENV_DIR="${VIRTUAL_ENV:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"

TARGET_SCRIPT="$PROJECT_ROOT/${1:-experiments/ops/cutedsl/vector_add.py}"
#TARGET_SCRIPT="$PROJECT_ROOT/${1:-experiments/ops/cutile/vector_add.py}"

CUDA_VISIBLE_DEVICES=0 ncu \
  --set detailed \
  --kernel-name-base function \
  --kernel-name 'regex:vector_add' \
  --launch-count 1 \
  -f \
  -o /tmp/poiesis/vector_add \
  "$PYTHON_BIN" "$TARGET_SCRIPT"

ncu --import /tmp/poiesis/vector_add.ncu-rep \
  --page details --print-details all | tee "/tmp/poiesis/cutedsl_vector_add.txt"
#  --page details --print-details all | tee "/tmp/poiesis/cutile_vector_add_n8192_t1024.txt"
#  --page details --print-details all | tee "/tmp/poiesis/cutile_vector_add.txt"


ncu --import /tmp/poiesis/vector_add.ncu-rep \
  --page source --print-source sass | tee "/tmp/poiesis/cutedsl_vector_add_sass.txt"
#  --page source --print-source sass | tee "/tmp/poiesis/cutile_vector_add_sass.txt"
#  --page source --print-source sass | tee "/tmp/poiesis/cutile_vector_add_sass.txt"
