#!/usr/bin/env bash
set -euo pipefail

# Compute normalization statistics for the same RMBench product recipe used by training.
# Keep this script aligned with build_rmbench_meta.sh so both outputs describe the same split.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
SIMVLA_DIR="${ROOT}/third_party/SimVLA"

# Dataset recipe: change these only when creating a new product version.
DATA_DIR="${ROOT}/datasets/raw/rmbench/data"
TASK_CONFIG="demo_clean"
TRAIN_EPISODES_PER_TASK=48
EVAL_EPISODES_PER_TASK=2
PRODUCT_DIR="${ROOT}/datasets/product/rmbench_demo_clean_train48_eval2"
OUTPUT_PATH="${PRODUCT_DIR}/norm_stats.json"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

python "${SIMVLA_DIR}/compute_rmbench_norm_stats.py" \
  --data_dir "${DATA_DIR}" \
  --task_config "${TASK_CONFIG}" \
  --split train \
  --train_episodes_per_task "${TRAIN_EPISODES_PER_TASK}" \
  --eval_episodes_per_task "${EVAL_EPISODES_PER_TASK}" \
  --output "${OUTPUT_PATH}"
