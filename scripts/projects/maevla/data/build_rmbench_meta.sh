#!/usr/bin/env bash
set -euo pipefail

# Build RMBench train/eval metadata from the current raw dataset.
# Edit only the variables in the "Dataset recipe" section when changing the split.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SIMVLA_DIR="${ROOT}/third_party/SimVLA"

# Dataset recipe: this defines which raw data slice becomes the reusable product.
DATA_DIR="${ROOT}/datasets/raw/rmbench/data"
TASK_CONFIG="demo_clean"
TRAIN_EPISODES_PER_TASK=48
EVAL_EPISODES_PER_TASK=2
PRODUCT_DIR="${ROOT}/datasets/product/rmbench_demo_clean_train48_eval2"
TRAIN_OUTPUT="${PRODUCT_DIR}/metas.json"
EVAL_OUTPUT="${PRODUCT_DIR}/eval_metas.json"

mkdir -p "$(dirname "${TRAIN_OUTPUT}")" "$(dirname "${EVAL_OUTPUT}")"

# Train split metadata used by training scripts.
python "${SIMVLA_DIR}/create_rmbench_meta.py" \
  --data_dir "${DATA_DIR}" \
  --task_config "${TASK_CONFIG}" \
  --split train \
  --train_episodes_per_task "${TRAIN_EPISODES_PER_TASK}" \
  --eval_episodes_per_task "${EVAL_EPISODES_PER_TASK}" \
  --output "${TRAIN_OUTPUT}"

# Eval split metadata kept next to the train product for later evaluation.
python "${SIMVLA_DIR}/create_rmbench_meta.py" \
  --data_dir "${DATA_DIR}" \
  --task_config "${TASK_CONFIG}" \
  --split eval \
  --train_episodes_per_task "${TRAIN_EPISODES_PER_TASK}" \
  --eval_episodes_per_task "${EVAL_EPISODES_PER_TASK}" \
  --output "${EVAL_OUTPUT}"
