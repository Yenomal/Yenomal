#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
OPENPI_DIR="${ROOT}/third_party/openpi"
DATASETS_ROOT="${ROOT}/datasets"
OUTPUTS_ROOT="${ROOT}/outputs"

PROJECT_NAME="maevla"
EXPERIMENT_NAME="pi05_baseline"
CONFIG_NAME="pi05_baseline_rmbench"
RUN_NAME="default"

GPU_IDS="0"
SAVE_INTERVAL="1000"
CHECKPOINT_BASE_DIR="${OUTPUTS_ROOT}/runs/${PROJECT_NAME}/${EXPERIMENT_NAME}"
OPENPI_PI05_WEIGHT_PATH="${DATASETS_ROOT}/checkpoints/pi05"
HF_LEROBOT_HOME="${DATASETS_ROOT}/lerobot"
HF_HOME="${ROOT}/.hf"
HUGGINGFACE_HUB_CACHE="${ROOT}/.hf/hub"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OPENPI_PI05_WEIGHT_PATH="${OPENPI_PI05_WEIGHT_PATH}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME}"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES=${#GPU_ARRAY[@]}

mkdir -p "${CHECKPOINT_BASE_DIR}"

pushd "${OPENPI_DIR}" >/dev/null

uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_PROCESSES}" \
  scripts/train_pytorch.py "${CONFIG_NAME}" \
  --exp_name "${RUN_NAME}" \
  --checkpoint_base_dir "${CHECKPOINT_BASE_DIR}" \
  --save_interval "${SAVE_INTERVAL}"

popd >/dev/null
