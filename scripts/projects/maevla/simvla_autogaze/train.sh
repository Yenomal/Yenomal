#!/usr/bin/env bash
set -euo pipefail

# Train the SimVLA + AutoGaze variant on the current default RMBench product.
# The defaults below mirror the current codebase assumptions and can be revised for a new experiment version.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SIMVLA_DIR="${ROOT}/third_party/SimVLA"
PRODUCT_DIR="${ROOT}/datasets/product/rmbench_demo_clean_train48_eval2"

# Experiment defaults.
GPU_IDS="0"
OUTPUT_DIR="${ROOT}/outputs/runs/maevla/simvla_autogaze"
INIT_CKPT="${ROOT}/datasets/checkpoints/simvla"
SMOLVLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"
AUTOGAZE_MODEL_PATH="nvidia/AutoGaze"
AUTOGAZE_SIGLIP_MODEL_PATH="google/siglip2-base-patch16-224"
TRAIN_META_PATH="${PRODUCT_DIR}/metas.json"
NORM_STATS_PATH="${PRODUCT_DIR}/norm_stats.json"

# Training hyperparameters for the AutoGaze variant.
LEARNING_RATE="1e-4"
LEARNING_COEF="0.1"
BATCH_SIZE="8"
NUM_ACTIONS="10"
ITERS="100000"
FREEZE_STEPS="1000"
WARMUP_STEPS="0"
SAVE_INTERVAL="5000"
LOG_INTERVAL="20"
NUM_WORKERS="4"
MAX_GRAD_NORM="1.0"
HIDDEN_SIZE="1024"
DEPTH="24"
NUM_HEADS="16"
IMAGE_SIZE="384"
MAIN_PROCESS_PORT="29514"
MIXED_PRECISION="bf16"
AUTOGAZE_HISTORY_LEN="8"
AUTOGAZE_PROJECTOR_HIDDEN_SIZE="1536"
AUTOGAZE_GAZING_RATIO="0.10"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TF_CPP_MIN_LOG_LEVEL=2

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES=${#GPU_ARRAY[@]}

mkdir -p "${OUTPUT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME="${HF_HOME:-${ROOT}/.hf}" \
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${ROOT}/.hf/hub}" \
accelerate launch \
  --num_processes="${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  "${SIMVLA_DIR}/train_smolvlm.py" \
  --models "${INIT_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_metas_path "${TRAIN_META_PATH}" \
  --smolvlm_model_path "${SMOLVLM_MODEL}" \
  --action_mode rmbench_joint \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --learning_coef "${LEARNING_COEF}" \
  --num_actions "${NUM_ACTIONS}" \
  --iters "${ITERS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --freeze_steps "${FREEZE_STEPS}" \
  --hidden_size "${HIDDEN_SIZE}" \
  --depth "${DEPTH}" \
  --num_heads "${NUM_HEADS}" \
  --num_workers "${NUM_WORKERS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --log_interval "${LOG_INTERVAL}" \
  --image_size "${IMAGE_SIZE}" \
  --norm_stats_path "${NORM_STATS_PATH}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --use_autogaze_obs_encoder \
  --autogaze_model_path "${AUTOGAZE_MODEL_PATH}" \
  --autogaze_siglip_model_path "${AUTOGAZE_SIGLIP_MODEL_PATH}" \
  --autogaze_history_len "${AUTOGAZE_HISTORY_LEN}" \
  --autogaze_projector_hidden_size "${AUTOGAZE_PROJECTOR_HIDDEN_SIZE}" \
  --autogaze_gazing_ratio "${AUTOGAZE_GAZING_RATIO}"
