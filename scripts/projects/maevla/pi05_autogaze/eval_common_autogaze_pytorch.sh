#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="${ROOT_DIR}/RMBench/policy/pi05"
PORT=19105

# -----------------------------
# User-tunable defaults
# -----------------------------
TASK_NAME="battery_try"
TASK_CONFIG="demo_clean"
TRAIN_CONFIG_NAME="pi05_aloha_common_autogaze_base"
MODEL_NAME="common_autogaze_stage_a"
CHECKPOINT_ID="latest"
SEED=0
GPU_ID=0
PI0_STEP=50
DEVICE="cuda"
CHECKPOINT_ROOT="${POLICY_DIR}/checkpoints"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MAEVLA_AUTOGAZE_STACK_CONFIG="${ROOT_DIR}/yenomal/projects/maevla/simvla_autogaze/config/autogaze_siglip_simvla.yaml"
export PYTHONPATH="${ROOT_DIR}/yenomal:${POLICY_DIR}/src:${PYTHONPATH:-}"

cd "${ROOT_DIR}/RMBench"

python script/policy_model_server.py \
  --port "${PORT}" \
  --config policy/pi05/deploy_policy.yml \
  --overrides \
  --policy_name pi05 \
  --train_config_name "${TRAIN_CONFIG_NAME}" \
  --model_name "${MODEL_NAME}" \
  --checkpoint_id "${CHECKPOINT_ID}" \
  --pi0_step "${PI0_STEP}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --device "${DEVICE}" &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

python script/eval_policy_client.py \
  --port "${PORT}" \
  --config policy/pi05/deploy_policy.yml \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --train_config_name "${TRAIN_CONFIG_NAME}" \
  --model_name "${MODEL_NAME}" \
  --checkpoint_id "${CHECKPOINT_ID}" \
  --ckpt_setting "${MODEL_NAME}" \
  --seed "${SEED}" \
  --pi0_step "${PI0_STEP}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --device "${DEVICE}" \
  --policy_name pi05
