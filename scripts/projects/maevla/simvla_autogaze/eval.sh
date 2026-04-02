#!/usr/bin/env bash
set -euo pipefail

# Evaluate the SimVLA + AutoGaze variant through RMBench's policy entry.
# The defaults below come from the current RMBench AutoGaze policy config.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PROJECT_DIR="${ROOT}/src/projects/maevla/simvla_autogaze"
COMMON_DIR="${ROOT}/src/common"
SIMVLA_DIR="${ROOT}/third_party/SimVLA"
cd "${ROOT}/third_party/RMBench"

POLICY_NAME="SimVLA_AutoGaze"
TASK_NAME="battery_try"
TASK_CONFIG="demo_clean"
CKPT_SETTING="default"
SEED="0"
CHECKPOINT_PATH="${ROOT}/outputs/runs/maevla/simvla_autogaze/ckpt-20000"
SMOLVLM_MODEL_PATH="HuggingFaceTB/SmolVLM-500M-Instruct"
NORM_STATS_PATH="${ROOT}/datasets/product/rmbench_demo_clean_train48_eval2/norm_stats.json"
INSTRUCTION_TYPE="unseen"
EXECUTE_HORIZON="5"
INTEGRATION_STEPS="10"
AUTOGAZE_MODEL_PATH="nvidia/AutoGaze"
AUTOGAZE_SIGLIP_MODEL_PATH="google/siglip2-base-patch16-224"
AUTOGAZE_HISTORY_LEN="8"
AUTOGAZE_PROJECTOR_HIDDEN_SIZE="1536"
AUTOGAZE_GAZING_RATIO="0.1"

PYTHONWARNINGS=ignore::UserWarning \
PYTHONPATH="${PROJECT_DIR}:${COMMON_DIR}:${SIMVLA_DIR}:${PYTHONPATH:-}" \
MAEVLA_VISUAL_GAZE_CONFIG="${PROJECT_DIR}/config/autogaze_siglip_simvla.yaml" \
python script/eval_policy.py --config "policy/${POLICY_NAME}/deploy_policy.yml" \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --ckpt_setting "${CKPT_SETTING}" \
  --seed "${SEED}" \
  --policy_name "${POLICY_NAME}" \
  --instruction_type "${INSTRUCTION_TYPE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --smolvlm_model_path "${SMOLVLM_MODEL_PATH}" \
  --norm_stats_path "${NORM_STATS_PATH}" \
  --execute_horizon "${EXECUTE_HORIZON}" \
  --integration_steps "${INTEGRATION_STEPS}" \
  --autogaze_model_path "${AUTOGAZE_MODEL_PATH}" \
  --autogaze_siglip_model_path "${AUTOGAZE_SIGLIP_MODEL_PATH}" \
  --autogaze_history_len "${AUTOGAZE_HISTORY_LEN}" \
  --autogaze_projector_hidden_size "${AUTOGAZE_PROJECTOR_HIDDEN_SIZE}" \
  --autogaze_gazing_ratio "${AUTOGAZE_GAZING_RATIO}"
