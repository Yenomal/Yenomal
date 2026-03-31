#!/usr/bin/env bash
set -euo pipefail

# Evaluate the SimVLA baseline through RMBench's policy entry.
# Edit the defaults below when you want a different task, checkpoint, or rollout setting.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}/third_party/RMBench"

# Evaluation defaults recovered from the current RMBench SimVLA policy config.
POLICY_NAME="SimVLA"
TASK_NAME="battery_try"
TASK_CONFIG="demo_clean"
CKPT_SETTING="default"
SEED="0"
CHECKPOINT_PATH="${ROOT}/outputs/runs/maevla/simvla_baseline/ckpt-20000"
SMOLVLM_MODEL_PATH="HuggingFaceTB/SmolVLM-500M-Instruct"
NORM_STATS_PATH="${ROOT}/datasets/product/rmbench_demo_clean_train48_eval2/norm_stats.json"
INSTRUCTION_TYPE="unseen"
EXECUTE_HORIZON="5"
INTEGRATION_STEPS="10"

PYTHONWARNINGS=ignore::UserWarning \
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
  --integration_steps "${INTEGRATION_STEPS}"
