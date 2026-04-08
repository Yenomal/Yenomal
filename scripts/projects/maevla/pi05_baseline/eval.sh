#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RMBENCH_DIR="${ROOT}/third_party/RMBench"
OPENPI_DIR="${ROOT}/third_party/openpi"
OPENPI_SRC_DIR="${OPENPI_DIR}/src"
OPENPI_CLIENT_SRC_DIR="${OPENPI_DIR}/packages/openpi-client/src"
OUTPUTS_ROOT="${ROOT}/outputs"

PROJECT_NAME="maevla"
EXPERIMENT_NAME="pi05_baseline"
POLICY_NAME="OpenPI"
OPENPI_CONFIG_NAME="pi05_baseline_rmbench"
RUN_NAME="default"

TASK_NAME="battery_try"
TASK_CONFIG="demo_clean"
CKPT_SETTING="default"
SEED="0"
INSTRUCTION_TYPE="unseen"
EXECUTE_HORIZON="1"
PYTHON_BIN="${PYTHON_BIN:-python}"

OPENPI_CHECKPOINT_DIR="${OUTPUTS_ROOT}/runs/${PROJECT_NAME}/${EXPERIMENT_NAME}/${OPENPI_CONFIG_NAME}/${RUN_NAME}"
HF_HOME="${ROOT}/.hf"
HUGGINGFACE_HUB_CACHE="${ROOT}/.hf/hub"

export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"

pushd "${RMBENCH_DIR}" >/dev/null

PYTHONWARNINGS=ignore::UserWarning \
PYTHONPATH="${OPENPI_SRC_DIR}:${OPENPI_CLIENT_SRC_DIR}:${PYTHONPATH:-}" \
"${PYTHON_BIN}" script/eval_policy.py --config "policy/${POLICY_NAME}/deploy_policy.yml" \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --ckpt_setting "${CKPT_SETTING}" \
  --seed "${SEED}" \
  --policy_name "${POLICY_NAME}" \
  --instruction_type "${INSTRUCTION_TYPE}" \
  --openpi_config_name "${OPENPI_CONFIG_NAME}" \
  --openpi_checkpoint_dir "${OPENPI_CHECKPOINT_DIR}" \
  --execute_horizon "${EXECUTE_HORIZON}"

popd >/dev/null
