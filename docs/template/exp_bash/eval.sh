#!/usr/bin/env bash
set -euo pipefail

# System environment
get_root() {
  git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel
}

ROOT="${YENOMAL_ROOT:-$(get_root)}"
THIRD_PARTY_ROOT="${ROOT}/third_party"
DATASETS_ROOT="${YENOMAL_DATASETS_ROOT:-${ROOT}/datasets}"
OUTPUTS_ROOT="${YENOMAL_OUTPUTS_ROOT:-${ROOT}/outputs}"

# Experiment identity.
PROJECT_NAME="maevla"
EXPERIMENT_NAME="REPLACE_ME"
TARGET_REPO="REPLACE_ME"
WORKDIR="${THIRD_PARTY_ROOT}/${TARGET_REPO}"

# Evaluation defaults.
PYTHON_BIN="${PYTHON_BIN:-python}"
POLICY_NAME="REPLACE_ME"
TASK_NAME="REPLACE_ME"
TASK_CONFIG="REPLACE_ME"
CKPT_SETTING="default"
SEED="0"
CHECKPOINT_PATH="${OUTPUTS_ROOT}/runs/${PROJECT_NAME}/${EXPERIMENT_NAME}/REPLACE_ME"
PRODUCT_DIR="${DATASETS_ROOT}/product/REPLACE_ME"
NORM_STATS_PATH="${PRODUCT_DIR}/norm_stats.json"

pushd "${WORKDIR}" >/dev/null

# Replace the command below with the actual evaluation entry for the target repo.
PYTHONWARNINGS=ignore::UserWarning \
"${PYTHON_BIN}" "REPLACE_ME.py" \
  --config "REPLACE_ME.yml" \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --ckpt_setting "${CKPT_SETTING}" \
  --seed "${SEED}" \
  --policy_name "${POLICY_NAME}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --norm_stats_path "${NORM_STATS_PATH}"

popd >/dev/null
