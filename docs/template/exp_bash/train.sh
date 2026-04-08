#!/usr/bin/env bash
set -euo pipefail

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

# Dataset / checkpoint / output locations.
PRODUCT_DIR="${DATASETS_ROOT}/product/REPLACE_ME"
INIT_CKPT_DIR="${DATASETS_ROOT}/checkpoints/REPLACE_ME"
RUN_OUTPUT_DIR="${OUTPUTS_ROOT}/runs/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# Runtime defaults.
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="0"

mkdir -p "${RUN_OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

pushd "${WORKDIR}" >/dev/null

# Replace the command below with the actual training entry for the target repo.
"${PYTHON_BIN}" "REPLACE_ME.py" \
  --train_metas_path "${PRODUCT_DIR}/metas.json" \
  --norm_stats_path "${PRODUCT_DIR}/norm_stats.json" \
  --models "${INIT_CKPT_DIR}" \
  --output_dir "${RUN_OUTPUT_DIR}"

popd >/dev/null
