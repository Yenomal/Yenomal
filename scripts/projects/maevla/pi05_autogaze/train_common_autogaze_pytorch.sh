#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="${ROOT_DIR}/RMBench/policy/pi05"

# -----------------------------
# User-tunable defaults
# -----------------------------
TRAIN_CONFIG_NAME="pi05_aloha_common_autogaze_base"
REPO_ID="${POLICY_DIR}/processed_data/your_dataset"
ASSET_ID="your_dataset"
MODEL_NAME="common_autogaze_stage_a"
STAGE="stage_a"
HISTORY_LEN=8

DEVICE="cuda"
PRECISION="bfloat16"
SEED=42

BATCH_SIZE=8
NUM_WORKERS=0
NUM_STEPS=1000
LOG_INTERVAL=10
SAVE_INTERVAL=200
WARMUP_STEPS=50

VISUAL_LR=1e-4
ACTION_LR=2e-5
SIGLIP_LR=5e-6
FINAL_LR_RATIO=0.1
WEIGHT_DECAY=1e-4
GRAD_CLIP_NORM=1.0

INIT_WEIGHT_PATH=""
CHECKPOINT_BASE_DIR="${POLICY_DIR}/checkpoints"
ASSETS_BASE_DIR="${POLICY_DIR}/assets"
WANDB_PROJECT="pi05-common-autogaze"
DISABLE_WANDB=0

export MAEVLA_AUTOGAZE_STACK_CONFIG="${ROOT_DIR}/yenomal/projects/maevla/simvla_autogaze/config/autogaze_siglip_simvla.yaml"
export PYTHONPATH="${ROOT_DIR}/yenomal:${POLICY_DIR}/src:${PYTHONPATH:-}"

NORM_STATS_PATH="${ASSETS_BASE_DIR}/${TRAIN_CONFIG_NAME}/${ASSET_ID}/norm_stats.json"

cd "${ROOT_DIR}"

if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "[train] norm stats not found at ${NORM_STATS_PATH}, computing them first..."
  python "${POLICY_DIR}/scripts/compute_norm_stats.py" \
    "${TRAIN_CONFIG_NAME}" \
    --repo-id "${REPO_ID}" \
    --asset-id "${ASSET_ID}" \
    --assets-base-dir "${ASSETS_BASE_DIR}"
fi

CMD=(
  python "${POLICY_DIR}/scripts/train_common_autogaze_pytorch.py"
  --train-config-name "${TRAIN_CONFIG_NAME}"
  --repo-id "${REPO_ID}"
  --asset-id "${ASSET_ID}"
  --model-name "${MODEL_NAME}"
  --stage "${STAGE}"
  --history-len "${HISTORY_LEN}"
  --device "${DEVICE}"
  --precision "${PRECISION}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-steps "${NUM_STEPS}"
  --log-interval "${LOG_INTERVAL}"
  --save-interval "${SAVE_INTERVAL}"
  --warmup-steps "${WARMUP_STEPS}"
  --visual-lr "${VISUAL_LR}"
  --action-lr "${ACTION_LR}"
  --siglip-lr "${SIGLIP_LR}"
  --final-lr-ratio "${FINAL_LR_RATIO}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
  --assets-base-dir "${ASSETS_BASE_DIR}"
  --wandb-project "${WANDB_PROJECT}"
)

if [[ -n "${INIT_WEIGHT_PATH}" ]]; then
  CMD+=(--init-weight-path "${INIT_WEIGHT_PATH}")
fi

if [[ "${DISABLE_WANDB}" == "1" ]]; then
  CMD+=(--disable-wandb)
fi

echo "[train] running: ${CMD[*]}"
"${CMD[@]}"
