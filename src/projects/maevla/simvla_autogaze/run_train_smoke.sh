#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXPERIMENT_DIR="${ROOT}/src/projects/maevla/experiments/simvla_autogaze"
SIMVLA_DIR="${ROOT}/third_party/SimVLA"
SANDBOX_DIR="${EXPERIMENT_DIR}/sandbox"
SMOLVLM_SNAPSHOT="${ROOT}/.hf/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47"
AUTOGAZE_SNAPSHOT="${ROOT}/.hf/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a"
SIGLIP_SNAPSHOT="${ROOT}/.hf/hub/models--google--siglip2-base-patch16-224/snapshots/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"

export PYTHONPATH="${SANDBOX_DIR}:${ROOT}/src:${SIMVLA_DIR}:${PYTHONPATH:-}"
export MAEVLA_AUTOGAZE_STACK_CONFIG="${EXPERIMENT_DIR}/config/autogaze_siglip_simvla.yaml"
export HF_HOME="${HF_HOME:-${ROOT}/.hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${ROOT}/.hf/hub}"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${SIMVLA_DIR}/train_smolvlm.py" \
  --models "${ROOT}/datasets/checkpoints/simvla" \
  --output_dir "${ROOT}/outputs/smoke/maevla/simvla_autogaze_common_stack" \
  --train_metas_path "${ROOT}/datasets/metas/maevla/rmbench/rmbench_all_train_metas.json" \
  --smolvlm_model_path "${SMOLVLM_SNAPSHOT}" \
  --action_mode rmbench_joint \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --learning_coef 0.1 \
  --num_actions 10 \
  --iters 1 \
  --warmup_steps 0 \
  --freeze_steps 0 \
  --hidden_size 1024 \
  --depth 24 \
  --num_heads 16 \
  --num_workers 0 \
  --save_interval 1 \
  --log_interval 1 \
  --image_size 384 \
  --norm_stats_path "${ROOT}/datasets/norm_stats/maevla/rmbench/rmbench_all_train_joint_norm.json" \
  --max_grad_norm 1.0 \
  --use_autogaze_obs_encoder \
  --autogaze_model_path "${AUTOGAZE_SNAPSHOT}" \
  --autogaze_siglip_model_path "${SIGLIP_SNAPSHOT}" \
  --autogaze_history_len 8 \
  --autogaze_projector_hidden_size 1536 \
  --autogaze_gazing_ratio 0.10
