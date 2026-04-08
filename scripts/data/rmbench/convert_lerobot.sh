ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RAW_DIR="${ROOT}/datasets/raw/rmbench/data"
OUTPUT_ROOT="${ROOT}/datasets/lerobot"

cd "${ROOT}/third_party/hdf5-to-lerobot-converter"

uv run convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir "${RAW_DIR}" \
  --repo-id rmbench_50train \
  --output-root "${OUTPUT_ROOT}" \
  --task "" \
  --mode image \
