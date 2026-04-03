ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd ${ROOT}/third_party/hdf5-to-lerobot-converter

uv run convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir /home/rui/Yenomal/datasets/raw/rmbench/data \
  --repo-id rmbench_50train \
  --output-root /home/rui/Yenomal/datasets/lerobot \
  --task "" \
  --mode image \