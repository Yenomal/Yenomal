#!/usr/bin/env bash
set -euo pipefail

# One-shot entrypoint for producing the current RMBench product.
# Run this when raw data is ready and metas/norm_stats need to be regenerated together.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

bash "${ROOT}/scripts/projects/maevla/data/build_rmbench_meta.sh"
bash "${ROOT}/scripts/projects/maevla/data/build_rmbench_norm.sh"
