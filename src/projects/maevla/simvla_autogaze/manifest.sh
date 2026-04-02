#!/usr/bin/env bash
set -euo pipefail


# experiment:
#   name: simvla_autogaze
#   project: maevla
#   description: "SimVLA + AutoGaze observation pipeline on RMBench."
#   commit: 15698a3061c13574b688a7d9c03b6b2bafea6038

# third_party_refs:
#   simvla_maevla:
#     commit: d3fa2a2b60a1bf48bc7ced273957446290c2c9d5
#     tag: simvla-maevla-simvla-autogaze-v1
#     status: pending_snapshot
#   rmbench:
#     commit: ead77f635489228b4d621fe49af0390c33ecb4e1
#     tag: rmbench-maevla-simvla-baseline-v1
#     status: pending_snapshot


ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

SIMVLA_PATH="${ROOT}/third_party/SimVLA"
RMBENCH_PATH="${ROOT}/third_party/RMBench"

YENOMAL_COMMIT=15698a3061c13574b688a7d9c03b6b2bafea6038
SIMVLA_COMMIT=d3fa2a2b60a1bf48bc7ced273957446290c2c9d5
RMBENCH_COMMIT=ead77f635489228b4d621fe49af0390c33ecb4e1

git -C "${ROOT}" checkout "${YENOMAL_COMMIT}"
git -C "${SIMVLA_PATH}" checkout "${SIMVLA_COMMIT}"
git -C "${RMBENCH_PATH}" checkout "${RMBENCH_COMMIT}"