#!/usr/bin/env bash
set -euo pipefail

# experiment:
#   name: pi05_baseline
#   project: maevla
#   description: "OpenPI PI05 baseline on RMBench."
#   commit: c04b8907d8573809abb56f7d049c7218f199e8c4

# third_party_refs:
#   openpi:
#     commit: cd114a088f1c4ec471c20854fbc1b8e61892b5d0
#     status: pending_snapshot
#   rmbench:
#     commit: b0c02b2071287e0d0d3900c205dbe5e326ccb9d8
#     status: pending_snapshot

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
OPENPI_PATH="${ROOT}/third_party/openpi"
RMBENCH_PATH="${ROOT}/third_party/RMBench"

YENOMAL_COMMIT="c04b8907d8573809abb56f7d049c7218f199e8c4"
OPENPI_COMMIT="cd114a088f1c4ec471c20854fbc1b8e61892b5d0"
RMBENCH_COMMIT="b0c02b2071287e0d0d3900c205dbe5e326ccb9d8"

git -C "${ROOT}" checkout "${YENOMAL_COMMIT}"
git -C "${OPENPI_PATH}" checkout "${OPENPI_COMMIT}"
git -C "${RMBENCH_PATH}" checkout "${RMBENCH_COMMIT}"
