#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

bash "${ROOT}/src/projects/maevla/simvla_autogaze/run_train_smoke.sh"
