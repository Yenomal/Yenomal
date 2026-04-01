#!/usr/bin/env bash
set -euo pipefail

# Temporary development shell for the Yenomal workspace.
# Usage:
#   source tempwork/dev_shell.sh
# or:
#   bash tempwork/dev_shell.sh

ROOT="/home/rui/Yenomal"
TEMPWORK="${ROOT}/tempwork"

mkdir -p "${TEMPWORK}"

ensure_link() {
  local target="$1"
  local link_path="$2"
  if [[ -L "${link_path}" ]]; then
    return 0
  fi
  if [[ -e "${link_path}" ]]; then
    echo "[yenomal] skip ${link_path} because a non-link path already exists"
    return 0
  fi
  ln -s "${target}" "${link_path}"
}

# Multi-repo style development view.
# ensure_link ../third_party/SimVLA "${TEMPWORK}/SimVLA"
ensure_link ../third_party/RMBench "${TEMPWORK}/RMBench"
ensure_link ../third_party/AutoGaze "${TEMPWORK}/AutoGaze"
ensure_link ../src "${TEMPWORK}/yenomal"
# ensure_link ../src/projects/maevla "${TEMPWORK}/maevla"
# ensure_link ../scripts/projects/maevla "${TEMPWORK}/maevla_scripts"

# Python resolves package names from the parent directory.
# With TEMPWORK on PYTHONPATH, imports can use:
#   from SimVLA.models ...
#   from AutoGaze.autogaze ...
#   from RMBench.script ...
#   from yenomal.common ...
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party:${PYTHONPATH:-}"

echo "[yenomal] ROOT=${ROOT}"
echo "[yenomal] TEMPWORK=${TEMPWORK}"
echo "[yenomal] PYTHONPATH=${PYTHONPATH}"

# If sourced, only export the environment.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  return 0
fi

# If executed directly, open an interactive shell with the environment already set.
exec /bin/bash --noprofile --norc -i
