#!/usr/bin/env bash
set -euo pipefail

# experiment:
#   name: REPLACE_ME
#   project: maevla
#   description: "Describe the experiment briefly."
#   monorepo_tag: REPLACE_ME

# third_party_refs:
#   repo1:
#     name: REPLACE_ME
#     commit: REPLACE_ME
#     tag: REPLACE_ME
#   repo2:
#     name: REPLACE_ME
#     commit: REPLACE_ME
#     tag: REPLACE_ME

get_root() {
  git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel
}

ROOT="${YENOMAL_ROOT:-$(get_root)}"
THIRD_PARTY_ROOT="${ROOT}/third_party"

YENOMAL_COMMIT="REPLACE_ME"
REPO1_NAME="REPLACE_ME"
REPO1_COMMIT="REPLACE_ME"
REPO2_NAME="REPLACE_ME"
REPO2_COMMIT="REPLACE_ME"

REPO1_PATH="${THIRD_PARTY_ROOT}/${REPO1_NAME}"
REPO2_PATH="${THIRD_PARTY_ROOT}/${REPO2_NAME}"

git -C "${ROOT}" checkout "${YENOMAL_COMMIT}"
git -C "${REPO1_PATH}" checkout "${REPO1_COMMIT}"
git -C "${REPO2_PATH}" checkout "${REPO2_COMMIT}"