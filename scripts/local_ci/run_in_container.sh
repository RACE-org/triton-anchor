#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${LOCAL_CI_CONFIG:-${SCRIPT_DIR}/config.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

sha="${1:?usage: run_in_container.sh <commit-sha>}"
LOCAL_CI_CONTAINER="${LOCAL_CI_CONTAINER:-triton-anchor-dev}"

pass_env=(
  GITEE_REPO_URL GITEE_BRANCH ANCHOR_DIR WORKSPACE
  BACKEND_PROFILE EXPECTED_TRITON_BACKEND BACKEND_PATH BACKEND_ENVSETUP BACKEND_ENVSETUP_ARGS BACKEND_TEST_COMMAND
  RUN_FLAGGEMS_TESTS FLAGGEMS_CLONE_DIR FLAGGEMS_REF FLAGGEMS_PIP_PACKAGES FLAGGEMS_TEST_COMMAND INSTALL_FLAGGEMS_PACKAGES
  LLVM_BUILD_DIR PPL_ROOT PACKAGE_TOOL PYTHON_BIN SOURCE_ENVSETUP FRONTEND_BUILD_COMMAND LOCAL_CI_ARTIFACT_ROOT
)

docker_args=()
for name in "${pass_env[@]}"; do
  if [[ -n "${!name:-}" ]]; then
    docker_args+=(-e "${name}=${!name}")
  fi
done

docker exec \
  "${docker_args[@]}" \
  -e LOCAL_CI_COMMIT="${sha}" \
  "${LOCAL_CI_CONTAINER}" \
  bash -lc '"${ANCHOR_DIR:-/workspace/triton-anchor}/scripts/local_ci/run_delivery_local.sh" "${LOCAL_CI_COMMIT}"'
