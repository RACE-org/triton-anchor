#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${LOCAL_CI_CONFIG:-${SCRIPT_DIR}/config.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

sha="${1:?usage: run_in_container.sh <commit-sha> [source-branch]}"
branch_override="${2:-}"
if [[ -n "${branch_override}" ]]; then
  GITEE_BRANCH="${branch_override}"
fi
LOCAL_CI_CONTAINER="${LOCAL_CI_CONTAINER:-triton-anchor-dev}"
CONTAINER_CI_RUNNER_DIR="${CONTAINER_CI_RUNNER_DIR:-/tmp/triton-anchor-local-ci-runner-${sha:0:12}-$$}"

pass_env=(
  GITEE_REPO_URL GITEE_BRANCH GITEE_USERNAME GITEE_TOKEN ANCHOR_DIR WORKSPACE
  BACKEND_PROFILE EXPECTED_TRITON_BACKEND BACKEND_PATH BACKEND_ENVSETUP BACKEND_ENVSETUP_ARGS BACKEND_TEST_COMMAND
  RUN_FLAGGEMS_TESTS FLAGGEMS_CLONE_DIR FLAGGEMS_REF FLAGGEMS_PIP_PACKAGES FLAGGEMS_TEST_OP FLAGGEMS_TEST_COMMAND INSTALL_FLAGGEMS_PACKAGES
  LLVM_BUILD_DIR PPL_ROOT PACKAGE_TOOL PYTHON_BIN PYTHON_VENV_ACTIVATE SOURCE_ENVSETUP FRONTEND_BUILD_COMMAND LOCAL_CI_ARTIFACT_ROOT
)

docker_args=()
for name in "${pass_env[@]}"; do
  if [[ -n "${!name:-}" ]]; then
    docker_args+=(-e "${name}=${!name}")
  fi
done

docker exec "${LOCAL_CI_CONTAINER}" mkdir -p "${CONTAINER_CI_RUNNER_DIR}"
docker cp "${SCRIPT_DIR}/." "${LOCAL_CI_CONTAINER}:${CONTAINER_CI_RUNNER_DIR}/"

docker exec \
  "${docker_args[@]}" \
  -e LOCAL_CI_COMMIT="${sha}" \
  -e LOCAL_CI_RUNNER_DIR="${CONTAINER_CI_RUNNER_DIR}" \
  "${LOCAL_CI_CONTAINER}" \
  bash -lc 'bash "${LOCAL_CI_RUNNER_DIR}/run_delivery_local.sh" "${LOCAL_CI_COMMIT}"'
