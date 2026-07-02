#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DELIVERY_ARTIFACT_DIR="${DELIVERY_ARTIFACT_DIR:-delivery-artifacts}"
SOURCE_ENVSETUP="${SOURCE_ENVSETUP:-1}"
EXPECTED_TRITON_BACKEND="${EXPECTED_TRITON_BACKEND:-}"
BACKEND_CLONE_DIR="${BACKEND_CLONE_DIR:-/tmp/triton-anchor-backend}"
BACKEND_ENVSETUP="${BACKEND_ENVSETUP:-}"
BACKEND_ENVSETUP_ARGS="${BACKEND_ENVSETUP_ARGS:-}"
BACKEND_TEST_COMMAND="${BACKEND_TEST_COMMAND:-}"

cd "${ROOT_DIR}"
mkdir -p "${DELIVERY_ARTIFACT_DIR}"

if [[ "${SOURCE_ENVSETUP}" == "1" ]]; then
  # Smoke validation imports libtriton and loads MLIR dialects, so it should
  # run under the same LLVM environment used for building.
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/envsetup.sh"
fi

resolve_backend_dir() {
  if [[ -n "${BACKEND_PATH:-}" ]]; then
    echo "${BACKEND_PATH}"
  elif [[ -n "${BACKEND_REPO_URL:-}" ]]; then
    echo "${BACKEND_CLONE_DIR}"
  else
    echo ""
  fi
}

source_backend_envsetup() {
  local backend_dir
  local setup_script="${BACKEND_ENVSETUP}"

  if [[ -z "${setup_script}" ]]; then
    return 0
  fi

  backend_dir="$(resolve_backend_dir)"
  if [[ -z "${backend_dir}" ]]; then
    echo "BACKEND_ENVSETUP is set but no backend directory is configured." >&2
    exit 1
  fi

  if [[ "${setup_script}" != /* ]]; then
    setup_script="${backend_dir}/${setup_script}"
  fi

  if [[ ! -f "${setup_script}" ]]; then
    echo "Backend envsetup script does not exist: ${setup_script}" >&2
    exit 1
  fi

  echo "Sourcing backend envsetup before smoke: ${setup_script} ${BACKEND_ENVSETUP_ARGS}"
  # shellcheck disable=SC1090,SC2086
  source "${setup_script}" ${BACKEND_ENVSETUP_ARGS}
}

run_with_log() {
  local name="$1"
  shift
  local log_file="${DELIVERY_ARTIFACT_DIR}/${name}.log"

  echo "Running ${name}; log: ${log_file}"
  set +e
  "$@" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ ${status} -ne 0 ]]; then
    echo "${name} failed with exit code ${status}" >&2
    exit "${status}"
  fi
}

run_backend_test_with_log() {
  local backend_dir="$1"
  local log_file="${DELIVERY_ARTIFACT_DIR}/backend-test.log"

  if [[ ! -d "${backend_dir}" ]]; then
    echo "Backend test directory does not exist: ${backend_dir}" >&2
    exit 1
  fi

  echo "Running backend test command in ${backend_dir}; log: ${log_file}"
  set +e
  (cd "${backend_dir}" && bash -lc "${BACKEND_TEST_COMMAND}") 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ ${status} -ne 0 ]]; then
    echo "backend-test failed with exit code ${status}" >&2
    exit "${status}"
  fi
}

source_backend_envsetup

run_with_log "verify-triton-anchor-import" \
  "${PYTHON_BIN}" -c "import triton_anchor; print('triton-anchor loaded', triton_anchor.__version__)"

run_with_log "verify-backend-discovery" \
  "${PYTHON_BIN}" -c "from triton.backends import backends; print(backends)"

if [[ -n "${EXPECTED_TRITON_BACKEND}" ]]; then
  run_with_log "verify-expected-backend" \
    "${PYTHON_BIN}" -c "from triton.backends import backends; expected='${EXPECTED_TRITON_BACKEND}'; print(backends); assert expected in backends, f'Expected backend {expected!r} was not discovered'"
fi

run_with_log "smoke-test" \
  "${PYTHON_BIN}" tests/test_smoke.py

if [[ -n "${BACKEND_TEST_COMMAND}" ]]; then
  backend_dir="$(resolve_backend_dir)"
  run_backend_test_with_log "${backend_dir}"
fi

echo "Delivery smoke test finished. Artifacts are in ${DELIVERY_ARTIFACT_DIR}"
