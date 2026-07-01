#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DELIVERY_ARTIFACT_DIR="${DELIVERY_ARTIFACT_DIR:-delivery-artifacts}"
SOURCE_ENVSETUP="${SOURCE_ENVSETUP:-1}"
EXPECTED_TRITON_BACKEND="${EXPECTED_TRITON_BACKEND:-}"

cd "${ROOT_DIR}"
mkdir -p "${DELIVERY_ARTIFACT_DIR}"

if [[ "${SOURCE_ENVSETUP}" == "1" ]]; then
  # Smoke validation imports libtriton and loads MLIR dialects, so it should
  # run under the same LLVM environment used for building.
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/envsetup.sh"
fi

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

echo "Delivery smoke test finished. Artifacts are in ${DELIVERY_ARTIFACT_DIR}"