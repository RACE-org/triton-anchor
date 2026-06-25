#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
#LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/home/cjl/0602/spine-triton/llvm-project/build}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/home/race_work/cjl/0606/llvm-project/build/installed}"
DIST_DIR="${DIST_DIR:-${ROOT_DIR}/dist}"
WHEEL_GLOB="${WHEEL_GLOB:-triton_anchor-*.whl}"
INSTALL_MODE="${INSTALL_MODE:-break-system-packages}"

echo "[1/4] Environment"
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "LLVM_BUILD_DIR=${LLVM_BUILD_DIR}"
echo "DIST_DIR=${DIST_DIR}"
echo "INSTALL_MODE=${INSTALL_MODE}"

if [[ ! -d "${LLVM_BUILD_DIR}" ]]; then
  echo "error: LLVM_BUILD_DIR does not exist: ${LLVM_BUILD_DIR}" >&2
  exit 1
fi

mkdir -p "${DIST_DIR}"

echo "[2/4] Build wheel"
(
  cd "${ROOT_DIR}"
  TRITON_ANCHOR_LLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
    "${PYTHON_BIN}" -m pip wheel . --no-build-isolation -w "${DIST_DIR}"
)

LATEST_WHEEL="$(find "${DIST_DIR}" -maxdepth 1 -name "${WHEEL_GLOB}" | sort | tail -n 1)"
if [[ -z "${LATEST_WHEEL}" ]]; then
  echo "error: no wheel matching ${WHEEL_GLOB} found in ${DIST_DIR}" >&2
  exit 1
fi

echo "[3/4] Install wheel"
case "${INSTALL_MODE}" in
  break-system-packages)
    "${PYTHON_BIN}" -m pip install --break-system-packages --force-reinstall "${LATEST_WHEEL}"
    ;;
  user)
    "${PYTHON_BIN}" -m pip install --user --force-reinstall "${LATEST_WHEEL}"
    ;;
  *)
    echo "error: unsupported INSTALL_MODE=${INSTALL_MODE}" >&2
    echo "supported values: break-system-packages, user" >&2
    exit 1
    ;;
esac

echo "[4/4] Run smoke test"
"${PYTHON_BIN}" "${ROOT_DIR}/tests/test_smoke.py"

echo "done: ${LATEST_WHEEL}"
