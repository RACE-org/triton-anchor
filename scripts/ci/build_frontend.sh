#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_WHEEL="${INSTALL_WHEEL:-1}"
NO_BUILD_ISOLATION="${NO_BUILD_ISOLATION:-1}"
PACKAGE_TOOL="${PACKAGE_TOOL:-auto}"
SOURCE_ENVSETUP="${SOURCE_ENVSETUP:-1}"

cd "${ROOT_DIR}"

if [[ "${SOURCE_ENVSETUP}" == "1" ]]; then
  # envsetup.sh exports LLVM_SYSPATH, LLVM include/lib/bin paths, and updates PATH.
  # It must be sourced in the current shell before building triton-anchor.
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/envsetup.sh"
fi

use_uv() {
  [[ "${PACKAGE_TOOL}" == "uv" ]] || { [[ "${PACKAGE_TOOL}" == "auto" ]] && command -v uv >/dev/null 2>&1; }
}

uv_pip() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    uv pip "$@"
  else
    uv pip --system "$@"
  fi
}

echo "Building triton-anchor from source at ${ROOT_DIR}"
if use_uv; then
  uv_pip install --upgrade build pip pybind11 setuptools wheel
else
  "${PYTHON_BIN}" -m pip install --upgrade build pip pybind11 setuptools wheel
fi

python_build_args=(--wheel)
uv_build_args=(--wheel)
if [[ "${NO_BUILD_ISOLATION}" == "1" ]]; then
  python_build_args+=(--no-isolation)
  uv_build_args+=(--no-build-isolation)
fi

if use_uv; then
  uv build "${uv_build_args[@]}"
else
  "${PYTHON_BIN}" -m build "${python_build_args[@]}"
fi

wheel_path="$(ls -t dist/triton_anchor-*.whl | head -n 1)"
if [[ -z "${wheel_path}" ]]; then
  echo "No triton-anchor wheel found under dist/." >&2
  exit 1
fi

echo "Built wheel: ${wheel_path}"

if [[ "${INSTALL_WHEEL}" == "1" ]]; then
  if use_uv; then
    uv_pip install --force-reinstall "${wheel_path}"
  else
    "${PYTHON_BIN}" -m pip install --force-reinstall "${wheel_path}"
  fi
  echo "Installed wheel: ${wheel_path}"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "wheel_path=${wheel_path}" >> "${GITHUB_OUTPUT}"
fi