#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND_INSTALL_ARGS="${BACKEND_INSTALL_ARGS:-}"
BACKEND_INSTALL_MODE="${BACKEND_INSTALL_MODE:-standard}"
BACKEND_CLONE_DIR="${BACKEND_CLONE_DIR:-/tmp/triton-anchor-backend}"
BACKEND_CLONE_SUBMODULES="${BACKEND_CLONE_SUBMODULES:-0}"
PACKAGE_TOOL="${PACKAGE_TOOL:-auto}"
SOURCE_ENVSETUP="${SOURCE_ENVSETUP:-1}"

if [[ "${SOURCE_ENVSETUP}" == "1" ]]; then
  # Backends often compile or import against the same LLVM/Triton environment.
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

if [[ -n "${BACKEND_PATH:-}" ]]; then
  backend_dir="${BACKEND_PATH}"
  echo "Using backend from BACKEND_PATH=${backend_dir}"
elif [[ -n "${BACKEND_REPO_URL:-}" ]]; then
  backend_dir="${BACKEND_CLONE_DIR}"
  rm -rf "${backend_dir}"

  clone_args=()
  if [[ "${BACKEND_CLONE_SUBMODULES}" == "1" ]]; then
    clone_args+=(--recursive)
  fi

  echo "Cloning backend repo ${BACKEND_REPO_URL}"
  git clone "${clone_args[@]}" "${BACKEND_REPO_URL}" "${backend_dir}"

  if [[ -n "${BACKEND_REF:-}" ]]; then
    git -C "${backend_dir}" checkout "${BACKEND_REF}"
  else
    echo "::warning::BACKEND_REF is not set; using the repository default branch."
  fi
else
  echo "No backend configured; skipping backend installation."
  exit 0
fi

if [[ ! -d "${backend_dir}" ]]; then
  echo "Backend directory does not exist: ${backend_dir}" >&2
  exit 1
fi

echo "Installing backend from ${backend_dir}"
install_target=("${backend_dir}")
if [[ "${BACKEND_INSTALL_MODE}" == "editable" ]]; then
  install_target=(-e "${backend_dir}")
elif [[ "${BACKEND_INSTALL_MODE}" != "standard" ]]; then
  echo "Unsupported BACKEND_INSTALL_MODE='${BACKEND_INSTALL_MODE}'. Use 'standard' or 'editable'." >&2
  exit 1
fi

if use_uv; then
  uv_pip install "${install_target[@]}" ${BACKEND_INSTALL_ARGS}
else
  "${PYTHON_BIN}" -m pip install "${install_target[@]}" ${BACKEND_INSTALL_ARGS}
fi