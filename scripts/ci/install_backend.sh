#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND_INSTALL_ARGS="${BACKEND_INSTALL_ARGS:-}"
BACKEND_CLONE_DIR="${BACKEND_CLONE_DIR:-/tmp/triton-anchor-backend}"
BACKEND_CLONE_SUBMODULES="${BACKEND_CLONE_SUBMODULES:-0}"
PACKAGE_TOOL="${PACKAGE_TOOL:-auto}"
SOURCE_ENVSETUP="${SOURCE_ENVSETUP:-1}"
BACKEND_ENVSETUP="${BACKEND_ENVSETUP:-}"
BACKEND_ENVSETUP_ARGS="${BACKEND_ENVSETUP_ARGS:-}"
BACKEND_SOURCE_ENVSETUP_BEFORE_INSTALL="${BACKEND_SOURCE_ENVSETUP_BEFORE_INSTALL:-1}"

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

source_backend_envsetup() {
  local backend_dir="$1"
  local setup_script="${BACKEND_ENVSETUP}"

  if [[ "${BACKEND_SOURCE_ENVSETUP_BEFORE_INSTALL}" != "1" || -z "${setup_script}" ]]; then
    return 0
  fi

  if [[ "${setup_script}" != /* ]]; then
    setup_script="${backend_dir}/${setup_script}"
  fi

  if [[ ! -f "${setup_script}" ]]; then
    echo "Backend envsetup script does not exist: ${setup_script}" >&2
    exit 1
  fi

  echo "Sourcing backend envsetup before install: ${setup_script} ${BACKEND_ENVSETUP_ARGS}"
  # shellcheck disable=SC1090,SC2086
  set +u
  source "${setup_script}" ${BACKEND_ENVSETUP_ARGS}
  set -u
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
  if [[ -n "${PREBUILT_DOWNLOAD_TOKEN:-}" && "${BACKEND_REPO_URL}" == https://github.com/* ]]; then
    auth_header="$(printf 'x-access-token:%s' "${PREBUILT_DOWNLOAD_TOKEN}" | base64 | tr -d '\n')"
    echo "Using PREBUILT_DOWNLOAD_TOKEN for authenticated GitHub backend clone."
    git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
      clone "${clone_args[@]}" "${BACKEND_REPO_URL}" "${backend_dir}"
  else
    git clone "${clone_args[@]}" "${BACKEND_REPO_URL}" "${backend_dir}"
  fi

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

source_backend_envsetup "${backend_dir}"

echo "Installing backend from ${backend_dir}"
if use_uv; then
  uv_pip install "${backend_dir}" ${BACKEND_INSTALL_ARGS}
else
  "${PYTHON_BIN}" -m pip install "${backend_dir}" ${BACKEND_INSTALL_ARGS}
fi