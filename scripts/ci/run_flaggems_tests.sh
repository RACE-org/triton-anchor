#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PACKAGE_TOOL="${PACKAGE_TOOL:-auto}"
WORKSPACE="${WORKSPACE:-/workspace}"
DELIVERY_ARTIFACT_DIR="${DELIVERY_ARTIFACT_DIR:-delivery-artifacts}"
SOURCE_ENVSETUP="${SOURCE_ENVSETUP:-1}"
BACKEND_CLONE_DIR="${BACKEND_CLONE_DIR:-/tmp/triton-anchor-backend}"
BACKEND_ENVSETUP="${BACKEND_ENVSETUP:-}"
BACKEND_ENVSETUP_ARGS="${BACKEND_ENVSETUP_ARGS:-}"
FLAGGEMS_CLONE_DIR="${FLAGGEMS_CLONE_DIR:-${WORKSPACE}/FlagGems}"
FLAGGEMS_REPO_URL="${FLAGGEMS_REPO_URL:-}"
FLAGGEMS_REF="${FLAGGEMS_REF:-}"
FLAGGEMS_PIP_PACKAGES="${FLAGGEMS_PIP_PACKAGES:-}"
FLAGGEMS_TEST_COMMAND="${FLAGGEMS_TEST_COMMAND:-}"

cd "${ROOT_DIR}"
mkdir -p "${DELIVERY_ARTIFACT_DIR}"

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

resolve_backend_dir() {
  if [[ -n "${BACKEND_PATH:-}" ]]; then
    echo "${BACKEND_PATH}"
  elif [[ -n "${BACKEND_REPO_URL:-}" ]]; then
    echo "${BACKEND_CLONE_DIR}"
  else
    echo ""
  fi
}

normalize_flaggems_repo_config() {
  local clean_url="${FLAGGEMS_REPO_URL%%\?*}"
  clean_url="${clean_url%%#*}"

  if [[ "${clean_url}" =~ ^https://github\.com/([^/]+)/([^/]+)/tree/(.+)$ ]]; then
    FLAGGEMS_REPO_URL="https://github.com/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}.git"
    if [[ -z "${FLAGGEMS_REF}" ]]; then
      FLAGGEMS_REF="${BASH_REMATCH[3]}"
    fi
  elif [[ "${clean_url}" =~ ^https://github\.com/([^/]+)/([^/]+)$ && "${clean_url}" != *.git ]]; then
    FLAGGEMS_REPO_URL="${clean_url}.git"
  else
    FLAGGEMS_REPO_URL="${clean_url}"
  fi

  export FLAGGEMS_REPO_URL FLAGGEMS_REF
}

clone_flaggems_repo() {
  if [[ -z "${FLAGGEMS_REPO_URL}" ]]; then
    echo "FLAGGEMS_REPO_URL is required when FlagGems tests are enabled." >&2
    exit 1
  fi
  if [[ -z "${FLAGGEMS_CLONE_DIR}" || "${FLAGGEMS_CLONE_DIR}" == "/" ]]; then
    echo "Unsafe FLAGGEMS_CLONE_DIR: ${FLAGGEMS_CLONE_DIR}" >&2
    exit 1
  fi

  rm -rf "${FLAGGEMS_CLONE_DIR}"
  echo "Cloning FlagGems repo ${FLAGGEMS_REPO_URL}"
  if [[ -n "${PREBUILT_DOWNLOAD_TOKEN:-}" && "${FLAGGEMS_REPO_URL}" == https://github.com/* ]]; then
    auth_header="$(printf 'x-access-token:%s' "${PREBUILT_DOWNLOAD_TOKEN}" | base64 | tr -d '\n')"
    echo "Using PREBUILT_DOWNLOAD_TOKEN for authenticated GitHub FlagGems clone."
    git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
      clone "${FLAGGEMS_REPO_URL}" "${FLAGGEMS_CLONE_DIR}"
  else
    git clone "${FLAGGEMS_REPO_URL}" "${FLAGGEMS_CLONE_DIR}"
  fi

  if [[ -n "${FLAGGEMS_REF}" ]]; then
    git -C "${FLAGGEMS_CLONE_DIR}" checkout "${FLAGGEMS_REF}"
  else
    echo "::warning::FLAGGEMS_REF is not set; using the repository default branch."
  fi
}

source_backend_envsetup() {
  local backend_dir="$1"
  local setup_script="${BACKEND_ENVSETUP}"

  if [[ -z "${setup_script}" ]]; then
    return 0
  fi

  if [[ "${setup_script}" != /* ]]; then
    setup_script="${backend_dir}/${setup_script}"
  fi

  if [[ ! -f "${setup_script}" ]]; then
    echo "Backend envsetup script does not exist: ${setup_script}" >&2
    exit 1
  fi

  echo "Sourcing backend envsetup before FlagGems tests: ${setup_script} ${BACKEND_ENVSETUP_ARGS}"
  # shellcheck disable=SC1090,SC2086
  set +u
  source "${setup_script}" ${BACKEND_ENVSETUP_ARGS}
  set -u
}

collect_flaggems_artifacts() {
  local backend_dir="$1"
  local out_dir="${DELIVERY_ARTIFACT_DIR}/flaggems"
  mkdir -p "${out_dir}"

  cp -f "${DELIVERY_ARTIFACT_DIR}/flaggems-test.log" "${out_dir}/" 2>/dev/null || true
  cp -f "${backend_dir}"/testop/batch_test_results_flaggems_*.csv "${out_dir}/" 2>/dev/null || true
  if [[ -d "${backend_dir}/testop/logs" ]]; then
    cp -a "${backend_dir}/testop/logs" "${out_dir}/" 2>/dev/null || true
  fi
}

if [[ "${SOURCE_ENVSETUP}" == "1" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/envsetup.sh"
fi

backend_dir="$(resolve_backend_dir)"
if [[ -z "${backend_dir}" || ! -d "${backend_dir}" ]]; then
  echo "FlagGems tests require an installed backend directory. Got: ${backend_dir:-<none>}" >&2
  exit 1
fi
if [[ -z "${FLAGGEMS_TEST_COMMAND}" ]]; then
  echo "FLAGGEMS_TEST_COMMAND is required when FlagGems tests are enabled." >&2
  exit 1
fi

normalize_flaggems_repo_config
clone_flaggems_repo

if [[ -n "${FLAGGEMS_PIP_PACKAGES}" ]]; then
  echo "Installing FlagGems test Python packages: ${FLAGGEMS_PIP_PACKAGES}"
  if use_uv; then
    uv_pip install ${FLAGGEMS_PIP_PACKAGES}
  else
    "${PYTHON_BIN}" -m pip install ${FLAGGEMS_PIP_PACKAGES}
  fi
fi

export FLAGGEMS_ROOT="${FLAGGEMS_CLONE_DIR}"
source_backend_envsetup "${backend_dir}"

log_file="${DELIVERY_ARTIFACT_DIR}/flaggems-test.log"
echo "Running FlagGems test command in ${backend_dir}; log: ${log_file}"
set +e
(cd "${backend_dir}" && bash -lc "${FLAGGEMS_TEST_COMMAND}") 2>&1 | tee "${log_file}"
status=${PIPESTATUS[0]}
set -e

collect_flaggems_artifacts "${backend_dir}"

if [[ ${status} -ne 0 ]]; then
  echo "FlagGems tests failed with exit code ${status}" >&2
  exit "${status}"
fi

echo "FlagGems tests finished. Artifacts are in ${DELIVERY_ARTIFACT_DIR}/flaggems"