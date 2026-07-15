#!/usr/bin/env bash
set -euo pipefail

if [[ "${LOCAL_CI_SCRIPT_STAGED:-0}" != "1" ]]; then
  source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  staged_dir="/tmp/triton-anchor-local-ci.$$"
  mkdir -p "${staged_dir}"
  cp -a "${source_dir}/." "${staged_dir}/"
  export LOCAL_CI_RUNNER_DIR="${staged_dir}"
  export LOCAL_CI_SCRIPT_STAGED="1"
  exec "${staged_dir}/run_delivery_local.sh" "$@"
fi

target_sha="${1:?usage: run_delivery_local.sh <commit-sha>}"

WORKSPACE="${WORKSPACE:-/workspace}"
ANCHOR_DIR="${ANCHOR_DIR:-${WORKSPACE}/triton-anchor}"
GITEE_REPO_URL="${GITEE_REPO_URL:-https://gitee.com/likehupochuan/triton-anchor.git}"
GITEE_BRANCH="${GITEE_BRANCH:-jiwang-delivery-ci}"
GITEE_USERNAME="${GITEE_USERNAME:-likehupochuan}"
GITEE_TOKEN="${GITEE_TOKEN:-}"
LOCAL_CI_GIT_ASKPASS=""
BACKEND_PROFILE="${BACKEND_PROFILE:-sophgo-cmodel}"
EXPECTED_TRITON_BACKEND="${EXPECTED_TRITON_BACKEND:-sophgo}"
BACKEND_PATH="${BACKEND_PATH:-${WORKSPACE}/triton-sophgo-backend}"
BACKEND_ENVSETUP="${BACKEND_ENVSETUP:-envsetup.sh}"
BACKEND_ENVSETUP_ARGS="${BACKEND_ENVSETUP_ARGS:-PIO_CMODEL}"
BACKEND_TEST_COMMAND="${BACKEND_TEST_COMMAND:-python3 tests/test_smoke.py && python3 tests/test_jit.py}"
RUN_FLAGGEMS_TESTS="${RUN_FLAGGEMS_TESTS:-false}"
FLAGGEMS_CLONE_DIR="${FLAGGEMS_CLONE_DIR:-${WORKSPACE}/FlagGems}"
FLAGGEMS_REF="${FLAGGEMS_REF:-}"
FLAGGEMS_PIP_PACKAGES="${FLAGGEMS_PIP_PACKAGES:-scipy pytest}"
FLAGGEMS_TEST_OP="${FLAGGEMS_TEST_OP:-abs}"
FLAGGEMS_TEST_COMMAND="${FLAGGEMS_TEST_COMMAND:-cd ${FLAGGEMS_CLONE_DIR} && python3 -m pytest -s tests/test_unary_pointwise_ops.py -m ${FLAGGEMS_TEST_OP} --record=log}"
INSTALL_FLAGGEMS_PACKAGES="${INSTALL_FLAGGEMS_PACKAGES:-1}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${WORKSPACE}/llvm-release}"
PPL_ROOT="${PPL_ROOT:-${WORKSPACE}/ppl-release}"
PACKAGE_TOOL="${PACKAGE_TOOL:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_VENV_ACTIVATE="${PYTHON_VENV_ACTIVATE:-/opt/venv/bin/activate}"
SOURCE_ENVSETUP="${SOURCE_ENVSETUP:-1}"
FRONTEND_BUILD_COMMAND="${FRONTEND_BUILD_COMMAND:-}"
LOCAL_CI_ARTIFACT_ROOT="${LOCAL_CI_ARTIFACT_ROOT:-${WORKSPACE}/local-ci-artifacts}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
DELIVERY_ARTIFACT_DIR="${DELIVERY_ARTIFACT_DIR:-${LOCAL_CI_ARTIFACT_ROOT}/${run_stamp}-${target_sha:0:12}}"

export WORKSPACE ANCHOR_DIR BACKEND_PROFILE EXPECTED_TRITON_BACKEND BACKEND_PATH
export BACKEND_ENVSETUP BACKEND_ENVSETUP_ARGS BACKEND_TEST_COMMAND
export RUN_FLAGGEMS_TESTS FLAGGEMS_CLONE_DIR FLAGGEMS_REF FLAGGEMS_PIP_PACKAGES FLAGGEMS_TEST_OP FLAGGEMS_TEST_COMMAND
export LLVM_BUILD_DIR PPL_ROOT PYTHON_BIN PYTHON_VENV_ACTIVATE GITHUB_SHA="${target_sha}" GITHUB_REF="refs/heads/${GITEE_BRANCH}"

mkdir -p "${DELIVERY_ARTIFACT_DIR}"

use_uv() {
  [[ "${PACKAGE_TOOL}" == "uv" ]] || { [[ "${PACKAGE_TOOL}" == "auto" ]] && command -v uv >/dev/null 2>&1; }
}

setup_gitee_git_auth() {
  if [[ -z "${GITEE_TOKEN}" ]]; then
    echo "GITEE_TOKEN is not set; git fetch will rely on existing credentials."
    export GIT_TERMINAL_PROMPT=0
    return 0
  fi

  local askpass
  askpass="$(mktemp /tmp/local-ci-gitee-askpass.XXXXXX)"
  cat > "${askpass}" <<'EOF'
#!/usr/bin/env sh
case "$1" in
  *Username*) printf '%s\n' "${GITEE_USERNAME:-likehupochuan}" ;;
  *) printf '%s\n' "${GITEE_TOKEN}" ;;
esac
EOF
  chmod 700 "${askpass}"
  export GITEE_USERNAME GITEE_TOKEN
  export GIT_ASKPASS="${askpass}"
  export GIT_TERMINAL_PROMPT=0
  LOCAL_CI_GIT_ASKPASS="${askpass}"
}

cleanup_gitee_git_auth() {
  if [[ -n "${LOCAL_CI_GIT_ASKPASS:-}" && -f "${LOCAL_CI_GIT_ASKPASS}" ]]; then
    rm -f "${LOCAL_CI_GIT_ASKPASS}"
  fi
}
run_logged() {
  local name="$1"
  shift
  local log_file="${DELIVERY_ARTIFACT_DIR}/${name}.log"
  echo "Running ${name}; log: ${log_file}"
  "$@" 2>&1 | tee "${log_file}"
}

rebuild_backend() {
  if [[ ! -d "${BACKEND_PATH}" ]]; then
    echo "Backend path does not exist: ${BACKEND_PATH}" >&2
    return 1
  fi

  local log_file="${DELIVERY_ARTIFACT_DIR}/backend-rebuild.log"
  echo "Running backend-rebuild; log: ${log_file}"
  set +e
  (
    set -euo pipefail
    cd "${BACKEND_PATH}"
    if use_uv; then
      uv pip install scikit-build-core pybind11
      uv pip uninstall triton-sophgo-backend triton_sophgo_backend || true
      rm -rf build dist *.egg-info
      uv build --wheel --no-build-isolation
      uv pip install --force-reinstall dist/triton_sophgo_backend-*.whl
    else
      "${PYTHON_BIN}" -m pip install scikit-build-core pybind11 build
      "${PYTHON_BIN}" -m pip uninstall -y triton-sophgo-backend triton_sophgo_backend || true
      rm -rf build dist *.egg-info
      "${PYTHON_BIN}" -m build --wheel --no-isolation
      "${PYTHON_BIN}" -m pip install --force-reinstall dist/triton_sophgo_backend-*.whl
    fi

    backend_wheel="$(find dist -maxdepth 1 -name 'triton_sophgo_backend-*.whl' -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
    if [[ -n "${backend_wheel}" ]]; then
      cp "${backend_wheel}" "${DELIVERY_ARTIFACT_DIR}/"
      ls -lh "${backend_wheel}" "${DELIVERY_ARTIFACT_DIR}/$(basename "${backend_wheel}")"
    fi
  ) 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

source_python_venv() {
  if [[ -z "${PYTHON_VENV_ACTIVATE}" ]]; then
    return 0
  fi
  if [[ ! -f "${PYTHON_VENV_ACTIVATE}" ]]; then
    echo "Python venv activate script does not exist: ${PYTHON_VENV_ACTIVATE}" >&2
    exit 1
  fi
  echo "Sourcing Python venv: ${PYTHON_VENV_ACTIVATE}"
  set +u
  # shellcheck disable=SC1090
  source "${PYTHON_VENV_ACTIVATE}"
  set -u
}

source_anchor_env() {
  if [[ "${SOURCE_ENVSETUP}" == "1" && -f "${ANCHOR_DIR}/envsetup.sh" ]]; then
    echo "Sourcing anchor envsetup.sh."
    set +u
    # shellcheck disable=SC1091
    source "${ANCHOR_DIR}/envsetup.sh"
    set -u
  fi
}

source_backend_env() {
  local setup_script="${BACKEND_ENVSETUP}"
  if [[ -z "${setup_script}" ]]; then
    return 0
  fi
  if [[ "${setup_script}" != /* ]]; then
    setup_script="${BACKEND_PATH}/${setup_script}"
  fi
  if [[ ! -f "${setup_script}" ]]; then
    echo "Backend envsetup script does not exist: ${setup_script}" >&2
    exit 1
  fi
  echo "Sourcing backend envsetup: ${setup_script} ${BACKEND_ENVSETUP_ARGS}"
  set +u
  # shellcheck disable=SC1090,SC2086
  source "${setup_script}" ${BACKEND_ENVSETUP_ARGS}
  set -u
}

git_commit() {
  local repo="$1"
  git -C "${repo}" rev-parse HEAD 2>/dev/null || true
}

write_summary() {
  local status="$1"
  set +e
  {
    echo "schema: triton-anchor-local-ci/v2"
    echo "status: ${status}"
    echo "target_sha: ${target_sha}"
    echo "branch: ${GITEE_BRANCH}"
    echo "anchor_dir: ${ANCHOR_DIR}"
    echo "anchor_commit: $(git_commit "${ANCHOR_DIR}")"
    echo "backend_profile: ${BACKEND_PROFILE}"
    echo "expected_backend: ${EXPECTED_TRITON_BACKEND}"
    echo "backend_path: ${BACKEND_PATH}"
    echo "backend_commit: $(git_commit "${BACKEND_PATH}")"
    echo "flaggems_enabled: ${RUN_FLAGGEMS_TESTS}"
    echo "flaggems_dir: ${FLAGGEMS_CLONE_DIR}"
    echo "flaggems_commit: $(git_commit "${FLAGGEMS_CLONE_DIR}")"
    echo "flaggems_test_op: ${FLAGGEMS_TEST_OP}"
    echo "flaggems_test_command: ${FLAGGEMS_TEST_COMMAND}"
    echo "llvm_build_dir: ${LLVM_BUILD_DIR}"
    echo "ppl_root: ${PPL_ROOT}"
    echo "artifact_dir: ${DELIVERY_ARTIFACT_DIR}"
  } > "${DELIVERY_ARTIFACT_DIR}/delivery-summary.txt"
  set -e
}

on_exit() {
  local status="$?"
  cleanup_gitee_git_auth
  write_summary "${status}"
  exit "${status}"
}
trap on_exit EXIT

cd "${ANCHOR_DIR}"
git config --global --add safe.directory "${ANCHOR_DIR}" || true
if git remote get-url gitee >/dev/null 2>&1; then
  git remote set-url gitee "${GITEE_REPO_URL}"
else
  git remote add gitee "${GITEE_REPO_URL}"
fi

setup_gitee_git_auth
git fetch --prune gitee "${GITEE_BRANCH}"
git checkout --detach "${target_sha}"
git reset --hard "${target_sha}"

cat <<EOF
Local CI commit: ${target_sha}
Anchor dir: ${ANCHOR_DIR}
Backend profile: ${BACKEND_PROFILE}
Backend path: ${BACKEND_PATH}
Run FlagGems: ${RUN_FLAGGEMS_TESTS}
Artifact dir: ${DELIVERY_ARTIFACT_DIR}
EOF

source_python_venv
source_anchor_env

if [[ -z "${FRONTEND_BUILD_COMMAND}" ]]; then
  if use_uv; then
    FRONTEND_BUILD_COMMAND="uv build --wheel --no-build-isolation"
  else
    FRONTEND_BUILD_COMMAND="${PYTHON_BIN} -m build --wheel --no-isolation"
  fi
fi
mkdir -p "${ANCHOR_DIR}/dist"
echo "Cleaning old frontend wheels under ${ANCHOR_DIR}/dist"
rm -f "${ANCHOR_DIR}"/dist/*.whl

run_logged frontend-build bash -lc "${FRONTEND_BUILD_COMMAND}"

wheel_path="$(find "${ANCHOR_DIR}/dist" -maxdepth 1 -name '*.whl' -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
if [[ -z "${wheel_path}" ]]; then
  echo "No built wheel found under ${ANCHOR_DIR}/dist" >&2
  exit 1
fi
{
  echo "Built frontend wheel: ${wheel_path}"
  ls -lh "${wheel_path}"
  sha256sum "${wheel_path}"
} | tee "${DELIVERY_ARTIFACT_DIR}/frontend-wheel-info.log"
if use_uv; then
  run_logged frontend-install uv pip install --force-reinstall --no-deps "${wheel_path}"
else
  run_logged frontend-install "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "${wheel_path}"
fi

source_python_venv
source_anchor_env

run_logged verify-triton-anchor-import "${PYTHON_BIN}" - <<'PY'
import triton_anchor
print("triton-anchor loaded", getattr(triton_anchor, "__version__", "unknown"))
PY

(cd "${ANCHOR_DIR}" && run_logged frontend-smoke "${PYTHON_BIN}" tests/test_smoke.py)

source_backend_env
rebuild_backend
source_python_venv
source_anchor_env
source_backend_env

run_logged verify-backend-discovery "${PYTHON_BIN}" - <<'PY'
from triton.backends import backends
print(backends)
PY

if [[ -n "${EXPECTED_TRITON_BACKEND}" ]]; then
  run_logged verify-expected-backend "${PYTHON_BIN}" - <<'PY'
import os
from triton.backends import backends
expected = os.environ["EXPECTED_TRITON_BACKEND"]
assert expected in backends, f"Expected backend {expected!r} was not discovered"
print(f"expected backend discovered: {expected}")
PY
fi

if [[ -n "${BACKEND_TEST_COMMAND}" ]]; then
  (cd "${BACKEND_PATH}" && run_logged backend-smoke-jit bash -lc "${BACKEND_TEST_COMMAND}")
fi

if [[ "${RUN_FLAGGEMS_TESTS}" == "true" ]]; then
  if [[ ! -d "${FLAGGEMS_CLONE_DIR}" ]]; then
    echo "FlagGems repo does not exist: ${FLAGGEMS_CLONE_DIR}" >&2
    exit 1
  fi
  if [[ -n "${FLAGGEMS_REF}" ]]; then
    git -C "${FLAGGEMS_CLONE_DIR}" checkout "${FLAGGEMS_REF}"
  fi
  if [[ "${INSTALL_FLAGGEMS_PACKAGES}" != "0" && -n "${FLAGGEMS_PIP_PACKAGES}" ]]; then
    if use_uv; then
      run_logged flaggems-deps uv pip install ${FLAGGEMS_PIP_PACKAGES}
    else
      run_logged flaggems-deps "${PYTHON_BIN}" -m pip install ${FLAGGEMS_PIP_PACKAGES}
    fi
  fi
  export FLAGGEMS_ROOT="${FLAGGEMS_CLONE_DIR}"
  source_backend_env
  (cd "${BACKEND_PATH}" && run_logged flaggems bash -lc "${FLAGGEMS_TEST_COMMAND}")

fi

echo "Local CI finished successfully. Artifacts are in ${DELIVERY_ARTIFACT_DIR}"
