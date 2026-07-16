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
GITEE_RESULTS_REPO_URL="${GITEE_RESULTS_REPO_URL:-${GITEE_REPO_URL}}"
GITEE_RESULTS_BRANCH="${GITEE_RESULTS_BRANCH:-local-ci-results}"
LOCAL_CI_BASE_SHA="${LOCAL_CI_BASE_SHA:-}"
LOCAL_CI_BASE_REF="${LOCAL_CI_BASE_REF:-}"
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
RUN_COMPILE_BENCHMARK="${RUN_COMPILE_BENCHMARK:-true}"
COMPILE_BENCHMARK_KERNELS="${COMPILE_BENCHMARK_KERNELS:-add,mm,softmax,layernorm}"
COMPILE_BENCHMARK_REPEAT="${COMPILE_BENCHMARK_REPEAT:-5}"
COMPILE_BENCHMARK_WARMUP="${COMPILE_BENCHMARK_WARMUP:-1}"
COMPILE_BENCHMARK_THRESHOLD="${COMPILE_BENCHMARK_THRESHOLD:-0.20}"
COMPILE_BENCHMARK_TIMEOUT="${COMPILE_BENCHMARK_TIMEOUT:-30m}"
COMPILE_TIME_STATUS="disabled"
MAX_JOBS="${MAX_JOBS:-1}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
NINJAFLAGS="${NINJAFLAGS:--j1}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
DELIVERY_ARTIFACT_DIR="${DELIVERY_ARTIFACT_DIR:-${LOCAL_CI_ARTIFACT_ROOT}/${run_stamp}-${target_sha:0:12}}"

export WORKSPACE ANCHOR_DIR BACKEND_PROFILE EXPECTED_TRITON_BACKEND BACKEND_PATH
export BACKEND_ENVSETUP BACKEND_ENVSETUP_ARGS BACKEND_TEST_COMMAND
export RUN_FLAGGEMS_TESTS FLAGGEMS_CLONE_DIR FLAGGEMS_REF FLAGGEMS_PIP_PACKAGES FLAGGEMS_TEST_OP FLAGGEMS_TEST_COMMAND
export LLVM_BUILD_DIR PPL_ROOT PYTHON_BIN PYTHON_VENV_ACTIVATE GITHUB_SHA="${target_sha}" GITHUB_REF="refs/heads/${GITEE_BRANCH}"
export BACKEND_PROFILE MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL NINJAFLAGS UV_LINK_MODE

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

safe_path_part() {
  local value="$1"
  value="${value//\//_}"
  value="$(printf '%s' "${value}" | tr -c 'A-Za-z0-9._-' '_')"
  value="${value##_}"
  value="${value%%_}"
  printf '%s' "${value:-default}"
}

fetch_compile_baseline() {
  local sha="$1"
  local output="$2"
  local safe_profile
  safe_profile="$(safe_path_part "${BACKEND_PROFILE}")"
  local rel_path="compile-time/by-sha/${sha}/${safe_profile}/latest.json"

  if git remote get-url gitee-results >/dev/null 2>&1; then
    git remote set-url gitee-results "${GITEE_RESULTS_REPO_URL}"
  else
    git remote add gitee-results "${GITEE_RESULTS_REPO_URL}"
  fi
  if ! git fetch -q --depth=1 gitee-results \
    "refs/heads/${GITEE_RESULTS_BRANCH}:refs/remotes/gitee-results/${GITEE_RESULTS_BRANCH}"; then
    echo "Compile-time results branch is not available: ${GITEE_RESULTS_BRANCH}" >&2
    return 1
  fi
  if ! git show "gitee-results/${GITEE_RESULTS_BRANCH}:${rel_path}" > "${output}"; then
    rm -f "${output}"
    echo "No cached compile-time baseline at ${rel_path}" >&2
    return 1
  fi
  echo "Loaded compile-time baseline for ${sha}: ${rel_path}"
}

run_compile_benchmark() {
  if [[ "${RUN_COMPILE_BENCHMARK}" != "true" ]]; then
    COMPILE_TIME_STATUS="disabled"
    return 0
  fi
  if [[ ! -f "${LOCAL_CI_RUNNER_DIR}/compile_benchmark.py" ]]; then
    echo "Compile benchmark script is missing from the trusted runner snapshot." >&2
    return 1
  fi
  if [[ ! -f "${LOCAL_CI_RUNNER_DIR}/compare_compile_time.py" ]]; then
    echo "Compile comparison script is missing from the trusted runner snapshot." >&2
    return 1
  fi

  local candidate_json="${DELIVERY_ARTIFACT_DIR}/compile-benchmark.json"
  local candidate_csv="${DELIVERY_ARTIFACT_DIR}/compile-benchmark.csv"
  export FLAGGEMS_ROOT="${FLAGGEMS_CLONE_DIR}"
  source_backend_env
  run_logged compile-benchmark timeout "${COMPILE_BENCHMARK_TIMEOUT}" \
    "${PYTHON_BIN}" "${LOCAL_CI_RUNNER_DIR}/compile_benchmark.py" \
      --backend "${EXPECTED_TRITON_BACKEND:-sophgo}" \
      --vendor "${EXPECTED_TRITON_BACKEND:-sophgo}" \
      --flaggems-root "${FLAGGEMS_CLONE_DIR}" \
      --kernels "${COMPILE_BENCHMARK_KERNELS}" \
      --repeat "${COMPILE_BENCHMARK_REPEAT}" \
      --warmup "${COMPILE_BENCHMARK_WARMUP}" \
      --output-json "${candidate_json}" \
      --output-csv "${candidate_csv}"

  COMPILE_TIME_STATUS="pass"
  if [[ -n "${LOCAL_CI_BASE_SHA}" ]]; then
    local baseline_json="${DELIVERY_ARTIFACT_DIR}/compile-benchmark-base.json"
    fetch_compile_baseline "${LOCAL_CI_BASE_SHA}" "${baseline_json}" || true
    "${PYTHON_BIN}" "${LOCAL_CI_RUNNER_DIR}/compare_compile_time.py" \
      --baseline-json "${baseline_json}" \
      --candidate-json "${candidate_json}" \
      --base-sha "${LOCAL_CI_BASE_SHA}" \
      --candidate-sha "${target_sha}" \
      --kernels "${COMPILE_BENCHMARK_KERNELS}" \
      --threshold "${COMPILE_BENCHMARK_THRESHOLD}" \
      --output-json "${DELIVERY_ARTIFACT_DIR}/compile-time-comparison.json" \
      --output-markdown "${DELIVERY_ARTIFACT_DIR}/compile-time-comparison.md" \
      2>&1 | tee "${DELIVERY_ARTIFACT_DIR}/compile-time-comparison.log"
    COMPILE_TIME_STATUS="$("${PYTHON_BIN}" -c \
      'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
      "${DELIVERY_ARTIFACT_DIR}/compile-time-comparison.json")"
  fi
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
    echo "base_sha: ${LOCAL_CI_BASE_SHA}"
    echo "base_ref: ${LOCAL_CI_BASE_REF}"
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
    echo "compile_time_status: ${COMPILE_TIME_STATUS}"
    echo "compile_time_threshold: ${COMPILE_BENCHMARK_THRESHOLD}"
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

if [[ ("${RUN_FLAGGEMS_TESTS}" == "true" || "${RUN_COMPILE_BENCHMARK}" == "true") \
  && "${INSTALL_FLAGGEMS_PACKAGES}" != "0" && -n "${FLAGGEMS_PIP_PACKAGES}" ]]; then
  if use_uv; then
    run_logged flaggems-deps uv pip install ${FLAGGEMS_PIP_PACKAGES}
  else
    run_logged flaggems-deps "${PYTHON_BIN}" -m pip install ${FLAGGEMS_PIP_PACKAGES}
  fi
fi

if [[ "${RUN_FLAGGEMS_TESTS}" == "true" ]]; then
  if [[ ! -d "${FLAGGEMS_CLONE_DIR}" ]]; then
    echo "FlagGems repo does not exist: ${FLAGGEMS_CLONE_DIR}" >&2
    exit 1
  fi
  if [[ -n "${FLAGGEMS_REF}" ]]; then
    git -C "${FLAGGEMS_CLONE_DIR}" checkout "${FLAGGEMS_REF}"
  fi
  export FLAGGEMS_ROOT="${FLAGGEMS_CLONE_DIR}"
  source_backend_env
  (cd "${BACKEND_PATH}" && run_logged flaggems bash -lc "${FLAGGEMS_TEST_COMMAND}")

fi

run_compile_benchmark

echo "Local CI finished successfully. Artifacts are in ${DELIVERY_ARTIFACT_DIR}"
