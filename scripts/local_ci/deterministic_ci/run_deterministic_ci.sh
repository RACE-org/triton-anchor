#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK_TMP_TOOL="${LOCAL_CI_ROOT}/shared/task_tmp.py"
target_sha="${1:?usage: run_deterministic_ci.sh <commit-sha>}"
if [[ ! "${target_sha}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Local CI target SHA must be a lowercase 40-character hexadecimal commit." >&2
  exit 2
fi


if [[ "${LOCAL_CI_SCRIPT_STAGED:-0}" != "1" ]]; then
  task_tmp_root="$(mktemp -d "/tmp/triton-anchor-local-ci-task.${target_sha:0:12}.XXXXXX")"
  "${PYTHON_BIN:-python3}" "${TASK_TMP_TOOL}" prepare \
    --path "${task_tmp_root}" \
    --target-sha "${target_sha}"
  staged_dir="${task_tmp_root}/runner"
  cp -a "${LOCAL_CI_ROOT}/." "${staged_dir}/"
  export LOCAL_CI_RUNNER_DIR="${staged_dir}"
  export LOCAL_CI_SCRIPT_STAGED="1"
  export LOCAL_CI_TASK_TMP_ROOT="${task_tmp_root}"
  export TMPDIR="${task_tmp_root}/tmp"

  cleanup_staged_task_tmp() {
    local status="$?"
    local cleanup_status=0
    trap - EXIT INT TERM
    set +e
    "${PYTHON_BIN:-python3}" "${TASK_TMP_TOOL}" cleanup \
      --path "${task_tmp_root}" \
      --target-sha "${target_sha}" || cleanup_status=$?
    if [[ ${cleanup_status} -ne 0 ]]; then
      echo "Failed to clean Local CI task temporary root: ${task_tmp_root}" >&2
      if [[ ${status} -eq 0 ]]; then
        status="${cleanup_status}"
      fi
    fi
    exit "${status}"
  }
  trap cleanup_staged_task_tmp EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  bash "${staged_dir}/deterministic_ci/run_deterministic_ci.sh" "$@"
  exit 0
fi

RUNNER_ROOT="${LOCAL_CI_RUNNER_DIR:-${LOCAL_CI_ROOT}}"
PERFORMANCE_SCRIPT_DIR="${RUNNER_ROOT}/deterministic_ci/performance"
FLAGGEMS_SCRIPT_DIR="${RUNNER_ROOT}/deterministic_ci/flaggems"
DUMP_ARTIFACT_TOOL="${RUNNER_ROOT}/shared/dump_artifacts.py"
TASK_TMP_TOOL="${RUNNER_ROOT}/shared/task_tmp.py"
# shellcheck disable=SC1091
source "${RUNNER_ROOT}/shared/path_utils.sh"

WORKSPACE="${WORKSPACE:-/workspace}"
ANCHOR_DIR="${ANCHOR_DIR:-${WORKSPACE}/triton-anchor}"
GITEE_REPO_URL="${GITEE_REPO_URL:-https://gitee.com/race-org/triton-anchor-local-ci-results.git}"
GITEE_BRANCH="${GITEE_BRANCH:?GITEE_BRANCH must be provided by the Local CI poller}"
GITEE_USERNAME="${GITEE_USERNAME:-likehupochuan}"
GITEE_TOKEN="${GITEE_TOKEN:-}"
GITEE_RESULTS_REPO_URL="${GITEE_RESULTS_REPO_URL:-${GITEE_REPO_URL}}"
GITEE_RESULTS_BRANCH="${GITEE_RESULTS_BRANCH:-local-ci-results}"
LOCAL_CI_BASE_SHA="${LOCAL_CI_BASE_SHA:-}"
LOCAL_CI_BASE_REF="${LOCAL_CI_BASE_REF:-}"
TRUSTED_ANCHOR_ENVSETUP="${TRUSTED_ANCHOR_ENVSETUP:-${RUNNER_ROOT}/trusted/envsetup.sh}"
LOCAL_CI_GIT_ASKPASS=""
LOCAL_CI_PROFILE_NAME="${LOCAL_CI_PROFILE_NAME:-${BACKEND_PROFILE:-sophgo-cmodel}}"
LOCAL_CI_LLVM_HASH="${LOCAL_CI_LLVM_HASH:-unknown}"
RUN_BACKEND_STAGES="${RUN_BACKEND_STAGES:-true}"
BACKEND_SKIP_REASON="${BACKEND_SKIP_REASON:-}"
FRONTEND_ONLY_BACKEND_SKIP_REASON="当前没有部署可供测试的厂商后端，未执行后端构建、JIT、FlagGems 和性能验证。"
if [[ ! "${LOCAL_CI_LLVM_HASH}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LOCAL_CI_LLVM_HASH must be selected by the Local CI poller." >&2
  exit 2
fi
case "${RUN_BACKEND_STAGES}" in
  true)
    BACKEND_PROFILE="${BACKEND_PROFILE:-}"
    EXPECTED_TRITON_BACKEND="${EXPECTED_TRITON_BACKEND:-}"
    BACKEND_PATH="${BACKEND_PATH:-}"
    BACKEND_ENVSETUP="${BACKEND_ENVSETUP:-}"
    BACKEND_ENVSETUP_ARGS="${BACKEND_ENVSETUP_ARGS:-}"
    BACKEND_TEST_COMMAND="${BACKEND_TEST_COMMAND:-}"
    BACKEND_UNINSTALL_PACKAGES="${BACKEND_UNINSTALL_PACKAGES:-}"
    BACKEND_WHEEL_PATTERN="${BACKEND_WHEEL_PATTERN:-}"
    for required_backend_value in \
      BACKEND_PROFILE EXPECTED_TRITON_BACKEND BACKEND_PATH \
      BACKEND_TEST_COMMAND BACKEND_WHEEL_PATTERN; do
      if [[ -z "${!required_backend_value}" ]]; then
        echo "RUN_BACKEND_STAGES=true requires ${required_backend_value} from the selected profile." >&2
        exit 2
      fi
    done
    ;;
  false)
    BACKEND_SKIP_REASON="${FRONTEND_ONLY_BACKEND_SKIP_REASON}"
    BACKEND_PROFILE="${BACKEND_PROFILE:-${LOCAL_CI_PROFILE_NAME}}"
    EXPECTED_TRITON_BACKEND="${EXPECTED_TRITON_BACKEND-}"
    BACKEND_PATH="${BACKEND_PATH-}"
    BACKEND_ENVSETUP="${BACKEND_ENVSETUP-}"
    BACKEND_ENVSETUP_ARGS="${BACKEND_ENVSETUP_ARGS-}"
    BACKEND_TEST_COMMAND="${BACKEND_TEST_COMMAND-}"
    BACKEND_UNINSTALL_PACKAGES="${BACKEND_UNINSTALL_PACKAGES-}"
    BACKEND_WHEEL_PATTERN="${BACKEND_WHEEL_PATTERN-}"
    ;;
  *)
    echo "RUN_BACKEND_STAGES must be true or false." >&2
    exit 2
    ;;
esac
RUN_FLAGGEMS_TESTS="${RUN_FLAGGEMS_TESTS:-false}"
FLAGGEMS_CLONE_DIR="${FLAGGEMS_CLONE_DIR:-${WORKSPACE}/FlagGems}"
FLAGGEMS_REF="${FLAGGEMS_REF:-}"
FLAGGEMS_PIP_PACKAGES="${FLAGGEMS_PIP_PACKAGES:-scipy pytest}"
FLAGGEMS_TEST_MODE="${FLAGGEMS_TEST_MODE:-sample}"
FLAGGEMS_SAMPLE_SIZE="${FLAGGEMS_SAMPLE_SIZE:-6}"
FLAGGEMS_RANDOM_SEED="${FLAGGEMS_RANDOM_SEED:-}"
FLAGGEMS_TEST_OP="${FLAGGEMS_TEST_OP:-abs}"
FLAGGEMS_TEST_COMMAND="${FLAGGEMS_TEST_COMMAND:-}"
FLAGGEMS_PYTEST_ARGS="${FLAGGEMS_PYTEST_ARGS:---ref cpu -vs}"
FLAGGEMS_IDLE_TIMEOUT_SECONDS="${FLAGGEMS_IDLE_TIMEOUT_SECONDS:-300}"
FLAGGEMS_TOTAL_TIMEOUT_SECONDS="${FLAGGEMS_TOTAL_TIMEOUT_SECONDS:-6000}"
FLAGGEMS_FULL_TIMEOUT_EXTENSION_SECONDS="${FLAGGEMS_FULL_TIMEOUT_EXTENSION_SECONDS:-1800}"
FLAGGEMS_FULL_HARD_TIMEOUT_SECONDS="${FLAGGEMS_FULL_HARD_TIMEOUT_SECONDS:-14400}"
FLAGGEMS_CLEAR_CACHE="${FLAGGEMS_CLEAR_CACHE:-1}"
FLAGGEMS_WHITELIST="${FLAGGEMS_WHITELIST:-${FLAGGEMS_SCRIPT_DIR}/flaggems_pass_whitelist.tsv}"
FLAGGEMS_FULL_LIST="${FLAGGEMS_FULL_LIST:-${FLAGGEMS_SCRIPT_DIR}/flaggems_all_ops.tsv}"
INSTALL_FLAGGEMS_PACKAGES="${INSTALL_FLAGGEMS_PACKAGES:-1}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${WORKSPACE}/llvm-release}"
PPL_ROOT="${PPL_ROOT:-${WORKSPACE}/ppl-release}"
PACKAGE_TOOL="${PACKAGE_TOOL:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_VENV_ACTIVATE="${PYTHON_VENV_ACTIVATE:-/opt/venv/bin/activate}"
SOURCE_ENVSETUP="${SOURCE_ENVSETUP:-1}"
FRONTEND_BUILD_MODE="${FRONTEND_BUILD_MODE:-fresh}"
FRONTEND_BUILD_COMMAND="${FRONTEND_BUILD_COMMAND:-}"
LOCAL_CI_ARTIFACT_ROOT="${LOCAL_CI_ARTIFACT_ROOT:-${WORKSPACE}/local-ci-artifacts}"
RUN_COMPILE_BENCHMARK="${RUN_COMPILE_BENCHMARK:-true}"
COMPILE_BENCHMARK_KERNELS="${COMPILE_BENCHMARK_KERNELS:-add,mm,softmax,layernorm}"
COMPILE_BENCHMARK_REPEAT="${COMPILE_BENCHMARK_REPEAT:-5}"
COMPILE_BENCHMARK_WARMUP="${COMPILE_BENCHMARK_WARMUP:-1}"
COMPILE_BENCHMARK_THRESHOLD="${COMPILE_BENCHMARK_THRESHOLD:-0.20}"
COMPILE_BENCHMARK_TIMEOUT="${COMPILE_BENCHMARK_TIMEOUT:-30m}"
COMPILE_TIME_STATUS="not_run"
RUN_PASS_PROFILE="${RUN_PASS_PROFILE:-true}"
PASS_PROFILE_KERNELS="${PASS_PROFILE_KERNELS:-${COMPILE_BENCHMARK_KERNELS}}"
PASS_PROFILE_REPEAT="${PASS_PROFILE_REPEAT:-3}"
PASS_PROFILE_WARMUP="${PASS_PROFILE_WARMUP:-1}"
PASS_PROFILE_THRESHOLD="${PASS_PROFILE_THRESHOLD:-0.20}"
PASS_PROFILE_MIN_BASE_MS="${PASS_PROFILE_MIN_BASE_MS:-1.0}"
PASS_PROFILE_MIN_DELTA_MS="${PASS_PROFILE_MIN_DELTA_MS:-1.0}"
PASS_PROFILE_TIMEOUT="${PASS_PROFILE_TIMEOUT:-30m}"
PASS_PROFILE_STATUS="not_run"
RUN_IR_SERIALIZATION_BENCHMARK="${RUN_IR_SERIALIZATION_BENCHMARK:-true}"
IR_SERIALIZATION_KERNELS="${IR_SERIALIZATION_KERNELS:-${COMPILE_BENCHMARK_KERNELS}}"
IR_SERIALIZATION_REPEAT="${IR_SERIALIZATION_REPEAT:-20}"
IR_SERIALIZATION_WARMUP="${IR_SERIALIZATION_WARMUP:-3}"
IR_SERIALIZATION_METRICS="${IR_SERIALIZATION_METRICS:-serialize,deserialize}"
IR_SERIALIZATION_THRESHOLD="${IR_SERIALIZATION_THRESHOLD:-0.20}"
IR_SERIALIZATION_MIN_BASE_MS="${IR_SERIALIZATION_MIN_BASE_MS:-0.05}"
IR_SERIALIZATION_MIN_DELTA_MS="${IR_SERIALIZATION_MIN_DELTA_MS:-0.05}"
IR_SERIALIZATION_TIMEOUT="${IR_SERIALIZATION_TIMEOUT:-30m}"
IR_SERIALIZATION_STATUS="not_run"
FRONTEND_BUILD_STATUS="not_run"
FRONTEND_CHECKOUT_MODE="unknown"
FRONTEND_BUILD_CACHE_PRESERVED="false"
FRONTEND_SMOKE_STATUS="not_run"
BACKEND_REBUILD_STATUS="not_run"
BACKEND_SMOKE_JIT_STATUS="not_run"
FLAGGEMS_STATUS="disabled"
if [[ "${RUN_FLAGGEMS_TESTS}" == "true" ]]; then
  FLAGGEMS_STATUS="not_run"
fi
if [[ "${RUN_BACKEND_STAGES}" == "false" ]]; then
  RUN_FLAGGEMS_TESTS="false"
  RUN_COMPILE_BENCHMARK="false"
  RUN_PASS_PROFILE="false"
  RUN_IR_SERIALIZATION_BENCHMARK="false"
  INSTALL_FLAGGEMS_PACKAGES="0"
  BACKEND_REBUILD_STATUS="skipped"
  BACKEND_SMOKE_JIT_STATUS="skipped"
  FLAGGEMS_STATUS="skipped"
  COMPILE_TIME_STATUS="skipped"
  PASS_PROFILE_STATUS="skipped"
  IR_SERIALIZATION_STATUS="skipped"
fi
LOCAL_CI_RESULT_STATUS=0
TRITON_DUMP_CLEANUP_STATUS="not_run"
MAX_JOBS="${MAX_JOBS:-1}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
NINJAFLAGS="${NINJAFLAGS:--j1}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
DELIVERY_ARTIFACT_DIR="${DELIVERY_ARTIFACT_DIR:-${LOCAL_CI_ARTIFACT_ROOT}/${run_stamp}-${target_sha:0:12}}"
FLAGGEMS_SELECTED_FILE="${FLAGGEMS_SELECTED_FILE:-${DELIVERY_ARTIFACT_DIR}/flaggems-selected.txt}"
SOPHGO_TRITON_DUMP_DIR="/workspace/triton-dump-dir"
ROOT_TRITON_DUMP_DIR="/root/.triton/dump"
LOCAL_CI_TASK_TMP_ROOT="${LOCAL_CI_TASK_TMP_ROOT:?LOCAL_CI_TASK_TMP_ROOT must be provided by the Local CI task owner}"
LOCAL_CI_TASK_TMP_DIR="${LOCAL_CI_TASK_TMP_ROOT}/tmp"
LOCAL_CI_TASK_DUMP_ROOT="${LOCAL_CI_TASK_TMP_ROOT}/dump"
LOCAL_CI_TASK_CREDENTIAL_DIR="${LOCAL_CI_TASK_TMP_ROOT}/credentials"
LOCAL_CI_TASK_BENCHMARK_ROOT="${LOCAL_CI_TASK_TMP_ROOT}/benchmark"
TMPDIR="${LOCAL_CI_TASK_TMP_DIR}"
FAILURE_IR_ARTIFACT_DIR="${DELIVERY_ARTIFACT_DIR}/failure-ir"

export WORKSPACE ANCHOR_DIR BACKEND_PROFILE EXPECTED_TRITON_BACKEND BACKEND_PATH
export BACKEND_ENVSETUP BACKEND_ENVSETUP_ARGS BACKEND_TEST_COMMAND
export BACKEND_UNINSTALL_PACKAGES BACKEND_WHEEL_PATTERN
export RUN_FLAGGEMS_TESTS FLAGGEMS_CLONE_DIR FLAGGEMS_REF FLAGGEMS_PIP_PACKAGES FLAGGEMS_TEST_MODE FLAGGEMS_SAMPLE_SIZE FLAGGEMS_RANDOM_SEED FLAGGEMS_TEST_OP FLAGGEMS_TEST_COMMAND FLAGGEMS_PYTEST_ARGS FLAGGEMS_IDLE_TIMEOUT_SECONDS FLAGGEMS_TOTAL_TIMEOUT_SECONDS FLAGGEMS_FULL_TIMEOUT_EXTENSION_SECONDS FLAGGEMS_FULL_HARD_TIMEOUT_SECONDS FLAGGEMS_CLEAR_CACHE FLAGGEMS_WHITELIST FLAGGEMS_FULL_LIST FLAGGEMS_SELECTED_FILE
export LLVM_BUILD_DIR PPL_ROOT PYTHON_BIN PYTHON_VENV_ACTIVATE FRONTEND_BUILD_MODE GITHUB_SHA="${target_sha}" GITHUB_REF="refs/heads/${GITEE_BRANCH}"
export BACKEND_PROFILE MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL NINJAFLAGS UV_LINK_MODE
export LOCAL_CI_PROFILE_NAME LOCAL_CI_LLVM_HASH RUN_BACKEND_STAGES BACKEND_SKIP_REASON
export LOCAL_CI_TASK_TMP_ROOT TMPDIR

if [[ ! -r "${TASK_TMP_TOOL}" ]]; then
  echo "Task temporary directory tool is missing from the trusted runner snapshot: ${TASK_TMP_TOOL}" >&2
  exit 2
fi
"${PYTHON_BIN}" "${TASK_TMP_TOOL}" validate \
  --path "${LOCAL_CI_TASK_TMP_ROOT}" \
  --target-sha "${target_sha}" >/dev/null

task_tmp_path="$(realpath -e -- "${LOCAL_CI_TASK_TMP_ROOT}")"
artifact_path="$(realpath -m -- "${DELIVERY_ARTIFACT_DIR}")"
if [[ "${artifact_path}" == "${task_tmp_path}" || "${artifact_path}" == "${task_tmp_path}"/* ]]; then
  echo "DELIVERY_ARTIFACT_DIR must be outside the task temporary root: ${artifact_path}" >&2
  exit 2
fi
mkdir -p "${DELIVERY_ARTIFACT_DIR}"

use_uv() {
  [[ "${PACKAGE_TOOL}" == "uv" ]] || { [[ "${PACKAGE_TOOL}" == "auto" ]] && command -v uv >/dev/null 2>&1; }
}

prune_uv_cache_for_ci() {
  local prune_status=0
  if ! command -v uv >/dev/null 2>&1; then
    return 0
  fi

  echo "Pruning unused uv CI cache entries."
  uv cache prune --ci || prune_status=$?
  if [[ ${prune_status} -ne 0 ]]; then
    echo "Warning: uv cache prune --ci failed with exit code ${prune_status}; Local CI result is unchanged." >&2
  fi
  return 0
}

setup_gitee_git_auth() {
  if [[ -z "${GITEE_TOKEN}" ]]; then
    echo "GITEE_TOKEN is not set; git fetch will rely on existing credentials."
    export GIT_TERMINAL_PROMPT=0
    return 0
  fi

  local askpass
  askpass="$(mktemp "${LOCAL_CI_TASK_CREDENTIAL_DIR}/gitee-askpass.XXXXXX")"
  write_gitee_askpass "${askpass}"
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

validated_anchor_checkout_path() {
  if [[ "${WORKSPACE}" != /* || "${ANCHOR_DIR}" != /* ]]; then
    echo "WORKSPACE and ANCHOR_DIR must be absolute paths." >&2
    return 1
  fi
  if [[ -L "${ANCHOR_DIR}" ]]; then
    echo "Refusing to replace symlinked ANCHOR_DIR: ${ANCHOR_DIR}" >&2
    return 1
  fi

  local workspace_path anchor_path protected protected_path
  workspace_path="$(realpath -m -- "${WORKSPACE}")"
  anchor_path="$(realpath -m -- "${ANCHOR_DIR}")"
  if [[ "${anchor_path}" == "${workspace_path}" || "${anchor_path}" != "${workspace_path}"/* ]]; then
    echo "Refusing to replace ANCHOR_DIR outside WORKSPACE: ${anchor_path}" >&2
    return 1
  fi
  if command -v mountpoint >/dev/null 2>&1 \
    && [[ -e "${anchor_path}" ]] \
    && mountpoint -q "${anchor_path}"; then
    echo "Refusing to replace mounted ANCHOR_DIR: ${anchor_path}" >&2
    return 1
  fi

  for protected in \
    "${BACKEND_PATH}" \
    "${FLAGGEMS_CLONE_DIR}" \
    "${LLVM_BUILD_DIR}" \
    "${PPL_ROOT}" \
    "${LOCAL_CI_ARTIFACT_ROOT}"; do
    [[ -n "${protected}" ]] || continue
    protected_path="$(realpath -m -- "${protected}")"
    if [[ "${anchor_path}" == "${protected_path}" \
      || "${anchor_path}" == "${protected_path}"/* \
      || "${protected_path}" == "${anchor_path}"/* ]]; then
      echo "Refusing overlapping checkout/protected paths: ${anchor_path}, ${protected_path}" >&2
      return 1
    fi
  done

  printf '%s\n' "${anchor_path}"
}

fresh_checkout_anchor() {
  local anchor_path checked_out_sha
  anchor_path="$(validated_anchor_checkout_path)"

  echo "Removing previous frontend checkout: ${anchor_path}"
  rm -rf -- "${anchor_path}"
  mkdir -p -- "$(dirname "${anchor_path}")"

  echo "Cloning ${GITEE_BRANCH} from ${GITEE_REPO_URL}"
  git clone \
    --origin gitee \
    --branch "${GITEE_BRANCH}" \
    --single-branch \
    --no-checkout \
    "${GITEE_REPO_URL}" \
    "${anchor_path}"
  git config --global --add safe.directory "${anchor_path}" || true
  git -C "${anchor_path}" checkout --detach "${target_sha}"
  git -C "${anchor_path}" reset --hard "${target_sha}"
  git -C "${anchor_path}" clean -ffdx

  checked_out_sha="$(git -C "${anchor_path}" rev-parse HEAD)"
  if [[ "${checked_out_sha}" != "${target_sha}" ]]; then
    echo "Fresh checkout SHA mismatch: expected ${target_sha}, got ${checked_out_sha}" >&2
    return 1
  fi
}

incremental_checkout_anchor() {
  local anchor_path checked_out_sha
  anchor_path="$(validated_anchor_checkout_path)"

  if [[ ! -e "${anchor_path}" ]]; then
    echo "No existing frontend checkout; using a cold clone for incremental mode."
    fresh_checkout_anchor
    return
  fi
  if [[ ! -d "${anchor_path}/.git" ]] \
    || ! git -C "${anchor_path}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Incremental frontend checkout is not a Git worktree: ${anchor_path}" >&2
    echo "Use FRONTEND_BUILD_MODE=fresh to replace it." >&2
    return 1
  fi

  git config --global --add safe.directory "${anchor_path}" || true
  if git -C "${anchor_path}" remote get-url gitee >/dev/null 2>&1; then
    git -C "${anchor_path}" remote set-url gitee "${GITEE_REPO_URL}"
  else
    git -C "${anchor_path}" remote add gitee "${GITEE_REPO_URL}"
  fi

  echo "Fetching ${GITEE_BRANCH} into existing frontend checkout."
  git -C "${anchor_path}" fetch \
    --prune \
    --no-tags \
    gitee \
    "refs/heads/${GITEE_BRANCH}"
  if ! git -C "${anchor_path}" cat-file -e "${target_sha}^{commit}" 2>/dev/null; then
    echo "Target SHA was not fetched from ${GITEE_BRANCH}: ${target_sha}" >&2
    return 1
  fi

  git -C "${anchor_path}" checkout --detach "${target_sha}"
  git -C "${anchor_path}" reset --hard "${target_sha}"
  git -C "${anchor_path}" clean -ffdx -e /build/

  checked_out_sha="$(git -C "${anchor_path}" rev-parse HEAD)"
  if [[ "${checked_out_sha}" != "${target_sha}" ]]; then
    echo "Incremental checkout SHA mismatch: expected ${target_sha}, got ${checked_out_sha}" >&2
    return 1
  fi
}

prepare_anchor_checkout() {
  case "${FRONTEND_BUILD_MODE}" in
    fresh)
      fresh_checkout_anchor
      ;;
    incremental)
      incremental_checkout_anchor
      ;;
    *)
      echo "Invalid FRONTEND_BUILD_MODE=${FRONTEND_BUILD_MODE@Q}; expected fresh or incremental." >&2
      return 2
      ;;
  esac
}

prune_task_dumps() {
  "${PYTHON_BIN}" "${DUMP_ARTIFACT_TOOL}" prune \
    --profile task-dumps --root / >/dev/null
}

clean_task_dump_stage() {
  local stage_name="$1"
  local safe_stage
  safe_stage="$(safe_path_part "${stage_name}")"
  "${PYTHON_BIN}" "${TASK_TMP_TOOL}" clean-owned \
    --path "${LOCAL_CI_TASK_TMP_ROOT}" \
    --target-sha "${target_sha}" \
    --relative "dump/${safe_stage}" >/dev/null
}

prepare_dump_stage() {
  local stage_name="$1"
  local safe_stage
  safe_stage="$(safe_path_part "${stage_name}")"
  prune_task_dumps || return $?
  mkdir -p "${LOCAL_CI_TASK_DUMP_ROOT}/${safe_stage}" || return $?
  printf '%s' "${LOCAL_CI_TASK_DUMP_ROOT}/${safe_stage}"
}

collect_failure_ir() {
  local stage_name="$1"
  local task_dump_dir="$2"
  local collection_log="${DELIVERY_ARTIFACT_DIR}/failure-ir-collection.log"
  "${PYTHON_BIN}" "${DUMP_ARTIFACT_TOOL}" collect \
    --output-dir "${FAILURE_IR_ARTIFACT_DIR}" \
    --stage "${stage_name}" \
    --target-sha "${target_sha}" \
    --source "task=${task_dump_dir}" \
    --source "sophgo=${SOPHGO_TRITON_DUMP_DIR}" \
    --source "root=${ROOT_TRITON_DUMP_DIR}" \
    >> "${collection_log}" 2>&1
}

finish_dump_stage() {
  local stage_name="$1"
  local task_dump_dir="$2"
  local stage_status="$3"
  local collection_status=0
  local cleanup_status=0
  local task_cleanup_status=0
  if [[ ${stage_status} -ne 0 ]]; then
    collect_failure_ir "${stage_name}" "${task_dump_dir}" || collection_status=$?
    if [[ ${collection_status} -ne 0 ]]; then
      echo "Failed to collect useful IR for ${stage_name}; see ${DELIVERY_ARTIFACT_DIR}/failure-ir-collection.log" >&2
    fi
  fi
  prune_task_dumps || cleanup_status=$?
  clean_task_dump_stage "${stage_name}" || task_cleanup_status=$?
  if [[ ${task_cleanup_status} -ne 0 ]]; then
    echo "Failed to clean task-owned Triton dump directory after ${stage_name}." >&2
    if [[ ${cleanup_status} -eq 0 ]]; then
      cleanup_status="${task_cleanup_status}"
    fi
  fi
  if [[ ${cleanup_status} -ne 0 ]]; then
    echo "Failed to clean managed Triton dump directories after ${stage_name}." >&2
  fi
  return "${cleanup_status}"
}

run_logged() {
  local name="$1"
  shift
  local log_file="${DELIVERY_ARTIFACT_DIR}/${name}.log"
  local task_dump_dir=""
  local stage_status=0
  local cleanup_status=0
  local had_errexit=0
  local pipeline_statuses=()
  local pipeline_status
  if [[ $- == *e* ]]; then
    had_errexit=1
  fi
  if ! task_dump_dir="$(prepare_dump_stage "${name}")"; then
    echo "Cannot prepare managed Triton dump directory for ${name}." >&2
    return 1
  fi
  echo "Running ${name}; log: ${log_file}"
  set +e
  (
    export TRITON_DUMP_DIR="${task_dump_dir}"
    "$@"
  ) 2>&1 | tee "${log_file}"
  pipeline_statuses=("${PIPESTATUS[@]}")
  for pipeline_status in "${pipeline_statuses[@]}"; do
    if [[ ${pipeline_status} -ne 0 ]]; then
      stage_status="${pipeline_status}"
    fi
  done
  finish_dump_stage "${name}" "${task_dump_dir}" "${stage_status}" || cleanup_status=$?
  if [[ ${stage_status} -eq 0 && ${cleanup_status} -ne 0 ]]; then
    stage_status="${cleanup_status}"
  fi
  if [[ ${had_errexit} -eq 1 ]]; then
    set -e
  else
    set +e
  fi
  return "${stage_status}"
}

frontend_package_installed() {
  "${PYTHON_BIN}" -c \
    'from importlib.metadata import distribution; distribution("triton-anchor")' \
    >/dev/null 2>&1
}

uninstall_installed_frontend() {
  if ! frontend_package_installed; then
    echo "No previously installed triton-anchor distribution found." \
      | tee "${DELIVERY_ARTIFACT_DIR}/frontend-uninstall.log"
    return 0
  fi

  if use_uv; then
    run_logged frontend-uninstall uv pip uninstall triton-anchor
  else
    run_logged frontend-uninstall \
      "${PYTHON_BIN}" -m pip uninstall -y triton-anchor
  fi

  if frontend_package_installed; then
    echo "triton-anchor is still installed after uninstall." >&2
    return 1
  fi
}

mark_stage_failed() {
  local status_var="$1"
  local stage_name="$2"
  local message="$3"
  printf -v "${status_var}" '%s' "fail"
  LOCAL_CI_RESULT_STATUS=1
  echo "${stage_name} failed: ${message}" >&2
}

run_recorded_stage() {
  local status_var="$1"
  local stage_name="$2"
  shift 2

  local stage_exit=0
  set +e
  "$@"
  stage_exit=$?
  set -e

  if [[ ${stage_exit} -eq 0 ]]; then
    if [[ "${!status_var}" == "not_run" || "${!status_var}" == "running" ]]; then
      printf -v "${status_var}" '%s' "pass"
    fi
    echo "${stage_name} status: ${!status_var}"
  else
    mark_stage_failed "${status_var}" "${stage_name}" "exit ${stage_exit}"
  fi
  return 0
}

run_recorded_stage_in_dir() {
  local status_var="$1"
  local stage_name="$2"
  local directory="$3"
  local log_name="$4"
  shift 4
  run_recorded_stage "${status_var}" "${stage_name}" \
    run_logged_in_dir "${directory}" "${log_name}" "$@"
}

run_logged_in_dir() {
  local directory="$1"
  local log_name="$2"
  shift 2
  (cd "${directory}" && run_logged "${log_name}" "$@")
}

rebuild_backend() {
  if [[ ! -d "${BACKEND_PATH}" ]]; then
    echo "Backend path does not exist: ${BACKEND_PATH}" >&2
    return 1
  fi
  run_logged_in_dir "${BACKEND_PATH}" backend-rebuild rebuild_backend_command
}

rebuild_backend_command() {
  set -euo pipefail
  local backend_wheel=""
  if use_uv; then
    uv pip install scikit-build-core pybind11
    if [[ -n "${BACKEND_UNINSTALL_PACKAGES}" ]]; then
      # shellcheck disable=SC2086
      uv pip uninstall ${BACKEND_UNINSTALL_PACKAGES} || true
    fi
    rm -rf build dist *.egg-info
    uv build --wheel --no-build-isolation
  else
    "${PYTHON_BIN}" -m pip install scikit-build-core pybind11 build
    if [[ -n "${BACKEND_UNINSTALL_PACKAGES}" ]]; then
      # shellcheck disable=SC2086
      "${PYTHON_BIN}" -m pip uninstall -y ${BACKEND_UNINSTALL_PACKAGES} || true
    fi
    rm -rf build dist *.egg-info
    "${PYTHON_BIN}" -m build --wheel --no-isolation
  fi

  backend_wheel="$(
    find dist -maxdepth 1 -type f -name "${BACKEND_WHEEL_PATTERN}" \
      -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
  )"
  if [[ -z "${backend_wheel}" ]]; then
    echo "Backend build did not produce a wheel matching ${BACKEND_WHEEL_PATTERN}." >&2
    return 1
  fi
  if use_uv; then
    uv pip install --force-reinstall "${backend_wheel}"
  else
    "${PYTHON_BIN}" -m pip install --force-reinstall "${backend_wheel}"
  fi
  cp "${backend_wheel}" "${DELIVERY_ARTIFACT_DIR}/"
  ls -lh "${backend_wheel}" "${DELIVERY_ARTIFACT_DIR}/$(basename "${backend_wheel}")"
}

source_python_venv() {
  if [[ -z "${PYTHON_VENV_ACTIVATE}" ]]; then
    return 0
  fi
  if [[ ! -f "${PYTHON_VENV_ACTIVATE}" ]]; then
    echo "Python venv activate script does not exist: ${PYTHON_VENV_ACTIVATE}" >&2
    return 1
  fi
  echo "Sourcing Python venv: ${PYTHON_VENV_ACTIVATE}"
  set +u
  # shellcheck disable=SC1090
  source "${PYTHON_VENV_ACTIVATE}"
  set -u
}

source_anchor_env() {
  if [[ "${SOURCE_ENVSETUP}" != "1" ]]; then
    if [[ -n "${LOCAL_CI_BASE_SHA}" ]]; then
      echo "PR Local CI requires SOURCE_ENVSETUP=1 and a trusted base envsetup.sh." >&2
      return 1
    fi
    return 0
  fi

  local envsetup_file="${ANCHOR_DIR}/envsetup.sh"
  if [[ -n "${LOCAL_CI_BASE_SHA}" ]]; then
    if [[ -f "${ANCHOR_DIR}/envsetup.sh" ]]; then
      echo "Syntax-checking candidate envsetup.sh without sourcing it."
      bash -n "${ANCHOR_DIR}/envsetup.sh"
    fi
    if [[ ! -f "${TRUSTED_ANCHOR_ENVSETUP}" ]]; then
      echo "Trusted base envsetup.sh is required for PR Local CI." >&2
      return 1
    fi
    envsetup_file="${TRUSTED_ANCHOR_ENVSETUP}"
    echo "Sourcing trusted base envsetup.sh: ${envsetup_file}"
  elif [[ ! -f "${envsetup_file}" ]]; then
    echo "Push commit has no envsetup.sh to source."
    return 0
  else
    echo "Sourcing push commit envsetup.sh."
  fi

  if [[ -f "${envsetup_file}" ]]; then
    local command_tmp_dir="${TMPDIR}"
    bash -n "${envsetup_file}"
    set +u
    # shellcheck disable=SC1090
    source "${envsetup_file}"
    export TMPDIR="${command_tmp_dir}"
    set -u
  fi
}

source_backend_env() {
  local setup_script="${BACKEND_ENVSETUP}"
  local command_dump_dir="${TRITON_DUMP_DIR:-}"
  local command_tmp_dir="${TMPDIR}"
  local setup_status=0
  if [[ -z "${setup_script}" ]]; then
    return 0
  fi
  if [[ "${setup_script}" != /* ]]; then
    setup_script="${BACKEND_PATH}/${setup_script}"
  fi
  if [[ ! -f "${setup_script}" ]]; then
    echo "Backend envsetup script does not exist: ${setup_script}" >&2
    return 1
  fi
  echo "Sourcing backend envsetup: ${setup_script} ${BACKEND_ENVSETUP_ARGS}"
  set +u
  # shellcheck disable=SC1090,SC2086
  source "${setup_script}" ${BACKEND_ENVSETUP_ARGS} || setup_status=$?
  if [[ -n "${command_dump_dir}" ]]; then
    export TRITON_DUMP_DIR="${command_dump_dir}"
  fi
  export TMPDIR="${command_tmp_dir}"
  set -u
  return "${setup_status}"
}

fetch_performance_baseline() {
  local result_dir="$1"
  local branch_label="$2"
  local cache_label="$3"
  local sha="$4"
  local output="$5"
  local safe_profile
  safe_profile="$(safe_path_part "${BACKEND_PROFILE}")"
  local rel_path="${result_dir}/by-sha/${sha}/${safe_profile}/latest.json"

  if git remote get-url gitee-results >/dev/null 2>&1; then
    git remote set-url gitee-results "${GITEE_RESULTS_REPO_URL}"
  else
    git remote add gitee-results "${GITEE_RESULTS_REPO_URL}"
  fi
  if ! git fetch -q --depth=1 gitee-results \
    "refs/heads/${GITEE_RESULTS_BRANCH}:refs/remotes/gitee-results/${GITEE_RESULTS_BRANCH}"; then
    echo "${branch_label} results branch is not available: ${GITEE_RESULTS_BRANCH}" >&2
    return 1
  fi
  if ! git show "gitee-results/${GITEE_RESULTS_BRANCH}:${rel_path}" > "${output}"; then
    rm -f "${output}"
    echo "No cached ${cache_label} baseline at ${rel_path}" >&2
    return 1
  fi
  echo "Loaded ${cache_label} baseline for ${sha}: ${rel_path}"
}

fetch_compile_baseline() {
  fetch_performance_baseline "compile-time" "Compile-time" "compile-time" "$@"
}

fetch_pass_profile_baseline() {
  fetch_performance_baseline "pass-profile" "Pass-profile" "pass-profile" "$@"
}

fetch_ir_serialization_baseline() {
  fetch_performance_baseline \
    "ir-serialization" "IR serialization" "IR serialization" "$@"
}

read_comparison_status() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["status"])
PY
}

run_compile_benchmark() {
  if [[ "${RUN_COMPILE_BENCHMARK}" != "true" ]]; then
    COMPILE_TIME_STATUS="disabled"
    return 0
  fi
  COMPILE_TIME_STATUS="running"
  if [[ ! -f "${PERFORMANCE_SCRIPT_DIR}/compile_benchmark.py" ]]; then
    echo "Compile benchmark script is missing from the trusted runner snapshot." >&2
    COMPILE_TIME_STATUS="fail"
    return 1
  fi
  if [[ ! -f "${PERFORMANCE_SCRIPT_DIR}/compare_compile_time.py" ]]; then
    echo "Compile comparison script is missing from the trusted runner snapshot." >&2
    COMPILE_TIME_STATUS="fail"
    return 1
  fi

  local candidate_json="${DELIVERY_ARTIFACT_DIR}/compile-benchmark.json"
  local candidate_csv="${DELIVERY_ARTIFACT_DIR}/compile-benchmark.csv"
  export FLAGGEMS_ROOT="${FLAGGEMS_CLONE_DIR}"
  if ! source_backend_env; then
    COMPILE_TIME_STATUS="fail"
    return 1
  fi
  if ! run_logged compile-benchmark timeout "${COMPILE_BENCHMARK_TIMEOUT}" \
    "${PYTHON_BIN}" "${PERFORMANCE_SCRIPT_DIR}/compile_benchmark.py" \
      --backend "${EXPECTED_TRITON_BACKEND}" \
      --vendor "${EXPECTED_TRITON_BACKEND}" \
      --flaggems-root "${FLAGGEMS_CLONE_DIR}" \
      --kernels "${COMPILE_BENCHMARK_KERNELS}" \
      --repeat "${COMPILE_BENCHMARK_REPEAT}" \
      --warmup "${COMPILE_BENCHMARK_WARMUP}" \
      --cache-root "${LOCAL_CI_TASK_BENCHMARK_ROOT}/compile/cache" \
      --dump-root "${LOCAL_CI_TASK_BENCHMARK_ROOT}/compile/dump" \
      --output-json "${candidate_json}" \
      --output-csv "${candidate_csv}"; then
    COMPILE_TIME_STATUS="fail"
    return 1
  fi

  COMPILE_TIME_STATUS="pass"
  if [[ -n "${LOCAL_CI_BASE_SHA}" ]]; then
    local baseline_json="${DELIVERY_ARTIFACT_DIR}/compile-benchmark-base.json"
    if [[ ! -f "${baseline_json}" ]]; then
      echo "Compile-time baseline was not prefetched for ${LOCAL_CI_BASE_SHA}; comparison will report a warning." >&2
    fi
    if ! "${PYTHON_BIN}" "${PERFORMANCE_SCRIPT_DIR}/compare_compile_time.py" \
      --baseline-json "${baseline_json}" \
      --candidate-json "${candidate_json}" \
      --base-sha "${LOCAL_CI_BASE_SHA}" \
      --candidate-sha "${target_sha}" \
      --kernels "${COMPILE_BENCHMARK_KERNELS}" \
      --threshold "${COMPILE_BENCHMARK_THRESHOLD}" \
      --output-json "${DELIVERY_ARTIFACT_DIR}/compile-time-comparison.json" \
      --output-markdown "${DELIVERY_ARTIFACT_DIR}/compile-time-comparison.md" \
      2>&1 | tee "${DELIVERY_ARTIFACT_DIR}/compile-time-comparison.log"; then
      COMPILE_TIME_STATUS="fail"
      return 1
    fi
    local comparison_status=""
    if ! comparison_status="$(read_comparison_status \
      "${DELIVERY_ARTIFACT_DIR}/compile-time-comparison.json")"; then
      COMPILE_TIME_STATUS="fail"
      return 1
    fi
    COMPILE_TIME_STATUS="${comparison_status}"
  fi
  return 0
}

run_pass_profile() {
  if [[ "${RUN_PASS_PROFILE}" != "true" ]]; then
    PASS_PROFILE_STATUS="disabled"
    return 0
  fi
  PASS_PROFILE_STATUS="running"
  if [[ ! -f "${PERFORMANCE_SCRIPT_DIR}/pass_profile_benchmark.py" ]]; then
    echo "Pass profile script is missing from the trusted runner snapshot." >&2
    PASS_PROFILE_STATUS="fail"
    return 1
  fi
  if [[ ! -f "${PERFORMANCE_SCRIPT_DIR}/compare_pass_profile.py" ]]; then
    echo "Pass profile comparison script is missing from the trusted runner snapshot." >&2
    PASS_PROFILE_STATUS="fail"
    return 1
  fi

  local candidate_json="${DELIVERY_ARTIFACT_DIR}/pass-profile.json"
  local candidate_events_csv="${DELIVERY_ARTIFACT_DIR}/pass-profile-events.csv"
  local candidate_summary_csv="${DELIVERY_ARTIFACT_DIR}/pass-profile-summary.csv"
  local hotspots_md="${DELIVERY_ARTIFACT_DIR}/pass-profile-hotspots.md"
  export FLAGGEMS_ROOT="${FLAGGEMS_CLONE_DIR}"
  if ! source_backend_env; then
    PASS_PROFILE_STATUS="fail"
    return 1
  fi
  if ! run_logged pass-profile timeout "${PASS_PROFILE_TIMEOUT}" \
    "${PYTHON_BIN}" "${PERFORMANCE_SCRIPT_DIR}/pass_profile_benchmark.py" \
      --backend "${EXPECTED_TRITON_BACKEND}" \
      --vendor "${EXPECTED_TRITON_BACKEND}" \
      --flaggems-root "${FLAGGEMS_CLONE_DIR}" \
      --kernels "${PASS_PROFILE_KERNELS}" \
      --repeat "${PASS_PROFILE_REPEAT}" \
      --warmup "${PASS_PROFILE_WARMUP}" \
      --cache-root "${LOCAL_CI_TASK_BENCHMARK_ROOT}/pass-profile/cache" \
      --dump-root "${LOCAL_CI_TASK_BENCHMARK_ROOT}/pass-profile/dump" \
      --output-json "${candidate_json}" \
      --output-events-csv "${candidate_events_csv}" \
      --output-summary-csv "${candidate_summary_csv}" \
      --output-hotspots-markdown "${hotspots_md}"; then
    PASS_PROFILE_STATUS="fail"
    return 1
  fi

  PASS_PROFILE_STATUS="pass"
  if [[ -n "${LOCAL_CI_BASE_SHA}" ]]; then
    local baseline_json="${DELIVERY_ARTIFACT_DIR}/pass-profile-base.json"
    if [[ ! -f "${baseline_json}" ]]; then
      echo "Pass-profile baseline was not prefetched for ${LOCAL_CI_BASE_SHA}; comparison will report a warning." >&2
    fi
    if ! "${PYTHON_BIN}" "${PERFORMANCE_SCRIPT_DIR}/compare_pass_profile.py" \
      --baseline-json "${baseline_json}" \
      --candidate-json "${candidate_json}" \
      --base-sha "${LOCAL_CI_BASE_SHA}" \
      --candidate-sha "${target_sha}" \
      --kernels "${PASS_PROFILE_KERNELS}" \
      --threshold "${PASS_PROFILE_THRESHOLD}" \
      --min-base-ms "${PASS_PROFILE_MIN_BASE_MS}" \
      --min-delta-ms "${PASS_PROFILE_MIN_DELTA_MS}" \
      --output-json "${DELIVERY_ARTIFACT_DIR}/pass-profile-comparison.json" \
      --output-csv "${DELIVERY_ARTIFACT_DIR}/pass-profile-comparison.csv" \
      --output-markdown "${DELIVERY_ARTIFACT_DIR}/pass-profile-comparison.md" \
      2>&1 | tee "${DELIVERY_ARTIFACT_DIR}/pass-profile-comparison.log"; then
      PASS_PROFILE_STATUS="fail"
      return 1
    fi
    local comparison_status=""
    if ! comparison_status="$(read_comparison_status \
      "${DELIVERY_ARTIFACT_DIR}/pass-profile-comparison.json")"; then
      PASS_PROFILE_STATUS="fail"
      return 1
    fi
    PASS_PROFILE_STATUS="${comparison_status}"
  fi
  return 0
}

run_ir_serialization_benchmark() {
  if [[ "${RUN_IR_SERIALIZATION_BENCHMARK}" != "true" ]]; then
    IR_SERIALIZATION_STATUS="disabled"
    return 0
  fi
  IR_SERIALIZATION_STATUS="running"
  if [[ ! -f "${PERFORMANCE_SCRIPT_DIR}/ir_serialization_benchmark.py" ]]; then
    echo "IR serialization benchmark is missing from the trusted runner snapshot." >&2
    IR_SERIALIZATION_STATUS="fail"
    return 1
  fi
  if [[ ! -f "${PERFORMANCE_SCRIPT_DIR}/compare_ir_serialization.py" ]]; then
    echo "IR serialization comparison is missing from the trusted runner snapshot." >&2
    IR_SERIALIZATION_STATUS="fail"
    return 1
  fi

  local candidate_json="${DELIVERY_ARTIFACT_DIR}/ir-serialization.json"
  local candidate_csv="${DELIVERY_ARTIFACT_DIR}/ir-serialization.csv"
  local candidate_markdown="${DELIVERY_ARTIFACT_DIR}/ir-serialization-summary.md"
  export FLAGGEMS_ROOT="${FLAGGEMS_CLONE_DIR}"
  if ! source_backend_env; then
    IR_SERIALIZATION_STATUS="fail"
    return 1
  fi
  if ! run_logged ir-serialization timeout "${IR_SERIALIZATION_TIMEOUT}" \
    "${PYTHON_BIN}" "${PERFORMANCE_SCRIPT_DIR}/ir_serialization_benchmark.py" \
      --backend "${EXPECTED_TRITON_BACKEND}" \
      --vendor "${EXPECTED_TRITON_BACKEND}" \
      --flaggems-root "${FLAGGEMS_CLONE_DIR}" \
      --kernels "${IR_SERIALIZATION_KERNELS}" \
      --repeat "${IR_SERIALIZATION_REPEAT}" \
      --warmup "${IR_SERIALIZATION_WARMUP}" \
      --work-root "${LOCAL_CI_TASK_BENCHMARK_ROOT}/ir-serialization" \
      --output-json "${candidate_json}" \
      --output-csv "${candidate_csv}" \
      --output-markdown "${candidate_markdown}"; then
    IR_SERIALIZATION_STATUS="fail"
    return 1
  fi

  IR_SERIALIZATION_STATUS="pass"
  if [[ -n "${LOCAL_CI_BASE_SHA}" ]]; then
    local baseline_json="${DELIVERY_ARTIFACT_DIR}/ir-serialization-base.json"
    if [[ ! -f "${baseline_json}" ]]; then
      echo "IR serialization baseline was not prefetched for ${LOCAL_CI_BASE_SHA}; comparison will report a warning." >&2
    fi
    if ! "${PYTHON_BIN}" "${PERFORMANCE_SCRIPT_DIR}/compare_ir_serialization.py" \
      --baseline-json "${baseline_json}" \
      --candidate-json "${candidate_json}" \
      --base-sha "${LOCAL_CI_BASE_SHA}" \
      --candidate-sha "${target_sha}" \
      --kernels "${IR_SERIALIZATION_KERNELS}" \
      --metrics "${IR_SERIALIZATION_METRICS}" \
      --threshold "${IR_SERIALIZATION_THRESHOLD}" \
      --min-base-ms "${IR_SERIALIZATION_MIN_BASE_MS}" \
      --min-delta-ms "${IR_SERIALIZATION_MIN_DELTA_MS}" \
      --output-json "${DELIVERY_ARTIFACT_DIR}/ir-serialization-comparison.json" \
      --output-csv "${DELIVERY_ARTIFACT_DIR}/ir-serialization-comparison.csv" \
      --output-markdown "${DELIVERY_ARTIFACT_DIR}/ir-serialization-comparison.md" \
      2>&1 | tee "${DELIVERY_ARTIFACT_DIR}/ir-serialization-comparison.log"; then
      IR_SERIALIZATION_STATUS="fail"
      return 1
    fi
    local comparison_status=""
    if ! comparison_status="$(read_comparison_status \
      "${DELIVERY_ARTIFACT_DIR}/ir-serialization-comparison.json")"; then
      IR_SERIALIZATION_STATUS="fail"
      return 1
    fi
    IR_SERIALIZATION_STATUS="${comparison_status}"
  fi
  return 0
}

git_commit() {
  local repo="$1"
  if [[ -z "${repo}" ]]; then
    return 0
  fi
  git -C "${repo}" rev-parse HEAD 2>/dev/null || true
}

write_summary() {
  local status="$1"
  local failure_ir_file_count=0
  if [[ -d "${FAILURE_IR_ARTIFACT_DIR}" ]]; then
    failure_ir_file_count="$(
      find "${FAILURE_IR_ARTIFACT_DIR}" -type f \
        \( -name '*.ttir' -o -name '*.linalg' -o -name '*.pplir' \) \
        | wc -l
    )"
  fi
  set +e
  {
    actual_anchor_commit="$(git_commit "${ANCHOR_DIR}")"
    tested_sha_kind="commit"
    if [[ "${GITEE_BRANCH}" =~ ^ci/pr-[0-9]+/.+ ]]; then
      tested_sha_kind="pr_merge"
    fi
    echo "schema: triton-anchor-local-ci/v3"
    echo "status: ${status}"
    echo "target_sha: ${target_sha}"
    echo "tested_sha: ${target_sha}"
    echo "tested_sha_kind: ${tested_sha_kind}"
    echo "actual_checkout_sha: ${actual_anchor_commit}"
    echo "run_id: ${LOCAL_CI_RUN_ID:-}"
    echo "base_sha: ${LOCAL_CI_BASE_SHA}"
    echo "base_ref: ${LOCAL_CI_BASE_REF}"
    echo "branch: ${GITEE_BRANCH}"
    echo "anchor_dir: ${ANCHOR_DIR}"
    echo "anchor_commit: ${actual_anchor_commit}"
    echo "frontend_build_mode: ${FRONTEND_BUILD_MODE}"
    echo "frontend_checkout_mode: ${FRONTEND_CHECKOUT_MODE}"
    echo "frontend_build_cache_preserved: ${FRONTEND_BUILD_CACHE_PRESERVED}"
    echo "frontend_uninstall_before_build: true"
    echo "ci_profile: ${LOCAL_CI_PROFILE_NAME}"
    echo "llvm_hash: ${LOCAL_CI_LLVM_HASH}"
    echo "backend_stages_enabled: ${RUN_BACKEND_STAGES}"
    echo "backend_skip_reason: ${BACKEND_SKIP_REASON}"
    echo "backend_profile: ${BACKEND_PROFILE}"
    echo "expected_backend: ${EXPECTED_TRITON_BACKEND}"
    echo "backend_path: ${BACKEND_PATH}"
    echo "backend_commit: $(git_commit "${BACKEND_PATH}")"
    echo "flaggems_enabled: ${RUN_FLAGGEMS_TESTS}"
    echo "flaggems_dir: ${FLAGGEMS_CLONE_DIR}"
    echo "flaggems_commit: $(git_commit "${FLAGGEMS_CLONE_DIR}")"
    echo "flaggems_test_mode: ${FLAGGEMS_TEST_MODE}"
    echo "flaggems_sample_size: ${FLAGGEMS_SAMPLE_SIZE}"
    echo "flaggems_random_seed: ${FLAGGEMS_RANDOM_SEED}"
    echo "flaggems_whitelist: ${FLAGGEMS_WHITELIST}"
    echo "flaggems_full_list: ${FLAGGEMS_FULL_LIST}"
    echo "flaggems_selected_file: ${FLAGGEMS_SELECTED_FILE}"
    echo "flaggems_test_op: ${FLAGGEMS_TEST_OP}"
    echo "flaggems_test_command: ${FLAGGEMS_TEST_COMMAND}"
    echo "flaggems_idle_timeout_seconds: ${FLAGGEMS_IDLE_TIMEOUT_SECONDS}"
    echo "flaggems_total_timeout_seconds: ${FLAGGEMS_TOTAL_TIMEOUT_SECONDS}"
    echo "flaggems_full_timeout_extension_seconds: ${FLAGGEMS_FULL_TIMEOUT_EXTENSION_SECONDS}"
    echo "flaggems_full_hard_timeout_seconds: ${FLAGGEMS_FULL_HARD_TIMEOUT_SECONDS}"
    echo "llvm_build_dir: ${LLVM_BUILD_DIR}"
    echo "ppl_root: ${PPL_ROOT}"
    echo "artifact_dir: ${DELIVERY_ARTIFACT_DIR}"
    echo "failure_ir_artifact_dir: $([[ -d "${FAILURE_IR_ARTIFACT_DIR}" ]] && printf '%s' "${FAILURE_IR_ARTIFACT_DIR}")"
    echo "failure_ir_file_count: ${failure_ir_file_count}"
    echo "triton_dump_cleanup_status: ${TRITON_DUMP_CLEANUP_STATUS}"
    echo "frontend_build_status: ${FRONTEND_BUILD_STATUS}"
    echo "frontend_smoke_status: ${FRONTEND_SMOKE_STATUS}"
    echo "backend_rebuild_status: ${BACKEND_REBUILD_STATUS}"
    echo "backend_smoke_jit_status: ${BACKEND_SMOKE_JIT_STATUS}"
    echo "flaggems_status: ${FLAGGEMS_STATUS}"
    echo "compile_time_status: ${COMPILE_TIME_STATUS}"
    echo "compile_time_threshold: ${COMPILE_BENCHMARK_THRESHOLD}"
    echo "pass_profile_status: ${PASS_PROFILE_STATUS}"
    echo "pass_profile_threshold: ${PASS_PROFILE_THRESHOLD}"
    echo "ir_serialization_status: ${IR_SERIALIZATION_STATUS}"
    echo "ir_serialization_threshold: ${IR_SERIALIZATION_THRESHOLD}"
  } > "${DELIVERY_ARTIFACT_DIR}/delivery-summary.txt"
  set -e
}

finalize_running_statuses() {
  local status_var
  for status_var in \
    FRONTEND_BUILD_STATUS FRONTEND_SMOKE_STATUS BACKEND_REBUILD_STATUS \
    BACKEND_SMOKE_JIT_STATUS FLAGGEMS_STATUS COMPILE_TIME_STATUS \
    PASS_PROFILE_STATUS IR_SERIALIZATION_STATUS; do
    if [[ "${!status_var}" == "running" ]]; then
      printf -v "${status_var}" '%s' "fail"
    fi
  done
}

required_stages_passed() {
  local status
  local required_statuses=(
    "${FRONTEND_BUILD_STATUS}"
    "${FRONTEND_SMOKE_STATUS}"
  )
  if [[ "${RUN_BACKEND_STAGES}" == "true" ]]; then
    required_statuses+=(
      "${BACKEND_REBUILD_STATUS}"
      "${BACKEND_SMOKE_JIT_STATUS}"
    )
  fi
  for status in "${required_statuses[@]}"; do
    case "${status}" in
      pass|success) ;;
      *) return 1 ;;
    esac
  done
}

on_exit() {
  local status="$?"
  local dump_cleanup_status=0
  trap - EXIT
  set +e
  if [[ ${status} -eq 0 && ${LOCAL_CI_RESULT_STATUS} -ne 0 ]]; then
    status="${LOCAL_CI_RESULT_STATUS}"
  fi
  finalize_running_statuses
  if [[ ${status} -eq 0 ]] && ! required_stages_passed; then
    echo "One or more required Local CI stages did not pass; forcing overall failure." >&2
    status=1
  fi
  cleanup_gitee_git_auth
  if [[ ${status} -ne 0 ]]; then
    collect_failure_ir "unhandled-exit" "${LOCAL_CI_TASK_DUMP_ROOT}" || true
  fi
  prune_task_dumps || dump_cleanup_status=$?
  if [[ ${dump_cleanup_status} -eq 0 ]]; then
    TRITON_DUMP_CLEANUP_STATUS="pass"
  else
    TRITON_DUMP_CLEANUP_STATUS="fail"
    echo "Failed to clean managed Triton dump directories during Local CI exit." >&2
    if [[ ${status} -eq 0 ]]; then
      status="${dump_cleanup_status}"
    fi
  fi
  prune_uv_cache_for_ci
  write_summary "${status}"
  exit "${status}"
}
trap on_exit EXIT

if [[ ! -r "${DUMP_ARTIFACT_TOOL}" ]]; then
  echo "Dump artifact tool is missing from the trusted runner snapshot: ${DUMP_ARTIFACT_TOOL}" >&2
  exit 2
fi
prune_task_dumps

FRONTEND_BUILD_STATUS="running"
case "${FRONTEND_BUILD_MODE}" in
  fresh)
    FRONTEND_CHECKOUT_MODE="fresh_clone"
    FRONTEND_BUILD_CACHE_PRESERVED="false"
    ;;
  incremental)
    if [[ -e "${ANCHOR_DIR}" ]]; then
      FRONTEND_CHECKOUT_MODE="incremental_fetch"
    else
      FRONTEND_CHECKOUT_MODE="incremental_cold_clone"
    fi
    if [[ -d "${ANCHOR_DIR}/build" ]]; then
      FRONTEND_BUILD_CACHE_PRESERVED="true"
    fi
    ;;
  *)
    echo "Invalid FRONTEND_BUILD_MODE=${FRONTEND_BUILD_MODE@Q}; expected fresh or incremental." >&2
    exit 2
    ;;
esac

setup_gitee_git_auth
run_logged frontend-checkout prepare_anchor_checkout
cd "${ANCHOR_DIR}"

if [[ -n "${LOCAL_CI_BASE_SHA}" ]]; then
  if [[ "${RUN_COMPILE_BENCHMARK}" == "true" ]]; then
    fetch_compile_baseline \
      "${LOCAL_CI_BASE_SHA}" \
      "${DELIVERY_ARTIFACT_DIR}/compile-benchmark-base.json" || true
  fi
  if [[ "${RUN_PASS_PROFILE}" == "true" ]]; then
    fetch_pass_profile_baseline \
      "${LOCAL_CI_BASE_SHA}" \
      "${DELIVERY_ARTIFACT_DIR}/pass-profile-base.json" || true
  fi
  if [[ "${RUN_IR_SERIALIZATION_BENCHMARK}" == "true" ]]; then
    fetch_ir_serialization_baseline \
      "${LOCAL_CI_BASE_SHA}" \
      "${DELIVERY_ARTIFACT_DIR}/ir-serialization-base.json" || true
  fi
fi

cleanup_gitee_git_auth
unset GITEE_TOKEN GITEE_USERNAME GIT_ASKPASS
LOCAL_CI_GIT_ASKPASS=""

cat <<EOF
Local CI commit: ${target_sha}
Anchor dir: ${ANCHOR_DIR}
Frontend build mode: ${FRONTEND_BUILD_MODE}
Backend profile: ${BACKEND_PROFILE}
Backend path: ${BACKEND_PATH}
Run FlagGems: ${RUN_FLAGGEMS_TESTS}
Artifact dir: ${DELIVERY_ARTIFACT_DIR}
EOF

source_python_venv
uninstall_installed_frontend
source_anchor_env

if [[ -z "${FRONTEND_BUILD_COMMAND}" ]]; then
  if use_uv; then
    FRONTEND_BUILD_COMMAND="uv build --wheel --no-build-isolation"
  else
    FRONTEND_BUILD_COMMAND="${PYTHON_BIN} -m build --wheel --no-isolation"
  fi
fi
if [[ "${FRONTEND_BUILD_MODE}" == "fresh" ]]; then
  echo "Clearing frontend build cache and wheel output under ${ANCHOR_DIR}"
  rm -rf -- "${ANCHOR_DIR}/build" "${ANCHOR_DIR}/dist"
else
  echo "Preserving frontend build cache at ${ANCHOR_DIR}/build"
  echo "Clearing frontend wheel output under ${ANCHOR_DIR}/dist"
  rm -rf -- "${ANCHOR_DIR}/dist"
fi
find "${ANCHOR_DIR}" -maxdepth 1 -name '*.egg-info' -exec rm -rf -- {} +
mkdir -p "${ANCHOR_DIR}/dist"

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
FRONTEND_BUILD_STATUS="pass"

run_recorded_stage_in_dir FRONTEND_SMOKE_STATUS "Frontend smoke" \
  "${ANCHOR_DIR}" frontend-smoke "${PYTHON_BIN}" tests/test_smoke.py

if [[ "${RUN_BACKEND_STAGES}" == "true" ]]; then
BACKEND_REBUILD_STATUS="running"
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
BACKEND_REBUILD_STATUS="pass"

if [[ -n "${BACKEND_TEST_COMMAND}" ]]; then
  run_recorded_stage_in_dir BACKEND_SMOKE_JIT_STATUS "Backend smoke and JIT" \
    "${BACKEND_PATH}" backend-smoke-jit bash -lc "${BACKEND_TEST_COMMAND}"
else
  BACKEND_SMOKE_JIT_STATUS="disabled"
fi

if [[ ("${RUN_FLAGGEMS_TESTS}" == "true" || "${RUN_COMPILE_BENCHMARK}" == "true" \
  || "${RUN_PASS_PROFILE}" == "true" || "${RUN_IR_SERIALIZATION_BENCHMARK}" == "true") \
  && "${INSTALL_FLAGGEMS_PACKAGES}" != "0" && -n "${FLAGGEMS_PIP_PACKAGES}" ]]; then
  if use_uv; then
    run_logged flaggems-deps uv pip install ${FLAGGEMS_PIP_PACKAGES}
  else
    run_logged flaggems-deps "${PYTHON_BIN}" -m pip install ${FLAGGEMS_PIP_PACKAGES}
  fi
fi

if [[ "${RUN_FLAGGEMS_TESTS}" == "true" ]]; then
  FLAGGEMS_STATUS="running"
  if [[ ! -d "${FLAGGEMS_CLONE_DIR}" ]]; then
    mark_stage_failed FLAGGEMS_STATUS "FlagGems" "repo does not exist: ${FLAGGEMS_CLONE_DIR}"
  elif [[ -n "${FLAGGEMS_REF}" ]] && ! git -C "${FLAGGEMS_CLONE_DIR}" checkout "${FLAGGEMS_REF}"; then
    mark_stage_failed FLAGGEMS_STATUS "FlagGems" "cannot checkout ${FLAGGEMS_REF}"
  else
    export FLAGGEMS_ROOT="${FLAGGEMS_CLONE_DIR}"
    if ! source_backend_env; then
      mark_stage_failed FLAGGEMS_STATUS "FlagGems" "backend environment setup failed"
    elif [[ -n "${FLAGGEMS_TEST_COMMAND}" ]]; then
      run_recorded_stage_in_dir FLAGGEMS_STATUS "FlagGems" \
        "${BACKEND_PATH}" flaggems bash -lc "${FLAGGEMS_TEST_COMMAND}"
    else
      flaggems_runner="${FLAGGEMS_SCRIPT_DIR}/batch_test_flaggems.py"
      if [[ ! -f "${flaggems_runner}" ]]; then
        mark_stage_failed FLAGGEMS_STATUS "FlagGems" \
          "batch runner does not exist: ${flaggems_runner}"
      else
        run_recorded_stage FLAGGEMS_STATUS "FlagGems" run_logged flaggems \
          "${PYTHON_BIN}" "${flaggems_runner}" \
            --mode "${FLAGGEMS_TEST_MODE}" \
            --sample-size "${FLAGGEMS_SAMPLE_SIZE}" \
            --seed "${FLAGGEMS_RANDOM_SEED}" \
            --op "${FLAGGEMS_TEST_OP}" \
            --whitelist "${FLAGGEMS_WHITELIST}" \
            --full-list "${FLAGGEMS_FULL_LIST}" \
            --flaggems-dir "${FLAGGEMS_CLONE_DIR}" \
            --python-bin "${PYTHON_BIN}" \
            --artifact-dir "${DELIVERY_ARTIFACT_DIR}" \
            --selected-output "${FLAGGEMS_SELECTED_FILE}" \
            --pytest-args="${FLAGGEMS_PYTEST_ARGS}" \
            --idle-timeout-seconds "${FLAGGEMS_IDLE_TIMEOUT_SECONDS}" \
            --total-timeout-seconds "${FLAGGEMS_TOTAL_TIMEOUT_SECONDS}" \
            --full-timeout-extension-seconds "${FLAGGEMS_FULL_TIMEOUT_EXTENSION_SECONDS}" \
            --full-hard-timeout-seconds "${FLAGGEMS_FULL_HARD_TIMEOUT_SECONDS}" \
            --clear-cache "${FLAGGEMS_CLEAR_CACHE}"
      fi
    fi
  fi
else
  FLAGGEMS_STATUS="disabled"
fi

run_recorded_stage COMPILE_TIME_STATUS "Compile-time benchmark" run_compile_benchmark
run_recorded_stage PASS_PROFILE_STATUS "Pass profile" run_pass_profile
run_recorded_stage IR_SERIALIZATION_STATUS "IR serialization" run_ir_serialization_benchmark
else
  echo "Skipping backend-dependent stages: ${BACKEND_SKIP_REASON}"
fi

if [[ ${LOCAL_CI_RESULT_STATUS} -ne 0 ]]; then
  echo "Local CI finished with one or more failed stages. Artifacts are in ${DELIVERY_ARTIFACT_DIR}" >&2
  exit "${LOCAL_CI_RESULT_STATUS}"
fi
echo "Local CI finished successfully. Artifacts are in ${DELIVERY_ARTIFACT_DIR}"
