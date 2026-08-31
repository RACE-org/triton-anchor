#!/usr/bin/env bash
set -euo pipefail

LOCAL_CI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${LOCAL_CI_ROOT}/shared/path_utils.sh"

CONFIG_FILE="${LOCAL_CI_CONFIG:-${LOCAL_CI_ROOT}/config.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
LOCAL_CI_CONFIG="${CONFIG_FILE}"

LOCAL_CI_STATE_DIR="${LOCAL_CI_STATE_DIR:-/home/localci/local_ci/local-ci-state}"
LOCAL_CI_SCRIPT_DIR="${LOCAL_CI_SCRIPT_DIR:-${LOCAL_CI_ROOT}}"

GITEE_REPO_URL="${GITEE_REPO_URL:-https://gitee.com/race-org/triton-anchor-local-ci-results.git}"
GITEE_OWNER="${GITEE_OWNER:-race-org}"
GITEE_REPO="${GITEE_REPO:-triton-anchor-local-ci-results}"
GITEE_BRANCHES="${GITEE_BRANCHES:-}"
GITEE_POLL_ALL_BRANCHES="${GITEE_POLL_ALL_BRANCHES:-1}"
GITEE_BRANCH_INCLUDE_REGEX="${GITEE_BRANCH_INCLUDE_REGEX:-^ci/(pr-[0-9]+/.+|push/.+|full/.+)$}"
GITEE_TOKEN="${GITEE_TOKEN:-}"
LOCAL_CI_POLL_INTERVAL="${LOCAL_CI_POLL_INTERVAL:-60}"
LOCAL_CI_ONCE="${LOCAL_CI_ONCE:-0}"
LOCAL_CI_HEALTH_ENABLED="${LOCAL_CI_HEALTH_ENABLED:-1}"
LOCAL_CI_HEALTH_DIR="${LOCAL_CI_HEALTH_DIR:-${LOCAL_CI_STATE_DIR%/}/health}"
LOCAL_CI_WORKER_ID="${LOCAL_CI_WORKER_ID:-local-ci-worker}"
LOCAL_CI_HEARTBEAT_INTERVAL_SECONDS="${LOCAL_CI_HEARTBEAT_INTERVAL_SECONDS:-60}"
LOCAL_CI_MAINTENANCE_ENABLED="${LOCAL_CI_MAINTENANCE_ENABLED:-0}"
LOCAL_CI_MAINTENANCE_INTERVAL_SECONDS="${LOCAL_CI_MAINTENANCE_INTERVAL_SECONDS:-86400}"
LOCAL_CI_SUCCESS_RETENTION_DAYS="${LOCAL_CI_SUCCESS_RETENTION_DAYS:-14}"
LOCAL_CI_FAILURE_RETENTION_DAYS="${LOCAL_CI_FAILURE_RETENTION_DAYS:-28}"
LOCAL_CI_INCOMPLETE_RETENTION_DAYS="${LOCAL_CI_INCOMPLETE_RETENTION_DAYS:-7}"
LOCAL_CI_DOCKER_ORPHAN_GRACE_HOURS="${LOCAL_CI_DOCKER_ORPHAN_GRACE_HOURS:-72}"
GITEE_RESULT_CONTEXT="${GITEE_RESULT_CONTEXT:-local-ci/sophgo-cmodel}"
GITEE_RESULTS_BRANCH="${GITEE_RESULTS_BRANCH:-local-ci-results}"
GITEE_RESULTS_OWNER="${GITEE_RESULTS_OWNER:-${GITEE_OWNER}}"
GITEE_RESULTS_REPO="${GITEE_RESULTS_REPO:-${GITEE_REPO}}"
GITEE_RESULTS_REPO_URL="${GITEE_RESULTS_REPO_URL:-${GITEE_REPO_URL}}"
PUBLISH_GITEE_RESULTS="${PUBLISH_GITEE_RESULTS:-1}"
GITEE_USERNAME="${GITEE_USERNAME:-likehupochuan}"
GITEE_WEB_URL="${GITEE_WEB_URL:-https://gitee.com/${GITEE_OWNER}/${GITEE_REPO}}"
GITEE_RESULTS_WEB_URL="${GITEE_RESULTS_WEB_URL:-https://gitee.com/${GITEE_RESULTS_OWNER}/${GITEE_RESULTS_REPO}}"
LOCAL_CI_CONTAINER="${LOCAL_CI_CONTAINER:-anchor-sophgo-ci-prod}"
LOCAL_CI_WORKSPACE_HOST="${LOCAL_CI_WORKSPACE_HOST:-/home/localci/local_ci/workspace}"
LOCAL_CI_ARTIFACT_HOST_ROOTS="${LOCAL_CI_ARTIFACT_HOST_ROOTS:-${LOCAL_CI_WORKSPACE_HOST%/}/local-ci-artifacts}"
BACKEND_PROFILE="${BACKEND_PROFILE:-sophgo-cmodel}"
LOCAL_CI_PROFILE_DIR="${LOCAL_CI_PROFILE_DIR:-}"
LOCAL_CI_PROFILE_NAME="${LOCAL_CI_PROFILE_NAME:-${BACKEND_PROFILE}}"
LOCAL_CI_LLVM_HASH="${LOCAL_CI_LLVM_HASH:-}"
LOCAL_CI_PROFILE_FILE="${LOCAL_CI_PROFILE_FILE:-}"
RUN_BACKEND_STAGES="${RUN_BACKEND_STAGES:-true}"
BACKEND_SKIP_REASON="${BACKEND_SKIP_REASON:-}"
FRONTEND_ONLY_BACKEND_SKIP_REASON="当前没有部署可供测试的厂商后端，未执行后端构建、JIT、FlagGems 和性能验证。"
RUN_COMPILE_BENCHMARK="${RUN_COMPILE_BENCHMARK:-true}"
RUN_PASS_PROFILE="${RUN_PASS_PROFILE:-true}"
RUN_IR_SERIALIZATION_BENCHMARK="${RUN_IR_SERIALIZATION_BENCHMARK:-true}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_AI_CI_HOME="${CODEX_AI_CI_HOME:-}"
RUN_CODEX_AI_CI="${RUN_CODEX_AI_CI:-false}"
CODEX_AI_CI_BRANCH_REGEX="${CODEX_AI_CI_BRANCH_REGEX:-^ci/(push/.+|pr-[0-9]+/.+)$}"
CODEX_AI_CI_WORKSPACE_ROOT="${CODEX_AI_CI_WORKSPACE_ROOT:-${LOCAL_CI_STATE_DIR%/}/codex-workspaces}"
CODEX_AI_CI_TIMEOUT_SECONDS="${CODEX_AI_CI_TIMEOUT_SECONDS:-3600}"
CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS="${CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS:-1500}"
CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS="${CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS:-900}"
CODEX_AI_CI_REASONING_EFFORT="${CODEX_AI_CI_REASONING_EFFORT:-medium}"
CODEX_AI_CI_MIN_GENERATED_TEST_CASES="${CODEX_AI_CI_MIN_GENERATED_TEST_CASES:-1}"
CODEX_AI_CI_MAX_GENERATED_TEST_CASES="${CODEX_AI_CI_MAX_GENERATED_TEST_CASES:-15}"
CODEX_AI_CI_MAX_GENERATED_TEST_FILES="${CODEX_AI_CI_MAX_GENERATED_TEST_FILES:-5}"
CODEX_AI_CI_MAX_TEST_COMMANDS="${CODEX_AI_CI_MAX_TEST_COMMANDS:-50}"
CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS="${CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS:-900}"
CODEX_AI_CI_TEST_BUDGET_SECONDS="${CODEX_AI_CI_TEST_BUDGET_SECONDS:-2700}"
CODEX_AI_CI_REPORT_RESERVE_SECONDS="${CODEX_AI_CI_REPORT_RESERVE_SECONDS:-450}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export GITEE_TOKEN GITEE_USERNAME GITEE_WEB_URL GITEE_RESULTS_WEB_URL WORKSPACE LOCAL_CI_WORKSPACE_HOST LOCAL_CI_CONFIG LOCAL_CI_CONTAINER

mkdir -p "${LOCAL_CI_STATE_DIR}"
export GIT_TERMINAL_PROMPT=0
if [[ -n "${GITEE_TOKEN}" ]]; then
  gitee_askpass="${LOCAL_CI_STATE_DIR}/gitee-askpass.sh"
  write_gitee_askpass "${gitee_askpass}"
  export GIT_ASKPASS="${gitee_askpass}"
fi

lock_file="${LOCAL_CI_STATE_DIR}/poll.lock"

case "${GITEE_POLL_ALL_BRANCHES}" in
  0|1) ;;
  *)
    echo "GITEE_POLL_ALL_BRANCHES must be 0 or 1" >&2
    exit 1
    ;;
esac
case "${LOCAL_CI_MAINTENANCE_ENABLED}" in
  0|1) ;;
  *)
    echo "LOCAL_CI_MAINTENANCE_ENABLED must be 0 or 1" >&2
    exit 1
    ;;
esac
for positive_name in \
  LOCAL_CI_MAINTENANCE_INTERVAL_SECONDS LOCAL_CI_SUCCESS_RETENTION_DAYS \
  LOCAL_CI_FAILURE_RETENTION_DAYS LOCAL_CI_INCOMPLETE_RETENTION_DAYS \
  LOCAL_CI_DOCKER_ORPHAN_GRACE_HOURS; do
  if [[ ! "${!positive_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${positive_name} must be a positive integer" >&2
    exit 1
  fi
done
if [[ "${GITEE_POLL_ALL_BRANCHES}" == "0" && -z "${GITEE_BRANCHES//[[:space:],]/}" ]]; then
  echo "GITEE_BRANCHES is required when GITEE_POLL_ALL_BRANCHES=0" >&2
  exit 1
fi

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "Another local-ci poller is already running: ${lock_file}" >&2
  exit 1
fi

if [[ "${1:-}" == "--once" ]]; then
  LOCAL_CI_ONCE="1"
fi

HEALTH_TOOL="${LOCAL_CI_ROOT}/maintenance/local_ci_health.py"
HEALTH_HEARTBEAT_PID=""

health_call() {
  if [[ "${LOCAL_CI_HEALTH_ENABLED}" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "${HEALTH_TOOL}" ]]; then
    echo "Warning: Local CI health tool is unavailable: ${HEALTH_TOOL}" >&2
    return 0
  fi
  if ! "${PYTHON_BIN}" "${HEALTH_TOOL}" "$@"; then
    echo "Warning: Local CI health update failed: $*" >&2
  fi
  return 0
}

stop_health_heartbeat() {
  if [[ -n "${HEALTH_HEARTBEAT_PID}" ]]; then
    kill "${HEALTH_HEARTBEAT_PID}" >/dev/null 2>&1 || true
    wait "${HEALTH_HEARTBEAT_PID}" >/dev/null 2>&1 || true
  fi
}

trap stop_health_heartbeat EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

health_call poller-start \
  --health-dir "${LOCAL_CI_HEALTH_DIR}" \
  --worker-id "${LOCAL_CI_WORKER_ID}" \
  --pid "$$"
if [[ "${LOCAL_CI_HEALTH_ENABLED}" == "1" && -f "${HEALTH_TOOL}" ]]; then
  "${PYTHON_BIN}" "${HEALTH_TOOL}" heartbeat \
    --health-dir "${LOCAL_CI_HEALTH_DIR}" \
    --worker-id "${LOCAL_CI_WORKER_ID}" \
    --parent-pid "$$" \
    --interval "${LOCAL_CI_HEARTBEAT_INTERVAL_SECONDS}" &
  HEALTH_HEARTBEAT_PID="$!"
fi

latest_sha() {
  local branch="$1"
  git ls-remote "${GITEE_REPO_URL}" "refs/heads/${branch}" | awk '{print $1}'
}

list_branches() {
  if [[ "${GITEE_POLL_ALL_BRANCHES}" == "1" ]]; then
    git ls-remote --heads "${GITEE_REPO_URL}" |
      awk '{sub(/^refs\/heads\//, "", $2); print $2}'
    return 0
  fi

  printf '%s' "${GITEE_BRANCHES}" | tr -s ',[:space:]' '\n' | awk 'NF'
}

branch_is_enabled() {
  local branch="$1"
  if [[ -z "${branch}" ]]; then
    return 1
  fi
  case "${branch}" in
    ci/meta/*) return 1 ;;
  esac
  if [[ "${branch}" == "${GITEE_RESULTS_BRANCH}" ]]; then
    return 1
  fi
  if [[ -n "${GITEE_BRANCH_INCLUDE_REGEX}" && ! "${branch}" =~ ${GITEE_BRANCH_INCLUDE_REGEX} ]]; then
    return 1
  fi
  return 0
}

metadata_ref_for_task() {
  local branch="$1"
  if [[ "${branch}" =~ ^ci/pr-([0-9]+)/(.+)$ ]]; then
    printf 'ci/meta/pr-%s/%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return 0
  fi
  return 1
}

read_verified_llvm_hash() {
  local branch="$1"
  local expected_sha="$2"
  local checkout_dir=""
  local actual_sha=""
  local llvm_hash=""

  if [[ ! "${expected_sha}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Cannot read LLVM hash without an exact task commit." >&2
    return 1
  fi
  checkout_dir="$(mktemp -d "${LOCAL_CI_STATE_DIR}/llvm-hash.XXXXXX")"
  git -C "${checkout_dir}" init -q
  git -C "${checkout_dir}" remote add origin "${GITEE_REPO_URL}"
  if ! git -C "${checkout_dir}" fetch -q --depth=1 origin \
    "+refs/heads/${branch}:refs/local-ci/llvm-hash"; then
    rm -rf -- "${checkout_dir}"
    echo "Unable to fetch ${branch} while resolving the Local CI profile." >&2
    return 1
  fi
  actual_sha="$(git -C "${checkout_dir}" rev-parse refs/local-ci/llvm-hash)"
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    rm -rf -- "${checkout_dir}"
    echo "Task ref ${branch} moved while resolving the Local CI profile." >&2
    return 1
  fi
  if ! git -C "${checkout_dir}" cat-file -e \
    "${expected_sha}:triton/cmake/llvm-hash.txt" 2>/dev/null; then
    rm -rf -- "${checkout_dir}"
    echo "Commit ${expected_sha} has no triton/cmake/llvm-hash.txt." >&2
    return 1
  fi
  if ! llvm_hash="$(
    git -C "${checkout_dir}" show \
      "${expected_sha}:triton/cmake/llvm-hash.txt"
  )"; then
    rm -rf -- "${checkout_dir}"
    echo "Unable to read the LLVM hash from commit ${expected_sha}." >&2
    return 1
  fi
  rm -rf -- "${checkout_dir}"
  if [[ ! "${llvm_hash}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Commit ${expected_sha} contains an invalid LLVM hash." >&2
    return 1
  fi
  printf '%s' "${llvm_hash}"
}

select_task_profile() {
  local task_branch="$1"
  local task_sha="$2"
  local base_branch="$3"
  local base_sha="$4"
  local selected_hash=""
  local candidate_hash=""
  local resolution=""
  local configured_profile_dir="${LOCAL_CI_PROFILE_DIR}"

  PROFILE_SELECTION_ERROR=""
  if [[ "${task_branch}" =~ ^ci/pr-[0-9]+/.+$ ]]; then
    if ! selected_hash="$(read_verified_llvm_hash "${base_branch}" "${base_sha}")"; then
      PROFILE_SELECTION_ERROR="Unable to read the trusted PR base LLVM hash."
      return 1
    fi
    if ! candidate_hash="$(read_verified_llvm_hash "${task_branch}" "${task_sha}")"; then
      PROFILE_SELECTION_ERROR="Unable to read the tested PR LLVM hash."
      return 1
    fi
    if [[ "${candidate_hash}" != "${selected_hash}" ]]; then
      PROFILE_SELECTION_ERROR="The tested PR changes triton/cmake/llvm-hash.txt relative to its trusted base."
      echo "${PROFILE_SELECTION_ERROR}" >&2
      return 1
    fi
  else
    if ! selected_hash="$(read_verified_llvm_hash "${task_branch}" "${task_sha}")"; then
      PROFILE_SELECTION_ERROR="Unable to read the tested commit LLVM hash."
      return 1
    fi
  fi
  LOCAL_CI_LLVM_HASH="${selected_hash}"

  if ! resolution="$(
    "${PYTHON_BIN:-python3}" \
      "${LOCAL_CI_RUNNER_DIR}/shared/resolve_ci_profile.py" \
      --profile-dir "${configured_profile_dir}" \
      --llvm-hash "${selected_hash}"
  )"; then
    PROFILE_SELECTION_ERROR="No trusted Local CI profile is available for LLVM hash ${selected_hash}."
    return 1
  fi

  LOCAL_CI_PROFILE_FILE="${resolution}"
  LOCAL_CI_PROFILE_NAME=""
  LOCAL_CI_CONTAINER=""
  LOCAL_CI_WORKSPACE_HOST=""
  LLVM_BUILD_DIR=""
  PYTHON_VENV_ACTIVATE=""
  BACKEND_PROFILE=""
  RUN_BACKEND_STAGES=""
  BACKEND_SKIP_REASON=""
  EXPECTED_TRITON_BACKEND=""
  BACKEND_PATH=""
  BACKEND_ENVSETUP=""
  BACKEND_ENVSETUP_ARGS=""
  BACKEND_TEST_COMMAND=""
  BACKEND_UNINSTALL_PACKAGES=""
  BACKEND_WHEEL_PATTERN=""
  RUN_FLAGGEMS_TESTS=""
  RUN_COMPILE_BENCHMARK=""
  RUN_PASS_PROFILE=""
  RUN_IR_SERIALIZATION_BENCHMARK=""
  # Profile files live outside the task checkout and are controlled by the server.
  # shellcheck disable=SC1090
  source "${LOCAL_CI_PROFILE_FILE}"
  LOCAL_CI_PROFILE_DIR="${configured_profile_dir}"
  LOCAL_CI_PROFILE_FILE="${resolution}"
  LOCAL_CI_LLVM_HASH="${selected_hash}"

  if [[ ! "${LOCAL_CI_PROFILE_NAME:-}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    PROFILE_SELECTION_ERROR="The selected Local CI profile has an invalid LOCAL_CI_PROFILE_NAME."
    echo "${PROFILE_SELECTION_ERROR}" >&2
    return 1
  fi
  if [[ ! "${LOCAL_CI_CONTAINER:-}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    PROFILE_SELECTION_ERROR="The selected Local CI profile has an invalid LOCAL_CI_CONTAINER."
    echo "${PROFILE_SELECTION_ERROR}" >&2
    return 1
  fi
  if [[ "${LOCAL_CI_WORKSPACE_HOST:-}" != /* ]]; then
    PROFILE_SELECTION_ERROR="The selected Local CI profile must use an absolute LOCAL_CI_WORKSPACE_HOST."
    echo "${PROFILE_SELECTION_ERROR}" >&2
    return 1
  fi
  if [[ "${LLVM_BUILD_DIR:-}" != /* ]]; then
    PROFILE_SELECTION_ERROR="The selected Local CI profile must use an absolute LLVM_BUILD_DIR."
    echo "${PROFILE_SELECTION_ERROR}" >&2
    return 1
  fi
  if [[ "${PYTHON_VENV_ACTIVATE:-}" != /* ]]; then
    PROFILE_SELECTION_ERROR="The selected Local CI profile must use an absolute PYTHON_VENV_ACTIVATE."
    echo "${PROFILE_SELECTION_ERROR}" >&2
    return 1
  fi
  for profile_switch in \
    RUN_FLAGGEMS_TESTS RUN_COMPILE_BENCHMARK RUN_PASS_PROFILE \
    RUN_IR_SERIALIZATION_BENCHMARK; do
    if [[ "${!profile_switch:-}" != "true" && "${!profile_switch:-}" != "false" ]]; then
      PROFILE_SELECTION_ERROR="The selected Local CI profile must set ${profile_switch} to true or false."
      echo "${PROFILE_SELECTION_ERROR}" >&2
      return 1
    fi
  done
  case "${RUN_BACKEND_STAGES:-}" in
    true)
      BACKEND_SKIP_REASON=""
      for required_backend_value in \
        BACKEND_PROFILE EXPECTED_TRITON_BACKEND BACKEND_PATH BACKEND_TEST_COMMAND \
        BACKEND_WHEEL_PATTERN; do
        if [[ -z "${!required_backend_value:-}" ]]; then
          PROFILE_SELECTION_ERROR="A backend-enabled Local CI profile must provide ${required_backend_value}."
          echo "${PROFILE_SELECTION_ERROR}" >&2
          return 1
        fi
      done
      if [[ "${BACKEND_PATH}" != /* ]]; then
        PROFILE_SELECTION_ERROR="A backend-enabled Local CI profile must use an absolute BACKEND_PATH."
        echo "${PROFILE_SELECTION_ERROR}" >&2
        return 1
      fi
      ;;
    false)
      BACKEND_SKIP_REASON="${FRONTEND_ONLY_BACKEND_SKIP_REASON}"
      RUN_FLAGGEMS_TESTS="false"
      RUN_COMPILE_BENCHMARK="false"
      RUN_PASS_PROFILE="false"
      RUN_IR_SERIALIZATION_BENCHMARK="false"
      INSTALL_FLAGGEMS_PACKAGES="0"
      EXPECTED_TRITON_BACKEND=""
      BACKEND_PATH=""
      BACKEND_ENVSETUP=""
      BACKEND_ENVSETUP_ARGS=""
      BACKEND_TEST_COMMAND=""
      BACKEND_UNINSTALL_PACKAGES=""
      BACKEND_WHEEL_PATTERN=""
      ;;
    *)
      PROFILE_SELECTION_ERROR="The selected Local CI profile must set RUN_BACKEND_STAGES to true or false."
      echo "${PROFILE_SELECTION_ERROR}" >&2
      return 1
      ;;
  esac

  export LOCAL_CI_PROFILE_FILE LOCAL_CI_PROFILE_NAME LOCAL_CI_LLVM_HASH
  export RUN_BACKEND_STAGES BACKEND_SKIP_REASON LOCAL_CI_CONTAINER LLVM_BUILD_DIR
  export LOCAL_CI_WORKSPACE_HOST PYTHON_VENV_ACTIVATE
  echo "Selected Local CI profile ${LOCAL_CI_PROFILE_NAME} for LLVM ${LOCAL_CI_LLVM_HASH}."
}

flaggems_mode_for_branch() {
  local branch="$1"
  case "${branch}" in
    ci/full/*) printf 'full' ;;
    *) printf '%s' "${FLAGGEMS_TEST_MODE:-sample}" ;;
  esac
}

run_codex_ai_ci_for_run() {
  local sha="$1"
  local run_dir="$2"
  local base_sha="$3"
  local base_ref="$4"
  local branch="$5"
  local local_ci_status="$6"
  local task_metadata_file="$7"
  local head_sha="$8"
  local head_ref="$9"
  local run_id
  run_id="$(basename "${run_dir}")"

  CODEX_BIN="${CODEX_BIN}" \
    CODEX_AI_CI_HOME="${CODEX_AI_CI_HOME}" \
    CODEX_AI_CI_TIMEOUT_SECONDS="${CODEX_AI_CI_TIMEOUT_SECONDS}" \
    CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS="${CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS}" \
    CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS="${CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS}" \
    CODEX_AI_CI_REASONING_EFFORT="${CODEX_AI_CI_REASONING_EFFORT}" \
    CODEX_AI_CI_WORKSPACE_ROOT="${CODEX_AI_CI_WORKSPACE_ROOT}" \
    CODEX_AI_CI_MIN_GENERATED_TEST_CASES="${CODEX_AI_CI_MIN_GENERATED_TEST_CASES}" \
    CODEX_AI_CI_MAX_GENERATED_TEST_CASES="${CODEX_AI_CI_MAX_GENERATED_TEST_CASES}" \
    CODEX_AI_CI_MAX_GENERATED_TEST_FILES="${CODEX_AI_CI_MAX_GENERATED_TEST_FILES}" \
    CODEX_AI_CI_MAX_TEST_COMMANDS="${CODEX_AI_CI_MAX_TEST_COMMANDS}" \
    CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS="${CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS}" \
    CODEX_AI_CI_TEST_BUDGET_SECONDS="${CODEX_AI_CI_TEST_BUDGET_SECONDS}" \
    CODEX_AI_CI_REPORT_RESERVE_SECONDS="${CODEX_AI_CI_REPORT_RESERVE_SECONDS}" \
    LOCAL_CI_CONTAINER="${LOCAL_CI_CONTAINER}" \
    LOCAL_CI_ARTIFACT_ROOT="${LOCAL_CI_ARTIFACT_ROOT:-/workspace/local-ci-artifacts}" \
    LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-}" \
    PYTHON_VENV_ACTIVATE="${PYTHON_VENV_ACTIVATE:-}" \
    TRUSTED_ANCHOR_ENVSETUP="${LOCAL_CI_RUNNER_DIR}/trusted/envsetup.sh" \
    CODEX_TEST_PYTHON_BIN="${PYTHON_BIN:-python3}" \
    PPL_ROOT="${PPL_ROOT:-}" \
    PACKAGE_TOOL="${PACKAGE_TOOL:-auto}" \
    FRONTEND_BUILD_MODE="${FRONTEND_BUILD_MODE:-}" \
    BACKEND_PROFILE="${BACKEND_PROFILE:-}" \
    EXPECTED_TRITON_BACKEND="${EXPECTED_TRITON_BACKEND:-}" \
    FLAGGEMS_CLONE_DIR="${FLAGGEMS_CLONE_DIR:-}" \
    MAX_JOBS="${MAX_JOBS:-1}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}" \
    NINJAFLAGS="${NINJAFLAGS:--j1}" \
    UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
    LOCAL_CI_RUN_ID="${run_id}" \
    SOURCE_ENVSETUP="${SOURCE_ENVSETUP:-1}" \
    ANCHOR_DIR="${ANCHOR_DIR:-}" \
    BACKEND_PATH="${BACKEND_PATH:-}" \
    BACKEND_ENVSETUP="${BACKEND_ENVSETUP:-}" \
    BACKEND_ENVSETUP_ARGS="${BACKEND_ENVSETUP_ARGS:-}" \
    BACKEND_UNINSTALL_PACKAGES="${BACKEND_UNINSTALL_PACKAGES:-}" \
    BACKEND_WHEEL_PATTERN="${BACKEND_WHEEL_PATTERN:-}" \
    LOCAL_CI_PROFILE_NAME="${LOCAL_CI_PROFILE_NAME}" \
    LOCAL_CI_LLVM_HASH="${LOCAL_CI_LLVM_HASH}" \
    RUN_BACKEND_STAGES="${RUN_BACKEND_STAGES}" \
    BACKEND_SKIP_REASON="${BACKEND_SKIP_REASON}" \
    LOCAL_CI_EXECUTION_MODE="${LOCAL_CI_EXECUTION_MODE:-full}" \
    bash "${LOCAL_CI_RUNNER_DIR}/codex_ai/run_codex_ai_ci.sh" \
    "${GITEE_REPO_URL}" "${run_dir}" "${sha}" "${base_sha}" "${base_ref}" \
    "${branch}" "${local_ci_status}" "${task_metadata_file}" "${head_sha}" "${head_ref}"
}

publish_result() {
  local sha="$1"
  local status="$2"
  local run_id="$3"
  local run_dir="$4"
  local branch="$5"
  local head_sha="${6:-}"
  if [[ "${PUBLISH_GITEE_RESULTS}" != "1" ]]; then
    echo "PUBLISH_GITEE_RESULTS is not 1; skip publishing Gitee result branch and commit comment."
    return 0
  fi
  local args=(
    --owner "${GITEE_OWNER}"
    --repo "${GITEE_REPO}"
    --repo-url "${GITEE_REPO_URL}"
    --results-owner "${GITEE_RESULTS_OWNER}"
    --results-repo "${GITEE_RESULTS_REPO}"
    --results-repo-url "${GITEE_RESULTS_REPO_URL}"
    --results-web-url "${GITEE_RESULTS_WEB_URL}"
    --sha "${sha}"
    --source-branch "${branch}"
    --run-id "${run_id}"
    --run-dir "${run_dir}"
    --exit-code "${status}"
    --results-branch "${GITEE_RESULTS_BRANCH}"
    --context "${GITEE_RESULT_CONTEXT}"
    --execution-mode "${LOCAL_CI_EXECUTION_MODE:-full}"
    --ci-profile "${LOCAL_CI_PROFILE_NAME:-unavailable}"
    --llvm-hash "${LOCAL_CI_LLVM_HASH:-unavailable}"
    --backend-stages-enabled "${RUN_BACKEND_STAGES:-false}"
    --backend-skip-reason "${BACKEND_SKIP_REASON:-}"
  )
  if [[ -n "${head_sha}" ]]; then
    args+=(--head-sha "${head_sha}")
  fi
  "${PYTHON_BIN:-python3}" \
    "${LOCAL_CI_RUNNER_DIR}/results/publish_gitee_result.py" "${args[@]}"
}

stage_runner_scripts() {
  local run_id="$1"
  local required_path
  if [[ ! -d "${LOCAL_CI_SCRIPT_DIR}" ]]; then
    echo "LOCAL_CI_SCRIPT_DIR does not exist: ${LOCAL_CI_SCRIPT_DIR}" >&2
    return 1
  fi
  for required_path in \
    poll_gitee_and_run.sh \
    orchestration/run_deterministic_ci_in_container.sh \
    orchestration/fetch_task_metadata.sh \
    deterministic_ci/run_deterministic_ci.sh \
    deterministic_ci/flaggems/batch_test_flaggems.py \
    deterministic_ci/flaggems/select_flaggems_tests.py \
    deterministic_ci/flaggems/flaggems_all_ops.tsv \
    deterministic_ci/flaggems/flaggems_pass_whitelist.tsv \
    deterministic_ci/performance/compile_benchmark.py \
    deterministic_ci/performance/common.py \
    deterministic_ci/performance/compare_compile_time.py \
    deterministic_ci/performance/pass_profile_benchmark.py \
    deterministic_ci/performance/compare_pass_profile.py \
    deterministic_ci/performance/ir_serialization_benchmark.py \
    deterministic_ci/performance/compare_ir_serialization.py \
    codex_ai/run_codex_ai_ci.sh \
    codex_ai/classify_codex_review_context.py \
    codex_ai/prepare_codex_checkout.sh \
    codex_ai/setup_codex_ai_container.sh \
    codex_ai/validate_codex_ai_credentials.py \
    codex_ai/build_codex_ai_report.py \
    codex_ai/codex_jsonl_evidence.py \
    codex_ai/render_codex_ai_report.py \
    codex_ai/codex_ai_analysis.schema.json \
    codex_ai/codex_ai_report.schema.json \
    codex_ai/prompts/codex_ai_success.md \
    codex_ai/prompts/codex_ai_failure.md \
    results/publish_gitee_result.py \
    results/bridge_gitee_to_github_status.py \
    shared/finding_locations.py \
    shared/dump_artifacts.py \
    shared/task_tmp.py \
    shared/result_paths.py \
    shared/path_utils.sh \
    shared/resolve_ci_profile.py \
    maintenance/local_ci_health.py \
    shared/validate_task_metadata.py; do
    if [[ ! -f "${LOCAL_CI_SCRIPT_DIR}/${required_path}" ]]; then
      echo "LOCAL_CI_SCRIPT_DIR is not a complete Local CI root; missing ${required_path}" >&2
      return 1
    fi
  done

  local staged_dir="${LOCAL_CI_STATE_DIR}/runner/${run_id}"
  rm -rf "${staged_dir}"
  mkdir -p "${staged_dir}"
  cp -a "${LOCAL_CI_SCRIPT_DIR}/." "${staged_dir}/"
  find "${staged_dir}" -type d -name __pycache__ -prune -exec rm -rf -- {} +
  find "${staged_dir}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
  printf '%s' "${staged_dir}"
}

prepare_trusted_envsetup() {
  local runner_dir="$1"
  local task_branch="$2"
  local base_branch="$3"
  local base_sha="$4"
  local checkout_dir=""
  local trusted_file="${runner_dir}/trusted/envsetup.sh"

  rm -f -- "${trusted_file}"
  if [[ ! "${task_branch}" =~ ^ci/pr-[0-9]+/.+$ ]]; then
    return 0
  fi
  if [[ -z "${base_branch}" || ! "${base_sha}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "PR task has no verified base revision for trusted envsetup.sh." >&2
    return 1
  fi

  checkout_dir="$(mktemp -d "${LOCAL_CI_STATE_DIR}/trusted-base.XXXXXX")"
  git -C "${checkout_dir}" init -q
  git -C "${checkout_dir}" remote add origin "${GITEE_REPO_URL}"
  if ! git -C "${checkout_dir}" fetch -q --depth=1 origin \
    "+refs/heads/${base_branch}:refs/local-ci/trusted-base"; then
    rm -rf -- "${checkout_dir}"
    echo "Unable to fetch trusted PR base ref ${base_branch}." >&2
    return 1
  fi
  if [[ "$(git -C "${checkout_dir}" rev-parse refs/local-ci/trusted-base)" != "${base_sha}" ]]; then
    rm -rf -- "${checkout_dir}"
    echo "Trusted PR base ref moved while preparing envsetup.sh." >&2
    return 1
  fi
  if ! git -C "${checkout_dir}" cat-file -e "${base_sha}:envsetup.sh" 2>/dev/null; then
    rm -rf -- "${checkout_dir}"
    echo "Trusted base commit has no envsetup.sh." >&2
    return 1
  fi
  mkdir -p "$(dirname "${trusted_file}")"
  git -C "${checkout_dir}" show "${base_sha}:envsetup.sh" > "${trusted_file}"
  rm -rf -- "${checkout_dir}"
  chmod 600 "${trusted_file}"
  bash -n "${trusted_file}"
  echo "Prepared trusted envsetup.sh from base ${base_sha}."
}

cached_benchmark_exists() {
  local kind="$1"
  local sha="$2"
  local safe_profile
  safe_profile="$(safe_path_part "${BACKEND_PROFILE}")"
  local rel_path="${kind}/by-sha/${sha}/${safe_profile}/latest.json"
  local checkout_dir
  checkout_dir="$(mktemp -d "${LOCAL_CI_STATE_DIR}/baseline-check.XXXXXX")"
  local status=1

  git -C "${checkout_dir}" init -q
  git -C "${checkout_dir}" remote add origin "${GITEE_RESULTS_REPO_URL}"
  if git -C "${checkout_dir}" fetch -q --depth=1 origin \
    "refs/heads/${GITEE_RESULTS_BRANCH}:refs/remotes/origin/${GITEE_RESULTS_BRANCH}"; then
    if git -C "${checkout_dir}" cat-file -e "origin/${GITEE_RESULTS_BRANCH}:${rel_path}" 2>/dev/null; then
      status=0
    fi
  fi

  rm -rf "${checkout_dir}"
  return "${status}"
}

compile_baseline_exists() {
  cached_benchmark_exists "compile-time" "$1"
}

pass_profile_baseline_exists() {
  cached_benchmark_exists "pass-profile" "$1"
}

ir_serialization_baseline_exists() {
  cached_benchmark_exists "ir-serialization" "$1"
}

run_once() (
  local branch="$1"
  local sha
  sha="$(latest_sha "${branch}")"
  if [[ -z "${sha}" ]]; then
    echo "No commit found at ${GITEE_REPO_URL} refs/heads/${branch}" >&2
    return 1
  fi

  local safe_branch
  safe_branch="$(safe_path_part "${branch}")"
  local last_file="${LOCAL_CI_STATE_DIR}/last-processed-${safe_branch}.sha"
  local last=""
  if [[ -f "${last_file}" ]]; then
    last="$(<"${last_file}")"
  fi

  if [[ "${sha}" == "${last}" ]]; then
    echo "No new commit on ${branch}: ${sha}"
    return 0
  fi

  local run_id
  run_id="$(date -u +%Y%m%dT%H%M%SZ)-${sha:0:12}"
  local run_dir="${LOCAL_CI_STATE_DIR}/runs/${safe_branch}/${run_id}"
  mkdir -p "${run_dir}"

  local health_task_started=1
  local health_result_status="error"
  local health_result_exit_code=1
  local health_publish_status=-1
  local health_failure_code="task_interrupted"
  finish_health_task() {
    local shell_status="$?"
    trap - EXIT
    if [[ "${health_failure_code}" == "task_interrupted" ]]; then
      health_result_exit_code="${shell_status}"
    fi
    if [[ "${health_task_started}" == "1" ]]; then
      health_call task-finish \
        --health-dir "${LOCAL_CI_HEALTH_DIR}" \
        --worker-id "${LOCAL_CI_WORKER_ID}" \
        --branch "${branch}" \
        --sha "${sha}" \
        --run-id "${run_id}" \
        --profile "${LOCAL_CI_PROFILE_NAME:-unknown}" \
        --status "${health_result_status}" \
        --exit-code "${health_result_exit_code}" \
        --publish-status "${health_publish_status}" \
        --failure-code "${health_failure_code}"
    fi
    exit "${shell_status}"
  }
  trap finish_health_task EXIT
  health_call task-start \
    --health-dir "${LOCAL_CI_HEALTH_DIR}" \
    --worker-id "${LOCAL_CI_WORKER_ID}" \
    --branch "${branch}" \
    --sha "${sha}" \
    --run-id "${run_id}" \
    --profile "resolving" \
    --container "${LOCAL_CI_CONTAINER:-}" \
    --stage "preparing"

  echo "Detected new commit on ${branch}: ${sha}"
  echo "Run directory: ${run_dir}"

  LOCAL_CI_RUNNER_DIR="$(stage_runner_scripts "${run_id}")"
  export LOCAL_CI_RUNNER_DIR
  echo "Runner script snapshot: ${LOCAL_CI_RUNNER_DIR}"

  local base_branch=""
  local base_sha=""
  local head_branch=""
  local head_sha=""
  if [[ "${branch}" =~ ^ci/pr-([0-9]+)/(.+)$ ]]; then
    base_branch="ci/base/pr-${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    head_branch="ci/head/pr-${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    base_sha="$(latest_sha "${base_branch}")"
    head_sha="$(latest_sha "${head_branch}")"
    if [[ -z "${base_sha}" ]]; then
      echo "No exact PR base SHA found for ${branch}; refusing to run head-only Local CI." >&2
      return 1
    fi
    if [[ -z "${head_sha}" ]]; then
      echo "No exact PR head SHA found for ${branch}; refusing to run PR Local CI without base/head identity." >&2
      return 1
    fi
  fi

  local task_metadata_file=""
  local task_metadata_ref=""
  local execution_mode="full"
  if task_metadata_ref="$(metadata_ref_for_task "${branch}")"; then
    task_metadata_file="${run_dir}/task-metadata.json"
    local task_metadata_message=""
    if task_metadata_message="$(
      bash "${LOCAL_CI_RUNNER_DIR}/orchestration/fetch_task_metadata.sh" \
        "${GITEE_REPO_URL}" "${task_metadata_ref}" "${branch}" "${sha}" \
        "${task_metadata_file}" "${base_sha}" "${head_sha}" 2>&1
    )"; then
      echo "Fetched PR task metadata from ${task_metadata_ref}."
      if [[ -n "${task_metadata_message}" ]]; then
        printf '%s\n' "${task_metadata_message}" >&2
      fi
    else
      rm -f -- "${task_metadata_file}"
      task_metadata_file=""
      echo "Warning: PR task metadata is unavailable; Local CI and Codex AI will continue." >&2
      if [[ -n "${task_metadata_message}" ]]; then
        printf '%s\n' "${task_metadata_message}" >&2
      fi
    fi
  fi
  if [[ -n "${task_metadata_file}" ]]; then
    execution_mode="$(
      "${PYTHON_BIN:-python3}" -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("execution_mode", "full"))' \
        "${task_metadata_file}"
    )"
  fi
  echo "Local CI execution mode: ${execution_mode}"
  LOCAL_CI_EXECUTION_MODE="${execution_mode}"
  health_call task-stage \
    --health-dir "${LOCAL_CI_HEALTH_DIR}" \
    --run-id "${run_id}" \
    --stage "preparing" \
    --execution-mode "${execution_mode}"

  local PROFILE_SELECTION_ERROR=""
  local profile_selection_status=0
  if select_task_profile "${branch}" "${sha}" "${base_branch}" "${base_sha}"; then
    :
  else
    profile_selection_status=$?
    echo "Local CI profile selection failed: ${PROFILE_SELECTION_ERROR}" >&2
  fi
  health_call task-stage \
    --health-dir "${LOCAL_CI_HEALTH_DIR}" \
    --run-id "${run_id}" \
    --stage "profile-selected" \
    --profile "${LOCAL_CI_PROFILE_NAME:-unavailable}" \
    --container "${LOCAL_CI_CONTAINER:-}"

  local flaggems_test_mode
  flaggems_test_mode="$(flaggems_mode_for_branch "${branch}")"
  echo "FlagGems test mode: ${flaggems_test_mode}"

  if [[ ${profile_selection_status} -eq 0 \
    && "${branch}" =~ ^ci/pr-[0-9]+/.+$ \
    && "${execution_mode}" != "codex_only" ]]; then
    if [[ -z "${base_sha}" ]]; then
      echo "Skipping PR performance baseline because exact base SHA is unavailable." >&2
    else
      local missing_baseline=0
      if [[ "${RUN_COMPILE_BENCHMARK}" == "true" ]]; then
        if compile_baseline_exists "${base_sha}"; then
          echo "Using cached compile-time baseline for ${base_sha}."
        else
          echo "Compile-time baseline missing for ${base_sha}."
          missing_baseline=1
        fi
      fi
      if [[ "${RUN_PASS_PROFILE}" == "true" ]]; then
        if pass_profile_baseline_exists "${base_sha}"; then
          echo "Using cached pass-profile baseline for ${base_sha}."
        else
          echo "Pass-profile baseline missing for ${base_sha}."
          missing_baseline=1
        fi
      fi
      if [[ "${RUN_IR_SERIALIZATION_BENCHMARK}" == "true" ]]; then
        if ir_serialization_baseline_exists "${base_sha}"; then
          echo "Using cached IR serialization baseline for ${base_sha}."
        else
          echo "IR serialization baseline missing for ${base_sha}."
          missing_baseline=1
        fi
      fi
      if [[ "${missing_baseline}" == "1" ]]; then
        local base_run_id
        base_run_id="$(date -u +%Y%m%dT%H%M%SZ)-${base_sha:0:12}-base"
        local base_run_dir="${LOCAL_CI_STATE_DIR}/runs/$(safe_path_part "${base_branch}")/${base_run_id}"
        mkdir -p "${base_run_dir}"
        echo "Running base task once to populate missing performance baseline(s) for ${base_sha}."
        health_call task-stage \
          --health-dir "${LOCAL_CI_HEALTH_DIR}" \
          --run-id "${run_id}" \
          --stage "performance-baseline"

        local base_status=0
        set +e
        LOCAL_CI_BASE_SHA="" LOCAL_CI_BASE_REF="" GITEE_BRANCH="${base_branch}" \
          LOCAL_CI_RUN_ID="${base_run_id}" FLAGGEMS_TEST_MODE="${flaggems_test_mode}" \
          RUN_FLAGGEMS_TESTS=false \
          bash "${LOCAL_CI_RUNNER_DIR}/orchestration/run_deterministic_ci_in_container.sh" \
            "${base_sha}" "${base_branch}" 2>&1 |
          tee "${base_run_dir}/local-ci.log"
        base_status=${PIPESTATUS[0]}
        set -e
        echo "{\"sha\":\"${base_sha}\",\"status\":${base_status},\"run_dir\":\"${base_run_dir}\"}" \
          > "${base_run_dir}/result.json"
        publish_result "${base_sha}" "${base_status}" "${base_run_id}" "${base_run_dir}" "${base_branch}" || true
        if [[ ${base_status} -ne 0 ]]; then
          echo "Base task failed; continuing candidate task with a missing-baseline warning." >&2
        fi
      fi
    fi
  fi

  local status=0
  local nonexecuted_artifact_dir=""
  if [[ ${profile_selection_status} -ne 0 ]]; then
    status=1
    health_result_status="error"
    health_failure_code="profile_selection_failed"
    LOCAL_CI_PROFILE_NAME="unavailable"
    RUN_BACKEND_STAGES="false"
    BACKEND_SKIP_REASON="${PROFILE_SELECTION_ERROR}"
    nonexecuted_artifact_dir="$(
      mktemp -d "/tmp/triton-anchor-profile-error.${sha:0:12}.XXXXXX"
    )"
    cat > "${nonexecuted_artifact_dir}/delivery-summary.txt" <<EOF
schema: triton-anchor-local-ci/v3
status: 1
target_sha: ${sha}
tested_sha: ${sha}
tested_sha_kind: $([[ "${branch}" =~ ^ci/pr-[0-9]+/.+$ ]] && printf 'pr_merge' || printf 'commit')
actual_checkout_sha: not_run
branch: ${branch}
run_id: ${run_id}
execution_mode: ${execution_mode}
ci_profile: unavailable
llvm_hash: ${LOCAL_CI_LLVM_HASH:-unavailable}
backend_stages_enabled: false
backend_skip_reason: ${PROFILE_SELECTION_ERROR}
artifact_dir: ${nonexecuted_artifact_dir}
frontend_build_status: skipped
frontend_smoke_status: skipped
backend_rebuild_status: skipped
backend_smoke_jit_status: skipped
flaggems_status: skipped
compile_time_status: skipped
pass_profile_status: skipped
ir_serialization_status: skipped
EOF
    {
      echo "Local CI did not start because no trusted execution profile was selected."
      echo "${PROFILE_SELECTION_ERROR}"
      echo "Artifact dir: ${nonexecuted_artifact_dir}"
    } | tee "${run_dir}/local-ci.log"
  elif [[ "${execution_mode}" == "codex_only" ]]; then
    prepare_trusted_envsetup "${LOCAL_CI_RUNNER_DIR}" "${branch}" \
      "${base_branch}" "${base_sha}"
    nonexecuted_artifact_dir="$(
      mktemp -d "/tmp/triton-anchor-docs-only.${sha:0:12}.XXXXXX"
    )"
    cat > "${nonexecuted_artifact_dir}/delivery-summary.txt" <<EOF
schema: triton-anchor-local-ci/v3
status: 0
target_sha: ${sha}
tested_sha: ${sha}
tested_sha_kind: pr_merge
actual_checkout_sha: not_run
branch: ${branch}
run_id: ${run_id}
execution_mode: codex_only
ci_profile: ${LOCAL_CI_PROFILE_NAME}
llvm_hash: ${LOCAL_CI_LLVM_HASH}
backend_stages_enabled: ${RUN_BACKEND_STAGES}
backend_skip_reason: ${BACKEND_SKIP_REASON}
artifact_dir: ${nonexecuted_artifact_dir}
frontend_build_status: skipped
frontend_smoke_status: skipped
backend_rebuild_status: skipped
backend_smoke_jit_status: skipped
flaggems_status: skipped
compile_time_status: skipped
pass_profile_status: skipped
ir_serialization_status: skipped
EOF
    echo "Skipping deterministic Local CI for documentation-only PR." |
      tee "${run_dir}/local-ci.log"
  else
    health_call task-stage \
      --health-dir "${LOCAL_CI_HEALTH_DIR}" \
      --run-id "${run_id}" \
      --stage "deterministic-ci"
    prepare_trusted_envsetup "${LOCAL_CI_RUNNER_DIR}" "${branch}" \
      "${base_branch}" "${base_sha}"
    set +e
    LOCAL_CI_BASE_SHA="${base_sha}" LOCAL_CI_BASE_REF="${base_branch}" GITEE_BRANCH="${branch}" \
      LOCAL_CI_RUN_ID="${run_id}" FLAGGEMS_TEST_MODE="${flaggems_test_mode}" \
      bash "${LOCAL_CI_RUNNER_DIR}/orchestration/run_deterministic_ci_in_container.sh" \
        "${sha}" "${branch}" 2>&1 |
      tee "${run_dir}/local-ci.log"
    status=${PIPESTATUS[0]}
    set -e
  fi
  health_result_exit_code="${status}"
  if [[ ${profile_selection_status} -eq 0 ]]; then
    if [[ ${status} -eq 0 ]]; then
      health_result_status="success"
      health_failure_code=""
    else
      health_result_status="failure"
      health_failure_code="deterministic_ci_failed"
    fi
  fi

  local codex_ai_base_sha=""
  local codex_ai_base_ref=""
  if [[ -n "${base_branch}" ]]; then
    codex_ai_base_sha="${base_sha}"
    codex_ai_base_ref="${base_branch}"
  elif [[ -n "${last}" ]]; then
    codex_ai_base_sha="${last}"
  fi

  local codex_ai_ci_status="skipped"
  local codex_ai_ci_verdict="NOT_RUN"
  local codex_ai_test_status="NOT_RUN"
  local codex_ai_failure_code=""
  local codex_ai_mode="not_run"
  if [[ ${profile_selection_status} -eq 0 \
    && ("${execution_mode}" == "codex_only" \
      || ("${RUN_CODEX_AI_CI}" == "true" \
        && (-z "${CODEX_AI_CI_BRANCH_REGEX}" || "${branch}" =~ ${CODEX_AI_CI_BRANCH_REGEX}))) ]]; then
    codex_ai_ci_verdict="UNKNOWN"
    codex_ai_mode="full"
    if [[ ${status} -ne 0 ]]; then
      codex_ai_mode="analysis_only"
    fi
    health_call task-stage \
      --health-dir "${LOCAL_CI_HEALTH_DIR}" \
      --run-id "${run_id}" \
      --stage "codex-ai"
    echo "Running non-blocking Codex AI CI for ${sha} (${codex_ai_mode})." |
      tee -a "${run_dir}/local-ci.log"
    local codex_ai_ci_exit=0
    set +e
    run_codex_ai_ci_for_run \
      "${sha}" "${run_dir}" "${codex_ai_base_sha}" "${codex_ai_base_ref}" "${branch}" "${status}" \
      "${task_metadata_file}" "${head_sha}" "${head_branch}" 2>&1 |
      tee -a "${run_dir}/local-ci.log"
    codex_ai_ci_exit=${PIPESTATUS[0]}
    set -e
    local codex_ai_summary="${run_dir}/codex-ai-ci-summary.txt"
    if [[ -f "${codex_ai_summary}" ]]; then
      local parsed_codex_ai_verdict
      parsed_codex_ai_verdict="$(awk -F ': ' '$1 == "report_verdict" { print $2; exit }' "${codex_ai_summary}")"
      case "${parsed_codex_ai_verdict}" in
        PASS | WARNING | FAIL) codex_ai_ci_verdict="${parsed_codex_ai_verdict}" ;;
      esac
      local parsed_codex_ai_test_status
      parsed_codex_ai_test_status="$(awk -F ': ' '$1 == "test_execution_status" { print $2; exit }' "${codex_ai_summary}")"
      case "${parsed_codex_ai_test_status}" in
        not_run | passed | stable_failure | flaky_failure | infrastructure_failure | test_generation_error | insufficient_evidence | unavailable)
          codex_ai_test_status="${parsed_codex_ai_test_status}"
          ;;
      esac
      codex_ai_failure_code="$(awk -F ': ' '$1 == "failure_code" { print $2; exit }' "${codex_ai_summary}")"
    fi
    if [[ ${codex_ai_ci_exit} -eq 0 ]]; then
      codex_ai_ci_status="pass"
    else
      codex_ai_ci_status="fail"
      echo "Codex AI CI failed but does not change the deterministic local-ci result." |
        tee -a "${run_dir}/local-ci.log"
    fi
  else
    echo "Codex AI CI skipped for ${branch}." | tee -a "${run_dir}/local-ci.log"
  fi

  "${PYTHON_BIN:-python3}" - "${run_dir}/result.json" "${sha}" "${status}" \
    "${codex_ai_ci_status}" "${codex_ai_mode}" "${codex_ai_ci_verdict}" \
    "${codex_ai_test_status}" "${codex_ai_failure_code}" "${run_dir}" "${base_sha}" "${head_sha}" \
    "${branch}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
(
    tested_sha,
    status,
    codex_ai_ci_status,
    codex_ai_ci_mode,
    codex_ai_ci_verdict,
    codex_ai_test_status,
    codex_ai_failure_code,
    run_dir,
    base_sha,
    head_sha,
    branch,
) = sys.argv[2:]
result = {
    "sha": tested_sha,
    "target_sha": tested_sha,
    "tested_sha": tested_sha,
    "tested_sha_kind": "pr_merge" if branch.startswith("ci/pr-") else "commit",
    "status": int(status),
    "codex_ai_ci_status": codex_ai_ci_status,
    "codex_ai_ci_mode": codex_ai_ci_mode,
    "codex_ai_ci_verdict": codex_ai_ci_verdict,
    "codex_ai_test_status": codex_ai_test_status,
    "codex_ai_failure_code": codex_ai_failure_code,
    "run_dir": run_dir,
}
if base_sha:
    result["base_sha"] = base_sha
if head_sha:
    result["head_sha"] = head_sha
output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
PY

  if [[ -n "${nonexecuted_artifact_dir}" ]]; then
    echo "Artifact dir: ${nonexecuted_artifact_dir}" >> "${run_dir}/local-ci.log"
  fi
  local publish_status=0
  health_call task-stage \
    --health-dir "${LOCAL_CI_HEALTH_DIR}" \
    --run-id "${run_id}" \
    --stage "publishing"
  set +e
  publish_result "${sha}" "${status}" "${run_id}" "${run_dir}" "${branch}" "${head_sha}"
  publish_status=$?
  set -e
  health_publish_status="${publish_status}"
  if [[ ${publish_status} -ne 0 ]]; then
    health_result_status="error"
    health_result_exit_code="${publish_status}"
    health_failure_code="result_publish_failed"
  elif [[ ${profile_selection_status} -eq 0 && ${status} -ne 0 ]]; then
    health_result_status="failure"
    health_result_exit_code="${status}"
    health_failure_code="deterministic_ci_failed"
  fi
  if [[ -n "${nonexecuted_artifact_dir}" ]]; then
    rm -rf -- "${nonexecuted_artifact_dir}"
  fi

  if [[ ${publish_status} -eq 0 ]]; then
    echo "${sha}" > "${last_file}"
    if [[ ${status} -eq 0 ]]; then
      echo "local-ci passed and result was published; marked ${sha} processed."
    else
      echo "local-ci failed and result was published; marked ${sha} processed."
    fi
  else
    echo "local-ci result publish failed; ${sha} was not marked processed and will be retried." >&2
  fi

  if [[ ${status} -ne 0 ]]; then
    return "${status}"
  fi
  return "${publish_status}"
)

run_all_once() {
  local status=0
  local branch_output=""
  local -a branches=()
  local branch

  health_call poller-update \
    --health-dir "${LOCAL_CI_HEALTH_DIR}" \
    --worker-id "${LOCAL_CI_WORKER_ID}" \
    --pid "$$" \
    --phase started
  if ! branch_output="$(list_branches)"; then
    health_call poller-update \
      --health-dir "${LOCAL_CI_HEALTH_DIR}" \
      --worker-id "${LOCAL_CI_WORKER_ID}" \
      --pid "$$" \
      --phase finished \
      --status error \
      --error-code "gitee_branch_discovery_failed"
    return 1
  fi
  while IFS= read -r branch; do
    if ! branch_is_enabled "${branch}"; then
      continue
    fi
    branches+=("${branch}")
  done < <(printf '%s\n' "${branch_output}" | awk 'NF' | sort -u)
  health_call poller-update \
    --health-dir "${LOCAL_CI_HEALTH_DIR}" \
    --worker-id "${LOCAL_CI_WORKER_ID}" \
    --pid "$$" \
    --phase started \
    --task-ref-count "${#branches[@]}"
  for branch in "${branches[@]}"; do
    run_once "${branch}" || status=1
  done
  health_call poller-update \
    --health-dir "${LOCAL_CI_HEALTH_DIR}" \
    --worker-id "${LOCAL_CI_WORKER_ID}" \
    --pid "$$" \
    --phase finished \
    --status success \
    --task-ref-count "${#branches[@]}"
  return "${status}"
}

run_maintenance_if_due() {
  if [[ "${LOCAL_CI_MAINTENANCE_ENABLED}" != "1" || "${LOCAL_CI_ONCE}" == "1" ]]; then
    return 0
  fi
  local maintenance_script="${LOCAL_CI_SCRIPT_DIR}/maintenance/manage_local_ci_state.py"
  local report="${LOCAL_CI_STATE_DIR}/maintenance/latest.json"
  if [[ ! -f "${maintenance_script}" ]]; then
    echo "Local CI maintenance script is missing: ${maintenance_script}" >&2
    return 1
  fi
  if [[ -f "${report}" ]]; then
    local last_run now
    last_run="$(stat -c %Y "${report}")"
    now="$(date +%s)"
    if ((now - last_run < LOCAL_CI_MAINTENANCE_INTERVAL_SECONDS)); then
      return 0
    fi
  fi

  local args=(
    --apply
    --state-dir "${LOCAL_CI_STATE_DIR}"
    --success-days "${LOCAL_CI_SUCCESS_RETENTION_DAYS}"
    --failure-days "${LOCAL_CI_FAILURE_RETENTION_DAYS}"
    --incomplete-days "${LOCAL_CI_INCOMPLETE_RETENTION_DAYS}"
    --docker-orphan-grace-hours "${LOCAL_CI_DOCKER_ORPHAN_GRACE_HOURS}"
    --report "${report}"
  )
  local artifact_root
  while IFS= read -r artifact_root; do
    if [[ -n "${artifact_root}" ]]; then
      args+=(--artifact-root "${artifact_root}")
    fi
  done < <(printf '%s' "${LOCAL_CI_ARTIFACT_HOST_ROOTS}" | tr ':' '\n')
  echo "Running Local CI retention maintenance."
  "${PYTHON_BIN:-python3}" "${maintenance_script}" "${args[@]}"
}

while true; do
  loop_status=0
  run_all_once || loop_status=$?
  if [[ "${LOCAL_CI_ONCE}" == "1" ]]; then
    exit "${loop_status}"
  fi
  run_maintenance_if_due || echo "Local CI maintenance failed; polling will continue." >&2
  sleep "${LOCAL_CI_POLL_INTERVAL}"
done
