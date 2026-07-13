#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE="${LOCAL_CI_CONFIG:-${SCRIPT_DIR}/config.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

GITEE_REPO_URL="${GITEE_REPO_URL:-https://gitee.com/likehupochuan/triton-anchor.git}"
GITEE_OWNER="${GITEE_OWNER:-likehupochuan}"
GITEE_REPO="${GITEE_REPO:-triton-anchor}"
GITEE_BRANCH="${GITEE_BRANCH:-jiwang-delivery-ci}"
GITEE_BRANCHES="${GITEE_BRANCHES:-${GITEE_BRANCH}}"
GITEE_POLL_ALL_BRANCHES="${GITEE_POLL_ALL_BRANCHES:-0}"
GITEE_BRANCH_INCLUDE_REGEX="${GITEE_BRANCH_INCLUDE_REGEX:-}"
GITEE_TOKEN="${GITEE_TOKEN:-}"
LOCAL_CI_STATE_DIR="${LOCAL_CI_STATE_DIR:-/root/projects/test/local-ci-state}"
LOCAL_CI_POLL_INTERVAL="${LOCAL_CI_POLL_INTERVAL:-60}"
LOCAL_CI_ONCE="${LOCAL_CI_ONCE:-0}"
GITEE_RESULT_CONTEXT="${GITEE_RESULT_CONTEXT:-local-ci/sophgo-cmodel}"
GITEE_RESULTS_BRANCH="${GITEE_RESULTS_BRANCH:-local-ci-results}"
GITEE_RESULTS_OWNER="${GITEE_RESULTS_OWNER:-${GITEE_OWNER}}"
GITEE_RESULTS_REPO="${GITEE_RESULTS_REPO:-${GITEE_REPO}}"
GITEE_RESULTS_REPO_URL="${GITEE_RESULTS_REPO_URL:-${GITEE_REPO_URL}}"
PUBLISH_GITEE_RESULTS="${PUBLISH_GITEE_RESULTS:-1}"
GITEE_USERNAME="${GITEE_USERNAME:-${GITEE_OWNER}}"
GITEE_WEB_URL="${GITEE_WEB_URL:-https://gitee.com/${GITEE_OWNER}/${GITEE_REPO}}"
GITEE_RESULTS_WEB_URL="${GITEE_RESULTS_WEB_URL:-https://gitee.com/${GITEE_RESULTS_OWNER}/${GITEE_RESULTS_REPO}}"
LOCAL_CI_WORKSPACE_HOST="${LOCAL_CI_WORKSPACE_HOST:-/root/projects/test/workspace}"
export GITEE_TOKEN GITEE_USERNAME GITEE_WEB_URL GITEE_RESULTS_WEB_URL WORKSPACE LOCAL_CI_WORKSPACE_HOST

mkdir -p "${LOCAL_CI_STATE_DIR}"
lock_file="${LOCAL_CI_STATE_DIR}/poll.lock"

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "Another local-ci poller is already running: ${lock_file}" >&2
  exit 1
fi

latest_sha() {
  local branch="$1"
  git ls-remote "${GITEE_REPO_URL}" "refs/heads/${branch}" | awk '{print $1}'
}

safe_path_part() {
  local value="$1"
  value="${value//\//_}"
  value="$(printf '%s' "${value}" | tr -c 'A-Za-z0-9._-' '_')"
  value="${value##_}"
  value="${value%%_}"
  printf '%s' "${value:-default}"
}

list_branches() {
  if [[ "${GITEE_POLL_ALL_BRANCHES}" == "1" ]]; then
    git ls-remote --heads "${GITEE_REPO_URL}" |
      awk '{sub(/^refs\/heads\//, "", $2); print $2}'
    return 0
  fi

  printf '%s\n' ${GITEE_BRANCHES}
}

branch_is_enabled() {
  local branch="$1"
  if [[ -z "${branch}" ]]; then
    return 1
  fi
  if [[ "${branch}" == "${GITEE_RESULTS_BRANCH}" ]]; then
    return 1
  fi
  if [[ -n "${GITEE_BRANCH_INCLUDE_REGEX}" && ! "${branch}" =~ ${GITEE_BRANCH_INCLUDE_REGEX} ]]; then
    return 1
  fi
  return 0
}

publish_result() {
  local sha="$1"
  local status="$2"
  local run_id="$3"
  local run_dir="$4"
  local branch="$5"
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
  )
  "${SCRIPT_DIR}/publish_gitee_result.py" "${args[@]}"
}

run_once() {
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

  echo "Detected new commit on ${branch}: ${sha}"
  echo "Run directory: ${run_dir}"

  local status=0
  set +e
  GITEE_BRANCH="${branch}" "${SCRIPT_DIR}/run_in_container.sh" "${sha}" 2>&1 | tee "${run_dir}/local-ci.log"
  status=${PIPESTATUS[0]}
  set -e

  if [[ ${status} -eq 0 ]]; then
    echo "${sha}" > "${last_file}"
  else
    echo "local-ci failed; ${sha} was not marked processed and will be retried." >&2
  fi

  echo "{\"sha\":\"${sha}\",\"status\":${status},\"run_dir\":\"${run_dir}\"}" > "${run_dir}/result.json"
  publish_result "${sha}" "${status}" "${run_id}" "${run_dir}" "${branch}" || true
  return "${status}"
}

run_all_once() {
  local status=0
  local branch
  while IFS= read -r branch; do
    if ! branch_is_enabled "${branch}"; then
      continue
    fi
    run_once "${branch}" || status=1
  done < <(list_branches | awk 'NF' | sort -u)
  return "${status}"
}

if [[ "${1:-}" == "--once" ]]; then
  LOCAL_CI_ONCE="1"
fi

while true; do
  run_all_once || true
  if [[ "${LOCAL_CI_ONCE}" == "1" ]]; then
    break
  fi
  sleep "${LOCAL_CI_POLL_INTERVAL}"
done
