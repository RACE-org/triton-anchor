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
GITEE_TOKEN="${GITEE_TOKEN:-}"
LOCAL_CI_STATE_DIR="${LOCAL_CI_STATE_DIR:-/root/projects/test/local-ci-state}"
LOCAL_CI_POLL_INTERVAL="${LOCAL_CI_POLL_INTERVAL:-60}"
LOCAL_CI_ONCE="${LOCAL_CI_ONCE:-0}"
GITEE_RESULT_CONTEXT="${GITEE_RESULT_CONTEXT:-local-ci/sophgo-cmodel}"
GITEE_RESULTS_BRANCH="${GITEE_RESULTS_BRANCH:-local-ci-results}"
PUBLISH_GITEE_RESULTS="${PUBLISH_GITEE_RESULTS:-1}"
GITEE_USERNAME="${GITEE_USERNAME:-${GITEE_OWNER}}"
GITEE_WEB_URL="${GITEE_WEB_URL:-https://gitee.com/${GITEE_OWNER}/${GITEE_REPO}}"
LOCAL_CI_WORKSPACE_HOST="${LOCAL_CI_WORKSPACE_HOST:-/root/projects/test/workspace}"
export GITEE_TOKEN GITEE_USERNAME GITEE_WEB_URL WORKSPACE LOCAL_CI_WORKSPACE_HOST

mkdir -p "${LOCAL_CI_STATE_DIR}"
lock_file="${LOCAL_CI_STATE_DIR}/poll.lock"
last_file="${LOCAL_CI_STATE_DIR}/last-processed-${GITEE_BRANCH}.sha"

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "Another local-ci poller is already running: ${lock_file}" >&2
  exit 1
fi

latest_sha() {
  git ls-remote "${GITEE_REPO_URL}" "refs/heads/${GITEE_BRANCH}" | awk '{print $1}'
}

publish_result() {
  local sha="$1"
  local status="$2"
  local run_id="$3"
  local run_dir="$4"
  if [[ "${PUBLISH_GITEE_RESULTS}" != "1" ]]; then
    echo "PUBLISH_GITEE_RESULTS is not 1; skip publishing Gitee result branch and commit comment."
    return 0
  fi
  local args=(
    --owner "${GITEE_OWNER}"
    --repo "${GITEE_REPO}"
    --repo-url "${GITEE_REPO_URL}"
    --sha "${sha}"
    --source-branch "${GITEE_BRANCH}"
    --run-id "${run_id}"
    --run-dir "${run_dir}"
    --exit-code "${status}"
    --results-branch "${GITEE_RESULTS_BRANCH}"
    --context "${GITEE_RESULT_CONTEXT}"
  )
  "${SCRIPT_DIR}/publish_gitee_result.py" "${args[@]}"
}

run_once() {
  local sha
  sha="$(latest_sha)"
  if [[ -z "${sha}" ]]; then
    echo "No commit found at ${GITEE_REPO_URL} refs/heads/${GITEE_BRANCH}" >&2
    return 1
  fi

  local last=""
  if [[ -f "${last_file}" ]]; then
    last="$(<"${last_file}")"
  fi

  if [[ "${sha}" == "${last}" ]]; then
    echo "No new commit on ${GITEE_BRANCH}: ${sha}"
    return 0
  fi

  local run_id
  run_id="$(date -u +%Y%m%dT%H%M%SZ)-${sha:0:12}"
  local run_dir="${LOCAL_CI_STATE_DIR}/runs/${run_id}"
  mkdir -p "${run_dir}"

  echo "Detected new commit on ${GITEE_BRANCH}: ${sha}"
  echo "Run directory: ${run_dir}"

  local status=0
  set +e
  "${SCRIPT_DIR}/run_in_container.sh" "${sha}" 2>&1 | tee "${run_dir}/local-ci.log"
  status=${PIPESTATUS[0]}
  set -e

  if [[ ${status} -eq 0 ]]; then
    echo "${sha}" > "${last_file}"
  else
    echo "local-ci failed; ${sha} was not marked processed and will be retried." >&2
  fi

  echo "{\"sha\":\"${sha}\",\"status\":${status},\"run_dir\":\"${run_dir}\"}" > "${run_dir}/result.json"
  publish_result "${sha}" "${status}" "${run_id}" "${run_dir}" || true
  return "${status}"
}

if [[ "${1:-}" == "--once" ]]; then
  LOCAL_CI_ONCE="1"
fi

while true; do
  run_once || true
  if [[ "${LOCAL_CI_ONCE}" == "1" ]]; then
    break
  fi
  sleep "${LOCAL_CI_POLL_INTERVAL}"
done
