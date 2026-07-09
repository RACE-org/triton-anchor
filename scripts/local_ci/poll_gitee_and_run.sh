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
GITEE_STATUS_CONTEXT="${GITEE_STATUS_CONTEXT:-local-ci/sophgo-cmodel}"
export GITEE_TOKEN

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

post_status() {
  local sha="$1"
  local state="$2"
  local description="$3"
  local target_url="${4:-}"
  local args=(
    --owner "${GITEE_OWNER}"
    --repo "${GITEE_REPO}"
    --sha "${sha}"
    --state "${state}"
    --context "${GITEE_STATUS_CONTEXT}"
    --description "${description}"
  )

  if [[ -n "${target_url}" ]]; then
    args+=(--target-url "${target_url}")
  fi

  "${SCRIPT_DIR}/post_gitee_status.py" "${args[@]}"
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
  post_status "${sha}" "pending" "local-ci started" || true

  local status=0
  set +e
  "${SCRIPT_DIR}/run_in_container.sh" "${sha}" 2>&1 | tee "${run_dir}/local-ci.log"
  status=${PIPESTATUS[0]}
  set -e

  echo "${sha}" > "${last_file}"

  if [[ ${status} -eq 0 ]]; then
    post_status "${sha}" "success" "local-ci passed" || true
  else
    post_status "${sha}" "failure" "local-ci failed with exit code ${status}" || true
  fi

  echo "{\"sha\":\"${sha}\",\"status\":${status},\"run_dir\":\"${run_dir}\"}" > "${run_dir}/result.json"
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
