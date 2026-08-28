#!/usr/bin/env bash
set -uo pipefail

usage="run_codex_ai_ci.sh <repo-url> <output-dir> <target-sha> <base-sha> <base-ref> <branch> [local-ci-status] [task-metadata-file] [head-sha] [head-ref]"
repo_url="${1:?usage: ${usage}}"
output_dir="${2:?usage: ${usage}}"
target_sha="${3:?usage: ${usage}}"
requested_base_sha="${4:-}"
requested_base_ref="${5:-}"
branch="${6:?usage: ${usage}}"
local_ci_status="${7:-0}"
task_metadata_file="${8:-}"
requested_head_sha="${9:-}"
requested_head_ref="${10:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_AI_CI_HOME="${CODEX_AI_CI_HOME:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CODEX_AI_CI_TIMEOUT_SECONDS="${CODEX_AI_CI_TIMEOUT_SECONDS:-3600}"
CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS="${CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS:-1500}"
CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS="${CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS:-900}"
CODEX_AI_CI_REASONING_EFFORT="${CODEX_AI_CI_REASONING_EFFORT:-medium}"
CODEX_AI_CI_WORKSPACE_ROOT="${CODEX_AI_CI_WORKSPACE_ROOT:-${TMPDIR:-/tmp}/triton-anchor-codex-ai}"
CODEX_AI_CI_MIN_GENERATED_TEST_CASES="${CODEX_AI_CI_MIN_GENERATED_TEST_CASES:-1}"
CODEX_AI_CI_MAX_GENERATED_TEST_CASES="${CODEX_AI_CI_MAX_GENERATED_TEST_CASES:-15}"
CODEX_AI_CI_MAX_GENERATED_TEST_FILES="${CODEX_AI_CI_MAX_GENERATED_TEST_FILES:-5}"
CODEX_AI_CI_MAX_TEST_COMMANDS="${CODEX_AI_CI_MAX_TEST_COMMANDS:-50}"
CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS="${CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS:-900}"
CODEX_AI_CI_TEST_BUDGET_SECONDS="${CODEX_AI_CI_TEST_BUDGET_SECONDS:-2700}"
CODEX_AI_CI_REPORT_RESERVE_SECONDS="${CODEX_AI_CI_REPORT_RESERVE_SECONDS:-450}"
CODEX_AI_CI_CPUS="${CODEX_AI_CI_CPUS:-12}"
CODEX_AI_CI_MEMORY="${CODEX_AI_CI_MEMORY:-48g}"
CODEX_AI_CI_MEMORY_SWAP="${CODEX_AI_CI_MEMORY_SWAP:-48g}"
CODEX_AI_CI_PIDS_LIMIT="${CODEX_AI_CI_PIDS_LIMIT:-4096}"
LOCAL_CI_CONTAINER="${LOCAL_CI_CONTAINER:-anchor-sophgo-ci-prod}"
LOCAL_CI_ARTIFACT_ROOT="${LOCAL_CI_ARTIFACT_ROOT:-/workspace/local-ci-artifacts}"
LOCAL_CI_PROFILE_NAME="${LOCAL_CI_PROFILE_NAME:-legacy}"
LOCAL_CI_LLVM_HASH="${LOCAL_CI_LLVM_HASH:-unknown}"
LOCAL_CI_EXECUTION_MODE="${LOCAL_CI_EXECUTION_MODE:-full}"
RUN_BACKEND_STAGES="${RUN_BACKEND_STAGES:-true}"
BACKEND_SKIP_REASON="${BACKEND_SKIP_REASON:-}"
FRONTEND_ONLY_BACKEND_SKIP_REASON="当前没有部署可供测试的厂商后端，未执行后端构建、JIT、FlagGems 和性能验证。"
backend_validation_scope="full"
if [[ "${RUN_BACKEND_STAGES}" == "false" ]]; then
  backend_validation_scope="frontend_only"
  BACKEND_SKIP_REASON="${FRONTEND_ONLY_BACKEND_SKIP_REASON}"
fi
PYTHON_VENV_ACTIVATE="${PYTHON_VENV_ACTIVATE:-/opt/venv/bin/activate}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-}"
SOURCE_ENVSETUP="${SOURCE_ENVSETUP:-1}"
TRUSTED_ANCHOR_ENVSETUP="${TRUSTED_ANCHOR_ENVSETUP:-${LOCAL_CI_ROOT}/trusted/envsetup.sh}"
CODEX_TEST_PYTHON_BIN="${CODEX_TEST_PYTHON_BIN:-python3}"
PPL_ROOT="${PPL_ROOT:-}"
PACKAGE_TOOL="${PACKAGE_TOOL:-auto}"
FRONTEND_BUILD_MODE="${FRONTEND_BUILD_MODE:-}"
BACKEND_PROFILE="${BACKEND_PROFILE:-}"
EXPECTED_TRITON_BACKEND="${EXPECTED_TRITON_BACKEND:-}"
FLAGGEMS_CLONE_DIR="${FLAGGEMS_CLONE_DIR:-}"
MAX_JOBS="${MAX_JOBS:-1}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
NINJAFLAGS="${NINJAFLAGS:--j1}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
LOCAL_CI_RUN_ID="${LOCAL_CI_RUN_ID:-}"
ANCHOR_DIR="${ANCHOR_DIR:-/workspace/triton-anchor}"
if [[ "${RUN_BACKEND_STAGES}" == "false" ]]; then
  BACKEND_PATH=""
  BACKEND_ENVSETUP=""
  BACKEND_ENVSETUP_ARGS=""
else
  BACKEND_PATH="${BACKEND_PATH:-}"
  BACKEND_ENVSETUP="${BACKEND_ENVSETUP:-}"
  BACKEND_ENVSETUP_ARGS="${BACKEND_ENVSETUP_ARGS:-}"
fi

container_codex_bin="/usr/local/bin/codex"
container_codex_home="/root/.codex"
container_workspace_root="/codex-workspace"
container_checkout_dir="${container_workspace_root}/checkout"
container_input_dir="${container_workspace_root}/input"
container_analysis_json_path="${container_workspace_root}/codex-ai-analysis.json"
container_schema_path="${container_workspace_root}/codex-ai-analysis.schema.json"
container_jsonl_recorder_path="${container_workspace_root}/codex-jsonl-evidence.py"
container_local_ci_log="${container_input_dir}/local-ci.log"
container_changed_files_manifest="${container_input_dir}/codex-changed-files-manifest.json"
container_trusted_envsetup="${container_input_dir}/trusted-envsetup.sh"
container_anchor_envsetup=""
if [[ "${branch}" =~ ^ci/pr-[0-9]+/.+ && "${SOURCE_ENVSETUP}" == "1" ]]; then
  container_anchor_envsetup="${container_trusted_envsetup}"
fi

log_path="${output_dir}/codex-ai-ci.log"
codex_jsonl_path="${output_dir}/codex-ai-events.jsonl"
analysis_json_path="${output_dir}/codex-ai-analysis.json"
report_json_path="${output_dir}/codex-ai-report.json"
report_path="${output_dir}/codex-ai-report.md"
comment_path="${output_dir}/codex-ai-comment.md"
summary_path="${output_dir}/codex-ai-ci-summary.txt"
workspace_status_path="${output_dir}/codex-workspace-status.txt"
workspace_patch_path="${output_dir}/codex-workspace.patch"
generated_files_path="${output_dir}/codex-generated-files.tar.gz"
command_ledger_path="${output_dir}/codex-command-ledger.json"
schema_path="${SCRIPT_DIR}/codex_ai_analysis.schema.json"
renderer_path="${SCRIPT_DIR}/render_codex_ai_report.py"
report_builder_path="${SCRIPT_DIR}/build_codex_ai_report.py"
jsonl_evidence_path="${SCRIPT_DIR}/codex_jsonl_evidence.py"
review_context_classifier="${SCRIPT_DIR}/classify_codex_review_context.py"
checkout_helper="${SCRIPT_DIR}/prepare_codex_checkout.sh"
credentials_validator="${SCRIPT_DIR}/validate_codex_ai_credentials.py"
task_metadata_validator="${LOCAL_CI_ROOT}/shared/validate_task_metadata.py"
task_metadata_output_path="${output_dir}/task-metadata.json"
prompt_dir="${SCRIPT_DIR}/prompts"
success_prompt_template="${prompt_dir}/codex_ai_success.md"
failure_prompt_template="${prompt_dir}/codex_ai_failure.md"
changed_files_manifest_path="${output_dir}/codex-changed-files-manifest.json"
analysis_manifest_path="${output_dir}/codex-analysis-files-manifest.json"

status="fail"
exit_code=1
actual_sha=""
base_sha=""
base_source=""
diff_mode="unresolved"
diff_command=""
diff_revisions=()
changed_file_count=0
changed_files_manifest_json="[]"
changed_files_manifest_available="false"
failure_reason=""
marker_found="false"
report_format_valid="false"
report_verdict="UNKNOWN"
test_execution_status="UNKNOWN"
generated_test_file_count="UNKNOWN"
test_command_count="UNKNOWN"
max_test_command_duration_seconds="UNKNOWN"
total_test_command_duration_seconds="UNKNOWN"
test_generation_expected="false"
constraint_status="warning"
constraint_reason="尚未获得可校验的测试执行信息。"
turn_completed="false"
startup_progress="false"
startup_timed_out="false"
prepare_timed_out="false"
prepare_timeout_phase=""
prepare_started_seconds=0
prepare_deadline_seconds=0
prepare_duration_seconds="UNKNOWN"
snapshot_duration_seconds="UNKNOWN"
container_start_duration_seconds="UNKNOWN"
input_setup_duration_seconds="UNKNOWN"
command_executed="false"
command_ledger_available="false"
generated_archive_available="false"
workspace_dirty="false"
workspace_dir=""
workspace_parent=""
artifact_dir=""
host_codex_bin=""
credential_integrity_status="not_checked"
credential_integrity_reason="尚未校验独立凭据文件。"
credential_hashes_initialized="false"
config_sha256_before=""
auth_sha256_before=""
change_request_context_status="not_applicable"
change_request_context_reason="当前任务不是 PR，功能声明上下文不适用。"
change_request_context_diagnostic=""
change_request_context_pr_number=""
change_request_context_json=""
ephemeral_container=""
ephemeral_image=""
analysis_mode="full"
review_context_profile="unclassified"
review_context_hint="尚未根据变更文件生成审查上下文策略。"
changed_file_groups_json="{}"
failure_code=""
start_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
start_seconds="${SECONDS}"
if [[ "${local_ci_status}" != "0" ]]; then
  analysis_mode="analysis_only"
  constraint_reason="Local CI 失败诊断尚未完成轻量约束校验。"
fi
if [[ "${branch}" =~ ^ci/pr-[0-9]+/.+ ]]; then
  change_request_context_status="not_checked"
  change_request_context_reason="尚未校验 PR 功能声明元数据。"
fi

cleanup() {
  if [[ -n "${ephemeral_container}" ]]; then
    case "${ephemeral_container}" in
      anchor-codex-ai-*)
        docker rm -f "${ephemeral_container}" >/dev/null 2>&1 || true
        ;;
    esac
  fi
  if [[ -n "${ephemeral_image}" ]]; then
    case "${ephemeral_image}" in
      triton-anchor-codex-ai-snapshot:*)
        docker image rm -f "${ephemeral_image}" >/dev/null 2>&1 || true
        ;;
    esac
  fi
  if [[ -n "${workspace_parent}" && -d "${workspace_parent}" ]]; then
    rm -rf -- "${workspace_parent}"
  fi
}
trap cleanup EXIT

write_summary() {
  local duration_seconds="$((SECONDS - start_seconds))"
  {
    echo "schema: triton-anchor-codex-ai-ci/v3"
    echo "status: ${status}"
    echo "exit_code: ${exit_code}"
    echo "target_sha: ${target_sha}"
    echo "tested_sha: ${target_sha}"
    echo "actual_sha: ${actual_sha}"
    echo "requested_base_sha: ${requested_base_sha}"
    echo "requested_base_ref: ${requested_base_ref}"
    echo "requested_head_sha: ${requested_head_sha}"
    echo "requested_head_ref: ${requested_head_ref}"
    echo "base_sha: ${base_sha}"
    echo "base_source: ${base_source}"
    echo "diff_mode: ${diff_mode}"
    echo "branch: ${branch}"
    echo "repo_source: gitee"
    echo "local_ci_status: ${local_ci_status}"
    echo "analysis_mode: ${analysis_mode}"
    echo "execution_mode: ephemeral_container"
    echo "local_ci_execution_mode: ${LOCAL_CI_EXECUTION_MODE}"
    echo "ci_profile: ${LOCAL_CI_PROFILE_NAME}"
    echo "llvm_hash: ${LOCAL_CI_LLVM_HASH}"
    echo "backend_stages_enabled: ${RUN_BACKEND_STAGES}"
    echo "backend_skip_reason: ${BACKEND_SKIP_REASON}"
    echo "source_container: ${LOCAL_CI_CONTAINER}"
    echo "ephemeral_container: ${ephemeral_container}"
    echo "snapshot_image: ${ephemeral_image}"
    echo "workspace_dir: ${workspace_dir}"
    echo "container_workspace_dir: ${container_checkout_dir}"
    echo "artifact_dir: ${artifact_dir}"
    echo "output_dir: ${output_dir}"
    echo "changed_file_count: ${changed_file_count}"
    echo "changed_files_manifest_available: ${changed_files_manifest_available}"
    echo "review_context_profile: ${review_context_profile}"
    echo "review_context_hint: ${review_context_hint}"
    echo "changed_file_groups_json: ${changed_file_groups_json}"
    echo "started_at: ${start_time}"
    echo "duration_seconds: ${duration_seconds}"
    echo "timeout_seconds: ${CODEX_AI_CI_TIMEOUT_SECONDS}"
    echo "prepare_timeout_seconds: ${CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS}"
    echo "prepare_timed_out: ${prepare_timed_out}"
    echo "prepare_timeout_phase: ${prepare_timeout_phase}"
    echo "prepare_duration_seconds: ${prepare_duration_seconds}"
    echo "snapshot_duration_seconds: ${snapshot_duration_seconds}"
    echo "container_start_duration_seconds: ${container_start_duration_seconds}"
    echo "input_setup_duration_seconds: ${input_setup_duration_seconds}"
    echo "reasoning_effort: ${CODEX_AI_CI_REASONING_EFFORT}"
    echo "startup_timeout_seconds: ${CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS}"
    echo "startup_progress: ${startup_progress}"
    echo "startup_timed_out: ${startup_timed_out}"
    echo "min_generated_test_cases: ${CODEX_AI_CI_MIN_GENERATED_TEST_CASES}"
    echo "max_generated_test_cases: ${CODEX_AI_CI_MAX_GENERATED_TEST_CASES}"
    echo "max_generated_test_files: ${CODEX_AI_CI_MAX_GENERATED_TEST_FILES}"
    echo "max_test_commands: ${CODEX_AI_CI_MAX_TEST_COMMANDS}"
    echo "recommended_command_timeout_seconds: ${CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS}"
    echo "container_cpus: ${CODEX_AI_CI_CPUS}"
    echo "container_memory: ${CODEX_AI_CI_MEMORY}"
    echo "container_memory_swap: ${CODEX_AI_CI_MEMORY_SWAP}"
    echo "container_pids_limit: ${CODEX_AI_CI_PIDS_LIMIT}"
    echo "test_budget_seconds: ${CODEX_AI_CI_TEST_BUDGET_SECONDS}"
    echo "report_reserve_seconds: ${CODEX_AI_CI_REPORT_RESERVE_SECONDS}"
    echo "marker_found: ${marker_found}"
    echo "report_format_valid: ${report_format_valid}"
    echo "report_verdict: ${report_verdict}"
    echo "test_execution_status: ${test_execution_status}"
    echo "generated_test_file_count: ${generated_test_file_count}"
    echo "test_command_count: ${test_command_count}"
    echo "max_test_command_duration_seconds: ${max_test_command_duration_seconds}"
    echo "total_test_command_duration_seconds: ${total_test_command_duration_seconds}"
    echo "test_generation_expected: ${test_generation_expected}"
    echo "constraint_status: ${constraint_status}"
    echo "constraint_reason: ${constraint_reason}"
    echo "turn_completed: ${turn_completed}"
    echo "command_executed: ${command_executed}"
    echo "command_ledger_available: ${command_ledger_available}"
    echo "generated_archive_available: ${generated_archive_available}"
    echo "workspace_dirty: ${workspace_dirty}"
    echo "credential_integrity_status: ${credential_integrity_status}"
    echo "credential_integrity_reason: ${credential_integrity_reason}"
    echo "change_request_context_status: ${change_request_context_status}"
    echo "change_request_context_reason: ${change_request_context_reason}"
    echo "change_request_context_diagnostic: ${change_request_context_diagnostic}"
    echo "change_request_context_pr_number: ${change_request_context_pr_number}"
    echo "failure_code: ${failure_code}"
    echo "failure_reason: ${failure_reason}"
  } > "${summary_path}"
}

load_execution_metadata() {
  local execution_metadata
  local constraint_fields=()
  execution_metadata="$(
    "${PYTHON_BIN}" -c '
import json
import sys

execution = json.load(open(sys.argv[1], encoding="utf-8"))["test_execution"]
max_files = int(sys.argv[2])
max_commands = int(sys.argv[3])
recommended_timeout = int(sys.argv[4])
test_budget = int(sys.argv[5])
generated_files = execution["generated_test_files"]
commands = execution["commands"]
durations = [float(command["duration_seconds"]) for command in commands]
max_duration = max(durations, default=0.0)
total_duration = sum(durations)
reasons = []

if len(generated_files) > max_files:
    reasons.append(
        f"生成测试文件数量 {len(generated_files)} 超过限制 {max_files}"
    )
if len(commands) > max_commands:
    reasons.append(
        f"测试、构建、lint 或诊断命令数量 {len(commands)} 超过限制 {max_commands}"
    )
if max_duration > recommended_timeout:
    reasons.append(
        f"单条命令最长耗时 {max_duration:g} 秒超过建议上限 "
        f"{recommended_timeout} 秒"
    )
if total_duration > test_budget:
    reasons.append(
        f"测试和诊断命令累计耗时 {total_duration:g} 秒超过建议预算 "
        f"{test_budget} 秒"
    )
constraint_status = "warning" if reasons else "pass"
constraint_reason = (
    "；".join(reasons)
    if reasons
    else "生成测试文件、执行测试或诊断命令的数量和耗时均在轻量约束范围内。"
)

print(
    execution["status"],
    len(generated_files),
    len(commands),
    f"{max_duration:g}",
    f"{total_duration:g}",
    constraint_status,
    constraint_reason,
    sep="\t",
)
' "${report_json_path}" \
      "${CODEX_AI_CI_MAX_GENERATED_TEST_FILES}" \
      "${CODEX_AI_CI_MAX_TEST_COMMANDS}" \
      "${CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS}" \
      "${CODEX_AI_CI_TEST_BUDGET_SECONDS}" 2>> "${log_path}"
  )" || return 1
  [[ -n "${execution_metadata}" ]] || return 1
  IFS=$'\t' read -r -a constraint_fields <<< "${execution_metadata}"
  [[ "${#constraint_fields[@]}" -eq 7 ]] || return 1
  test_execution_status="${constraint_fields[0]}"
  generated_test_file_count="${constraint_fields[1]}"
  test_command_count="${constraint_fields[2]}"
  max_test_command_duration_seconds="${constraint_fields[3]}"
  total_test_command_duration_seconds="${constraint_fields[4]}"
  constraint_status="${constraint_fields[5]}"
  constraint_reason="${constraint_fields[6]}"
}

refresh_command_ledger_state() {
  local ledger_metadata=""
  local ledger_count=""
  local ledger_max_duration=""
  local ledger_total_duration=""
  local ledger_failed_count=""
  command_executed="false"
  if [[ "${command_ledger_available}" != "true" || ! -r "${command_ledger_path}" ]]; then
    return 1
  fi
  if ! ledger_metadata="$(
    "${PYTHON_BIN}" - "${command_ledger_path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    document = json.load(stream)
if not isinstance(document, list):
    raise SystemExit("command ledger is not an array")
for index, item in enumerate(document):
    if not isinstance(item, dict):
        raise SystemExit(f"command ledger item {index} is not an object")
    command = item.get("command")
    exit_code = item.get("exit_code")
    duration = item.get("duration_seconds", 0.0)
    if not isinstance(command, str) or not command.strip():
        raise SystemExit(f"command ledger item {index} has no command")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise SystemExit(f"command ledger item {index} has no integer exit code")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or duration < 0
    ):
        raise SystemExit(f"command ledger item {index} has an invalid duration")
durations = [float(item.get("duration_seconds", 0.0)) for item in document]
print(
    f"{len(document)}\t"
    f"{max(durations, default=0.0):.3f}\t"
    f"{sum(durations):.3f}\t"
    f"{sum(item['exit_code'] != 0 for item in document)}"
)
PY
  )"; then
    echo "Codex 命令执行记录不可读取。" >> "${log_path}"
    command_ledger_available="false"
    return 1
  fi
  IFS=$'\t' read -r ledger_count ledger_max_duration ledger_total_duration \
    ledger_failed_count \
    <<< "${ledger_metadata}"
  test_command_count="${ledger_count}"
  max_test_command_duration_seconds="${ledger_max_duration}"
  total_test_command_duration_seconds="${ledger_total_duration}"
  if ((ledger_count > 0)); then
    command_executed="true"
    if ((ledger_failed_count > 0)); then
      test_execution_status="insufficient_evidence"
    else
      test_execution_status="passed"
    fi
  else
    test_execution_status="not_run"
  fi
}

fallback_command_ledger_fact() {
  if [[ "${command_ledger_available}" != "true" || ! -r "${command_ledger_path}" ]]; then
    echo "本次验证或诊断命令的执行事实不可确认。"
    return 0
  fi
  "${PYTHON_BIN}" - "${command_ledger_path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    ledger = json.load(stream)
passed = sum(item["exit_code"] == 0 for item in ledger)
failed = len(ledger) - passed
duration = sum(float(item.get("duration_seconds", 0.0)) for item in ledger)
if not ledger:
    print("本次没有执行新的验证或诊断命令。")
elif failed:
    print(
        f"已保留 {len(ledger)} 条验证或诊断命令记录："
        f"{passed} 条成功、{failed} 条失败，总耗时 {duration:.3f} 秒；"
        "失败记录对整体结论的影响仍需结合对应命令和根因判断。"
    )
else:
    print(
        f"已保留 {len(ledger)} 条验证或诊断命令记录，均成功，"
        f"总耗时 {duration:.3f} 秒。"
    )
PY
}

write_fallback_command_ledger_table() {
  if [[ "${command_ledger_available}" != "true" || ! -r "${command_ledger_path}" ]]; then
    echo "命令执行记录不可确认。"
    return 0
  fi
  if ! "${PYTHON_BIN}" - "${command_ledger_path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    ledger = json.load(stream)

def cell(value: object, limit: int = 1000) -> str:
    text = str(value).replace("|", "\\|").replace("`", "\\`").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"

if not ledger:
    print("未记录新增验证或诊断命令。")
else:
    print("| 命令 | 退出码 | 耗时（秒） |")
    print("| --- | ---: | ---: |")
    for item in ledger:
        print(
            f"| `{cell(item['command'])}` | {item['exit_code']} | "
            f"{float(item.get('duration_seconds', 0.0)):.3f} |"
        )
PY
  then
    echo "命令执行记录不可确认。"
  fi
}

write_fallback_changed_files_table() {
  local max_rows="${1:-0}"
  if [[ "${changed_files_manifest_available}" != "true" || ! -r "${changed_files_manifest_path}" ]]; then
    echo "变更文件清单尚未生成或不可确认。"
    return 0
  fi
  if ! "${PYTHON_BIN}" - "${changed_files_manifest_path}" "${max_rows}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
max_rows = int(sys.argv[2])
if not isinstance(manifest, list):
    raise SystemExit("changed-files manifest is not an array")

labels = {
    "added": "新增",
    "modified": "修改",
    "deleted": "删除",
    "renamed": "重命名",
}

for index, item in enumerate(manifest):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise SystemExit(f"changed-files manifest item {index} is invalid")
    if item.get("change_type") not in labels:
        raise SystemExit(f"changed-files manifest item {index} has an invalid type")
paths = [item["path"] for item in manifest]
if len(paths) != len(set(paths)):
    raise SystemExit("changed-files manifest contains duplicate paths")

def cell(value: object, limit: int = 500) -> str:
    text = str(value).replace("|", "\\|").replace("`", "\\`").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"

print("| 文件 | 类型 | 改动说明 | 影响 |")
print("| --- | --- | --- | --- |")
if not manifest:
    print("| 无 | 无 | 本次差异没有变更文件。 | 不适用。 |")
shown = manifest if max_rows <= 0 else manifest[:max_rows]
for item in shown:
    change_type = labels[item["change_type"]]
    path = cell(item["path"])
    print(
        f"| `{path}` | {change_type} | "
        "自动审查未完成，未能可靠归纳该文件的具体改动。 | "
        "该文件的行为影响仍需结合代码差异人工核对。 |"
    )
if len(shown) < len(manifest):
    print(
        f"| 其余 {len(manifest) - len(shown)} 个文件 | 省略 | "
        "评论长度受限，完整文件清单保留在任务结果产物中。 | "
        "需结合完整差异人工核对。 |"
    )
PY
  then
    echo "变更文件清单不可读取。"
  fi
}

limit_public_comment() {
  "${PYTHON_BIN}" - "${comment_path}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
limit = 58_000
if len(text) <= limit:
    raise SystemExit(0)

changed_heading = "\n### 变更文件\n"
if changed_heading in text:
    changed_start = text.index(changed_heading)
    details_end = text.find("\n</details>", changed_start)
    if details_end >= 0:
        tail_start = details_end + len("\n</details>")
        text = (
            text[:changed_start]
            + changed_heading
            + "\n变更文件较多，公开评论已按长度上限省略文件表；"
            "完整清单保留在本次任务结果产物中。\n"
            + text[tail_start:]
        )

if len(text) > limit:
    validation_heading = "\n### 验证情况\n"
    if validation_heading in text:
        tail = text[text.index(validation_heading) :]
        marker = "\n\n（前文已按评论长度上限截断。）\n"
        available = max(limit - len(marker) - len(tail), 0)
        text = text[:available].rstrip() + marker + tail

if len(text) > limit:
    suffix = "\n\n（评论已按长度上限截断，完整信息保留在任务结果产物中。）\n"
    text = text[: limit - len(suffix)].rstrip() + suffix

temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(text, encoding="utf-8")
temporary.replace(path)
PY
}

write_failure_report() {
  local rendered_diff_mode="two-point"
  local repository_root_args=()
  local fallback_assessment_label="无法判断"
  local fallback_contributor_goal="Codex AI 自动审查未完成，未能可靠归纳贡献者的修改目标。"
  local fallback_expected_behavior="Codex AI 自动审查未完成，未能可靠归纳贡献者声明的预期行为。"
  local fallback_implementation_summary="当前没有足够证据判断代码是否实现了声明目标。"
  local fallback_validation_fact="本次验证或诊断命令的执行事实不可确认。"
  local fallback_changed_file_count="不可确认"
  local public_failure_reason
  report_verdict="WARNING"
  test_execution_status="unavailable"
  generated_test_file_count="UNKNOWN"
  test_command_count="UNKNOWN"
  max_test_command_duration_seconds="UNKNOWN"
  total_test_command_duration_seconds="UNKNOWN"
  constraint_status="warning"
  constraint_reason="Codex AI 自动审查未完成，无法确认测试执行是否符合轻量约束。"
  refresh_command_ledger_state || true
  fallback_validation_fact="$(fallback_command_ledger_fact)"
  if [[ "${generated_archive_available}" == "true" ]]; then
    fallback_validation_fact+=" 已收集的任务级测试文件归档已保留在结果产物中，具体内容需人工核对。"
  fi
  if [[ "${changed_files_manifest_available}" == "true" ]]; then
    fallback_changed_file_count="${changed_file_count}"
  fi
  if [[ "${diff_mode}" == "merge-base" ]]; then
    rendered_diff_mode="merge-base"
  fi
  if [[ "${change_request_context_status}" == "not_applicable" ]]; then
    fallback_assessment_label="不适用"
    fallback_contributor_goal="当前任务不是 PR，因此没有需要对照的贡献者功能声明。"
    fallback_expected_behavior="当前任务不是 PR，因此贡献者预期行为不适用。"
    fallback_implementation_summary="当前任务不是 PR，不进行贡献者声明对照；Codex AI 审查未完成。"
  fi
  if [[ -n "${workspace_dir}" && -d "${workspace_dir}" ]]; then
    repository_root_args=(--repository-root "${workspace_dir}")
  fi

  public_failure_reason="$(codex_failure_public_reason "${failure_code:-codex_execution_failed}")"
  local command_ledger_state="unavailable"
  local generated_archive_state="unavailable"
  if [[ "${command_ledger_available}" == "true" ]]; then
    command_ledger_state="available"
  fi
  if [[ "${generated_archive_available}" == "true" ]]; then
    generated_archive_state="available"
  fi
  if [[ "${changed_files_manifest_available}" == "true" \
    && -r "${report_builder_path}" \
    && -r "${changed_files_manifest_path}" ]] && \
    "${PYTHON_BIN}" "${report_builder_path}" build-fallback \
      --output "${report_json_path}" \
      --manifest "${changed_files_manifest_path}" \
      --command-ledger "${command_ledger_path}" \
      --command-ledger-state "${command_ledger_state}" \
      --generated-archive "${generated_files_path}" \
      --generated-archive-state "${generated_archive_state}" \
      --failure-reason "${public_failure_reason}" \
      --change-request-context-status "${change_request_context_status}" \
      >> "${log_path}" 2>&1; then
    load_execution_metadata || true
  else
    echo "Codex AI CI 最末级报告：无法生成标准 fallback 数据。" >> "${log_path}"
  fi

  if [[ "${changed_files_manifest_available}" == "true" \
    && -r "${renderer_path}" \
    && -r "${changed_files_manifest_path}" ]] && \
    "${PYTHON_BIN}" "${renderer_path}" \
      --input "${report_json_path}" \
      --output "${report_path}" \
      --comment-output "${comment_path}" \
      --branch "${branch}" \
      --base-sha "${base_sha:-不可用}" \
      --requested-base-sha "${requested_base_sha}" \
      --diff-mode "${rendered_diff_mode}" \
      --target-sha "${target_sha}" \
      --head-sha "${requested_head_sha}" \
      --tested-sha-kind "$([[ "${branch}" =~ ^ci/pr-[0-9]+/.+ ]] && printf '%s' pr_merge || printf '%s' commit)" \
      --local-ci-status "${local_ci_status}" \
      --local-ci-execution-mode "${LOCAL_CI_EXECUTION_MODE}" \
      --backend-validation-scope "${backend_validation_scope}" \
      --changed-file-count "${changed_file_count}" \
      --changed-files-manifest "${changed_files_manifest_path}" \
      --constraint-status "${constraint_status}" \
      --constraint-reason "${constraint_reason}" \
      "${repository_root_args[@]}" \
      >/dev/null 2>> "${log_path}"; then
    return 0
  fi

  {
    echo "# Codex AI 自动审查报告"
    echo
    echo "## 元数据"
    echo
    echo "| 字段 | 值 |"
    echo "| --- | --- |"
    echo "| 报告格式 | \`triton-anchor-codex-ai-report/v3\` |"
    echo "| 分支 | \`${branch}\` |"
    echo "| 请求的基础提交 | \`${requested_base_sha:-不可用}\` |"
    echo "| PR Head 提交 | \`${requested_head_sha:-不可用}\` |"
    echo "| 实际审查起点 | \`${base_sha:-不可用}\` |"
    echo "| 差异模式 | \`${diff_mode}\` |"
    echo "| 测试提交 | \`${target_sha}\` |"
    echo "| 变更文件数 | ${fallback_changed_file_count} |"
    echo "| 生成时间（UTC） | \`${start_time}\` |"
    echo
    echo "## 结论"
    echo
    echo "**警告**"
    echo
    echo "## 摘要"
    echo
    echo "Codex AI 自动审查未完成：${failure_reason}"
    echo
    echo "## 贡献者目标与实现情况"
    echo
    echo "- 判断：${fallback_assessment_label}"
    echo "- 修改目标：${fallback_contributor_goal}"
    echo "- 预期行为：${fallback_expected_behavior}"
    echo "- 实现情况：${fallback_implementation_summary}"
    echo "- 判断依据："
    echo "  - Codex AI 自动审查未完成，当前没有足够证据判断贡献者声明和实际实现是否一致。"
    echo
    echo "## 合入建议"
    echo
    echo "本次 AI 意见不可用；确定性门禁事实不受影响，请结合门禁结果和人工审查决定是否合入。"
    echo
    echo "## 具体文件变更"
    echo
    write_fallback_changed_files_table
    echo
    echo "## 行为覆盖"
    echo
    echo "正常、边界、错误、兼容和集成路径均未获得可信结论。"
    echo
    echo "## 关键问题"
    echo
    echo "分析未完成，无法给出可靠的问题结论。"
    echo
    echo "## 建议测试"
    echo
    echo "无。"
    echo
    echo "## 测试执行"
    echo
    echo "- ${fallback_validation_fact}"
    echo
    write_fallback_command_ledger_table
    echo
    echo "## 剩余风险"
    echo
    echo "- 本次 Codex AI 自动审查未完成，当前代码差异仍需人工检查。"
    echo
    echo "## 执行标记"
    echo
    echo "CODEX_AI_CI_FAILED"
  } > "${report_path}"
  {
    echo "## Codex AI 自动审查"
    echo
    echo "> Codex AI 自动审查仅供参考且不阻塞合入；本地确定性 CI 检查结果才是合入门禁。"
    echo
    echo "### 审查摘要"
    echo
    if [[ "${LOCAL_CI_EXECUTION_MODE}" == "codex_only" ]]; then
      echo "- 本地确定性 CI 检查：**按策略跳过**；本次任务只执行 Codex AI 自动审查。"
    elif [[ "${backend_validation_scope}" == "frontend_only" && "${local_ci_status}" == "0" ]]; then
      echo "- 本地确定性 CI 检查：**通过（仅前端范围）**；本次未执行厂商后端验证。"
    elif [[ "${backend_validation_scope}" == "frontend_only" ]]; then
      echo "- 本地确定性 CI 检查：**失败（前端范围）**；本次未执行厂商后端验证。"
    elif [[ "${local_ci_status}" == "0" ]]; then
      echo "- 本地确定性 CI 检查：**通过**。"
    else
      echo "- 本地确定性 CI 检查：**失败**。"
    fi
    echo "- 合入建议：本次 AI 意见不可用；确定性门禁事实不受影响，请结合门禁结果和人工审查决定是否合入。"
    echo
    echo "$(codex_failure_public_reason "${failure_code:-codex_execution_failed}")。"
    echo
    echo "### 贡献者目标与实现情况"
    echo
    echo "- 判断：**${fallback_assessment_label}**"
    echo "- 贡献者目标：${fallback_contributor_goal}"
    echo "- 预期效果：${fallback_expected_behavior}"
    echo "- 当前实现情况：${fallback_implementation_summary}"
    echo "- 判断依据："
    echo "  - Codex AI 自动审查未完成，当前没有足够证据可靠判断贡献者声明和实际实现是否一致。"
    echo
    echo "### 需要处理的问题"
    echo
    echo "本次审查未形成可确认的具体代码问题结论。"
    echo
    echo "### 验证情况"
    echo
    echo "- 验证内容与结果："
    echo "  - ${fallback_validation_fact}"
    echo "- 限制与未覆盖："
    echo "  - $(codex_failure_public_reason "${failure_code:-codex_execution_failed}")。"
    if [[ "${backend_validation_scope}" == "frontend_only" ]]; then
      echo "  - ${BACKEND_SKIP_REASON}"
    fi
    echo "  - 本次 AI 审查未完成，代码差异仍需人工核对。"
    echo
    echo "### 剩余风险"
    echo
    echo "- 本次 AI 审查未完成，可能仍有未被识别的代码风险。"
    echo
    echo "### 变更文件"
    echo
    echo "<details>"
    echo "<summary>展开文件级变更表</summary>"
    echo
    write_fallback_changed_files_table 50
    echo
    echo "</details>"
  } > "${comment_path}"
  limit_public_comment
}

append_credential_integrity_warning() {
  if [[ "${credential_integrity_status}" != "warning" ]]; then
    return 0
  fi
  {
    echo
    echo "## 凭据完整性"
    echo
    echo "- 状态：警告"
    echo "- 说明：${credential_integrity_reason}"
  } >> "${report_path}"
  {
    echo
    echo "### Codex AI CI 凭据完整性警告"
    echo
    echo "${credential_integrity_reason}"
  } >> "${comment_path}"
  echo "Codex AI CI 凭据完整性警告：${credential_integrity_reason}" >> "${log_path}"
}

append_change_request_context_warning() {
  local public_context_reason=""
  case "${change_request_context_status}" in
    missing)
      public_context_reason="未取得与当前 PR 测试提交匹配的功能声明元数据；本次审查已结束，功能声明对照不可确认。"
      ;;
    invalid)
      public_context_reason="PR 功能声明元数据未通过校验；本次审查已结束，功能声明对照不可确认。"
      ;;
    *) return 0 ;;
  esac
  {
    echo
    echo "## PR 功能声明上下文"
    echo
    echo "- 状态：警告"
    echo "- 说明：${public_context_reason}"
  } >> "${report_path}"
  {
    echo
    echo "### PR 功能声明上下文警告"
    echo
    echo "${public_context_reason}"
  } >> "${comment_path}"
  echo "PR 功能声明上下文警告：${public_context_reason}" >> "${log_path}"
}

codex_failure_code_for_reason() {
  local reason="$1"
  case "${reason}" in
    *"Codex CLI"* | *"找不到可执行的 Codex"*) echo "codex_cli_unavailable" ;;
    *"CODEX_AI_CI_HOME"* | *"独立凭据"* | *"config.toml"* | *"auth.json"*) echo "credential_validation_failed" ;;
    *"提示词"* | *"渲染"*) echo "prompt_render_failed" ;;
    *"容器准备阶段"* | *"环境快照超时"*) echo "container_prepare_timeout" ;;
    *"硬超时"*) echo "timeout" ;;
    *"启动阶段"* | *"首个有效进展"*) echo "startup_timeout" ;;
    *"CODEX_AI_CI_COMPLETE"*) echo "missing_completion_marker" ;;
    *"turn.completed"*) echo "missing_turn_completed" ;;
    *"没有执行任何"*) echo "no_command_executed" ;;
    *"语义载荷"* | *"schema"*) echo "analysis_contract_failed" ;;
    *"可信报告输入"*) echo "trusted_report_input_failed" ;;
    *"内部报告契约"*) echo "report_contract_failed" ;;
    *"执行事实元数据"*) echo "report_metadata_failed" ;;
    *"容器"* | *"Docker socket"* | *"镜像"*) echo "container_setup_failed" ;;
    *"checkout"* | *"差异"* | *"变更文件清单"*) echo "checkout_or_diff_failed" ;;
    *"宿主机缺少"* | *"Python"*) echo "prerequisite_failed" ;;
    *) echo "codex_execution_failed" ;;
  esac
}

codex_failure_public_reason() {
  case "$1" in
    codex_cli_unavailable) echo "Codex AI 自动审查工具在当前环境中不可用" ;;
    credential_validation_failed) echo "Codex 审查凭据校验未通过" ;;
    prompt_render_failed) echo "Codex 审查输入准备失败" ;;
    container_prepare_timeout) echo "Codex 审查运行环境准备超时" ;;
    timeout) echo "Codex 自动审查执行超时" ;;
    startup_timeout) echo "Codex 自动审查启动阶段超时" ;;
    missing_completion_marker | missing_turn_completed) echo "Codex 自动审查没有完整结束" ;;
    no_command_executed) echo "Codex 自动审查未获得可确认的审查或验证操作记录" ;;
    analysis_contract_failed | schema_validation_failed) echo "Codex 审查结果整理失败" ;;
    trusted_report_input_failed) echo "Codex 审查所需的任务证据校验失败" ;;
    report_contract_failed) echo "Codex 审查报告生成失败" ;;
    report_metadata_failed) echo "Codex 审查执行记录读取失败" ;;
    invalid_finding_location) echo "Codex 问题定位信息校验未通过" ;;
    container_setup_failed) echo "Codex 审查运行环境启动失败" ;;
    checkout_or_diff_failed) echo "Codex 审查代码或差异准备失败" ;;
    prerequisite_failed) echo "Codex 审查运行环境缺少必要组件" ;;
    *) echo "Codex 自动审查执行异常" ;;
  esac
}

set_failure_reason() {
  failure_reason="$1"
  if [[ -z "${failure_code}" ]]; then
    failure_code="$(codex_failure_code_for_reason "${failure_reason}")"
  fi
}

fail_ai_ci() {
  set_failure_reason "$1"
  verify_credential_integrity
  write_failure_report
  append_change_request_context_warning
  append_credential_integrity_warning
  limit_public_comment
  echo "Codex AI CI 失败：${failure_code} ${failure_reason}" >> "${log_path}"
  write_summary
  echo "Codex AI CI：失败（${failure_code}：${failure_reason}）"
  exit 1
}

resolve_codex_binary() {
  if [[ "${CODEX_BIN}" == */* ]]; then
    host_codex_bin="${CODEX_BIN}"
  else
    host_codex_bin="$(command -v "${CODEX_BIN}" 2>/dev/null || true)"
  fi
  if [[ -z "${host_codex_bin}" || ! -x "${host_codex_bin}" ]]; then
    fail_ai_ci "宿主机上找不到可执行的 Codex CLI：${CODEX_BIN}"
  fi
}

credential_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

capture_credential_hashes() {
  config_sha256_before="$(credential_sha256 "${CODEX_AI_CI_HOME}/config.toml")"
  auth_sha256_before="$(credential_sha256 "${CODEX_AI_CI_HOME}/auth.json")"
  credential_hashes_initialized="true"
  credential_integrity_status="pass"
  credential_integrity_reason="独立凭据文件在任务执行前后保持不变。"
}

verify_credential_integrity() {
  local config_sha256_after
  local auth_sha256_after
  if [[ "${credential_hashes_initialized}" != "true" ]]; then
    return 0
  fi
  if [[ ! -f "${CODEX_AI_CI_HOME}/config.toml" || ! -f "${CODEX_AI_CI_HOME}/auth.json" ]]; then
    credential_integrity_status="warning"
    credential_integrity_reason="任务执行期间独立凭据文件被删除或替换；系统未自动恢复文件。"
    return 0
  fi
  config_sha256_after="$(credential_sha256 "${CODEX_AI_CI_HOME}/config.toml" 2>/dev/null || true)"
  auth_sha256_after="$(credential_sha256 "${CODEX_AI_CI_HOME}/auth.json" 2>/dev/null || true)"
  if [[ -z "${config_sha256_after}" || -z "${auth_sha256_after}" ]]; then
    credential_integrity_status="warning"
    credential_integrity_reason="任务结束时无法重新计算独立凭据文件哈希；系统未修改或恢复文件。"
  elif [[ "${config_sha256_after}" != "${config_sha256_before}" || "${auth_sha256_after}" != "${auth_sha256_before}" ]]; then
    credential_integrity_status="warning"
    credential_integrity_reason="任务执行期间独立凭据文件内容发生变化；系统未自动恢复文件，请人工检查。"
  else
    credential_integrity_status="pass"
    credential_integrity_reason="独立凭据文件在任务执行前后保持不变。"
  fi
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  case "${value}" in
    "" | *[!0-9]*) fail_ai_ci "${name} 必须是正整数" ;;
  esac
  if (( 10#${value} <= 0 )); then
    fail_ai_ci "${name} 必须大于 0"
  fi
}

render_prompt_template() {
  local template_path="$1"
  shift
  "${PYTHON_BIN}" - "${template_path}" "$@" <<'PY'
import sys
from pathlib import Path
from string import Template

template_path = Path(sys.argv[1])
items = sys.argv[2:]
if len(items) % 2:
    print("提示词模板变量必须按名称和值成对传入", file=sys.stderr)
    raise SystemExit(2)
values = dict(zip(items[0::2], items[1::2]))
try:
    rendered = Template(template_path.read_text(encoding="utf-8")).substitute(values)
except (OSError, UnicodeError, KeyError, ValueError) as exc:
    print(f"无法渲染提示词模板 {template_path}: {exc}", file=sys.stderr)
    raise SystemExit(2)
sys.stdout.write(rendered)
PY
}

build_unavailable_change_request_context() {
  "${PYTHON_BIN}" - "${change_request_context_status}" \
    "${change_request_context_reason}" <<'PY'
import json
import sys

print(
    json.dumps(
        {"status": sys.argv[1], "reason": sys.argv[2]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
)
PY
}

load_change_request_context() {
  if [[ ! "${branch}" =~ ^ci/pr-[0-9]+/.+ ]]; then
    change_request_context_status="not_applicable"
    change_request_context_reason="当前任务不是 PR，功能声明上下文不适用。"
    change_request_context_json="$(build_unavailable_change_request_context)" || \
      fail_ai_ci "无法生成非 PR 功能声明上下文"
    return 0
  fi

  if [[ -z "${task_metadata_file}" || ! -f "${task_metadata_file}" ]]; then
    change_request_context_status="missing"
    change_request_context_reason="未取得与当前 PR 测试提交匹配的功能声明元数据；功能声明对照不可确认，审查依据代码差异和测试证据进行。"
    change_request_context_json="$(build_unavailable_change_request_context)" || \
      fail_ai_ci "无法生成 PR 元数据缺失上下文"
    echo "${change_request_context_reason}" >> "${log_path}"
    return 0
  fi

  if [[ "${task_metadata_file}" != "${task_metadata_output_path}" ]]; then
    rm -f -- "${task_metadata_output_path}"
  fi
  local validation_message=""
  if ! validation_message="$(
    "${PYTHON_BIN}" "${task_metadata_validator}" \
      --input "${task_metadata_file}" \
      --output "${task_metadata_output_path}" \
      --task-ref "${branch}" \
      --target-sha "${target_sha}" \
      --base-sha "${requested_base_sha}" \
      --head-sha "${requested_head_sha}" 2>&1
  )"; then
    rm -f -- "${task_metadata_output_path}"
    validation_message="${validation_message//$'\n'/ }"
    change_request_context_status="invalid"
    change_request_context_reason="PR 功能声明元数据未通过校验；功能声明对照不可确认，审查依据代码差异和测试证据进行。"
    change_request_context_diagnostic="${validation_message}"
    change_request_context_json="$(build_unavailable_change_request_context)" || \
      fail_ai_ci "无法生成 PR 元数据无效上下文"
    if [[ -n "${validation_message}" ]]; then
      printf '%s\n' "${validation_message}" >> "${log_path}"
    fi
    echo "${change_request_context_reason}" >> "${log_path}"
    return 0
  fi

  if [[ -n "${validation_message}" ]]; then
    printf '%s\n' "${validation_message}" >> "${log_path}"
  fi
  local context_parts=()
  mapfile -t context_parts < <(
    "${PYTHON_BIN}" - "${task_metadata_output_path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    metadata = json.load(stream)
context = {
    "status": "available",
    "pr_number": metadata["pr_number"],
    "title": metadata["title"],
    "description": metadata["description"],
    "captured_at": metadata["captured_at"],
    "title_truncated": metadata["title_truncated"],
    "description_truncated": metadata["description_truncated"],
}
for key in (
    "event_kind",
    "base_branch",
    "base_sha",
    "head_branch",
    "head_sha",
    "head_repo",
    "tested_ref",
    "tested_sha",
    "tested_sha_kind",
):
    if key in metadata:
        context[key] = metadata[key]
print(metadata["pr_number"])
print(json.dumps(context, ensure_ascii=False, separators=(",", ":")))
print(metadata.get("base_sha", ""))
print(metadata.get("head_sha", ""))
print(metadata.get("base_task_ref", ""))
print(metadata.get("head_task_ref", ""))
PY
  )
  if [[ "${#context_parts[@]}" -lt 2 || -z "${context_parts[1]}" ]]; then
    rm -f -- "${task_metadata_output_path}"
    change_request_context_status="invalid"
    change_request_context_reason="规范化后的 PR 功能声明元数据无法读取；功能声明对照不可确认，审查依据代码差异和测试证据进行。"
    change_request_context_json="$(build_unavailable_change_request_context)" || \
      fail_ai_ci "无法生成 PR 元数据读取失败上下文"
    echo "${change_request_context_reason}" >> "${log_path}"
    return 0
  fi

  local metadata_base_sha="${context_parts[2]:-}"
  local metadata_head_sha="${context_parts[3]:-}"
  local metadata_base_ref="${context_parts[4]:-}"
  local metadata_head_ref="${context_parts[5]:-}"
  if [[ -z "${requested_base_sha}" && -n "${metadata_base_sha}" ]]; then
    requested_base_sha="${metadata_base_sha}"
  fi
  if [[ -z "${requested_head_sha}" && -n "${metadata_head_sha}" ]]; then
    requested_head_sha="${metadata_head_sha}"
  fi
  if [[ -z "${requested_base_ref}" && -n "${metadata_base_ref}" ]]; then
    requested_base_ref="${metadata_base_ref}"
  fi
  if [[ -z "${requested_head_ref}" && -n "${metadata_head_ref}" ]]; then
    requested_head_ref="${metadata_head_ref}"
  fi

  change_request_context_status="available"
  change_request_context_reason="已校验并载入与当前 PR 测试提交匹配的功能声明元数据。"
  change_request_context_pr_number="${context_parts[0]}"
  change_request_context_json="${context_parts[1]}"
}

validate_prerequisites() {
  local integer_name
  local prompt_template
  local integer_names=(
    CODEX_AI_CI_TIMEOUT_SECONDS
    CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS
    CODEX_AI_CI_MIN_GENERATED_TEST_CASES
    CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS
    CODEX_AI_CI_MAX_GENERATED_TEST_CASES
    CODEX_AI_CI_MAX_GENERATED_TEST_FILES
    CODEX_AI_CI_MAX_TEST_COMMANDS
    CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS
    CODEX_AI_CI_TEST_BUDGET_SECONDS
    CODEX_AI_CI_REPORT_RESERVE_SECONDS
    CODEX_AI_CI_PIDS_LIMIT
  )
  for integer_name in "${integer_names[@]}"; do
    validate_positive_integer "${integer_name}" "${!integer_name}"
  done
  if [[ ! "${CODEX_AI_CI_CPUS}" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]]; then
    fail_ai_ci "CODEX_AI_CI_CPUS 必须是正数"
  fi
  if [[ ! "${CODEX_AI_CI_MEMORY}" =~ ^[1-9][0-9]*[kKmMgG]$ ]]; then
    fail_ai_ci "CODEX_AI_CI_MEMORY 必须是 Docker 支持的正整数容量"
  fi
  if [[ ! "${CODEX_AI_CI_MEMORY_SWAP}" =~ ^[1-9][0-9]*[kKmMgG]$ ]]; then
    fail_ai_ci "CODEX_AI_CI_MEMORY_SWAP 必须是 Docker 支持的正整数容量"
  fi

  if ((
    10#${CODEX_AI_CI_MIN_GENERATED_TEST_CASES} >
      10#${CODEX_AI_CI_MAX_GENERATED_TEST_CASES}
  )); then
    fail_ai_ci "生成测试用例下限不能大于上限"
  fi

  case "${local_ci_status}" in
    "" | *[!0-9]*) fail_ai_ci "Local CI 状态必须是非负整数" ;;
  esac
  case "${LOCAL_CI_CONTAINER}" in
    "" | *[!A-Za-z0-9_.-]*) fail_ai_ci "Local CI 容器名称无效：${LOCAL_CI_CONTAINER}" ;;
  esac
  case "${RUN_BACKEND_STAGES}" in
    true | false) ;;
    *) fail_ai_ci "RUN_BACKEND_STAGES 必须是 true 或 false" ;;
  esac
  case "${LOCAL_CI_EXECUTION_MODE}" in
    full | codex_only) ;;
    *) fail_ai_ci "LOCAL_CI_EXECUTION_MODE 必须是 full 或 codex_only" ;;
  esac

  if ! command -v timeout >/dev/null 2>&1; then
    fail_ai_ci "宿主机缺少 timeout 命令"
  fi
  if ! command -v docker >/dev/null 2>&1; then
    fail_ai_ci "宿主机缺少 docker 命令"
  fi
  if ! command -v git >/dev/null 2>&1; then
    fail_ai_ci "宿主机缺少 git 命令"
  fi
  if ! command -v sha256sum >/dev/null 2>&1; then
    fail_ai_ci "宿主机缺少 sha256sum 命令"
  fi
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    fail_ai_ci "宿主机找不到 Python：${PYTHON_BIN}"
  fi
  if [[ ! -r "${schema_path}" ]]; then
    fail_ai_ci "语义分析 schema 不可读：${schema_path}"
  fi
  if [[ ! -r "${renderer_path}" ]]; then
    fail_ai_ci "报告渲染器不可读：${renderer_path}"
  fi
  if [[ ! -r "${report_builder_path}" ]]; then
    fail_ai_ci "canonical 报告构建器不可读：${report_builder_path}"
  fi
  if [[ ! -r "${jsonl_evidence_path}" ]]; then
    fail_ai_ci "Codex JSONL 证据工具不可读：${jsonl_evidence_path}"
  fi
  if [[ ! -r "${review_context_classifier}" ]]; then
    fail_ai_ci "审查上下文分类器不可读：${review_context_classifier}"
  fi
  if [[ ! -r "${checkout_helper}" ]]; then
    fail_ai_ci "checkout helper 不可读：${checkout_helper}"
  fi
  if [[ ! -r "${credentials_validator}" ]]; then
    fail_ai_ci "独立凭据校验器不可读：${credentials_validator}"
  fi
  if [[ ! -r "${task_metadata_validator}" ]]; then
    fail_ai_ci "PR 功能声明元数据校验器不可读：${task_metadata_validator}"
  fi
  for prompt_template in \
    "${success_prompt_template}" \
    "${failure_prompt_template}"; do
    [[ -r "${prompt_template}" ]] || \
      fail_ai_ci "Codex 提示词模板不可读：${prompt_template}"
  done

  resolve_codex_binary
  if [[ -z "${CODEX_AI_CI_HOME}" ]]; then
    fail_ai_ci "必须设置独立的 CODEX_AI_CI_HOME"
  fi
  local credential_validation_output
  if ! credential_validation_output="$(
    "${PYTHON_BIN}" "${credentials_validator}" \
      --codex-home "${CODEX_AI_CI_HOME}" \
      --personal-codex-home "${HOME}/.codex" \
      --quiet 2>&1
  )"; then
    [[ -n "${credential_validation_output}" ]] && \
      echo "${credential_validation_output}" >> "${log_path}"
    fail_ai_ci "${credential_validation_output:-Codex AI CI 独立凭据校验失败}"
  fi
  if [[ -n "${credential_validation_output}" ]]; then
    echo "${credential_validation_output}" >> "${log_path}"
  fi
  capture_credential_hashes

  if [[ "$(docker inspect --format '{{.State.Running}}' "${LOCAL_CI_CONTAINER}" 2>> "${log_path}" || true)" != "true" ]]; then
    fail_ai_ci "Local CI 容器未运行：${LOCAL_CI_CONTAINER}"
  fi
  if docker inspect --format '{{range .Mounts}}{{println .Destination}}{{end}}' \
    "${LOCAL_CI_CONTAINER}" 2>> "${log_path}" | grep -Fxq '/var/run/docker.sock'; then
    fail_ai_ci "Local CI 容器挂载了 Docker socket，拒绝将其传递给 Codex 容器"
  fi
}

discover_artifact_dir() {
  local candidate=""
  local resolved_root=""
  local resolved_candidate=""
  if [[ ! -f "${output_dir}/local-ci.log" ]]; then
    return 0
  fi
  candidate="$(
    sed -n 's/.*Artifacts are in \([^[:space:]]*\).*/\1/p' \
      "${output_dir}/local-ci.log" | tail -n 1
  )"
  if [[ -z "${candidate}" ]]; then
    return 0
  fi
  if [[ -z "${ephemeral_container}" ]]; then
    artifact_dir=""
    return 0
  fi
  resolved_root="$(
    docker exec --user 0 "${ephemeral_container}" \
      readlink -e -- "${LOCAL_CI_ARTIFACT_ROOT}" 2>> "${log_path}" || true
  )"
  resolved_candidate="$(
    docker exec --user 0 "${ephemeral_container}" \
      readlink -e -- "${candidate}" 2>> "${log_path}" || true
  )"
  if [[ -z "${resolved_root}" || -z "${resolved_candidate}" \
    || "${resolved_candidate}" == "${resolved_root}" \
    || "${resolved_candidate}" != "${resolved_root}"/* ]]; then
    echo "忽略不在预期 artifact 根目录中的日志路径：${candidate}" >> "${log_path}"
    artifact_dir=""
    return 0
  fi
  artifact_dir="${resolved_candidate}"
}

diff_requires_generated_tests() {
  local changed_path
  local diff_args=(
    -C "${workspace_dir}"
    diff
    --name-only
    --diff-filter=ACDMRTUXB
    --find-renames
    "${diff_revisions[@]}"
  )
  while IFS= read -r changed_path; do
    [[ -z "${changed_path}" ]] && continue
    case "${changed_path}" in
      docs/* | *.md | *.markdown | *.rst)
        ;;
      README | README.* | LICENSE | LICENSE.* | NOTICE | NOTICE.*)
        ;;
      ROADMAP.md | SECURITY.md)
        ;;
      *)
        return 0
        ;;
    esac
  done < <(git "${diff_args[@]}")
  return 1
}

generate_changed_files_manifest() {
  changed_files_manifest_available="false"
  local diff_args=(
    -C "${workspace_dir}"
    diff
    --name-status
    -z
    --diff-filter=ACDMRTUXB
    --find-renames
    "${diff_revisions[@]}"
  )
  if ! git "${diff_args[@]}" | "${PYTHON_BIN}" -c '
import json
import sys

fields = sys.stdin.buffer.read().split(b"\0")
if fields and fields[-1] == b"":
    fields.pop()

manifest = []
index = 0
while index < len(fields):
    status = fields[index].decode("ascii")
    index += 1
    code = status[:1]
    if code in {"R", "C"}:
        if index + 1 >= len(fields):
            raise SystemExit("incomplete rename/copy record in git diff")
        previous_path = fields[index].decode("utf-8", "replace")
        path = fields[index + 1].decode("utf-8", "replace")
        index += 2
        if code == "R":
            manifest.append({
                "path": path,
                "change_type": "renamed",
                "previous_path": previous_path,
            })
        else:
            manifest.append({"path": path, "change_type": "added"})
        continue
    if index >= len(fields):
        raise SystemExit("incomplete path record in git diff")
    path = fields[index].decode("utf-8", "replace")
    index += 1
    change_type = {
        "A": "added",
        "D": "deleted",
        "M": "modified",
        "T": "modified",
        "U": "modified",
        "X": "modified",
        "B": "modified",
    }.get(code)
    if change_type is None:
        raise SystemExit(f"unsupported git diff status: {status}")
    manifest.append({"path": path, "change_type": change_type})

paths = [item["path"] for item in manifest]
if len(paths) != len(set(paths)):
    raise SystemExit("duplicate path in git diff manifest")
json.dump(manifest, sys.stdout, ensure_ascii=False, separators=(",", ":"))
sys.stdout.write("\n")
' > "${changed_files_manifest_path}"; then
    return 1
  fi

  if ! changed_file_count="$(${PYTHON_BIN} -c '
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
' "${changed_files_manifest_path}")"; then
    return 1
  fi
  changed_files_manifest_available="true"
  if ! "${PYTHON_BIN}" "${report_builder_path}" prepare-manifest \
    --input "${changed_files_manifest_path}" \
    --output "${analysis_manifest_path}" >> "${log_path}" 2>&1; then
    return 1
  fi
  changed_files_manifest_json="$(<"${analysis_manifest_path}")"
}

classify_review_context() {
  local context_lines=()
  mapfile -t context_lines < <(
    "${PYTHON_BIN}" "${review_context_classifier}" \
      "${changed_files_manifest_path}" "${analysis_mode}"
  )
  if [[ "${#context_lines[@]}" -ne 3 ]]; then
    review_context_profile="unclassified"
    review_context_hint="无法生成文件分组摘要；按标准项目专项审查执行。"
    changed_file_groups_json="{}"
    return 0
  fi
  review_context_profile="${context_lines[0]}"
  review_context_hint="${context_lines[1]}"
  changed_file_groups_json="${context_lines[2]}"
  printf '%s\n' "${changed_file_groups_json}" > "${output_dir}/codex-context-summary.json"
}

run_prepare_command() {
  local phase="$1"
  shift
  local remaining_seconds="$((prepare_deadline_seconds - SECONDS))"
  local command_started_seconds="${SECONDS}"
  local command_exit=0

  if ((remaining_seconds <= 0)); then
    prepare_timed_out="true"
    prepare_timeout_phase="${phase}"
    return 124
  fi

  echo "Codex 容器准备开始：${phase}（剩余预算 ${remaining_seconds} 秒）。" >> "${log_path}"
  timeout --signal=TERM --kill-after=30s "${remaining_seconds}s" \
    "$@" >> "${log_path}" 2>&1
  command_exit=$?
  echo "Codex 容器准备结束：${phase}（耗时 $((SECONDS - command_started_seconds)) 秒，退出码 ${command_exit}）。" \
    >> "${log_path}"
  if [[ ${command_exit} -eq 124 || ${command_exit} -eq 137 ]]; then
    prepare_timed_out="true"
    prepare_timeout_phase="${phase}"
  fi
  return "${command_exit}"
}

run_prepare_capture() {
  local output_variable="$1"
  local phase="$2"
  shift 2
  local remaining_seconds="$((prepare_deadline_seconds - SECONDS))"
  local command_started_seconds="${SECONDS}"
  local command_exit=0
  local captured_output=""

  if ((remaining_seconds <= 0)); then
    prepare_timed_out="true"
    prepare_timeout_phase="${phase}"
    return 124
  fi

  echo "Codex 容器准备开始：${phase}（剩余预算 ${remaining_seconds} 秒）。" >> "${log_path}"
  captured_output="$(
    timeout --signal=TERM --kill-after=30s "${remaining_seconds}s" \
      "$@" 2>> "${log_path}"
  )"
  command_exit=$?
  echo "Codex 容器准备结束：${phase}（耗时 $((SECONDS - command_started_seconds)) 秒，退出码 ${command_exit}）。" \
    >> "${log_path}"
  if [[ ${command_exit} -eq 124 || ${command_exit} -eq 137 ]]; then
    prepare_timed_out="true"
    prepare_timeout_phase="${phase}"
  fi
  if [[ ${command_exit} -eq 0 ]]; then
    printf -v "${output_variable}" '%s' "${captured_output}"
  fi
  return "${command_exit}"
}

fail_prepare_step() {
  local reason="$1"
  prepare_duration_seconds="$((SECONDS - prepare_started_seconds))"
  if [[ "${snapshot_duration_seconds}" == "UNKNOWN" ]]; then
    snapshot_duration_seconds="$((SECONDS - phase_started_seconds))"
  elif [[ "${container_start_duration_seconds}" == "UNKNOWN" ]]; then
    container_start_duration_seconds="$((SECONDS - phase_started_seconds))"
  elif [[ "${input_setup_duration_seconds}" == "UNKNOWN" ]]; then
    input_setup_duration_seconds="$((SECONDS - phase_started_seconds))"
  fi
  if [[ "${prepare_timed_out}" == "true" ]]; then
    failure_code="container_prepare_timeout"
    fail_ai_ci "Codex 容器准备阶段超过 ${CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS} 秒（阶段：${prepare_timeout_phase}）"
  fi
  fail_ai_ci "${reason}"
}

create_ephemeral_container() {
  local resource_key
  local workspace_rw
  local container_running
  local mount_destinations
  local copied_sha
  local phase_started_seconds
  resource_key="$(date -u +%Y%m%dT%H%M%SZ)-${target_sha:0:12}-$$"
  resource_key="${resource_key,,}"
  ephemeral_container="anchor-codex-ai-${resource_key}"
  ephemeral_image="triton-anchor-codex-ai-snapshot:${resource_key}"
  prepare_started_seconds="${SECONDS}"
  prepare_deadline_seconds="$((SECONDS + 10#${CODEX_AI_CI_PREPARE_TIMEOUT_SECONDS}))"

  echo "正在从 ${LOCAL_CI_CONTAINER} 创建本次任务的临时镜像 ${ephemeral_image}。" >> "${log_path}"
  phase_started_seconds="${SECONDS}"
  if ! run_prepare_command "environment_snapshot" \
    docker commit \
      --change "LABEL triton-anchor.role=codex-ai-snapshot" \
      --change "LABEL triton-anchor.run-id=${LOCAL_CI_RUN_ID}" \
      --change "LABEL triton-anchor.target-sha=${target_sha}" \
      "${LOCAL_CI_CONTAINER}" "${ephemeral_image}"; then
    snapshot_duration_seconds="$((SECONDS - phase_started_seconds))"
    fail_prepare_step "无法从本次 Local CI 容器创建环境快照"
  fi
  snapshot_duration_seconds="$((SECONDS - phase_started_seconds))"

  phase_started_seconds="${SECONDS}"
  if ! run_prepare_command "container_start" docker run -dit \
    --name "${ephemeral_container}" \
    --hostname "${ephemeral_container}" \
    --cpus "${CODEX_AI_CI_CPUS}" \
    --memory "${CODEX_AI_CI_MEMORY}" \
    --memory-swap "${CODEX_AI_CI_MEMORY_SWAP}" \
    --pids-limit "${CODEX_AI_CI_PIDS_LIMIT}" \
    --label "triton-anchor.role=codex-ai" \
    --label "triton-anchor.run-id=${LOCAL_CI_RUN_ID}" \
    --label "triton-anchor.target-sha=${target_sha}" \
    --volumes-from "${LOCAL_CI_CONTAINER}:ro" \
    --entrypoint /bin/bash \
    "${ephemeral_image}" \
    -lc 'trap : TERM INT; while :; do sleep 3600; done'; then
    container_start_duration_seconds="$((SECONDS - phase_started_seconds))"
    fail_prepare_step "无法启动本次任务的临时 Codex 容器"
  fi

  if ! run_prepare_capture container_running "inspect_container_running" \
    docker inspect --format '{{.State.Running}}' "${ephemeral_container}"; then
    fail_prepare_step "无法检查临时 Codex 容器状态"
  fi
  if [[ "${container_running}" != "true" ]]; then
    fail_prepare_step "临时 Codex 容器启动后未保持运行"
  fi
  if ! run_prepare_capture mount_destinations "inspect_container_mounts" \
    docker inspect --format '{{range .Mounts}}{{println .Destination}}{{end}}' \
      "${ephemeral_container}"; then
    fail_prepare_step "无法检查临时 Codex 容器挂载"
  fi
  if grep -Fxq '/var/run/docker.sock' <<< "${mount_destinations}"; then
    fail_prepare_step "临时 Codex 容器意外挂载了 Docker socket"
  fi
  if ! run_prepare_capture workspace_rw "inspect_workspace_mode" \
    docker inspect \
      --format '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{println .RW}}{{end}}{{end}}' \
      "${ephemeral_container}"; then
    fail_prepare_step "无法检查临时 Codex 容器的 /workspace 挂载模式"
  fi
  if [[ "${workspace_rw}" != "false" ]]; then
    fail_prepare_step "临时 Codex 容器没有以只读方式复用 /workspace"
  fi
  container_start_duration_seconds="$((SECONDS - phase_started_seconds))"

  phase_started_seconds="${SECONDS}"
  if ! run_prepare_command "workspace_setup" docker exec --user 0 "${ephemeral_container}" \
    mkdir -p \
      "${container_codex_home}" \
      "$(dirname "${container_codex_bin}")" \
      "${container_checkout_dir}" \
      "${container_input_dir}"; then
    fail_prepare_step "无法在临时容器中创建 Codex 工作目录"
  fi

  if ! run_prepare_command "copy_codex_cli" docker cp "${host_codex_bin}" \
    "${ephemeral_container}:${container_codex_bin}"; then
    fail_prepare_step "无法把 Codex CLI 复制到临时容器"
  fi
  if ! run_prepare_command "configure_codex_cli" \
    docker exec --user 0 "${ephemeral_container}" \
      chmod +x "${container_codex_bin}"; then
    fail_prepare_step "无法设置容器内 Codex CLI 的执行权限"
  fi
  for config_file in config.toml auth.json; do
    if ! run_prepare_command "copy_${config_file}" docker cp \
      "${CODEX_AI_CI_HOME}/${config_file}" \
      "${ephemeral_container}:${container_codex_home}/${config_file}"; then
      fail_prepare_step "无法把 ${config_file} 复制到临时容器"
    fi
  done

  if ! run_prepare_command "copy_checkout" docker cp "${workspace_dir}/." \
    "${ephemeral_container}:${container_checkout_dir}"; then
    fail_prepare_step "无法把经过验证的 checkout 复制到临时容器"
  fi
  if [[ "${branch}" =~ ^ci/pr-[0-9]+/.+ && "${SOURCE_ENVSETUP}" == "1" ]]; then
    if ! run_prepare_command "copy_trusted_envsetup" docker cp \
      "${TRUSTED_ANCHOR_ENVSETUP}" \
      "${ephemeral_container}:${container_trusted_envsetup}"; then
      fail_prepare_step "无法把可信目标分支环境脚本复制到临时容器"
    fi
  fi
  if ! run_prepare_command "copy_analysis_schema" docker cp "${schema_path}" \
    "${ephemeral_container}:${container_schema_path}"; then
    fail_prepare_step "无法把语义分析 schema 复制到临时容器"
  fi
  if ! run_prepare_command "copy_changed_files_manifest" docker cp \
    "${analysis_manifest_path}" \
    "${ephemeral_container}:${container_changed_files_manifest}"; then
    fail_prepare_step "无法把变更文件清单复制到临时容器"
  fi
  if ! run_prepare_command "copy_jsonl_evidence_tool" docker cp \
    "${jsonl_evidence_path}" \
    "${ephemeral_container}:${container_jsonl_recorder_path}"; then
    fail_prepare_step "无法把 Codex JSONL 证据工具复制到临时容器"
  fi
  if [[ -f "${output_dir}/local-ci.log" ]]; then
    if ! run_prepare_command "copy_local_ci_log" docker cp \
      "${output_dir}/local-ci.log" \
      "${ephemeral_container}:${container_local_ci_log}"; then
      fail_prepare_step "无法把 Local CI 日志复制到临时容器"
    fi
  fi

  if ! run_prepare_command "configure_workspace_ownership" \
    docker exec --user 0 "${ephemeral_container}" \
      chown -R 0:0 \
        "${container_codex_home}" \
        "${container_workspace_root}" \
        "${container_codex_bin}"; then
    fail_prepare_step "无法修正临时容器内 Codex 文件的所有权"
  fi
  if ! run_prepare_command "configure_credential_permissions" \
    docker exec --user 0 "${ephemeral_container}" \
      chmod 600 \
        "${container_codex_home}/config.toml" \
        "${container_codex_home}/auth.json"; then
    fail_prepare_step "无法收紧临时容器内 Codex 配置文件权限"
  fi

  copied_sha="$(
    docker exec --user 0 "${ephemeral_container}" \
      git -C "${container_checkout_dir}" rev-parse HEAD 2>> "${log_path}" || true
  )"
  copied_sha="${copied_sha//$'\r'/}"
  if [[ "${copied_sha}" != "${target_sha}" ]]; then
    fail_prepare_step "容器内 checkout 的 SHA 与目标 SHA 不一致"
  fi
  if ! run_prepare_command "verify_codex_cli" \
    docker exec --user 0 "${ephemeral_container}" \
      "${container_codex_bin}" --version; then
    fail_prepare_step "Codex CLI 无法在临时容器中启动"
  fi
  input_setup_duration_seconds="$((SECONDS - phase_started_seconds))"
  prepare_duration_seconds="$((SECONDS - prepare_started_seconds))"
}

collect_container_workspace() {
  local container_untracked_list="${container_workspace_root}/untracked-files.list"
  local container_generated_files="${container_workspace_root}/codex-generated-files.tar.gz"

  docker exec --user 0 "${ephemeral_container}" \
    git -C "${container_checkout_dir}" status --short --untracked-files=all \
    > "${workspace_status_path}" 2>> "${log_path}" || true
  docker exec --user 0 "${ephemeral_container}" \
    git -C "${container_checkout_dir}" diff --binary HEAD \
    > "${workspace_patch_path}" 2>> "${log_path}" || true
  if docker exec --user 0 "${ephemeral_container}" bash -c \
    'set -euo pipefail; cd "$1"; {
       git diff --name-only --diff-filter=ACMRTUXB -z HEAD;
       git ls-files --others --exclude-standard -z;
     } | sort -zu > "$2";
     tar --null --verbatim-files-from --files-from="$2" -czf "$3"' \
    bash "${container_checkout_dir}" "${container_untracked_list}" \
    "${container_generated_files}" >> "${log_path}" 2>&1; then
    if docker exec --user 0 "${ephemeral_container}" cat "${container_generated_files}" \
      > "${generated_files_path}" 2>> "${log_path}"; then
      generated_archive_available="true"
    fi
  fi
}

if ! mkdir -p "${output_dir}" || [[ ! -w "${output_dir}" ]]; then
  echo "Codex AI CI：失败（输出目录不可写：${output_dir}）" >&2
  exit 1
fi
: > "${log_path}"
: > "${codex_jsonl_path}"
: > "${analysis_json_path}"
: > "${report_json_path}"
: > "${report_path}"
: > "${comment_path}"
: > "${workspace_status_path}"
: > "${workspace_patch_path}"
printf '[]\n' > "${command_ledger_path}"
command_ledger_available="true"

validate_prerequisites
load_change_request_context

if ! workspace_dir="$(
  bash "${checkout_helper}" \
    "${repo_url}" \
    "${branch}" \
    "${CODEX_AI_CI_WORKSPACE_ROOT}" \
    "codex-ai" \
    "${target_sha}" \
    "${requested_base_ref}" \
    "${requested_base_sha}" \
    "${requested_head_ref}" \
    "${requested_head_sha}" 2>> "${log_path}"
)"; then
  fail_ai_ci "无法创建一次性分析 checkout"
fi
workspace_parent="$(dirname "${workspace_dir}")"
actual_sha="$(git -C "${workspace_dir}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${actual_sha}" != "${target_sha}" ]]; then
  fail_ai_ci "checkout 的 SHA 与目标 SHA 不一致"
fi

if [[ "${branch}" =~ ^ci/pr-[0-9]+/.+ ]]; then
  if [[ -z "${requested_base_ref}" ]]; then
    fail_ai_ci "PR Codex 审查缺少目标分支引用"
  fi
  if [[ -z "${requested_base_sha}" ]]; then
    fail_ai_ci "PR Codex 审查缺少目标分支精确 SHA"
  fi
  if [[ -z "${requested_head_sha}" ]]; then
    fail_ai_ci "PR Codex 审查缺少贡献分支精确 SHA"
  fi
  if [[ "${SOURCE_ENVSETUP}" == "1" && ! -r "${TRUSTED_ANCHOR_ENVSETUP}" ]]; then
    fail_ai_ci "PR Codex 审查缺少可信目标分支环境脚本"
  fi
  if ! git -C "${workspace_dir}" cat-file -e "${requested_base_sha}^{commit}" 2>/dev/null; then
    fail_ai_ci "PR 目标分支提交在 Codex checkout 中不可用：${requested_base_sha}"
  fi
  if ! git -C "${workspace_dir}" cat-file -e "${requested_head_sha}^{commit}" 2>/dev/null; then
    fail_ai_ci "PR head 提交在 Codex checkout 中不可用：${requested_head_sha}"
  fi
  if ! base_sha="$(
    git -C "${workspace_dir}" merge-base "${requested_base_sha}" "${requested_head_sha}" 2>/dev/null
  )"; then
    fail_ai_ci "PR head 与目标分支提交没有共同祖先"
  fi
  base_source="merge-base"
  diff_mode="merge-base"
  diff_revisions=("${requested_base_sha}...${requested_head_sha}")
  diff_command="git diff --find-renames ${requested_base_sha}...${requested_head_sha}"
else
  if [[ -n "${requested_base_sha}" ]] &&
    git -C "${workspace_dir}" cat-file -e "${requested_base_sha}^{commit}" 2>/dev/null; then
    base_sha="$(
      git -C "${workspace_dir}" rev-parse "${requested_base_sha}^{commit}" 2>/dev/null
    )"
    base_source="previous-push"
  elif base_sha="$(git -C "${workspace_dir}" rev-parse "${target_sha}^" 2>/dev/null)"; then
    base_source="target-parent"
  else
    if ! base_sha="$(git -C "${workspace_dir}" mktree </dev/null)"; then
      fail_ai_ci "无法创建空树作为审查基线"
    fi
    base_source="empty-tree"
  fi
  diff_mode="two-point"
  diff_revisions=("${base_sha}" "${target_sha}")
  diff_command="git diff --find-renames ${base_sha} ${target_sha}"
fi

if ! generate_changed_files_manifest; then
  fail_ai_ci "无法生成待审查代码差异的标准文件清单"
fi
classify_review_context

if diff_requires_generated_tests; then
  test_generation_expected="true"
fi

create_ephemeral_container
discover_artifact_dir

if [[ "${analysis_mode}" == "full" ]]; then
  selected_prompt_template="${success_prompt_template}"
else
  selected_prompt_template="${failure_prompt_template}"
fi

if ! prompt="$(
  render_prompt_template "${selected_prompt_template}" \
    REPOSITORY_ROOT "${container_checkout_dir}" \
    BRANCH "${branch}" \
    REQUESTED_BASE_REF "${requested_base_ref:-不适用}" \
    REQUESTED_BASE_SHA "${requested_base_sha:-不适用}" \
    REQUESTED_HEAD_REF "${requested_head_ref:-不适用}" \
    REQUESTED_HEAD_SHA "${requested_head_sha:-不适用}" \
    BASE_SHA "${base_sha}" \
    TARGET_SHA "${target_sha}" \
    LOCAL_CI_STATUS "${local_ci_status}" \
    ANALYSIS_MODE "${analysis_mode}" \
    DIFF_MODE "${diff_mode}" \
    DIFF_COMMAND "${diff_command}" \
    CHANGE_REQUEST_CONTEXT_JSON "${change_request_context_json}" \
    REVIEW_CONTEXT_PROFILE "${review_context_profile}" \
    REVIEW_CONTEXT_HINT "${review_context_hint}" \
    CHANGED_FILE_COUNT "${changed_file_count}" \
    CHANGED_FILES_MANIFEST_JSON "${changed_files_manifest_json}" \
    CHANGED_FILE_GROUPS_JSON "${changed_file_groups_json}" \
    CHANGED_FILES_MANIFEST_PATH "${container_changed_files_manifest}" \
    LOCAL_CI_LOG "${container_local_ci_log}" \
    ARTIFACT_DIR "${artifact_dir:-未识别到具体目录}" \
    TEST_GENERATION_EXPECTED "${test_generation_expected}" \
    MIN_GENERATED_TEST_CASES "${CODEX_AI_CI_MIN_GENERATED_TEST_CASES}" \
    MAX_GENERATED_TEST_CASES "${CODEX_AI_CI_MAX_GENERATED_TEST_CASES}" \
    MAX_GENERATED_TEST_FILES "${CODEX_AI_CI_MAX_GENERATED_TEST_FILES}" \
    MAX_TEST_COMMANDS "${CODEX_AI_CI_MAX_TEST_COMMANDS}" \
    RECOMMENDED_COMMAND_TIMEOUT_SECONDS "${CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS}" \
    TEST_BUDGET_SECONDS "${CODEX_AI_CI_TEST_BUDGET_SECONDS}" \
    CODEX_TIMEOUT_SECONDS "${CODEX_AI_CI_TIMEOUT_SECONDS}" \
    REPORT_RESERVE_SECONDS "${CODEX_AI_CI_REPORT_RESERVE_SECONDS}"
)"; then
  fail_ai_ci "无法渲染 Codex AI CI 提示词模板：${selected_prompt_template}"
fi

set +e
printf '%s\n' "${prompt}" | timeout --signal=TERM --kill-after=30s \
  "${CODEX_AI_CI_TIMEOUT_SECONDS}s" \
  docker exec -i \
    --user 0 \
    --workdir "${container_checkout_dir}" \
    --env "GIT_OPTIONAL_LOCKS=0" \
    --env "HOME=/root" \
    --env "CODEX_HOME=${container_codex_home}" \
    --env "AI_ANALYSIS_MODE=${analysis_mode}" \
    --env "AI_CODEX_BIN=${container_codex_bin}" \
    --env "AI_SCHEMA_PATH=${container_schema_path}" \
    --env "AI_ANALYSIS_PATH=${container_analysis_json_path}" \
    --env "AI_JSONL_RECORDER_PATH=${container_jsonl_recorder_path}" \
    --env "AI_REASONING_EFFORT=${CODEX_AI_CI_REASONING_EFFORT}" \
    --env "AI_PYTHON_VENV_ACTIVATE=${PYTHON_VENV_ACTIVATE}" \
    --env "AI_LLVM_BUILD_DIR=${LLVM_BUILD_DIR}" \
    --env "AI_SOURCE_ENVSETUP=${SOURCE_ENVSETUP}" \
    --env "AI_ANCHOR_ENVSETUP=${container_anchor_envsetup}" \
    --env "AI_CHECKOUT_DIR=${container_checkout_dir}" \
    --env "AI_TARGET_SHA=${target_sha}" \
    --env "AI_BRANCH=${branch}" \
    --env "AI_LOCAL_CI_RUN_ID=${LOCAL_CI_RUN_ID}" \
    --env "AI_TEST_PYTHON_BIN=${CODEX_TEST_PYTHON_BIN}" \
    --env "AI_PPL_ROOT=${PPL_ROOT}" \
    --env "AI_PACKAGE_TOOL=${PACKAGE_TOOL}" \
    --env "AI_FRONTEND_BUILD_MODE=${FRONTEND_BUILD_MODE}" \
    --env "AI_BACKEND_PROFILE=${BACKEND_PROFILE}" \
    --env "AI_EXPECTED_TRITON_BACKEND=${EXPECTED_TRITON_BACKEND}" \
    --env "AI_FLAGGEMS_CLONE_DIR=${FLAGGEMS_CLONE_DIR}" \
    --env "AI_MAX_JOBS=${MAX_JOBS}" \
    --env "AI_CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}" \
    --env "AI_NINJAFLAGS=${NINJAFLAGS}" \
    --env "AI_UV_LINK_MODE=${UV_LINK_MODE}" \
    --env "AI_BACKEND_PATH=${BACKEND_PATH}" \
    --env "AI_BACKEND_ENVSETUP=${BACKEND_ENVSETUP}" \
    --env "AI_BACKEND_ENVSETUP_ARGS=${BACKEND_ENVSETUP_ARGS}" \
    --env "AI_RUN_BACKEND_STAGES=${RUN_BACKEND_STAGES}" \
    "${ephemeral_container}" \
    bash -lc '
      bootstrap_status=0
      set +u
      export TMPDIR=/tmp/triton-anchor-codex-tmp
      export TRITON_DUMP_DIR=/tmp/triton-anchor-codex-dump
      mkdir -p "${TMPDIR}" "${TRITON_DUMP_DIR}" || bootstrap_status=1
      if [[ -n "${AI_PYTHON_VENV_ACTIVATE}" && -f "${AI_PYTHON_VENV_ACTIVATE}" ]]; then
        source "${AI_PYTHON_VENV_ACTIVATE}" || bootstrap_status=1
      else
        echo "Codex AI CI 环境提示：Python venv 激活脚本不存在。" >&2
        bootstrap_status=1
      fi
      if [[ -n "${AI_LLVM_BUILD_DIR}" ]]; then
        export LLVM_BUILD_DIR="${AI_LLVM_BUILD_DIR}"
      fi
      export WORKSPACE=/workspace
      export ANCHOR_DIR="${AI_CHECKOUT_DIR}"
      export GITHUB_SHA="${AI_TARGET_SHA}"
      export GITHUB_REF="refs/heads/${AI_BRANCH}"
      export LOCAL_CI_RUN_ID="${AI_LOCAL_CI_RUN_ID}"
      export PYTHON_BIN="${AI_TEST_PYTHON_BIN}"
      export PPL_ROOT="${AI_PPL_ROOT}"
      export PACKAGE_TOOL="${AI_PACKAGE_TOOL}"
      export FRONTEND_BUILD_MODE="${AI_FRONTEND_BUILD_MODE}"
      export BACKEND_PROFILE="${AI_BACKEND_PROFILE}"
      export EXPECTED_TRITON_BACKEND="${AI_EXPECTED_TRITON_BACKEND}"
      export FLAGGEMS_CLONE_DIR="${AI_FLAGGEMS_CLONE_DIR}"
      export FLAGGEMS_ROOT="${AI_FLAGGEMS_CLONE_DIR}"
      export MAX_JOBS="${AI_MAX_JOBS}"
      export CMAKE_BUILD_PARALLEL_LEVEL="${AI_CMAKE_BUILD_PARALLEL_LEVEL}"
      export NINJAFLAGS="${AI_NINJAFLAGS}"
      export UV_LINK_MODE="${AI_UV_LINK_MODE}"
      if [[ "${AI_SOURCE_ENVSETUP}" == "1" ]]; then
        anchor_setup="${AI_ANCHOR_ENVSETUP}"
        if [[ -n "${anchor_setup}" && -f "${anchor_setup}" ]]; then
          source "${anchor_setup}" || bootstrap_status=1
        elif [[ -n "${anchor_setup}" ]]; then
          echo "Codex AI CI 环境提示：前端环境脚本不存在。" >&2
          bootstrap_status=1
        elif [[ -f "${AI_CHECKOUT_DIR}/envsetup.sh" ]]; then
          source "${AI_CHECKOUT_DIR}/envsetup.sh" || bootstrap_status=1
        fi
      fi
      if [[ "${AI_RUN_BACKEND_STAGES}" == "true" ]]; then
        backend_setup="${AI_BACKEND_ENVSETUP}"
        if [[ -n "${backend_setup}" && "${backend_setup}" != /* ]]; then
          backend_setup="${AI_BACKEND_PATH}/${backend_setup}"
        fi
        if [[ -n "${backend_setup}" && -f "${backend_setup}" ]]; then
          # shellcheck disable=SC2086
          source "${backend_setup}" ${AI_BACKEND_ENVSETUP_ARGS} || bootstrap_status=1
        elif [[ -n "${backend_setup}" ]]; then
          echo "Codex AI CI 环境提示：后端环境脚本不存在。" >&2
          bootstrap_status=1
        fi
      fi
      export TMPDIR=/tmp/triton-anchor-codex-tmp
      export TRITON_DUMP_DIR=/tmp/triton-anchor-codex-dump
      mkdir -p "${TMPDIR}" "${TRITON_DUMP_DIR}" || bootstrap_status=1
      set -u
      if [[ ${bootstrap_status} -ne 0 ]]; then
        if [[ "${AI_ANALYSIS_MODE}" == "full" ]]; then
          echo "CODEX_AI_CI_BOOTSTRAP_FAILED_BEFORE_EXEC" >&2
          echo "Codex AI CI 无法继承确定性 CI 的验证环境。" >&2
          exit 78
        fi
        export CODEX_AI_ENVIRONMENT_STATUS="incomplete"
        echo "Codex AI CI 验证环境不完整；继续执行静态失败诊断。" >&2
      else
        export CODEX_AI_ENVIRONMENT_STATUS="ready"
      fi
      unset GITEE_TOKEN GITEE_USERNAME GIT_ASKPASS
      set -o pipefail
      "${AI_CODEX_BIN}" exec \
          --ephemeral \
          --json \
          --sandbox danger-full-access \
          --ignore-rules \
          --config "model_reasoning_effort=\"${AI_REASONING_EFFORT}\"" \
          --output-schema "${AI_SCHEMA_PATH}" \
          --output-last-message "${AI_ANALYSIS_PATH}" \
          - |
        python3 "${AI_JSONL_RECORDER_PATH}" record
    ' > "${codex_jsonl_path}" 2>> "${log_path}" &
codex_exec_pid=$!
startup_deadline=$((SECONDS + 10#${CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS}))
while kill -0 "${codex_exec_pid}" 2>/dev/null; do
  if grep -Eq '"type"[[:space:]]*:[[:space:]]*"(item\.(started|completed)|turn\.completed)"' \
    "${codex_jsonl_path}"; then
    startup_progress="true"
    break
  fi
  if ((SECONDS >= startup_deadline)); then
    startup_timed_out="true"
    echo "Codex AI CI 启动 watchdog：${CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS} 秒内未出现首个有效进展事件。" \
      >> "${log_path}"
    kill -ALRM "${codex_exec_pid}" 2>/dev/null || \
      kill -TERM "${codex_exec_pid}" 2>/dev/null || true
    break
  fi
  sleep 2
done
wait "${codex_exec_pid}"
exit_code=$?
if grep -Eq '"type"[[:space:]]*:[[:space:]]*"(item\.(started|completed)|turn\.completed)"' \
  "${codex_jsonl_path}"; then
  startup_progress="true"
fi
set -e

docker exec --user 0 "${ephemeral_container}" cat "${container_analysis_json_path}" \
  > "${analysis_json_path}" 2>> "${log_path}" || true
collect_container_workspace

if [[ -s "${workspace_status_path}" ]]; then
  workspace_dirty="true"
fi
if ! "${PYTHON_BIN}" "${jsonl_evidence_path}" extract \
  --input "${codex_jsonl_path}" \
  --output "${command_ledger_path}" >> "${log_path}" 2>&1; then
  command_ledger_available="false"
  echo "Codex JSONL 命令证据提取失败。" >> "${log_path}"
else
  command_ledger_available="true"
fi
if "${PYTHON_BIN}" "${jsonl_evidence_path}" has-event \
  --input "${codex_jsonl_path}" --type "turn.completed"; then
  turn_completed="true"
fi
refresh_command_ledger_state || true
if [[ ${exit_code} -eq 0 ]]; then
  "${PYTHON_BIN}" "${report_builder_path}" build \
    --analysis "${analysis_json_path}" \
    --output "${report_json_path}" \
    --manifest "${changed_files_manifest_path}" \
    --command-ledger "${command_ledger_path}" \
    --generated-archive "${generated_files_path}" \
    --repository-root "${workspace_dir}" \
    --analysis-mode "${analysis_mode}" \
    --test-generation-expected "${test_generation_expected}" \
    >> "${log_path}" 2>&1 || true
  if grep -Fq "CODEX_AI_CI_COMPLETE" "${report_json_path}"; then
    marker_found="true"
  fi
  execution_metadata_available="false"
  if load_execution_metadata; then
    execution_metadata_available="true"
  fi

  renderer_args=(
    --input "${report_json_path}"
    --output "${report_path}"
    --comment-output "${comment_path}"
    --branch "${branch}"
    --base-sha "${base_sha}"
    --requested-base-sha "${requested_base_sha}"
    --diff-mode "${diff_mode}"
    --target-sha "${target_sha}"
    --head-sha "${requested_head_sha}"
    --tested-sha-kind "$([[ "${branch}" =~ ^ci/pr-[0-9]+/.+ ]] && printf '%s' pr_merge || printf '%s' commit)"
    --local-ci-status "${local_ci_status}"
    --local-ci-execution-mode "${LOCAL_CI_EXECUTION_MODE}"
    --backend-validation-scope "${backend_validation_scope}"
    --changed-file-count "${changed_file_count}"
    --changed-files-manifest "${changed_files_manifest_path}"
    --repository-root "${workspace_dir}"
    --constraint-status "${constraint_status}"
    --constraint-reason "${constraint_reason}"
  )
  if report_verdict="$(
    "${PYTHON_BIN}" "${renderer_path}" "${renderer_args[@]}" 2>> "${log_path}"
  )"; then
    if [[ "${execution_metadata_available}" == "true" ]]; then
      report_format_valid="true"
    else
      report_verdict="UNKNOWN"
    fi
  else
    report_verdict="UNKNOWN"
  fi
fi

if [[ "${startup_timed_out}" == "true" ]]; then
  set_failure_reason "Codex 启动阶段超过 ${CODEX_AI_CI_STARTUP_TIMEOUT_SECONDS} 秒仍未出现首个有效进展"
elif [[ ${exit_code} -eq 124 || ${exit_code} -eq 137 ]]; then
  set_failure_reason "Codex 执行超过 ${CODEX_AI_CI_TIMEOUT_SECONDS} 秒硬超时"
elif [[ ${exit_code} -eq 78 ]] && \
  grep -Fq "CODEX_AI_CI_BOOTSTRAP_FAILED_BEFORE_EXEC" "${log_path}"; then
  failure_code="container_setup_failed"
  set_failure_reason "Codex 无法继承确定性 CI 的验证环境"
elif [[ ${exit_code} -ne 0 ]]; then
  set_failure_reason "Codex exec 异常退出，退出码为 ${exit_code}"
elif [[ "${report_format_valid}" != "true" ]]; then
  builder_failure_tail="$(grep -F "Invalid Codex AI analysis:" "${log_path}" | tail -n 1 || true)"
  trusted_input_failure_tail="$(grep -F "Invalid Codex AI trusted input:" "${log_path}" | tail -n 1 || true)"
  renderer_failure_tail="$(grep -F "Invalid Codex AI report:" "${log_path}" | tail -n 1 || true)"
  if [[ -n "${trusted_input_failure_tail}" ]]; then
    failure_code="trusted_report_input_failed"
    set_failure_reason "可信报告输入校验失败：${trusted_input_failure_tail}"
  elif [[ -n "${builder_failure_tail}" ]]; then
    failure_code="analysis_contract_failed"
    set_failure_reason "Codex 语义载荷未满足公开结构契约：${builder_failure_tail}"
  elif [[ -n "${renderer_failure_tail}" ]]; then
    failure_code="report_contract_failed"
    set_failure_reason "Runner 生成的内部报告契约校验失败：${renderer_failure_tail}"
  else
    failure_code="report_metadata_failed"
    set_failure_reason "Runner 无法读取结构化报告的执行事实元数据"
  fi
elif [[ "${marker_found}" != "true" ]]; then
  set_failure_reason "runner 生成的结构化报告缺少 CODEX_AI_CI_COMPLETE 标记"
elif [[ "${turn_completed}" != "true" ]]; then
  set_failure_reason "Codex JSONL 日志中没有 turn.completed 事件"
elif [[ "${command_executed}" != "true" ]]; then
  set_failure_reason "Codex 没有执行任何用于检查代码或日志的命令"
else
  status="pass"
fi

verify_credential_integrity
if [[ "${status}" != "pass" ]]; then
  write_failure_report
fi
append_change_request_context_warning
append_credential_integrity_warning
limit_public_comment
write_summary
if [[ "${status}" == "pass" ]]; then
  echo "Codex AI CI：完成（结论 ${report_verdict}；测试状态 ${test_execution_status}；约束 ${constraint_status}；报告 ${report_path}）"
  exit 0
fi

echo "Codex AI CI：失败（${failure_code:-codex_execution_failed}：${failure_reason}）"
exit 1
