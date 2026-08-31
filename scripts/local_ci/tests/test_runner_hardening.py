from pathlib import Path
import os
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "deterministic_ci/run_deterministic_ci.sh").read_text()
POLLER = (ROOT / "poll_gitee_and_run.sh").read_text()
ORCHESTRATOR = (
    ROOT / "orchestration/run_deterministic_ci_in_container.sh"
).read_text()
CODEX_RUNNER = (ROOT / "codex_ai/run_codex_ai_ci.sh").read_text()
CODEX_FAILURE_PROMPT = (ROOT / "codex_ai/prompts/codex_ai_failure.md").read_text()


def test_pr_sources_only_base_envsetup_from_runner_snapshot() -> None:
    assert 'TRUSTED_ANCHOR_ENVSETUP="${TRUSTED_ANCHOR_ENVSETUP:-${RUNNER_ROOT}/trusted/envsetup.sh}"' in RUNNER
    pr_block = RUNNER[RUNNER.index('if [[ -n "${LOCAL_CI_BASE_SHA}" ]]') :]
    pr_block = pr_block[: pr_block.index("elif")]
    assert 'bash -n "${ANCHOR_DIR}/envsetup.sh"' in pr_block
    assert 'source "${ANCHOR_DIR}/envsetup.sh"' not in pr_block
    assert 'envsetup_file="${TRUSTED_ANCHOR_ENVSETUP}"' in pr_block
    assert 'source "${TRUSTED_ANCHOR_ENVSETUP}"' not in RUNNER
    assert 'source "${envsetup_file}"' in RUNNER
    assert 'git -C "${checkout_dir}" show "${base_sha}:envsetup.sh"' in POLLER
    assert 'TRUSTED_ANCHOR_ENVSETUP="${TRUSTED_ANCHOR_ENVSETUP:-${LOCAL_CI_ROOT}/trusted/envsetup.sh}"' in CODEX_RUNNER
    assert '"copy_trusted_envsetup" docker cp' in CODEX_RUNNER
    assert 'anchor_setup="${AI_ANCHOR_ENVSETUP}"' in CODEX_RUNNER


def test_candidate_exit_zero_cannot_override_required_stage_failure() -> None:
    assert "required_stages_passed()" in RUNNER
    for status in (
        "FRONTEND_BUILD_STATUS",
        "FRONTEND_SMOKE_STATUS",
        "BACKEND_REBUILD_STATUS",
        "BACKEND_SMOKE_JIT_STATUS",
    ):
        assert status in RUNNER
    assert "forcing overall failure" in RUNNER


def test_frontend_only_profile_skips_every_backend_dependent_stage() -> None:
    assert 'if [[ "${RUN_BACKEND_STAGES}" == "true" ]]; then' in RUNNER
    assert 'BACKEND_REBUILD_STATUS="skipped"' in RUNNER
    assert 'BACKEND_SMOKE_JIT_STATUS="skipped"' in RUNNER
    assert 'FLAGGEMS_STATUS="skipped"' in RUNNER
    assert 'COMPILE_TIME_STATUS="skipped"' in RUNNER
    assert 'PASS_PROFILE_STATUS="skipped"' in RUNNER
    assert 'IR_SERIALIZATION_STATUS="skipped"' in RUNNER
    assert 'required_statuses=(' in RUNNER
    assert 'required_statuses+=(' in RUNNER


def test_backend_environment_starts_after_frontend_validation() -> None:
    main_start = RUNNER.index("Local CI commit: ${target_sha}")
    verify_frontend = RUNNER.index(
        "run_logged verify-triton-anchor-import", main_start
    )
    frontend_smoke = RUNNER.index(
        'FRONTEND_SMOKE_STATUS "Frontend smoke"', verify_frontend
    )
    prepare_backend = RUNNER.index("source_backend_env\n", frontend_smoke)
    rebuild_backend = RUNNER.index("rebuild_backend\n", prepare_backend)
    refresh_backend = RUNNER.index("source_backend_env\n", rebuild_backend)
    verify_backend = RUNNER.index(
        "run_logged verify-backend-discovery", refresh_backend
    )

    assert (
        verify_frontend
        < frontend_smoke
        < prepare_backend
        < rebuild_backend
        < refresh_backend
        < verify_backend
    )


def test_frontend_only_empty_backend_path_passes_checkout_overlap_guard(
    tmp_path: Path,
) -> None:
    start = RUNNER.index("validated_anchor_checkout_path() {")
    end = RUNNER.index("\n}\n\nfresh_checkout_anchor()", start) + 3
    function = RUNNER[start:end]
    workspace = tmp_path / "workspace"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "ANCHOR_DIR": str(workspace / "triton-anchor"),
            "BACKEND_PATH": "",
            "FLAGGEMS_CLONE_DIR": str(workspace / "FlagGems"),
            "LLVM_BUILD_DIR": str(workspace / "llvm"),
            "PPL_ROOT": str(workspace / "ppl"),
            "LOCAL_CI_ARTIFACT_ROOT": str(workspace / "artifacts"),
        }
    )
    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{function}\nvalidated_anchor_checkout_path"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(workspace / "triton-anchor")


def test_codex_budget_defaults_are_consistent_at_the_poller_boundary() -> None:
    assert 'CODEX_AI_CI_MAX_TEST_COMMANDS="${CODEX_AI_CI_MAX_TEST_COMMANDS:-50}"' in POLLER
    assert (
        'CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS="${CODEX_AI_CI_RECOMMENDED_COMMAND_TIMEOUT_SECONDS:-900}"'
        in POLLER
    )
    assert 'CODEX_AI_CI_TIMEOUT_SECONDS="${CODEX_AI_CI_TIMEOUT_SECONDS:-3600}"' in POLLER
    assert 'CODEX_AI_CI_TEST_BUDGET_SECONDS="${CODEX_AI_CI_TEST_BUDGET_SECONDS:-2700}"' in POLLER
    assert 'CODEX_AI_CI_REPORT_RESERVE_SECONDS="${CODEX_AI_CI_REPORT_RESERVE_SECONDS:-450}"' in POLLER


def test_retention_governance_defaults_are_wired() -> None:
    for expected in (
        'LOCAL_CI_MAINTENANCE_INTERVAL_SECONDS="${LOCAL_CI_MAINTENANCE_INTERVAL_SECONDS:-86400}"',
        'LOCAL_CI_SUCCESS_RETENTION_DAYS="${LOCAL_CI_SUCCESS_RETENTION_DAYS:-14}"',
        'LOCAL_CI_FAILURE_RETENTION_DAYS="${LOCAL_CI_FAILURE_RETENTION_DAYS:-28}"',
    ):
        assert expected in POLLER
    assert 'run_maintenance_if_due || echo "Local CI maintenance failed; polling will continue."' in POLLER
def test_codex_ephemeral_container_has_ownership_labels() -> None:
    for expected in (
        '--label "triton-anchor.run-id=${LOCAL_CI_RUN_ID}"',
        'LABEL triton-anchor.role=codex-ai-snapshot',
    ):
        assert expected in CODEX_RUNNER


def test_poller_uses_base_hash_for_pr_and_rejects_candidate_hash_change() -> None:
    assert (
        'selected_hash="$(read_verified_llvm_hash "${base_branch}" "${base_sha}")"'
        in POLLER
    )
    assert (
        'candidate_hash="$(read_verified_llvm_hash "${task_branch}" "${task_sha}")"'
        in POLLER
    )
    assert 'if [[ "${candidate_hash}" != "${selected_hash}" ]]' in POLLER
    assert "The tested PR changes triton/cmake/llvm-hash.txt" in POLLER


def test_push_profile_uses_tested_hash_and_unknown_hash_cannot_fall_back() -> None:
    assert (
        'selected_hash="$(read_verified_llvm_hash "${task_branch}" "${task_sha}")"'
        in POLLER
    )
    assert '--profile-dir "${configured_profile_dir}"' in POLLER
    assert "No trusted Local CI profile is available" in POLLER
    assert 'run_once() (' in POLLER


def test_profile_capability_is_passed_to_container_and_codex() -> None:
    for variable in (
        "LOCAL_CI_PROFILE_NAME",
        "LOCAL_CI_LLVM_HASH",
        "RUN_BACKEND_STAGES",
        "BACKEND_SKIP_REASON",
    ):
        assert f'{variable}="${{{variable}}}"' in POLLER
    for field in (
        "ci_profile: ${LOCAL_CI_PROFILE_NAME}",
        "llvm_hash: ${LOCAL_CI_LLVM_HASH}",
        "backend_stages_enabled: ${RUN_BACKEND_STAGES}",
        "backend_skip_reason: ${BACKEND_SKIP_REASON}",
    ):
        assert field in RUNNER
    assert 'LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-}" \\' in POLLER
    assert 'export LLVM_BUILD_DIR="${AI_LLVM_BUILD_DIR}"' in CODEX_RUNNER
    assert 'PPL_ROOT="${PPL_ROOT:-}" \\' in POLLER
    assert 'export PPL_ROOT="${AI_PPL_ROOT}"' in CODEX_RUNNER
    assert 'export LOCAL_CI_RUN_ID="${AI_LOCAL_CI_RUN_ID}"' in CODEX_RUNNER
    assert 'export TMPDIR=/tmp/triton-anchor-codex-tmp' in CODEX_RUNNER
    assert 'if [[ "${AI_ANALYSIS_MODE}" == "full" ]]' in CODEX_RUNNER
    assert "CODEX_AI_ENVIRONMENT_STATUS" in CODEX_FAILURE_PROMPT
    assert 'LOCAL_CI_EXECUTION_MODE="${LOCAL_CI_EXECUTION_MODE:-full}"' in POLLER


def test_backend_enabled_profile_cannot_fall_back_to_sophgo_values() -> None:
    for variable in (
        "BACKEND_PROFILE",
        "EXPECTED_TRITON_BACKEND",
        "BACKEND_PATH",
        "BACKEND_TEST_COMMAND",
        "BACKEND_WHEEL_PATTERN",
    ):
        assert f'{variable}=""' in POLLER
    assert "requires ${required_backend_value} from the selected profile" in RUNNER
    assert 'EXPECTED_TRITON_BACKEND:-sophgo' not in RUNNER
    assert 'BACKEND_PATH:-${WORKSPACE}/triton-sophgo-backend' not in RUNNER
    assert 'find dist -maxdepth 1 -type f -name "${BACKEND_WHEEL_PATTERN}"' in RUNNER
    assert "BACKEND_PROFILE EXPECTED_TRITON_BACKEND BACKEND_PATH" in POLLER

    profile_source = ORCHESTRATOR.index('source "${resolved_profile}"')
    global_config_source = ORCHESTRATOR.index('source "${CONFIG_FILE}"')
    for variable in (
        "BACKEND_PROFILE",
        "EXPECTED_TRITON_BACKEND",
        "BACKEND_PATH",
        "BACKEND_ENVSETUP",
        "BACKEND_ENVSETUP_ARGS",
        "BACKEND_TEST_COMMAND",
        "BACKEND_UNINSTALL_PACKAGES",
        "BACKEND_WHEEL_PATTERN",
    ):
        cleared = ORCHESTRATOR.index(f'{variable}=""', global_config_source)
        assert cleared < profile_source

    assert "/workspace/triton-sophgo-backend" not in CODEX_RUNNER


def test_manual_fallback_preserves_mixed_command_ledger_facts(tmp_path: Path) -> None:
    start = CODEX_RUNNER.index("refresh_command_ledger_state() {")
    end = CODEX_RUNNER.index("write_fallback_changed_files_table() {", start)
    functions = CODEX_RUNNER[start:end]
    ledger = tmp_path / "ledger.json"
    log = tmp_path / "runner.log"
    ledger.write_text(
        '[{"command":"python3 -m pytest ok.py","exit_code":0,"duration_seconds":1.25},'
        '{"command":"python3 -m pytest fail.py","exit_code":2,"duration_seconds":2.5}]',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": "python3",
            "command_ledger_path": str(ledger),
            "command_ledger_available": "true",
            "command_executed": "false",
            "test_execution_status": "unavailable",
            "test_command_count": "UNKNOWN",
            "max_test_command_duration_seconds": "UNKNOWN",
            "total_test_command_duration_seconds": "UNKNOWN",
            "log_path": str(log),
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f"set -uo pipefail\n{functions}\n"
                "refresh_command_ledger_state\n"
                "fallback_command_ledger_fact\n"
                "write_fallback_command_ledger_table\n"
                'printf "STATE=%s COUNT=%s MAX=%s TOTAL=%s\\n" '
                '"${test_execution_status}" "${test_command_count}" '
                '"${max_test_command_duration_seconds}" '
                '"${total_test_command_duration_seconds}"'
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "已保留 2 条验证或诊断命令记录：1 条成功、1 条失败" in result.stdout
    assert "| `python3 -m pytest ok.py` | 0 | 1.250 |" in result.stdout
    assert "| `python3 -m pytest fail.py` | 2 | 2.500 |" in result.stdout
    assert "STATE=insufficient_evidence COUNT=2 MAX=2.500 TOTAL=3.750" in result.stdout


def test_manual_fallback_comment_keeps_required_sections_under_length_limit(
    tmp_path: Path,
) -> None:
    start = CODEX_RUNNER.index("limit_public_comment() {")
    end = CODEX_RUNNER.index("\n}\n\nwrite_failure_report()", start) + 3
    function = CODEX_RUNNER[start:end]
    comment = tmp_path / "comment.md"
    comment.write_text(
        "## Codex AI 自动审查\n"
        + ("审查说明。" * 2_000)
        + "\n### 验证情况\n\n- 验证内容与结果：\n  - 已保留命令记录。\n"
        + "- 限制与未覆盖：\n  - 审查未完成。\n"
        + "\n### 剩余风险\n\n- 仍需人工核对。\n"
        + "\n### 变更文件\n\n"
        + "<details>\n<summary>展开文件级变更表</summary>\n\n"
        + ("| `very-long-file.py` | 修改 | 说明 | 影响 |\n" * 2_000)
        + "\n</details>\n\n"
        + "### PR 功能声明上下文警告\n\n"
        + "未取得与当前 PR 测试提交匹配的功能声明元数据。\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update({"PYTHON_BIN": "python3", "comment_path": str(comment)})
    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{function}\nlimit_public_comment"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    limited = comment.read_text(encoding="utf-8")
    assert len(limited) <= 58_000
    for heading in (
        "### 验证情况",
        "### 剩余风险",
        "### 变更文件",
        "### PR 功能声明上下文警告",
    ):
        assert heading in limited


def test_pr_without_trusted_envsetup_fails_closed() -> None:
    assert "Trusted base commit has no envsetup.sh" in POLLER
    assert "Trusted base envsetup.sh is required for PR Local CI" in RUNNER
    assert "PR Local CI requires SOURCE_ENVSETUP=1" in RUNNER


def test_runner_requires_the_actual_task_ref() -> None:
    env = os.environ.copy()
    env.pop("GITEE_BRANCH", None)
    env["LOCAL_CI_SCRIPT_STAGED"] = "1"
    result = subprocess.run(
        ["bash", str(ROOT / "deterministic_ci/run_deterministic_ci.sh"), "a" * 40],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "GITEE_BRANCH must be provided by the Local CI poller" in result.stderr


def test_single_branch_poller_requires_an_explicit_branch_list() -> None:
    with tempfile.TemporaryDirectory() as directory:
        env = os.environ.copy()
        env.update(
            {
                "LOCAL_CI_STATE_DIR": directory,
                "LOCAL_CI_ONCE": "1",
                "GITEE_POLL_ALL_BRANCHES": "0",
                "GITEE_BRANCHES": "",
                "LOCAL_CI_CONFIG": str(Path(directory) / "missing.env"),
            }
        )
        result = subprocess.run(
            ["bash", str(ROOT / "poll_gitee_and_run.sh")],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    assert result.returncode != 0
    assert "GITEE_BRANCHES is required" in result.stderr
