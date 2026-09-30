"""Retry diagnosis must reflect evidence, not conversational activity."""

import json

import pytest

from agent_ci.protocol import atomic_json
from agent_ci.recovery import (
    completed_tools, diagnose_report, recovery_prompt, retry_observation, write_handoff,
)


def test_diagnostics_identify_missing_records_and_parameters(tmp_path):
    path = tmp_path / "agent-result.json"
    policy = {"required_checks": ["change_validation", "flaggems"],
              "required_reviews": ["architecture"],
              "required_parameters": {"flaggems": {"mode": "full"}}}
    atomic_json(path, {"status": "pass", "checks": [
        {"tool_id": "flaggems", "status": "pass", "parameters": {"mode": "impact"}},
    ], "reviews": []})
    result = diagnose_report(path, policy)
    assert result["report"] is None
    assert {issue["field"] for issue in result["issues"]} == {
        "checks.change_validation", "reviews.architecture", "checks.flaggems.parameters",
    }
    prompt = recovery_prompt(result)
    assert "change_validation" in prompt and "architecture" in prompt and '"full"' in prompt
    assert "不能只修改报告" in prompt


@pytest.mark.parametrize("document,field", [
    ([], "agent-result.json"),
    ({"status": []}, "status"),
    ({"status": "pass", "checks": {}, "reviews": []}, "checks"),
    ({"status": "pass", "checks": [{"tool_id": "a", "status": "pass"}] * 2, "reviews": []}, "checks"),
    ({"status": "pass", "checks": [], "reviews": [{"kind": "architecture"}]}, "reviews"),
])
def test_malformed_reports_return_diagnostics_not_exceptions(tmp_path, document, field):
    path = tmp_path / "agent-result.json"
    atomic_json(path, document)
    result = diagnose_report(path, {})
    assert result["report"] is None
    assert field in {issue["field"] for issue in result["issues"]}


def test_missing_invalid_and_oversized_reports(tmp_path):
    path = tmp_path / "agent-result.json"
    assert diagnose_report(path, {})["issues"][0]["code"] == "report_missing"
    for data in ("{broken", " " * (2 * 1024 * 1024 + 1)):
        path.write_text(data)
        assert diagnose_report(path, {})["issues"][0]["code"] == "report_unreadable"


@pytest.mark.parametrize("status", ["fail", "infra_error", "cancelled"])
def test_terminal_report_does_not_require_more_validation(tmp_path, status):
    path = tmp_path / "agent-result.json"
    report = {"status": status, "checks": [], "reviews": []}
    atomic_json(path, report)
    assert diagnose_report(path, {"required_checks": ["frontend_build"]})["report"] == report


def test_check_failure_is_final_even_with_pass_top_level(tmp_path):
    path = tmp_path / "agent-result.json"
    report = {"status": "pass", "checks": [{"tool_id": "actual", "status": "fail"}], "reviews": []}
    atomic_json(path, report)
    assert diagnose_report(path, {"required_reviews": ["architecture"]})["report"] == report


def test_summary_churn_and_oscillating_gaps_do_not_reset_progress(tmp_path):
    path = tmp_path / "agent-result.json"
    policy = {"required_checks": ["a", "b", "c"]}
    previous = None
    for names, expected in ((["a"], True), (["a"], False), (["b"], True), (["a"], False), (["b"], False)):
        atomic_json(path, {"status": "pass", "summary": str(previous), "checks": [
            {"tool_id": name, "status": "pass"} for name in names
        ], "reviews": []})
        previous, progressed = retry_observation(diagnose_report(path, policy), [], previous)
        assert progressed is expected


def test_completed_tools_require_current_source_and_environment(tmp_path):
    environment = {"variants": {"candidate": {"source_sha": "a" * 40, "environment_fingerprint": "env-a"}}}
    path = tmp_path / "candidate/frontend_build/result.json"
    row = {"tool_id": "frontend_build", "status": "pass", "target_sha": "a" * 40,
           "environment_fingerprint": "env-a", "parameters": {"jobs": 2}}
    atomic_json(path, row)
    first = completed_tools(tmp_path, environment)
    assert len(first) == 1
    atomic_json(path, {**row, "duration_seconds": 77, "summary": "rewritten"})
    assert completed_tools(tmp_path, environment) == first
    for change in ({"target_sha": "b" * 40}, {"environment_fingerprint": "env-b"}, {"status": "ready"}):
        atomic_json(path, {**row, **change})
        assert completed_tools(tmp_path, environment) == []


def test_diagnostic_reader_rejects_symlink_and_special_files(tmp_path):
    import os
    target = tmp_path / "private.json"
    target.write_text('{"secret":"do not read"}')
    path = tmp_path / "agent-result.json"
    path.symlink_to(target)
    assert diagnose_report(path, {})["issues"][0]["code"] == "report_unreadable"
    path.unlink()
    os.mkfifo(path)
    assert diagnose_report(path, {})["issues"][0]["code"] == "report_unreadable"
    path.unlink()
    path.symlink_to(path)
    assert diagnose_report(path, {})["issues"][0]["code"] == "report_unreadable"


def test_handoff_copies_evidence_and_notes_but_never_session_or_install_state(tmp_path):
    old, current = tmp_path / "old", tmp_path / "new"
    artifacts = old / "artifacts"
    artifacts.mkdir(parents=True)
    (current / "artifacts").mkdir(parents=True)
    atomic_json(artifacts / "agent-result.json", {"status": "pass", "checks": [
        {"tool_id": "change_validation", "evidence": ["ai_custom_tools/test_case.py", "missing.log"]},
    ]})
    (artifacts / "ai_custom_tools").mkdir()
    (artifacts / "ai_custom_tools/test_case.py").write_text("assert 1 + 1 == 2\n")
    (artifacts / "recovery-notes.md").write_text("Fixed dependencies; fixture-secret; next: backend smoke")
    (old / "codex-session.json").write_text('{"session_id":"private-session"}')
    environment = {"variants": {"candidate": {"source_sha": "a" * 40, "environment_fingerprint": "old-env"}}}
    tool = {"tool_id": "frontend_install", "status": "pass", "target_sha": "a" * 40,
            "environment_fingerprint": "old-env"}
    atomic_json(artifacts / "candidate/frontend_install/result.json", tool)
    atomic_json(artifacts / "candidate/frontend_install/installation.json", {"task_venv": "/task/candidate/venv"})
    result = write_handoff(current, {"task_id": "frozen"}, {}, {},
                           [(old, {"checkpoint": {"environment": environment}})],
                           lambda value: value.replace("fixture-secret", "[redacted]"))
    assert result["environment_rebuilt"] is True and result["current_results"] == []
    prior = result["previous_runs"][0]
    assert prior["installation_valid"] is False
    assert "recovery/old/candidate/frontend_install/result.json" in prior["files"]
    assert "recovery/old/ai_custom_tools/test_case.py" in prior["files"]
    assert "missing.log" in prior["omitted"]
    assert not (current / "artifacts/agent-result.json").exists()
    assert not (current / "codex-session.json").exists()
    assert not (current / "artifacts/candidate/frontend_install/installation.json").exists()
    notes = (current / "artifacts/recovery/old/recovery-notes.md").read_text()
    assert "[redacted]" in notes and "fixture-secret" not in notes


def test_handoff_bounds_and_unsafe_paths_are_reported_without_aborting(tmp_path, monkeypatch):
    from agent_ci import recovery
    old, current = tmp_path / "old", tmp_path / "new"
    for directory in (old, current):
        (directory / "artifacts").mkdir(parents=True)
    (old / "artifacts/recovery-notes.md").write_text("repair notes")
    (old / "artifacts/agent-result.json").write_text("{}")
    monkeypatch.setattr(recovery, "MAX_REQUIRED_FILES", 1)
    prior = write_handoff(current, {"task_id": "t"}, {}, {}, [(old, {})], str)["previous_runs"][0]
    assert len(prior["files"]) == 1 and "agent-result.json" in prior["omitted"]
    (old / "artifacts/recovery-notes.md").unlink()
    secret = tmp_path / "unrelated"
    secret.write_text("private")
    (old / "artifacts/recovery-notes.md").symlink_to(secret)
    prior = write_handoff(current, {"task_id": "t"}, {}, {}, [(old, {})], str)["previous_runs"][0]
    assert "recovery-notes.md" in prior["omitted"]
    assert not any(path.endswith("recovery-notes.md") for path in prior["files"])


def test_same_environment_handoff_keeps_current_evidence_separate(tmp_path):
    (tmp_path / "artifacts").mkdir()
    observation = {"issues": [{"code": "record_missing", "field": "checks.change_validation", "message": "missing"}]}
    result = write_handoff(tmp_path, {"task_id": "t"}, {}, observation, [], str)
    assert result["environment_rebuilt"] is False
    assert result["process_state"] == "inspect_before_retry"
    assert result["previous_runs"] == [] and result["issues"] == observation["issues"]
    missing = write_handoff(tmp_path, {"task_id": "t"}, {}, observation, [], str, rebuilt=True)
    assert missing["environment_rebuilt"] is True and missing["history_available"] is False
