"""Result sealing and retrying Git publication against a real local bare remote."""

import json
import subprocess
from unittest.mock import patch

import pytest

from agent_ci.delivery import MAX_FILE_BYTES, MAX_OPTIONAL_FILES, MAX_REQUIRED_FILES, seal_result
from agent_ci.protocol import (
    ContractError,
    TASK_SCHEMA,
    metadata_digest,
    result_task_prefix,
    task_id,
)
from agent_ci.relay import GitRelay


def task(event_kind="pull_request"):
    value = {
        "schema": TASK_SCHEMA,
        "repository": "RACE-org/triton-anchor",
        "event_kind": event_kind,
        "pr_number": 7 if event_kind == "pull_request" else 0,
        "target_branch": "main",
        "tested_sha": "a" * 40,
        "base_sha": "b" * 40,
        "head_sha": "c" * 40,
        "worker_revision_sha": "d" * 40,
        "llvm_hash": "e" * 40,
        "full": False,
        "draft": False,
        "state": "open",
        "title": "Fix behavior",
        "description": "Implementation and validation",
        "labels": [],
        "captured_at": "2026-09-11T00:00:00Z",
    }
    value["metadata_digest"] = metadata_digest(value)
    value["task_id"] = task_id(value)
    prefix = f"ci/pr-7/{value['task_id']}" if value["pr_number"] else f"ci/branch/{value['task_id']}"
    value.update(
        task_ref=prefix + "/tested", base_task_ref=prefix + "/base",
        head_task_ref=prefix + "/head",
    )
    return value


def answer():
    return {
        "status": "pass",
        "summary": "Validated behavior",
        "checks": [{
                "tool_id": "frontend_tests", "status": "pass",
                "summary": "Tests passed", "evidence": [],
            }],
        "reviews": [
            {"kind": "pr_info", "status": "pass", "summary": "Clear"},
            {
                "kind": "architecture", "status": "pass",
                "summary": "Compatible", "evidence": ["src/file.py:12"],
            },
        ],
    }


def seal(tmp_path, value, event_kind="pull_request"):
    return seal_result(
        task(event_kind), "20260911-run", value, {"required_checks": ["frontend_tests"]},
        {"profile": "test"}, tmp_path / "run", tmp_path / "sealed",
    )


def performance_environment():
    return {"variants": {"candidate": {
        "profile": "triton-3.0",
        "llvm_hash": "e" * 40,
        "backend_enabled": True,
        "backend_profile": "sophgo-cmodel",
        "environment_fingerprint": "performance-environment",
    }}}


def backend_smoke_result(tmp_path, status="pass", **changes):
    value = {
        "tool_id": "backend_smoke", "status": status, "exit_code": 0 if status == "pass" else 1,
        "target_sha": task()["tested_sha"], "variant": "candidate",
        "llvm_hash": "e" * 40, "environment_fingerprint": "performance-environment",
        "details": {"smoke_success": {
            "tool": "backend_smoke", "status": "passed",
            "task_id": task()["task_id"], "target_sha": task()["tested_sha"],
        }} if status == "pass" else {},
        **changes,
    }
    path = tmp_path / "run/artifacts/candidate/backend_smoke/result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def seal_backend(tmp_path, value, **kwargs):
    return seal_result(
        task(), "20260911-backend", value, {"required_checks": ["frontend_tests"]},
        performance_environment(), tmp_path / "run", tmp_path / "sealed", **kwargs,
    )


@pytest.mark.parametrize("status", ["pass", "fail", "infra_error", "cancelled"])
def test_backend_smoke_is_collected_without_an_agent_named_check(tmp_path, status):
    backend_smoke_result(tmp_path, status)
    value = answer()
    value["checks"].append({"tool_id": "custom_jit_validation", "status": "pass"})
    result = seal_backend(tmp_path, value)
    checks = {row["tool_id"]: row for row in result["checks"]}
    assert checks["backend_smoke"]["status"] == status
    assert checks["custom_jit_validation"]["status"] == "pass"
    assert checks["backend_smoke"]["evidence"] == ["candidate/backend_smoke/result.json"]
    assert (tmp_path / "sealed/artifacts/candidate/backend_smoke/result.json").is_file()
    assert result["status"] == ("infra_error" if status == "cancelled" else status)


@pytest.mark.parametrize("changes", [
    {"target_sha": "b" * 40}, {"variant": "base"}, {"llvm_hash": "f" * 40},
    {"environment_fingerprint": "old-environment"}, {"tool_id": "backend_build"},
    {"status": "unknown"}, {"exit_code": 1}, {"details": {}},
    {"details": {"smoke_success": {"tool": "backend_smoke", "status": "passed",
                                  "task_id": "another-task", "target_sha": "a" * 40}}},
])
def test_backend_smoke_rejects_mismatched_or_incomplete_runner_evidence(tmp_path, changes):
    backend_smoke_result(tmp_path, **changes)
    with pytest.raises(ContractError, match="backend_smoke"):
        seal_backend(tmp_path, answer())


def test_backend_smoke_preserves_final_revalidation_and_does_not_invent_results(tmp_path):
    assert all(row["tool_id"] != "backend_smoke" for row in seal_backend(tmp_path, answer())["checks"])
    path = backend_smoke_result(tmp_path)
    base = tmp_path / "run/artifacts/base/backend_smoke/result.json"
    base.parent.mkdir(parents=True)
    path.rename(base)
    assert all(row["tool_id"] != "backend_smoke" for row in seal_backend(tmp_path, answer())["checks"])
    backend_smoke_result(tmp_path, "infra_error")
    value = answer()
    value["checks"].append({"tool_id": "backend_smoke", "status": "pass",
                            "summary": "Revalidated after repair", "evidence": []})
    result = seal_backend(tmp_path, value)
    smoke = [row for row in result["checks"] if row["tool_id"] == "backend_smoke"]
    assert len(smoke) == 1
    assert smoke[0]["status"] == result["status"] == "pass"
    assert smoke[0]["summary"] == "Revalidated after repair"
    assert all(row["tool_id"] != "backend_smoke" for row in
               seal_backend(tmp_path, answer(), collect_business=False)["checks"])
    # A front-end-only environment does not acquire a backend result from leftovers.
    assert all(row["tool_id"] != "backend_smoke" for row in seal(tmp_path, answer())["checks"])


def performance_runner_result(tool_id):
    kernels = ["add", "mm", "softmax", "layernorm"]
    metrics = ["serialize", "write_text", "read_text", "deserialize", "roundtrip"]
    repeat = 20 if tool_id == "ir_serialization" else 3
    metadata = {
        "commit_sha": "a" * 40,
        "environment_fingerprint": "performance-environment",
        "profile_id": "triton-3.0",
        "llvm_revision": "e" * 40,
        "kernels": kernels,
        "repeat": repeat,
        "warmup": 1,
    }
    if tool_id == "compile_time":
        summary = {kernel: {
            "all_correct": True,
            "compile_est": {"median_ms": 10.0, "count": 3},
        } for kernel in kernels}
    elif tool_id == "pass_profile":
        summary = {kernel: {"passes": {
            "Canonicalizer": {"wall_ms": {"median_ms": 2.0, "count": 3}},
        }} for kernel in kernels}
    else:
        summary = {kernel: {"metrics": {
            metric: {"median_ms": float(index + 1), "count": 20}
            for index, metric in enumerate(metrics)
        }} for kernel in kernels}
    return {
        "tool_id": tool_id,
        "status": "pass",
        "parameters": {},
        "target_sha": "a" * 40,
        "variant": "candidate",
        "llvm_hash": "e" * 40,
        "environment_fingerprint": "performance-environment",
        "details": {
            "candidate": {"metadata": metadata, "summary": summary},
            "comparison": {"status": "not_comparable", "reason": "baseline_missing"},
        },
    }


def test_standard_performance_is_sealed_from_trusted_runner_results(tmp_path):
    run = tmp_path / "run/artifacts/candidate"
    for tool_id in ("compile_time", "pass_profile", "ir_serialization"):
        directory = run / tool_id
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(
            json.dumps(performance_runner_result(tool_id))
        )
    value = answer()
    value["checks"].append({
        "tool_id": "compile_time",
        "status": "fail",
        "summary": "Agent summary remains human-authored",
        "parameters": {"kernels": ["mm"]},
        "details": {"candidate": {"summary": {"spoofed": {}}}},
    })
    result = seal_result(
        task(), "20260911-performance", value,
        {"required_checks": ["frontend_tests"]}, performance_environment(),
        tmp_path / "run", tmp_path / "sealed",
    )
    checks = {check["tool_id"]: check for check in result["checks"]}
    assert set(checks) >= {
        "compile_time", "pass_profile", "ir_serialization",
    }
    for tool_id, repeat in (
        ("compile_time", 3), ("pass_profile", 3), ("ir_serialization", 20),
    ):
        check = checks[tool_id]
        assert check["status"] == "pass"
        assert check["parameters"] == {
            "kernels": ["add", "mm", "softmax", "layernorm"],
            "repeat": repeat,
            "warmup": 1,
        }
        assert set(check["details"]["candidate"]["summary"]) == {
            "add", "mm", "softmax", "layernorm",
        }
    assert "spoofed" not in checks["compile_time"]["details"]["candidate"]["summary"]
    assert checks["compile_time"]["summary"] == "Agent summary remains human-authored"

    invalid = performance_runner_result("ir_serialization")
    invalid["parameters"] = {"repeat": 19}
    (run / "ir_serialization/result.json").write_text(json.dumps(invalid))
    with pytest.raises(ContractError, match="sampling metadata differs"):
        seal_result(
            task(), "20260911-performance-invalid", value,
            {"required_checks": ["frontend_tests"]}, performance_environment(),
            tmp_path / "run", tmp_path / "invalid-sealed",
        )


@pytest.mark.parametrize("missing", ["check", "architecture", "pr_info"])
@pytest.mark.parametrize("event_kind", ["pull_request", "push", "manual"])
def test_incomplete_minimum_or_review_cannot_claim_pass(tmp_path, missing, event_kind):
    value = answer()
    if missing == "check":
        value["checks"] = []
    elif missing == "pr_info" and event_kind != "pull_request":
        value["reviews"][0]["status"] = "not_applicable"
    else:
        value["reviews"] = [row for row in value["reviews"] if row["kind"] != missing]
    result = seal(tmp_path, value, event_kind)
    required = missing != "pr_info" or event_kind == "pull_request"
    assert result["status"] == ("infra_error" if required else "pass")
    assert not result["blocking_reasons"]
    assert bool(result["limitations"]) == required


@pytest.mark.parametrize("location", ["top", "review", "both"])
def test_high_risk_review_blocks_without_explicit_blocking_flag(tmp_path, location):
    value = answer()
    finding = {"severity": "high", "summary": "Public behavior broken"}
    if location in {"top", "both"}:
        value["findings"] = [finding]
    if location in {"review", "both"}:
        value["reviews"][1]["findings"] = [finding]
    result = seal(tmp_path, value)
    assert result["status"] == "fail"
    assert result["findings"] == [finding]
    assert all("findings" not in review for review in result["reviews"])


def test_required_evidence_must_publish_while_optional_files_use_remaining_budget(tmp_path):
    value = answer()
    value["checks"][0]["evidence"] = ["report.txt"]
    initial = seal(tmp_path, value)
    assert initial["status"] == "infra_error"
    assert initial["checks"][0]["status"] == "pass"
    assert initial["evidence_delivery"]["status"] == "incomplete"
    artifacts = tmp_path / "run/artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "report.txt").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    result = seal(tmp_path, value)
    assert result["status"] == "infra_error"
    assert result["checks"][0]["status"] == "pass"
    assert result["evidence_delivery"]["status"] == "incomplete"
    assert result["artifacts"][0]["omitted"]
    assert not (tmp_path / "sealed/artifacts/report.txt").exists()

    (artifacts / "report.txt").write_text("required evidence\n")
    (artifacts / "optional.log").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    value["artifacts"] = ["optional.log"]
    result = seal(tmp_path, value)
    assert result["status"] == "pass"
    assert result["evidence_delivery"]["status"] == "incomplete"
    assert [row["path"] for row in result["artifacts"]] == [
        "report.txt", "optional.log",
    ]
    assert "omitted" not in result["artifacts"][0]
    assert result["artifacts"][1]["omitted"]


@pytest.mark.parametrize("required_overflow", [False, True])
def test_evidence_count_limits_are_separate_and_deduplicated(tmp_path, required_overflow):
    value = answer()
    required = [f"required-{index}.txt" for index in range(MAX_REQUIRED_FILES + required_overflow)]
    optional = [f"optional-{index}.txt" for index in range(MAX_OPTIONAL_FILES + 1)]
    value["checks"][0]["evidence"] = required + required[:1]
    value["artifacts"] = ["missing-optional.txt"] + required[:1] + optional + optional[:1]
    artifacts = tmp_path / "run/artifacts"
    artifacts.mkdir(parents=True)
    for name in required + optional:
        (artifacts / name).write_text("evidence\n")

    result = seal(tmp_path, value)

    assert result["status"] == ("infra_error" if required_overflow else "pass")
    assert result["checks"][0]["status"] == "pass"
    assert [row["path"] for row in result["artifacts"] if not row.get("omitted")] == (
        required[:MAX_REQUIRED_FILES] + optional[:MAX_OPTIONAL_FILES]
    )
    assert len(result["artifacts"]) == len(required) + len(optional) + 1
    omitted = result["evidence_delivery"]["omitted"]
    assert any(row["required"] for row in omitted) == required_overflow
    assert (tmp_path / "sealed/result.json").is_file()


def test_published_result_json_is_readable_without_changing_its_data(tmp_path):
    result = seal(tmp_path, answer())
    text = (tmp_path / "sealed/result.json").read_text()
    assert text.startswith("{\n  ")
    assert "\n  \"checks\":" in text
    assert json.loads(text) == result


def test_lightweight_diff_evidence_can_pass_but_cannot_replace_explicit_full(tmp_path):
    from agent_ci.policy import minimum_checks

    artifacts = tmp_path / "run/artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "validation.txt").write_text("Reviewed diff: ordinary comment only; diff check passed.\n")
    value = answer()
    value["checks"] = [{
        "tool_id": "change_validation", "status": "pass",
        "summary": "Ordinary comment only; reviewed diff and ran a lightweight check; build unnecessary",
        "evidence": ["validation.txt"],
    }]
    for full in (False, True):
        policy = minimum_checks(
            [{"path": "python/triton_anchor/__init__.py"}], backend_enabled=True, full=full,
        )
        result = seal_result(
            task(), "20260911-run", value, policy, {"profile": "test"},
            tmp_path / "run", tmp_path / "sealed",
        )
        assert result["status"] == ("infra_error" if full else "pass")


@pytest.mark.parametrize("missing", ["summary", "evidence", "file"])
def test_change_validation_requires_reasoning_and_evidence(tmp_path, missing):
    artifacts = tmp_path / "run/artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "validation.txt").write_text("Completed targeted validation\n")
    value = answer()
    check = {
        "tool_id": "change_validation", "status": "pass",
        "summary": "Targeted validation for the changed behavior",
        "evidence": ["validation.txt"],
    }
    if missing == "file":
        (artifacts / "validation.txt").unlink()
    else:
        check[missing] = " " if missing == "summary" else []
    value["checks"] = [check]
    result = seal_result(
        task(), "20260911-run", value, {"required_checks": ["change_validation"]},
        {"profile": "test"}, tmp_path / "run", tmp_path / "sealed",
    )
    assert result["status"] == "infra_error"
    assert result["checks"][0]["status"] == "pass"
    assert not result["blocking_reasons"]
    assert result["limitations"]


def test_code_failure_and_tool_limitations_have_separate_explanations(tmp_path):
    value = answer()
    value["checks"][0].update(status="fail", summary="Plugin discovery regression reproduced")
    value["checks"].append({"tool_id": "control_plane", "status": "infra_error",
                            "summary": "Checker rejects an in-tree symlink"})
    value["reviews"][1].update(status="fail", summary="Plugin discovery violates the architecture contract")
    value["blocking_reasons"] = ["Plugin discovery regression reproduced"]
    value["findings"] = [{"severity": "high", "blocking": True,
                           "summary": "Installed plugins are not discovered",
                           "qualification": "Directory scanning omits entry-point plugins",
                           "code_evidence": ["src/backends.py:17"]},
                          {"severity": "high", "summary": "Initialization leaks a resource"}]
    value["limitations"] = ["The checker limitation does not affect the independently reproduced regression"]
    result = seal(tmp_path, value)
    assert result["status"] == "fail"
    assert result["blocking_reasons"] == [finding["summary"] for finding in value["findings"]]
    assert result["checks"][0]["summary"] == value["checks"][0]["summary"]
    assert result["reviews"][1]["summary"] == value["reviews"][1]["summary"]
    assert any("control_plane" in reason for reason in result["limitations"])
    assert value["limitations"][0] in result["limitations"]
    assert result["findings"] == value["findings"]


@pytest.mark.parametrize("source", ["check", "review", "agent", "summary"])
def test_failure_without_blocking_findings_keeps_a_reason(tmp_path, source):
    value = answer()
    value["findings"] = [{"severity": "low", "summary": "Improve error wording"}]
    if source in {"check", "agent"}:
        value["checks"][0].update(status="fail", summary="Regression reproduced")
    elif source == "review":
        value["reviews"][1].update(status="fail", summary="Architecture contract broken")
    else:
        value.update(status="fail", summary="Independent validation failed")
    if source == "agent":
        value["blocking_reasons"] = ["Independent blocker"]
    expected = {"check": "frontend_tests：Regression reproduced",
                "review": "必要审查未通过：architecture — Architecture contract broken",
                "agent": "Independent blocker", "summary": "Independent validation failed"}
    result = seal(tmp_path, value)
    assert result["status"] == "fail"
    assert result["blocking_reasons"] == [expected[source]]


def test_limitations_do_not_turn_a_completed_review_into_a_code_failure(tmp_path):
    value = answer()
    value["limitations"] = ["Optional performance comparison unavailable; correctness verified independently"]
    result = seal(tmp_path, value)
    assert result["status"] == "pass"
    assert result["limitations"] == value["limitations"]
    value["checks"][0].update(status="infra_error", summary="Required test dependency unavailable")
    result = seal(tmp_path, value)
    assert result["status"] == "infra_error"
    assert not result["blocking_reasons"]
    assert any("Required test dependency" in reason for reason in result["limitations"])


@pytest.mark.parametrize("status", ["warning", "limited"])
@pytest.mark.parametrize("required", [False, True])
def test_advisory_and_limited_checks_do_not_override_minimum_validation(tmp_path, status, required):
    value = answer()
    row = {"tool_id": "frontend_tests" if required else "supplemental", "status": status,
           "summary": "Supplemental observation", "limitation": "Optional coverage unavailable"}
    if required:
        value["checks"] = [row]
    else:
        value["checks"].append(row)
    result = seal(tmp_path, value)
    assert result["status"] == ("infra_error" if required else "pass")
    assert not result["blocking_reasons"]
    assert result["limitations"] == ([row["limitation"]] if required or status == "limited" else [])


def test_repaired_checks_and_format_warning_keep_final_outcome_and_single_limit(tmp_path):
    value = answer()
    value["checks"][0].update(details={"initial_status": "infra_error", "repair": "Corrected import mode"})
    explanation = "缺少补充回归套件；必要行为已通过独立定向用例验证。"
    value["checks"] += [
        {"tool_id": "diff_check", "status": "warning", "summary": "文档行尾空白", "details": {"exit_code": 2}},
        {"tool_id": "control_plane", "status": "limited", "summary": "没有现成回归套件", "limitation": explanation},
    ]
    value["findings"] = [{"severity": "low", "blocking": False, "summary": "文档行尾空白"}]
    value["limitations"] = [explanation]
    result = seal(tmp_path, value)
    assert result["status"] == "pass"
    assert result["blocking_reasons"] == []
    assert result["limitations"] == [explanation]
    assert result["checks"][0]["details"]["initial_status"] == "infra_error"


def test_only_selected_files_are_published_and_text_is_redacted(tmp_path):
    artifacts = tmp_path / "run/artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "report.xml").write_text("<output>fixture-private-token</output>")
    (artifacts / "all-commands.log").write_text("private full trace")
    value = answer()
    value["checks"][0]["evidence"] = ["report.xml"]
    with patch.dict("os.environ", {"GITEE_TOKEN": "fixture-private-token"}):
        result = seal(tmp_path, value)
    assert result["status"] == "pass"
    assert "fixture-private-token" not in (tmp_path / "sealed/artifacts/report.xml").read_text()
    assert "fixture-private-token" in (artifacts / "report.xml").read_text()
    assert not (tmp_path / "sealed/artifacts/all-commands.log").exists()
    assert [row["path"] for row in result["artifacts"]] == ["report.xml"]


def test_git_result_and_selected_file_commit_together_and_retry_is_idempotent(tmp_path):
    artifacts = tmp_path / "run/artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "report.txt").write_text("ok")
    value = answer()
    value["checks"][0]["evidence"] = ["report.txt"]
    result = seal(tmp_path, value)
    remote = tmp_path / "relay.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    relay = GitRelay(str(remote), tmp_path / "cache", allow_local=True)
    original = relay.git
    lost = False

    def lost_response(args, **kwargs):
        nonlocal lost
        response = original(args, **kwargs)
        if args[0] == "push" and not lost:
            lost = True
            raise RuntimeError("successful push response lost")
        return response

    with patch.object(relay, "git", side_effect=lost_response):
        first = relay.publish_result(task(), result["run_id"], tmp_path / "sealed")
    second = relay.publish_result(task(), result["run_id"], tmp_path / "sealed")
    assert first == second
    branch = relay.results_branch
    count = subprocess.check_output(["git", "rev-list", "--count", branch], cwd=remote).strip()
    assert count == b"1"
    message = subprocess.check_output(
        ["git", "log", "-1", "--format=%s", branch], cwd=remote,
    ).decode().strip()
    assert message == f"local-ci: pass {task()['head_sha'][:12]} {result['run_id']}"
    prefix = f"{branch}:{result_task_prefix(task())}/{result['run_id']}"
    published = json.loads(subprocess.check_output(["git", "show", prefix + "/result.json"], cwd=remote))
    assert published == result
    assert subprocess.check_output(["git", "show", prefix + "/artifacts/report.txt"], cwd=remote) == b"ok"

    full_task = task("push")
    full_task.update(target_branch="triton_v3.0", full=True)
    full_task["metadata_digest"] = metadata_digest(full_task)
    full_task["task_id"] = task_id(full_task)
    full_task.update(
        task_ref=f"ci/branch/{full_task['task_id']}/tested",
        base_task_ref=f"ci/branch/{full_task['task_id']}/base",
        head_task_ref=f"ci/branch/{full_task['task_id']}/head",
    )
    document = {
        "schema": "triton-anchor-local-ci/flaggems-v1",
        "mode": "full",
        "sample_size": 1,
        "seed": "",
        "summary": {"total": 1, "passed": 1, "failed": 0, "timed_out": 0, "status": "pass"},
        "results": [{"index": 1, "op": "add", "test_status": "成功"}],
    }
    tool_dir = tmp_path / "full-run/artifacts/candidate/flaggems"
    tool_dir.mkdir(parents=True)
    (tool_dir / "flaggems-summary.json").write_text(json.dumps(document))
    (tool_dir / "result.json").write_text(json.dumps({
        "tool_id": "flaggems", "status": "pass", "target_sha": full_task["tested_sha"],
        "parameters": {"mode": "full"}, "details": {"flaggems-summary": document},
    }))
    full_answer = answer()
    full_answer["checks"] = [{
        "tool_id": "flaggems", "status": "pass", "summary": "Full suite passed",
        "parameters": {"mode": "full"},
        "evidence": ["candidate/flaggems/flaggems-summary.json"],
        "details": {"flaggems-summary": document},
    }]
    full_answer["artifacts"] = ["candidate/flaggems/flaggems-summary.json"]
    full_result = seal_result(
        full_task, "20260911-full", full_answer,
        {"required_checks": ["flaggems"], "required_parameters": {"flaggems": {"mode": "full"}}},
        {"profile": "triton-3.0"}, tmp_path / "full-run", tmp_path / "full-sealed",
    )
    full_check = full_result["checks"][0]
    assert "flaggems-summary" not in full_check.get("details", {})
    assert full_check["evidence"] == []
    assert full_result["artifacts"] == []
    business_file = tmp_path / "full-sealed/business/flaggems-summary.json"
    assert json.loads(business_file.read_text()) == document
    business_bytes = business_file.read_bytes()
    business_file.unlink()
    with pytest.raises(ContractError, match="business result is missing"):
        relay.publish_result(full_task, full_result["run_id"], tmp_path / "full-sealed")
    business_file.write_bytes(business_bytes)
    relay.publish_result(full_task, full_result["run_id"], tmp_path / "full-sealed")
    relay.publish_result(full_task, full_result["run_id"], tmp_path / "full-sealed")
    business_path = (
        f"{branch}:runs/ci_full_flaggems/{full_task['tested_sha']}/"
        f"{full_result['run_id']}/flaggems-summary.json"
    )
    assert json.loads(subprocess.check_output(["git", "show", business_path], cwd=remote)) == document
    assert subprocess.check_output(["git", "rev-list", "--count", branch], cwd=remote).strip() == b"2"

    (tmp_path / "sealed/artifacts/report.txt").write_text("changed")
    with pytest.raises(ContractError):
        relay.publish_result(task(), result["run_id"], tmp_path / "sealed")


def test_result_directory_distinguishes_event_branch_and_pr_number():
    pull_request = task()
    pull_request["target_branch"] = "release/3.0"
    assert result_task_prefix(pull_request).startswith(
        "runs/pr/branch-release%2F3.0/pr-7/"
    )
    assert result_task_prefix(pull_request).endswith("/" + pull_request["head_sha"])
    push = {**pull_request, "event_kind": "push", "pr_number": 0}
    assert result_task_prefix(push).startswith("runs/push/branch-release%2F3.0/")


@pytest.mark.parametrize("grouped", [False, True])
def test_pre_upgrade_sealed_outbox_retries_original_result_path(tmp_path, grouped):
    result = seal(tmp_path, answer())
    value = task()
    old_prefix = (result_task_prefix(value, legacy=True) if grouped
                  else f"runs/{value['task_id']}")
    directory = tmp_path / old_prefix / result["run_id"] / "sealed"
    directory.parent.mkdir(parents=True)
    (tmp_path / "sealed").rename(directory)
    remote = tmp_path / "relay.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    relay = GitRelay(str(remote), tmp_path / "cache", allow_local=True)
    for _ in range(2):
        relay.publish_result(value, result["run_id"], directory)
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", relay.results_branch], cwd=remote,
    ).decode().splitlines()
    assert paths == [f"{old_prefix}/{result['run_id']}/result.json"]
    assert subprocess.check_output(
        ["git", "rev-list", "--count", relay.results_branch], cwd=remote,
    ).strip() == b"1"



def test_oversized_seal_never_leaves_a_result_commit_point(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_ci.delivery.MAX_RESULT_BYTES", 128)
    with pytest.raises(ContractError, match="exceeds"):
        seal(tmp_path, answer())
    assert not (tmp_path / "sealed/result.json").exists()
