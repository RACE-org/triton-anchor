"""Seal a concise result and selected evidence for a single Git publication."""

from __future__ import annotations

import os
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .protocol import (
    CHECK_STATUSES,
    ContractError,
    RESULT_SCHEMA,
    RESULT_STATUSES,
    RUN_ID,
    atomic_json,
    full_flaggems_result_path,
    validate_result,
    validate_task,
    within,
)

MAX_REQUIRED_FILES = 32
MAX_OPTIONAL_FILES = 8
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_FULL_FLAGGEMS_BYTES = 10 * 1024 * 1024
FULL_FLAGGEMS_ARTIFACT = "candidate/flaggems/flaggems-summary.json"
PERFORMANCE_KERNELS = ("add", "mm", "softmax", "layernorm")
PERFORMANCE_METRICS = (
    "serialize",
    "write_text",
    "read_text",
    "deserialize",
    "roundtrip",
)
PERFORMANCE_TOOLS = {
    "compile_time": 3,
    "pass_profile": 3,
    "ir_serialization": 20,
}


def _redactor(redact):
    if redact is not None:
        return redact
    secrets = [
        value
        for key, value in os.environ.items()
        if len(value) > 3
        and any(
            word in key.upper() for word in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")
        )
    ]

    def clean(text):
        for value in secrets:
            text = text.replace(value, "[redacted]")
        return text

    return clean


def _clean(value, redact):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_clean(item, redact) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item, redact) for key, item in value.items()}
    return value


def _records(value, identity):
    if not isinstance(value, list):
        raise ContractError(f"Agent {identity} records must be a list")
    result = []
    seen = set()
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get(identity), str):
            raise ContractError(f"Agent record needs {identity}")
        if row[identity] in seen or row.get("status") not in CHECK_STATUSES:
            raise ContractError(f"Duplicate or invalid {identity} result")
        if "limitation" in row and not isinstance(row["limitation"], str):
            raise ContractError("Check limitation must be an explanation")
        seen.add(row[identity])
        evidence = row.get("evidence", [])
        if not isinstance(evidence, list) or any(
            not isinstance(item, str) for item in evidence
        ):
            raise ContractError("Evidence must be a list of paths or review references")
        result.append(
            {
                **row,
                "summary": str(row.get("summary", "")),
                "evidence": evidence,
            }
        )
    return result


def required_parameters_match(check, expected):
    """Match the small trusted policy parameter set against an Agent check."""
    if not expected:
        return True
    actual = check.get("parameters", {})
    if not isinstance(actual, dict):
        return False
    if check.get("tool_id") == "flaggems" and "mode" not in actual:
        actual = {
            **actual,
            "mode": check.get("details", {})
            .get("flaggems-summary", {})
            .get("mode"),
        }
    return all(actual.get(name) == value for name, value in expected.items())


def validate_full_flaggems(document):
    """Validate the stable raw document consumed by the full-operator Dashboard."""
    if (
        not isinstance(document, dict)
        or document.get("schema") != "triton-anchor-local-ci/flaggems-v1"
        or document.get("mode") != "full"
    ):
        raise ContractError("Full FlagGems report must use mode=full")
    rows = document.get("results")
    summary = document.get("summary")
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) for row in rows)
        or not isinstance(summary, dict)
        or summary.get("total") != len(rows)
    ):
        raise ContractError("Full FlagGems report is missing complete operator results")
    counts = (summary.get("passed"), summary.get("failed"), summary.get("timed_out"))
    if any(type(value) is not int or value < 0 for value in counts) or sum(counts) != len(rows):
        raise ContractError("Full FlagGems summary counts do not match its operator results")
    if summary.get("status") not in {"pass", "fail"}:
        raise ContractError("Full FlagGems report needs a terminal summary status")
    return document


def _seal_full_flaggems(task, run_id, checks, source, destination, redact):
    """Seal trusted full data separately and remove its large body from result.json."""
    target = destination / "business" / "flaggems-summary.json"
    target.unlink(missing_ok=True)
    check = next((row for row in checks if row.get("tool_id") == "flaggems"), None)
    if not task.get("full") or check is None:
        return

    full_parameters = required_parameters_match(check, {"mode": "full"})
    details = check.get("details")
    if isinstance(details, dict):
        # result.json keeps the check conclusion; the operator table has its own
        # immutable publication path and is joined back only in the Dashboard feed.
        details.pop("flaggems-summary", None)
        if not details:
            check.pop("details", None)
    check["evidence"] = [
        path for path in check.get("evidence", []) if path != FULL_FLAGGEMS_ARTIFACT
    ]
    if check.get("status") not in {"pass", "fail"}:
        return
    if not full_parameters:
        raise ContractError("Completed full FlagGems check is missing mode=full")

    tool_dir = source / "candidate" / "flaggems"
    raw_path, tool_path = tool_dir / "flaggems-summary.json", tool_dir / "result.json"
    if not raw_path.is_file() or not tool_path.is_file():
        raise ContractError("Completed full FlagGems check is missing its runner report")
    raw = raw_path.read_bytes()
    if len(raw) > MAX_FULL_FLAGGEMS_BYTES:
        raise ContractError("Full FlagGems report exceeds the business-result budget")
    try:
        document = validate_full_flaggems(json.loads(raw))
        tool = json.loads(tool_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("Full FlagGems runner report is not valid JSON") from exc
    if (
        not isinstance(tool, dict)
        or tool.get("tool_id") != "flaggems"
        or tool.get("target_sha") != task["tested_sha"]
        or tool.get("status") != check["status"]
        or (tool.get("parameters") or {}).get("mode") != "full"
        or (tool.get("details") or {}).get("flaggems-summary") != document
    ):
        raise ContractError("Full FlagGems report does not match the trusted runner result")
    document = _clean(document, redact)
    check["business_result"] = full_flaggems_result_path(task, run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, document, pretty=True)


def _finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _valid_timing(value):
    return (
        isinstance(value, dict)
        and _finite_number(value.get("median_ms"))
        and type(value.get("count")) is int
        and value["count"] > 0
    )


def _performance_parameters(tool_id, value):
    """Resolve runner defaults while allowing supported targeted measurements."""
    if not isinstance(value, dict) or set(value) - {"kernels", "repeat", "warmup"}:
        raise ContractError(f"{tool_id} has unsupported performance parameters")
    defaults = {
        "kernels": list(PERFORMANCE_KERNELS),
        "repeat": PERFORMANCE_TOOLS[tool_id],
        "warmup": 1,
    }
    resolved = {key: value.get(key, default) for key, default in defaults.items()}
    kernels = resolved["kernels"]
    if (
        not isinstance(kernels, list)
        or not kernels
        or any(not isinstance(kernel, str) or kernel not in PERFORMANCE_KERNELS for kernel in kernels)
        or type(resolved["repeat"]) is not int
        or not 1 <= resolved["repeat"] <= 100
        or type(resolved["warmup"]) is not int
        or not 0 <= resolved["warmup"] <= 20
    ):
        raise ContractError(f"{tool_id} has unsupported performance parameters")
    return resolved


def _validate_performance_details(tool_id, document, task, environment, parameters):
    """Validate measurements against the runner's actual identity and sampling."""
    kernels = parameters["kernels"]
    details = document.get("details")
    candidate = details.get("candidate") if isinstance(details, dict) else None
    comparison = details.get("comparison") if isinstance(details, dict) else None
    if not isinstance(candidate, dict) or not isinstance(comparison, dict):
        raise ContractError(f"{tool_id} is missing candidate or comparison details")
    metadata, summary = candidate.get("metadata"), candidate.get("summary")
    runtime = environment.get("variants", {}).get("candidate", environment)
    if (
        not isinstance(metadata, dict)
        or not isinstance(summary, dict)
        or set(summary) != set(kernels)
        or metadata.get("commit_sha") != task["tested_sha"]
        or metadata.get("environment_fingerprint")
        != runtime.get("environment_fingerprint")
        or metadata.get("profile_id") != runtime.get("profile")
        or metadata.get("llvm_revision") != runtime.get("llvm_hash")
        or metadata.get("kernels") != kernels
        or metadata.get("repeat") != parameters["repeat"]
        or metadata.get("warmup") != parameters["warmup"]
    ):
        raise ContractError(f"{tool_id} candidate identity or sampling metadata differs")

    for kernel in kernels:
        row = summary[kernel]
        if not isinstance(row, dict):
            raise ContractError(f"{tool_id} has an invalid {kernel} summary")
        if tool_id == "compile_time":
            timing = row.get("compile_est", {})
            valid = row.get("all_correct") is True and _valid_timing(timing)
        elif tool_id == "pass_profile":
            passes = row.get("passes")
            timings = [
                item.get("wall_ms")
                for item in passes.values()
                if isinstance(item, dict)
            ] if isinstance(passes, dict) else []
            valid = bool(timings) and all(_valid_timing(timing) for timing in timings)
        else:
            metrics = row.get("metrics")
            valid = isinstance(metrics, dict) and all(
                _valid_timing(metrics.get(metric))
                for metric in PERFORMANCE_METRICS
            )
        if not valid:
            raise ContractError(f"{tool_id} has an invalid {kernel} measurement")

    if comparison.get("status") == "not_comparable":
        if not isinstance(comparison.get("reason"), str):
            raise ContractError(f"{tool_id} has an invalid comparison reason")
        return details
    if comparison.get("candidate_sha") != task["tested_sha"]:
        raise ContractError(f"{tool_id} comparison belongs to another candidate")
    if tool_id == "compile_time":
        rows = comparison.get("kernels")
        keys = {
            row.get("kernel") for row in rows if isinstance(row, dict)
        } if isinstance(rows, list) else set()
        expected = set(kernels)
    elif tool_id == "pass_profile":
        rows = comparison.get("passes")
        keys = {
            row.get("kernel") for row in rows if isinstance(row, dict)
        } if isinstance(rows, list) else set()
        expected = set(kernels)
    else:
        rows = comparison.get("rows")
        keys = {
            (row.get("kernel"), row.get("metric"))
            for row in rows if isinstance(row, dict)
        } if isinstance(rows, list) else set()
        expected = {
            (kernel, metric)
            for kernel in kernels
            for metric in PERFORMANCE_METRICS
        }
    if not isinstance(rows, list) or not expected <= keys:
        raise ContractError(f"{tool_id} comparison is missing measured kernel rows")
    return details


def _seal_performance(task, checks, source, environment):
    """Replace Agent-projected performance data with trusted runner results."""
    indexed = {row.get("tool_id"): row for row in checks}
    runtime = environment.get("variants", {}).get("candidate", environment)
    for tool_id in PERFORMANCE_TOOLS:
        check = indexed.get(tool_id)
        path = source / "candidate" / tool_id / "result.json"
        if not path.is_file():
            if check is not None:
                check.pop("details", None)
                if check.get("status") in {"pass", "fail", "infra_error", "cancelled"}:
                    raise ContractError(f"{tool_id} check is missing its trusted runner result")
            continue
        try:
            document = json.loads(path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"{tool_id} runner result is not valid JSON") from exc
        if (
            not isinstance(document, dict)
            or document.get("tool_id") != tool_id
            or document.get("target_sha") != task["tested_sha"]
            or document.get("variant") != "candidate"
            or document.get("llvm_hash") != runtime.get("llvm_hash")
            or document.get("environment_fingerprint")
            != runtime.get("environment_fingerprint")
            or document.get("status") not in CHECK_STATUSES
        ):
            raise ContractError(f"{tool_id} runner identity or status differs")
        parameters = _performance_parameters(tool_id, document.get("parameters"))
        if document["status"] == "pass":
            details = _validate_performance_details(
                tool_id, document, task, environment, parameters
            )
        else:
            details = document.get("details") if isinstance(document.get("details"), dict) else None
        if check is None:
            check = {
                "tool_id": tool_id,
                "status": document["status"],
                "summary": "固定性能脚本结果由 Worker 校验并封存",
                "evidence": [],
            }
            checks.append(check)
            indexed[tool_id] = check
        check["status"] = document["status"]
        check["parameters"] = parameters
        evidence = f"candidate/{tool_id}/result.json"
        if evidence not in check["evidence"]:
            check["evidence"] = [*check["evidence"], evidence]
        if details:
            check["details"] = details
        else:
            check.pop("details", None)


def _seal_backend_smoke(task, checks, source, environment):
    """Fill an omitted backend check from this run's candidate runner evidence.

    An explicit Agent conclusion may include a later repair or native-command
    revalidation, so an earlier runner attempt must not overwrite it.
    """
    if any(check["tool_id"] == "backend_smoke" for check in checks):
        return
    runtime = environment.get("variants", {}).get("candidate", environment)
    if runtime.get("backend_enabled") is not True:
        return
    evidence = "candidate/backend_smoke/result.json"
    path = within(source, evidence)
    if not path.is_file():
        return
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ContractError("backend_smoke runner result exceeds the evidence budget")
    try:
        document = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("backend_smoke runner result is not valid JSON") from exc
    if (
        not isinstance(document, dict)
        or document.get("tool_id") != "backend_smoke"
        or document.get("target_sha") != task["tested_sha"]
        or document.get("variant") != "candidate"
        or not runtime.get("llvm_hash")
        or document.get("llvm_hash") != runtime["llvm_hash"]
        or not runtime.get("environment_fingerprint")
        or document.get("environment_fingerprint") != runtime["environment_fingerprint"]
        or document.get("status") not in CHECK_STATUSES
    ):
        raise ContractError("backend_smoke runner identity or status differs")
    details = document.get("details")
    receipt = details.get("smoke_success") if isinstance(details, dict) else None
    if document["status"] == "pass" and (
        document.get("exit_code") != 0
        or not isinstance(receipt, dict)
        or receipt.get("tool") != "backend_smoke"
        or receipt.get("status") != "passed"
        or receipt.get("task_id") != task["task_id"]
        or receipt.get("target_sha") != task["tested_sha"]
    ):
        raise ContractError("backend_smoke pass is missing its matching completion record")
    checks.append({
        "tool_id": "backend_smoke",
        "display_name": "后端基本功能与 JIT 验证",
        "status": document["status"],
        "summary": "本次候选环境的后端基本功能与 JIT 工具结果，由 Worker 校验并补齐。"
        + str(document.get("reason") or ""),
        "evidence": [evidence],
        "details": details if isinstance(details, dict) else {},
    })


def seal_result(
    task,
    run_id,
    agent_result,
    policy,
    environment,
    source_dir,
    destination,
    *,
    redact=None,
    collect_business=True,
):
    """The host supplies identity; Codex supplies checks, reviews and selected files."""
    validate_task(task)
    if (
        not isinstance(run_id, str)
        or not RUN_ID.fullmatch(run_id)
        or not isinstance(agent_result, dict)
    ):
        raise ContractError("Invalid run or Agent result")
    redact = _redactor(redact)
    checks = _records(agent_result.get("checks", []), "tool_id")
    source = Path(source_dir) / "artifacts"
    if collect_business:
        _seal_performance(task, checks, source, environment)
        _seal_backend_smoke(task, checks, source, environment)
    reviews = _records(agent_result.get("reviews", []), "kind")
    findings = agent_result.get("findings", [])
    if not isinstance(findings, list):
        raise ContractError("Findings must be a list")
    findings = list(findings)
    for review in reviews:
        # Fold older nested findings into the single published findings list.
        for finding in review.pop("findings", []):
            if finding not in findings:
                findings.append(finding)
    if any(not isinstance(finding, dict) for finding in findings):
        raise ContractError("Findings must be objects")
    reasons = agent_result.get("blocking_reasons", [])
    if not isinstance(reasons, list):
        raise ContractError("Blocking reasons must be a list")
    reasons = [str(reason) for reason in reasons]
    failure_diagnostics = []
    limitations = agent_result.get("limitations", [])
    if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
        raise ContractError("Limitations must be a list of explanations")
    limitations = list(limitations)
    incomplete = False
    failed = False
    selected = {check["tool_id"]: check for check in checks}

    def limitation(row, fallback):
        return row.get("limitation") or fallback

    for tool_id in policy.get("required_checks", []):
        check = selected.get(tool_id, {})
        if check.get("status") not in {"pass", "fail"}:
            limitations.append(limitation(check,
                f"最低必检未通过：{tool_id}"
                + (f" — {check['summary']}" if check.get("summary") else "")
            ))
            incomplete = True
        elif check.get("status") == "pass" and tool_id == "change_validation" and (
            not check["summary"].strip() or not check["evidence"]
        ):
            limitations.append("变更验证必须说明影响范围、选测理由并提供实际证据文件")
            incomplete = True
        elif not required_parameters_match(
            check, policy.get("required_parameters", {}).get(tool_id, {})
        ):
            limitations.append(f"最低必检参数不符合任务要求：{tool_id}")
            incomplete = True
    for check in checks:
        if check["status"] == "limited" and check["tool_id"] not in policy.get("required_checks", []):
            limitations.append(limitation(check, f"{check['tool_id']}：{check['summary'] or '验证受限'}"))
        if check["status"] in {"fail", "infra_error", "cancelled"}:
            reason = f"{check['tool_id']}：{check['summary'] or check['status']}"
            destination_reasons = failure_diagnostics if check["status"] == "fail" else limitations
            if check["status"] == "fail" or check["tool_id"] not in policy.get("required_checks", []):
                destination_reasons.append(reason if check["status"] == "fail" else limitation(check, reason))
            failed |= check["status"] == "fail"
            incomplete |= check["status"] != "fail"
    reviewed = {review["kind"]: review for review in reviews}
    required_reviews = ("pr_info", "architecture") if task["event_kind"] == "pull_request" else ("architecture",)
    for kind in required_reviews:
        review = reviewed.get(kind, {})
        if review.get("status") != "pass":
            reason = (
                f"必要审查未通过：{kind}"
                + (f" — {review['summary']}" if review.get("summary") else "")
            )
            if review.get("status") == "fail":
                failure_diagnostics.append(reason)
            else:
                limitations.append(limitation(review, reason))
            failed |= review.get("status") == "fail"
            incomplete |= review.get("status") != "fail"
    for review in reviews:
        if review["kind"] not in required_reviews and review["status"] in {"infra_error", "cancelled", "limited"}:
            limitations.append(limitation(review, f"{review['kind']}：{review['summary'] or review['status']}"))
    blocking_summaries = []
    for finding in findings:
        if finding.get("blocking") is True or finding.get("severity") in {
            "high",
            "critical",
        }:
            failed = True
            blocking_summaries.append(str(finding.get("summary") or "高风险审查发现"))
    requested = agent_result.get("status", "pass")
    if requested not in RESULT_STATUSES:
        raise ContractError("Invalid Agent result status")
    failed |= requested == "fail" or bool(agent_result.get("blocking_reasons"))
    incomplete |= requested == "infra_error"
    if requested == "cancelled":
        status = "cancelled"
    elif failed:
        status = "fail"
    else:
        status = "infra_error" if incomplete else "pass"
    # Findings own defect conclusions; diagnostics are a fallback, not extra issues.
    reasons = blocking_summaries or reasons or failure_diagnostics
    summary = str(agent_result.get("summary", ""))
    if status == "fail" and not reasons:
        reasons.append(summary or status)
    elif status in {"infra_error", "cancelled"} and not limitations:
        limitations.append(summary or status)

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if collect_business:
        _seal_full_flaggems(task, run_id, checks, source, destination, redact)
    else:
        # A previous attempt may have staged full data before failing later.
        (destination / "business" / "flaggems-summary.json").unlink(missing_ok=True)
    selected = agent_result.get("artifacts", [])
    if not isinstance(selected, list):
        raise ContractError("Agent artifacts must be a list")
    if task.get("full"):
        selected = [
            entry
            for entry in selected
            if (
                entry
                if isinstance(entry, str)
                else entry.get("path", "") if isinstance(entry, dict) else None
            )
            != FULL_FLAGGEMS_ARTIFACT
        ]
    required = [path for check in checks for path in check["evidence"]]
    # Required check evidence is copied first so optional files cannot consume
    # its byte budget. Agent-selected files use the remaining budget.
    chosen = required + list(selected)
    required_paths = set(required)
    artifacts = []
    omitted_evidence = []
    seen = set()
    total = 0
    uploaded = {True: 0, False: 0}
    for entry in chosen:
        if not isinstance(entry, (str, dict)):
            raise ContractError("Artifact selection needs a relative path")
        path = entry if isinstance(entry, str) else entry.get("path", "")
        if path in seen:
            continue
        seen.add(path)
        incoming = within(source, path)
        required_file = path in required_paths
        limit = MAX_REQUIRED_FILES if required_file else MAX_OPTIONAL_FILES
        row = {
            "path": path,
            "label": path if isinstance(entry, str) else str(entry.get("label", path)),
        }
        if uploaded[required_file] >= limit:
            category = "必传证据" if required_file else "选传附件"
            row["omitted"] = f"超过 {limit} 份{category}上限，保留在 CI 主机"
        elif not incoming.is_file():
            row["omitted"] = "文件未生成，未上传"
        elif incoming.stat().st_size > MAX_FILE_BYTES:
            row["omitted"] = "文件超过 2 MiB，保留在 CI 主机"
        else:
            data = incoming.read_bytes()
            try:
                data = redact(data.decode("utf-8")).encode("utf-8")
            except UnicodeDecodeError:
                pass
            if len(data) > MAX_FILE_BYTES or total + len(data) > MAX_TOTAL_BYTES:
                row["omitted"] = "超出 10 MiB 发布预算，保留在 CI 主机"
            else:
                target = within(destination / "artifacts", redact(path))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                row["size"] = len(data)
                total += len(data)
                uploaded[required_file] += 1
        if row.get("omitted"):
            omitted_evidence.append({
                "path": path,
                "reason": row["omitted"],
                "required": path in required_paths,
            })
            if path in required_paths:
                limitations.append(
                    f"必传检查证据未完整发布，整体结论待确认：{path}（{row['omitted']}）"
                )
                if status == "pass":
                    status = "infra_error"
        artifacts.append(row)
    result = {
        "schema": RESULT_SCHEMA,
        "task": task,
        "run_id": run_id,
        "status": status,
        "summary": summary,
        "checks": checks,
        "reviews": reviews,
        "findings": findings,
        "blocking_reasons": list(dict.fromkeys(reasons)),
        "limitations": list(dict.fromkeys(limitations)),
        "artifacts": artifacts,
        "evidence_delivery": {
            "status": "incomplete" if omitted_evidence else "complete",
            "omitted": omitted_evidence,
        },
        "environment": environment,
        "policy": policy,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    result = _clean(result, redact)
    result["task"] = task
    validate_result(result, task)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode()
    if len(encoded) + 1 > MAX_RESULT_BYTES:
        raise ContractError("Result summary exceeds 2 MiB")
    # result.json is the commit point used by restart recovery.
    atomic_json(destination / "result.json", result, pretty=True)
    return result
