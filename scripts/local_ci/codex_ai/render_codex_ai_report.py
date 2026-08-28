#!/usr/bin/env python3
"""Validate structured Codex output and render the fixed local-CI report."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHARED_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPT_DIR))
from finding_locations import (  # noqa: E402
    normalized_repository_path,
    parse_finding_line_range as parse_shared_finding_line_range,
)

ROOT_KEYS = {
    "verdict",
    "summary",
    "merge_recommendation",
    "change_request_assessment",
    "changed_files",
    "behavior_coverage",
    "findings",
    "unlocated_findings",
    "suggested_tests",
    "residual_risks",
    "test_execution",
    "completion_marker",
}
CHANGE_REQUEST_ASSESSMENT_KEYS = {
    "status",
    "contributor_goal",
    "expected_behavior",
    "implementation_summary",
    "evidence",
}
CHANGED_FILE_KEYS = {
    "path",
    "change_type",
    "summary",
    "impact",
    "validation_strategy",
}
MANIFEST_FILE_KEYS = {"path", "change_type"}
MANIFEST_RENAME_KEYS = {"path", "change_type", "previous_path"}
BEHAVIOR_COVERAGE_KEYS = {
    "normal",
    "boundary",
    "error",
    "compatibility",
    "integration",
}
BEHAVIOR_ITEM_KEYS = {"scope", "strategy", "result"}
FINDING_KEYS = {
    "id",
    "severity",
    "category",
    "file",
    "line",
    "code_role",
    "title",
    "evidence",
    "impact",
    "fix_direction",
}
UNLOCATED_FINDING_KEYS = {
    "id",
    "severity",
    "category",
    "trusted_file",
    "reported_line",
    "location_issue",
    "code_role",
    "title",
    "evidence",
    "impact",
    "fix_direction",
}
TEST_KEYS = {"id", "priority", "target", "description"}
TEST_EXECUTION_KEYS = {
    "evidence_level",
    "status",
    "summary",
    "generated_test_files",
    "commands",
}
EVIDENCE_LEVELS = {
    "not_needed",
    "sufficient",
    "insufficient",
    "test_generation_error",
    "unavailable",
}
COMMAND_KEYS = {
    "id",
    "command",
    "role",
    "exit_code",
    "duration_seconds",
    "status",
    "evidence",
    "purpose",
}
COMMAND_ROLES = {"validation", "diagnostic", "unclassified"}
SEVERITIES = {"HIGH", "MEDIUM", "LOW"}
TEST_EXECUTION_STATUSES = {
    "not_run",
    "passed",
    "stable_failure",
    "flaky_failure",
    "infrastructure_failure",
    "test_generation_error",
    "insufficient_evidence",
    "unavailable",
}
COMMAND_STATUSES = {
    "passed",
    "failed",
    "stable_failure",
    "flaky_failure",
    "infrastructure_failure",
    "not_executed",
}
CHANGE_REQUEST_ASSESSMENT_STATUSES = {
    "implemented",
    "partially_implemented",
    "not_implemented",
    "not_assessable",
    "not_applicable",
}
CHANGE_TYPES = {"modified", "added", "deleted", "renamed"}
CATEGORIES = {
    "algorithm",
    "business-logic",
    "state-management",
    "cache-consistency",
    "concurrency",
    "resource-lifecycle",
    "data-integrity",
    "correctness",
    "regression",
    "security",
    "api-compatibility",
    "performance",
    "test-gap",
    "other",
}
CHINESE_TEXT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
MAX_ASSESSMENT_EVIDENCE_ITEMS = 8
MAX_TEST_EXECUTION_SUMMARY_ITEMS = 10
INTERNAL_COMMENT_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(AI|FILE|TEST|RUN)-0*([1-9][0-9]*)"
    r"(?![A-Za-z0-9_.-])[ \t]*",
    re.IGNORECASE,
)
PUBLIC_COMMENT_ID_TEMPLATES = {
    "AI": "问题 {number}",
    "FILE": "变更文件 {number}",
    "TEST": "建议测试 {number}",
    "RUN": "相关验证",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comment-output", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--requested-base-sha", default="")
    parser.add_argument(
        "--diff-mode",
        choices=("two-point", "merge-base"),
        default="two-point",
    )
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--local-ci-status", default="")
    parser.add_argument(
        "--local-ci-execution-mode",
        choices=("full", "codex_only", "unavailable"),
        default="full",
    )
    parser.add_argument(
        "--backend-validation-scope",
        choices=("full", "frontend_only", "unavailable"),
        default="full",
    )
    parser.add_argument("--tested-sha-kind", default="commit")
    parser.add_argument("--changed-file-count", required=True, type=int)
    parser.add_argument("--changed-files-manifest", required=True)
    parser.add_argument("--repository-root", default="")
    parser.add_argument(
        "--constraint-status",
        choices=("pass", "warning", "not_applicable"),
        default="pass",
    )
    parser.add_argument(
        "--constraint-reason",
        default="未发现测试数量或耗时超出轻量约束。",
    )
    return parser.parse_args()


def require_exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{location} keys mismatch; missing={missing}, extra={extra}")


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def require_chinese_string(value: Any, location: str) -> str:
    text = require_string(value, location)
    if not CHINESE_TEXT_RE.search(text):
        raise ValueError(f"{location} must contain Chinese explanatory text")
    return text


def assessment_evidence_items(value: Any, location: str) -> list[str]:
    if isinstance(value, str):
        return [require_chinese_string(value, location)]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty string or array")
    if len(value) > MAX_ASSESSMENT_EVIDENCE_ITEMS:
        raise ValueError(
            f"{location} must contain at most {MAX_ASSESSMENT_EVIDENCE_ITEMS} items"
        )
    items = [
        require_chinese_string(item, f"{location}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(set(items)) != len(items):
        raise ValueError(f"{location} must not contain duplicate items")
    return items


def test_execution_summary_items(value: Any, location: str) -> list[str]:
    if isinstance(value, str):
        return [require_chinese_string(value, location)]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty string or array")
    if len(value) > MAX_TEST_EXECUTION_SUMMARY_ITEMS:
        raise ValueError(
            f"{location} must contain at most "
            f"{MAX_TEST_EXECUTION_SUMMARY_ITEMS} items"
        )
    items = [
        require_chinese_string(item, f"{location}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(set(items)) != len(items):
        raise ValueError(f"{location} must not contain duplicate items")
    return items


def public_comment_identifier(
    match: re.Match[str], identifier_descriptions: dict[str, str]
) -> str:
    number = int(match.group(2))
    identifier = f"{match.group(1).upper()}-{number:03d}"
    if identifier in identifier_descriptions:
        return identifier_descriptions[identifier]
    template = PUBLIC_COMMENT_ID_TEMPLATES[match.group(1).upper()]
    return template.format(number=number)


def validate_changed_files_manifest(document: Any) -> list[dict[str, str]]:
    if not isinstance(document, list):
        raise ValueError("changed files manifest must be an array")
    manifest: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, item in enumerate(document):
        location = f"changed_files_manifest[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{location} must be an object")
        change_type = require_string(item.get("change_type"), f"{location}.change_type")
        expected_keys = (
            MANIFEST_RENAME_KEYS if change_type == "renamed" else MANIFEST_FILE_KEYS
        )
        require_exact_keys(item, expected_keys, location)
        if change_type not in CHANGE_TYPES:
            raise ValueError(f"{location}.change_type is invalid")
        path = require_string(item["path"], f"{location}.path")
        if path in seen_paths:
            raise ValueError(f"duplicate manifest path: {path}")
        seen_paths.add(path)
        normalized = {"path": path, "change_type": change_type}
        if change_type == "renamed":
            normalized["previous_path"] = require_string(
                item["previous_path"], f"{location}.previous_path"
            )
        manifest.append(normalized)
    return manifest


def parse_finding_line_range(value: Any, location: str) -> tuple[int, int]:
    text = require_string(value, location)
    line_range = parse_shared_finding_line_range(text)
    if line_range is None:
        raise ValueError(f"{location} must be a positive, ordered line number or range")
    return line_range


def validate_finding_location(
    finding_file: str,
    line_range: tuple[int, int],
    expected_files: list[dict[str, str]],
    repository_root: Path | None,
    location: str,
) -> None:
    relative_path = normalized_repository_path(finding_file)
    if relative_path is None:
        raise ValueError(
            f"{location}.file must be a normalized repository-relative path"
        )

    change_types = {item["path"]: item["change_type"] for item in expected_files}
    change_type = change_types.get(finding_file)
    if change_type is None:
        raise ValueError(f"{location}.file must be a changed file in the Git diff")
    if change_type == "deleted":
        raise ValueError(
            f"{location}.file is deleted; anchor the finding to a retained changed call site"
        )

    if repository_root is None:
        return
    source_path = repository_root.joinpath(*relative_path.parts)
    try:
        resolved_source = source_path.resolve(strict=True)
        resolved_source.relative_to(repository_root)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{location}.file is not a readable file in the review checkout"
        ) from exc
    if not resolved_source.is_file():
        raise ValueError(
            f"{location}.file is not a regular file in the review checkout"
        )
    try:
        source_lines = resolved_source.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as exc:
        raise ValueError(
            f"{location}.file could not be read from the review checkout"
        ) from exc
    start, end = line_range
    if end > len(source_lines):
        raise ValueError(
            f"{location}.line {start}-{end} is outside {finding_file} with "
            f"{len(source_lines)} lines"
        )


def command_target_groups(
    commands: list[dict[str, Any]], role: str
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for command in commands:
        if command["role"] == role:
            groups.setdefault(command["purpose"], []).append(command)
    return groups


def command_target_status(commands: list[dict[str, Any]]) -> str:
    methods: dict[str, list[dict[str, Any]]] = {}
    for command in commands:
        methods.setdefault(command["command"], []).append(command)
    failed_indexes = [
        index
        for index, command in enumerate(commands)
        if command["exit_code"] != 0
    ]
    clean_methods = {
        command_text
        for command_text, method_commands in methods.items()
        if method_commands
        and all(command["status"] == "passed" for command in method_commands)
    }
    if not failed_indexes:
        return (
            "passed"
            if any(command["status"] == "passed" for command in commands)
            else "insufficient_evidence"
        )
    if any(
        index > failed_indexes[-1] and command["command"] in clean_methods
        for index, command in enumerate(commands)
    ):
        return "passed"

    failed_statuses = {
        command["status"]
        for command in commands
        if command["exit_code"] != 0
    }
    if failed_statuses == {"infrastructure_failure"}:
        return "infrastructure_failure"
    if failed_statuses == {"flaky_failure"}:
        return "flaky_failure"
    if failed_statuses == {"stable_failure"}:
        return "stable_failure"
    return "insufficient_evidence"


def validate_report(
    document: Any,
    expected_files: list[dict[str, str]],
    repository_root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("report root must be an object")
    if repository_root is not None:
        repository_root = repository_root.resolve()
    require_exact_keys(document, ROOT_KEYS, "report")

    verdict = require_string(document["verdict"], "verdict")
    if verdict not in {"PASS", "WARNING", "FAIL"}:
        raise ValueError(f"unsupported verdict: {verdict}")
    require_chinese_string(document["summary"], "summary")
    require_chinese_string(document["merge_recommendation"], "merge_recommendation")
    assessment = document["change_request_assessment"]
    if not isinstance(assessment, dict):
        raise ValueError("change_request_assessment must be an object")
    require_exact_keys(
        assessment,
        CHANGE_REQUEST_ASSESSMENT_KEYS,
        "change_request_assessment",
    )
    assessment_status = require_string(
        assessment["status"], "change_request_assessment.status"
    )
    if assessment_status not in CHANGE_REQUEST_ASSESSMENT_STATUSES:
        raise ValueError("change_request_assessment.status is invalid")
    for key in {
        "contributor_goal",
        "expected_behavior",
        "implementation_summary",
    }:
        require_chinese_string(assessment[key], f"change_request_assessment.{key}")
    assessment_evidence_items(
        assessment["evidence"], "change_request_assessment.evidence"
    )
    if document["completion_marker"] != "CODEX_AI_CI_COMPLETE":
        raise ValueError("completion_marker is invalid")

    changed_files = document["changed_files"]
    if not isinstance(changed_files, list):
        raise ValueError("changed_files must be an array")
    actual_files: dict[str, str] = {}
    for index, changed_file in enumerate(changed_files):
        location = f"changed_files[{index}]"
        if not isinstance(changed_file, dict):
            raise ValueError(f"{location} must be an object")
        require_exact_keys(changed_file, CHANGED_FILE_KEYS, location)
        path = require_string(changed_file["path"], f"{location}.path")
        change_type = require_string(
            changed_file["change_type"], f"{location}.change_type"
        )
        if change_type not in CHANGE_TYPES:
            raise ValueError(f"{location}.change_type is invalid")
        if path in actual_files:
            raise ValueError(f"duplicate changed_files path: {path}")
        actual_files[path] = change_type
        for key in {"summary", "impact", "validation_strategy"}:
            require_chinese_string(changed_file[key], f"{location}.{key}")

    expected_map = {
        item["path"]: item["change_type"]
        for item in expected_files
    }
    if actual_files != expected_map:
        missing = sorted(set(expected_map) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_map))
        mismatched = sorted(
            path
            for path in set(actual_files) & set(expected_map)
            if actual_files[path] != expected_map[path]
        )
        raise ValueError(
            "changed_files does not match Git diff manifest; "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )

    behavior_coverage = document["behavior_coverage"]
    if not isinstance(behavior_coverage, dict):
        raise ValueError("behavior_coverage must be an object")
    require_exact_keys(
        behavior_coverage, BEHAVIOR_COVERAGE_KEYS, "behavior_coverage"
    )
    for behavior_name in BEHAVIOR_COVERAGE_KEYS:
        behavior = behavior_coverage[behavior_name]
        location = f"behavior_coverage.{behavior_name}"
        if not isinstance(behavior, dict):
            raise ValueError(f"{location} must be an object")
        require_exact_keys(behavior, BEHAVIOR_ITEM_KEYS, location)
        for key in BEHAVIOR_ITEM_KEYS:
            require_chinese_string(behavior[key], f"{location}.{key}")

    findings = document["findings"]
    if not isinstance(findings, list):
        raise ValueError("findings must be an array")
    finding_ids: set[str] = set()
    for index, finding in enumerate(findings):
        location = f"findings[{index}]"
        if not isinstance(finding, dict):
            raise ValueError(f"{location} must be an object")
        require_exact_keys(finding, FINDING_KEYS, location)
        finding_id = require_string(finding["id"], f"{location}.id")
        if not re.fullmatch(r"AI-[0-9]{3,}", finding_id):
            raise ValueError(f"{location}.id has an invalid format")
        if finding_id in finding_ids:
            raise ValueError(f"duplicate finding id: {finding_id}")
        finding_ids.add(finding_id)
        severity = require_string(finding["severity"], f"{location}.severity")
        category = require_string(finding["category"], f"{location}.category")
        if severity not in SEVERITIES:
            raise ValueError(f"{location}.severity is invalid")
        if category not in CATEGORIES:
            raise ValueError(f"{location}.category is invalid")
        finding_file = require_string(finding["file"], f"{location}.file")
        line_range = parse_finding_line_range(finding["line"], f"{location}.line")
        validate_finding_location(
            finding_file,
            line_range,
            expected_files,
            repository_root,
            location,
        )
        for key in {"code_role", "title", "evidence", "impact", "fix_direction"}:
            require_chinese_string(finding[key], f"{location}.{key}")

    unlocated_findings = document["unlocated_findings"]
    if not isinstance(unlocated_findings, list):
        raise ValueError("unlocated_findings must be an array")
    for index, finding in enumerate(unlocated_findings):
        location = f"unlocated_findings[{index}]"
        if not isinstance(finding, dict):
            raise ValueError(f"{location} must be an object")
        require_exact_keys(finding, UNLOCATED_FINDING_KEYS, location)
        finding_id = require_string(finding["id"], f"{location}.id")
        if not re.fullmatch(r"AI-[0-9]{3,}", finding_id):
            raise ValueError(f"{location}.id has an invalid format")
        if finding_id in finding_ids:
            raise ValueError(f"duplicate finding id: {finding_id}")
        finding_ids.add(finding_id)
        severity = require_string(finding["severity"], f"{location}.severity")
        category = require_string(finding["category"], f"{location}.category")
        if severity not in SEVERITIES:
            raise ValueError(f"{location}.severity is invalid")
        if category not in CATEGORIES:
            raise ValueError(f"{location}.category is invalid")
        if finding["trusted_file"]:
            trusted_file = require_string(
                finding["trusted_file"], f"{location}.trusted_file"
            )
            if trusted_file not in {
                item["path"] for item in expected_files if item["change_type"] != "deleted"
            }:
                raise ValueError(
                    f"{location}.trusted_file must be a retained changed file"
                )
        elif not isinstance(finding["trusted_file"], str):
            raise ValueError(f"{location}.trusted_file must be a string")
        require_string(finding["reported_line"], f"{location}.reported_line")
        for key in {
            "location_issue",
            "code_role",
            "title",
            "evidence",
            "impact",
            "fix_direction",
        }:
            require_chinese_string(finding[key], f"{location}.{key}")

    tests = document["suggested_tests"]
    if not isinstance(tests, list):
        raise ValueError("suggested_tests must be an array")
    test_ids: set[str] = set()
    for index, test in enumerate(tests):
        location = f"suggested_tests[{index}]"
        if not isinstance(test, dict):
            raise ValueError(f"{location} must be an object")
        require_exact_keys(test, TEST_KEYS, location)
        test_id = require_string(test["id"], f"{location}.id")
        if not re.fullmatch(r"TEST-[0-9]{3,}", test_id):
            raise ValueError(f"{location}.id has an invalid format")
        if test_id in test_ids:
            raise ValueError(f"duplicate test id: {test_id}")
        test_ids.add(test_id)
        priority = require_string(test["priority"], f"{location}.priority")
        if priority not in SEVERITIES:
            raise ValueError(f"{location}.priority is invalid")
        require_string(test["target"], f"{location}.target")
        require_chinese_string(test["description"], f"{location}.description")

    residual_risks = document["residual_risks"]
    if not isinstance(residual_risks, list):
        raise ValueError("residual_risks must be an array")
    for index, risk in enumerate(residual_risks):
        require_chinese_string(risk, f"residual_risks[{index}]")
    test_execution = document["test_execution"]
    if not isinstance(test_execution, dict):
        raise ValueError("test_execution must be an object")
    require_exact_keys(test_execution, TEST_EXECUTION_KEYS, "test_execution")
    evidence_level = require_string(
        test_execution["evidence_level"], "test_execution.evidence_level"
    )
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError(f"unsupported evidence level: {evidence_level}")
    execution_status = require_string(
        test_execution["status"], "test_execution.status"
    )
    if execution_status not in TEST_EXECUTION_STATUSES:
        raise ValueError(f"unsupported test execution status: {execution_status}")
    test_execution_summary_items(
        test_execution["summary"], "test_execution.summary"
    )

    generated_test_files = test_execution["generated_test_files"]
    if not isinstance(generated_test_files, list):
        raise ValueError("test_execution.generated_test_files must be an array")
    for index, file_name in enumerate(generated_test_files):
        require_string(file_name, f"test_execution.generated_test_files[{index}]")

    commands = test_execution["commands"]
    if not isinstance(commands, list):
        raise ValueError("test_execution.commands must be an array")
    command_ids: set[str] = set()
    for index, command in enumerate(commands):
        location = f"test_execution.commands[{index}]"
        if not isinstance(command, dict):
            raise ValueError(f"{location} must be an object")
        require_exact_keys(command, COMMAND_KEYS, location)
        command_id = require_string(command["id"], f"{location}.id")
        if not re.fullmatch(r"RUN-[0-9]{3,}", command_id):
            raise ValueError(f"{location}.id has an invalid format")
        if command_id in command_ids:
            raise ValueError(f"duplicate command id: {command_id}")
        command_ids.add(command_id)
        command_text = require_string(command["command"], f"{location}.command")
        command_role = require_string(command["role"], f"{location}.role")
        if command_role not in COMMAND_ROLES:
            raise ValueError(f"{location}.role is invalid")
        purpose = require_chinese_string(command["purpose"], f"{location}.purpose")
        if len(purpose) > 120:
            raise ValueError(
                f"{location}.purpose must contain at most 120 characters"
            )
        if INTERNAL_COMMENT_ID_RE.search(purpose):
            raise ValueError(f"{location}.purpose must not contain an internal ID")
        if not isinstance(command["exit_code"], int):
            raise ValueError(f"{location}.exit_code must be an integer")
        duration = command["duration_seconds"]
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise ValueError(f"{location}.duration_seconds must be a number")
        if duration < 0:
            raise ValueError(f"{location}.duration_seconds must not be negative")
        command_status = require_string(command["status"], f"{location}.status")
        if command_status not in COMMAND_STATUSES:
            raise ValueError(f"{location}.status is invalid")
        if command_status == "passed" and command["exit_code"] != 0:
            raise ValueError(f"{location}.passed command must have exit_code 0")
        if command_status in {
            "failed",
            "stable_failure",
            "flaky_failure",
            "infrastructure_failure",
        } and command["exit_code"] == 0:
            raise ValueError(
                f"{location}.{command_status} command must have a non-zero exit_code"
            )
        require_chinese_string(command["evidence"], f"{location}.evidence")

    validation_target_statuses = [
        command_target_status(target_commands)
        for target_commands in command_target_groups(commands, "validation").values()
    ]
    if execution_status == "passed":
        if not validation_target_statuses or any(
            status != "passed" for status in validation_target_statuses
        ):
            raise ValueError(
                "test_execution.status passed requires at least one completed "
                "validation target and all validation targets to pass"
            )
    elif execution_status == "not_run" and validation_target_statuses:
        raise ValueError(
            "test_execution.status not_run cannot contain executed validation commands"
        )
    elif execution_status == "unavailable" and commands:
        raise ValueError(
            "test_execution.status unavailable cannot contain command records"
        )
    elif execution_status == "stable_failure":
        if "stable_failure" not in validation_target_statuses or any(
            status not in {"passed", "stable_failure"}
            for status in validation_target_statuses
        ):
            raise ValueError(
                "test_execution.status stable_failure requires an unresolved "
                "stable validation target"
            )
    elif execution_status == "flaky_failure":
        if "flaky_failure" not in validation_target_statuses or any(
            status not in {"passed", "flaky_failure"}
            for status in validation_target_statuses
        ):
            raise ValueError(
                "test_execution.status flaky_failure requires an unresolved "
                "flaky validation target"
            )
    elif execution_status == "infrastructure_failure":
        if "infrastructure_failure" not in validation_target_statuses or any(
            status not in {"passed", "infrastructure_failure"}
            for status in validation_target_statuses
        ):
            raise ValueError(
                "test_execution.status infrastructure_failure requires an unresolved "
                "infrastructure-limited validation target"
            )

    warning_execution_statuses = {
        "stable_failure",
        "flaky_failure",
        "infrastructure_failure",
        "test_generation_error",
        "insufficient_evidence",
        "unavailable",
    }
    expected_verdict = (
        "FAIL"
        if any(
            finding["severity"] == "HIGH"
            for finding in findings + unlocated_findings
        )
        else "WARNING"
        if (
                findings
                or unlocated_findings
                or evidence_level
                in {"insufficient", "test_generation_error", "unavailable"}
            or execution_status in warning_execution_statuses
        )
        else "PASS"
    )
    if verdict != expected_verdict:
        raise ValueError(
            f"verdict {verdict} does not match findings and test_execution.status; "
            f"expected {expected_verdict}"
        )
    return document


def inline(value: Any) -> str:
    text = " ".join(str(value).split())
    return html.escape(text, quote=False).replace("|", "\\|").replace("`", "'")


CATEGORY_LABELS = {
    "algorithm": "算法错误",
    "business-logic": "业务逻辑错误",
    "state-management": "状态错误",
    "cache-consistency": "缓存一致性错误",
    "concurrency": "并发错误",
    "resource-lifecycle": "资源生命周期错误",
    "data-integrity": "数据完整性问题",
    "correctness": "正确性错误",
    "regression": "行为回归",
    "security": "安全问题",
    "api-compatibility": "接口兼容性问题",
    "performance": "性能问题",
    "test-gap": "测试缺口",
    "other": "其他问题",
}
SEVERITY_LABELS = {
    "HIGH": "高风险",
    "MEDIUM": "中风险",
    "LOW": "低风险",
}
VERDICT_LABELS = {
    "PASS": "通过",
    "WARNING": "警告",
    "FAIL": "失败",
}
COMMENT_VERDICT_LABELS = {
    "PASS": "通过",
    "WARNING": "需关注（非阻塞）",
    "FAIL": "失败",
}
TEST_EXECUTION_STATUS_LABELS = {
    "not_run": "未执行",
    "passed": "正式验证目标均已完成",
    "stable_failure": "可稳定复现的失败",
    "flaky_failure": "非确定性失败",
    "infrastructure_failure": "受环境限制，未完全执行",
    "test_generation_error": "测试生成失败",
    "insufficient_evidence": "证据不足",
    "unavailable": "未获得可信结果",
}
EVIDENCE_LEVEL_LABELS = {
    "not_needed": "无需额外动态验证",
    "sufficient": "证据充分",
    "insufficient": "证据不足",
    "test_generation_error": "测试生成失败",
    "unavailable": "未获得可信判断",
}
COMMAND_STATUS_LABELS = {
    "passed": "通过",
    "failed": "失败（尚未完成稳定性或基础设施归因）",
    "stable_failure": "可稳定复现的失败",
    "flaky_failure": "非确定性失败",
    "infrastructure_failure": "环境限制导致未完成",
    "not_executed": "未执行",
}
COMMAND_ROLE_LABELS = {
    "validation": "正式验证",
    "diagnostic": "诊断",
    "unclassified": "未分类",
}
CONSTRAINT_STATUS_LABELS = {
    "pass": "通过",
    "warning": "警告",
    "not_applicable": "不适用",
}
CHANGE_TYPE_LABELS = {
    "modified": "修改",
    "added": "新增",
    "deleted": "删除",
    "renamed": "重命名",
}
CHANGE_REQUEST_ASSESSMENT_LABELS = {
    "implemented": "已实现",
    "partially_implemented": "部分实现",
    "not_implemented": "未实现",
    "not_assessable": "无法判断",
    "not_applicable": "不适用",
}
BEHAVIOR_LABELS = {
    "normal": "正常路径",
    "boundary": "边界路径",
    "error": "错误路径",
    "compatibility": "兼容路径",
    "integration": "集成路径",
}
BEHAVIOR_ORDER = ("normal", "boundary", "error", "compatibility", "integration")
MAX_COMMENT_LENGTH = 58_000
MAX_COMMENT_ASSESSMENT_EVIDENCE_ITEMS = 4
MAX_COMMENT_TEST_SUMMARY_ITEMS = 6
MAX_COMMENT_VALIDATION_COMMAND_ITEMS = 6
MAX_COMMENT_VALIDATION_LIMIT_ITEMS = 6
MAX_COMMENT_RESIDUAL_RISK_ITEMS = 6
REPORT_NORMALIZATION_RISK_PREFIX = "报告完整性提醒："
COMMAND_RECORD_NORMALIZATION_RISK_RE = re.compile(
    r"^报告完整性提醒：[0-9]+ 条非零退出命令没有可匹配的用途说明"
)


def comment_inline(value: Any, limit: int = 2_000) -> str:
    text = inline(value).replace("@", "＠")
    if len(text) <= limit:
        return text
    return f"{text[: max(limit - 1, 0)]}…"


VALIDATION_SUMMARY_PREFIXES = (
    "Codex 说明：",
    "Codex 对验证证据的判断：",
    "Runner 校验：",
    "Runner 事实校验：",
)
INTERNAL_VALIDATION_ENUM_LABELS = {
    "not_needed": "不需要新增验证",
    "sufficient": "现有验证可支撑审查",
    "insufficient": "现有验证覆盖有限",
    "test_generation_error": "测试生成未完成",
    "not_run": "未执行新增命令",
    "passed": "所执行命令均成功",
    "stable_failure": "存在可稳定复现的失败",
    "flaky_failure": "重复执行结果不一致",
    "infrastructure_failure": "执行受运行环境限制",
    "insufficient_evidence": "失败记录尚不足以归因",
    "unavailable": "相关事实不可确认",
}
INTERNAL_VALIDATION_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:evidence_level|test_execution\.status|test_status)"
    r"\s*[:=]\s*("
    + "|".join(map(re.escape, INTERNAL_VALIDATION_ENUM_LABELS))
    + r")(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
INTERNAL_VALIDATION_ENUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(" + "|".join(map(re.escape, INTERNAL_VALIDATION_ENUM_LABELS))
    + r")(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
PUBLIC_NO_NEW_COMMAND_RE = re.compile(
    r"^(?:本次)?(?:不需要|无需|无须|没有必要|不必|不用|未|没有)"
    r"(?:再)?(?:执行|运行)?(?:任何)?"
    r"(?:新增(?:的)?|额外(?:的)?|新(?:的)?)(?:任何)?"
    r"(?:(?:验证|诊断|测试)(?:或(?:验证|诊断|测试))?)*命令[。.!！]?$"
)
INTERNAL_FAILURE_NARRATIVE_REPLACEMENTS = (
    (
        re.compile(r"结构化报告未通过 schema、固定格式或中文内容校验"),
        "自动审查结果整理阶段未能生成公开摘要",
    ),
    (
        re.compile(r"Codex (?:审查)?语义载荷未满足公开结构契约"),
        "Codex 自动审查结果整理阶段未能生成公开摘要",
    ),
    (
        re.compile(r"(?:Runner|自动检查) 生成的可信报告输入校验失败"),
        "Codex 自动审查结果汇总阶段未能核对代码差异与执行记录",
    ),
    (
        re.compile(r"(?:Runner|自动检查) 生成报告时内部契约校验失败"),
        "Codex 自动审查报告生成阶段未完成",
    ),
    (
        re.compile(r"(?:Runner|自动检查) 读取报告执行事实失败"),
        "Codex 自动审查验证结果汇总阶段未完成",
    ),
)
PUBLIC_INTERNAL_NARRATIVE_REPLACEMENTS = (
    (re.compile(r"\bunclassified\b", re.IGNORECASE), "用途未说明"),
    (re.compile(r"\bledger\b", re.IGNORECASE), "命令执行记录"),
    (re.compile(r"\btest_execution\.status\b", re.IGNORECASE), "正式验证状态"),
    (re.compile(r"\bmerge_recommendation\b", re.IGNORECASE), "合入建议"),
    (re.compile(r"\bverdict\b", re.IGNORECASE), "审查结论"),
    (re.compile(r"\bbuilder\b", re.IGNORECASE), "报告生成逻辑"),
    (re.compile(r"\brenderer\b", re.IGNORECASE), "公开评论生成逻辑"),
    (re.compile(r"\bschema\b", re.IGNORECASE), "报告格式"),
    (re.compile(r"\bcanonical\b", re.IGNORECASE), "标准报告"),
    (
        re.compile(r"\bderive_execution_status\b", re.IGNORECASE),
        "验证状态派生逻辑",
    ),
    (
        re.compile(r"\bpublic_unclassified_failure_items\b", re.IGNORECASE),
        "辅助检查公开说明逻辑",
    ),
)
PUBLIC_INTERNAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/workspace|/tmp)/[^\s`'\"，。；：！？（）()]+"
)
PUBLIC_CODE_SPAN_RE = re.compile(r"`([^`\r\n]+)`")


def replace_public_runner_term(text: str) -> str:
    text = re.sub(
        r"(?<![A-Za-z0-9_.-])Runner(?![A-Za-z0-9_.-])",
        "自动检查",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=自动检查)", "", text)
    return re.sub(r"(?<=自动检查)\s+(?=[\u3400-\u9fff，。；：！？])", "", text)


def public_code_span(match: re.Match[str]) -> str:
    text = " ".join(match.group(1).split())
    lowered = text.lower()
    if "/workspace/" in lowered or "/tmp/" in lowered:
        if "pytest" in lowered:
            return "定向测试命令"
        if re.search(r"(?:^|[\s/])(rg|grep)(?:\.exe)?(?:\s|$)", lowered):
            return "代码搜索命令"
        return "辅助检查命令"
    if re.match(r"^(?:python[0-9.]*\s+-m\s+pytest|pytest)(?:\s|$)", lowered):
        return "定向测试命令"
    if re.search(r"(?:^|[\s/])(rg|grep)(?:\.exe)?(?:\s|$)", lowered):
        return "代码搜索命令"
    if re.match(
        r"^(?:(?:/bin/)?(?:bash|sh|zsh)\s+-c\b|sed\b|cat\b|head\b|tail\b)",
        lowered,
    ):
        return "辅助信息读取命令"
    return match.group(0)


def public_narrative_text(
    value: str, identifier_descriptions: dict[str, str] | None = None
) -> str:
    text = value
    if identifier_descriptions is not None:
        text = INTERNAL_COMMENT_ID_RE.sub(
            lambda match: public_comment_identifier(match, identifier_descriptions),
            text,
        )
    text = PUBLIC_CODE_SPAN_RE.sub(public_code_span, text)
    text = PUBLIC_INTERNAL_PATH_RE.sub("任务内部路径", text)
    text = INTERNAL_VALIDATION_ASSIGNMENT_RE.sub(
        lambda match: INTERNAL_VALIDATION_ENUM_LABELS[match.group(1).lower()],
        text,
    )
    text = INTERNAL_VALIDATION_ENUM_RE.sub(
        lambda match: INTERNAL_VALIDATION_ENUM_LABELS[match.group(1).lower()],
        text,
    )
    for prefix in VALIDATION_SUMMARY_PREFIXES:
        text = text.replace(prefix, "")
    for pattern, replacement in INTERNAL_FAILURE_NARRATIVE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    for pattern, replacement in PUBLIC_INTERNAL_NARRATIVE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = text.removeprefix(REPORT_NORMALIZATION_RISK_PREFIX)
    return replace_public_runner_term(text).strip()


def public_validation_summary_item(
    value: str, identifier_descriptions: dict[str, str] | None = None
) -> str:
    text = public_narrative_text(value, identifier_descriptions)
    if PUBLIC_NO_NEW_COMMAND_RE.fullmatch("".join(text.split())):
        text = "本次未新增验证命令。"
    return text or "本次没有提供可公开展示的验证依据。"


PUBLIC_VALIDATION_LIMIT_RE = re.compile(
    r"尚未|未覆盖|未验证|未执行|未能|无法|未完成|"
    r"受.{0,20}限制|缺少.{0,20}(?:验证|测试|证据|环境)|"
    r"(?:验证|测试|证据).{0,20}(?:不足|有限)"
)


def is_public_validation_limit(value: str) -> bool:
    return bool(PUBLIC_VALIDATION_LIMIT_RE.search(value))


def unique_comment_items(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(item.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def limited_comment_items(
    items: list[str], limit: int, omitted_noun: str
) -> list[str]:
    unique_items = unique_comment_items(items)
    shown = unique_items[:limit]
    lines = [f"  - {comment_inline(item, 1_000)}" for item in shown]
    if len(unique_items) > len(shown):
        lines.append(
            f"  - 另有 {len(unique_items) - len(shown)} {omitted_noun}，请查看完整报告。"
        )
    return lines


def public_validation_artifact_items(test_execution: dict[str, Any]) -> list[str]:
    items: list[str] = []
    generated_files = test_execution["generated_test_files"]
    if generated_files:
        shown_files = generated_files[:3]
        paths = "、".join(comment_inline(path, 300) for path in shown_files)
        suffix = (
            f"，另有 {len(generated_files) - len(shown_files)} 个文件"
            if len(generated_files) > len(shown_files)
            else ""
        )
        items.append(
            f"生成了 {len(generated_files)} 个任务级测试文件：{paths}{suffix}。"
        )
    if not test_execution["commands"]:
        items.append(
            "本次命令执行事实不可确认。"
            if test_execution["status"] == "unavailable"
            else "本次未新增验证命令。"
        )
    return items


def exclude_seen_comment_items(
    items: list[str], seen_items: list[str]
) -> list[str]:
    seen = {comment_inline(item, 1_000) for item in seen_items}
    result: list[str] = []
    for item in items:
        normalized = comment_inline(item, 1_000)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(item)
    return result


def exclude_covered_comment_items(
    items: list[str], covering_items: list[str]
) -> list[str]:
    covering = [
        comment_inline(item, 1_000).rstrip("。！？；， ")
        for item in covering_items
    ]
    result: list[str] = []
    for item in items:
        normalized = comment_inline(item, 1_000).rstrip("。！？；， ")
        if any(
            normalized == cover
            or normalized in cover
            for cover in covering
        ):
            continue
        result.append(item)
    return result


def unresolved_diagnostic_groups(
    commands: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return unresolved_command_target_groups(commands, "diagnostic")


def unresolved_command_target_groups(
    commands: list[dict[str, Any]], role: str
) -> dict[str, list[dict[str, Any]]]:
    return {
        purpose: items
        for purpose, items in command_target_groups(commands, role).items()
        if command_target_status(items) != "passed"
    }


def public_unresolved_target_items(
    commands: list[dict[str, Any]],
    role: str,
    identifier_descriptions: dict[str, str] | None = None,
) -> list[str]:
    generic_evidence = {
        "执行结果来自可信CodexJSONL事件",
        "执行事实来自可信命令记录",
        "定向测试执行完成",
    }
    items: list[str] = []
    for purpose, target_commands in unresolved_command_target_groups(
        commands, role
    ).items():
        public_purpose = public_narrative_text(
            purpose, identifier_descriptions
        ).rstrip("。！？；， ")
        evidence_items = unique_comment_items(
            evidence
            for command in target_commands
            if command["exit_code"] != 0
            if (
                evidence := public_narrative_text(
                    command["evidence"], identifier_descriptions
                ).rstrip("。！？；， ")
            )
            and "".join(evidence.split()) not in generic_evidence
        )
        if evidence_items:
            items.append(
                f"{public_purpose}尚未完成；{evidence_items[-1]}。"
            )
        else:
            items.append(
                f"{public_purpose}尚未完成，对应目标的原因和影响仍待确认。"
            )
    return items


def public_unresolved_diagnostic_items(
    commands: list[dict[str, Any]],
    identifier_descriptions: dict[str, str] | None = None,
) -> list[str]:
    return public_unresolved_target_items(
        commands, "diagnostic", identifier_descriptions
    )


def public_comment_identifier_descriptions(
    document: dict[str, Any],
) -> dict[str, str]:
    descriptions = {
        f"FILE-{index:03d}": changed_file["path"]
        for index, changed_file in enumerate(document["changed_files"], start=1)
    }
    descriptions.update(
        {
            finding["id"]: public_narrative_text(finding["title"])
            for finding in document["findings"] + document["unlocated_findings"]
        }
    )
    descriptions.update(
        {
            test["id"]: public_narrative_text(test["description"])
            for test in document["suggested_tests"]
        }
    )
    descriptions.update(
        {
            command["id"]: public_narrative_text(command["purpose"])
            for command in document["test_execution"]["commands"]
        }
    )
    return descriptions


def public_validation_limit_items(
    document: dict[str, Any],
    args: argparse.Namespace | None = None,
    identifier_descriptions: dict[str, str] | None = None,
) -> list[str]:
    test_execution = document["test_execution"]
    execution_status = test_execution["status"]
    evidence_level = test_execution["evidence_level"]
    items: list[str] = []

    items.extend(
        public_unresolved_target_items(
            test_execution["commands"], "validation", identifier_descriptions
        )
    )
    items.extend(
        public_unresolved_diagnostic_items(
            test_execution["commands"], identifier_descriptions
        )
    )

    if execution_status == "stable_failure":
        items.append("可稳定复现的失败尚未经过修复后复测。")
    elif execution_status == "flaky_failure":
        items.append("重复执行结果不一致，仍需在可比且稳定的环境中复测。")
    elif execution_status == "infrastructure_failure":
        validation_target_statuses = [
            command_target_status(target_commands)
            for target_commands in command_target_groups(
                test_execution["commands"], "validation"
            ).values()
        ]
        if validation_target_statuses and all(
            status == "infrastructure_failure"
            for status in validation_target_statuses
        ):
            items.append("所执行的验证均受运行环境限制，当前没有完成预期覆盖。")
        else:
            items.append("部分验证受运行环境限制，当前没有完成全部预期覆盖。")
    elif execution_status == "test_generation_error":
        items.append("测试生成阶段未完成，当前没有形成预期的动态验证覆盖。")
    elif execution_status == "unavailable":
        items.append("预期验证是否执行及其结果仍待核对。")
    if evidence_level == "insufficient":
        items.append("现有验证尚未覆盖本次变更的全部风险。")
    elif evidence_level == "test_generation_error" and execution_status != "test_generation_error":
        items.append("Codex 报告测试生成过程未完成。")
    elif evidence_level == "unavailable":
        items.append("本次自动审查未形成完整的验证依据说明。")

    items.extend(
        "尚未执行："
        f"{public_narrative_text(test['description'], identifier_descriptions)}"
        for test in document["suggested_tests"]
    )
    if getattr(args, "backend_validation_scope", "full") == "frontend_only":
        items.append(
            "当前没有部署可供测试的厂商后端，未执行后端构建、JIT、"
            "FlagGems 和性能验证。"
        )
    elif getattr(args, "backend_validation_scope", "full") == "unavailable":
        items.append("本次后端验证范围不可确认。")
    if getattr(args, "constraint_status", "pass") == "warning":
        constraint_reason = public_validation_summary_item(
            getattr(args, "constraint_reason", "本次验证过程存在约束提醒。"),
            identifier_descriptions,
        )
        items.append(constraint_reason)
    return items


def render_report(document: dict[str, Any], args: argparse.Namespace) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assessment = document["change_request_assessment"]
    assessment_evidence = assessment_evidence_items(
        assessment["evidence"], "change_request_assessment.evidence"
    )
    if args.diff_mode == "merge-base":
        base_rows = [
            f"| 目标分支提交 | `{inline(args.requested_base_sha)}` |",
            f"| 实际审查起点（merge-base） | `{inline(args.base_sha)}` |",
        ]
    else:
        base_rows = [
            f"| 基础提交 | `{inline(args.base_sha)}` |",
        ]
    lines = [
        "# Codex AI 自动审查报告",
        "",
        "## 元数据",
        "",
        "| 字段 | 值 |",
        "| --- | --- |",
        "| 报告格式 | `triton-anchor-codex-ai-report/v3` |",
        f"| 分支 | `{inline(args.branch)}` |",
        *base_rows,
        f"| 测试提交 | `{inline(args.target_sha)}` |",
        f"| 测试提交类型 | `{inline(args.tested_sha_kind)}` |",
        *([f"| PR Head 提交 | `{inline(args.head_sha)}` |"] if args.head_sha else []),
        f"| 变更文件数 | {args.changed_file_count} |",
        f"| 生成时间（UTC） | `{generated_at}` |",
        "",
        "## 结论",
        "",
        f"**{VERDICT_LABELS[document['verdict']]}**",
        "",
        "## 摘要",
        "",
        inline(document["summary"]),
        "",
        "## 贡献者目标与实现情况",
        "",
        f"- 判断：{CHANGE_REQUEST_ASSESSMENT_LABELS[assessment['status']]}",
        f"- 修改目标：{inline(assessment['contributor_goal'])}",
        f"- 预期行为：{inline(assessment['expected_behavior'])}",
        f"- 实现情况：{inline(assessment['implementation_summary'])}",
        "- 判断依据：",
        *[f"  - {inline(item)}" for item in assessment_evidence],
        "",
        "## 合入建议",
        "",
        inline(document["merge_recommendation"]),
        "",
        "## 具体文件变更",
        "",
        "| 文件 | 类型 | 改动说明 | 影响 | 已执行验证或未执行原因 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for changed_file in document["changed_files"]:
        lines.append(
            f"| `{inline(changed_file['path'])}` | "
            f"{CHANGE_TYPE_LABELS[changed_file['change_type']]} | "
            f"{inline(changed_file['summary'])} | "
            f"{inline(changed_file['impact'])} | "
            f"{inline(changed_file['validation_strategy'])} |"
        )
    if not document["changed_files"]:
        lines.append("| 无 | 无 | 本次差异没有变更文件。 | 不适用。 | 不适用。 |")

    lines.extend([
        "",
        "## 行为覆盖",
        "",
        "| 路径 | 检查范围 | 验证策略 | 结果 |",
        "| --- | --- | --- | --- |",
    ])
    for behavior_name in BEHAVIOR_ORDER:
        behavior = document["behavior_coverage"][behavior_name]
        lines.append(
            f"| {BEHAVIOR_LABELS[behavior_name]} | "
            f"{inline(behavior['scope'])} | "
            f"{inline(behavior['strategy'])} | "
            f"{inline(behavior['result'])} |"
        )
    lines.extend(["", "## 关键问题", ""])

    findings = document["findings"]
    unlocated_findings = document["unlocated_findings"]
    if not findings and not unlocated_findings:
        lines.extend([public_no_findings_message(document), ""])
    for finding in findings:
        location = f"{inline(finding['file'])}:{inline(finding['line'])}"
        lines.extend([
            f"### {finding['id']}: {inline(finding['title'])}",
            "",
            "| 字段 | 值 |",
            "| --- | --- |",
            f"| 风险级别 | **{SEVERITY_LABELS[finding['severity']]}** |",
            f"| 类别 | {CATEGORY_LABELS[finding['category']]} |",
            f"| 位置 | `{location}` |",
            f"| 这段代码负责 | {inline(finding['code_role'])} |",
            f"| 证据 | {inline(finding['evidence'])} |",
            f"| 影响 | {inline(finding['impact'])} |",
            f"| 修复方向 | {inline(finding['fix_direction'])} |",
            "",
        ])
    for finding in unlocated_findings:
        trusted_file = finding["trusted_file"] or "未能映射可信变更文件"
        location = f"{inline(trusted_file)}；模型行号 {inline(finding['reported_line'])}"
        lines.extend([
            f"### {finding['id']}: {inline(finding['title'])}（定位待核对）",
            "",
            "| 字段 | 值 |",
            "| --- | --- |",
            f"| 风险级别 | **{SEVERITY_LABELS[finding['severity']]}** |",
            f"| 类别 | {CATEGORY_LABELS[finding['category']]} |",
            f"| 定位状态 | {inline(finding['location_issue'])} |",
            f"| 原始定位 | `{location}` |",
            f"| 代码职责 | {inline(finding['code_role'])} |",
            f"| 证据 | {inline(finding['evidence'])} |",
            f"| 影响 | {inline(finding['impact'])} |",
            f"| 修复方向 | {inline(finding['fix_direction'])} |",
            "",
        ])

    lines.extend(["## 建议测试", ""])
    tests = document["suggested_tests"]
    if not tests:
        lines.extend(["无。", ""])
    else:
        lines.extend([
            "| 编号 | 优先级 | 目标 | 说明 |",
            "| --- | --- | --- | --- |",
        ])
        for test in tests:
            lines.append(
                f"| {test['id']} | **{test['priority']}** | "
                f"`{inline(test['target'])}` | {inline(test['description'])} |"
            )
        lines.append("")

    test_execution = document["test_execution"]
    test_execution_summary = test_execution_summary_items(
        test_execution["summary"], "test_execution.summary"
    )
    lines.extend([
        "## 测试执行",
        "",
        f"- Codex 对验证证据的判断：{EVIDENCE_LEVEL_LABELS[test_execution['evidence_level']]}",
        f"- Runner 事实校验：{TEST_EXECUTION_STATUS_LABELS[test_execution['status']]}",
        "- 说明：",
        *[f"  - {inline(item)}" for item in test_execution_summary],
        "",
        "### 生成的测试文件",
        "",
    ])
    generated_test_files = test_execution["generated_test_files"]
    if generated_test_files:
        lines.extend(f"- `{inline(file_name)}`" for file_name in generated_test_files)
    else:
        lines.append("无。")
    lines.extend(["", "### 执行命令", ""])
    commands = test_execution["commands"]
    if commands:
        lines.extend([
            "| 编号 | 类型 | 功能 | 状态 | 退出码 | 耗时（秒） | 命令 | 证据 |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ])
        for command in commands:
            lines.append(
                f"| {command['id']} | "
                f"{COMMAND_ROLE_LABELS[command['role']]} | "
                f"{inline(command['purpose'])} | "
                f"{COMMAND_STATUS_LABELS[command['status']]} | "
                f"{command['exit_code']} | {command['duration_seconds']} | "
                f"`{inline(command['command'])}` | {inline(command['evidence'])} |"
            )
    else:
        lines.append("未记录执行命令。")
    lines.extend([
        "",
        "## 测试执行约束",
        "",
        f"- 状态：{CONSTRAINT_STATUS_LABELS[args.constraint_status]}",
        f"- 说明：{inline(args.constraint_reason)}",
        "",
        "## 剩余风险",
        "",
    ])
    risks = document["residual_risks"]
    if risks:
        lines.extend(f"- {inline(risk)}" for risk in risks)
    else:
        lines.append("未报告剩余风险。")
    lines.extend(["", "## 执行标记", "", "CODEX_AI_CI_COMPLETE", ""])
    return "\n".join(lines)


def deterministic_ci_comment_line(args: argparse.Namespace) -> str:
    status = getattr(args, "local_ci_status", "")
    execution_mode = getattr(args, "local_ci_execution_mode", "full")
    backend_scope = getattr(args, "backend_validation_scope", "full")
    if execution_mode == "codex_only":
        return (
            "按策略未执行确定性 CI；该状态不表示确定性测试通过，"
            "Codex AI 自动审查仍只提供补充意见。"
        )
    if execution_mode == "unavailable":
        return (
            "执行状态不可确认；当前不能据此判断确定性门禁结果，"
            "Codex AI 自动审查仍只提供补充意见。"
        )
    if backend_scope == "frontend_only":
        if status in {0, "0"}:
            return (
                "前端验证范围已通过；本次未执行厂商后端构建或运行验证，"
                "Codex AI 自动审查不改变这一门禁范围。"
            )
        if status:
            return (
                "前端验证范围未通过；本次未执行厂商后端构建或运行验证，"
                "最终仍以检查结果和复测为准。"
            )
        return (
            "前端验证结果尚未提供；本次未执行厂商后端构建或运行验证，"
            "Codex AI 自动审查只提供补充意见。"
        )
    if backend_scope == "unavailable":
        if status in {0, "0"}:
            return (
                "已执行范围通过，但后端验证范围不可确认；"
                "Codex AI 自动审查不改变这一事实。"
            )
        if status:
            return (
                "已执行范围未通过，且后端验证范围不可确认；"
                "最终仍以检查结果和复测为准。"
            )
        return (
            "检查结果和后端验证范围均不可确认；"
            "Codex AI 自动审查只提供补充意见。"
        )
    if status in {0, "0"}:
        return "已通过；Codex AI 自动审查只提供补充意见，不改变门禁结果。"
    if status:
        return "未通过；Codex AI 自动审查用于辅助定位原因，最终仍以检查结果和复测为准。"
    return "结果尚未提供；Codex AI 自动审查只提供补充意见。"


def public_no_findings_message(document: dict[str, Any]) -> str:
    test_execution = document["test_execution"]
    review_complete = (
        document["verdict"] == "PASS"
        and test_execution["evidence_level"] in {"not_needed", "sufficient"}
        and test_execution["status"] in {"not_run", "passed"}
    )
    if review_complete:
        return "基于当前代码差异和验证证据，本次审查未发现需要处理的具体代码缺陷。"
    return (
        "本次未形成可确认的具体代码问题；"
        "验证限制或未覆盖项仍需结合下文核对。"
    )


def has_public_validation_limitations(
    document: dict[str, Any], args: argparse.Namespace
) -> bool:
    test_execution = document["test_execution"]
    return bool(
        test_execution["evidence_level"]
        in {"insufficient", "test_generation_error", "unavailable"}
        or test_execution["status"]
        in {
            "stable_failure",
            "flaky_failure",
            "infrastructure_failure",
            "test_generation_error",
            "insufficient_evidence",
            "unavailable",
        }
        or unresolved_diagnostic_groups(test_execution["commands"])
        or document["suggested_tests"]
        or getattr(args, "local_ci_execution_mode", "full") != "full"
        or getattr(args, "backend_validation_scope", "full") != "full"
        or getattr(args, "constraint_status", "pass") == "warning"
    )


def render_comment(document: dict[str, Any], args: argparse.Namespace) -> str:
    findings = sorted(
        document["findings"],
        key=lambda finding: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[
            finding["severity"]
        ],
    )
    test_execution = document["test_execution"]
    test_execution_summary = test_execution_summary_items(
        test_execution["summary"], "test_execution.summary"
    )
    identifier_descriptions = public_comment_identifier_descriptions(document)
    assessment = document["change_request_assessment"]
    assessment_evidence = assessment_evidence_items(
        assessment["evidence"], "change_request_assessment.evidence"
    )
    shown_assessment_evidence = assessment_evidence[
        :MAX_COMMENT_ASSESSMENT_EVIDENCE_ITEMS
    ]
    assessment_evidence_lines = [
        "  - "
        f"{comment_inline(public_narrative_text(item, identifier_descriptions), 1_000)}"
        for item in shown_assessment_evidence
    ]
    if len(assessment_evidence) > len(shown_assessment_evidence):
        assessment_evidence_lines.append(
            f"  - 另有 {len(assessment_evidence) - len(shown_assessment_evidence)} "
            "条判断依据，请查看完整报告。"
        )
    review_complete = test_execution["evidence_level"] != "unavailable"
    lines = [
        "## Codex AI 自动审查",
        "",
        "> Codex AI 自动审查仅供参考且不阻塞合入；本地确定性 CI 检查结果才是合入门禁。",
        "",
        "### 审查摘要",
        "",
    ]
    if review_complete:
        lines.append(
            "- Codex AI 审查结论："
            f"**{COMMENT_VERDICT_LABELS[document['verdict']]}**"
        )
    lines.extend(
        [
            f"- 本地确定性 CI 检查：{deterministic_ci_comment_line(args)}",
            "- 合入建议："
            f"{comment_inline(public_narrative_text(document['merge_recommendation'], identifier_descriptions), 1_000)}",
            "",
            comment_inline(
                public_narrative_text(document["summary"], identifier_descriptions)
            ),
            "",
            "### 贡献者目标与实现情况",
            "",
            f"- 判断：**{CHANGE_REQUEST_ASSESSMENT_LABELS[assessment['status']]}**",
            "- 贡献者目标："
            f"{comment_inline(public_narrative_text(assessment['contributor_goal'], identifier_descriptions), 1_500)}",
            "- 预期效果："
            f"{comment_inline(public_narrative_text(assessment['expected_behavior'], identifier_descriptions), 1_500)}",
            "- 当前实现情况："
            f"{comment_inline(public_narrative_text(assessment['implementation_summary'], identifier_descriptions), 2_000)}",
            "- 判断依据：",
            *assessment_evidence_lines,
            "",
            "### 需要处理的问题",
            "",
        ]
    )
    unlocated_findings = document["unlocated_findings"]
    if not findings and not unlocated_findings:
        lines.extend([
            public_no_findings_message(document),
            "",
        ])
    else:
        shown_findings = findings[:5]
        for index, finding in enumerate(shown_findings, start=1):
            category = CATEGORY_LABELS.get(finding["category"], "其他问题")
            severity = SEVERITY_LABELS[finding["severity"]]
            location = (
                f"{comment_inline(finding['file'], 500)}:"
                f"{comment_inline(finding['line'], 100)}"
            )
            lines.append(
                f"#### {index}. [{severity}] {comment_inline(finding['title'], 400)}"
            )
            lines.append("")
            lines.append(f"- 问题类型：{category}")
            lines.append(f"- 代码定位：`{location}`")
            lines.append(f"- 这段代码负责：{comment_inline(finding['code_role'], 800)}")
            lines.append(f"- 核心证据：{comment_inline(finding['evidence'], 1_000)}")
            lines.append(f"- 影响：{comment_inline(finding['impact'], 1_000)}")
            lines.append(f"- 建议：{comment_inline(finding['fix_direction'], 1_000)}")
            lines.append("")
        remaining_slots = 5 - len(shown_findings)
        shown_unlocated = unlocated_findings[:remaining_slots]
        for index, finding in enumerate(
            shown_unlocated, start=len(shown_findings) + 1
        ):
            category = CATEGORY_LABELS.get(finding["category"], "其他问题")
            severity = SEVERITY_LABELS[finding["severity"]]
            trusted_file = finding["trusted_file"] or "未能映射可信变更文件"
            lines.extend([
                f"#### {index}. [{severity}·定位待核对] "
                f"{comment_inline(finding['title'], 400)}",
                "",
                f"- 问题类型：{category}",
                f"- 原始定位：`{comment_inline(trusted_file, 500)}；"
                f"模型行号 {comment_inline(finding['reported_line'], 100)}`",
                f"- 定位状态：{comment_inline(finding['location_issue'], 800)}",
                f"- 这段代码负责：{comment_inline(finding['code_role'], 800)}",
                f"- 核心证据：{comment_inline(finding['evidence'], 1_000)}",
                f"- 影响：{comment_inline(finding['impact'], 1_000)}",
                f"- 建议：{comment_inline(finding['fix_direction'], 1_000)}",
                "",
            ])
        shown_count = len(shown_findings) + len(shown_unlocated)
        total_count = len(findings) + len(unlocated_findings)
        if total_count > shown_count:
            lines.extend([
                f"另有 {total_count - shown_count} 个问题，请查看完整报告。",
                "",
            ])

    validation_summary_items = [
        public_validation_summary_item(item, identifier_descriptions)
        for item in test_execution_summary
    ]
    validation_basis_items = [
        item for item in validation_summary_items if not is_public_validation_limit(item)
    ]
    validation_summary_limit_items = [
        item for item in validation_summary_items if is_public_validation_limit(item)
    ]
    validation_artifact_items = public_validation_artifact_items(test_execution)
    validation_basis_items = unique_comment_items(validation_basis_items)
    validation_artifact_items = exclude_seen_comment_items(
        unique_comment_items(validation_artifact_items), validation_basis_items
    )
    if (
        not validation_basis_items
        and not validation_artifact_items
        and test_execution["status"] == "not_run"
    ):
        validation_artifact_items.append("本次未新增验证命令。")
    derived_validation_limit_items = public_validation_limit_items(
        document, args, identifier_descriptions
    )
    validation_summary_limit_items = exclude_covered_comment_items(
        validation_summary_limit_items, derived_validation_limit_items
    )
    derived_validation_limit_items = exclude_covered_comment_items(
        derived_validation_limit_items, validation_summary_limit_items
    )
    validation_limit_items = unique_comment_items(
        validation_summary_limit_items + derived_validation_limit_items
    )
    has_reported_validation_limits = bool(validation_limit_items)
    if not validation_limit_items:
        validation_limit_items = ["本次未报告额外的验证限制或未覆盖项。"]
    lines.extend([
        "### 验证情况",
        "",
        "- 验证内容与结果：",
        *limited_comment_items(
            validation_basis_items,
            MAX_COMMENT_TEST_SUMMARY_ITEMS,
            "条验证说明",
        ),
        *limited_comment_items(
            validation_artifact_items,
            MAX_COMMENT_VALIDATION_COMMAND_ITEMS,
            "条测试产物或执行说明",
        ),
        "- 限制与未覆盖：",
        *limited_comment_items(
            validation_limit_items,
            MAX_COMMENT_VALIDATION_LIMIT_ITEMS,
            "项限制或未覆盖内容",
        ),
        "",
    ])

    lines.extend([
        "### 剩余风险",
        "",
    ])
    residual_risks = [
        risk
        for risk in document["residual_risks"]
        if not risk.startswith(REPORT_NORMALIZATION_RISK_PREFIX)
    ]
    if residual_risks:
        shown_residual_risks = residual_risks[:MAX_COMMENT_RESIDUAL_RISK_ITEMS]
        lines.extend(
            "- "
            f"{comment_inline(public_narrative_text(risk, identifier_descriptions), 1_000)}"
            for risk in shown_residual_risks
        )
        if len(residual_risks) > len(shown_residual_risks):
            lines.append(
                f"- 另有 {len(residual_risks) - len(shown_residual_risks)} "
                "项剩余风险，请查看完整报告。"
            )
    else:
        lines.append(
            "除上述验证限制外，本次未报告其他剩余风险。"
            if has_reported_validation_limits
            or has_public_validation_limitations(document, args)
            else "本次未报告剩余风险。"
        )
    lines.append("")

    lines.extend([
        "### 变更文件",
        "",
        "<details>",
        "<summary>查看变更文件</summary>",
        "",
        "| 文件 | 类型 | 改动说明 | 影响 |",
        "| --- | --- | --- | --- |",
    ])
    table_suffix = ["", "</details>", ""]
    changed_files = document["changed_files"]
    if not changed_files:
        lines.append("| 无 | 无 | 本次差异没有变更文件。 | 不适用。 |")
    else:
        for index, changed_file in enumerate(changed_files):
            public_summary = public_narrative_text(
                changed_file["summary"], identifier_descriptions
            )
            public_impact = public_narrative_text(
                changed_file["impact"], identifier_descriptions
            )
            row = (
                f"| `{comment_inline(changed_file['path'], 500)}` | "
                f"{CHANGE_TYPE_LABELS[changed_file['change_type']]} | "
                f"{comment_inline(public_summary, 800)} | "
                f"{comment_inline(public_impact, 800)} |"
            )
            candidate = "\n".join([*lines, row, *table_suffix])
            if len(candidate) > MAX_COMMENT_LENGTH:
                remaining = len(changed_files) - index
                lines.append(
                    f"| 其余 {remaining} 个文件 | 省略 | "
                    "评论长度接近 GitHub 限制，请查看完整报告。 | "
                    "完整验证策略保留在完整报告中。 |"
                )
                break
            lines.append(row)
    lines.extend(table_suffix)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        input_path = Path(args.input)
        manifest_path = Path(args.changed_files_manifest)
        expected_files = validate_changed_files_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if len(expected_files) != args.changed_file_count:
            raise ValueError(
                "changed file count does not match manifest; "
                f"count={args.changed_file_count}, manifest={len(expected_files)}"
            )
        repository_root = None
        if args.repository_root:
            repository_root = Path(args.repository_root).resolve()
            if not repository_root.is_dir():
                raise ValueError("repository_root must be an existing directory")
        document = validate_report(
            json.loads(input_path.read_text(encoding="utf-8")),
            expected_files,
            repository_root,
        )
        require_chinese_string(args.constraint_reason, "constraint_reason")
        rendered = render_report(document, args)
        output_path = Path(args.output)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(rendered, encoding="utf-8")
        temporary_path.replace(output_path)
        comment_path = Path(args.comment_output)
        comment_temporary_path = comment_path.with_suffix(comment_path.suffix + ".tmp")
        comment_temporary_path.write_text(render_comment(document, args), encoding="utf-8")
        comment_temporary_path.replace(comment_path)
        print(document["verdict"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"Invalid Codex AI report: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
