#!/usr/bin/env python3
"""Build canonical Codex AI report v3 from a small analysis payload and trusted facts."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from typing import Any


CODEX_AI_SCRIPT_DIR = Path(__file__).resolve().parent
if str(CODEX_AI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_AI_SCRIPT_DIR))
SHARED_SCRIPT_DIR = CODEX_AI_SCRIPT_DIR.parent / "shared"
if str(SHARED_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPT_DIR))
from finding_locations import parse_finding_line_range  # noqa: E402
from codex_jsonl_evidence import normalize_command  # noqa: E402


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
FILE_ID_RE = re.compile(r"FILE-[0-9]{3,}")
INTERNAL_ID_RE = re.compile(r"\b(AI|TEST|RUN)-0*([1-9][0-9]*)\b[ \t]*", re.IGNORECASE)
SEVERITIES = {"HIGH", "MEDIUM", "LOW"}
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
ASSESSMENT_STATUSES = {
    "implemented",
    "partially_implemented",
    "not_implemented",
    "not_assessable",
    "not_applicable",
}
EVIDENCE_LEVELS = {
    "not_needed",
    "sufficient",
    "insufficient",
    "test_generation_error",
}
FAILURE_CLASSIFICATIONS = {
    "none",
    "product",
    "flaky",
    "infrastructure",
    "unknown",
}
COMMAND_ROLES = {"validation", "diagnostic"}
WARNING_EXECUTION_STATUSES = {
    "stable_failure",
    "flaky_failure",
    "infrastructure_failure",
    "test_generation_error",
    "insufficient_evidence",
}
REPORT_NORMALIZATION_RISK_PREFIX = "报告完整性提醒："
BEHAVIOR_LABELS = {
    "normal": "正常路径",
    "boundary": "边界路径",
    "error": "错误路径",
    "compatibility": "兼容路径",
    "integration": "集成路径",
}
ANALYSIS_KEYS = {
    "summary",
    "merge_recommendation",
    "change_request_assessment",
    "changed_files",
    "behavior_coverage",
    "findings",
    "suggested_tests",
    "residual_risks",
    "test_assessment",
}
ASSESSMENT_KEYS = {
    "status",
    "contributor_goal",
    "expected_behavior",
    "implementation_summary",
    "evidence",
}
CHANGED_FILE_KEYS = {"file_id", "summary", "impact", "validation_strategy"}
BEHAVIOR_ITEM_KEYS = {"scope", "strategy", "result"}
FINDING_KEYS = {
    "severity",
    "category",
    "file_id",
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
SUGGESTED_TEST_KEYS = {"priority", "target", "description"}
TEST_ASSESSMENT_KEYS = {"evidence_level", "summary", "commands"}
COMMAND_ANNOTATION_KEYS = {
    "command",
    "role",
    "purpose",
    "evidence",
    "failure_classification",
}


class InvalidFindingLocation(ValueError):
    """A model-provided finding location cannot be mapped to trusted source."""


class InvalidTrustedReportInput(ValueError):
    """A runner-produced input cannot be trusted for canonical report building."""


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def require_array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string")
    return value


def require_exact_keys(
    document: dict[str, Any], expected: set[str], location: str
) -> None:
    actual = set(document)
    if actual != expected:
        raise ValueError(
            f"{location} has invalid keys; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def text_or_default(value: Any, default: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    text = value.strip()
    if CHINESE_RE.search(text) is None:
        return f"Codex 原始说明：{text}"
    return text


def unique_in_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def command_purpose_or_default(value: Any) -> str:
    text = text_or_default(value, "Codex 执行的验证或诊断命令")
    replacements = {"AI": "相关问题", "TEST": "建议测试", "RUN": "相关验证"}
    text = INTERNAL_ID_RE.sub(
        lambda match: replacements[match.group(1).upper()], text
    ).strip()
    return text[:120].rstrip()


def normalized_repo_path(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{location} must be a normalized repository-relative path")
    return value


def load_manifest(path: Path) -> list[dict[str, str]]:
    document = load_json(path)
    if not isinstance(document, list):
        raise ValueError("changed_files_manifest must be an array")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(document):
        location = f"changed_files_manifest[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{location} must be an object")
        change_type = raw.get("change_type")
        if change_type not in {"added", "modified", "deleted", "renamed"}:
            raise ValueError(f"{location}.change_type is invalid")
        expected = (
            {"path", "change_type", "previous_path"}
            if change_type == "renamed"
            else {"path", "change_type"}
        )
        if set(raw) != expected:
            raise ValueError(f"{location} has invalid keys")
        item = {
            "path": normalized_repo_path(raw["path"], f"{location}.path"),
            "change_type": change_type,
        }
        if item["path"] in seen:
            raise ValueError(f"duplicate manifest path: {item['path']}")
        seen.add(item["path"])
        if change_type == "renamed":
            item["previous_path"] = normalized_repo_path(
                raw["previous_path"], f"{location}.previous_path"
            )
        result.append(item)
    return result


def prepare_manifest(input_path: Path, output_path: Path) -> None:
    write_json(
        output_path,
        [
            {"file_id": f"FILE-{index:03d}", **item}
            for index, item in enumerate(load_manifest(input_path), start=1)
        ],
    )


def parse_generated_archive(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    generated: list[str] = []
    with tarfile.open(path, mode="r:gz") as archive:
        for index, member in enumerate(archive.getmembers()):
            location = f"generated archive member[{index}]"
            relative = normalized_repo_path(member.name, location)
            if not member.isfile():
                raise ValueError(f"{location} must be a regular file: {relative}")
            if relative in generated:
                raise ValueError(f"duplicate generated archive path: {relative}")
            generated.append(relative)
    return generated


def is_test_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    filename = path.name.lower()
    directory_parts = {part.lower() for part in path.parts[:-1]}
    return (
        bool(directory_parts & {"test", "tests", "generated_tests"})
        or filename.startswith("test_")
        or filename.startswith("test-")
        or re.search(r"(?:^|[_-])tests?(?:[._-]|$)", filename) is not None
    )


def load_command_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    document = load_json(path)
    if not isinstance(document, list):
        raise ValueError("command_ledger must be an array")
    ledger: list[dict[str, Any]] = []
    for index, raw in enumerate(document):
        location = f"command_ledger[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{location} must be an object")
        command = raw.get("command")
        exit_code = raw.get("exit_code")
        duration = raw.get("duration_seconds", 0.0)
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"{location}.command must be a non-empty string")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            raise ValueError(f"{location}.exit_code must be an integer")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or duration < 0
        ):
            raise ValueError(f"{location}.duration_seconds must be non-negative")
        ledger.append(
            {
                "command": command.strip(),
                "exit_code": exit_code,
                "duration_seconds": round(float(duration), 3),
            }
        )
    return ledger


def validate_line(value: Any, location: str, file_path: str, root: Path) -> str:
    if not isinstance(value, str):
        raise InvalidFindingLocation(f"{location} must be a line number or range")
    line_range = parse_finding_line_range(value)
    if line_range is None:
        raise InvalidFindingLocation(
            f"{location} must be a positive, ordered line or line range"
        )
    _, end = line_range
    try:
        repository_root = root.resolve(strict=True)
    except OSError as exc:
        raise InvalidFindingLocation(
            f"{location} cannot resolve the review checkout"
        ) from exc
    candidate = repository_root.joinpath(*PurePosixPath(file_path).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository_root)
    except (OSError, ValueError) as exc:
        raise InvalidFindingLocation(
            f"{location} references an unreadable changed file"
        ) from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise InvalidFindingLocation(
            f"{location} references a non-regular changed file"
        )
    line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
    if end > line_count:
        raise InvalidFindingLocation(f"{location} exceeds the changed file line count")
    return value


def semantic_command_annotations(
    analysis: dict[str, Any]
) -> dict[str, deque[dict[str, Any]]]:
    assessment = require_object(analysis["test_assessment"], "test_assessment")
    result: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for index, value in enumerate(
        require_array(assessment["commands"], "test_assessment.commands")
    ):
        location = f"test_assessment.commands[{index}]"
        raw = require_object(value, location)
        require_exact_keys(raw, COMMAND_ANNOTATION_KEYS, location)
        command = require_string(raw["command"], f"{location}.command")
        role = require_string(raw["role"], f"{location}.role")
        if role not in COMMAND_ROLES:
            raise ValueError(f"{location}.role is invalid")
        purpose = require_string(raw["purpose"], f"{location}.purpose")
        evidence = require_string(raw["evidence"], f"{location}.evidence")
        classification = require_string(
            raw["failure_classification"], f"{location}.failure_classification"
        )
        if classification not in FAILURE_CLASSIFICATIONS:
            raise ValueError(f"{location}.failure_classification is invalid")
        normalized_command = normalize_command(command)
        if normalized_command is None:
            continue
        result[normalized_command].append(
            {
                "role": role,
                "purpose": command_purpose_or_default(purpose),
                "evidence": text_or_default(
                    evidence, "执行结果来自可信 Codex JSONL 事件。"
                ),
                "failure_classification": classification,
            }
        )
    return result


def build_commands(
    analysis: dict[str, Any] | None, ledger: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    annotations = (
        semantic_command_annotations(analysis)
        if analysis is not None
        else defaultdict(deque)
    )
    commands: list[dict[str, Any]] = []
    classifications: list[str] = []
    for index, fact in enumerate(ledger, start=1):
        command_key = normalize_command(fact["command"]) or fact["command"]
        if annotations[command_key]:
            annotation_queue = annotations[command_key]
            annotation = (
                annotation_queue.popleft()
                if len(annotation_queue) > 1
                else annotation_queue[0]
            )
        else:
            annotation = {
                "role": "unclassified",
                "purpose": "Codex 执行的验证或诊断命令",
                "evidence": "执行结果来自可信 Codex JSONL 事件。",
                "failure_classification": (
                    "none" if fact["exit_code"] == 0 else "unknown"
                ),
            }
        classification = annotation["failure_classification"]
        if fact["exit_code"] == 0:
            classification = "none"
        classifications.append(classification)
        commands.append(
            {
                "id": f"RUN-{index:03d}",
                "role": annotation["role"],
                "purpose": annotation["purpose"],
                "command": fact["command"],
                "exit_code": fact["exit_code"],
                "duration_seconds": fact["duration_seconds"],
                "status": "passed" if fact["exit_code"] == 0 else "failed",
                "evidence": annotation["evidence"],
            }
        )

    failed_indexes = [
        index for index, command in enumerate(commands) if command["exit_code"] != 0
    ]
    if not failed_indexes:
        return commands, classifications
    for index in failed_indexes:
        if classifications[index] == "infrastructure":
            commands[index]["status"] = "infrastructure_failure"

    outcomes: dict[str, list[int]] = defaultdict(list)
    for command in commands:
        outcomes[command["command"]].append(command["exit_code"])
    flaky = {
        text
        for text, exits in outcomes.items()
        if any(code == 0 for code in exits) and any(code != 0 for code in exits)
    }
    for index in failed_indexes:
        if (
            commands[index]["command"] in flaky
            and classifications[index] != "infrastructure"
        ):
            commands[index]["status"] = "flaky_failure"

    failure_indexes_by_text: dict[str, list[int]] = defaultdict(list)
    for index in failed_indexes:
        failure_indexes_by_text[commands[index]["command"]].append(index)
    stable = {
        text
        for text, indexes in failure_indexes_by_text.items()
        if len(indexes) >= 2
        and all(classifications[index] != "infrastructure" for index in indexes)
    }
    for index in failed_indexes:
        if (
            commands[index]["status"] == "failed"
            and commands[index]["command"] in stable
            and classifications[index] != "infrastructure"
        ):
            commands[index]["status"] = "stable_failure"
    return commands, classifications


def derive_execution_status(
    evidence_level: str,
    commands: list[dict[str, Any]],
) -> str:
    if evidence_level == "test_generation_error":
        return "test_generation_error"
    validation_commands = [
        command for command in commands if command["role"] == "validation"
    ]
    if not validation_commands:
        if evidence_level == "unavailable" and commands:
            return "insufficient_evidence"
        return "not_run"
    targets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for command in validation_commands:
        targets[command["purpose"]].append(command)

    unresolved_statuses: list[str] = []
    for target_commands in targets.values():
        methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for command in target_commands:
            methods[command["command"]].append(command)
        failed_indexes = [
            index
            for index, command in enumerate(target_commands)
            if command["exit_code"] != 0
        ]
        clean_methods = {
            command_text
            for command_text, method_commands in methods.items()
            if method_commands
            and all(command["exit_code"] == 0 for command in method_commands)
        }
        if not failed_indexes or any(
            index > failed_indexes[-1]
            and command["command"] in clean_methods
            for index, command in enumerate(target_commands)
        ):
            continue

        failed_statuses = {
            command["status"]
            for command in target_commands
            if command["exit_code"] != 0
        }
        if failed_statuses == {"infrastructure_failure"}:
            unresolved_statuses.append("infrastructure_failure")
        elif failed_statuses == {"flaky_failure"}:
            unresolved_statuses.append("flaky_failure")
        elif failed_statuses == {"stable_failure"}:
            unresolved_statuses.append("stable_failure")
        else:
            unresolved_statuses.append("insufficient_evidence")

    if not unresolved_statuses:
        return "passed"
    if len(set(unresolved_statuses)) == 1:
        return unresolved_statuses[0]
    return "insufficient_evidence"


def normalize_assessment(analysis: dict[str, Any]) -> dict[str, Any]:
    raw = require_object(
        analysis["change_request_assessment"], "change_request_assessment"
    )
    require_exact_keys(raw, ASSESSMENT_KEYS, "change_request_assessment")
    status = require_string(raw["status"], "change_request_assessment.status")
    if status not in ASSESSMENT_STATUSES:
        raise ValueError("change_request_assessment.status is invalid")
    contributor_goal = require_string(
        raw["contributor_goal"], "change_request_assessment.contributor_goal"
    )
    expected_behavior = require_string(
        raw["expected_behavior"], "change_request_assessment.expected_behavior"
    )
    implementation_summary = require_string(
        raw["implementation_summary"],
        "change_request_assessment.implementation_summary",
    )
    raw_evidence = require_array(
        raw["evidence"], "change_request_assessment.evidence"
    )
    checked_evidence = [
        require_string(item, f"change_request_assessment.evidence[{index}]")
        for index, item in enumerate(raw_evidence)
    ]
    evidence = [
        text_or_default(
            item,
            "现有证据不足，无法进一步确认贡献者声明。",
        )
        for item in unique_in_order(checked_evidence)[:8]
    ]
    if not evidence:
        evidence = ["现有证据不足，无法进一步确认贡献者声明。"]
    return {
        "status": status,
        "contributor_goal": text_or_default(
            contributor_goal, "当前上下文未提供可确认的贡献者目标。"
        ),
        "expected_behavior": text_or_default(
            expected_behavior, "当前上下文未提供可确认的预期行为。"
        ),
        "implementation_summary": text_or_default(
            implementation_summary, "本轮依据代码差异完成了辅助审查。"
        ),
        "evidence": evidence,
    }


def build_changed_files(
    analysis: dict[str, Any], manifest: list[dict[str, str]], commands: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    semantic_by_id: dict[str, dict[str, Any]] = {}
    ignored_count = 0
    expected_ids = {f"FILE-{index:03d}" for index in range(1, len(manifest) + 1)}
    for index, value in enumerate(
        require_array(analysis["changed_files"], "changed_files")
    ):
        location = f"changed_files[{index}]"
        raw = require_object(value, location)
        require_exact_keys(raw, CHANGED_FILE_KEYS, location)
        file_id_value = raw["file_id"]
        file_id = file_id_value if isinstance(file_id_value, str) else ""
        for key in {"summary", "impact", "validation_strategy"}:
            require_string(raw[key], f"{location}.{key}")
        if (
            FILE_ID_RE.fullmatch(file_id) is None
            or file_id not in expected_ids
            or file_id in semantic_by_id
        ):
            ignored_count += 1
            continue
        semantic_by_id[file_id] = raw
    missing_count = len(expected_ids - set(semantic_by_id))
    warnings = []
    if missing_count:
        warnings.append(
            f"Codex 的逐文件语义说明缺少 {missing_count} 个可信变更文件；"
            "报告已按 Git 清单保留这些文件，相关影响仍需人工核对。"
        )
    if ignored_count:
        warnings.append(
            f"Codex 的逐文件语义说明包含 {ignored_count} 个重复或无法映射的文件引用；"
            "这些引用未作为可信文件说明使用。"
        )
    changed_files = []
    for index, item in enumerate(manifest, start=1):
        semantic = semantic_by_id.get(f"FILE-{index:03d}", {})
        changed_files.append(
            {
                "path": item["path"],
                "change_type": item["change_type"],
                "summary": text_or_default(
                    semantic.get("summary"),
                    "Codex 未提供该文件的独立语义说明，已按可信 Git 清单保留。",
                ),
                "impact": text_or_default(
                    semantic.get("impact"),
                    "该文件的具体行为影响仍需结合代码差异人工核对。",
                ),
                "validation_strategy": text_or_default(
                    semantic.get("validation_strategy"),
                    (
                        "结合代码差异和本轮已执行的验证命令进行检查。"
                        if commands
                        else "未执行：本轮依据代码差异和已有 CI 证据完成审查。"
                    ),
                ),
            }
        )
    return changed_files, warnings


def build_behavior_coverage(analysis: dict[str, Any]) -> dict[str, dict[str, str]]:
    raw_coverage = require_object(analysis["behavior_coverage"], "behavior_coverage")
    require_exact_keys(raw_coverage, set(BEHAVIOR_LABELS), "behavior_coverage")
    result: dict[str, dict[str, str]] = {}
    for name, label in BEHAVIOR_LABELS.items():
        location = f"behavior_coverage.{name}"
        raw = require_object(raw_coverage[name], location)
        require_exact_keys(raw, BEHAVIOR_ITEM_KEYS, location)
        for key in BEHAVIOR_ITEM_KEYS:
            require_string(raw[key], f"{location}.{key}")
        result[name] = {
            "scope": text_or_default(
                raw.get("scope"), f"检查本次差异涉及的{label}。"
            ),
            "strategy": text_or_default(
                raw.get("strategy"),
                "结合代码差异、已有 CI 证据和本轮定向命令进行审查。",
            ),
            "result": text_or_default(
                raw.get("result"),
                "具体结果已汇总在审查摘要、关键问题和剩余风险中。",
            ),
        }
    return result


def build_findings(
    analysis: dict[str, Any],
    file_by_id: dict[str, dict[str, str]],
    repository_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    findings: list[dict[str, Any]] = []
    unlocated_findings: list[dict[str, Any]] = []
    for index, value in enumerate(
        require_array(analysis["findings"], "findings"), start=1
    ):
        location = f"findings[{index - 1}]"
        raw = require_object(value, location)
        require_exact_keys(raw, FINDING_KEYS, location)
        file_id_value = raw["file_id"]
        file_id = file_id_value if isinstance(file_id_value, str) else ""
        severity = require_string(raw["severity"], f"{location}.severity")
        category = require_string(raw["category"], f"{location}.category")
        if severity not in SEVERITIES:
            raise ValueError(f"{location}.severity is invalid")
        if category not in CATEGORIES:
            raise ValueError(f"{location}.category is invalid")
        for key in {"code_role", "title", "evidence", "impact", "fix_direction"}:
            require_string(raw[key], f"{location}.{key}")
        trusted = file_by_id.get(file_id)
        if (
            FILE_ID_RE.fullmatch(file_id) is None
            or trusted is None
            or trusted["change_type"] == "deleted"
        ):
            unlocated_findings.append(
                build_unlocated_finding(
                    index,
                    raw,
                    trusted_file="",
                    location_issue="模型提供的文件引用无法映射到可信的未删除变更文件。",
                )
            )
            continue
        try:
            line = validate_line(
                raw["line"], f"{location}.line", trusted["path"], repository_root
            )
        except InvalidFindingLocation:
            unlocated_findings.append(
                build_unlocated_finding(
                    index,
                    raw,
                    trusted_file=trusted["path"],
                    location_issue="模型提供的行号无法在可信审查 checkout 中验证。",
                )
            )
            continue
        findings.append(
            {
                "id": f"AI-{index:03d}",
                "severity": severity,
                "category": category,
                "file": trusted["path"],
                "line": line,
                "code_role": text_or_default(raw.get("code_role"), "该位置参与本次变更行为。"),
                "title": text_or_default(raw.get("title"), "需要检查的代码问题"),
                "evidence": text_or_default(raw.get("evidence"), "Codex 未提供完整问题证据。"),
                "impact": text_or_default(raw.get("impact"), "可能影响本次变更涉及的行为。"),
                "fix_direction": text_or_default(raw.get("fix_direction"), "请结合该位置补充修复并复测。"),
            }
        )
    warnings = []
    if unlocated_findings:
        warnings.append(
            f"Codex 有 {len(unlocated_findings)} 个问题未能通过可信代码定位校验；"
            "其完整语义已保留在“定位待核对的问题”中。"
        )
    return findings, unlocated_findings, warnings


def build_unlocated_finding(
    index: int,
    raw: dict[str, Any],
    *,
    trusted_file: str,
    location_issue: str,
) -> dict[str, Any]:
    return {
        "id": f"AI-{index:03d}",
        "severity": raw["severity"],
        "category": raw["category"],
        "trusted_file": trusted_file,
        "reported_line": (
            raw["line"].strip()
            if isinstance(raw["line"], str) and raw["line"].strip()
            else "未提供有效行号"
        ),
        "location_issue": location_issue,
        "code_role": text_or_default(raw.get("code_role"), "该位置参与本次变更行为。"),
        "title": text_or_default(raw.get("title"), "需要检查的代码问题"),
        "evidence": text_or_default(raw.get("evidence"), "Codex 未提供完整问题证据。"),
        "impact": text_or_default(raw.get("impact"), "可能影响本次变更涉及的行为。"),
        "fix_direction": text_or_default(
            raw.get("fix_direction"), "请重新定位该问题并结合证据修复和复测。"
        ),
    }


def build_suggested_tests(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, value in enumerate(
        require_array(analysis["suggested_tests"], "suggested_tests"), start=1
    ):
        location = f"suggested_tests[{index - 1}]"
        raw = require_object(value, location)
        require_exact_keys(raw, SUGGESTED_TEST_KEYS, location)
        priority = require_string(raw["priority"], f"{location}.priority")
        if priority not in SEVERITIES:
            raise ValueError(f"{location}.priority is invalid")
        target = require_string(raw["target"], f"{location}.target")
        description = require_string(raw["description"], f"{location}.description")
        result.append(
            {
                "id": f"TEST-{index:03d}",
                "priority": priority,
                "target": target.strip() or "未指定目标",
                "description": text_or_default(description, "建议补充相关定向测试。"),
            }
        )
    return result


def build_report(args: argparse.Namespace) -> None:
    raw_analysis = load_json(args.analysis)
    analysis = require_object(raw_analysis, "analysis root")
    require_exact_keys(analysis, ANALYSIS_KEYS, "analysis root")
    summary = require_string(analysis["summary"], "summary")
    merge_recommendation = require_string(
        analysis["merge_recommendation"], "merge_recommendation"
    )
    try:
        manifest = load_manifest(args.manifest)
        ledger = load_command_ledger(args.command_ledger)
        generated_archive = parse_generated_archive(args.generated_archive)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        tarfile.TarError,
    ) as exc:
        raise InvalidTrustedReportInput(str(exc)) from exc
    file_by_id = {
        f"FILE-{index:03d}": item for index, item in enumerate(manifest, start=1)
    }
    generated_files = [
        relative
        for relative in generated_archive
        if is_test_path(relative)
    ]
    commands, _ = build_commands(analysis, ledger)

    assessment = require_object(analysis["test_assessment"], "test_assessment")
    require_exact_keys(assessment, TEST_ASSESSMENT_KEYS, "test_assessment")
    evidence_level = require_string(
        assessment["evidence_level"], "test_assessment.evidence_level"
    )
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError("test_assessment.evidence_level is invalid")
    raw_execution_summary = require_array(
        assessment["summary"], "test_assessment.summary"
    )
    checked_execution_summary = [
        require_string(item, f"test_assessment.summary[{index}]")
        for index, item in enumerate(raw_execution_summary)
    ]
    codex_execution_summary = [
        "Codex 说明：" + text_or_default(
            item,
            "本轮验证说明未完整提供。",
        )
        for item in unique_in_order(checked_execution_summary)[:8]
    ]
    if not codex_execution_summary:
        codex_execution_summary = [
            "Codex 说明：本次不需要执行额外验证命令。"
            if evidence_level == "not_needed"
            else "Codex 说明：模型未提供验证证据说明。"
        ]

    findings, unlocated_findings, finding_warnings = build_findings(
        analysis, file_by_id, args.repository_root
    )
    suggested_tests = build_suggested_tests(analysis)
    changed_files, changed_file_warnings = build_changed_files(
        analysis, manifest, commands
    )
    unclassified_failure_count = sum(
        command["role"] == "unclassified" and command["exit_code"] != 0
        for command in commands
    )
    command_warnings = (
        [
            f"{unclassified_failure_count} 条非零退出命令没有可匹配的用途说明；"
            "其执行事实已保留，但未用于派生正式验证状态。"
        ]
        if unclassified_failure_count
        else []
    )
    normalization_warnings = finding_warnings + changed_file_warnings + command_warnings
    execution_summary = unique_in_order(codex_execution_summary)[:10]
    execution_status = derive_execution_status(
        evidence_level,
        commands,
    )
    residual_risks = [
        text_or_default(
            require_string(item, f"residual_risks[{index}]"),
            "仍有未覆盖的审查风险。",
        )
        for index, item in enumerate(
            require_array(analysis["residual_risks"], "residual_risks")
        )
    ]
    normalization_risks = [
        f"{REPORT_NORMALIZATION_RISK_PREFIX}{warning}"
        for warning in normalization_warnings
    ]
    residual_risks = unique_in_order(residual_risks + normalization_risks)
    for warning in normalization_warnings:
        print(f"Normalized Codex AI analysis: {warning}")
    behavior_coverage = build_behavior_coverage(analysis)
    verdict = (
        "FAIL"
        if any(
            finding["severity"] == "HIGH"
            for finding in findings + unlocated_findings
        )
        else "WARNING"
        if (
            findings
            or unlocated_findings
            or evidence_level in {"insufficient", "test_generation_error"}
            or execution_status in WARNING_EXECUTION_STATUSES
        )
        else "PASS"
    )
    normalized_merge_recommendation = text_or_default(
        merge_recommendation,
        "请结合本地确定性 CI 检查结果决定是否合入。",
    )
    report = {
        "verdict": verdict,
        "summary": text_or_default(
            summary, "本轮已完成代码差异的 Codex AI 自动审查。"
        ),
        "merge_recommendation": normalized_merge_recommendation,
        "change_request_assessment": normalize_assessment(analysis),
        "changed_files": changed_files,
        "behavior_coverage": behavior_coverage,
        "findings": findings,
        "unlocated_findings": unlocated_findings,
        "suggested_tests": suggested_tests,
        "residual_risks": residual_risks,
        "test_execution": {
            "evidence_level": evidence_level,
            "status": execution_status,
            "summary": execution_summary,
            "generated_test_files": generated_files,
            "commands": commands,
        },
        "completion_marker": "CODEX_AI_CI_COMPLETE",
    }
    write_json(args.output, report)


def build_fallback_changed_files(
    manifest: list[dict[str, str]], commands: list[dict[str, Any]]
) -> list[dict[str, str]]:
    validation_strategy = (
        f"自动审查未完成；已保留 {len(commands)} 条可信命令执行记录供人工核对。"
        if commands
        else "自动审查未完成；请结合代码差异和现有检查结果人工核对。"
    )
    return [
        {
            "path": item["path"],
            "change_type": item["change_type"],
            "summary": "自动审查未完成，未能可靠归纳该文件的具体改动。",
            "impact": "该文件的行为影响仍需结合代码差异人工核对。",
            "validation_strategy": validation_strategy,
        }
        for item in manifest
    ]


def build_fallback_report(args: argparse.Namespace) -> None:
    """Build a canonical warning report while preserving available trusted facts."""
    try:
        manifest = load_manifest(args.manifest)
        if args.command_ledger_state == "available":
            if not args.command_ledger.is_file():
                raise ValueError("available command ledger does not exist")
            ledger = load_command_ledger(args.command_ledger)
        else:
            ledger = []
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise InvalidTrustedReportInput(str(exc)) from exc

    generated_archive: list[str] = []
    archive_confirmed = False
    if args.generated_archive_state == "available":
        try:
            if not args.generated_archive.is_file():
                raise ValueError("available generated archive does not exist")
            generated_archive = parse_generated_archive(args.generated_archive)
            archive_confirmed = True
        except (OSError, ValueError, tarfile.TarError) as exc:
            print(f"Ignored untrusted generated archive in fallback report: {exc}")

    commands, _ = build_commands(None, ledger)
    generated_files = [
        relative for relative in generated_archive if is_test_path(relative)
    ]
    execution_status = (
        derive_execution_status("unavailable", commands)
        if args.command_ledger_state == "available"
        else "unavailable"
    )
    execution_summary = [
        "本次自动审查未完成，未形成完整的代码审查分析结果。"
    ]
    if commands:
        execution_summary.append(
            f"已从可信执行记录保留 {len(commands)} 条验证或诊断命令事实。"
        )
    if generated_files:
        execution_summary.append(
            f"已从任务级归档保留 {len(generated_files)} 个新增测试文件记录。"
        )
    elif not archive_confirmed:
        execution_summary.append("本次未能取得可信的任务级测试文件归档事实。")

    failure_reason = text_or_default(
        args.failure_reason,
        "自动审查执行未完成，具体原因未能可靠归纳。",
    )
    context_not_applicable = args.change_request_context_status == "not_applicable"
    assessment = {
        "status": "not_applicable" if context_not_applicable else "not_assessable",
        "contributor_goal": (
            "当前任务没有适用的变更请求功能声明。"
            if context_not_applicable
            else "自动审查未完成，未能可靠归纳贡献者的修改目标。"
        ),
        "expected_behavior": (
            "当前任务没有适用的变更请求预期行为声明。"
            if context_not_applicable
            else "自动审查未完成，未能可靠归纳贡献者声明的预期行为。"
        ),
        "implementation_summary": (
            "自动审查未完成，当前只能保留可信的文件和命令事实。"
        ),
        "evidence": [
            "本次仅保留了可由自动化流程确认的文件、命令和测试产物事实。"
        ],
    }
    behavior_coverage = {
        name: {
            "scope": f"本次未能完整判断变更涉及的{label}。",
            "strategy": "请结合代码差异、确定性检查结果和已保留的执行事实人工核对。",
            "result": "自动审查未完成，当前没有形成可信的行为覆盖结论。",
        }
        for name, label in BEHAVIOR_LABELS.items()
    }
    report = {
        "verdict": "WARNING",
        "summary": f"Codex AI 自动审查未完成：{failure_reason}",
        "merge_recommendation": (
            "请结合本地确定性 CI 检查结果和完整执行记录决定是否合入；"
            "本次未完成的自动审查结果不应单独作为判断依据。"
        ),
        "change_request_assessment": assessment,
        "changed_files": build_fallback_changed_files(manifest, commands),
        "behavior_coverage": behavior_coverage,
        "findings": [],
        "unlocated_findings": [],
        "suggested_tests": [],
        "residual_risks": unique_in_order(
            [
                "自动审查未完成，当前代码差异仍需结合可信检查结果人工核对。",
                *(
                    ["任务级测试文件归档不可确认，不能据此判断是否生成了新增测试。"]
                    if not archive_confirmed
                    else []
                ),
            ]
        ),
        "test_execution": {
            "evidence_level": "unavailable",
            "status": execution_status,
            "summary": execution_summary,
            "generated_test_files": generated_files,
            "commands": commands,
        },
        "completion_marker": "CODEX_AI_CI_COMPLETE",
    }
    write_json(args.output, report)


def parse_bool(value: str) -> bool:
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-manifest")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--analysis", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--command-ledger", type=Path, required=True)
    build.add_argument("--generated-archive", type=Path, required=True)
    build.add_argument("--repository-root", type=Path, required=True)
    build.add_argument("--analysis-mode", choices=("full", "analysis_only"), required=True)
    build.add_argument("--test-generation-expected", type=parse_bool, required=True)
    fallback = subparsers.add_parser("build-fallback")
    fallback.add_argument("--output", type=Path, required=True)
    fallback.add_argument("--manifest", type=Path, required=True)
    fallback.add_argument("--command-ledger", type=Path, required=True)
    fallback.add_argument(
        "--command-ledger-state",
        choices=("available", "unavailable"),
        required=True,
    )
    fallback.add_argument("--generated-archive", type=Path, required=True)
    fallback.add_argument(
        "--generated-archive-state",
        choices=("available", "unavailable"),
        required=True,
    )
    fallback.add_argument("--failure-reason", required=True)
    fallback.add_argument("--change-request-context-status", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "prepare-manifest":
            try:
                prepare_manifest(args.input, args.output)
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                raise InvalidTrustedReportInput(str(exc)) from exc
        elif args.command == "build":
            build_report(args)
        else:
            build_fallback_report(args)
    except InvalidTrustedReportInput as exc:
        print(f"Invalid Codex AI trusted input: {exc}")
        return 1
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, tarfile.TarError) as exc:
        print(f"Invalid Codex AI analysis: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
