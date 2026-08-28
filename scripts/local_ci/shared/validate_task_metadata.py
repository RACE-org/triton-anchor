#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_V1 = "triton-anchor-local-ci-task-metadata/v1"
SCHEMA_V2 = "triton-anchor-local-ci-task-metadata/v2"
TITLE_LIMIT = 500
DESCRIPTION_LIMIT = 8000
MAX_INPUT_BYTES = 64 * 1024
TASK_REF_RE = re.compile(r"ci/pr-([0-9]+)/(.+)")
SHA_RE = re.compile(r"[0-9a-f]{40}")
REQUIRED_V1_FIELDS = {
    "schema",
    "task_ref",
    "target_sha",
    "pr_number",
    "title",
    "description",
    "captured_at",
    "title_truncated",
    "description_truncated",
}
REQUIRED_V2_FIELDS = {
    *REQUIRED_V1_FIELDS,
    "event_kind",
    "base_task_ref",
    "head_task_ref",
    "tested_sha",
    "tested_ref",
    "tested_sha_kind",
    "base_branch",
    "base_sha",
    "head_branch",
    "head_sha",
    "head_repo",
    "target_branch",
    "worker_revision_sha",
}
OPTIONAL_V2_FIELDS = {"execution_mode"}


class MetadataError(ValueError):
    pass


def fail(message: str) -> None:
    raise MetadataError(message)


def validate_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        fail(f"{field} 必须是 40 位小写十六进制 SHA")
    return value


def validate_nonempty_string(value: Any, field: str, *, limit: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        fail(f"{field} 必须是非空字符串")
    return value


def normalize_text(value: Any, field: str) -> tuple[str, bool]:
    if not isinstance(value, str):
        fail(f"{field} 必须是字符串")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    removed_nul = "\x00" in normalized
    return normalized.replace("\x00", ""), removed_nul


def validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 64:
        fail("captured_at 必须是非空 UTC 时间字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail("captured_at 不是有效的 ISO 8601 时间")
    if parsed.tzinfo is None:
        fail("captured_at 必须包含时区")
    if parsed.utcoffset() != timedelta(0):
        fail("captured_at 必须使用 UTC")
    return value


def validate_common_fields(
    document: dict[str, Any],
    *,
    expected_task_ref: str,
    expected_target_sha: str,
) -> tuple[int, str, str, str, bool, bool, str, list[str]]:
    if document["task_ref"] != expected_task_ref:
        fail("task_ref 与当前 PR 任务不一致")
    if document["target_sha"] != expected_target_sha:
        fail("target_sha 与当前测试提交不一致")
    validate_sha(expected_target_sha, "期望的 target SHA")

    task_match = TASK_REF_RE.fullmatch(expected_task_ref)
    if not task_match:
        fail("元数据只能用于 ci/pr-<number>/<branch> 任务")
    expected_pr_number = int(task_match.group(1))
    task_branch = task_match.group(2)
    if isinstance(document["pr_number"], bool) or not isinstance(
        document["pr_number"], int
    ):
        fail("pr_number 必须是整数")
    if document["pr_number"] != expected_pr_number:
        fail("pr_number 与 task_ref 不一致")
    if not isinstance(document["title_truncated"], bool) or not isinstance(
        document["description_truncated"], bool
    ):
        fail("截断标记必须是布尔值")

    title, title_had_nul = normalize_text(document["title"], "title")
    description, description_had_nul = normalize_text(
        document["description"], "description"
    )
    title = title.strip()
    description = description.strip()
    if not title:
        fail("title 不能为空")

    warnings: list[str] = []
    if title_had_nul or description_had_nul:
        warnings.append("已移除标题或描述中的 NUL 字符")
    title_truncated = document["title_truncated"] or len(title) > TITLE_LIMIT
    description_truncated = (
        document["description_truncated"] or len(description) > DESCRIPTION_LIMIT
    )
    if len(title) > TITLE_LIMIT:
        warnings.append(f"title 超过 {TITLE_LIMIT} 字符，已截断")
    if len(description) > DESCRIPTION_LIMIT:
        warnings.append(f"description 超过 {DESCRIPTION_LIMIT} 字符，已截断")

    return (
        expected_pr_number,
        task_branch,
        title[:TITLE_LIMIT],
        description[:DESCRIPTION_LIMIT],
        title_truncated,
        description_truncated,
        validate_timestamp(document["captured_at"]),
        warnings,
    )


def validate_v1_document(
    document: dict[str, Any],
    *,
    expected_task_ref: str,
    expected_target_sha: str,
) -> tuple[dict[str, Any], list[str]]:
    missing = sorted(REQUIRED_V1_FIELDS - set(document))
    extra = sorted(set(document) - REQUIRED_V1_FIELDS)
    if missing:
        fail(f"元数据缺少字段：{', '.join(missing)}")
    if extra:
        fail(f"元数据包含未知字段：{', '.join(extra)}")

    (
        expected_pr_number,
        _,
        title,
        description,
        title_truncated,
        description_truncated,
        captured_at,
        warnings,
    ) = validate_common_fields(
        document,
        expected_task_ref=expected_task_ref,
        expected_target_sha=expected_target_sha,
    )

    canonical = {
        "schema": SCHEMA_V1,
        "task_ref": expected_task_ref,
        "target_sha": expected_target_sha,
        "pr_number": expected_pr_number,
        "title": title,
        "description": description,
        "captured_at": captured_at,
        "title_truncated": title_truncated,
        "description_truncated": description_truncated,
    }
    return canonical, warnings


def validate_v2_document(
    document: dict[str, Any],
    *,
    expected_task_ref: str,
    expected_target_sha: str,
    expected_base_sha: str = "",
    expected_head_sha: str = "",
) -> tuple[dict[str, Any], list[str]]:
    missing = sorted(REQUIRED_V2_FIELDS - set(document))
    extra = sorted(set(document) - REQUIRED_V2_FIELDS - OPTIONAL_V2_FIELDS)
    if missing:
        fail(f"元数据缺少字段：{', '.join(missing)}")
    if extra:
        fail(f"元数据包含未知字段：{', '.join(extra)}")

    (
        expected_pr_number,
        task_branch,
        title,
        description,
        title_truncated,
        description_truncated,
        captured_at,
        warnings,
    ) = validate_common_fields(
        document,
        expected_task_ref=expected_task_ref,
        expected_target_sha=expected_target_sha,
    )

    if document["event_kind"] != "pull_request":
        fail("event_kind 必须是 pull_request")
    if document["tested_sha_kind"] != "pr_merge":
        fail("tested_sha_kind 必须是 pr_merge")
    tested_sha = validate_sha(document["tested_sha"], "tested_sha")
    if tested_sha != expected_target_sha:
        fail("tested_sha 与当前测试提交不一致")
    if document["target_sha"] != tested_sha:
        fail("target_sha 必须等于 tested_sha")

    expected_tested_ref = f"refs/pull/{expected_pr_number}/merge"
    if document["tested_ref"] != expected_tested_ref:
        fail("tested_ref 必须指向当前 PR 的 GitHub test merge ref")
    if document["base_task_ref"] != f"ci/base/pr-{expected_pr_number}/{task_branch}":
        fail("base_task_ref 与当前 PR 任务不一致")
    if document["head_task_ref"] != f"ci/head/pr-{expected_pr_number}/{task_branch}":
        fail("head_task_ref 与当前 PR 任务不一致")
    if document["head_branch"] != task_branch:
        fail("head_branch 与当前 PR 任务不一致")

    base_sha = validate_sha(document["base_sha"], "base_sha")
    head_sha = validate_sha(document["head_sha"], "head_sha")
    if expected_base_sha and base_sha != expected_base_sha:
        fail("base_sha 与当前 PR base ref 不一致")
    if expected_head_sha and head_sha != expected_head_sha:
        fail("head_sha 与当前 PR head ref 不一致")

    base_branch = validate_nonempty_string(document["base_branch"], "base_branch")
    head_repo = validate_nonempty_string(document["head_repo"], "head_repo")
    target_branch = validate_nonempty_string(document["target_branch"], "target_branch")
    if target_branch != base_branch:
        fail("target_branch 必须等于 base_branch")
    worker_revision_sha = validate_sha(
        document["worker_revision_sha"], "worker_revision_sha"
    )
    execution_mode = document.get("execution_mode", "full")
    if execution_mode not in {"full", "codex_only"}:
        fail("execution_mode 必须是 full 或 codex_only")

    canonical = {
        "schema": SCHEMA_V2,
        "event_kind": "pull_request",
        "execution_mode": execution_mode,
        "task_ref": expected_task_ref,
        "base_task_ref": f"ci/base/pr-{expected_pr_number}/{task_branch}",
        "head_task_ref": f"ci/head/pr-{expected_pr_number}/{task_branch}",
        "target_sha": expected_target_sha,
        "tested_sha": tested_sha,
        "tested_ref": expected_tested_ref,
        "tested_sha_kind": "pr_merge",
        "base_branch": base_branch,
        "base_sha": base_sha,
        "head_branch": task_branch,
        "head_sha": head_sha,
        "head_repo": head_repo,
        "target_branch": target_branch,
        "worker_revision_sha": worker_revision_sha,
        "pr_number": expected_pr_number,
        "title": title,
        "description": description,
        "captured_at": captured_at,
        "title_truncated": title_truncated,
        "description_truncated": description_truncated,
    }
    return canonical, warnings


def validate_document(
    document: Any,
    *,
    expected_task_ref: str,
    expected_target_sha: str,
    expected_base_sha: str = "",
    expected_head_sha: str = "",
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(document, dict):
        fail("元数据顶层必须是 JSON 对象")
    schema = document.get("schema")
    if schema == SCHEMA_V1:
        return validate_v1_document(
            document,
            expected_task_ref=expected_task_ref,
            expected_target_sha=expected_target_sha,
        )
    if schema == SCHEMA_V2:
        return validate_v2_document(
            document,
            expected_task_ref=expected_task_ref,
            expected_target_sha=expected_target_sha,
            expected_base_sha=expected_base_sha,
            expected_head_sha=expected_head_sha,
        )
    fail(f"schema 必须是 {SCHEMA_V1} 或 {SCHEMA_V2}")


def read_document(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            fail(f"元数据文件超过 {MAX_INPUT_BYTES} 字节上限")
        return json.loads(path.read_text(encoding="utf-8"))
    except MetadataError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        fail(f"无法读取有效的 UTF-8 JSON：{path}")


def write_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="校验并规范化 Local CI PR 元数据")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-ref", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", default="")
    args = parser.parse_args()

    try:
        canonical, warnings = validate_document(
            read_document(Path(args.input)),
            expected_task_ref=args.task_ref,
            expected_target_sha=args.target_sha,
            expected_base_sha=args.base_sha,
            expected_head_sha=args.head_sha,
        )
        write_document(Path(args.output), canonical)
    except MetadataError as exc:
        print(f"PR 元数据校验失败：{exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"PR 元数据警告：{warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
