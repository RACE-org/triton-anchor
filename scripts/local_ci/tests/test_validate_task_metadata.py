from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "local_ci" / "shared"))

import validate_task_metadata as metadata


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
TESTED_SHA = "c" * 40
TASK_REF = "ci/pr-42/feature"


def v2_document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": metadata.SCHEMA_V2,
        "event_kind": "pull_request",
        "task_ref": TASK_REF,
        "base_task_ref": "ci/base/pr-42/feature",
        "head_task_ref": "ci/head/pr-42/feature",
        "target_sha": TESTED_SHA,
        "tested_sha": TESTED_SHA,
        "tested_ref": "refs/pull/42/merge",
        "tested_sha_kind": "pr_merge",
        "base_branch": "main",
        "base_sha": BASE_SHA,
        "head_branch": "feature",
        "head_sha": HEAD_SHA,
        "head_repo": "owner/repo",
        "target_branch": "main",
        "worker_revision_sha": "d" * 40,
        "pr_number": 42,
        "title": "Add merge-aware Local CI",
        "description": "Test the GitHub PR merge result.",
        "captured_at": "2026-08-03T01:02:03Z",
        "title_truncated": False,
        "description_truncated": False,
    }
    document.update(overrides)
    return document


def test_v2_metadata_records_pr_merge_identity() -> None:
    canonical, warnings = metadata.validate_document(
        v2_document(),
        expected_task_ref=TASK_REF,
        expected_target_sha=TESTED_SHA,
        expected_base_sha=BASE_SHA,
        expected_head_sha=HEAD_SHA,
    )

    assert warnings == []
    assert canonical["schema"] == metadata.SCHEMA_V2
    assert canonical["target_sha"] == TESTED_SHA
    assert canonical["tested_sha"] == TESTED_SHA
    assert canonical["tested_sha_kind"] == "pr_merge"
    assert canonical["tested_ref"] == "refs/pull/42/merge"
    assert canonical["base_sha"] == BASE_SHA
    assert canonical["head_sha"] == HEAD_SHA
    assert canonical["target_branch"] == "main"
    assert canonical["worker_revision_sha"] == "d" * 40
    assert canonical["execution_mode"] == "full"


def test_v2_metadata_validates_execution_mode() -> None:
    canonical, warnings = metadata.validate_document(
        v2_document(execution_mode="codex_only"),
        expected_task_ref=TASK_REF,
        expected_target_sha=TESTED_SHA,
        expected_base_sha=BASE_SHA,
        expected_head_sha=HEAD_SHA,
    )

    assert warnings == []
    assert canonical["execution_mode"] == "codex_only"
    with pytest.raises(metadata.MetadataError, match="execution_mode"):
        metadata.validate_document(
            v2_document(execution_mode="skip_all"),
            expected_task_ref=TASK_REF,
            expected_target_sha=TESTED_SHA,
            expected_base_sha=BASE_SHA,
            expected_head_sha=HEAD_SHA,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_sha", HEAD_SHA, "target_sha 与当前测试提交不一致"),
        ("tested_sha", HEAD_SHA, "tested_sha 与当前测试提交不一致"),
        ("tested_ref", "refs/pull/42/head", "tested_ref 必须指向当前 PR"),
        ("tested_sha_kind", "head", "tested_sha_kind 必须是 pr_merge"),
        ("base_sha", HEAD_SHA, "base_sha 与当前 PR base ref 不一致"),
        ("head_sha", BASE_SHA, "head_sha 与当前 PR head ref 不一致"),
        ("target_branch", "release", "target_branch 必须等于 base_branch"),
        ("worker_revision_sha", "invalid", "worker_revision_sha 必须是 40 位"),
    ],
)
def test_v2_metadata_rejects_ambiguous_or_stale_identity(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(metadata.MetadataError, match=message):
        metadata.validate_document(
            v2_document(**{field: value}),
            expected_task_ref=TASK_REF,
            expected_target_sha=TESTED_SHA,
            expected_base_sha=BASE_SHA,
            expected_head_sha=HEAD_SHA,
        )


def test_v1_metadata_remains_supported_for_legacy_context() -> None:
    canonical, warnings = metadata.validate_document(
        {
            "schema": metadata.SCHEMA_V1,
            "task_ref": TASK_REF,
            "target_sha": HEAD_SHA,
            "pr_number": 42,
            "title": "Legacy metadata",
            "description": "Old head-SHA metadata still normalizes when expected.",
            "captured_at": "2026-08-03T01:02:03Z",
            "title_truncated": False,
            "description_truncated": False,
        },
        expected_task_ref=TASK_REF,
        expected_target_sha=HEAD_SHA,
    )

    assert warnings == []
    assert canonical["schema"] == metadata.SCHEMA_V1
    assert canonical["target_sha"] == HEAD_SHA
