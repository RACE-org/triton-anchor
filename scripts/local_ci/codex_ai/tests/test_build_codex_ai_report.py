import copy
import importlib.util
import io
import json
import re
import tarfile
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest


CODEX_AI_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILDER = load_module("build_codex_ai_report", CODEX_AI_DIR / "build_codex_ai_report.py")
RENDERER = load_module("render_codex_ai_report", CODEX_AI_DIR / "render_codex_ai_report.py")


def analysis(command: str = "python3 -m pytest generated_tests/test_generated.py"):
    return {
        "summary": "未发现具体缺陷，定向验证结果支持当前修改。",
        "merge_recommendation": "当前未发现需要阻塞合入的问题。",
        "change_request_assessment": {
            "status": "not_applicable",
            "contributor_goal": "当前任务不是 PR，因此没有贡献者功能声明。",
            "expected_behavior": "当前任务不是 PR，因此预期行为声明不适用。",
            "implementation_summary": "本次仅依据代码差异完成自动审查。",
            "evidence": ["任务上下文明确标记为普通推送任务。"],
        },
        "changed_files": [
            {
                "file_id": "FILE-001",
                "summary": "调整了示例代码的执行逻辑。",
                "impact": "可能影响示例代码的返回结果。",
                "validation_strategy": "执行定向测试验证修改后的行为。",
            }
        ],
        "behavior_coverage": {
            name: {
                "scope": f"检查{name}对应的行为路径。",
                "strategy": "结合代码差异和定向验证检查。",
                "result": "未发现新的行为缺陷。",
            }
            for name in ("normal", "boundary", "error", "compatibility", "integration")
        },
        "findings": [],
        "suggested_tests": [],
        "residual_risks": ["本次仅覆盖与代码差异直接相关的路径。"],
        "test_assessment": {
            "evidence_level": "sufficient",
            "summary": ["生成并执行了一个定向测试。"],
            "commands": [
                {
                    "command": command,
                    "role": "validation",
                    "purpose": "生成代码路径定向测试",
                    "evidence": "定向测试执行完成。",
                    "failure_classification": "none",
                }
            ],
        },
    }


def finding(*, file_id: str = "FILE-001", line: str = "1") -> dict:
    return {
        "severity": "MEDIUM",
        "category": "correctness",
        "file_id": file_id,
        "line": line,
        "code_role": "该行负责计算返回结果。",
        "title": "返回结果错误",
        "evidence": "该表达式会产生错误结果。",
        "impact": "调用方会收到错误结果。",
        "fix_direction": "修正该表达式并补充测试。",
    }


def comment_args() -> Namespace:
    return Namespace(
        branch="CI_dev",
        base_sha="a" * 40,
        requested_base_sha="",
        diff_mode="two-point",
        target_sha="b" * 40,
        head_sha="",
        local_ci_status="0",
        tested_sha_kind="commit",
        changed_file_count=1,
        constraint_status="pass",
        constraint_reason="未发现测试数量或耗时超出轻量约束。",
    )


def set_nested(document: dict, path: tuple[object, ...], value: object) -> None:
    current: Any = document
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def write_archive(path: Path, entries: list[tuple[str, bytes, str]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == "file":
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            elif kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "outside"
                archive.addfile(member)
            else:
                raise AssertionError(kind)


def build(
    tmp_path: Path,
    document: dict,
    ledger: list[dict],
    archive_entries=None,
    *,
    test_generation_expected=True,
    source_text: str | None = "value = 1\n",
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir(exist_ok=True)
    if source_text is not None:
        (repository_root / "example.py").write_text(source_text, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    analysis_path = tmp_path / "analysis.json"
    ledger_path = tmp_path / "ledger.json"
    archive_path = tmp_path / "generated.tar.gz"
    output_path = tmp_path / "report.json"
    manifest = [{"path": "example.py", "change_type": "modified"}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    analysis_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    write_archive(
        archive_path,
        archive_entries
        if archive_entries is not None
        else [("generated_tests/test_generated.py", b"def test_x(): pass\n", "file")],
    )
    BUILDER.build_report(
        Namespace(
            analysis=analysis_path,
            output=output_path,
            manifest=manifest_path,
            command_ledger=ledger_path,
            generated_archive=archive_path,
            repository_root=repository_root,
            analysis_mode="full",
            test_generation_expected=test_generation_expected,
        )
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    RENDERER.validate_report(report, manifest, repository_root)
    return report


def build_fallback(
    tmp_path: Path,
    ledger: list[dict] | None,
    archive_entries: list[tuple[str, bytes, str]] | None,
    *,
    command_ledger_state: str = "available",
    generated_archive_state: str = "available",
):
    repository_root = tmp_path / "fallback-repo"
    repository_root.mkdir(exist_ok=True)
    (repository_root / "example.py").write_text("value = 1\n", encoding="utf-8")
    manifest = [{"path": "example.py", "change_type": "modified"}]
    manifest_path = tmp_path / "fallback-manifest.json"
    ledger_path = tmp_path / "fallback-ledger.json"
    archive_path = tmp_path / "fallback-generated.tar.gz"
    output_path = tmp_path / "fallback-report.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if ledger is not None:
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    if archive_entries is not None:
        write_archive(archive_path, archive_entries)
    BUILDER.build_fallback_report(
        Namespace(
            output=output_path,
            manifest=manifest_path,
            command_ledger=ledger_path,
            command_ledger_state=command_ledger_state,
            generated_archive=archive_path,
            generated_archive_state=generated_archive_state,
            failure_reason="结构化审查分析未能完成。",
            change_request_context_status="valid",
        )
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    RENDERER.validate_report(report, manifest, repository_root)
    return report


def test_builder_assigns_trusted_fields_and_pass_status(tmp_path):
    command = "python3 -m pytest generated_tests/test_generated.py"
    report = build(
        tmp_path,
        analysis(command),
        [{"command": command, "exit_code": 0, "duration_seconds": 0.2}],
    )
    assert report["verdict"] == "PASS"
    assert report["changed_files"][0]["path"] == "example.py"
    assert report["changed_files"][0]["summary"] == "调整了示例代码的执行逻辑。"
    assert report["behavior_coverage"]["normal"]["scope"] == "检查normal对应的行为路径。"
    assert report["test_execution"]["evidence_level"] == "sufficient"
    assert report["test_execution"]["status"] == "passed"
    assert report["test_execution"]["commands"][0] == {
        "id": "RUN-001",
        "role": "validation",
        "purpose": "生成代码路径定向测试",
        "command": command,
        "exit_code": 0,
        "duration_seconds": 0.2,
        "status": "passed",
        "evidence": "定向测试执行完成。",
    }
    assert report["completion_marker"] == "CODEX_AI_CI_COMPLETE"


def test_public_validation_uses_fact_groups_without_internal_status_labels(tmp_path):
    command = "python3 -m pytest generated_tests/test_generated.py"
    report = build(
        tmp_path,
        analysis(command),
        [{"command": command, "exit_code": 0, "duration_seconds": 0.2}],
    )

    comment = RENDERER.render_comment(report, comment_args())
    validation = comment.split("### 验证情况", 1)[1].split("### 剩余风险", 1)[0]

    assert "- 验证内容与结果：" in validation
    assert "  - 生成并执行了一个定向测试。" in validation
    assert "生成了 1 个任务级测试文件：generated_tests/test_generated.py。" in validation
    assert "'generated_tests/test_generated.py'" not in validation
    assert "生成代码路径定向测试执行成功" not in validation
    assert "- 限制与未覆盖：" in validation
    assert "  - 本次未报告额外的验证限制或未覆盖项。" in validation
    assert "- 验证依据：" not in validation
    assert "- 执行内容：" not in validation
    assert "- 执行结果：" not in validation
    assert "Codex 对验证证据的判断" not in comment
    assert "Runner 事实校验" not in comment
    assert "Codex 说明：" not in validation
    assert "Runner 校验：" not in validation


def test_diagnostic_failure_does_not_override_formal_validation(tmp_path):
    document = analysis()
    document["test_assessment"]["summary"] = [
        "报告契约定向测试共执行一百四十三个用例并全部通过。",
        "旧字段残留检查已通过等价方式完成，未发现旧字段残留。",
    ]
    document["test_assessment"]["commands"] = [
        {
            "command": "python3 -m pytest report_tests.py",
            "role": "validation",
            "purpose": "报告契约定向测试",
            "evidence": "一百四十三个用例全部通过。",
            "failure_classification": "none",
        },
        {
            "command": "rg old_field scripts/local_ci",
            "role": "diagnostic",
            "purpose": "旧字段残留搜索",
            "evidence": "搜索工具不可用；随后使用现有工具完成等价检查。",
            "failure_classification": "infrastructure",
        },
        {
            "command": "grep -R old_field scripts/local_ci",
            "role": "diagnostic",
            "purpose": "旧字段残留搜索",
            "evidence": "等价搜索已经完成，未发现旧字段残留。",
            "failure_classification": "none",
        },
    ]
    report = build(
        tmp_path,
        document,
        [
            {
                "command": "python3 -m pytest report_tests.py",
                "exit_code": 0,
                "duration_seconds": 0.1,
            },
            {
                "command": "rg old_field scripts/local_ci",
                "exit_code": 127,
                "duration_seconds": 0.1,
            },
            {
                "command": "grep -R old_field scripts/local_ci",
                "exit_code": 0,
                "duration_seconds": 0.1,
            },
        ],
    )

    assert report["test_execution"]["status"] == "passed"
    assert report["verdict"] == "PASS"
    assert [command["role"] for command in report["test_execution"]["commands"]] == [
        "validation",
        "diagnostic",
        "diagnostic",
    ]
    comment = RENDERER.render_comment(report, comment_args())
    validation = comment.split("### 验证情况", 1)[1].split("### 剩余风险", 1)[0]
    assert "一百四十三个用例并全部通过" in validation
    assert "旧字段残留检查已通过等价方式完成" in validation
    assert "搜索工具不可用" not in validation
    assert "执行记录" not in validation
    assert "条成功" not in validation


def test_validation_target_can_be_closed_by_an_alternative_method(tmp_path):
    document = analysis()
    document["test_assessment"]["summary"] = [
        "报告语法检查已通过等价方式完成。"
    ]
    document["test_assessment"]["commands"] = [
        {
            "command": "bash -n report_tests.py",
            "role": "validation",
            "purpose": "报告语法检查",
            "evidence": "所用解释器不适用于该文件，未形成语法结论。",
            "failure_classification": "product",
        },
        {
            "command": "python3 -m py_compile report_tests.py",
            "role": "validation",
            "purpose": "报告语法检查",
            "evidence": "使用正确解释器完成语法检查。",
            "failure_classification": "none",
        },
    ]
    report = build(
        tmp_path,
        document,
        [
            {
                "command": "bash -n report_tests.py",
                "exit_code": 2,
                "duration_seconds": 0.1,
            },
            {
                "command": "python3 -m py_compile report_tests.py",
                "exit_code": 0,
                "duration_seconds": 0.1,
            },
        ],
    )

    assert report["test_execution"]["status"] == "passed"
    assert report["verdict"] == "PASS"
    comment = RENDERER.render_comment(report, comment_args())
    validation = comment.split("### 验证情况", 1)[1].split("### 剩余风险", 1)[0]
    assert "报告语法检查已通过等价方式完成" in validation
    assert "所用解释器不适用" not in validation


@pytest.mark.parametrize(
    ("role", "expected_status", "expected_verdict"),
    [
        ("validation", "insufficient_evidence", "WARNING"),
        ("diagnostic", "passed", "PASS"),
    ],
)
def test_later_failure_is_not_closed_by_an_earlier_success(
    tmp_path, role, expected_status, expected_verdict
):
    document = analysis()
    successful = {
        "command": "python3 -m py_compile report_tests.py",
        "role": role,
        "purpose": "报告语法检查",
        "evidence": "等价语法检查曾执行成功。",
        "failure_classification": "none",
    }
    failed = {
        "command": "bash -n report_tests.py",
        "role": role,
        "purpose": "报告语法检查",
        "evidence": "后续检查使用了不适用的解释器，因此该目标仍未关闭。",
        "failure_classification": "product",
    }
    ledger = [
        {
            "command": successful["command"],
            "exit_code": 0,
            "duration_seconds": 0.1,
        },
        {"command": failed["command"], "exit_code": 2, "duration_seconds": 0.1},
    ]
    if role == "validation":
        document["test_assessment"]["commands"] = [successful, failed]
    else:
        validation_command = document["test_assessment"]["commands"][0]["command"]
        document["test_assessment"]["commands"].extend([successful, failed])
        ledger.insert(
            0,
            {
                "command": validation_command,
                "exit_code": 0,
                "duration_seconds": 0.1,
            },
        )
    document["test_assessment"]["summary"] = ["完成了与改动相关的静态审查。"]

    report = build(tmp_path, document, ledger)
    assert report["test_execution"]["status"] == expected_status
    assert report["verdict"] == expected_verdict
    validation = RENDERER.render_comment(report, comment_args()).split(
        "### 验证情况", 1
    )[1].split("### 剩余风险", 1)[0]
    assert "报告语法检查尚未完成" in validation
    assert "后续检查使用了不适用的解释器" in validation
    assert "等价语法检查曾执行成功" not in validation


@pytest.mark.parametrize(
    ("evidence_level", "expected_verdict"),
    [("sufficient", "PASS"), ("insufficient", "WARNING")],
)
def test_unresolved_diagnostic_verdict_follows_overall_evidence_level(
    tmp_path, evidence_level, expected_verdict
):
    document = analysis()
    document["residual_risks"] = []
    document["test_assessment"]["evidence_level"] = evidence_level
    document["test_assessment"]["summary"] = [
        "报告契约定向测试执行通过。",
    ]
    document["test_assessment"]["commands"] = [
        {
            "command": "python3 -m pytest report_tests.py",
            "role": "validation",
            "purpose": "报告契约定向测试",
            "evidence": "报告契约定向测试执行通过。",
            "failure_classification": "none",
        },
        {
            "command": "rg old_field scripts/local_ci",
            "role": "diagnostic",
            "purpose": "旧字段残留搜索",
            "evidence": "临时环境未提供搜索工具，旧字段残留检查尚未完成。",
            "failure_classification": "infrastructure",
        },
        {
            "command": "grep -R old_field scripts/local_ci",
            "role": "diagnostic",
            "purpose": "旧字段残留搜索",
            "evidence": "替代搜索也受当前环境限制，该目标仍未完成。",
            "failure_classification": "infrastructure",
        },
    ]
    report = build(
        tmp_path,
        document,
        [
            {
                "command": "python3 -m pytest report_tests.py",
                "exit_code": 0,
                "duration_seconds": 0.1,
            },
            {
                "command": "rg old_field scripts/local_ci",
                "exit_code": 127,
                "duration_seconds": 0.1,
            },
            {
                "command": "grep -R old_field scripts/local_ci",
                "exit_code": 2,
                "duration_seconds": 0.1,
            },
        ],
    )

    assert report["test_execution"]["status"] == "passed"
    assert report["verdict"] == expected_verdict
    assert report["merge_recommendation"] == document["merge_recommendation"]
    comment = RENDERER.render_comment(report, comment_args())
    validation = comment.split("### 验证情况", 1)[1].split(
        "### 剩余风险", 1
    )[0]
    assert "报告契约定向测试执行通过" in validation
    assert "旧字段残留搜索尚未完成" in validation
    assert "替代搜索也受当前环境限制" in validation
    assert validation.count("旧字段残留搜索尚未完成") == 1
    assert "执行记录" not in validation


def test_public_validation_sanitizes_internal_sources_and_enums(tmp_path):
    document = analysis()
    document["test_assessment"] = {
        "evidence_level": "insufficient",
        "summary": [
            "Runner 事实校验：test_execution.status=not_run；"
            "evidence_level=insufficient。",
            "未执行任何新增验证命令。",
        ],
        "commands": [],
    }
    document["residual_risks"] = []
    report = build(tmp_path, document, [], archive_entries=[])

    comment = RENDERER.render_comment(report, comment_args())
    assert "Runner" not in comment
    assert "test_execution.status" not in comment
    assert "evidence_level" not in comment
    assert "not_run" not in comment
    assert "insufficient" not in comment
    assert "本次未新增验证命令" in comment
    assert comment.count("本次未新增验证命令") == 1
    assert "现有验证覆盖有限" in comment


def test_public_validation_sanitizing_does_not_rewrite_finding_evidence(tmp_path):
    document = analysis()
    item = finding()
    item["evidence"] = (
        '代码证据为 `status == "passed"`，并调用 `Runner.run()`；'
        "常量 `AI-001` 也必须保持不变。"
    )
    document["findings"] = [item]
    report = build(
        tmp_path,
        document,
        [
            {
                "command": "python3 -m pytest generated_tests/test_generated.py",
                "exit_code": 0,
                "duration_seconds": 0.2,
            }
        ],
    )
    report["changed_files"][0]["path"] = "docs/AI-001-passed.py"

    comment = RENDERER.render_comment(report, comment_args())

    assert 'status == "passed"' in comment
    assert "Runner.run()" in comment
    assert "AI-001" in comment
    assert "`docs/AI-001-passed.py`" in comment


def test_no_findings_wording_requires_complete_review_and_evidence(tmp_path):
    complete = build(
        tmp_path,
        analysis(),
        [
            {
                "command": "python3 -m pytest generated_tests/test_generated.py",
                "exit_code": 0,
                "duration_seconds": 0.2,
            }
        ],
    )
    assert "本次审查未发现需要处理的具体代码缺陷" in RENDERER.render_comment(
        complete, comment_args()
    )

    limited_document = analysis()
    limited_document["test_assessment"] = {
        "evidence_level": "insufficient",
        "summary": ["当前验证覆盖仍然有限。"],
        "commands": [],
    }
    limited_document["residual_risks"] = []
    limited = build(tmp_path, limited_document, [], archive_entries=[])
    limited_comment = RENDERER.render_comment(limited, comment_args())
    assert "本次未形成可确认的具体代码问题" in limited_comment
    assert "本次审查未发现需要处理的具体代码缺陷" not in limited_comment
    assert "除上述验证限制外，本次未报告其他剩余风险" in limited_comment


def test_public_summary_only_shows_verdict_for_complete_review(tmp_path):
    complete = build(
        tmp_path,
        analysis(),
        [
            {
                "command": "python3 -m pytest generated_tests/test_generated.py",
                "exit_code": 0,
                "duration_seconds": 0.2,
            }
        ],
    )
    complete_comment = RENDERER.render_comment(complete, comment_args())
    assert "Codex 执行状态" not in complete_comment
    assert "Codex AI 审查结论：**通过**" in complete_comment
    assert "Codex 建议性结论" not in complete_comment
    assert "本地确定性 CI 检查：" in complete_comment

    incomplete = build_fallback(tmp_path, [], [])
    incomplete_comment = RENDERER.render_comment(incomplete, comment_args())
    assert "Codex 执行状态" not in incomplete_comment
    assert "Codex 建议性结论" not in incomplete_comment
    assert "Codex AI 审查结论" not in incomplete_comment


def test_public_summary_routes_known_limitations_to_limit_group(tmp_path):
    document = analysis()
    document["residual_risks"] = []
    document["test_assessment"]["summary"] = [
        "复用了与当前提交匹配的确定性检查日志。",
        "尚未在真实 GitHub Environment 审批流程中完成端到端验证。",
    ]
    report = build(
        tmp_path,
        document,
        [
            {
                "command": "python3 -m pytest generated_tests/test_generated.py",
                "exit_code": 0,
                "duration_seconds": 0.2,
            }
        ],
    )
    comment = RENDERER.render_comment(report, comment_args())
    validation = comment.split("### 验证情况", 1)[1].split("### 剩余风险", 1)[0]
    results, limits = validation.split("- 限制与未覆盖：", 1)
    assert "复用了与当前提交匹配的确定性检查日志" in results
    assert "尚未在真实 GitHub Environment" not in results
    assert "尚未在真实 GitHub Environment" in limits
    assert "除上述验证限制外，本次未报告其他剩余风险" in comment
    assert "本次未报告额外的验证限制或未覆盖项" not in limits


def test_frontend_only_scope_lists_the_complete_backend_limit(tmp_path):
    report = build(
        tmp_path,
        analysis(),
        [
            {
                "command": "python3 -m pytest generated_tests/test_generated.py",
                "exit_code": 0,
                "duration_seconds": 0.2,
            }
        ],
    )
    args = comment_args()
    args.backend_validation_scope = "frontend_only"
    comment = RENDERER.render_comment(report, args)
    assert (
        "当前没有部署可供测试的厂商后端，未执行后端构建、JIT、FlagGems 和性能验证。"
        in comment
    )


@pytest.mark.parametrize(
    ("execution_mode", "backend_scope", "status", "expected"),
    [
        ("full", "full", "0", "已通过；Codex AI 自动审查只提供补充意见"),
        ("full", "full", "1", "未通过；Codex AI 自动审查用于辅助定位原因"),
        ("codex_only", "full", "", "按策略未执行确定性 CI；该状态不表示确定性测试通过"),
        ("unavailable", "full", "", "执行状态不可确认；当前不能据此判断确定性门禁结果"),
        ("full", "frontend_only", "0", "前端验证范围已通过；本次未执行厂商后端"),
        ("full", "frontend_only", "1", "前端验证范围未通过；本次未执行厂商后端"),
        ("full", "unavailable", "0", "已执行范围通过，但后端验证范围不可确认"),
    ],
)
def test_deterministic_ci_public_modes_are_explicit_and_backward_compatible(
    execution_mode, backend_scope, status, expected
):
    args = comment_args()
    args.local_ci_execution_mode = execution_mode
    args.backend_validation_scope = backend_scope
    args.local_ci_status = status
    assert expected in RENDERER.deterministic_ci_comment_line(args)


@pytest.mark.parametrize(
    ("exits", "classification", "purpose", "evidence", "expected", "public_result"),
    [
        (
            [1],
            "product",
            "PR Comment 混合结果契约测试",
            "pytest 输出显示评论仍包含已禁止的“失败待归因”描述。"
            "因此本次未能确认新的失败说明格式符合预期，"
            "但不影响已经独立通过的 Python 语法和差异格式检查。",
            "insufficient_evidence",
            "pytest 输出显示评论仍包含已禁止的“失败待归因”描述",
        ),
        (
            [1, 1],
            "product",
            "生成代码路径定向测试",
            "生成代码路径定向测试重复执行仍失败，错误输出一致。",
            "stable_failure",
            "重复执行仍失败，错误输出一致",
        ),
        (
            [1, 0],
            "flaky",
            "生成代码路径定向测试",
            "生成代码路径定向测试重复执行结果不一致，稳定性待复核。",
            "flaky_failure",
            "重复执行结果不一致，稳定性待复核",
        ),
        (
            [2],
            "infrastructure",
            "生成代码路径定向测试",
            "运行环境缺少必要设备，生成代码路径定向测试受到限制。",
            "infrastructure_failure",
            "运行环境缺少必要设备",
        ),
    ],
)
def test_failure_status_is_derived_from_repeated_ledger(
    tmp_path, exits, classification, purpose, evidence, expected, public_result
):
    command = "python3 -m pytest generated_tests/test_generated.py"
    document = analysis(command)
    semantic = document["test_assessment"]["commands"][0]
    semantic["purpose"] = purpose
    semantic["evidence"] = evidence
    semantic["failure_classification"] = classification
    document["test_assessment"]["summary"] = [evidence]
    document["test_assessment"]["commands"] = [copy.deepcopy(semantic)]
    if expected == "flaky_failure":
        successful_retry = copy.deepcopy(semantic)
        successful_retry["evidence"] = "后续一次执行已经通过。"
        successful_retry["failure_classification"] = "none"
        document["test_assessment"]["commands"].append(successful_retry)
        document["test_assessment"]["summary"] = [
            "完成了与该测试相关的静态审查。"
        ]
    ledger = [
        {"command": command, "exit_code": code, "duration_seconds": 0.1}
        for code in exits
    ]
    report = build(tmp_path, document, ledger)
    assert report["test_execution"]["status"] == expected
    assert report["verdict"] == "WARNING"
    comment = RENDERER.render_comment(report, comment_args())
    assert "Codex AI 审查结论：**需关注（非阻塞）**" in comment
    validation = comment.split("### 验证情况", 1)[1].split("### 剩余风险", 1)[0]
    assert public_result in validation
    if expected == "flaky_failure":
        assert "后续一次执行已经通过" not in validation
    if expected == "infrastructure_failure":
        assert "所执行的验证均受运行环境限制" in validation
        assert "部分验证受运行环境限制" not in validation


def test_not_needed_with_no_commands_derives_not_run(tmp_path):
    document = analysis()
    document["test_assessment"] = {
        "evidence_level": "not_needed",
        "summary": ["本次只有文档变化，因此不需要执行测试。"],
        "commands": [],
    }
    report = build(tmp_path, document, [], archive_entries=[])
    assert report["test_execution"]["status"] == "not_run"
    assert report["verdict"] == "PASS"


def test_sufficient_reused_evidence_with_no_commands_keeps_runner_not_run(tmp_path):
    document = analysis()
    document["test_assessment"] = {
        "evidence_level": "sufficient",
        "summary": ["复用的确定性 Local CI 证据足以支撑当前审查结论。"],
        "commands": [],
    }
    report = build(tmp_path, document, [], archive_entries=[])
    assert report["test_execution"]["evidence_level"] == "sufficient"
    assert report["test_execution"]["status"] == "not_run"
    assert report["verdict"] == "PASS"


def test_insufficient_evidence_with_no_commands_warns_without_faking_runner_status(
    tmp_path,
):
    document = analysis()
    document["test_assessment"] = {
        "evidence_level": "insufficient",
        "summary": ["当前没有足够的动态验证证据。"],
        "commands": [],
    }
    report = build(tmp_path, document, [], archive_entries=[])
    assert report["test_execution"]["evidence_level"] == "insufficient"
    assert report["test_execution"]["status"] == "not_run"
    assert report["verdict"] == "WARNING"

    comment = RENDERER.render_comment(report, comment_args())
    validation = comment.split("### 验证情况", 1)[1].split("### 剩余风险", 1)[0]
    assert "  - 本次未新增验证命令。" in validation
    assert validation.count("本次未新增验证命令") == 1
    assert "本次没有新的命令执行结果" not in validation
    assert "  - 现有验证尚未覆盖本次变更的全部风险。" in validation


@pytest.mark.parametrize(
    "summary",
    [
        "本次未新增验证命令。",
        "本次无需执行额外验证命令",
        "本次没有必要执行新的诊断命令",
    ],
)
def test_no_command_fact_is_not_duplicated_when_codex_summary_already_states_it(
    tmp_path, summary
):
    document = analysis()
    document["test_assessment"] = {
        "evidence_level": "insufficient",
        "summary": [summary],
        "commands": [],
    }
    report = build(tmp_path, document, [], archive_entries=[])

    validation = RENDERER.render_comment(report, comment_args()).split(
        "### 验证情况", 1
    )[1].split("### 剩余风险", 1)[0]

    assert validation.count("本次未新增验证命令") == 1


def test_no_command_fact_is_not_duplicated_when_codex_summary_is_empty(tmp_path):
    document = analysis()
    document["test_assessment"] = {
        "evidence_level": "not_needed",
        "summary": [],
        "commands": [],
    }
    report = build(tmp_path, document, [], archive_entries=[])

    validation = RENDERER.render_comment(report, comment_args()).split(
        "### 验证情况", 1
    )[1].split("### 剩余风险", 1)[0]

    assert validation.count("本次未新增验证命令") == 1
    assert "本次不需要执行额外验证命令" not in validation


def test_public_narrative_fields_remove_internal_sources_and_enums(tmp_path):
    document = analysis()
    document["summary"] = "由Runner校验后仍是insufficient_evidence状态，但审查已完成。"
    document["merge_recommendation"] = "Codex 说明：结果为stable_failure，需要人工判断。"
    document["residual_risks"] = ["Runner仍需核对infrastructure_failure状态。"]
    report = build(
        tmp_path,
        document,
        [
            {
                "command": "python3 -m pytest generated_tests/test_generated.py",
                "exit_code": 0,
                "duration_seconds": 0.2,
            }
        ],
    )

    comment = RENDERER.render_comment(report, comment_args())
    for internal_term in (
        "Runner",
        "Codex 说明：",
        "insufficient_evidence",
        "stable_failure",
        "infrastructure_failure",
    ):
        assert internal_term not in comment


def test_unavailable_is_reserved_for_consistent_failure_fallback(tmp_path):
    document = analysis()
    report = build(tmp_path, document, [])
    report["test_execution"].update(
        {
            "evidence_level": "unavailable",
            "status": "unavailable",
            "summary": ["Runner 校验：没有形成可信的执行事实结论。"],
            "generated_test_files": [],
            "commands": [],
        }
    )
    report["verdict"] = "WARNING"
    RENDERER.validate_report(
        report,
        [{"path": "example.py", "change_type": "modified"}],
        tmp_path / "repo",
    )
    comment = RENDERER.render_comment(report, comment_args())
    validation = comment.split("### 验证情况", 1)[1].split("### 剩余风险", 1)[0]
    assert "本次命令执行事实不可确认" in validation
    assert "预期验证是否执行及其结果仍待核对" in validation
    assert "本次自动审查未形成完整的验证依据说明" in validation
    assert "Runner 校验：" not in validation

    invalid_report = json.loads(json.dumps(report))
    invalid_report["test_execution"]["commands"] = [
        {
            "id": "RUN-001",
            "command": "python3 -m pytest",
            "role": "unclassified",
            "purpose": "定向测试",
            "exit_code": 0,
            "duration_seconds": 0.1,
            "status": "not_executed",
            "evidence": "Runner 没有执行这条命令。",
        }
    ]
    with pytest.raises(ValueError, match="cannot contain command records"):
        RENDERER.validate_report(
            invalid_report,
            [{"path": "example.py", "change_type": "modified"}],
            tmp_path / "repo",
        )

    archive_report = json.loads(json.dumps(report))
    archive_report["test_execution"]["generated_test_files"] = [
        "generated_tests/test_generated.py"
    ]
    RENDERER.validate_report(
        archive_report,
        [{"path": "example.py", "change_type": "modified"}],
        tmp_path / "repo",
    )

    report["test_execution"]["status"] = "not_run"
    RENDERER.validate_report(
        report,
        [{"path": "example.py", "change_type": "modified"}],
        tmp_path / "repo",
    )


def test_semantic_evidence_can_be_sufficient_when_command_facts_are_unavailable(
    tmp_path,
):
    document = analysis()
    document["test_assessment"] = {
        "evidence_level": "sufficient",
        "summary": ["复用的确定性检查证据足以支持当前审查结论。"],
        "commands": [],
    }
    report = build(tmp_path, document, [], archive_entries=[])
    report["test_execution"]["status"] = "unavailable"
    report["verdict"] = "WARNING"

    RENDERER.validate_report(
        report,
        [{"path": "example.py", "change_type": "modified"}],
        tmp_path / "repo",
    )
    comment = RENDERER.render_comment(report, comment_args())
    assert "本次未形成可确认的具体代码问题" in comment
    assert "本次命令执行事实不可确认" in comment


def test_fallback_preserves_trusted_command_without_guessing_its_role(
    tmp_path,
):
    report = build_fallback(
        tmp_path,
        [
            {
                "command": "python3 -m pytest generated_tests/test_generated.py",
                "exit_code": 0,
                "duration_seconds": 0.2,
            }
        ],
        [("generated_tests/test_generated.py", b"def test_x(): pass\n", "file")],
    )

    assert report["verdict"] == "WARNING"
    assert report["test_execution"]["evidence_level"] == "unavailable"
    assert report["test_execution"]["status"] == "insufficient_evidence"
    assert report["test_execution"]["commands"][0]["status"] == "passed"
    assert report["test_execution"]["commands"][0]["role"] == "unclassified"
    assert report["test_execution"]["generated_test_files"] == [
        "generated_tests/test_generated.py"
    ]
    comment = RENDERER.render_comment(report, comment_args())
    assert "本次未形成可确认的具体代码问题" in comment
    assert "本次审查未发现需要处理的具体代码缺陷" not in comment
    assert "evidence_level" not in comment
    assert "test_execution.status" not in comment
    full_report = RENDERER.render_report(report, comment_args())
    assert "本次未形成可确认的具体代码问题" in full_report
    assert "未发现需要阻塞合并的关键问题" not in full_report


def test_fallback_keeps_repeated_failure_facts_without_guessing_validation_role(
    tmp_path,
):
    command = "python3 -m pytest generated_tests/test_generated.py"
    report = build_fallback(
        tmp_path,
        [
            {"command": command, "exit_code": 1, "duration_seconds": 0.2},
            {"command": command, "exit_code": 1, "duration_seconds": 0.3},
        ],
        [],
    )

    assert report["test_execution"]["status"] == "insufficient_evidence"
    assert {
        item["status"] for item in report["test_execution"]["commands"]
    } == {"stable_failure"}
    assert all(
        item["role"] == "unclassified"
        for item in report["test_execution"]["commands"]
    )


def test_fallback_distinguishes_empty_and_unavailable_command_ledger(tmp_path):
    empty = build_fallback(tmp_path, [], [])
    assert empty["test_execution"]["status"] == "not_run"
    assert empty["test_execution"]["commands"] == []

    unavailable = build_fallback(
        tmp_path,
        None,
        [("generated_tests/test_generated.py", b"def test_x(): pass\n", "file")],
        command_ledger_state="unavailable",
    )
    assert unavailable["test_execution"]["status"] == "unavailable"
    assert unavailable["test_execution"]["commands"] == []
    assert unavailable["test_execution"]["generated_test_files"] == [
        "generated_tests/test_generated.py"
    ]
    validation = RENDERER.render_comment(unavailable, comment_args()).split(
        "### 验证情况", 1
    )[1].split("### 剩余风险", 1)[0]
    assert validation.count("命令执行事实不可确认") == 1


def test_fallback_ignores_untrusted_archive_without_losing_report(tmp_path):
    report = build_fallback(
        tmp_path,
        [],
        [("../escape.py", b"x", "file")],
    )

    assert report["test_execution"]["status"] == "not_run"
    assert report["test_execution"]["generated_test_files"] == []
    assert any("测试文件归档不可确认" in risk for risk in report["residual_risks"])
    assert any(
        "未能取得可信的任务级测试文件归档事实" in item
        for item in report["test_execution"]["summary"]
    )


def test_fallback_rejects_missing_available_command_ledger(tmp_path):
    with pytest.raises(
        BUILDER.InvalidTrustedReportInput,
        match="available command ledger does not exist",
    ):
        build_fallback(tmp_path, None, [], command_ledger_state="available")


def test_not_needed_with_successful_inspection_ignores_generation_hint(tmp_path):
    document = analysis("git diff --stat")
    document["test_assessment"]["evidence_level"] = "not_needed"
    report = build(
        tmp_path,
        document,
        [{"command": "git diff --stat", "exit_code": 0, "duration_seconds": 0.1}],
        archive_entries=[],
        test_generation_expected=True,
    )
    assert report["test_execution"]["status"] == "passed"


def test_sufficient_existing_validation_derives_passed_without_generated_files(tmp_path):
    command = "python3 -m pytest scripts/local_ci/results/tests/test_local_ci_bridge.py -q"
    report = build(
        tmp_path,
        analysis(command),
        [{"command": command, "exit_code": 0, "duration_seconds": 0.2}],
        archive_entries=[],
        test_generation_expected=True,
    )
    assert report["test_execution"]["status"] == "passed"
    assert report["test_execution"]["generated_test_files"] == []


def test_insufficient_codex_evidence_is_not_upgraded_by_successful_command(tmp_path):
    command = "python3 -m pytest scripts/local_ci/results/tests/test_local_ci_bridge.py -q"
    document = analysis(command)
    document["test_assessment"]["evidence_level"] = "insufficient"
    document["test_assessment"]["summary"] = [
        "执行桥接单测并通过，当前没有列出仍需补充的验证项。"
    ]
    document["residual_risks"] = []
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.2}],
        archive_entries=[],
        test_generation_expected=True,
    )
    assert report["test_execution"]["evidence_level"] == "insufficient"
    assert report["test_execution"]["status"] == "passed"
    assert report["verdict"] == "WARNING"


def test_suggested_test_keeps_insufficient_with_successful_command(tmp_path):
    command = "python3 -m pytest scripts/local_ci/results/tests/test_local_ci_bridge.py -q"
    document = analysis(command)
    document["test_assessment"]["evidence_level"] = "insufficient"
    document["suggested_tests"] = [
        {
            "priority": "HIGH",
            "target": "scripts/local_ci/codex_ai/tests/test_local_ci_codex_container.sh",
            "description": "补跑容器契约以覆盖 runner、提示词和报告输出。",
        }
    ]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.2}],
        archive_entries=[],
        test_generation_expected=True,
    )
    assert report["test_execution"]["status"] == "passed"
    comment = RENDERER.render_comment(report, comment_args())
    validation = comment.split("### 验证情况", 1)[1].split("### 剩余风险", 1)[0]
    assert "尚未执行：补跑容器契约以覆盖自动检查、提示词和报告输出。" in validation


def test_generation_error_has_public_execution_facts_and_limit(tmp_path):
    document = analysis()
    document["test_assessment"] = {
        "evidence_level": "test_generation_error",
        "summary": ["创建定向测试时未能完成测试文件生成。"],
        "commands": [],
    }
    document["residual_risks"] = ["异常路径仍缺少动态验证。"]
    report = build(tmp_path, document, [], archive_entries=[])

    assert report["test_execution"]["status"] == "test_generation_error"
    comment = RENDERER.render_comment(report, comment_args())
    validation = comment.split("### 验证情况", 1)[1].split("### 剩余风险", 1)[0]
    assert "创建定向测试时未能完成测试文件生成" in validation
    assert "测试生成阶段未完成，当前没有形成预期的动态验证覆盖" in validation


def test_unannotated_success_is_not_promoted_to_formal_validation(tmp_path):
    command = "python3 -m pytest scripts/local_ci/results/tests/test_local_ci_bridge.py -q"
    document = analysis("not-an-executed-command")
    document["test_assessment"]["evidence_level"] = "insufficient"
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.2}],
        archive_entries=[],
        test_generation_expected=True,
    )
    assert report["test_execution"]["status"] == "not_run"
    assert report["test_execution"]["commands"][0]["role"] == "unclassified"


def test_unmatched_semantic_command_is_ignored(tmp_path):
    document = analysis("not-an-executed-command")
    document["test_assessment"]["evidence_level"] = "not_needed"
    report = build(tmp_path, document, [], archive_entries=[])
    assert report["test_execution"]["commands"] == []
    assert report["test_execution"]["status"] == "not_run"


def test_unknown_finding_file_id_becomes_explicit_residual_risk(tmp_path):
    document = analysis()
    document["findings"] = [finding(file_id="FILE-999")]
    command = document["test_assessment"]["commands"][0]["command"]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
    )
    assert report["findings"] == []
    assert report["unlocated_findings"][0]["title"] == "返回结果错误"
    assert report["verdict"] == "WARNING"
    assert report["test_execution"]["status"] == "passed"
    assert any("完整语义已保留" in risk for risk in report["residual_risks"])


def test_unlocated_high_finding_keeps_fail_verdict(tmp_path):
    document = analysis()
    high_finding = finding(file_id="FILE-999")
    high_finding["severity"] = "HIGH"
    document["findings"] = [high_finding]
    command = document["test_assessment"]["commands"][0]["command"]

    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
    )

    assert report["findings"] == []
    assert report["unlocated_findings"][0]["severity"] == "HIGH"
    assert report["verdict"] == "FAIL"


def test_unlocated_finding_keeps_full_semantics_in_public_comment(tmp_path):
    document = analysis()
    document["findings"] = [finding(file_id="FILE-999")]
    command = document["test_assessment"]["commands"][0]["command"]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
    )
    args = Namespace(
        branch="CI_dev",
        base_sha="a" * 40,
        requested_base_sha="",
        diff_mode="two-point",
        target_sha="b" * 40,
        head_sha="",
        local_ci_status="0",
        tested_sha_kind="commit",
        changed_file_count=1,
        constraint_status="pass",
        constraint_reason="未发现测试数量或耗时超出轻量约束。",
    )

    comment = RENDERER.render_comment(report, args)

    assert "[中风险·定位待核对] 返回结果错误" in comment
    assert "- 核心证据：该表达式会产生错误结果" in comment
    assert "该表达式会产生错误结果" in comment
    assert "调用方会收到错误结果" in comment
    assert "修正该表达式并补充测试" in comment
    assert "当前未发现需要阻塞合入的问题" in report["merge_recommendation"]
    assert report["merge_recommendation"] == document["merge_recommendation"]
    assert "结构化语义载荷" not in report["merge_recommendation"]


def test_public_comment_preserves_key_findings_within_length_budget(tmp_path):
    command = "python3 -m pytest generated_tests/test_generated.py"
    document = analysis(command)
    long_text = "这是用于验证评论长度预算的完整语义证据。" * 500
    document["change_request_assessment"]["evidence"] = [
        f"第 {index} 条判断依据。{long_text}" for index in range(8)
    ]
    document["findings"] = []
    for _ in range(10):
        item = finding()
        for key in ("code_role", "title", "evidence", "impact", "fix_direction"):
            item[key] = long_text
        document["findings"].append(item)
    document["residual_risks"] = [
        f"第 {index} 项剩余风险。{long_text}" for index in range(100)
    ]
    document["test_assessment"]["summary"] = [
        f"第 {index} 条验证说明。{long_text}" for index in range(8)
    ]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
    )
    args = Namespace(
        branch="CI_dev",
        base_sha="a" * 40,
        requested_base_sha="",
        diff_mode="two-point",
        target_sha="b" * 40,
        head_sha="",
        local_ci_status="0",
        tested_sha_kind="commit",
        changed_file_count=1,
        constraint_status="pass",
        constraint_reason="未发现测试数量或耗时超出轻量约束。",
    )

    comment = RENDERER.render_comment(report, args)

    assert len(comment) <= RENDERER.MAX_COMMENT_LENGTH
    assert "- 这段代码负责：" in comment
    assert "- 核心证据：" in comment
    assert "- 影响：" in comment
    assert "- 建议：" in comment
    assert "另有 4 条判断依据" in comment
    assert "另有 2 条验证说明" in comment
    assert "另有 94 项剩余风险" in comment
    assert "另有 5 个问题" in comment


def test_wide_finding_line_range_is_preserved(tmp_path):
    document = analysis()
    document["findings"] = [finding(line="2-20")]
    command = document["test_assessment"]["commands"][0]["command"]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
        source_text="".join(f"value_{index} = {index}\n" for index in range(1, 21)),
    )
    assert report["findings"][0]["line"] == "2-20"
    assert report["verdict"] == "WARNING"


def test_out_of_bounds_finding_becomes_explicit_residual_risk(tmp_path):
    document = analysis()
    document["findings"] = [finding(line="1-20")]
    command = document["test_assessment"]["commands"][0]["command"]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
        source_text="value = 1\n",
    )
    assert report["findings"] == []
    assert report["unlocated_findings"][0]["trusted_file"] == "example.py"
    assert any("完整语义已保留" in risk for risk in report["residual_risks"])


@pytest.mark.parametrize("line", ["0", "2-1", "not-a-line", "", None, 12])
def test_invalid_finding_line_is_preserved_as_unlocated(tmp_path, line):
    document = analysis()
    document["findings"] = [finding(line=line)]
    command = document["test_assessment"]["commands"][0]["command"]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
    )
    assert report["findings"] == []
    assert report["unlocated_findings"][0]["reported_line"]
    assert report["unlocated_findings"][0]["title"] == "返回结果错误"


def test_deleted_file_finding_is_preserved_as_unlocated(tmp_path):
    document = {"findings": [finding()]}
    valid, unlocated, warnings = BUILDER.build_findings(
        document,
        {"FILE-001": {"path": "example.py", "change_type": "deleted"}},
        tmp_path,
    )
    assert valid == []
    assert unlocated[0]["title"] == "返回结果错误"
    assert warnings


def test_unreadable_finding_file_becomes_explicit_residual_risk(tmp_path):
    document = analysis()
    document["findings"] = [finding(line="1")]
    command = document["test_assessment"]["commands"][0]["command"]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
        source_text=None,
    )
    assert report["findings"] == []
    assert report["unlocated_findings"][0]["trusted_file"] == "example.py"
    assert any("完整语义已保留" in risk for risk in report["residual_risks"])


def test_blank_finding_line_does_not_discard_report(tmp_path):
    document = analysis()
    document["findings"] = [finding(line="1")]
    command = document["test_assessment"]["commands"][0]["command"]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
        source_text="\nvalue = 1\n",
    )
    assert report["findings"][0]["line"] == "1"


def test_command_purpose_is_sanitized_after_language_normalization(tmp_path):
    document = analysis()
    document["test_assessment"]["commands"][0]["purpose"] = "RUN-001 " + "x" * 111
    command = document["test_assessment"]["commands"][0]["command"]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
    )
    purpose = report["test_execution"]["commands"][0]["purpose"]
    assert len(purpose) <= 120
    assert "RUN-001" not in purpose
    assert "相关验证" in purpose


@pytest.mark.parametrize(
    "path",
    [
        ("summary",),
        ("merge_recommendation",),
        ("change_request_assessment", "contributor_goal"),
        ("change_request_assessment", "expected_behavior"),
        ("change_request_assessment", "implementation_summary"),
        ("change_request_assessment", "evidence", 0),
        ("changed_files", 0, "summary"),
        ("changed_files", 0, "impact"),
        ("changed_files", 0, "validation_strategy"),
        *[
            ("behavior_coverage", behavior, field)
            for behavior in ("normal", "boundary", "error", "compatibility", "integration")
            for field in ("scope", "strategy", "result")
        ],
        *[
            ("findings", 0, field)
            for field in ("code_role", "title", "evidence", "impact", "fix_direction")
        ],
        ("suggested_tests", 0, "target"),
        ("suggested_tests", 0, "description"),
        ("residual_risks", 0),
        ("test_assessment", "summary", 0),
        ("test_assessment", "commands", 0, "purpose"),
        ("test_assessment", "commands", 0, "evidence"),
    ],
)
@pytest.mark.parametrize("replacement", ["", "English-only text."])
def test_all_human_readable_analysis_fields_are_safely_normalized(
    tmp_path, path, replacement
):
    command = "python3 -m pytest generated_tests/test_generated.py"
    document = analysis(command)
    document["findings"] = [finding()]
    document["suggested_tests"] = [
        {
            "priority": "LOW",
            "target": "generated_tests/test_generated.py",
            "description": "补充一个定向回归测试。",
        }
    ]
    set_nested(document, path, replacement)

    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
    )

    assert report["completion_marker"] == "CODEX_AI_CI_COMPLETE"


def test_blank_semantic_command_annotation_is_ignored(tmp_path):
    command = "python3 -m pytest generated_tests/test_generated.py"
    document = analysis(command)
    document["test_assessment"]["commands"][0]["command"] = ""

    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
    )

    assert report["test_execution"]["commands"][0]["purpose"] == (
        "Codex 执行的验证或诊断命令"
    )


@pytest.mark.parametrize(
    "changed_files",
    [
        [],
        [
            {
                "file_id": "FILE-999",
                "summary": "调整了示例代码。",
                "impact": "影响示例行为。",
                "validation_strategy": "执行定向检查。",
            }
        ],
    ],
)
def test_incomplete_changed_file_annotations_preserve_manifest(tmp_path, changed_files):
    document = analysis()
    document["changed_files"] = changed_files
    command = document["test_assessment"]["commands"][0]["command"]
    report = build(
        tmp_path,
        document,
        [{"command": command, "exit_code": 0, "duration_seconds": 0.1}],
    )
    assert report["changed_files"][0]["path"] == "example.py"
    assert "未提供该文件" in report["changed_files"][0]["summary"]
    assert report["test_execution"]["status"] == "passed"
    assert not any(
        "逐文件语义说明" in item for item in report["test_execution"]["summary"]
    )
    assert any("逐文件语义说明" in item for item in report["residual_risks"])
    assert report["verdict"] == "PASS"

    comment = RENDERER.render_comment(report, comment_args())
    validation = comment.split("### 验证情况", 1)[1].split("### 剩余风险", 1)[0]
    assert "逐文件语义说明" not in validation
    assert "逐文件语义说明" not in comment


def test_missing_behavior_category_is_rejected(tmp_path):
    document = analysis()
    del document["behavior_coverage"]["integration"]
    with pytest.raises(ValueError, match="behavior_coverage has invalid keys"):
        build(tmp_path, document, [])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evidence_level", "maybe", "test_assessment.evidence_level is invalid"),
        ("command_classification", "maybe", "failure_classification is invalid"),
        ("command_role", "maybe", "role is invalid"),
    ],
)
def test_invalid_semantic_enum_is_rejected(tmp_path, field, value, message):
    document = analysis()
    if field == "evidence_level":
        document["test_assessment"]["evidence_level"] = value
    elif field == "command_classification":
        document["test_assessment"]["commands"][0]["failure_classification"] = value
    else:
        document["test_assessment"]["commands"][0]["role"] = value
    with pytest.raises(ValueError, match=message):
        build(tmp_path, document, [])


def test_semantic_summary_overflow_and_duplicates_are_normalized(tmp_path):
    document = analysis()
    document["test_assessment"]["summary"] = ["重复说明。", "重复说明。"]
    report = build(tmp_path, document, [])
    assert report["test_execution"]["summary"] == ["Codex 说明：重复说明。"]

    document = analysis()
    document["change_request_assessment"]["evidence"] = [
        f"第 {index} 条判断依据。" for index in range(9)
    ]
    report = build(tmp_path, document, [])
    assert len(report["change_request_assessment"]["evidence"]) == 8


def test_workspace_archive_only_reports_test_paths(tmp_path):
    report = build(
        tmp_path,
        analysis(),
        [
            {
                "command": "python3 -m pytest generated_tests/test_generated.py",
                "exit_code": 0,
                "duration_seconds": 0.2,
            }
        ],
        archive_entries=[
            ("generated_tests/test_generated.py", b"def test_x(): pass\n", "file"),
            ("python/triton_anchor/runtime.py", b"changed = True\n", "file"),
            ("diagnostics/notes.txt", b"notes\n", "file"),
        ],
    )
    assert report["test_execution"]["generated_test_files"] == [
        "generated_tests/test_generated.py"
    ]


def test_english_and_missing_noncritical_fields_are_normalized(tmp_path):
    document = analysis()
    document["summary"] = "English-only summary."
    document["merge_recommendation"] = ""
    document["change_request_assessment"]["evidence"] = []
    document["test_assessment"]["summary"] = []
    report = build(
        tmp_path,
        document,
        [
            {
                "command": "python3 -m pytest generated_tests/test_generated.py",
                "exit_code": 0,
                "duration_seconds": 0.2,
            }
        ],
    )
    assert report["summary"].startswith("Codex 原始说明：")
    assert "确定性 CI" in report["merge_recommendation"]
    assert report["change_request_assessment"]["evidence"]
    assert report["test_execution"]["summary"]


def test_unreported_ledger_command_is_kept_as_unclassified_history(tmp_path):
    document = analysis("command-not-present-in-ledger")
    document["test_assessment"]["evidence_level"] = "not_needed"
    actual = "python3 -m pytest generated_tests/test_generated.py"
    report = build(
        tmp_path,
        document,
        [{"command": actual, "exit_code": 0, "duration_seconds": 0.2}],
    )
    assert report["test_execution"]["commands"][0]["command"] == actual
    assert report["test_execution"]["commands"][0]["role"] == "unclassified"
    assert report["test_execution"]["status"] == "not_run"


def test_unclassified_search_failures_do_not_override_formal_validation(tmp_path):
    validation_command = "python3 -m pytest report_tests.py"
    document = analysis(validation_command)
    document["test_assessment"]["summary"] = [
        "报告契约定向测试执行通过。",
    ]
    report = build(
        tmp_path,
        document,
        [
            {
                "command": validation_command,
                "exit_code": 0,
                "duration_seconds": 0.2,
            },
            {
                "command": "rg old_field scripts/local_ci",
                "exit_code": 127,
                "duration_seconds": 0.1,
            },
            {
                "command": "grep -R old_field scripts/local_ci",
                "exit_code": 1,
                "duration_seconds": 0.1,
            },
        ],
    )
    assert report["test_execution"]["status"] == "passed"
    assert report["verdict"] == "PASS"
    assert [
        command["role"] for command in report["test_execution"]["commands"]
    ] == ["validation", "unclassified", "unclassified"]
    assert any(
        "2 条非零退出命令没有可匹配的用途说明" in risk
        and "未用于派生正式验证状态" in risk
        for risk in report["residual_risks"]
    )
    assert report["merge_recommendation"] == document["merge_recommendation"]
    comment = RENDERER.render_comment(report, comment_args())
    assert "部分辅助检查没有形成可确认的结果" not in comment


@pytest.mark.parametrize(
    "entry",
    [
        ("../escape.py", b"x", "file"),
        ("generated_tests/link.py", b"", "symlink"),
    ],
)
def test_generated_archive_rejects_escape_and_symlink(tmp_path, entry):
    with pytest.raises(ValueError):
        build(tmp_path, analysis(), [], archive_entries=[entry])


def assert_closed_object_schema(node: dict, expected_keys: set[str]) -> None:
    assert node["type"] == "object"
    assert set(node["properties"]) == expected_keys
    assert set(node["required"]) == expected_keys
    assert node["additionalProperties"] is False


def test_analysis_schema_matches_builder_contract() -> None:
    schema = json.loads(
        (CODEX_AI_DIR / "codex_ai_analysis.schema.json").read_text(encoding="utf-8")
    )
    assert_closed_object_schema(schema, BUILDER.ANALYSIS_KEYS)
    properties = schema["properties"]
    assert_closed_object_schema(
        properties["change_request_assessment"], BUILDER.ASSESSMENT_KEYS
    )
    assert_closed_object_schema(
        properties["changed_files"]["items"], BUILDER.CHANGED_FILE_KEYS
    )
    file_id_pattern = properties["changed_files"]["items"]["properties"][
        "file_id"
    ]["pattern"]
    assert re.fullmatch(file_id_pattern, "FILE-1000")
    assert_closed_object_schema(
        properties["behavior_coverage"], set(BUILDER.BEHAVIOR_LABELS)
    )
    for name in BUILDER.BEHAVIOR_LABELS:
        assert_closed_object_schema(
            properties["behavior_coverage"]["properties"][name],
            BUILDER.BEHAVIOR_ITEM_KEYS,
        )
    assert_closed_object_schema(properties["findings"]["items"], BUILDER.FINDING_KEYS)
    assert_closed_object_schema(
        properties["suggested_tests"]["items"], BUILDER.SUGGESTED_TEST_KEYS
    )
    assessment = properties["test_assessment"]
    assert_closed_object_schema(assessment, BUILDER.TEST_ASSESSMENT_KEYS)
    assert_closed_object_schema(
        assessment["properties"]["commands"]["items"],
        BUILDER.COMMAND_ANNOTATION_KEYS,
    )
    assert set(
        properties["change_request_assessment"]["properties"]["status"]["enum"]
    ) == BUILDER.ASSESSMENT_STATUSES
    assert set(properties["findings"]["items"]["properties"]["severity"]["enum"]) == (
        BUILDER.SEVERITIES
    )
    assert set(properties["findings"]["items"]["properties"]["category"]["enum"]) == (
        BUILDER.CATEGORIES
    )
    assert set(assessment["properties"]["evidence_level"]["enum"]) == (
        BUILDER.EVIDENCE_LEVELS
    )
    assert set(
        assessment["properties"]["commands"]["items"]["properties"][
            "failure_classification"
        ]["enum"]
    ) == BUILDER.FAILURE_CLASSIFICATIONS
    assert set(
        assessment["properties"]["commands"]["items"]["properties"]["role"][
            "enum"
        ]
    ) == BUILDER.COMMAND_ROLES


def test_report_schema_matches_renderer_contract() -> None:
    schema = json.loads(
        (CODEX_AI_DIR / "codex_ai_report.schema.json").read_text(encoding="utf-8")
    )
    assert_closed_object_schema(schema, RENDERER.ROOT_KEYS)
    properties = schema["properties"]
    assert_closed_object_schema(
        properties["change_request_assessment"],
        RENDERER.CHANGE_REQUEST_ASSESSMENT_KEYS,
    )
    assert_closed_object_schema(
        properties["changed_files"]["items"], RENDERER.CHANGED_FILE_KEYS
    )
    assert_closed_object_schema(
        properties["behavior_coverage"], RENDERER.BEHAVIOR_COVERAGE_KEYS
    )
    for name in RENDERER.BEHAVIOR_COVERAGE_KEYS:
        assert_closed_object_schema(
            properties["behavior_coverage"]["properties"][name],
            RENDERER.BEHAVIOR_ITEM_KEYS,
        )
    assert_closed_object_schema(properties["findings"]["items"], RENDERER.FINDING_KEYS)
    assert_closed_object_schema(
        properties["unlocated_findings"]["items"],
        RENDERER.UNLOCATED_FINDING_KEYS,
    )
    assert_closed_object_schema(
        properties["suggested_tests"]["items"], RENDERER.TEST_KEYS
    )
    execution = properties["test_execution"]
    assert_closed_object_schema(execution, RENDERER.TEST_EXECUTION_KEYS)
    assert_closed_object_schema(
        execution["properties"]["commands"]["items"], RENDERER.COMMAND_KEYS
    )
    assert set(properties["verdict"]["enum"]) == {"PASS", "WARNING", "FAIL"}
    assert set(
        properties["change_request_assessment"]["properties"]["status"]["enum"]
    ) == RENDERER.CHANGE_REQUEST_ASSESSMENT_STATUSES
    assert set(properties["findings"]["items"]["properties"]["severity"]["enum"]) == (
        RENDERER.SEVERITIES
    )
    assert set(properties["findings"]["items"]["properties"]["category"]["enum"]) == (
        RENDERER.CATEGORIES
    )
    assert set(execution["properties"]["status"]["enum"]) == (
        RENDERER.TEST_EXECUTION_STATUSES
    )
    assert set(execution["properties"]["evidence_level"]["enum"]) == (
        RENDERER.EVIDENCE_LEVELS
    )
    assert execution["properties"]["summary"]["anyOf"][1]["maxItems"] == (
        RENDERER.MAX_TEST_EXECUTION_SUMMARY_ITEMS
    )
    assert set(
        execution["properties"]["commands"]["items"]["properties"]["status"][
            "enum"
        ]
    ) == RENDERER.COMMAND_STATUSES
    assert set(
        execution["properties"]["commands"]["items"]["properties"]["role"][
            "enum"
        ]
    ) == RENDERER.COMMAND_ROLES
    identifier_examples = {
        "findings": "AI-1000",
        "unlocated_findings": "AI-1000",
        "suggested_tests": "TEST-1000",
    }
    for key, value in identifier_examples.items():
        pattern = properties[key]["items"]["properties"]["id"]["pattern"]
        assert re.fullmatch(pattern, value)
    command_pattern = execution["properties"]["commands"]["items"]["properties"][
        "id"
    ]["pattern"]
    assert re.fullmatch(command_pattern, "RUN-1000")
