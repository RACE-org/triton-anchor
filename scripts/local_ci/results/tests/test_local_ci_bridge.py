from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "local_ci" / "results"))

import bridge_gitee_to_github_status as bridge


class BridgeArgumentTests(unittest.TestCase):
    def parse(self, *extra: str):
        argv = [
            "bridge",
            "--gitee-owner",
            "owner",
            "--gitee-repo",
            "results",
            "--gitee-web-url",
            "https://gitee.example/results",
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            return bridge.parse_args()

    def test_source_branch_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse()

    def test_status_sha_cannot_differ_from_tested_sha(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse(
                "--source-branch",
                "feature/demo",
                "--sha",
                "a" * 40,
                "--status-sha",
                "b" * 40,
            )

    def test_status_sha_requires_tested_sha(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse(
                "--source-branch",
                "feature/demo",
                "--status-sha",
                "a" * 40,
            )

    def test_status_sha_may_equal_tested_sha(self) -> None:
        args = self.parse(
            "--source-branch",
            "feature/demo",
            "--sha",
            "a" * 40,
            "--status-sha",
            "a" * 40,
        )
        self.assertEqual(args.source_branch, "feature/demo")


class CodexCommentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = bridge.Target(
            "feature",
            "ci/pr-42/feature",
            "a" * 40,
            "PR #42 feature",
            head_sha="b" * 40,
        )
        self.result = bridge.LocalCIResult(
            0,
            "https://gitee.example/results/run",
            "run-1",
            "pass",
            "pass",
            "pass",
            {},
            bridge.CodexAIResult(
                "pass",
                "WARNING",
                "stable_failure",
                "full",
                "warning",
                "生成测试和执行命令均在限制内。",
                "",
                (
                    "## Codex AI 代码审查\n\n"
                    "> 这是非阻塞的辅助审查；确定性 CI 结果仍是合入门禁。\n\n"
                    "### 审查摘要\n\n"
                    "- Codex 执行状态：**完成**。\n"
                    "- Codex 建议性结论（非阻塞）：**警告**\n\n"
                    "发现一个问题。\n\nCodex AI CI 已完成，Local CI 已通过。\n\n"
                    "### 贡献者目标与实现情况\n\n实现不完整。\n\n"
                    "### 需要处理的问题\n\n#### 1. [中风险] 示例问题\n\n"
                    "### 验证情况\n\nRUN-001 稳定复现问题。\n\n"
                    "### 变更文件\n\n<details></details>"
                ),
                "https://gitee.example/results/run/codex-ai-report.md",
                (
                    bridge.FindingLocation(
                        "AI-001", "python/example.py", "17-18"
                    ),
                ),
                "",
                (("RUN-001", "缓存版本失配定向测试"),),
            ),
        )

    def test_pr_number_is_derived_only_from_pr_task_refs(self) -> None:
        self.assertEqual(bridge.pr_number_from_task_ref(self.target.task_ref), 42)
        self.assertIsNone(bridge.pr_number_from_task_ref("ci/push/feature"))

    def test_validation_purpose_is_read_from_report_and_used_in_comment(self) -> None:
        report_json = json.dumps(
            {
                "test_execution": {
                    "commands": [
                        {"id": "RUN-001", "purpose": "Python 语法检查"}
                    ]
                }
            }
        )
        purposes = bridge.validation_purposes_from_report(report_json)

        self.assertEqual(purposes, (("RUN-001", "Python 语法检查"),))
        self.assertEqual(
            bridge.public_comment_text("RUN-001 已通过", purposes),
            "Python 语法检查已通过",
        )
        self.assertEqual(
            bridge.public_comment_text("RUN-001 已通过"),
            "相关验证已通过",
        )

    def test_four_digit_internal_ids_are_supported_and_hidden(self) -> None:
        report_json = json.dumps(
            {
                "findings": [
                    {
                        "id": "AI-1000",
                        "file": "python/example.py",
                        "line": "17",
                    }
                ],
                "test_execution": {
                    "commands": [
                        {"id": "RUN-1000", "purpose": "大规模变更定向检查"}
                    ]
                },
            }
        )
        purposes = bridge.validation_purposes_from_report(report_json)
        locations = bridge.finding_locations_from_report(report_json)
        public_text = bridge.public_comment_text(
            "RUN-1000 已验证 AI-1000", purposes
        )

        self.assertEqual(purposes, (("RUN-1000", "大规模变更定向检查"),))
        self.assertEqual(
            locations,
            (bridge.FindingLocation("AI-1000", "python/example.py", "17"),),
        )
        self.assertNotIn("RUN-1000", public_text)
        self.assertNotIn("AI-1000", public_text)

    def test_comment_body_contains_summary_link_and_stable_marker(self) -> None:
        body = bridge.codex_pr_comment_body(self.target, self.result)
        self.assertIn("发现一个问题。", body)
        self.assertIn("## Codex AI 自动审查", body)
        self.assertIn("### 审查摘要", body)
        self.assertIn("### 贡献者目标与实现情况", body)
        self.assertIn("### 变更文件", body)
        self.assertIn("- 测试提交：", body)
        self.assertIn("`aaaaaaaaaaaa`", body)
        self.assertIn("缓存版本失配定向测试稳定复现问题", body)
        self.assertIn("Codex AI 自动审查已完成，本地确定性 CI 检查已通过", body)
        self.assertIn("Codex AI 审查结论：**需关注（非阻塞）**", body)
        main, footer = body.split("\n---\n", 1)
        self.assertNotIn("Codex 执行状态", main)
        self.assertIn("Codex 执行状态：完成", footer)
        self.assertNotIn("Codex 建议性结论", body)
        self.assertIn(bridge.CODEX_COMMENT_MARKER, body)
        self.assertIn(bridge.codex_pr_commit_marker(self.target), body)

    def test_docs_only_comment_keeps_policy_skip_semantics(self) -> None:
        codex_ai = replace(
            self.result.codex_ai,
            comment_markdown=(
                "## Codex AI 自动审查\n\n"
                "> Codex AI 自动审查仅供参考且不阻塞合入。\n\n"
                "### 审查摘要\n\n"
                "- Codex 建议性结论（非阻塞）：**警告**\n\n"
                "本地确定性 CI 检查已通过。\n\n"
                "- 本地确定性 CI 检查：已通过；Codex AI 自动审查只提供补充意见。\n\n"
                "### 验证情况\n\n本次只检查文档。\n"
            ),
        )
        result = replace(
            self.result,
            codex_ai=codex_ai,
            execution_mode="codex_only",
        )

        body = bridge.codex_pr_comment_body(self.target, result)

        self.assertIn("本次仅含文档变更，按策略未执行确定性 CI", body)
        self.assertNotIn("本地确定性 CI 检查：已通过", body)
        self.assertNotIn("本地确定性 CI 检查已通过", body)
        self.assertIn("Codex AI 审查结论：**需关注（非阻塞）**", body)
        footer = body.split("\n---\n", 1)[1]
        self.assertNotIn("本地确定性", footer)
        self.assertIn("Codex 执行状态：完成", footer)

    def test_footer_keeps_failure_reason_without_repeating_status_or_verdict(self) -> None:
        cases = (
            (
                "PASS",
                "WARNING",
                "",
                "Codex 执行状态：完成",
                "",
            ),
            (
                "fail",
                "FAIL",
                "timeout",
                "Codex 执行状态：未完成",
                "未完成原因：Codex 自动审查执行超时",
            ),
            (
                "pass",
                "FAIL",
                "startup_timeout",
                "Codex 执行状态：未完成",
                "未完成原因：Codex 自动审查启动超时",
            ),
            (
                "skipped",
                "NOT_RUN",
                "",
                "Codex 执行状态：未运行",
                "状态说明：本次任务按策略未运行 Codex 自动审查",
            ),
            (
                "not_reported",
                "UNKNOWN",
                "",
                "Codex 执行状态：未完成",
                "未完成原因：尚未收到 Codex 自动审查结果",
            ),
        )
        for execution_status, verdict, failure_code, expected_status, reason in cases:
            with self.subTest(
                execution_status=execution_status,
                failure_code=failure_code,
            ):
                codex_ai = replace(
                    self.result.codex_ai,
                    execution_status=execution_status,
                    verdict=verdict,
                    failure_code=failure_code,
                )
                body = bridge.codex_pr_comment_body(
                    self.target, replace(self.result, codex_ai=codex_ai)
                )
                main, footer = body.split("\n---\n", 1)
                self.assertNotIn("Codex 自动审查状态", footer)
                self.assertNotIn("Codex 执行状态", main)
                self.assertIn(expected_status, footer)
                self.assertNotIn("建议性结论", footer)
                if reason:
                    self.assertIn(reason, footer)
                    self.assertNotIn("Codex AI 自动审查已完成", body)
                    self.assertNotIn("Codex AI 审查结论", body)
                else:
                    self.assertNotIn("未完成原因", footer)
                    self.assertIn(
                        "Codex AI 审查结论：**需关注（非阻塞）**", body
                    )

    def test_footer_only_links_nonempty_report_url(self) -> None:
        for report_url, expected in (
            (self.result.codex_ai.report_url, True),
            ("", False),
        ):
            with self.subTest(report_url=report_url):
                result = replace(
                    self.result,
                    codex_ai=replace(self.result.codex_ai, report_url=report_url),
                )
                body = bridge.codex_pr_comment_body(self.target, result)
                self.assertEqual(
                    "查看完整 Codex AI 自动审查报告" in body,
                    expected,
                )

    def test_comment_body_links_findings_to_the_reviewed_commit(self) -> None:
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            body = bridge.codex_pr_comment_body(self.target, self.result)

        self.assertIn("### 可点击代码定位", body)
        self.assertIn("提交者修复和审核者核对代码功能", body)
        self.assertIn("问题 1：", body)
        self.assertNotIn("AI-001", body)
        self.assertLess(
            body.index("### 可点击代码定位"),
            body.index("### 验证情况"),
        )
        self.assertIn(
            "https://github.com/owner/repo/blob/"
            f"{self.target.sha}/python/example.py#L17-L18",
            body,
        )

    def test_final_comment_does_not_rewrite_real_ids_in_evidence_or_paths(self) -> None:
        comment = self.result.codex_ai.comment_markdown.replace(
            "#### 1. [中风险] 示例问题",
            "#### 1. [中风险] 示例问题\n\n- 核心证据：常量 `AI-001` 保持不变。",
        ).replace(
            "<details></details>",
            "<details>\n\n| `docs/AI-001.md` | 修改 | 说明 | 影响 |\n\n</details>",
        )
        result = replace(
            self.result,
            codex_ai=replace(self.result.codex_ai, comment_markdown=comment),
        )

        body = bridge.codex_pr_comment_body(self.target, result)

        self.assertIn("常量 `AI-001` 保持不变", body)
        self.assertIn("`docs/AI-001.md`", body)

    def test_final_comment_length_keeps_links_sections_and_footer(self) -> None:
        oversized_comment = self.result.codex_ai.comment_markdown.replace(
            "<details></details>",
            "<details>\n" + ("| `very-long.py` | 修改 | 说明 | 影响 |\n" * 2_000) + "</details>",
        )
        result = replace(
            self.result,
            codex_ai=replace(
                self.result.codex_ai,
                comment_markdown=oversized_comment,
            ),
        )

        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            body = bridge.codex_pr_comment_body(self.target, result)

        self.assertLessEqual(len(body), bridge.MAX_CODEX_PR_COMMENT_LENGTH)
        self.assertIn("### 可点击代码定位", body)
        self.assertIn("### 验证情况", body)
        self.assertIn("### 变更文件", body)
        self.assertIn("公开评论已按长度上限省略文件表", body)
        self.assertIn(bridge.CODEX_COMMENT_MARKER, body)
        self.assertIn(bridge.codex_pr_commit_marker(self.target), body)

    def test_wide_finding_range_keeps_clickable_location(self) -> None:
        locations = bridge.finding_locations_from_report(
            json.dumps(
                {
                    "findings": [
                        {
                            "id": "AI-001",
                            "file": "python/example.py",
                            "line": "17-42",
                        }
                    ]
                }
            )
        )
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            links = bridge.github_finding_location_links(self.target, locations)

        self.assertEqual(
            locations,
            (bridge.FindingLocation("AI-001", "python/example.py", "17-42"),),
        )
        self.assertIn(
            "https://github.com/owner/repo/blob/"
            f"{self.target.sha}/python/example.py#L17-L42",
            links,
        )

    def test_unlocated_finding_with_trusted_file_gets_file_only_link(self) -> None:
        locations = bridge.finding_locations_from_report(
            json.dumps(
                {
                    "findings": [],
                    "unlocated_findings": [
                        {
                            "id": "AI-001",
                            "trusted_file": "python/example.py",
                            "reported_line": "9999",
                        },
                        {
                            "id": "AI-002",
                            "trusted_file": "",
                            "reported_line": "17",
                        },
                    ],
                }
            )
        )
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            links = bridge.github_finding_location_links(self.target, locations)

        self.assertEqual(
            locations,
            (bridge.FindingLocation("AI-001", "python/example.py", ""),),
        )
        file_url = (
            "https://github.com/owner/repo/blob/"
            f"{self.target.sha}/python/example.py"
        )
        self.assertIn(f"[python/example.py（具体行号待核对）]({file_url})", links)
        self.assertNotIn(f"{file_url}#L", links)
        self.assertNotIn("问题 2", links)

    def test_fork_pr_location_uses_head_repository(self) -> None:
        fork_target = bridge.Target(
            self.target.source_branch,
            self.target.task_ref,
            self.target.sha,
            self.target.label,
            "fork-owner/fork-repo",
        )
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            body = bridge.codex_pr_comment_body(fork_target, self.result)

        self.assertIn("https://github.com/fork-owner/fork-repo/blob/", body)

    @mock.patch.object(bridge, "get_github_json")
    def test_reconcile_pr_targets_use_test_merge_sha(self, get_json: mock.Mock) -> None:
        get_json.return_value = [
            {
                "number": 42,
                "head": {
                    "ref": "feature",
                    "sha": "a" * 40,
                    "repo": {"full_name": "owner/repo"},
                },
                "merge_commit_sha": "b" * 40,
            }
        ]
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            targets = bridge.list_open_pr_targets(10)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].sha, "b" * 40)
        self.assertEqual(targets[0].task_ref, "ci/pr-42/feature")

    @mock.patch.object(bridge, "get_github_json")
    def test_reconcile_pr_targets_do_not_fallback_to_head_sha(
        self, get_json: mock.Mock
    ) -> None:
        get_json.return_value = [
            {
                "number": 42,
                "head": {
                    "ref": "feature",
                    "sha": "a" * 40,
                    "repo": {"full_name": "owner/repo"},
                },
                "merge_commit_sha": None,
            }
        ]
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            self.assertEqual(bridge.list_open_pr_targets(10), [])

    @mock.patch.object(bridge, "get_github_json")
    def test_current_pr_matches_frozen_merge_result(self, get_json: mock.Mock) -> None:
        get_json.return_value = {
            "state": "open",
            "draft": False,
            "head": {"sha": "a" * 40},
            # The PR API base SHA may lag behind the base parent frozen in the
            # GitHub merge-result; freshness is bound to head/ref/merge SHA.
            "base": {"sha": "e" * 40, "ref": "CI_dev"},
            "merge_commit_sha": "c" * 40,
        }
        args = SimpleNamespace(
            pr_number="42",
            expected_head_sha="a" * 40,
            comparison_base_sha="b" * 40,
            target_branch="CI_dev",
        )
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            self.assertTrue(bridge.current_pr_matches(args, "c" * 40))
            self.assertFalse(bridge.current_pr_matches(args, "d" * 40))

    @mock.patch.object(bridge, "get_github_json")
    def test_current_pr_rejects_force_push_and_retarget(self, get_json: mock.Mock) -> None:
        get_json.return_value = {
            "state": "open",
            "draft": False,
            "head": {"sha": "d" * 40},
            "base": {"sha": "b" * 40, "ref": "main"},
            "merge_commit_sha": "c" * 40,
        }
        args = SimpleNamespace(
            pr_number="42",
            expected_head_sha="a" * 40,
            comparison_base_sha="b" * 40,
            target_branch="CI_dev",
        )
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            self.assertFalse(bridge.current_pr_matches(args, "c" * 40))

    @mock.patch.object(bridge, "request_json")
    @mock.patch.object(bridge, "get_github_json", return_value=[])
    def test_new_pr_comment_is_created(
        self, get_json: mock.Mock, request_json: mock.Mock
    ) -> None:
        request_json.return_value = (201, {}, "")
        with mock.patch.dict(
            os.environ,
            {"GITHUB_REPOSITORY": "owner/repo", "GITHUB_TOKEN": "token"},
        ):
            bridge.post_codex_pr_comment(self.target, self.result)

        get_json.assert_called_once_with(
            "/repos/owner/repo/issues/42/comments", {"per_page": "100"}
        )
        self.assertEqual(request_json.call_args.kwargs["method"], "POST")

    @mock.patch.object(bridge, "request_json")
    @mock.patch.object(bridge, "get_github_json")
    def test_existing_codex_comment_for_same_commit_is_updated(
        self, get_json: mock.Mock, request_json: mock.Mock
    ) -> None:
        get_json.return_value = [
            {
                "id": 99,
                "body": (
                    f"{bridge.CODEX_COMMENT_MARKER}\n"
                    f"{bridge.codex_pr_commit_marker(self.target)}"
                ),
                "user": {"type": "Bot"},
            }
        ]
        request_json.return_value = (200, {}, "")
        with mock.patch.dict(
            os.environ,
            {"GITHUB_REPOSITORY": "owner/repo", "GITHUB_TOKEN": "token"},
        ):
            bridge.post_codex_pr_comment(self.target, self.result)

        self.assertEqual(request_json.call_args.kwargs["method"], "PATCH")
        self.assertIn("/issues/comments/99", request_json.call_args.args[0])

    @mock.patch.object(bridge, "request_json")
    @mock.patch.object(bridge, "get_github_json")
    def test_existing_codex_comment_for_different_commit_is_not_updated(
        self, get_json: mock.Mock, request_json: mock.Mock
    ) -> None:
        other_target = bridge.Target(
            self.target.source_branch,
            self.target.task_ref,
            "b" * 40,
            self.target.label,
        )
        get_json.return_value = [
            {
                "id": 99,
                "body": (
                    f"{bridge.CODEX_COMMENT_MARKER}\n"
                    f"{bridge.codex_pr_commit_marker(other_target)}"
                ),
                "user": {"type": "Bot"},
            }
        ]
        request_json.return_value = (201, {}, "")
        with mock.patch.dict(
            os.environ,
            {"GITHUB_REPOSITORY": "owner/repo", "GITHUB_TOKEN": "token"},
        ):
            bridge.post_codex_pr_comment(self.target, self.result)

        self.assertEqual(request_json.call_args.kwargs["method"], "POST")
        self.assertIn("/issues/42/comments", request_json.call_args.args[0])

    def test_push_does_not_publish_pr_comment(self) -> None:
        push_target = bridge.Target(
            "feature",
            "ci/push/feature",
            "b" * 40,
            "branch feature",
        )
        with (
            mock.patch.object(bridge, "get_github_json") as get_json,
            mock.patch.object(bridge, "request_json") as request_json,
        ):
            bridge.post_codex_pr_comment(push_target, self.result)
        get_json.assert_not_called()
        request_json.assert_not_called()

    def test_codex_advisory_status_is_always_non_blocking(self) -> None:
        args = SimpleNamespace(context="local-ci/test")
        with mock.patch.object(bridge, "post_github_status") as post_status:
            bridge.post_codex_advisory_status(args, self.target, self.result)

        status_args = post_status.call_args.args
        self.assertEqual(status_args[0], self.target.sha)
        self.assertEqual(status_args[1], "success")
        self.assertEqual(status_args[2], "local-ci/test/codex-ai-advisory")
        self.assertIn("可稳定复现的失败", status_args[3])
        self.assertIn("非阻塞", status_args[3])
        self.assertEqual(status_args[4], self.result.codex_ai.report_url)

    def test_advisory_descriptions_cover_non_pass_states(self) -> None:
        base = self.result.codex_ai

        def changed(**values: str) -> bridge.CodexAIResult:
            payload = {
                name: getattr(base, name)
                for name in bridge.CodexAIResult.__dataclass_fields__
            }
            payload.update(values)
            return bridge.CodexAIResult(**payload)

        cases = (
            (
                changed(execution_status="fail", failure_code="timeout"),
                "Codex 自动审查执行超时",
            ),
            (changed(execution_status="skipped"), "未运行"),
            (
                changed(verdict="PASS", test_status="insufficient_evidence"),
                "证据不足",
            ),
            (
                changed(verdict="PASS", test_status="stable_failure"),
                "可稳定复现的失败",
            ),
            (
                changed(verdict="PASS", test_status="flaky_failure"),
                "非确定性失败",
            ),
            (
                changed(verdict="PASS", test_status="infrastructure_failure"),
                "受环境限制，未完全执行",
            ),
            (
                changed(verdict="PASS", test_status="test_generation_error"),
                "测试生成失败",
            ),
            (changed(verdict="FAIL", test_status="passed"), "失败"),
            (changed(verdict="WARNING", test_status="passed"), "需关注"),
            (
                changed(
                    verdict="PASS",
                    test_status="passed",
                    constraint_status="warning",
                ),
                "约束警告",
            ),
            (
                changed(
                    verdict="PASS",
                    test_status="passed",
                    constraint_status="pass",
                ),
                "通过",
            ),
        )
        for codex_ai, expected in cases:
            with self.subTest(expected=expected):
                description = bridge.codex_advisory_description(codex_ai)
                self.assertIn(expected, description)
                self.assertIn("非阻塞", description)

    def test_new_report_failure_codes_have_stable_public_labels(self) -> None:
        expected = {
            "container_prepare_timeout": "Codex 审查运行环境准备超时",
            "startup_timeout": "Codex 自动审查启动超时",
            "analysis_contract_failed": "Codex 自动审查结果整理阶段未能生成公开摘要",
            "schema_validation_failed": "Codex 自动审查结果整理阶段未能生成公开摘要",
            "trusted_report_input_failed": "Codex 自动审查结果汇总阶段未能核对代码差异与执行记录",
            "report_contract_failed": "Codex 自动审查报告生成阶段未完成",
            "report_metadata_failed": "Codex 自动审查验证结果汇总阶段未完成",
        }
        for code, label in expected.items():
            with self.subTest(code=code):
                public_label = bridge.public_failure_reason(code)
                self.assertEqual(public_label, label)
                self.assertNotRegex(public_label, r"Runner|schema|语义载荷|内部契约")

    def test_internal_failure_terms_are_rewritten_in_public_comment(self) -> None:
        cases = (
            (
                "结构化报告未通过 schema、固定格式或中文内容校验。",
                "自动审查结果整理阶段未能生成公开摘要。",
            ),
            (
                "Codex 审查语义载荷未满足公开结构契约。",
                "Codex 自动审查结果整理阶段未能生成公开摘要。",
            ),
            (
                "Runner 生成的可信报告输入校验失败。",
                "Codex 自动审查结果汇总阶段未能核对代码差异与执行记录。",
            ),
            (
                "Runner 生成报告时内部契约校验失败。",
                "Codex 自动审查报告生成阶段未完成。",
            ),
            (
                "Runner 读取报告执行事实失败。",
                "Codex 自动审查验证结果汇总阶段未完成。",
            ),
        )
        for original, expected in cases:
            with self.subTest(original=original):
                public_text = bridge.public_comment_text(original)
                self.assertEqual(public_text, expected)
                self.assertNotRegex(
                    public_text, r"Runner|schema|语义载荷|内部契约"
                )
        self.assertEqual(
            bridge.public_comment_text("Local CI profile 与 Local CI Gateway"),
            "Local CI profile 与 Local CI Gateway",
        )

    def test_codex_ai_output_is_single_line_json(self) -> None:
        encoded = bridge.codex_ai_output_json(self.result)
        self.assertNotIn("\n", encoded)
        payload = json.loads(encoded)
        self.assertEqual(payload["execution_status"], "pass")
        self.assertEqual(payload["verdict"], "WARNING")
        self.assertEqual(payload["test_status"], "stable_failure")
        self.assertIn("发现一个问题", payload["comment_markdown"])
        self.assertEqual(
            payload["finding_locations"],
            [{"id": "AI-001", "file": "python/example.py", "line": "17-18"}],
        )
        self.assertEqual(
            payload["report_url"],
            self.result.codex_ai.report_url,
        )

    def test_write_github_outputs_includes_codex_ai_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "github-output.txt"
            with mock.patch.dict(
                os.environ,
                {"GITHUB_OUTPUT": str(output_path)},
            ):
                bridge.write_github_outputs(self.result)

            values = dict(
                line.split("=", 1)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            )
        payload = json.loads(values["codex_ai_result"])
        self.assertEqual(payload["analysis_mode"], "full")
        self.assertEqual(payload["constraint_status"], "warning")

    @mock.patch.object(bridge, "gitee_content")
    def test_read_result_combines_result_summary_and_comment(
        self, gitee_content: mock.Mock
    ) -> None:
        def content(
            owner: str,
            repo: str,
            path: str,
            ref: str,
            token: str,
        ) -> str | None:
            del owner, repo, ref, token
            if path.endswith("/latest.txt"):
                return "run-1\n"
            if path.endswith("/publish-manifest.json"):
                return json.dumps(
                    {
                        "schema": "triton-anchor-local-ci-publish-manifest/v1",
                        "status": "passed",
                        "target_sha": self.target.sha,
                        "tested_sha": self.target.sha,
                        "run_id": "run-1",
                        "missing_expected_files": [],
                        "fallback": False,
                    }
                )
            if path.endswith("/delivery-summary.txt"):
                return (
                    "status: 0\n"
                    "backend_stages_enabled: true\n"
                    "frontend_build_status: pass\n"
                    "frontend_smoke_status: pass\n"
                    "backend_rebuild_status: pass\n"
                    "backend_smoke_jit_status: pass\n"
                    "flaggems_status: disabled\n"
                    "compile_time_status: disabled\n"
                    "pass_profile_status: disabled\n"
                    "ir_serialization_status: disabled\n"
                    "target_sha: " + self.target.sha + "\n"
                    "run_id: run-1\n"
                )
            if path.endswith("/result.json"):
                return json.dumps(
                    {
                        "codex_ai_ci_status": "pass",
                        "codex_ai_ci_mode": "full",
                        "codex_ai_ci_verdict": "PASS",
                        "codex_ai_test_status": "passed",
                    }
                )
            if path.endswith("/codex-ai-ci-summary.txt"):
                return (
                    "status: pass\n"
                    "analysis_mode: full\n"
                    "report_verdict: WARNING\n"
                    "test_execution_status: stable_failure\n"
                    "constraint_status: warning\n"
                    "constraint_reason: 定向测试数量符合约束。\n"
                    "failure_reason: \n"
                )
            if path.endswith("/codex-ai-comment.md"):
                return (
                    "## Codex AI 代码审查\n\n"
                    "### 审查摘要\n\n发现一个问题。\n\n"
                    "### 贡献者目标与实现情况\n\n实现不完整。\n\n"
                    "### 需要处理的问题\n\n1. 示例问题。\n\n"
                    "### 验证情况\n\n定向测试稳定复现。\n\n"
                    "### 变更文件\n\n<details></details>\n"
                )
            if path.endswith("/codex-ai-report.json"):
                return json.dumps(
                    {
                        "findings": [
                            {
                                "id": "AI-001",
                                "file": "python/example.py",
                                "line": "17",
                            }
                        ]
                    }
                )
            if path.endswith("/codex-ai-report.md"):
                return "# Codex AI 自动审查报告\n"
            return None

        gitee_content.side_effect = content
        args = SimpleNamespace(
            gitee_owner="owner",
            gitee_repo="results",
            gitee_results_branch="local-ci-results",
            gitee_web_url="https://gitee.example/results",
        )
        result = bridge.read_local_ci_result(args, self.target, "token")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.codex_ai.execution_status, "pass")
        self.assertEqual(result.codex_ai.verdict, "WARNING")
        self.assertEqual(result.codex_ai.test_status, "stable_failure")
        self.assertEqual(result.codex_ai.constraint_status, "warning")
        self.assertTrue(result.backend_stages_enabled)
        self.assertEqual(result.execution_mode, "full")
        self.assertIn("/blob/local-ci-results/", result.codex_ai.report_url)
        self.assertTrue(result.codex_ai.report_url.endswith("codex-ai-report.md"))
        self.assertEqual(
            result.codex_ai.finding_locations,
            (bridge.FindingLocation("AI-001", "python/example.py", "17"),),
        )
        self.assertIsNotNone(result.publish_manifest)
        assert result.publish_manifest is not None
        self.assertEqual(result.publish_manifest.target_sha, self.target.sha)

    def test_read_result_only_links_existing_full_report(self) -> None:
        cases = (
            ("# Codex AI 自动审查报告\n", (), True),
            (None, (), False),
            ("# Codex AI 自动审查报告\n", ("codex-ai-report.md",), False),
        )
        for report_markdown, missing_files, expected_url in cases:
            with self.subTest(
                report_exists=report_markdown is not None,
                missing_files=missing_files,
            ):

                def content(*_args: object) -> str | None:
                    path = str(_args[2])
                    if path.endswith("latest.txt"):
                        return "report-run\n"
                    if path.endswith("publish-manifest.json"):
                        return json.dumps(
                            {
                                "schema": "triton-anchor-local-ci-publish-manifest/v1",
                                "status": "passed",
                                "target_sha": self.target.sha,
                                "tested_sha": self.target.sha,
                                "run_id": "report-run",
                                "missing_expected_files": list(missing_files),
                                "fallback": False,
                            }
                        )
                    if path.endswith("delivery-summary.txt"):
                        return (
                            "status: 0\n"
                            "backend_stages_enabled: true\n"
                            f"target_sha: {self.target.sha}\n"
                            "run_id: report-run\n"
                            "frontend_build_status: pass\n"
                            "frontend_smoke_status: pass\n"
                            "backend_rebuild_status: pass\n"
                            "backend_smoke_jit_status: pass\n"
                        )
                    if path.endswith("codex-ai-ci-summary.txt"):
                        return "status: pass\nreport_verdict: PASS\n"
                    if path.endswith("codex-ai-comment.md"):
                        return "## Codex AI 自动审查\n"
                    if path.endswith("codex-ai-report.md"):
                        return report_markdown
                    return None

                args = SimpleNamespace(
                    gitee_owner="owner",
                    gitee_repo="results",
                    gitee_results_branch="local-ci-results",
                    gitee_web_url="https://gitee.example/results",
                )
                with mock.patch.object(
                    bridge, "gitee_content", side_effect=content
                ):
                    result = bridge.read_local_ci_result(args, self.target, "token")

                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(bool(result.codex_ai.report_url), expected_url)

    def test_required_stages_follow_trusted_backend_profile(self) -> None:
        cases = (
            ("true", "pass", "pass", "pass", "pass", 0, "pass", True),
            ("true", "pass", "pass", "pass", "not_run", 1, "pass", True),
            ("false", "pass", "pass", "skipped", "skipped", 0, "skipped", False),
            ("false", "pass", "pass", "fail", "skipped", 1, "fail", False),
            ("false", "skipped", "pass", "skipped", "skipped", 1, "skipped", False),
            ("true", "pass", "pass", "skipped", "skipped", 1, "fail", True),
            (None, "pass", "pass", "skipped", "skipped", 1, "fail", True),
            ("invalid", "pass", "pass", "pass", "pass", 1, "pass", True),
        )
        for (
            profile_value,
            frontend_build,
            frontend_smoke,
            backend_rebuild,
            backend_smoke,
            expected_exit,
            expected_backend_status,
            expected_backend_enabled,
        ) in cases:
            with self.subTest(profile_value=profile_value, expected_exit=expected_exit):
                profile_line = (
                    f"backend_stages_enabled: {profile_value}\n"
                    if profile_value is not None
                    else ""
                )
                dependent_status = (
                    "skipped" if profile_value == "false" else "disabled"
                )

                def content(*_args: object) -> str | None:
                    path = str(_args[2])
                    if path.endswith("latest.txt"):
                        return "profile-run\n"
                    if path.endswith("delivery-summary.txt"):
                        return (
                            "status: 0\n"
                            f"target_sha: {self.target.sha}\n"
                            "run_id: profile-run\n"
                            f"{profile_line}"
                            f"frontend_build_status: {frontend_build}\n"
                            f"frontend_smoke_status: {frontend_smoke}\n"
                            f"backend_rebuild_status: {backend_rebuild}\n"
                            f"backend_smoke_jit_status: {backend_smoke}\n"
                            f"flaggems_status: {dependent_status}\n"
                            f"compile_time_status: {dependent_status}\n"
                            f"pass_profile_status: {dependent_status}\n"
                            f"ir_serialization_status: {dependent_status}\n"
                        )
                    return None

                args = SimpleNamespace(
                    gitee_owner="owner",
                    gitee_repo="results",
                    gitee_results_branch="local-ci-results",
                    gitee_web_url="https://gitee.example/results",
                )
                with mock.patch.object(
                    bridge, "gitee_content", side_effect=content
                ):
                    result = bridge.read_local_ci_result(args, self.target, "token")

                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.exit_code, expected_exit)
                self.assertEqual(
                    result.stage_statuses["backend_rebuild"],
                    expected_backend_status,
                )
                self.assertEqual(
                    result.backend_stages_enabled,
                    expected_backend_enabled,
                )

    @mock.patch.object(bridge, "gitee_content")
    def test_codex_only_pr_accepts_explicitly_skipped_stages(
        self,
        gitee_content: mock.Mock,
    ) -> None:
        def content(*_args: object) -> str | None:
            path = str(_args[2])
            if path.endswith("latest.txt"):
                return "run-3\n"
            if path.endswith("delivery-summary.txt"):
                return (
                    "status: 0\n"
                    "execution_mode: codex_only\n"
                    f"target_sha: {self.target.sha}\n"
                    "run_id: run-3\n"
                    "frontend_build_status: skipped\n"
                    "frontend_smoke_status: skipped\n"
                    "backend_rebuild_status: skipped\n"
                    "backend_smoke_jit_status: skipped\n"
                    "flaggems_status: skipped\n"
                    "compile_time_status: skipped\n"
                    "pass_profile_status: skipped\n"
                    "ir_serialization_status: skipped\n"
                )
            return None

        gitee_content.side_effect = content
        args = SimpleNamespace(
            gitee_owner="owner",
            gitee_repo="results",
            gitee_results_branch="local-ci-results",
            gitee_web_url="https://gitee.example/results",
        )
        result = bridge.read_local_ci_result(args, self.target, "token")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.exit_code, 0)

        push_target = bridge.Target(
            "feature", "ci/push/feature", self.target.sha, "feature"
        )
        push_result = bridge.read_local_ci_result(args, push_target, "token")
        self.assertIsNotNone(push_result)
        assert push_result is not None
        self.assertEqual(push_result.exit_code, 1)
        self.assertEqual(push_result.stage_statuses["frontend_build"], "fail")


if __name__ == "__main__":
    unittest.main()
