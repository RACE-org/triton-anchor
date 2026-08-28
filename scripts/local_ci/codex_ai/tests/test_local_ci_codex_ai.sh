#!/usr/bin/env bash
set -euo pipefail

codex_ai_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${codex_ai_dir}/../../.." && pwd)"
renderer="${repo_root}/scripts/local_ci/codex_ai/render_codex_ai_report.py"
schema="${repo_root}/scripts/local_ci/codex_ai/codex_ai_report.schema.json"
analysis_schema="${repo_root}/scripts/local_ci/codex_ai/codex_ai_analysis.schema.json"
test_root="$(mktemp -d /tmp/local-ci-codex-report-test.XXXXXX)"
trap 'rm -rf -- "${test_root}"' EXIT

python3 - "${schema}" "${analysis_schema}" <<'PY'
import json
import sys
from pathlib import Path

schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
analysis_schema = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
unsupported_keywords = {
    "allOf",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "not",
    "oneOf",
    "then",
    "else",
    "uniqueItems",
}


def validate(node, location="$"):
    if isinstance(node, dict):
        unsupported = unsupported_keywords.intersection(node)
        assert not unsupported, f"{location}: unsupported keywords: {sorted(unsupported)}"
        if node.get("type") == "object":
            properties = node.get("properties", {})
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(properties)
        for key, value in node.items():
            validate(value, f"{location}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            validate(value, f"{location}[{index}]")



validate(schema)
validate(analysis_schema)
assert "verdict" not in analysis_schema["properties"]
assert "completion_marker" not in analysis_schema["properties"]
assert "test_execution" not in analysis_schema["properties"]
assert set(analysis_schema["properties"]) == {
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
assert set(analysis_schema["properties"]["changed_files"]["items"]["properties"]) == {
    "file_id", "summary", "impact", "validation_strategy"
}
assert set(analysis_schema["properties"]["behavior_coverage"]["properties"]) == {
    "normal", "boundary", "error", "compatibility", "integration"
}
assert set(analysis_schema["properties"]["test_assessment"]["properties"]) == {
    "evidence_level", "summary", "commands"
}
assert analysis_schema["properties"]["change_request_assessment"]["properties"]["evidence"]["maxItems"] == 8
assert analysis_schema["properties"]["test_assessment"]["properties"]["summary"]["maxItems"] == 8
assert analysis_schema["properties"]["test_assessment"]["properties"]["commands"]["items"]["properties"]["purpose"]["maxLength"] == 120
assert "pattern" not in analysis_schema["properties"]["summary"]
PY

valid_json="${test_root}/valid.json"
report_md="${test_root}/report.md"
comment_md="${test_root}/comment.md"
manifest_json="${test_root}/changed-files.json"
cat > "${manifest_json}" <<'JSON'
[
  {"path": "python/example.py", "change_type": "modified"}
]
JSON
cat > "${valid_json}" <<'JSON'
{
  "verdict": "WARNING",
  "summary": "发现一个可能引起行为回归的中风险问题。",
  "merge_recommendation": "建议修复缓存版本校验问题并重新运行定向测试后合入。",
  "change_request_assessment": {
    "status": "partially_implemented",
    "contributor_goal": "贡献者希望修复缓存命中后的状态读取逻辑。",
    "expected_behavior": "版本变化后旧缓存应失效，调用方应读取当前状态。",
    "implementation_summary": "正常缓存命中已调整，但版本失配路径仍未完整处理。",
    "evidence": [
      "代码差异缺少版本校验。",
      "RUN-001 复现了过期状态。"
    ]
  },
  "changed_files": [
    {
      "path": "python/example.py",
      "change_type": "modified",
      "summary": "调整了缓存命中后的状态读取逻辑。",
      "impact": "可能影响版本变化后的缓存一致性。",
      "validation_strategy": "通过版本变化后的缓存失效用例验证。"
    }
  ],
  "behavior_coverage": {
    "normal": {
      "scope": "缓存版本一致时的正常命中路径。",
      "strategy": "执行现有缓存命中测试。",
      "result": "正常命中路径验证通过。"
    },
    "boundary": {
      "scope": "版本号刚好发生变化的边界路径。",
      "strategy": "生成版本变化后的缓存失效用例。",
      "result": "发现缓存没有及时失效。"
    },
    "error": {
      "scope": "缓存内容不可用时的错误路径。",
      "strategy": "检查异常处理分支和已有测试。",
      "result": "未发现新的错误处理问题。"
    },
    "compatibility": {
      "scope": "旧版本缓存记录的兼容读取路径。",
      "strategy": "检查版本字段缺失时的处理逻辑。",
      "result": "旧记录仍可按既有规则处理。"
    },
    "integration": {
      "scope": "缓存模块与调用方之间的集成路径。",
      "strategy": "执行调用方定向回归测试。",
      "result": "除版本失配路径外未发现集成回归。"
    }
  },
  "findings": [
    {
      "id": "AI-001",
      "severity": "MEDIUM",
      "category": "regression",
      "file": "python/example.py",
      "line": "17",
      "code_role": "该条件决定缓存命中后是否继续复用旧状态。",
      "title": "缓存命中后返回了过期状态",
      "evidence": "新分支直接复用缓存值，但没有核对当前版本号。",
      "impact": "调用方可能读取到上一次任务遗留的状态。",
      "fix_direction": "在复用缓存前校验版本号，并为失配路径补充测试。"
    }
  ],
  "unlocated_findings": [],
  "suggested_tests": [
    {
      "id": "TEST-001",
      "priority": "MEDIUM",
      "target": "python/tests/test_example.py",
      "description": "增加版本变化后缓存必须失效的回归测试。"
    }
  ],
  "residual_risks": [
    "本次只执行了定向测试，尚未覆盖并发更新场景。"
  ],
  "test_execution": {
    "evidence_level": "sufficient",
    "status": "passed",
    "summary": [
      "RUN-001 执行的缓存失效测试已经通过。",
      "验证覆盖了版本变化后的缓存失效路径。"
    ],
    "generated_test_files": [
      "python/tests/test_generated_cache.py"
    ],
    "commands": [
      {
        "id": "RUN-001",
        "command": "python3 -m pytest python/tests/test_generated_cache.py",
        "role": "validation",
        "purpose": "缓存版本失配定向测试",
        "exit_code": 0,
        "duration_seconds": 0.2,
        "status": "passed",
        "evidence": "定向测试共执行一个用例并通过。"
      }
    ]
  },
  "completion_marker": "CODEX_AI_CI_COMPLETE"
}
JSON

verdict="$(python3 "${renderer}" \
  --input "${valid_json}" \
  --output "${report_md}" \
  --comment-output "${comment_md}" \
  --branch "ci/push/ai-ci" \
  --base-sha "$(printf 'a%.0s' {1..40})" \
  --target-sha "$(printf 'b%.0s' {1..40})" \
  --changed-file-count 1 \
  --changed-files-manifest "${manifest_json}" \
  --constraint-status warning \
  --constraint-reason "测试命令数量超过轻量约束。")"
[[ "${verdict}" == "WARNING" ]]

grep -Fq "# Codex AI 自动审查报告" "${report_md}"
grep -Fq "## 元数据" "${report_md}"
grep -Fq "## 结论" "${report_md}"
grep -Fq "**警告**" "${report_md}"
grep -Fq 'triton-anchor-codex-ai-report/v3' "${report_md}"
grep -Fq "## 贡献者目标与实现情况" "${report_md}"
grep -Fq "判断：部分实现" "${report_md}"
grep -Fq "## 合入建议" "${report_md}"
grep -Fq "## 具体文件变更" "${report_md}"
grep -Fq "## 行为覆盖" "${report_md}"
grep -Fq "## 关键问题" "${report_md}"
grep -Fq "缓存命中后返回了过期状态" "${report_md}"
grep -Fq "这段代码负责" "${report_md}"
grep -Fq "## 建议测试" "${report_md}"
grep -Fq "## 测试执行" "${report_md}"
grep -Fq -- "- Codex 对验证证据的判断：证据充分" "${report_md}"
grep -Fq -- "- Runner 事实校验：正式验证目标均已完成" "${report_md}"
grep -Fq "## 测试执行约束" "${report_md}"
grep -Fq "状态：警告" "${report_md}"
grep -Fq "测试命令数量超过轻量约束" "${report_md}"
grep -Fq "## 剩余风险" "${report_md}"
grep -Fq "## Codex AI 自动审查" "${comment_md}"
grep -Fq "仅供参考且不阻塞合入" "${comment_md}"
grep -Fq "### 审查摘要" "${comment_md}"
grep -Fq "本地确定性 CI 检查：" "${comment_md}"
grep -Fq "Codex AI 审查结论：" "${comment_md}"
grep -Fq "Codex AI 审查结论：**需关注（非阻塞）**" "${comment_md}"
if grep -Fq "Codex 执行状态" "${comment_md}"; then
  echo "PR 评论不应显示 Codex 执行状态" >&2
  exit 1
fi
grep -Fq "### 贡献者目标与实现情况" "${comment_md}"
grep -Fq "贡献者目标：贡献者希望修复缓存命中后的状态读取逻辑" "${comment_md}"
grep -Fq -- "- 判断依据：" "${comment_md}"
grep -Fq -- "  - 代码差异缺少版本校验。" "${comment_md}"
grep -Fq -- "  - 缓存版本失配定向测试复现了过期状态。" "${comment_md}"
grep -Fq "### 验证情况" "${comment_md}"
grep -Fq -- "- 验证内容与结果：" "${comment_md}"
if grep -Fq "条执行记录" "${comment_md}"; then
  echo "PR 评论不应根据全部命令生成固定统计文案" >&2
  exit 1
fi
grep -Fq -- "- 限制与未覆盖：" "${comment_md}"
grep -Fq -- "  - 验证覆盖了版本变化后的缓存失效路径。" "${comment_md}"
if grep -Eq -- "^- (验证依据|执行内容|执行结果)：" "${comment_md}"; then
  echo "PR 评论仍使用旧的验证分组" >&2
  exit 1
fi
if grep -Eq "Codex 对验证证据的判断|Runner 事实校验|Codex 说明：|Runner 校验：" "${comment_md}"; then
  echo "PR 评论仍暴露内部验证状态或来源标签" >&2
  exit 1
fi
grep -Fq "### 需要处理的问题" "${comment_md}"
grep -Fq "### 变更文件" "${comment_md}"
grep -Fq "这段代码负责：该条件决定缓存命中后是否继续复用旧状态" "${comment_md}"
grep -Fq -- "- 核心证据：新分支直接复用缓存值，但没有核对当前版本号。" "${comment_md}"
grep -Fq "<details>" "${comment_md}"
if grep -Eq '(^|[^A-Za-z0-9_])(AI|TEST|RUN)-[0-9]{3,}([^A-Za-z0-9_]|$)' "${comment_md}"; then
  echo "PR 评论不应包含内部结构化编号" >&2
  exit 1
fi
if grep -Eq '^## (Metadata|Verdict|Summary|Findings|Suggested Tests|Test Execution|Residual Risks|Execution)$' "${report_md}"; then
  echo "报告仍包含英文模板标题" >&2
  exit 1
fi
python3 - "${report_md}" "${comment_md}" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).read_text(encoding="utf-8")
Path(sys.argv[2]).read_text(encoding="utf-8")
PY

invalid_verdict="${test_root}/invalid-verdict.json"
sed 's/"verdict": "WARNING"/"verdict": "PASS"/' "${valid_json}" > "${invalid_verdict}"
if python3 "${renderer}" \
  --input "${invalid_verdict}" \
  --output "${test_root}/invalid-verdict.md" \
  --comment-output "${test_root}/invalid-verdict-comment.md" \
  --branch test --base-sha a --target-sha b --changed-file-count 1 \
  --changed-files-manifest "${manifest_json}" \
  >/dev/null 2>&1; then
  echo "渲染器接受了与 findings 不一致的 verdict" >&2
  exit 1
fi

english_report="${test_root}/english.json"
sed 's/发现一个可能引起行为回归的中风险问题。/English-only summary./' \
  "${valid_json}" > "${english_report}"
if python3 "${renderer}" \
  --input "${english_report}" \
  --output "${test_root}/english.md" \
  --comment-output "${test_root}/english-comment.md" \
  --branch test --base-sha a --target-sha b --changed-file-count 1 \
  --changed-files-manifest "${manifest_json}" \
  >/dev/null 2>&1; then
  echo "渲染器接受了英文说明性字段" >&2
  exit 1
fi

wrong_file_report="${test_root}/wrong-file.json"
sed 's#"path": "python/example.py"#"path": "python/other.py"#' \
  "${valid_json}" > "${wrong_file_report}"
if python3 "${renderer}" \
  --input "${wrong_file_report}" \
  --output "${test_root}/wrong-file.md" \
  --comment-output "${test_root}/wrong-file-comment.md" \
  --branch test --base-sha a --target-sha b --changed-file-count 1 \
  --changed-files-manifest "${manifest_json}" \
  >/dev/null 2>&1; then
  echo "渲染器接受了与 Git diff 清单不一致的 changed_files" >&2
  exit 1
fi

duplicate_file_report="${test_root}/duplicate-file.json"
python3 - "${valid_json}" "${duplicate_file_report}" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
document["changed_files"].append(dict(document["changed_files"][0]))
Path(sys.argv[2]).write_text(
    json.dumps(document, ensure_ascii=False), encoding="utf-8"
)
PY
if python3 "${renderer}" \
  --input "${duplicate_file_report}" \
  --output "${test_root}/duplicate-file.md" \
  --comment-output "${test_root}/duplicate-file-comment.md" \
  --branch test --base-sha a --target-sha b --changed-file-count 1 \
  --changed-files-manifest "${manifest_json}" \
  >/dev/null 2>&1; then
  echo "渲染器接受了重复的 changed_files 条目" >&2
  exit 1
fi

missing_behavior_report="${test_root}/missing-behavior.json"
python3 - "${valid_json}" "${missing_behavior_report}" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
del document["behavior_coverage"]["integration"]
Path(sys.argv[2]).write_text(
    json.dumps(document, ensure_ascii=False), encoding="utf-8"
)
PY
if python3 "${renderer}" \
  --input "${missing_behavior_report}" \
  --output "${test_root}/missing-behavior.md" \
  --comment-output "${test_root}/missing-behavior-comment.md" \
  --branch test --base-sha a --target-sha b --changed-file-count 1 \
  --changed-files-manifest "${manifest_json}" \
  >/dev/null 2>&1; then
  echo "渲染器接受了缺少集成路径的 behavior_coverage" >&2
  exit 1
fi

invalid_assessment="${test_root}/invalid-assessment.json"
sed 's/"status": "partially_implemented"/"status": "unknown"/' \
  "${valid_json}" > "${invalid_assessment}"
if python3 "${renderer}" \
  --input "${invalid_assessment}" \
  --output "${test_root}/invalid-assessment.md" \
  --comment-output "${test_root}/invalid-assessment-comment.md" \
  --branch test --base-sha a --target-sha b --changed-file-count 1 \
  --changed-files-manifest "${manifest_json}" \
  >/dev/null 2>&1; then
  echo "渲染器接受了未知的贡献目标实现状态" >&2
  exit 1
fi

inconsistent_execution_report="${test_root}/inconsistent-execution.json"
python3 - "${valid_json}" "${inconsistent_execution_report}" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
document["test_execution"]["commands"][0]["exit_code"] = 1
Path(sys.argv[2]).write_text(
    json.dumps(document, ensure_ascii=False), encoding="utf-8"
)
PY
if python3 "${renderer}" \
  --input "${inconsistent_execution_report}" \
  --output "${test_root}/inconsistent-execution.md" \
  --comment-output "${test_root}/inconsistent-execution-comment.md" \
  --branch test --base-sha a --target-sha b --changed-file-count 1 \
  --changed-files-manifest "${manifest_json}" \
  >/dev/null 2>&1; then
  echo "渲染器接受了 passed 状态与非零退出码矛盾的命令" >&2
  exit 1
fi

echo "Codex AI 中文报告格式与内容校验：通过"
