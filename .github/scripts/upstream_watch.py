#!/usr/bin/env python3
"""Generate upstream Triton change reports for triton-anchor."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ImpactRule:
    prefix: str
    level: str
    handling: str


IMPACT_RULES = (
    ImpactRule(
        "include/triton/Dialect/Triton/IR/",
        "高",
        "Op 定义变更 -> 可能影响所有后端",
    ),
    ImpactRule(
        "lib/Dialect/Triton/Transforms/",
        "高",
        "TTIR Pass 变更 -> 影响 7-pass pipeline",
    ),
    ImpactRule(
        "python/src/passes.cc",
        "中",
        "Pybind 接口变更 -> 需同步 anchor_passes.cc",
    ),
    ImpactRule(
        "python/triton/compiler/",
        "中",
        "编译器入口变更 -> 需评估",
    ),
    ImpactRule(
        "python/triton/language/",
        "中",
        "DSL 语义变更 -> FlagGems 兼容性",
    ),
    ImpactRule(
        "python/triton/runtime/",
        "低",
        "运行时变更 -> 通常不影响前端",
    ),
    ImpactRule(
        "third_party/",
        "低",
        "NVIDIA/AMD 后端 -> 与我们无关",
    ),
)

LEVEL_ORDER = {"高": 0, "中": 1, "低": 2, "未分类": 3}


def run_git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def ensure_repo(repo_url: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        run_git(["remote", "set-url", "origin", repo_url], repo_dir)
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(repo_dir)],
        check=True,
        text=True,
    )


def classify_path(path: str) -> dict[str, str]:
    for rule in IMPACT_RULES:
        if path == rule.prefix.rstrip("/") or path.startswith(rule.prefix):
            return {
                "level": rule.level,
                "category": rule.prefix,
                "handling": rule.handling,
            }
    return {
        "level": "未分类",
        "category": "未命中规则",
        "handling": "需人工判断是否影响 triton-anchor",
    }


def fetch_ref(repo_dir: Path, ref: str, depth: int) -> str:
    run_git(
        [
            "fetch",
            "--no-tags",
            f"--depth={depth}",
            "origin",
            f"+{ref}:refs/upstream-watch/{ref}",
        ],
        repo_dir,
    )
    return run_git(["rev-parse", f"refs/upstream-watch/{ref}"], repo_dir)


def commits_since(repo_dir: Path, ref: str, since: str, max_commits: int) -> list[dict[str, str]]:
    refname = f"refs/upstream-watch/{ref}"
    log_format = "%H%x09%cs%x09%an%x09%s"
    output = run_git(
        [
            "log",
            refname,
            f"--since={since}",
            f"--max-count={max_commits}",
            f"--format={log_format}",
        ],
        repo_dir,
    )
    commits: list[dict[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        sha, date, author, subject = line.split("\t", 3)
        commits.append(
            {
                "sha": sha,
                "short_sha": sha[:12],
                "date": date,
                "author": author,
                "subject": subject,
            }
        )
    return commits


def changed_files(repo_dir: Path, commit: str) -> list[dict[str, str]]:
    output = run_git(
        ["diff-tree", "--no-commit-id", "--name-status", "-r", "--find-renames", commit],
        repo_dir,
    )
    files: list[dict[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        path = parts[-1]
        impact = classify_path(path)
        files.append({"status": status, "path": path, **impact})
    return files


def summarize_branch(
    repo_dir: Path, ref: str, since: str, max_commits: int, fetch_depth: int
) -> dict[str, object]:
    head = fetch_ref(repo_dir, ref, fetch_depth)
    commits = commits_since(repo_dir, ref, since, max_commits)

    level_counts = {level: 0 for level in LEVEL_ORDER}
    file_rows: list[dict[str, str]] = []
    for commit in commits:
        files = changed_files(repo_dir, commit["sha"])
        commit["files"] = files
        for file_change in files:
            level_counts[file_change["level"]] += 1
            file_rows.append({"commit": commit["short_sha"], **file_change})

    file_rows.sort(key=lambda row: (LEVEL_ORDER[row["level"]], row["path"], row["commit"]))
    return {
        "ref": ref,
        "head": head,
        "head_short": head[:12],
        "commit_count": len(commits),
        "level_counts": level_counts,
        "commits": commits,
        "files": file_rows,
    }


def markdown_report(data: dict[str, object]) -> str:
    generated_at = data["generated_at"]
    refs = data["refs"]
    since = data["since"]
    repo_url = data["repo_url"]

    lines = [
        "# Triton 上游变更监控报告",
        "",
        f"- 上游仓库: `{repo_url}`",
        f"- 检查范围: `{since}` 至 `{generated_at}`",
        f"- 监控分支: {', '.join(f'`{ref}`' for ref in refs)}",
        "",
        "## 影响分类规则",
        "",
        "| 变更位置 | 影响级别 | 处理方式 |",
        "| --- | --- | --- |",
    ]
    for rule in IMPACT_RULES:
        lines.append(f"| `{rule.prefix}` | {rule.level} | {rule.handling} |")
    lines.extend(
        [
            "| 其他路径 | 未分类 | 需人工判断是否影响 triton-anchor |",
            "",
            "## 分支概览",
            "",
            "| 上游分支 | HEAD | Commit 数 | 高 | 中 | 低 | 未分类 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for branch in data["branches"]:
        counts = branch["level_counts"]
        lines.append(
            "| `{ref}` | `{head}` | {commits} | {high} | {medium} | {low} | {unknown} |".format(
                ref=branch["ref"],
                head=branch["head_short"],
                commits=branch["commit_count"],
                high=counts["高"],
                medium=counts["中"],
                low=counts["低"],
                unknown=counts["未分类"],
            )
        )

    for branch in data["branches"]:
        lines.extend(["", f"## `{branch['ref']}`", ""])
        if branch["commit_count"] == 0:
            lines.append("本次检查窗口内没有新增 commit。")
            continue

        lines.extend(
            [
                "### 高/中影响文件",
                "",
                "| 影响级别 | Commit | 状态 | 文件 | 处理方式 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        important_files = [
            row for row in branch["files"] if row["level"] in {"高", "中"}
        ]
        if important_files:
            for row in important_files:
                lines.append(
                    f"| {row['level']} | `{row['commit']}` | `{row['status']}` | "
                    f"`{row['path']}` | {md_cell(row['handling'])} |"
                )
        else:
            lines.append("| - | - | - | - | 未发现高/中影响文件 |")

        lines.extend(
            [
                "",
                "### Commit 列表",
                "",
                "| 日期 | Commit | 作者 | 标题 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for commit in branch["commits"]:
            lines.append(
                f"| {commit['date']} | `{commit['short_sha']}` | "
                f"{md_cell(commit['author'])} | {md_cell(commit['subject'])} |"
            )

        low_or_unknown = [
            row for row in branch["files"] if row["level"] in {"低", "未分类"}
        ]
        if low_or_unknown:
            lines.extend(
                [
                    "",
                    "<details>",
                    "<summary>低影响/未分类文件</summary>",
                    "",
                    "| 影响级别 | Commit | 状态 | 文件 | 处理方式 |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for row in low_or_unknown:
                lines.append(
                    f"| {row['level']} | `{row['commit']}` | `{row['status']}` | "
                    f"`{row['path']}` | {md_cell(row['handling'])} |"
                )
            lines.extend(["", "</details>"])

    lines.append("")
    return "\n".join(lines)


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-url",
        default="https://github.com/triton-lang/triton.git",
        help="Upstream Triton repository URL.",
    )
    parser.add_argument(
        "--refs",
        nargs="+",
        default=[
            "release/3.0.x",
            "release/3.1.x",
            "release/3.2.x",
            "release/3.3.x",
            "release/3.4.x",
            "release/3.5.x",
            "release/3.6.x",
        ],
        help="Upstream refs to monitor.",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Git log --since value. Overrides --since-days.",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=7,
        help="Default monitoring window in days.",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=200,
        help="Maximum commits to report per ref.",
    )
    parser.add_argument(
        "--fetch-depth",
        type=int,
        default=500,
        help="Fetch depth for each monitored ref.",
    )
    parser.add_argument(
        "--work-dir",
        default=os.environ.get("RUNNER_TEMP", ".upstream-watch"),
        help="Directory used for the upstream clone.",
    )
    parser.add_argument(
        "--report",
        default="upstream-watch-report.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--json",
        default="upstream-watch-report.json",
        help="JSON report output path.",
    )
    parser.add_argument(
        "--fresh-clone",
        action="store_true",
        help="Remove the cached upstream clone before fetching.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    since = args.since or f"{args.since_days} days ago"
    work_dir = Path(args.work_dir).resolve()
    repo_dir = work_dir / "triton-upstream"

    if args.fresh_clone and repo_dir.exists():
        shutil.rmtree(repo_dir)

    ensure_repo(args.repo_url, repo_dir)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    data: dict[str, object] = {
        "repo_url": args.repo_url,
        "refs": args.refs,
        "since": since,
        "generated_at": generated_at,
        "branches": [],
    }

    for ref in args.refs:
        try:
            data["branches"].append(
                summarize_branch(repo_dir, ref, since, args.max_commits, args.fetch_depth)
            )
        except subprocess.CalledProcessError as exc:
            print(f"error: failed to inspect {ref}: {exc.stderr}", file=sys.stderr)
            return exc.returncode

    report_path = Path(args.report)
    json_path = Path(args.json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(markdown_report(data), encoding="utf-8")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
