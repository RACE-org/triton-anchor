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


@dataclass(frozen=True)
class ImpactProfile:
    name: str
    refs: tuple[str, ...]
    rules: tuple[ImpactRule, ...]
    unused_paths: tuple[str, ...] = ()


COMMON_RULES = (
    # Rules are first-match-wins. Keep specific paths before the generic
    # vendored-source fallbacks near the end.
    ImpactRule(
        "python/triton/backends/__init__.py",
        "高",
        "本地定制后端发现 -> 必须核对 entry_points 插件机制",
    ),
    ImpactRule(
        "python/triton/__init__.py",
        "高",
        "本地裁剪入口 -> 必须重放 ops 导出修改",
    ),
    ImpactRule(
        "cmake/llvm-hash.txt",
        "高",
        "LLVM 基线变化 -> 可能导致 C++/MLIR API 不兼容",
    ),
    ImpactRule(
        "include/triton/Dialect/Triton/",
        "高",
        "TTIR 方言定义和 Pass 接口 -> 影响 AST、pipeline 和所有后端",
    ),
    ImpactRule(
        "lib/Dialect/Triton/",
        "高",
        "TTIR 方言和 Pass 实现 -> 影响语义、7-pass pipeline 和所有后端",
    ),
    ImpactRule(
        "include/triton/Analysis/",
        "高",
        "核心分析接口 -> 影响 TTIR 优化及 Linalg/TritonGPU 转换",
    ),
    ImpactRule(
        "lib/Analysis/",
        "高",
        "核心分析实现 -> 影响 AxisInfo、内存分配和转换结果",
    ),
    ImpactRule(
        "include/triton/Dialect/TritonGPU/",
        "高",
        "TritonGPU IR 和 Pass 接口 -> 影响 AnchorIR GPU Track",
    ),
    ImpactRule(
        "lib/Dialect/TritonGPU/",
        "高",
        "TritonGPU IR 和 Pass 实现 -> 影响布局、类型和 GPU Track pipeline",
    ),
    ImpactRule(
        "python/src/ir.cc",
        "高",
        "Pybind IR 核心接口 -> 影响 AST 到 TTIR 生成",
    ),
    ImpactRule(
        "python/src/passes.cc",
        "中",
        "Pybind Pass 接口 -> 需同步 anchor_passes.cc",
    ),
    ImpactRule(
        "python/src/passes.h",
        "中",
        "Pybind Pass 声明 -> 需核对 passes.cc 和 anchor 绑定",
    ),
    ImpactRule(
        "include/triton/Dialect/TritonNvidiaGPU/",
        "中",
        "直接编译的 NVIDIA 核心方言 -> 需验证兼容构建和导出 target",
    ),
    ImpactRule(
        "lib/Dialect/TritonNvidiaGPU/",
        "中",
        "直接编译的 NVIDIA 核心实现 -> 需验证兼容构建和链接",
    ),
    ImpactRule(
        "python/triton/language/extra/",
        "低",
        "CUDA/HIP 等扩展 DSL -> 通常不影响通用 DSL",
    ),
    ImpactRule(
        "python/triton/compiler/",
        "中",
        "AST/编译器入口 -> 需评估 TTIR 生成和后端调用链",
    ),
    ImpactRule(
        "python/triton/language/",
        "中",
        "通用 DSL 语义 -> 需验证 FlagGems 兼容性",
    ),
    ImpactRule(
        "python/triton/backends/",
        "中",
        "后端插件接口 -> 影响独立 backend 包注册和发现",
    ),
    ImpactRule(
        "python/triton/runtime/",
        "中",
        "JIT/cache/driver 接口 -> 影响运行时和后端插件兼容性",
    ),
    ImpactRule(
        "python/src/",
        "中",
        "C++ Pybind 绑定 -> 直接影响 libtriton.so 构建和 API",
    ),
    ImpactRule(
        "include/triton/Tools/",
        "中",
        "公共工具接口 -> LinearLayout 等组件可能影响 GPU Track",
    ),
    ImpactRule(
        "lib/Tools/",
        "中",
        "公共工具实现 -> LinearLayout 等组件可能影响 GPU Track",
    ),
    ImpactRule(
        "third_party/f2reduce/",
        "中",
        "anchor 直接编译的依赖 -> 需验证构建兼容性",
    ),
    ImpactRule(
        "cmake/",
        "中",
        "Vendored 构建配置 -> 需核对 LLVM 查找和编译选项",
    ),
    ImpactRule(
        "python/triton/tools/",
        "低",
        "辅助命令行工具 -> 通常不影响核心前端 pipeline",
    ),
    ImpactRule(
        "python/triton/testing.py",
        "低",
        "测试辅助 API -> 仅需评估测试兼容性",
    ),
    ImpactRule(
        "include/",
        "中",
        "Vendored C++ 头文件新增或调整 -> 需判断构建和接口影响",
    ),
    ImpactRule(
        "lib/",
        "中",
        "Vendored C++ 实现新增或调整 -> 需判断构建和 pipeline 影响",
    ),
    ImpactRule(
        "python/",
        "中",
        "Vendored Python/C++ 前端新增或调整 -> 需判断同步范围",
    ),
)

UNUSED_RULE = ImpactRule(
    "非同步范围",
    "低",
    "anchor 未使用或未 vendoring -> 无需逐项展开，升级时仅检查是否出现新依赖",
)

UNUSED_PATHS = (
    ".github/",
    "benchmarks/",
    "docs/",
    "test/",
    "third_party/",
    "python/benchmarks/",
    "python/examples/",
    "python/test/",
    "python/tutorials/",
    "python/triton/ops/",
    "CMakeLists.txt",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
    "setup.py",
    "python/MANIFEST.in",
    "python/pyproject.toml",
    "python/setup.py",
)

UNUSED_PATH_EXCEPTIONS = ("third_party/f2reduce/",)

PROFILE_30_33_RULES = (
    ImpactRule(
        "python/src/main.cc",
        "高",
        "3.0/3.3 anchor 本地定制入口 -> 必须重放 init_triton_anchor 修改",
    ),
    ImpactRule(
        "include/triton/Conversion/TritonToTritonGPU/",
        "高",
        "TTIR 到 TritonGPU 接口 -> 影响 AnchorIR GPU Track",
    ),
    ImpactRule(
        "lib/Conversion/TritonToTritonGPU/",
        "高",
        "TTIR 到 TritonGPU 转换 -> 影响 AnchorIR GPU Track",
    ),
    ImpactRule(
        "include/triton/Conversion/",
        "中",
        "3.0/3.3 anchor 使用的转换接口 -> 影响 GPU Track 或 LLVM 接口",
    ),
    ImpactRule(
        "lib/Conversion/",
        "中",
        "3.0/3.3 anchor 使用的转换实现 -> 影响 GPU Track 或 LLVM 接口",
    ),
    ImpactRule(
        "include/triton/Target/",
        "中",
        "3.0/3.3 anchor 使用的 LLVMIR Target 接口 -> 影响链接和代码生成",
    ),
    ImpactRule(
        "lib/Target/",
        "中",
        "3.0/3.3 anchor 使用的 LLVMIR Target 实现 -> 影响链接和代码生成",
    ),
)

PROFILE_30 = ImpactProfile(
    name="3.0 Profile",
    refs=("release/3.0.x", "release/3.1.x", "release/3.2.x"),
    rules=(
        ImpactRule(
            "lib/Conversion/TritonGPUToLLVM/CMakeLists.txt",
            "高",
            "3.0 本地构建补丁 -> 必须核对 NVGPUIR 依赖",
        ),
        *PROFILE_30_33_RULES,
    ),
)

PROFILE_33 = ImpactProfile(
    name="3.3 Profile",
    refs=("release/3.3.x", "release/3.4.x", "release/3.5.x"),
    rules=(
        *PROFILE_30_33_RULES,
        ImpactRule(
            "lib/Instrumentation/",
            "中",
            "3.3 Instrumentation -> 直接参与核心库构建",
        ),
    ),
)

PROFILE_36 = ImpactProfile(
    name="3.6 Profile",
    refs=("release/3.6.x",),
    rules=(
        ImpactRule(
            "include/triton/Dialect/Gluon/",
            "中",
            "3.6 Gluon 方言接口 -> 直接参与构建并影响实验前端",
        ),
        ImpactRule(
            "lib/Dialect/Gluon/",
            "中",
            "3.6 Gluon 方言实现 -> 直接参与构建并影响实验前端",
        ),
        ImpactRule(
            "python/src/gluon_ir.cc",
            "中",
            "3.6 Gluon Pybind 接口 -> 影响 libtriton.so API",
        ),
        ImpactRule(
            "python/triton/experimental/gluon/",
            "中",
            "3.6 Gluon Python 前端 -> 由 namespace package 打包",
        ),
        ImpactRule(
            "third_party/proton/Dialect/",
            "中",
            "3.6 Proton Dialect -> 由 CMake 直接编译",
        ),
    ),
    unused_paths=(
        "include/triton/Conversion/",
        "lib/Conversion/",
        "include/triton/Target/",
        "lib/Target/",
        "lib/Instrumentation/",
        "include/triton/Dialect/TritonInstrument/",
        "lib/Dialect/TritonInstrument/",
        "lib/Plugins/",
    ),
)

IMPACT_PROFILES = (PROFILE_30, PROFILE_33, PROFILE_36)

PROFILE_UNUSED_RULE = ImpactRule(
    "该版本非同步范围",
    "低",
    "官方上游可能存在，但该 anchor Profile 未同步或未使用 -> 无需逐项展开",
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


def path_matches(path: str, prefix: str) -> bool:
    return path == prefix.rstrip("/") or path.startswith(prefix)


def profile_for_ref(ref: str | None) -> ImpactProfile | None:
    if ref is None:
        return None
    for profile in IMPACT_PROFILES:
        if any(
            ref == candidate or ref.endswith(f"/{candidate}")
            for candidate in profile.refs
        ):
            return profile
    return None


def match_rules(path: str, rules: tuple[ImpactRule, ...]) -> dict[str, str] | None:
    for rule in rules:
        if path_matches(path, rule.prefix):
            return {
                "level": rule.level,
                "category": rule.prefix,
                "handling": rule.handling,
            }
    return None


def classify_path(path: str, ref: str | None = None) -> dict[str, str]:
    profile = profile_for_ref(ref)
    if profile is not None:
        impact = match_rules(path, profile.rules)
        if impact is not None:
            return impact
        if any(path_matches(path, prefix) for prefix in profile.unused_paths):
            return {
                "level": PROFILE_UNUSED_RULE.level,
                "category": f"{profile.name} 非同步范围",
                "handling": PROFILE_UNUSED_RULE.handling,
            }

    is_unused = any(path_matches(path, prefix) for prefix in UNUSED_PATHS)
    is_exception = any(
        path_matches(path, prefix) for prefix in UNUSED_PATH_EXCEPTIONS
    )
    if is_unused and not is_exception:
        return {
            "level": UNUSED_RULE.level,
            "category": UNUSED_RULE.prefix,
            "handling": UNUSED_RULE.handling,
        }

    impact = match_rules(path, COMMON_RULES)
    if impact is not None:
        return impact
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


def changed_files(repo_dir: Path, commit: str, ref: str) -> list[dict[str, str]]:
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
        impact = classify_path(path, ref)
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
        files = changed_files(repo_dir, commit["sha"], ref)
        commit["files"] = files
        for file_change in files:
            level_counts[file_change["level"]] += 1
            file_rows.append({"commit": commit["short_sha"], **file_change})

    file_rows.sort(key=lambda row: (LEVEL_ORDER[row["level"]], row["path"], row["commit"]))
    profile = profile_for_ref(ref)
    return {
        "ref": ref,
        "profile": profile.name if profile is not None else "公共规则",
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
        "| 适用范围 | 变更位置 | 影响级别 | 处理方式 |",
        "| --- | --- | --- | --- |",
    ]
    for rule in sorted(
        COMMON_RULES, key=lambda item: (LEVEL_ORDER[item.level], item.prefix)
    ):
        lines.append(f"| 公共 | `{rule.prefix}` | {rule.level} | {rule.handling} |")
    lines.append(
        f"| 公共 | `{UNUSED_RULE.prefix}` | {UNUSED_RULE.level} | {UNUSED_RULE.handling} |"
    )

    selected_profiles: list[ImpactProfile] = []
    for branch in data["branches"]:
        profile = profile_for_ref(branch["ref"])
        if profile is not None and profile not in selected_profiles:
            selected_profiles.append(profile)
    for profile in selected_profiles:
        for rule in sorted(
            profile.rules, key=lambda item: (LEVEL_ORDER[item.level], item.prefix)
        ):
            lines.append(
                f"| {profile.name} | `{rule.prefix}` | {rule.level} | {rule.handling} |"
            )
        if profile.unused_paths:
            lines.append(
                f"| {profile.name} | `{PROFILE_UNUSED_RULE.prefix}` | "
                f"{PROFILE_UNUSED_RULE.level} | {PROFILE_UNUSED_RULE.handling} |"
            )

    lines.extend(
        [
            "| 公共 | 其他路径 | 未分类 | 需人工判断是否影响 triton-anchor |",
            "",
            "## 分支概览",
            "",
            "| 上游分支 | Profile | HEAD | Commit 数 | 高 | 中 | 低 | 未分类 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for branch in data["branches"]:
        counts = branch["level_counts"]
        lines.append(
            "| `{ref}` | {profile} | `{head}` | {commits} | {high} | {medium} | {low} | {unknown} |".format(
                ref=branch["ref"],
                profile=branch["profile"],
                head=branch["head_short"],
                commits=branch["commit_count"],
                high=counts["高"],
                medium=counts["中"],
                low=counts["低"],
                unknown=counts["未分类"],
            )
        )

    for branch in data["branches"]:
        lines.extend(
            ["", f"## `{branch['ref']}`", "", f"Profile: {branch['profile']}", ""]
        )
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
