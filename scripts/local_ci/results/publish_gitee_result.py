#!/usr/bin/env python3
"""Publish local-ci logs to a Gitee results repository and add a short commit comment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SHARED_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPT_DIR))

from result_paths import (  # noqa: E402
    gitee_tree_url,
    result_commit_dir,
    result_run_dir,
    safe_path_part,
)


class PublishBudgetExceeded(RuntimeError):
    pass


class PublishBudget:
    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("GITEE result size limit must be positive")
        self.max_bytes = max_bytes
        self.used_bytes = 0

    def copy(self, source: Path, destination: Path) -> None:
        size = source.stat().st_size
        if size > self.max_bytes or self.used_bytes + size > self.max_bytes:
            raise PublishBudgetExceeded(
                f"Gitee result payload exceeds {self.max_bytes} bytes at {source}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        self.used_bytes += size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True, help="Source Gitee code repository owner for commit comments.")
    parser.add_argument("--repo", required=True, help="Source Gitee code repository name for commit comments.")
    parser.add_argument("--repo-url", required=True, help="Source Gitee code repository URL; kept for compatibility.")
    parser.add_argument("--results-owner", default="")
    parser.add_argument("--results-repo", default="")
    parser.add_argument("--results-repo-url", default="")
    parser.add_argument("--results-web-url", default="")
    parser.add_argument("--sha", required=True)
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--results-branch", default="local-ci-results")
    parser.add_argument("--context", default="local-ci/sophgo-cmodel")
    parser.add_argument("--execution-mode", default="full")
    parser.add_argument("--ci-profile", default="unavailable")
    parser.add_argument("--llvm-hash", default="unavailable")
    parser.add_argument(
        "--backend-stages-enabled", choices=("true", "false"), default="true"
    )
    parser.add_argument("--backend-skip-reason", default="")
    parser.add_argument(
        "--max-publish-bytes",
        type=int,
        default=int(os.getenv("GITEE_RESULT_MAX_BYTES", "268435456")),
    )
    return parser.parse_args()


def run_git(args: list[str], cwd: Path, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        check=check,
        text=True,
    )


def atomic_write_text(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def push_with_rebase_retry(
    worktree: Path,
    env: dict[str, str],
    results_branch: str,
    *,
    attempts: int = 3,
) -> bool:
    refspec = f"HEAD:refs/heads/{results_branch}"
    for attempt in range(1, attempts + 1):
        push = run_git(["push", "origin", refspec], worktree, env, check=False)
        if push.returncode == 0:
            if attempt > 1:
                print(f"Gitee result push succeeded after {attempt} attempts.")
            return True
        if attempt == attempts:
            break
        print(
            f"Gitee result push failed on attempt {attempt}; fetching and rebasing before retry.",
            file=sys.stderr,
        )
        fetch = run_git(
            ["fetch", "--depth=50", "origin", f"refs/heads/{results_branch}:refs/remotes/origin/{results_branch}"],
            worktree,
            env,
            check=False,
        )
        if fetch.returncode != 0:
            continue
        rebase = run_git(["rebase", f"origin/{results_branch}"], worktree, env, check=False)
        if rebase.returncode != 0:
            run_git(["rebase", "--abort"], worktree, env, check=False)
            print("Gitee result rebase failed; not overwriting remote results branch.", file=sys.stderr)
            return False
    return False


def make_git_env(tmpdir: Path, token: str, username: str) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        askpass = tmpdir / "gitee-askpass.sh"
        newline = chr(10)
        askpass.write_text(
            newline.join([
                "#!/usr/bin/env sh",
                'case "$1" in',
                '  *Username*) echo "${GITEE_USERNAME}" ;;',
                '  *) echo "${GITEE_TOKEN}" ;;',
                "esac",
            ])
            + newline
        )
        askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
        env["GIT_ASKPASS"] = str(askpass)
        env["GITEE_USERNAME"] = username
        env["GITEE_TOKEN"] = token
    return env


def discover_artifact_dir(run_log: Path) -> str:
    if not run_log.exists():
        return ""
    pattern = re.compile(r"(?:Artifact dir:|Artifacts are in)\s+(\S+)")
    found = ""
    for line in run_log.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            found = match.group(1)
    return found


def map_container_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    direct = Path(path_text)
    if direct.exists():
        return direct

    container_workspace = os.getenv("WORKSPACE", "/workspace").rstrip("/")
    host_workspace = os.getenv("LOCAL_CI_WORKSPACE_HOST", "").rstrip("/")
    if host_workspace and path_text.startswith(container_workspace + "/"):
        mapped = Path(host_workspace) / path_text[len(container_workspace) + 1 :]
        if mapped.exists():
            return mapped
    return None


PUBLISHED_ARTIFACT_FILES = (
    "delivery-summary.txt",
    "verify-triton-anchor-import.log",
    "frontend-smoke.log",
    "backend-smoke-jit.log",
    "flaggems-selected.txt",
    "flaggems-summary.csv",
    "flaggems-summary.json",
    "flaggems-summary.md",
    "compile-benchmark.json",
    "compile-benchmark.csv",
    "compile-time-comparison.json",
    "compile-time-comparison.md",
    "pass-profile.json",
    "pass-profile-summary.csv",
    "pass-profile-hotspots.md",
    "pass-profile-comparison.json",
    "pass-profile-comparison.csv",
    "pass-profile-comparison.md",
    "ir-serialization.json",
    "ir-serialization.csv",
    "ir-serialization-summary.md",
    "ir-serialization-comparison.json",
    "ir-serialization-comparison.csv",
    "ir-serialization-comparison.md",
)

PUBLISHED_RUN_FILES = (
    "task-metadata.json",
    "codex-ai-ci.log",
    "codex-ai-report.json",
    "codex-ai-report.md",
    "codex-ai-comment.md",
    "codex-ai-ci-summary.txt",
    "codex-context-summary.json",
    "codex-changed-files-manifest.json",
    "codex-workspace-status.txt",
    "codex-workspace.patch",
    "codex-generated-files.tar.gz",
)

REQUIRED_RESULT_FILES = (
    "delivery-summary.txt",
    "result.json",
)


def parse_summary_value(summary_path: Path, key: str) -> str:
    if not summary_path.is_file():
        return ""
    prefix = f"{key}:"
    for line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def build_publish_manifest(
    target_dir: Path,
    *,
    args: argparse.Namespace,
    rel_dir: Path,
    artifact_dir_text: str,
    artifact_dir: Path | None,
    copied: list[str],
    missing_expected: list[str],
    fallback: bool,
) -> None:
    summary_path = target_dir / "delivery-summary.txt"
    manifest = {
        "schema": "triton-anchor-local-ci-publish-manifest/v1",
        "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "passed" if args.exit_code == 0 else "failed",
        "exit_code": args.exit_code,
        "source_branch": args.source_branch,
        "target_sha": args.sha,
        "tested_sha": args.sha,
        "run_id": args.run_id,
        "context": args.context,
        "result_dir": rel_dir.as_posix(),
        "artifact_dir": artifact_dir_text or "",
        "artifact_dir_mapped": str(artifact_dir) if artifact_dir is not None else "",
        "fallback": fallback,
        "copied_files": sorted(copied),
        "missing_expected_files": sorted(missing_expected),
        "delivery_summary": {
            "schema": parse_summary_value(summary_path, "schema"),
            "status": parse_summary_value(summary_path, "status"),
            "target_sha": parse_summary_value(summary_path, "target_sha"),
            "tested_sha": parse_summary_value(summary_path, "tested_sha"),
            "tested_sha_kind": parse_summary_value(summary_path, "tested_sha_kind"),
            "actual_checkout_sha": parse_summary_value(summary_path, "actual_checkout_sha"),
            "run_id": parse_summary_value(summary_path, "run_id"),
        },
    }
    (target_dir / "publish-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_results(
    run_dir: Path,
    target_dir: Path,
    args: argparse.Namespace,
    rel_dir: Path,
    budget: PublishBudget,
) -> Path | None:
    artifact_dir_text = discover_artifact_dir(run_dir / "local-ci.log")
    artifact_dir = map_container_path(artifact_dir_text)
    if not artifact_dir or not artifact_dir.exists():
        return None

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing_expected = []
    for file_name in PUBLISHED_ARTIFACT_FILES:
        source = artifact_dir / file_name
        if source.is_file():
            budget.copy(source, target_dir / file_name)
            copied.append(file_name)

    for file_name in PUBLISHED_RUN_FILES:
        source = run_dir / file_name
        if source.is_file():
            budget.copy(source, target_dir / file_name)
            copied.append(file_name)

    result_json = run_dir / "result.json"
    if result_json.is_file():
        budget.copy(result_json, target_dir / "result.json")
        copied.append("result.json")

    for required_file in REQUIRED_RESULT_FILES:
        if required_file not in copied:
            missing_expected.append(required_file)

    if not copied:
        shutil.rmtree(target_dir)
        return None
    build_publish_manifest(
        target_dir,
        args=args,
        rel_dir=rel_dir,
        artifact_dir_text=artifact_dir_text,
        artifact_dir=artifact_dir,
        copied=copied,
        missing_expected=missing_expected,
        fallback=False,
    )
    return target_dir


def publish_performance_cache(
    worktree: Path,
    result_dir: Path | None,
    sha: str,
    *,
    cache_kind: str,
    source_name: str,
    label: str,
    sidecars: tuple[tuple[str, str], ...] = (),
) -> Path | None:
    if result_dir is None:
        return None
    source_json = result_dir / source_name
    if not source_json.is_file():
        return None

    try:
        document = json.loads(source_json.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"Cannot publish {label} cache from {source_json}: {exc}", file=sys.stderr)
        return None
    metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    profile = metadata.get("backend_profile") or metadata.get("backend") or "default"
    cache_dir = worktree / cache_kind / "by-sha" / sha / safe_path_part(str(profile))
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_json, cache_dir / "latest.json")
    for sidecar_name, target_name in sidecars:
        sidecar = result_dir / sidecar_name
        if sidecar.is_file():
            shutil.copy2(sidecar, cache_dir / target_name)
    return cache_dir


def publish_compile_time_cache(worktree: Path, result_dir: Path | None, sha: str) -> Path | None:
    return publish_performance_cache(
        worktree,
        result_dir,
        sha,
        cache_kind="compile-time",
        source_name="compile-benchmark.json",
        label="compile-time",
        sidecars=(("compile-benchmark.csv", "latest.csv"),),
    )


def publish_pass_profile_cache(worktree: Path, result_dir: Path | None, sha: str) -> Path | None:
    return publish_performance_cache(
        worktree,
        result_dir,
        sha,
        cache_kind="pass-profile",
        source_name="pass-profile.json",
        label="pass-profile",
        sidecars=(("pass-profile-summary.csv", "latest-summary.csv"),),
    )


def publish_ir_serialization_cache(
    worktree: Path, result_dir: Path | None, sha: str
) -> Path | None:
    return publish_performance_cache(
        worktree,
        result_dir,
        sha,
        cache_kind="ir-serialization",
        source_name="ir-serialization.json",
        label="IR serialization",
        sidecars=(
            ("ir-serialization.csv", "latest.csv"),
            ("ir-serialization-summary.md", "latest.md"),
        ),
    )


def ir_dashboard_rows(worktree: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cache_root = worktree / "ir-serialization" / "by-sha"
    if not cache_root.is_dir():
        return rows

    for result_file in cache_root.glob("*/*/latest.json"):
        try:
            document = json.loads(result_file.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"Skipping invalid IR dashboard input {result_file}: {exc}", file=sys.stderr)
            continue
        if not isinstance(document, dict):
            continue
        metadata = document.get("metadata", {})
        summary = document.get("summary", {})
        if not isinstance(metadata, dict) or not isinstance(summary, dict):
            continue
        sha = result_file.parents[1].name
        profile = result_file.parent.name
        generated_at = str(metadata.get("generated_at") or "")
        for kernel, kernel_data in summary.items():
            if not isinstance(kernel_data, dict):
                continue
            metrics = kernel_data.get("metrics", {})
            if not isinstance(metrics, dict):
                continue

            def median(metric: str) -> float | None:
                value = metrics.get(metric, {})
                value = value.get("median_ms") if isinstance(value, dict) else None
                return float(value) if isinstance(value, (int, float)) else None

            rows.append(
                {
                    "generated_at": generated_at,
                    "sha": sha,
                    "profile": profile,
                    "kernel": str(kernel),
                    "module_count": kernel_data.get("module_count"),
                    "ir_bytes": kernel_data.get("ir_bytes"),
                    "serialize_median_ms": median("serialize"),
                    "deserialize_median_ms": median("deserialize"),
                    "roundtrip_median_ms": median("roundtrip"),
                    "result_path": str(result_file.relative_to(worktree)).replace("\\", "/"),
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["generated_at"]),
            str(row["sha"]),
            str(row["profile"]),
            str(row["kernel"]),
        ),
        reverse=True,
    )
    return rows


def write_ir_serialization_dashboard(worktree: Path, limit: int = 100) -> tuple[Path, Path]:
    rows = ir_dashboard_rows(worktree)[:limit]
    dashboard_dir = worktree / "ir-serialization"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dashboard_dir / "dashboard.csv"
    fieldnames = [
        "generated_at",
        "sha",
        "profile",
        "kernel",
        "module_count",
        "ir_bytes",
        "serialize_median_ms",
        "deserialize_median_ms",
        "roundtrip_median_ms",
        "result_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)

    def format_ms(value: object) -> str:
        return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"

    lines = [
        "# IR serialization performance dashboard",
        "",
        "Latest SHA-indexed measurements, newest first. Times are medians in milliseconds.",
        "",
        "| Time (UTC) | Commit | Profile | Kernel | Modules | IR bytes | Serialize | Deserialize | Round-trip |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        short_sha = str(row["sha"])[:12]
        result_link = str(row["result_path"])
        if result_link.startswith("ir-serialization/"):
            result_link = result_link[len("ir-serialization/") :]
        lines.append(
            f"| {row['generated_at'] or 'unknown'} | [`{short_sha}`]({result_link}) | "
            f"{row['profile']} | {row['kernel']} | {row['module_count']} | {row['ir_bytes']} | "
            f"{format_ms(row['serialize_median_ms'])} | "
            f"{format_ms(row['deserialize_median_ms'])} | "
            f"{format_ms(row['roundtrip_median_ms'])} |"
        )
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | 0 | 0 | n/a | n/a | n/a |")
    markdown_path = dashboard_dir / "dashboard.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path, csv_path


def write_fallback_results(
    run_dir: Path,
    target_dir: Path,
    args: argparse.Namespace,
    rel_dir: Path,
    budget: PublishBudget,
) -> Path:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for file_name in ("local-ci.log", "result.json", *PUBLISHED_RUN_FILES):
        source = run_dir / file_name
        if source.is_file():
            budget.copy(source, target_dir / file_name)
            copied.append(file_name)

    artifact_dir_text = discover_artifact_dir(run_dir / "local-ci.log") or "unavailable"
    tested_sha_kind = "pr_merge" if args.source_branch.startswith("ci/pr-") else "commit"
    backend_stages_enabled = args.backend_stages_enabled == "true"
    deterministic_stages_skipped = args.execution_mode == "codex_only"
    frontend_status = "skipped" if deterministic_stages_skipped else "unavailable"
    backend_status = (
        "skipped"
        if deterministic_stages_skipped or not backend_stages_enabled
        else "unavailable"
    )
    backend_skip_reason = " ".join(args.backend_skip_reason.split())
    summary_lines = [
        "schema: triton-anchor-local-ci/v3",
        f"status: {args.exit_code}",
        f"target_sha: {args.sha}",
        f"tested_sha: {args.sha}",
        f"tested_sha_kind: {tested_sha_kind}",
        f"actual_checkout_sha: unavailable",
        f"branch: {args.source_branch}",
        f"run_id: {args.run_id}",
        f"execution_mode: {args.execution_mode}",
        f"ci_profile: {args.ci_profile or 'unavailable'}",
        f"llvm_hash: {args.llvm_hash or 'unavailable'}",
        f"backend_stages_enabled: {args.backend_stages_enabled}",
        f"backend_skip_reason: {backend_skip_reason}",
        f"artifact_dir: {artifact_dir_text}",
        f"frontend_build_status: {frontend_status}",
        f"frontend_smoke_status: {frontend_status}",
        f"backend_rebuild_status: {backend_status}",
        f"backend_smoke_jit_status: {backend_status}",
        f"flaggems_status: {backend_status}",
        f"compile_time_status: {backend_status}",
        f"pass_profile_status: {backend_status}",
        f"ir_serialization_status: {backend_status}",
        "note: artifact directory was unavailable; published host-side local-ci logs.",
        f"copied_files: {', '.join(copied) if copied else 'none'}",
    ]
    (target_dir / "delivery-summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    copied.append("delivery-summary.txt")
    missing_expected = [
        required_file
        for required_file in REQUIRED_RESULT_FILES
        if required_file not in copied
    ]
    build_publish_manifest(
        target_dir,
        args=args,
        rel_dir=rel_dir,
        artifact_dir_text=artifact_dir_text,
        artifact_dir=None,
        copied=copied,
        missing_expected=missing_expected,
        fallback=True,
    )
    return target_dir


def write_size_limit_result(
    run_dir: Path, target_dir: Path, args: argparse.Namespace, rel_dir: Path
) -> Path:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "sha": args.sha,
        "target_sha": args.sha,
        "tested_sha": args.sha,
        "status": 88,
        "failure_code": "gitee_result_size_limit",
        "run_dir": str(run_dir),
    }
    (run_dir / "gitee-result-size-limit.json").write_text(
        json.dumps(
            {
                "schema": "triton-anchor-local-ci-output-limit/v1",
                "failure_code": "gitee_result_size_limit",
                "limit_bytes": args.max_publish_bytes,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    (target_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (target_dir / "delivery-summary.txt").write_text(
        "\n".join(
            (
                "schema: triton-anchor-local-ci/v3",
                "status: 88",
                f"target_sha: {args.sha}",
                f"tested_sha: {args.sha}",
                f"branch: {args.source_branch}",
                f"run_id: {args.run_id}",
                "failure_code: gitee_result_size_limit",
                f"publish_limit_bytes: {args.max_publish_bytes}",
                "note: oversized result files were not published.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    build_publish_manifest(
        target_dir,
        args=args,
        rel_dir=rel_dir,
        artifact_dir_text="unavailable",
        artifact_dir=None,
        copied=["delivery-summary.txt", "result.json"],
        missing_expected=[],
        fallback=True,
    )
    return target_dir


def post_commit_comment(owner: str, repo: str, sha: str, token: str, body: str) -> None:
    path_owner = urllib.parse.quote(owner, safe="")
    path_repo = urllib.parse.quote(repo, safe="")
    path_sha = urllib.parse.quote(sha, safe="")
    url = f"https://gitee.com/api/v5/repos/{path_owner}/{path_repo}/commits/{path_sha}/comments"
    data = urllib.parse.urlencode({"access_token": token, "body": body}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
            print(f"Posted Gitee commit comment for {sha}: HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"Failed to post Gitee commit comment: HTTP {exc.code} {exc.reason}", file=sys.stderr)
        if error_body:
            print(error_body[:2000], file=sys.stderr)
        raise


def main() -> int:
    args = parse_args()
    try:
        publish_budget = PublishBudget(args.max_publish_bytes)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    token = os.getenv("GITEE_TOKEN", "")
    if not token:
        print("GITEE_TOKEN is not set; cannot publish Gitee result branch.", file=sys.stderr)
        return 1

    results_owner = args.results_owner or args.owner
    results_repo = args.results_repo or args.repo
    results_repo_url = args.results_repo_url or args.repo_url
    results_web_url = (
        args.results_web_url
        or os.getenv("GITEE_RESULTS_WEB_URL", "")
        or os.getenv("GITEE_WEB_URL", "")
        or f"https://gitee.com/{results_owner}/{results_repo}"
    ).rstrip("/")

    username = os.getenv("GITEE_USERNAME", args.owner)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    status_text = "passed" if args.exit_code == 0 else "failed"
    try:
        rel_dir = result_run_dir(
            args.source_branch, args.sha, args.run_id, args.head_sha
        )
        commit_dir = result_commit_dir(args.source_branch, args.sha, args.head_sha)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    result_url = gitee_tree_url(results_web_url, args.results_branch, rel_dir)

    with tempfile.TemporaryDirectory(prefix="triton-anchor-local-ci-results-") as tmp:
        tmp_path = Path(tmp)
        worktree = tmp_path / "repo"
        worktree.mkdir()
        git_env = make_git_env(tmp_path, token, username)

        run_git(["init", "-q"], worktree, git_env)
        run_git(["config", "user.name", "triton-anchor-local-ci"], worktree, git_env)
        run_git(["config", "user.email", "triton-anchor-local-ci@example.invalid"], worktree, git_env)
        run_git(["remote", "add", "origin", results_repo_url], worktree, git_env)

        fetch = run_git(
            ["fetch", "--depth=1", "origin", f"refs/heads/{args.results_branch}:refs/remotes/origin/{args.results_branch}"],
            worktree,
            git_env,
            check=False,
        )
        if fetch.returncode == 0:
            run_git(["checkout", "-q", "-B", args.results_branch, f"origin/{args.results_branch}"], worktree, git_env)
        else:
            run_git(["checkout", "-q", "--orphan", args.results_branch], worktree, git_env)

        target_dir = worktree / rel_dir
        try:
            copied_result_dir = copy_results(
                run_dir, target_dir, args, rel_dir, publish_budget
            )
            if copied_result_dir is None:
                print(
                    "Artifact result directory was unavailable; publishing fallback host logs.",
                    file=sys.stderr,
                )
                copied_result_dir = write_fallback_results(
                    run_dir, target_dir, args, rel_dir, publish_budget
                )
        except PublishBudgetExceeded as exc:
            print(str(exc), file=sys.stderr)
            args.exit_code = 88
            status_text = "failed"
            copied_result_dir = write_size_limit_result(
                run_dir, target_dir, args, rel_dir
            )

        compile_cache_dir = publish_compile_time_cache(worktree, copied_result_dir, args.sha)
        if compile_cache_dir is not None:
            print(f"Prepared compile-time cache: {compile_cache_dir.relative_to(worktree)}")
        pass_profile_cache_dir = publish_pass_profile_cache(worktree, copied_result_dir, args.sha)
        if pass_profile_cache_dir is not None:
            print(f"Prepared pass-profile cache: {pass_profile_cache_dir.relative_to(worktree)}")
        ir_serialization_cache_dir = publish_ir_serialization_cache(
            worktree, copied_result_dir, args.sha
        )
        if ir_serialization_cache_dir is not None:
            print(
                "Prepared IR serialization cache: "
                f"{ir_serialization_cache_dir.relative_to(worktree)}"
            )
        dashboard_markdown, dashboard_csv = write_ir_serialization_dashboard(worktree)
        print(
            "Updated IR serialization dashboard: "
            f"{dashboard_markdown.relative_to(worktree)}, "
            f"{dashboard_csv.relative_to(worktree)}"
        )

        latest_dir = worktree / commit_dir
        latest_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(latest_dir / "latest.txt", f"{args.run_id}\n")

        index = worktree / "index.md"
        index.write_text(
            "# Triton Anchor Local CI Results\n\n"
            "Result directories are grouped under runs/ci_full/, "
            "runs/ci_pr/, and runs/ci_push/.\n\n"
            "- Full: runs/ci_full/ci_full_<branch>/<commit>/<run-id>/\n"
            "- PR: runs/ci_pr/ci_pr-<number>_<branch>/h-<head12>_m-<merge12>/<run-id>/\n"
            "- PR base: runs/ci_pr/ci_base_pr-<number>_<branch>/<commit>/<run-id>/\n"
            "- Push: runs/ci_push/ci_push_<branch>/<commit>/<run-id>/\n\n"
            "Compile-time baselines are stored under "
            "compile-time/by-sha/<commit>/<backend-profile>/latest.json.\n"
            "Pass-profile baselines are stored under "
            "pass-profile/by-sha/<commit>/<backend-profile>/latest.json.\n"
            "IR serialization baselines are stored under "
            "ir-serialization/by-sha/<commit>/<backend-profile>/latest.json.\n\n"
            "IR serialization dashboard: [dashboard.md](ir-serialization/dashboard.md) "
            "([CSV](ir-serialization/dashboard.csv)).\n"
        )

        run_git(["add", "-A"], worktree, git_env)
        diff = run_git(["diff", "--cached", "--quiet"], worktree, git_env, check=False)
        if diff.returncode == 0:
            print("No Gitee result changes to publish.")
        else:
            run_git(["commit", "-q", "-m", f"local-ci: {status_text} {args.sha[:12]} {args.run_id}"], worktree, git_env)
            if not push_with_rebase_retry(worktree, git_env, args.results_branch):
                print("Failed to push Gitee local-ci results after retry.", file=sys.stderr)
                return 1
            print(f"Published Gitee local-ci results to {results_owner}/{results_repo}: {result_url}")

    comment_body = (
        f"local-ci {status_text}\n\n"
        f"- Branch: {args.source_branch}\n"
        f"- Commit: {args.sha}\n"
        f"- Run: {args.run_id}\n"
        f"- Context: {args.context}\n"
        f"- Exit code: {args.exit_code}\n"
        f"- Logs: {result_url}\n"
    )
    try:
        post_commit_comment(args.owner, args.repo, args.sha, token, comment_body)
    except Exception as exc:
        print(f"Warning: Gitee commit comment failed after results were published: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
