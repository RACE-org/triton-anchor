#!/usr/bin/env python3
"""Publish local-ci logs to a Gitee results repository and add a short commit comment."""

from __future__ import annotations

import argparse
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
from pathlib import Path


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
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--results-branch", default="local-ci-results")
    parser.add_argument("--context", default="local-ci/sophgo-cmodel")
    return parser.parse_args()


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "default"


def run_git(args: list[str], cwd: Path, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        check=check,
        text=True,
    )


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
    "frontend-install.log",
    "backend-smoke-jit.log",
    "flaggems.log",
)


def copy_results(run_dir: Path, target_dir: Path) -> Path | None:
    artifact_dir_text = discover_artifact_dir(run_dir / "local-ci.log")
    artifact_dir = map_container_path(artifact_dir_text)
    if not artifact_dir or not artifact_dir.exists():
        return None

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for file_name in PUBLISHED_ARTIFACT_FILES:
        source = artifact_dir / file_name
        if source.is_file():
            shutil.copy2(source, target_dir / file_name)
            copied.append(file_name)

    if not copied:
        shutil.rmtree(target_dir)
        return None
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
    token = os.getenv("GITEE_TOKEN", "")
    if not token:
        print("GITEE_TOKEN is not set; skip publishing Gitee result branch and commit comment.")
        return 0

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
    safe_branch = safe_path_part(args.source_branch)
    rel_dir = Path("runs") / safe_branch / args.sha / args.run_id
    quoted_branch = urllib.parse.quote(args.results_branch, safe="")
    quoted_rel_dir = urllib.parse.quote(str(rel_dir), safe="/")
    result_url = f"{results_web_url}/tree/{quoted_branch}/{quoted_rel_dir}"

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
        copy_results(run_dir, target_dir)

        latest_dir = worktree / "runs" / safe_branch / args.sha
        latest_dir.mkdir(parents=True, exist_ok=True)
        (latest_dir / "latest.txt").write_text(f"{args.run_id}\n")

        index = worktree / "index.md"
        index.write_text(
            "# Triton Anchor Local CI Results\n\n"
            "Result directories are stored under runs/<branch>/<commit>/<run-id>/.\n"
        )

        run_git(["add", "-A"], worktree, git_env)
        diff = run_git(["diff", "--cached", "--quiet"], worktree, git_env, check=False)
        if diff.returncode == 0:
            print("No Gitee result changes to publish.")
        else:
            run_git(["commit", "-q", "-m", f"local-ci: {status_text} {args.sha[:12]} {args.run_id}"], worktree, git_env)
            run_git(["push", "origin", f"HEAD:refs/heads/{args.results_branch}"], worktree, git_env)
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
    post_commit_comment(args.owner, args.repo, args.sha, token, comment_body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
