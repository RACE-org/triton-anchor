#!/usr/bin/env python3
"""Bridge Gitee local-ci results back to GitHub commit statuses."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


RESULT_NOT_READY_EXIT_CODE = 3
RESULT_FAILED_EXIT_CODE = 10


@dataclass(frozen=True)
class Target:
    source_branch: str
    task_ref: str
    sha: str
    label: str


@dataclass(frozen=True)
class LocalCIResult:
    exit_code: int | None
    target_url: str
    run_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gitee-owner", required=True)
    parser.add_argument("--gitee-repo", required=True)
    parser.add_argument("--gitee-results-branch", default="local-ci-results")
    parser.add_argument("--gitee-web-url", required=True)
    parser.add_argument("--source-branch", default="jiwang-delivery-ci")
    parser.add_argument("--reconcile-source-branches", default="")
    parser.add_argument("--task-ref", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--context", default="local-ci/sophgo-cmodel")
    parser.add_argument("--mode", choices=("single", "reconcile"), default="single")
    parser.add_argument("--set-pending", action="store_true")
    parser.add_argument("--max-prs", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=0)
    parser.add_argument("--require-result", action="store_true")
    parser.add_argument("--exit-with-result", action="store_true")
    return parser.parse_args()


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "default"


def request_json(url: str, method: str = "GET", token: str = "", data: dict | None = None) -> tuple[int, object | None, str]:
    body = None
    headers = {"Accept": "application/json"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(response_body) if response_body else None, response_body
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(response_body) if response_body else None
        except json.JSONDecodeError:
            parsed = None
        return exc.code, parsed, response_body


def gitee_content(owner: str, repo: str, path: str, ref: str, token: str) -> str | None:
    quoted_owner = urllib.parse.quote(owner, safe="")
    quoted_repo = urllib.parse.quote(repo, safe="")
    quoted_path = urllib.parse.quote(path, safe="/")
    params = {"ref": ref}
    if token:
        params["access_token"] = token
    query = urllib.parse.urlencode(params)
    url = f"https://gitee.com/api/v5/repos/{quoted_owner}/{quoted_repo}/contents/{quoted_path}?{query}"
    status, payload, raw = request_json(url)
    if status == 404:
        return None
    if status != 200:
        raise RuntimeError(f"Gitee content request failed: HTTP {status}: {raw[:500]}")
    if isinstance(payload, list):
        if not payload:
            return None
        raise RuntimeError(f"Gitee content response is a directory listing, not a file object: {raw[:500]}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Gitee content response is not a file object: {raw[:500]}")

    content = payload.get("content")
    encoding = payload.get("encoding")
    if not isinstance(content, str):
        raise RuntimeError(f"Gitee content response has no content field: {raw[:500]}")
    if encoding == "base64":
        return base64.b64decode(content).decode("utf-8", errors="replace")
    return content


def parse_summary_status(summary: str) -> int | None:
    for line in summary.splitlines():
        if line.startswith("status:"):
            value = line.split(":", 1)[1].strip()
            try:
                return int(value)
            except ValueError:
                return None
    return None


def github_api_url(path: str, params: dict[str, str] | None = None) -> str:
    url = f"https://api.github.com{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return url


def github_repo() -> str:
    return os.environ["GITHUB_REPOSITORY"]


def github_token() -> str:
    return os.environ["GITHUB_TOKEN"]


def github_status_url(sha: str) -> str:
    return github_api_url(f"/repos/{github_repo()}/statuses/{sha}")


def post_github_status(sha: str, state: str, context: str, description: str, target_url: str = "") -> None:
    payload = {
        "state": state,
        "context": context,
        "description": description[:140],
    }
    if target_url:
        payload["target_url"] = target_url

    status, _, raw = request_json(github_status_url(sha), method="POST", token=github_token(), data=payload)
    if status not in (200, 201):
        raise RuntimeError(f"GitHub status update failed: HTTP {status}: {raw[:500]}")


def get_github_json(path: str, params: dict[str, str] | None = None) -> object | None:
    status, payload, raw = request_json(github_api_url(path, params), token=github_token())
    if status == 404:
        return None
    if status != 200:
        raise RuntimeError(f"GitHub API request failed: HTTP {status}: {raw[:500]}")
    return payload


def github_branch_head(branch: str) -> str | None:
    quoted_branch = urllib.parse.quote(branch, safe="")
    payload = get_github_json(f"/repos/{github_repo()}/branches/{quoted_branch}")
    if not isinstance(payload, dict):
        return None
    commit = payload.get("commit")
    if not isinstance(commit, dict):
        return None
    sha = commit.get("sha")
    return sha if isinstance(sha, str) else None


def list_open_pr_targets(limit: int) -> list[Target]:
    targets: list[Target] = []
    page = 1
    per_page = min(max(limit, 1), 100)
    while len(targets) < limit:
        payload = get_github_json(
            f"/repos/{github_repo()}/pulls",
            {"state": "open", "per_page": str(per_page), "page": str(page)},
        )
        if not isinstance(payload, list) or not payload:
            break
        for item in payload:
            if len(targets) >= limit:
                break
            if not isinstance(item, dict):
                continue
            head = item.get("head")
            if not isinstance(head, dict):
                continue
            repo = head.get("repo")
            if not isinstance(repo, dict) or repo.get("full_name") != github_repo():
                print(f"Skip fork PR #{item.get('number')}: local CI only supports same-repository PRs")
                continue
            branch = head.get("ref")
            sha = head.get("sha")
            number = item.get("number")
            if isinstance(branch, str) and isinstance(sha, str) and isinstance(number, int):
                targets.append(Target(branch, f"ci/pr-{number}", sha, f"PR #{number}"))
        if len(payload) < per_page:
            break
        page += 1
    return targets


def gitee_result_url(web_url: str, results_branch: str, rel_dir: str) -> str:
    quoted_branch = urllib.parse.quote(results_branch, safe="")
    quoted_rel_dir = urllib.parse.quote(rel_dir, safe="/")
    return f"{web_url.rstrip('/')}/tree/{quoted_branch}/{quoted_rel_dir}"


def read_local_ci_result(args: argparse.Namespace, target: Target, gitee_token: str) -> LocalCIResult | None:
    safe_task_ref = safe_path_part(target.task_ref)
    commit_dir = f"runs/{safe_task_ref}/{target.sha}"
    latest_path = f"{commit_dir}/latest.txt"
    run_id_text = gitee_content(
        args.gitee_owner,
        args.gitee_repo,
        latest_path,
        args.gitee_results_branch,
        gitee_token,
    )
    if not run_id_text:
        print(f"No Gitee local CI result yet for {target.label} ({target.task_ref}): {latest_path}")
        return None

    run_id = run_id_text.strip().splitlines()[0]
    rel_dir = f"{commit_dir}/{run_id}"
    summary_path = f"{rel_dir}/delivery-summary.txt"
    summary = gitee_content(
        args.gitee_owner,
        args.gitee_repo,
        summary_path,
        args.gitee_results_branch,
        gitee_token,
    )
    if not summary:
        print(f"Gitee local CI run exists but summary is missing for {target.label}: {summary_path}")
        return None

    return LocalCIResult(
        parse_summary_status(summary),
        gitee_result_url(args.gitee_web_url, args.gitee_results_branch, rel_dir),
        run_id,
    )


def sync_target(args: argparse.Namespace, target: Target, set_pending: bool) -> LocalCIResult | None:
    gitee_token = os.getenv("GITEE_TOKEN", "")
    if set_pending:
        post_github_status(target.sha, "pending", args.context, "Waiting for Gitee local CI result")

    timeout = max(args.timeout_seconds, 0)
    interval = max(args.poll_interval_seconds, 1)
    deadline = time.monotonic() + timeout

    while True:
        result = read_local_ci_result(args, target, gitee_token)
        if result is not None:
            if result.exit_code == 0:
                post_github_status(target.sha, "success", args.context, "Gitee local CI passed", result.target_url)
                print(f"Gitee local CI passed for {target.label}: {result.target_url}")
            else:
                post_github_status(
                    target.sha,
                    "failure",
                    args.context,
                    f"Gitee local CI failed: status {result.exit_code}",
                    result.target_url,
                )
                print(f"Gitee local CI failed for {target.label}: {result.target_url}")
            return result

        if timeout == 0 or time.monotonic() >= deadline:
            print(f"No available Gitee local CI result for {target.label}; leaving GitHub status pending.")
            return None

        sleep_seconds = min(interval, max(1, int(deadline - time.monotonic())))
        print(f"Waiting {sleep_seconds}s before checking Gitee local CI result again...")
        time.sleep(sleep_seconds)


def reconcile_targets(args: argparse.Namespace) -> list[Target]:
    targets: list[Target] = []
    seen: set[tuple[str, str]] = set()
    configured_branches = args.reconcile_source_branches.strip()
    source_branches = re.split(r"[\s,]+", configured_branches) if configured_branches else [args.source_branch]

    for source_branch in source_branches:
        if not source_branch:
            continue
        branch_sha = github_branch_head(source_branch)
        if branch_sha:
            target = Target(
                source_branch,
                f"ci/push/{source_branch}",
                branch_sha,
                f"branch {source_branch}",
            )
            targets.append(target)
            seen.add((target.task_ref, target.sha))
        else:
            print(f"Source branch not found on GitHub: {source_branch}")

    for target in list_open_pr_targets(args.max_prs):
        key = (target.task_ref, target.sha)
        if key not in seen:
            targets.append(target)
            seen.add(key)
    return targets


def main() -> int:
    args = parse_args()

    if args.mode == "single":
        if not args.sha:
            print("--sha is required in single mode", file=sys.stderr)
            return 2
        target = Target(
            args.source_branch,
            args.task_ref or args.source_branch,
            args.sha,
            args.source_branch,
        )
        result = sync_target(args, target, args.set_pending)
        if result is None:
            return RESULT_NOT_READY_EXIT_CODE if args.require_result else 0
        if args.exit_with_result and result.exit_code != 0:
            return RESULT_FAILED_EXIT_CODE
        return 0

    updated = 0
    for target in reconcile_targets(args):
        if sync_target(args, target, set_pending=False) is not None:
            updated += 1
    print(f"Reconciled {updated} target(s) with available Gitee local CI results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
