#!/usr/bin/env python3
"""Bridge Gitee local-ci results back to a GitHub commit status."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gitee-owner", required=True)
    parser.add_argument("--gitee-repo", required=True)
    parser.add_argument("--gitee-results-branch", default="local-ci-results")
    parser.add_argument("--gitee-web-url", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--context", default="local-ci/sophgo-cmodel")
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
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


def github_status_url(sha: str) -> str:
    repository = os.environ["GITHUB_REPOSITORY"]
    return f"https://api.github.com/repos/{repository}/statuses/{sha}"


def post_github_status(sha: str, state: str, context: str, description: str, target_url: str = "") -> None:
    token = os.environ["GITHUB_TOKEN"]
    payload = {
        "state": state,
        "context": context,
        "description": description[:140],
    }
    if target_url:
        payload["target_url"] = target_url

    status, _, raw = request_json(github_status_url(sha), method="POST", token=token, data=payload)
    if status not in (200, 201):
        raise RuntimeError(f"GitHub status update failed: HTTP {status}: {raw[:500]}")


def gitee_result_url(web_url: str, results_branch: str, rel_dir: str) -> str:
    quoted_branch = urllib.parse.quote(results_branch, safe="")
    quoted_rel_dir = urllib.parse.quote(rel_dir, safe="/")
    return f"{web_url.rstrip('/')}/tree/{quoted_branch}/{quoted_rel_dir}"


def main() -> int:
    args = parse_args()
    gitee_token = os.getenv("GITEE_TOKEN", "")
    safe_branch = safe_path_part(args.source_branch)
    commit_dir = f"runs/{safe_branch}/{args.sha}"
    latest_path = f"{commit_dir}/latest.txt"

    post_github_status(args.sha, "pending", args.context, "Waiting for Gitee local CI result")

    deadline = time.monotonic() + args.timeout_seconds
    last_message = "result not found yet"
    while time.monotonic() < deadline:
        try:
            run_id_text = gitee_content(
                args.gitee_owner,
                args.gitee_repo,
                latest_path,
                args.gitee_results_branch,
                gitee_token,
            )
            if run_id_text:
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
                if summary:
                    exit_code = parse_summary_status(summary)
                    target_url = gitee_result_url(args.gitee_web_url, args.gitee_results_branch, rel_dir)
                    if exit_code == 0:
                        post_github_status(args.sha, "success", args.context, "Gitee local CI passed", target_url)
                        print(f"Gitee local CI passed: {target_url}")
                        return 0
                    post_github_status(args.sha, "failure", args.context, f"Gitee local CI failed: status {exit_code}", target_url)
                    print(f"Gitee local CI failed: {target_url}")
                    return 1
                last_message = f"run {run_id} found but delivery-summary.txt is missing"
            else:
                last_message = f"{latest_path} not found"
        except Exception as exc:  # noqa: BLE001 - keep polling through transient Gitee errors.
            last_message = str(exc)
            print(f"Waiting for Gitee local CI result: {last_message}", file=sys.stderr)

        time.sleep(args.poll_interval_seconds)

    timeout_url = gitee_result_url(args.gitee_web_url, args.gitee_results_branch, commit_dir)
    post_github_status(args.sha, "error", args.context, "Timed out waiting for Gitee local CI", timeout_url)
    print(f"Timed out waiting for Gitee local CI: {last_message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
