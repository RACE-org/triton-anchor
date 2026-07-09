#!/usr/bin/env python3
"""Post a commit status to Gitee if GITEE_TOKEN is configured."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--state", required=True, choices=["pending", "success", "failure", "error"])
    parser.add_argument("--context", default="local-ci/sophgo-cmodel")
    parser.add_argument("--description", default="")
    parser.add_argument("--target-url", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("GITEE_TOKEN", "")
    if not token:
        print("GITEE_TOKEN is not set; skip posting Gitee commit status.")
        return 0

    path_owner = urllib.parse.quote(args.owner, safe="")
    path_repo = urllib.parse.quote(args.repo, safe="")
    path_sha = urllib.parse.quote(args.sha, safe="")
    url = f"https://gitee.com/api/v5/repos/{path_owner}/{path_repo}/statuses/{path_sha}"

    payload = {
        "access_token": token,
        "state": args.state,
        "context": args.context,
        "description": args.description[:140],
    }
    if args.target_url:
        payload["target_url"] = args.target_url

    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(f"Posted Gitee status {args.state} for {args.sha}: HTTP {response.status}")
            if body:
                print(body[:1000])
    except Exception as exc:
        print(f"Failed to post Gitee status: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

