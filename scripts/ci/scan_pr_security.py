#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private key material",
        re.compile(r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----"),
    ),
    (
        "GitHub token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "GitHub fine-grained token",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
    ),
    (
        "Slack token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    ),
    (
        "AWS access key identifier",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
)

NETWORK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "new Python network module import",
        re.compile(
            r"^\s*(?:"
            r"from\s+(?:aiohttp|ftplib|httpx|paramiko|requests|smtplib|socket|"
            r"urllib(?:\.[A-Za-z_][A-Za-z0-9_]*)?|websocket|websockets)\b|"
            r"import\s+[^#]*\b(?:aiohttp|ftplib|httpx|paramiko|requests|smtplib|"
            r"socket|urllib(?:\.[A-Za-z_][A-Za-z0-9_]*)?|websocket|websockets)\b"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "new Python network request",
        re.compile(
            r"\b(?:"
            r"aiohttp\.|httpx\.|requests\.|urllib\.request|urlopen\s*\(|"
            r"socket\.socket\s*\(|socket\.create_connection\s*\(|"
            r"HTTPConnection\s*\(|HTTPSConnection\s*\(|"
            r"websocket\.create_connection\s*\(|websockets\.connect\s*\("
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "new shell network command",
        re.compile(
            r"(?:^|[;&|]\s*|\b(?:exec|sudo|timeout)\s+)\s*"
            r"(?:curl|ftp|nc|ncat|netcat|scp|sftp|socat|ssh|wget)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "new remote Git operation",
        re.compile(
            r"\bgit\s+(?:clone|fetch|ls-remote|pull|push)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "new PowerShell network request",
        re.compile(
            r"\b(?:Invoke-RestMethod|Invoke-WebRequest|System\.Net\.WebClient)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "new JavaScript network request",
        re.compile(
            r"\b(?:axios\.(?:delete|get|patch|post|put)|"
            r"https?\.(?:get|request)|new\s+WebSocket)\s*\(",
            re.IGNORECASE,
        ),
    ),
    (
        "new native socket or libcurl dependency",
        re.compile(
            r"#\s*include\s*[<\"](?:curl/curl\.h|sys/socket\.h|winsock2\.h)[>\"]",
            re.IGNORECASE,
        ),
    ),
)

EXECUTION_BLOCKING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "remote content is piped directly into a shell",
        re.compile(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:ba)?sh\b", re.IGNORECASE),
    ),
    (
        "dynamic shell evaluation is not allowed",
        re.compile(
            r"(?:\beval\s+|\bshell\s*=\s*True\b|\bos\.system\s*\()"
        ),
    ),
    (
        "workflow requests sudo privileges",
        re.compile(r"\bsudo\b"),
    ),
    (
        "world-writable permission requested",
        re.compile(r"\bchmod\s+(?:-[A-Za-z]+\s+)?777\b"),
    ),
)

WARNING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "local subprocess execution should be reviewed",
        re.compile(
            r"\bsubprocess\.(?:call|check_call|check_output|Popen|run)\s*\("
        ),
    ),
)

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
SECURITY_RELEVANT_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".cjs",
    ".cmake",
    ".cpp",
    ".cxx",
    ".dll",
    ".exe",
    ".go",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".lock",
    ".mjs",
    ".php",
    ".pl",
    ".ps1",
    ".psm1",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".sh",
    ".so",
    ".toml",
    ".ts",
    ".tsx",
    ".whl",
    ".yaml",
    ".yml",
    ".zsh",
}
SECURITY_RELEVANT_NAMES = {
    "cmakelists.txt",
    "dockerfile",
    "makefile",
    "pipfile",
}
PROTECTED_PATH_PREFIXES = (
    ".github/actions/",
    # ".github/workflows/",
    # "docker/",
    # "scripts/ci/",
    # "scripts/dashboard/",
    # "scripts/local_ci/",
)
PROTECTED_FILES = {
    ".gitmodules",
}
TEMP_CI_REVIEW_PATH_PREFIXES = (
    ".github/workflows/",
    "docker/",
    "docs/",
    "scripts/ci/",
    "scripts/dashboard/",
    "scripts/local_ci/",
)
DEPENDENCY_CONTROL_FILES = {
    "pipfile",
    "pipfile.lock",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
}
DEPENDENCY_SOURCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "new direct URL or VCS dependency source",
        re.compile(
            r"(?:^|\s)@?\s*(?:git\+|https?://|ssh://|git://)",
            re.IGNORECASE,
        ),
    ),
    (
        "new custom Python package index or link source",
        re.compile(
            r"^\s*(?:-i\b|--index-url\b|--extra-index-url\b|--find-links\b|"
            r"--trusted-host\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "new setuptools dependency_links source",
        re.compile(r"\bdependency_links\s*=", re.IGNORECASE),
    ),
)
PROTECTED_PATH_MESSAGE = (
    "ordinary pull requests may not modify trusted CI or runner control files"
)
UNSCANNED_PATCH_MESSAGE = (
    "GitHub did not provide a textual patch; this changed file was not scanned."
)
DEPENDENCY_REVIEW_MESSAGE = (
    "dependency or build control file changed; review before authorization"
)


@dataclass(frozen=True)
class Finding:
    """表示安全检查生成的一条 GitHub 标注。"""

    level: str
    filename: str
    line: int
    message: str


def github_api_get(url: str, token: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "triton-anchor-security-gate",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_pr_files(
    api_url: str, repository: str, pr_number: str, token: str
) -> list[dict[str, object]]:
    query = urllib.parse.urlencode({"per_page": "100"})
    url = f"{api_url}/repos/{repository}/pulls/{pr_number}/files?{query}"
    payload = github_api_get(url, token)
    if not isinstance(payload, list):
        raise RuntimeError(
            "GitHub API returned an unexpected pull-request file payload"
        )
    if len(payload) >= 100:
        raise RuntimeError(
            "pull request changes 100 or more files; split it before security review"
        )
    return [item for item in payload if isinstance(item, dict)]


def fetch_pr_head_sha(
    api_url: str, repository: str, pr_number: str, token: str
) -> str:
    url = f"{api_url}/repos/{repository}/pulls/{pr_number}"
    payload = github_api_get(url, token)
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub API returned an unexpected pull-request payload")
    head = payload.get("head")
    if not isinstance(head, dict) or not isinstance(head.get("sha"), str):
        raise RuntimeError("GitHub API pull-request payload did not contain head.sha")
    return str(head["sha"])


def require_pr_head(
    api_url: str,
    repository: str,
    pr_number: str,
    token: str,
    expected_head_sha: str,
) -> None:
    actual_head_sha = fetch_pr_head_sha(api_url, repository, pr_number, token)
    if actual_head_sha != expected_head_sha:
        raise RuntimeError(
            "pull request changed during security review: "
            f"expected {expected_head_sha}, found {actual_head_sha}"
        )


def requires_text_scan(filename: str) -> bool:
    path = filename.lower()
    name = path.rsplit("/", 1)[-1]
    suffix = Path(path).suffix
    return (
        suffix in SECURITY_RELEVANT_SUFFIXES
        or name in SECURITY_RELEVANT_NAMES
        or name.startswith("dockerfile.")
        or is_dependency_control_path(filename)
    )


def is_protected_ci_path(filename: str) -> bool:
    path = filename.replace("\\", "/").lower()
    return path.startswith(PROTECTED_PATH_PREFIXES) or path in PROTECTED_FILES


def is_dependency_control_path(filename: str) -> bool:
    path = filename.replace("\\", "/").lower()
    name = path.rsplit("/", 1)[-1]
    return (
        path in DEPENDENCY_CONTROL_FILES
        or name.startswith("requirements")
        and name.endswith(".txt")
    )


def has_ci_review_exception(filename: str) -> bool:
    path = filename.replace("\\", "/").lower()
    return path.startswith(TEMP_CI_REVIEW_PATH_PREFIXES)


def added_lines(patch: str) -> Iterator[tuple[int, str]]:
    current_line: int | None = None
    for raw_line in patch.splitlines():
        hunk = HUNK.match(raw_line)
        if hunk:
            current_line = int(hunk.group(1))
            continue
        if current_line is None:
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            yield current_line, raw_line[1:]
            current_line += 1
        elif not raw_line.startswith("-"):
            current_line += 1


def git_output(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    return completed.stdout


def restore_missing_patches(
    files: list[dict[str, object]],
    repository: Path,
    comparison_base_sha: str,
    expected_head_sha: str,
    tested_sha: str,
) -> list[dict[str, object]]:
    missing = [
        item
        for item in files
        if not isinstance(item.get("patch"), str)
        and item.get("status") != "removed"
        and requires_text_scan(str(item.get("filename", "")))
    ]
    if not missing:
        return files

    checked_out_sha = git_output(repository, "rev-parse", "HEAD").strip()
    if checked_out_sha != tested_sha:
        raise RuntimeError("diff fallback did not check out the frozen merge result")

    parents = git_output(
        repository, "show", "-s", "--format=%P", tested_sha
    ).strip().split()
    if parents != [comparison_base_sha, expected_head_sha]:
        raise RuntimeError("diff fallback merge parents do not match the authorized base/head")

    restored: list[dict[str, object]] = []
    for original in files:
        item = dict(original)
        filename = str(item.get("filename", ""))
        if original in missing:
            patch = git_output(
                repository,
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--unified=3",
                comparison_base_sha,
                tested_sha,
                "--",
                filename,
            )
            if any(HUNK.match(line) for line in patch.splitlines()):
                item["patch"] = patch
        restored.append(item)
    return restored


def scan(files: list[dict[str, object]]) -> tuple[list[Finding], list[Finding]]:

    blocking: list[Finding] = []
    warnings: list[Finding] = []

    for item in files:
        filename = str(item.get("filename", "<unknown>"))
        patch = item.get("patch")

        if is_protected_ci_path(filename):
            blocking.append(
                Finding(
                    "error",
                    filename,
                    1,
                    PROTECTED_PATH_MESSAGE,
                )
            )
            continue

        dependency_control_path = is_dependency_control_path(filename)

        if not isinstance(patch, str):
            if item.get("status") == "removed":
                continue
            must_scan = requires_text_scan(filename)
            finding = Finding(
                "error" if must_scan else "warning",
                filename,
                1,
                UNSCANNED_PATCH_MESSAGE,
            )
            if must_scan:
                blocking.append(finding)
            else:
                warnings.append(finding)
            continue

        for line_number, line in added_lines(patch):
            for message, pattern in CREDENTIAL_PATTERNS:
                if pattern.search(line):
                    blocking.append(Finding("error", filename, line_number, message))

            for message, pattern in NETWORK_PATTERNS:
                if has_ci_review_exception(filename):
                    continue
                if pattern.search(line):
                    blocking.append(Finding("error", filename, line_number, message))

            if dependency_control_path:
                for message, pattern in DEPENDENCY_SOURCE_PATTERNS:
                    if pattern.search(line):
                        blocking.append(
                            Finding("error", filename, line_number, message)
                        )

            for message, pattern in EXECUTION_BLOCKING_PATTERNS:
                if (
                    message == "dynamic shell evaluation is not allowed"
                    and Path(filename).suffix.lower()
                    in {".md", ".markdown", ".rst"}
                ):
                    continue
                if pattern.search(line):
                    blocking.append(Finding("error", filename, line_number, message))

            for message, pattern in WARNING_PATTERNS:
                if pattern.search(line):
                    warnings.append(Finding("warning", filename, line_number, message))

        if dependency_control_path:
            warnings.append(
                Finding(
                    "warning",
                    filename,
                    1,
                    DEPENDENCY_REVIEW_MESSAGE,
                )
            )

    return blocking, warnings


def print_findings(findings: list[Finding]) -> None:

    for finding in findings:
        display = f"{finding.filename}:{finding.line}: {finding.message}"
        command_value = (
            display.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        )
        print(f"::{finding.level}::{command_value}")
        print(f"{finding.level.upper()}: {display}")


def append_summary(mode: str, findings: list[Finding]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    title = (
        "Blocking security findings"
        if mode == "block"
        else "Security review warnings"
    )
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write(f"## {title}\n\n")
        if not findings:
            summary.write("No findings.\n")
            return
        for finding in findings:
            summary.write(f"- `{finding.filename}:{finding.line}`: {finding.message}\n")


def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("block", "warn"), required=True)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--pr-number", default=os.environ.get("PR_NUMBER", ""))
    parser.add_argument(
        "--api-url",
        default=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )
    parser.add_argument("--token", default=os.environ.get("GH_TOKEN", ""))
    parser.add_argument(
        "--expected-head-sha", default=os.environ.get("EXPECTED_HEAD_SHA", "")
    )
    parser.add_argument(
        "--comparison-base-sha", default=os.environ.get("COMPARISON_BASE_SHA", "")
    )
    parser.add_argument("--tested-sha", default=os.environ.get("TESTED_SHA", ""))
    parser.add_argument(
        "--diff-repository", default=os.environ.get("DIFF_REPOSITORY", "")
    )
    args = parser.parse_args()

    if (
        not args.repository
        or not args.pr_number
        or not args.token
        or not args.expected_head_sha
    ):
        parser.error(
            "repository, pr number, expected head SHA, and GitHub token are required"
        )

    api_url = args.api_url.rstrip("/")
    require_pr_head(
        api_url,
        args.repository,
        args.pr_number,
        args.token,
        args.expected_head_sha,
    )
    files = fetch_pr_files(api_url, args.repository, args.pr_number, args.token)
    if any(
        not isinstance(item.get("patch"), str)
        and item.get("status") != "removed"
        and requires_text_scan(str(item.get("filename", "")))
        for item in files
    ):
        if (
            not args.comparison_base_sha
            or not args.tested_sha
            or not args.diff_repository
        ):
            raise RuntimeError(
                "text diff fallback requires comparison base SHA, tested SHA, "
                "and diff repository"
            )
        files = restore_missing_patches(
            files,
            Path(args.diff_repository),
            args.comparison_base_sha,
            args.expected_head_sha,
            args.tested_sha,
        )
    require_pr_head(
        api_url,
        args.repository,
        args.pr_number,
        args.token,
        args.expected_head_sha,
    )
    blocking, warnings = scan(files)
    selected = blocking if args.mode == "block" else warnings
    print_findings(selected)
    append_summary(args.mode, selected)

    if args.mode == "block" and blocking:
        print(
            "Security gate blocked this PR: "
            f"{len(blocking)} blocking security finding(s)."
        )
        return 1

    print(f"Security {args.mode} scan completed: {len(selected)} finding(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
