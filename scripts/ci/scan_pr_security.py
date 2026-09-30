#!/usr/bin/env python3

from __future__ import annotations

import os
import re
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
        re.compile(r"(?:\beval\s+|\bshell\s*=\s*True\b|\bos\.system\s*\()"),
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
        re.compile(r"\bsubprocess\.(?:call|check_call|check_output|Popen|run)\s*\("),
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
                    # Network capability alone is advisory, not a code execution finding.
                    warnings.append(Finding("warning", filename, line_number, message))

            if dependency_control_path:
                for message, pattern in DEPENDENCY_SOURCE_PATTERNS:
                    if pattern.search(line):
                        blocking.append(
                            Finding("error", filename, line_number, message)
                        )

            for message, pattern in EXECUTION_BLOCKING_PATTERNS:
                if message == "dynamic shell evaluation is not allowed" and Path(
                    filename
                ).suffix.lower() in {".md", ".markdown", ".rst"}:
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
        "Blocking security findings" if mode == "block" else "Security review warnings"
    )
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write(f"## {title}\n\n")
        if not findings:
            summary.write("No findings.\n")
            return
        for finding in findings:
            summary.write(f"- `{finding.filename}:{finding.line}`: {finding.message}\n")
