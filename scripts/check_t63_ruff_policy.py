#!/usr/bin/env python3
"""Fail-closed policy audit for the Triton 3.6 Ruff release gate."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tokenize
from collections.abc import Iterable
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "triton-anchor-t63-ruff-policy-v1"
RUFF_VERSION = "0.15.22"
TOMLI_VERSION = "2.2.1"
SHARED_GATE = "scripts/run_t63_ruff_gates.sh"
RELEASE_BRANCH = "triton_v3.6"
CI_INSTALL_COMMAND = f"pip install 'ruff=={RUFF_VERSION}' 'tomli=={TOMLI_VERSION}'"
CI_GATE_COMMAND = f"bash {SHARED_GATE}"
RUFF_CONFIG_NAMES = frozenset({"ruff.toml", ".ruff.toml", "pyproject.toml"})
FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "ignore",
        "extend-ignore",
        "per-file-ignores",
        "extend-per-file-ignores",
    }
)
FORBIDDEN_GATE_FLAGS = (
    "--fix",
    "--unsafe-fixes",
    "--ignore",
    "--extend-ignore",
    "--per-file-ignores",
    "--extend-per-file-ignores",
)
FILE_NOQA_RE = re.compile(r"^#\s*(?:ruff|flake8)\s*:\s*noqa\b", re.IGNORECASE)
INLINE_NOQA_RE = re.compile(r"^#\s*noqa\b", re.IGNORECASE)


class RuffConfigParserUnavailable(RuntimeError):
    """Raised when Python has neither stdlib tomllib nor the pinned backport."""


def finding(
    rule: str,
    path: str,
    message: str,
    *,
    line: int | None = None,
) -> dict[str, object]:
    """Build one stable, JSON-serializable policy finding."""
    result: dict[str, object] = {
        "message": message,
        "path": path,
        "rule": rule,
    }
    if line is not None:
        result["line"] = line
    return result


def display_path(path: Path, repo_root: Path) -> str:
    """Return a repository-relative display path when possible."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return os.fspath(path)


def resolve_repo_input(repo_root: Path, value: str, label: str) -> Path:
    """Resolve a required input and reject paths outside the repository."""
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"{label} escapes repository root: {value!r}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {value!r}")
    return resolved


def read_nul_manifest(manifest: Path) -> tuple[list[str], list[dict[str, object]]]:
    """Read a NUL-delimited manifest without line-oriented path assumptions."""
    issues: list[dict[str, object]] = []
    try:
        raw = manifest.read_bytes()
    except OSError as error:
        return [], [finding("manifest-read", os.fspath(manifest), str(error))]

    if raw and not raw.endswith(b"\0"):
        issues.append(
            finding(
                "manifest-format",
                os.fspath(manifest),
                "changed manifest must end with a NUL byte",
            )
        )
    entries = raw.split(b"\0")
    if entries and entries[-1] == b"":
        entries.pop()
    if any(entry == b"" for entry in entries):
        issues.append(
            finding(
                "manifest-format",
                os.fspath(manifest),
                "changed manifest contains an empty path entry",
            )
        )

    paths = [os.fsdecode(entry) for entry in entries if entry]
    if len(paths) != len(set(paths)):
        issues.append(
            finding(
                "manifest-duplicate",
                os.fspath(manifest),
                "changed manifest contains duplicate paths",
            )
        )
    return paths, issues


def audit_changed_sources(
    repo_root: Path,
    manifest: Path,
) -> list[dict[str, object]]:
    """Audit changed Python comments using Python's lexical semantics."""
    paths, issues = read_nul_manifest(manifest)
    for relative_text in paths:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            issues.append(
                finding(
                    "manifest-path",
                    relative_text,
                    "changed path must be repository-relative and cannot traverse '..'",
                )
            )
            continue
        if relative.suffix not in {".py", ".pyi"}:
            issues.append(
                finding(
                    "manifest-path",
                    relative_text,
                    "changed manifest may contain only .py and .pyi files",
                )
            )
            continue

        source_path = repo_root / relative
        try:
            resolved = source_path.resolve(strict=True)
        except OSError as error:
            issues.append(finding("source-read", relative_text, str(error)))
            continue
        if not resolved.is_relative_to(repo_root) or not resolved.is_file():
            issues.append(
                finding(
                    "manifest-path",
                    relative_text,
                    "changed source resolves outside the repository or is not a file",
                )
            )
            continue

        try:
            source_bytes = resolved.read_bytes()
            comments = (
                token
                for token in tokenize.tokenize(io.BytesIO(source_bytes).readline)
                if token.type == tokenize.COMMENT
            )
            for token in comments:
                comment = token.string
                if FILE_NOQA_RE.match(comment):
                    issues.append(
                        finding(
                            "file-noqa",
                            relative_text,
                            "file-level Ruff/Flake8 noqa is not approved",
                            line=token.start[0],
                        )
                    )
                elif INLINE_NOQA_RE.match(comment):
                    issues.append(
                        finding(
                            "inline-noqa",
                            relative_text,
                            "no local noqa exception is approved for this release gate",
                            line=token.start[0],
                        )
                    )
        except (OSError, SyntaxError, UnicodeError, tokenize.TokenError) as error:
            issues.append(finding("source-tokenize", relative_text, str(error)))
    return issues


def tracked_files(repo_root: Path) -> tuple[list[str], list[dict[str, object]]]:
    """Return Git's tracked paths as a NUL-safe list."""
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        return [], [finding("git-ls-files", ".", str(error))]
    if completed.returncode != 0:
        detail = os.fsdecode(completed.stderr).strip() or "git ls-files failed"
        return [], [finding("git-ls-files", ".", detail)]
    return [os.fsdecode(item) for item in completed.stdout.split(b"\0") if item], []


def load_toml(path: Path) -> dict[str, Any]:
    """Load TOML with the standard parser available on supported runners."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError as error:
            raise RuffConfigParserUnavailable(
                "Python has neither tomllib nor the required tomli==2.2.1"
            ) from error
    with path.open("rb") as stream:
        parsed = tomllib.load(stream)
    if not isinstance(parsed, dict):
        raise TypeError("TOML root must be a table")
    return parsed


def walk_forbidden_keys(value: Any) -> Iterable[str]:
    """Yield forbidden Ruff keys found at any depth of a Ruff configuration."""
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        normalized = str(key).strip().lower().replace("_", "-")
        if normalized in FORBIDDEN_CONFIG_KEYS:
            yield normalized
        yield from walk_forbidden_keys(child)


def audit_ruff_configs(repo_root: Path) -> list[dict[str, object]]:
    """Reject tracked Ruff configuration capable of weakening the full gate."""
    tracked, issues = tracked_files(repo_root)
    for relative_text in tracked:
        relative = Path(relative_text)
        if relative.name not in RUFF_CONFIG_NAMES:
            continue
        path = repo_root / relative
        try:
            parsed = load_toml(path)
            if relative.name == "pyproject.toml":
                tool = parsed.get("tool", {})
                if not isinstance(tool, dict) or "ruff" not in tool:
                    continue
                ruff_config = tool["ruff"]
            else:
                ruff_config = parsed
            forbidden = sorted(set(walk_forbidden_keys(ruff_config)))
            for key in forbidden:
                issues.append(
                    finding(
                        "ruff-config-bypass",
                        relative.as_posix(),
                        f"tracked Ruff configuration sets forbidden key {key!r}",
                    )
                )
        except RuffConfigParserUnavailable as error:
            issues.append(
                finding(
                    "ruff-config-parser-unavailable",
                    relative.as_posix(),
                    str(error),
                )
            )
        except (OSError, TypeError, UnicodeError, ValueError) as error:
            issues.append(finding("ruff-config-parse", relative.as_posix(), str(error)))
    return issues


def shell_tokens(source: str) -> Iterable[tuple[int, str]]:
    """Yield shell-like tokens while ignoring ordinary Bash comments."""
    logical = source.replace("\\\r\n", "").replace("\\\n", "")
    for line_number, line in enumerate(logical.splitlines(), start=1):
        lexer = shlex.shlex(line, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        try:
            for token in lexer:
                yield line_number, token
        except ValueError:
            # A syntax error is independently caught by the gate's bash -n check. Scan
            # the raw line here so a malformed quote cannot hide a forbidden option.
            for token in line.split():
                yield line_number, token


def audit_gate(repo_root: Path, gate: Path) -> list[dict[str, object]]:
    """Reject mutating and lint-weakening options in the shared Bash gate."""
    relative = display_path(gate, repo_root)
    try:
        source = gate.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [finding("gate-read", relative, str(error))]

    issues: list[dict[str, object]] = []
    for line_number, token in shell_tokens(source):
        for flag in FORBIDDEN_GATE_FLAGS:
            if token == flag or token.startswith(f"{flag}="):
                issues.append(
                    finding(
                        "gate-forbidden-option",
                        relative,
                        f"shared gate contains forbidden Ruff option {flag}",
                        line=line_number,
                    )
                )
                break
    return issues


def lint_job(workflow: str) -> str | None:
    """Extract the lint job text without requiring a YAML dependency."""
    match = re.search(
        r"(?ms)^  lint:\s*$\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\s*$|\Z)",
        workflow,
    )
    return match.group("body") if match else None


def workflow_event_body(workflow: str, event: str) -> str | None:
    """Extract one top-level workflow event block using the canonical layout."""
    trigger = re.search(
        r"(?ms)^on:\s*$\n(?P<body>.*?)(?=^[A-Za-z_][A-Za-z0-9_-]*:\s*$)",
        workflow,
    )
    if trigger is None:
        return None
    match = re.search(
        rf"(?ms)^  {re.escape(event)}:\s*$\n"
        r"(?P<body>.*?)(?=^  [A-Za-z_][A-Za-z0-9_-]*:\s*$|\Z)",
        trigger.group("body"),
    )
    return match.group("body") if match else None


def audit_workflow(repo_root: Path, workflow_path: Path) -> list[dict[str, object]]:
    """Enforce the PR-head, exact-pin, and shared-gate workflow contract."""
    relative = display_path(workflow_path, repo_root)
    try:
        workflow = workflow_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [finding("workflow-read", relative, str(error))]
    lint = lint_job(workflow)
    if lint is None:
        return [finding("workflow-lint-job", relative, "dedicated lint job is missing")]

    issues: list[dict[str, object]] = []

    def require(condition: bool, rule: str, message: str) -> None:
        if not condition:
            issues.append(finding(rule, relative, message))

    require(
        re.search(r"(?m)^defaults:\s*", workflow) is None,
        "workflow-top-level-defaults",
        "top-level run defaults can alter the shared gate execution context",
    )
    require(
        re.search(
            r"(?m)^  (?:PATH|BASH_ENV|ENV|SHELLOPTS):\s*",
            workflow,
        )
        is None,
        "workflow-global-shell-environment",
        "top-level shell-control environment variables are forbidden",
    )
    require(
        re.search(r"(?m)^    if:\s*", lint) is None,
        "workflow-lint-job-condition",
        "lint job cannot have a job-level if condition",
    )
    require(
        re.search(r"(?m)^    (?:needs|defaults|container):\s*", lint) is None,
        "workflow-lint-job-context",
        "lint job cannot use needs, defaults, or a container execution context",
    )

    action_sequence = re.findall(
        r"(?m)^\s+(?:-\s+)?uses:\s*(\S+)\s*$",
        lint,
    )
    require(
        action_sequence
        == [
            "actions/checkout@v4",
            "actions/setup-python@v5",
            "actions/upload-artifact@v4",
        ],
        "workflow-action-sequence",
        "lint job must use only checkout, setup-python, and upload-artifact "
        "in canonical order",
    )

    gate_step = re.search(
        r"(?ms)^      - name: Run deterministic T6\.3 Ruff release gates\s*$\n"
        r"(?P<body>.*?)(?=^      - |\Z)",
        lint,
    )
    require(
        gate_step is not None,
        "workflow-gate-step",
        "canonical shared-gate step is missing",
    )
    if gate_step is not None:
        gate_body = gate_step.group("body")
        require(
            re.search(
                r"(?m)^\s+(?:if|shell|working-directory):\s*",
                gate_body,
            )
            is None,
            "workflow-gate-step-control",
            "shared-gate step cannot alter or skip its execution context",
        )
        require(
            re.findall(r"(?m)^\s+run:\s*(.*?)\s*$", gate_body) == [CI_GATE_COMMAND],
            "workflow-gate-step-command",
            "canonical shared-gate step must own the exact gate command",
        )
        require(
            re.findall(
                r"(?m)^          ([A-Za-z_][A-Za-z0-9_]*):",
                gate_body,
            )
            == ["PRODUCTION_CANDIDATE_SHA", "T63_RUFF_EVIDENCE_DIR"],
            "workflow-gate-step-environment",
            "shared-gate environment must contain only the two approved bindings",
        )

    upload_step = re.search(
        r"(?ms)^      - name: Upload Ruff gate evidence\s*$\n"
        r"(?P<body>.*?)(?=^      - |\Z)",
        lint,
    )
    require(
        upload_step is not None,
        "workflow-evidence-upload",
        "Ruff evidence upload step is missing",
    )
    if upload_step is not None:
        upload_body = upload_step.group("body")
        require(
            re.findall(r"(?m)^\s+uses:\s*(\S+)\s*$", upload_body)
            == ["actions/upload-artifact@v4"],
            "workflow-evidence-upload-owner",
            "canonical evidence step must own the upload-artifact action",
        )
        require(
            re.search(r"(?m)^\s+if:\s*always\(\)\s*$", upload_body) is not None,
            "workflow-evidence-upload-always",
            "Ruff evidence upload must run with if: always()",
        )
        require(
            re.search(
                r"(?m)^\s+path:\s*\$\{\{\s*runner\.temp\s*\}\}"
                r"/t63-ruff-evidence\s*$",
                upload_body,
            )
            is not None,
            "workflow-evidence-upload-path",
            "Ruff evidence upload must use the gate evidence directory",
        )
        require(
            re.search(
                r"(?m)^\s+if-no-files-found:\s*error\s*$",
                upload_body,
            )
            is not None,
            "workflow-evidence-upload-fail-closed",
            "Ruff evidence upload must fail when evidence is missing",
        )

    checkout = re.search(
        r"(?ms)^      - uses: actions/checkout@v4\s*$\n"
        r"(?P<body>.*?)(?=^      - |\Z)",
        lint,
    )
    checkout_body = checkout.group("body") if checkout else ""
    require(
        checkout is not None
        and re.search(
            r"(?m)^\s+(?:if|shell|working-directory):\s*",
            checkout_body,
        )
        is None,
        "workflow-checkout-step-control",
        "canonical checkout cannot be disabled or redirected",
    )
    require(
        checkout is not None and "fetch-depth: 0" in checkout_body,
        "workflow-checkout-depth",
        "lint checkout must use fetch-depth: 0",
    )
    require(
        checkout is not None
        and re.search(
            r"(?m)^\s*ref:\s*\$\{\{\s*"
            r"github\.event\.pull_request\.head\.sha\s*\|\|\s*github\.sha\s*\}\}\s*$",
            checkout_body,
        )
        is not None,
        "workflow-pr-head",
        "lint checkout must explicitly use the PR head SHA with the push fallback",
    )
    require(
        re.search(
            rf"pip\s+install[^\n]*['\"]?ruff=={re.escape(RUFF_VERSION)}['\"]?",
            lint,
        )
        is not None,
        "workflow-ruff-pin",
        f"lint job must install exact ruff=={RUFF_VERSION}",
    )
    require(
        f"tomli=={TOMLI_VERSION}" in lint,
        "workflow-tomli-pin",
        f"Python 3.10 lint job must install exact tomli=={TOMLI_VERSION}",
    )

    run_values = [
        match.group("value").strip()
        for match in re.finditer(
            r"(?m)^\s+run:\s*(?P<value>[^\n]*)$",
            lint,
        )
    ]
    require(
        run_values == [CI_INSTALL_COMMAND, CI_GATE_COMMAND],
        "workflow-shared-gate",
        "lint job run steps must be the exact pinned install and shared gate commands",
    )
    if "continue-on-error" in lint or "|| true" in lint:
        issues.append(
            finding(
                "workflow-error-bypass",
                relative,
                "lint job cannot downgrade gate failures",
            )
        )
    if re.search(
        r"(?mi)^\s*(?:-\s*)?(?:run:\s*)?(?:python(?:3)?\s+-m\s+)?ruff\s+(?:check|format)\b",
        lint,
    ):
        issues.append(
            finding(
                "workflow-direct-ruff",
                relative,
                "lint job must not invoke Ruff outside the shared gate",
            )
        )

    for event in ("push", "pull_request"):
        event_body = workflow_event_body(workflow, event)
        require(
            event_body is not None and RELEASE_BRANCH in event_body,
            "workflow-release-trigger",
            f"workflow {event} trigger must include {RELEASE_BRANCH}",
        )
    return issues


def write_result(output: Path, issues: list[dict[str, object]]) -> None:
    """Write the fixed policy schema deterministically, including on failure."""
    ordered = sorted(
        issues,
        key=lambda item: (
            str(item.get("path", "")),
            int(item.get("line", 0)),
            str(item.get("rule", "")),
            str(item.get("message", "")),
        ),
    )
    payload = {
        "schema": POLICY_SCHEMA,
        "status": "FAIL" if ordered else "PASS",
        "findings": ordered,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--changed-manifest", required=True)
    parser.add_argument("--gate-script", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output = Path(args.output_json).resolve()
    issues: list[dict[str, object]] = []
    try:
        repo_root = Path(args.repo_root).resolve(strict=True)
        if not repo_root.is_dir():
            raise ValueError(f"repository root is not a directory: {repo_root}")
        manifest = Path(args.changed_manifest).resolve()
        gate = resolve_repo_input(repo_root, args.gate_script, "gate script")
        workflow = resolve_repo_input(repo_root, args.workflow, "workflow")
        issues.extend(audit_changed_sources(repo_root, manifest))
        issues.extend(audit_ruff_configs(repo_root))
        issues.extend(audit_gate(repo_root, gate))
        issues.extend(audit_workflow(repo_root, workflow))
    except (OSError, RuntimeError, ValueError) as error:
        issues.append(finding("scanner-error", ".", str(error)))

    try:
        write_result(output, issues)
    except OSError as error:
        print(f"cannot write policy result {output}: {error}", file=sys.stderr)
        return 2
    if issues:
        print(
            f"T6.3 Ruff policy scanner: FAIL ({len(issues)} finding(s))",
            file=sys.stderr,
        )
        return 1
    print("T6.3 Ruff policy scanner: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
