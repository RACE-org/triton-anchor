"""Executable policy regression tests for the Triton 3.6 Ruff release gate."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import runpy
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_RELATIVE = Path("scripts/run_t63_ruff_gates.sh")
SCANNER_RELATIVE = Path("scripts/check_t63_ruff_policy.py")
WORKFLOW_RELATIVE = Path(".github/workflows/ci.yml")
GATE_SCRIPT = REPO_ROOT / GATE_RELATIVE
POLICY_SCANNER = REPO_ROOT / SCANNER_RELATIVE
CI_WORKFLOW = REPO_ROOT / WORKFLOW_RELATIVE

RUFF_VERSION = "0.15.22"
T63_BASE_SHA = "9ceba3e222fd2c9af9cbaf6a440beb23911c197b"
POLICY_SCHEMA = "triton-anchor-t63-ruff-policy-v1"
GATE_CONTEXT_SCHEMA = "triton-anchor-t63-ruff-gate-context-v1"
GATE_EXIT_SCHEMA = "triton-anchor-t63-ruff-gate-exits-v1"
EXPECTED_MANIFESTS = {
    "ALL_PYTHON_FILES": b"all.py\0changed.py\0",
    "T63_CHANGED_PYTHON_FILES": b"changed.py\0",
    "T63_DELETED_PYTHON_FILES": b"",
}
RUFF_CRITICAL_ARGS = (
    "check",
    "--isolated",
    "--no-cache",
    "--no-fix",
    "--target-version",
    "py310",
    "--select",
    "E9,F63,F7,F82",
)
RUFF_CHANGED_ARGS = (
    "check",
    "--isolated",
    "--no-cache",
    "--no-fix",
    "--target-version",
    "py310",
)
RUFF_FORMAT_ARGS = (
    "format",
    "--isolated",
    "--no-cache",
    "--target-version",
    "py310",
    "--check",
)


def _required_text(path: Path) -> str:
    assert path.is_file(), f"required release-gate implementation is missing: {path}"
    return path.read_text(encoding="utf-8")


def _tracked_path(relative: Path) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _flatten_shell(source: str) -> str:
    return re.sub(r"\\\n\s*", " ", source)


def _bash_array(source: str, name: str) -> tuple[str, ...]:
    match = re.search(
        rf"(?ms)^\s*(?:(?:readonly|declare)\s+(?:-[A-Za-z]+\s+)?)?"
        rf"{re.escape(name)}\s*=\s*\((.*?)^\s*\)",
        source,
    )
    assert match is not None, f"{name} must be a literal Bash array"
    return tuple(shlex.split(match.group(1), comments=True, posix=True))


def _lint_job(workflow: str) -> str:
    match = re.search(
        r"(?ms)^  lint:\s*$\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\s*$)",
        workflow,
    )
    assert match is not None, "CI must retain a dedicated lint job"
    return match.group("body")


def _assert_release_branch_triggers(workflow: str) -> None:
    trigger = re.search(
        r"(?ms)^on:\s*$\n(?P<body>.*?)(?=^[A-Za-z_][A-Za-z0-9_-]*:\s*$)",
        workflow,
    )
    assert trigger is not None, "workflow must have a top-level on: trigger"
    for event in ("push", "pull_request"):
        event_block = re.search(
            rf"(?ms)^  {re.escape(event)}:\s*$\n"
            r"(?P<body>.*?)(?=^  [A-Za-z_][A-Za-z0-9_-]*:\s*$|\Z)",
            trigger.group("body"),
        )
        assert event_block is not None, f"workflow trigger is missing {event}"
        assert "triton_v3.6" in event_block.group("body"), (
            f"{event} trigger must include triton_v3.6"
        )


def _assert_ci_contract(workflow: str) -> None:
    lint = _lint_job(workflow)

    assert "fetch-depth: 0" in lint
    assert re.search(
        r"(?m)^\s*ref:\s*\$\{\{[^}\n]*github\.event\.pull_request\.head\.sha",
        lint,
    ), "pull requests must checkout github.event.pull_request.head.sha explicitly"
    assert re.search(
        rf"pip\s+install[^\n]*['\"]?ruff=={re.escape(RUFF_VERSION)}['\"]?",
        lint,
    )
    assert f"bash {GATE_RELATIVE.as_posix()}" in lint
    assert "continue-on-error" not in lint
    assert not re.search(
        rf"bash\s+{re.escape(GATE_RELATIVE.as_posix())}[^\n]*"
        r"\|\|\s*(?:true\b|:\s*(?:#.*)?$|echo\b)",
        lint,
        re.MULTILINE,
    )
    assert not re.search(
        r"(?mi)^\s*(?:(?:-\s*)?run:\s*)?"
        r"(?:"
        r"(?:env(?:\s+[A-Za-z_][A-Za-z0-9_]*=[^\s]+)*\s+)?"
        r"(?:python(?:3)?\s+-m\s+)?ruff\s+(?:check|format)\b"
        r"|bash\s+-c\s+['\"][^'\"\n]*\bruff\s+(?:check|format)\b"
        r")",
        lint,
    )
    assert "T63_RUFF_EVIDENCE_DIR" in lint
    assert "actions/upload-artifact@v4" in lint


def _with_extra_ci_run_step(workflow: str, command: str) -> str:
    lint = _lint_job(workflow)
    step = re.search(r"(?m)^(?P<indent>\s*)-\s+(?:name|uses|run):", lint)
    assert step is not None, "lint job must contain a YAML step list"
    indent = step.group("indent")
    trailing_newlines = lint[len(lint.rstrip("\n")) :]
    mutated_lint = (
        lint.rstrip("\n")
        + f"\n{indent}- name: forbidden direct Ruff fixture"
        + f"\n{indent}  run: {command}\n"
        + trailing_newlines
    )
    mutated = workflow.replace(lint, mutated_lint, 1)
    assert f"bash {GATE_RELATIVE.as_posix()}" in _lint_job(mutated)
    assert command in _lint_job(mutated)
    return mutated


def _with_extra_ci_multiline_run_step(workflow: str, command: str) -> str:
    lint = _lint_job(workflow)
    step = re.search(r"(?m)^(?P<indent>\s*)-\s+(?:name|uses|run):", lint)
    assert step is not None, "lint job must contain a YAML step list"
    indent = step.group("indent")
    trailing_newlines = lint[len(lint.rstrip("\n")) :]
    mutated_lint = (
        lint.rstrip("\n")
        + f"\n{indent}- name: forbidden multiline Ruff fixture"
        + f"\n{indent}  run: |"
        + f"\n{indent}    {command}\n"
        + trailing_newlines
    )
    mutated = workflow.replace(lint, mutated_lint, 1)
    assert f"bash {GATE_RELATIVE.as_posix()}" in _lint_job(mutated)
    assert command in _lint_job(mutated)
    return mutated


def _with_shared_gate_bypass(workflow: str, suffix: str) -> str:
    lint = _lint_job(workflow)
    invocation = f"bash {GATE_RELATIVE.as_posix()}"
    mutated_lint = lint.replace(invocation, f"{invocation} {suffix}", 1)
    assert mutated_lint != lint, "fixture requires the shared gate invocation"
    return workflow.replace(lint, mutated_lint, 1)


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_shared_gate_and_scanner_are_tracked_executables() -> None:
    for relative, path in (
        (GATE_RELATIVE, GATE_SCRIPT),
        (SCANNER_RELATIVE, POLICY_SCANNER),
    ):
        assert path.is_file(), (
            f"required shared policy component is missing: {relative}"
        )
        assert _tracked_path(relative), f"policy component is not tracked: {relative}"
        assert path.stat().st_mode & stat.S_IXUSR, (
            f"policy component is not owner-executable: {relative}"
        )


def test_gate_is_valid_bash_and_safe_to_source(tmp_path: Path) -> None:
    _required_text(GATE_SCRIPT)
    syntax = subprocess.run(
        ["bash", "-n", str(GATE_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    marker = tmp_path / "unexpected-tool-call"
    fake_bin = tmp_path / "bin"
    fake_source = '#!/bin/sh\n: > "$T63_FAKE_TOOL_MARKER"\nexit 97\n'
    for name in ("git", "ruff", "python", "python3"):
        _write_executable(fake_bin / name, fake_source)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "T63_FAKE_TOOL_MARKER": str(marker),
        }
    )
    sourced = subprocess.run(
        [
            "bash",
            "-c",
            (
                'set -euo pipefail; source "$1"; declare -F main; '
                "declare -F run_changed_gates"
            ),
            "bash",
            str(GATE_SCRIPT),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert sourced.returncode == 0, sourced.stderr
    assert "main" in sourced.stdout
    assert "run_changed_gates" in sourced.stdout
    assert not marker.exists(), "sourcing the gate executed Git, Ruff, or the scanner"


def test_run_changed_gates_empty_array_is_explicit_pass_without_invocation(
    tmp_path: Path,
) -> None:
    _required_text(GATE_SCRIPT)
    marker = tmp_path / "unexpected-ruff-call"
    fake_bin = tmp_path / "bin"
    fake_source = '#!/bin/sh\n: > "$T63_FAKE_TOOL_MARKER"\nexit 97\n'
    for name in ("ruff", "python", "python3"):
        _write_executable(fake_bin / name, fake_source)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "T63_FAKE_TOOL_MARKER": str(marker),
        }
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                'set -euo pipefail; source "$1"; EMPTY_CHANGED=(); '
                "run_changed_gates EMPTY_CHANGED"
            ),
            "bash",
            str(GATE_SCRIPT),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lowered = completed.stdout.lower()
    assert re.search(r"changed[^\n]*count=0", lowered)
    assert re.search(
        r"changed[_ -]full[^\n]*status=pass[^\n]*(?:not[_ -]invoked|invoked=false)",
        lowered,
    )
    assert re.search(
        r"changed[_ -]format[^\n]*status=pass[^\n]*(?:not[_ -]invoked|invoked=false)",
        lowered,
    )
    assert not marker.exists(), "empty changed-file gates invoked Ruff"


def test_gate_uses_fixed_base_and_nul_safe_python_manifests() -> None:
    source = _required_text(GATE_SCRIPT)
    flat = _flatten_shell(source)

    assert re.search(
        rf"(?m)^\s*(?:readonly\s+)?T63_BASE_SHA=(?:['\"])?{T63_BASE_SHA}(?:['\"])?\s*$",
        source,
    )
    assert "git ls-files -z -- '*.py' '*.pyi'" in flat
    assert "git diff" in flat
    assert "--diff-filter=ACM" in flat
    assert "--name-only -z" in flat
    assert "--no-renames" in flat
    assert "LC_ALL=C sort -z" in flat
    assert "count" in source.lower()
    assert "sha256" in source.lower()
    assert (
        len(re.findall(r"\b(?:mapfile|readarray)\b[^\n]*\s-d\s+['\"]?['\"]?", source))
        >= 2
    )
    for name in (
        "ALL_PYTHON_FILES",
        "T63_CHANGED_PYTHON_FILES",
        "T63_DELETED_PYTHON_FILES",
    ):
        assert name in source


def test_gate_binds_candidate_to_clean_head_tree_before_ruff() -> None:
    source = _required_text(GATE_SCRIPT)
    flat = _flatten_shell(source)

    required_checks = (
        "PRODUCTION_CANDIDATE_SHA",
        "git rev-parse HEAD",
        "git write-tree",
        "HEAD^{tree}",
        "git merge-base --is-ancestor",
        "git diff --quiet --",
        "git diff --cached --quiet --",
    )
    for check in required_checks:
        assert check in flat

    first_gate_call = source.find('"${RUFF_CRITICAL_ARGS[@]}" --')
    assert first_gate_call >= 0, "critical Ruff gate invocation is missing"
    for check in ("git write-tree", "git merge-base --is-ancestor"):
        assert source.find(check) < first_gate_call, (
            f"candidate check must run before Ruff: {check}"
        )


def test_gate_uses_exact_non_mutating_ruff_command_domains() -> None:
    source = _required_text(GATE_SCRIPT)
    flat = _flatten_shell(source)

    assert re.search(
        rf"(?m)^\s*(?:readonly\s+)?RUFF_VERSION=(?:['\"])?{re.escape(RUFF_VERSION)}(?:['\"])?\s*$",
        source,
    )
    assert _bash_array(source, "RUFF_CRITICAL_ARGS") == RUFF_CRITICAL_ARGS
    assert _bash_array(source, "RUFF_CHANGED_ARGS") == RUFF_CHANGED_ARGS
    assert _bash_array(source, "RUFF_FORMAT_ARGS") == RUFF_FORMAT_ARGS
    assert re.search(r"\bruff\s+--version\b", flat)
    assert '"${RUFF_CRITICAL_ARGS[@]}" -- "${ALL_PYTHON_FILES[@]}"' in flat
    assert '"${RUFF_CHANGED_ARGS[@]}" -- "${T63_CHANGED_PYTHON_FILES[@]}"' in flat
    assert '"${RUFF_FORMAT_ARGS[@]}" -- "${T63_CHANGED_PYTHON_FILES[@]}"' in flat


def test_gate_rejects_python_deletion_or_rename_away() -> None:
    source = _required_text(GATE_SCRIPT)
    flat = _flatten_shell(source)

    assert "--diff-filter=D" in flat
    assert re.search(
        r"\$\{#T63_DELETED_PYTHON_FILES\[@\]\}\s*(?:(?:-ne|!=)|>\s*0)\s*0?",
        flat,
    )
    deletion_check = flat.find("${#T63_DELETED_PYTHON_FILES[@]}")
    first_gate_call = flat.find('"${RUFF_CRITICAL_ARGS[@]}" --')
    assert 0 <= deletion_check < first_gate_call
    assert re.search(
        r"T63_DELETED_PYTHON_FILES.{0,500}(?:return|exit)\s+[1-9]",
        flat,
    )


def test_gate_aggregates_all_four_hard_statuses_without_bypass() -> None:
    source = _required_text(GATE_SCRIPT)
    lowered = source.lower()

    assert "|| true" not in source
    assert "continue-on-error" not in lowered
    assert "--unsafe-fixes" not in source
    assert re.search(r"(?<!no-)--fix(?:\s|['\"]|$)", source) is None
    assert SCANNER_RELATIVE.as_posix() in source
    for status_name in (
        "RUFF_CRITICAL_EXIT",
        "RUFF_CHANGED_EXIT",
        "RUFF_FORMAT_EXIT",
        "RUFF_POLICY_EXIT",
    ):
        assert status_name in source
    assert re.search(r"(?:return|exit)\s+[1-9]", source)


def test_ci_pins_ruff_checks_pr_head_and_runs_shared_gate_as_hard_failure() -> None:
    workflow = _required_text(CI_WORKFLOW)
    _assert_ci_contract(workflow)


def test_ci_push_and_pull_request_triggers_cover_triton_v36() -> None:
    _assert_release_branch_triggers(_required_text(CI_WORKFLOW))


@pytest.mark.parametrize(
    "direct_command",
    (
        "ruff check .",
        "python -m ruff check .",
        "ruff format --check .",
        "env ruff check .",
        "bash -c 'ruff check .'",
    ),
)
def test_ci_contract_rejects_direct_ruff_commands(direct_command: str) -> None:
    workflow = _required_text(CI_WORKFLOW)
    mutated = _with_extra_ci_run_step(workflow, direct_command)
    with pytest.raises(AssertionError):
        _assert_ci_contract(mutated)


def test_ci_contract_rejects_multiline_env_ruff_command() -> None:
    workflow = _required_text(CI_WORKFLOW)
    mutated = _with_extra_ci_multiline_run_step(workflow, "env ruff check .")
    with pytest.raises(AssertionError):
        _assert_ci_contract(mutated)


@pytest.mark.parametrize("suffix", ("|| :", "|| echo ignored"))
def test_ci_contract_rejects_shared_gate_failure_bypasses(suffix: str) -> None:
    workflow = _required_text(CI_WORKFLOW)
    mutated = _with_shared_gate_bypass(workflow, suffix)
    with pytest.raises(AssertionError):
        _assert_ci_contract(mutated)


@pytest.mark.parametrize(
    "mutation",
    ("synthetic-head", "missing-ref", "shallow-checkout"),
)
def test_ci_contract_rejects_pr_checkout_identity_mutations(mutation: str) -> None:
    workflow = _required_text(CI_WORKFLOW)
    lint = _lint_job(workflow)
    if mutation == "synthetic-head":
        mutated_lint = lint.replace(
            "github.event.pull_request.head.sha",
            "github.sha",
            1,
        )
    elif mutation == "missing-ref":
        mutated_lint = re.sub(r"(?m)^\s*ref:.*\n", "", lint, count=1)
    else:
        mutated_lint = lint.replace("fetch-depth: 0", "fetch-depth: 1", 1)
    assert mutated_lint != lint, f"fixture could not apply {mutation} mutation"
    mutated = workflow.replace(lint, mutated_lint, 1)
    with pytest.raises(AssertionError):
        _assert_ci_contract(mutated)


def _fake_git_source() -> str:
    return (
        f"#!{sys.executable}\n"
        + r"""
import os
import sys


args = sys.argv[1:]
base = os.environ["FAKE_BASE_SHA"]
candidate = os.environ["FAKE_CANDIDATE_SHA"]
tree = os.environ["FAKE_TREE_SHA"]
dirty = os.environ.get("FAKE_GIT_DIRTY", "")


def has_filter(value):
    joined = " ".join(args)
    return f"--diff-filter={value}" in args or f"--diff-filter {value}" in joined


if not args:
    raise SystemExit(2)
if args[0] == "rev-parse":
    if "--show-toplevel" in args:
        print(os.environ["FAKE_REPO_ROOT"])
    elif any("tree" in argument for argument in args):
        print(tree)
    elif any(base in argument for argument in args):
        print(base)
    else:
        print(candidate)
    raise SystemExit(0)
if args[0] == "write-tree":
    print("3" * 40 if dirty == "tree" else tree)
    raise SystemExit(0)
if args[0] == "merge-base":
    raise SystemExit(1 if dirty == "ancestry" else 0)
if args[0] == "ls-files":
    if "-z" in args:
        sys.stdout.buffer.write(b"all.py\0changed.py\0")
    raise SystemExit(0)
if args[0] == "diff":
    if "--quiet" in args:
        cached = "--cached" in args
        if dirty == "cached" and cached:
            raise SystemExit(1)
        if dirty == "worktree" and not cached:
            raise SystemExit(1)
        raise SystemExit(0)
    if has_filter("ACM"):
        sys.stdout.buffer.write(b"changed.py\0")
    elif has_filter("D"):
        sys.stdout.buffer.write(b"")
    raise SystemExit(0)
if args[0] in {"status", "hash-object"}:
    raise SystemExit(0)
print("unexpected fake git argv: " + repr(args), file=sys.stderr)
raise SystemExit(2)
""".lstrip()
    )


def _fake_ruff_and_scanner_source() -> str:
    real_python = str(Path(sys.executable).resolve())
    return (
        f"#!{real_python}\nREAL_PYTHON = {real_python!r}\n"
        + r"""
import json
import os
import sys
from pathlib import Path


SCANNER_NAME = "check_t63_ruff_policy.py"


def record(tool, stage, arguments):
    with open(os.environ["FAKE_CALL_LOG"], "a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"tool": tool, "stage": stage, "argv": list(arguments)},
                sort_keys=True,
            )
            + "\n"
        )


def run_ruff(arguments):
    arguments = list(arguments)
    if arguments == ["--version"]:
        record("ruff", "version", arguments)
        print("ruff " + os.environ.get("FAKE_RUFF_VERSION", "0.15.22"))
        raise SystemExit(0)
    if arguments and arguments[0] == "check":
        stage = "critical" if "--select" in arguments else "changed"
    elif arguments and arguments[0] == "format":
        stage = "format"
    else:
        stage = "unknown"
    record("ruff", stage, arguments)
    print("[]")
    raise SystemExit(23 if os.environ.get("FAKE_FAIL_STAGE") == stage else 0)


def run_scanner(arguments):
    arguments = list(arguments)
    record("scanner", "scanner", arguments)
    status = "FAIL" if os.environ.get("FAKE_FAIL_STAGE") == "scanner" else "PASS"
    if "--output-json" in arguments:
        output = Path(arguments[arguments.index("--output-json") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema": "triton-anchor-t63-ruff-policy-v1",
                    "status": status,
                    "findings": [],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    raise SystemExit(29 if status == "FAIL" else 0)


program = Path(sys.argv[0]).name
arguments = sys.argv[1:]
if program in {"python", "python3"}:
    if arguments[:2] == ["-m", "ruff"]:
        run_ruff(arguments[2:])
    if arguments and Path(arguments[0]).name == SCANNER_NAME:
        run_scanner(arguments[1:])
    if arguments[:1] == ["-c"] and len(arguments) > 1 and "ruff" in arguments[1]:
        print("0.15.22")
        raise SystemExit(0)
    os.execv(REAL_PYTHON, [REAL_PYTHON, *arguments])
if program == "ruff":
    run_ruff(arguments)
if program == SCANNER_NAME:
    run_scanner(arguments)
raise SystemExit(2)
""".lstrip()
    )


def _initialize_full_gate_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    gate_source = _required_text(GATE_SCRIPT)
    workflow_source = _required_text(CI_WORKFLOW)
    fixture_root = tmp_path / "full-gate-repository"
    fake_bin = tmp_path / "fake-bin"
    call_log = tmp_path / "tool-calls.jsonl"
    evidence = tmp_path / "evidence"

    (fixture_root / GATE_RELATIVE.parent).mkdir(parents=True)
    (fixture_root / WORKFLOW_RELATIVE.parent).mkdir(parents=True)
    (fixture_root / GATE_RELATIVE).write_text(gate_source, encoding="utf-8")
    (fixture_root / WORKFLOW_RELATIVE).write_text(workflow_source, encoding="utf-8")
    (fixture_root / "all.py").write_text("ALL = 1\n", encoding="utf-8")
    (fixture_root / "changed.py").write_text("CHANGED = 1\n", encoding="utf-8")
    (fixture_root / GATE_RELATIVE).chmod(0o755)

    fake_tool_source = _fake_ruff_and_scanner_source()
    _write_executable(fake_bin / "git", _fake_git_source())
    for name in ("ruff", "python", "python3"):
        _write_executable(fake_bin / name, fake_tool_source)
    _write_executable(fixture_root / SCANNER_RELATIVE, fake_tool_source)

    candidate = "1" * 40
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "FAKE_CALL_LOG": str(call_log),
            "FAKE_BASE_SHA": T63_BASE_SHA,
            "FAKE_CANDIDATE_SHA": candidate,
            "FAKE_TREE_SHA": "2" * 40,
            "FAKE_REPO_ROOT": str(fixture_root),
            "FAKE_RUFF_VERSION": RUFF_VERSION,
            "PRODUCTION_CANDIDATE_SHA": candidate,
            "T63_RUFF_EVIDENCE_DIR": str(evidence),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return fixture_root, call_log, evidence, environment


def _run_full_gate(
    fixture_root: Path,
    call_log: Path,
    environment: dict[str, str],
    *,
    fail_stage: str = "",
    dirty: str = "",
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]]]:
    environment = environment.copy()
    environment["FAKE_FAIL_STAGE"] = "" if fail_stage == "wrong-version" else fail_stage
    environment["FAKE_RUFF_VERSION"] = (
        "0.16.4" if fail_stage == "wrong-version" else RUFF_VERSION
    )
    environment["FAKE_GIT_DIRTY"] = dirty
    completed = subprocess.run(
        ["bash", str(fixture_root / GATE_RELATIVE)],
        cwd=fixture_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    calls = []
    if call_log.is_file():
        calls = [
            json.loads(line)
            for line in call_log.read_text(encoding="utf-8").splitlines()
            if line
        ]
    return completed, calls


def _evidence_manifests(evidence: Path) -> dict[str, Path]:
    assert evidence.is_dir(), f"gate did not create evidence directory: {evidence}"
    nul_files = sorted(path for path in evidence.rglob("*.nul") if path.is_file())
    assert len(nul_files) == 3, (
        "evidence must contain exactly the all/changed/deleted NUL manifests; "
        f"found={nul_files!r}"
    )
    manifests = {}
    for name, token in (
        ("ALL_PYTHON_FILES", "all"),
        ("T63_CHANGED_PYTHON_FILES", "changed"),
        ("T63_DELETED_PYTHON_FILES", "deleted"),
    ):
        matches = [path for path in nul_files if token in path.name.lower()]
        assert len(matches) == 1, (
            f"evidence must identify exactly one {name} .nul file by name; "
            f"found={matches!r}"
        )
        manifests[name] = matches[0]
    assert len(set(manifests.values())) == 3
    return manifests


def _resolve_cli_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _assert_manifest_evidence(
    stdout: str,
    name: str,
    path: Path,
    expected: bytes,
) -> None:
    assert path.is_file(), f"missing persisted manifest: {path}"
    manifest = path.read_bytes()
    assert manifest == expected, (
        f"persisted {name} bytes differ: actual={manifest!r}, expected={expected!r}"
    )
    count = expected.count(b"\0")
    digest = hashlib.sha256(expected).hexdigest()
    assert manifest.count(b"\0") == count
    assert hashlib.sha256(manifest).hexdigest() == digest
    assert re.search(rf"(?i){re.escape(name)}[^\n]*count={count}\b", stdout)
    assert re.search(rf"(?i){re.escape(name)}[^\n]*sha256={digest}\b", stdout)


def _argv0(arguments: list[str]) -> bytes:
    return b"\0".join(os.fsencode(argument) for argument in arguments) + b"\0"


def _assert_gate_evidence_bundle(
    evidence: Path,
    fixture_root: Path,
    manifest_paths: dict[str, Path],
    scanner_output: Path,
    fail_stage: str,
) -> None:
    context_path = evidence / "gate-context.json"
    assert context_path.is_file()
    context = json.loads(context_path.read_text(encoding="utf-8"))
    expected_context = {
        "schema": GATE_CONTEXT_SCHEMA,
        "base_sha": T63_BASE_SHA,
        "candidate_sha": "1" * 40,
        "head_sha": "1" * 40,
        "head_tree_sha": "2" * 40,
        "index_tree_sha": "2" * 40,
        "base_is_ancestor": True,
        "tracked_worktree_clean": True,
        "cached_index_clean": True,
        "ruff_version": f"ruff {RUFF_VERSION}",
    }
    assert {key: context.get(key) for key in expected_context} == expected_context

    policy_argv = [
        "python3",
        str(fixture_root / SCANNER_RELATIVE),
        "--repo-root",
        str(fixture_root),
        "--changed-manifest",
        str(manifest_paths["T63_CHANGED_PYTHON_FILES"]),
        "--gate-script",
        GATE_RELATIVE.as_posix(),
        "--workflow",
        WORKFLOW_RELATIVE.as_posix(),
        "--output-json",
        str(scanner_output),
    ]
    expected_argv = {
        "critical": [
            "ruff",
            *RUFF_CRITICAL_ARGS,
            "--",
            "all.py",
            "changed.py",
        ],
        "changed": ["ruff", *RUFF_CHANGED_ARGS, "--", "changed.py"],
        "format": ["ruff", *RUFF_FORMAT_ARGS, "--", "changed.py"],
        "policy": policy_argv,
    }
    expected_raw = {
        "critical": b"[]\n",
        "changed": b"[]\n",
        "format": b"[]\n",
        "policy": b"",
    }
    expected_exits = {
        "critical": 23 if fail_stage == "critical" else 0,
        "changed": 23 if fail_stage == "changed" else 0,
        "format": 23 if fail_stage == "format" else 0,
        "policy": 29 if fail_stage == "scanner" else 0,
    }

    required_paths = {context_path, scanner_output, *manifest_paths.values()}
    for stage in ("critical", "changed", "format", "policy"):
        argv_path = evidence / f"{stage}.argv0"
        raw_path = evidence / f"{stage}.raw.log"
        required_paths.update((argv_path, raw_path))
        assert argv_path.read_bytes() == _argv0(expected_argv[stage])
        assert raw_path.read_bytes() == expected_raw[stage]

    exit_path = evidence / "gate-exit-summary.json"
    required_paths.add(exit_path)
    exit_summary = json.loads(exit_path.read_text(encoding="utf-8"))
    assert exit_summary.get("schema") == GATE_EXIT_SCHEMA
    assert exit_summary.get("exits") == expected_exits

    checksums_path = evidence / "SHA256SUMS"
    assert checksums_path.is_file()
    checksums = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        assert separator and re.fullmatch(r"[0-9a-f]{64}", digest)
        assert relative and not Path(relative).is_absolute()
        checksums[relative] = digest
    for path in required_paths:
        relative = path.relative_to(evidence).as_posix()
        assert checksums[relative] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "fail_stage",
    ("", "critical", "changed", "format", "scanner", "wrong-version"),
    ids=(
        "success",
        "critical-fails",
        "changed-fails",
        "format-fails",
        "scanner-fails",
        "wrong-version",
    ),
)
def test_full_gate_invokes_every_hard_gate_with_exact_argv_and_aggregates_failures(
    tmp_path: Path,
    fail_stage: str,
) -> None:
    fixture_root, call_log, evidence, environment = _initialize_full_gate_fixture(
        tmp_path
    )
    completed, calls = _run_full_gate(
        fixture_root,
        call_log,
        environment,
        fail_stage=fail_stage,
    )

    if fail_stage:
        assert completed.returncode != 0
    else:
        assert completed.returncode == 0, completed.stderr

    ruff_calls = [call for call in calls if call["tool"] == "ruff"]
    version_calls = [call for call in ruff_calls if call["stage"] == "version"]
    assert version_calls == [
        {"tool": "ruff", "stage": "version", "argv": ["--version"]}
    ]
    injected_version = "0.16.4" if fail_stage == "wrong-version" else RUFF_VERSION
    assert f"ruff {injected_version}" in f"{completed.stdout}\n{completed.stderr}"

    if fail_stage == "wrong-version":
        assert completed.returncode != 0
        return

    substantive = [call for call in ruff_calls if call["stage"] != "version"]
    assert [call["stage"] for call in substantive] == [
        "critical",
        "changed",
        "format",
    ]
    assert substantive[0]["argv"] == [
        *RUFF_CRITICAL_ARGS,
        "--",
        "all.py",
        "changed.py",
    ]
    assert substantive[1]["argv"] == [*RUFF_CHANGED_ARGS, "--", "changed.py"]
    assert substantive[2]["argv"] == [*RUFF_FORMAT_ARGS, "--", "changed.py"]
    scanner_calls = [call for call in calls if call["tool"] == "scanner"]
    assert len(scanner_calls) == 1
    manifest_paths = _evidence_manifests(evidence)
    for name, expected in EXPECTED_MANIFESTS.items():
        _assert_manifest_evidence(
            completed.stdout,
            name,
            manifest_paths[name],
            expected,
        )

    scanner_argv = scanner_calls[0]["argv"]
    assert isinstance(scanner_argv, list)
    assert len(scanner_argv) == 10
    assert scanner_argv[0] == "--repo-root"
    assert _resolve_cli_path(scanner_argv[1], fixture_root) == fixture_root.resolve()
    assert scanner_argv[2] == "--changed-manifest"
    assert (
        _resolve_cli_path(scanner_argv[3], fixture_root)
        == manifest_paths["T63_CHANGED_PYTHON_FILES"].resolve()
    )
    assert scanner_argv[4:8] == [
        "--gate-script",
        GATE_RELATIVE.as_posix(),
        "--workflow",
        WORKFLOW_RELATIVE.as_posix(),
    ]
    assert scanner_argv[8] == "--output-json"
    scanner_output = _resolve_cli_path(scanner_argv[9], fixture_root)
    assert scanner_output.is_relative_to(evidence.resolve())
    expected_status = "FAIL" if fail_stage == "scanner" else "PASS"
    expected_scanner_bytes = (
        json.dumps(
            {
                "schema": POLICY_SCHEMA,
                "status": expected_status,
                "findings": [],
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    assert scanner_output.is_file()
    assert scanner_output.read_bytes() == expected_scanner_bytes
    _assert_gate_evidence_bundle(
        evidence,
        fixture_root,
        manifest_paths,
        scanner_output,
        fail_stage,
    )


@pytest.mark.parametrize("dirty", ("worktree", "cached", "tree"))
def test_full_gate_rejects_non_candidate_state_before_tools(
    tmp_path: Path,
    dirty: str,
) -> None:
    fixture_root, call_log, _evidence, environment = _initialize_full_gate_fixture(
        tmp_path
    )
    completed, calls = _run_full_gate(
        fixture_root,
        call_log,
        environment,
        dirty=dirty,
    )

    assert completed.returncode != 0
    assert not [call for call in calls if call["tool"] in {"ruff", "scanner"}]


def test_manifest_sha256_failure_hard_fails_before_ruff_or_scanner(
    tmp_path: Path,
) -> None:
    fixture_root, call_log, _evidence, environment = _initialize_full_gate_fixture(
        tmp_path
    )
    fake_bin = Path(environment["PATH"].split(os.pathsep, 1)[0])
    _write_executable(
        fake_bin / "sha256sum",
        "#!/bin/sh\nprintf 'injected sha256sum failure\\n' >&2\nexit 71\n",
    )

    completed, calls = _run_full_gate(fixture_root, call_log, environment)

    assert completed.returncode != 0
    assert "injected sha256sum failure" in completed.stderr
    assert not [call for call in calls if call["tool"] in {"ruff", "scanner"}]


def _initialize_scanner_fixture(
    tmp_path: Path,
    *,
    python_source: str,
    python_relative: Path = Path("src/policy_fixture.py"),
    extra_files: dict[Path, str] | None = None,
    gate_suffix: str = "",
    workflow_override: str | None = None,
) -> tuple[Path, Path, Path]:
    gate_source = _required_text(GATE_SCRIPT) + gate_suffix
    scanner_source = _required_text(POLICY_SCANNER)
    workflow_source = (
        _required_text(CI_WORKFLOW) if workflow_override is None else workflow_override
    )

    fixture_root = tmp_path / "policy-repository"
    (fixture_root / GATE_RELATIVE.parent).mkdir(parents=True)
    (fixture_root / SCANNER_RELATIVE.parent).mkdir(parents=True, exist_ok=True)
    (fixture_root / WORKFLOW_RELATIVE.parent).mkdir(parents=True)
    (fixture_root / python_relative.parent).mkdir(parents=True, exist_ok=True)
    (fixture_root / GATE_RELATIVE).write_text(gate_source, encoding="utf-8")
    (fixture_root / SCANNER_RELATIVE).write_text(scanner_source, encoding="utf-8")
    (fixture_root / GATE_RELATIVE).chmod(0o755)
    (fixture_root / SCANNER_RELATIVE).chmod(0o755)
    (fixture_root / WORKFLOW_RELATIVE).write_text(workflow_source, encoding="utf-8")
    (fixture_root / python_relative).write_text(python_source, encoding="utf-8")
    for relative, content in (extra_files or {}).items():
        target = fixture_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=fixture_root, check=True)
    subprocess.run(["git", "add", "--all"], cwd=fixture_root, check=True)

    manifest = tmp_path / "changed-python-files.nul"
    manifest.write_bytes(os.fsencode(python_relative.as_posix()) + b"\0")
    output = tmp_path / "policy-result.json"
    return fixture_root, manifest, output


def _run_scanner(
    fixture_root: Path,
    manifest: Path,
    output: Path,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(fixture_root / SCANNER_RELATIVE),
            "--repo-root",
            str(fixture_root),
            "--changed-manifest",
            str(manifest),
            "--gate-script",
            GATE_RELATIVE.as_posix(),
            "--workflow",
            WORKFLOW_RELATIVE.as_posix(),
            "--output-json",
            str(output),
        ],
        cwd=fixture_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert output.is_file(), (
        "policy scanner must emit JSON even on failure; "
        f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == POLICY_SCHEMA
    assert isinstance(payload["findings"], list)
    return completed, payload


def _run_scanner_without_toml_parsers(
    fixture_root: Path,
    manifest: Path,
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, dict[str, object]]:
    namespace = runpy.run_path(
        str(fixture_root / SCANNER_RELATIVE),
        run_name="t63_policy_scanner_python310_fixture",
    )
    real_import = builtins.__import__

    def import_without_toml_parser(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name in {"tomllib", "tomli"}:
            raise ImportError(f"injected missing TOML parser: {name}")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_toml_parser)
    main = namespace["main"]
    try:
        return_code = main(
            [
                "--repo-root",
                str(fixture_root),
                "--changed-manifest",
                str(manifest),
                "--gate-script",
                GATE_RELATIVE.as_posix(),
                "--workflow",
                WORKFLOW_RELATIVE.as_posix(),
                "--output-json",
                str(output),
            ]
        )
    except ImportError as error:
        pytest.fail(f"missing TOML parser must fail closed with JSON: {error}")
    assert output.is_file(), "Python 3.10 parser failure must still emit policy JSON"
    payload = json.loads(output.read_text(encoding="utf-8"))
    return return_code, payload


def test_policy_scanner_is_nul_safe_and_uses_tokenize_comment_semantics(
    tmp_path: Path,
) -> None:
    relative = Path("src/name with spaces\nand newline.py")
    fixture_root, manifest, output = _initialize_scanner_fixture(
        tmp_path,
        python_relative=relative,
        python_source=(
            'marker = "# noqa: F401"\n'
            '# Prose mentioning "# ruff: noqa" is not a directive.\n'
            "value = 1\n"
        ),
    )
    completed, payload = _run_scanner(fixture_root, manifest, output)

    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == "PASS"
    assert payload["findings"] == []


def test_python310_toml_fallback_does_not_treat_tool_other_as_ruff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = Path("pyproject.toml")
    fixture_root, manifest, output = _initialize_scanner_fixture(
        tmp_path,
        python_source="value = 1\n",
        extra_files={relative: '[tool.other]\nignore = ["F401"]\n'},
    )
    return_code, payload = _run_scanner_without_toml_parsers(
        fixture_root,
        manifest,
        output,
        monkeypatch,
    )

    if return_code != 0:
        assert payload["status"] == "FAIL"
        assert any(
            finding.get("rule") == "ruff-config-parser-unavailable"
            for finding in payload["findings"]
        ), "a missing dependency may fail closed, but tool.other is not a Ruff bypass"
    else:
        assert payload == {
            "schema": POLICY_SCHEMA,
            "status": "PASS",
            "findings": [],
        }


@pytest.mark.parametrize(
    "content",
    (
        '[tool]\nruff.lint.ignore = ["F401"]\n',
        '[tool.ruff.lint]\nignore = ["F401"]\n',
        '[tool]\nruff = { lint = { ignore = ["F401"] } }\n',
    ),
    ids=("dotted-key", "ruff-table", "inline-table"),
)
def test_python310_toml_fallback_rejects_every_ruff_ignore_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    relative = Path("pyproject.toml")
    fixture_root, manifest, output = _initialize_scanner_fixture(
        tmp_path,
        python_source="value = 1\n",
        extra_files={relative: content},
    )
    return_code, payload = _run_scanner_without_toml_parsers(
        fixture_root,
        manifest,
        output,
        monkeypatch,
    )

    assert return_code != 0
    assert payload["status"] == "FAIL"
    assert any(
        finding.get("path") == relative.as_posix() for finding in payload["findings"]
    )


@pytest.mark.parametrize(
    "python_source",
    [
        "# ruff: noqa\nvalue = 1\n",
        "# ruff: noqa: F401\nvalue = 1\n",
        "# flake8: noqa\nvalue = 1\n",
        "# flake8: noqa: F401\nvalue = 1\n",
        "value = 1  # noqa\n",
        "value = 1  # noqa: ALL\n",
        "value = 1  # noqa: F401 - no exception is approved for this gate\n",
    ],
    ids=(
        "ruff-file-level",
        "ruff-file-level-code",
        "flake8-file-level",
        "flake8-file-level-code",
        "bare-inline",
        "all-inline",
        "unapproved-exact-inline",
    ),
)
def test_policy_scanner_rejects_unapproved_suppression_directives(
    tmp_path: Path,
    python_source: str,
) -> None:
    relative = Path("src/suppressed.py")
    fixture_root, manifest, output = _initialize_scanner_fixture(
        tmp_path,
        python_relative=relative,
        python_source=python_source,
    )
    completed, payload = _run_scanner(fixture_root, manifest, output)

    assert completed.returncode != 0
    assert payload["status"] == "FAIL"
    assert any(
        finding.get("path") == relative.as_posix()
        and isinstance(finding.get("line"), int)
        for finding in payload["findings"]
    )


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        (Path("ruff.toml"), '[lint]\nignore = ["F401"]\n'),
        (Path(".ruff.toml"), '[lint]\nextend-ignore = ["F401"]\n'),
        (
            Path("pyproject.toml"),
            '[tool.ruff.lint]\nper-file-ignores = {"*.py" = ["ALL"]}\n',
        ),
        (
            Path("pyproject.toml"),
            '[tool.ruff.lint]\nper-file-ignores = {"*.py" = ["F401"]}\n',
        ),
        (
            Path("pyproject.toml"),
            '[tool.ruff.lint]\nextend-per-file-ignores = {"*.py" = ["F401"]}\n',
        ),
        (
            Path("pyproject.toml"),
            '[tool.ruff.lint]\nextend-ignore = ["F401"]\n',
        ),
    ],
    ids=(
        "global-ignore",
        "dot-ruff-extend-ignore",
        "blanket-per-file-ignore",
        "wildcard-specific-code-per-file-ignore",
        "extend-per-file-ignores",
        "pyproject-extend-ignore",
    ),
)
def test_policy_scanner_rejects_tracked_ruff_config_bypasses(
    tmp_path: Path,
    relative: Path,
    content: str,
) -> None:
    fixture_root, manifest, output = _initialize_scanner_fixture(
        tmp_path,
        python_source="value = 1\n",
        extra_files={relative: content},
    )
    completed, payload = _run_scanner(fixture_root, manifest, output)

    assert completed.returncode != 0
    assert payload["status"] == "FAIL"
    assert any(
        finding.get("path") == relative.as_posix() for finding in payload["findings"]
    )


@pytest.mark.parametrize(
    "gate_suffix",
    [
        "\nruff check --ignore F401 -- .\n",
        "\nruff check --extend-ignore F401 -- .\n",
        "\nruff check --per-file-ignores '*.py:F401' -- .\n",
        "\nruff check --unsafe-fixes -- .\n",
        "\nruff check --fix -- .\n",
    ],
    ids=("ignore", "extend-ignore", "per-file-ignores", "unsafe-fixes", "fix"),
)
def test_policy_scanner_rejects_mutating_or_weakened_gate_commands(
    tmp_path: Path,
    gate_suffix: str,
) -> None:
    fixture_root, manifest, output = _initialize_scanner_fixture(
        tmp_path,
        python_source="value = 1\n",
        gate_suffix=gate_suffix,
    )
    completed, payload = _run_scanner(fixture_root, manifest, output)

    assert completed.returncode != 0
    assert payload["status"] == "FAIL"
    assert any(
        finding.get("path") == GATE_RELATIVE.as_posix()
        for finding in payload["findings"]
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "direct-ruff",
        "direct-python-ruff",
        "direct-env-ruff",
        "direct-bash-c-ruff",
        "direct-multiline-env-ruff",
        "gate-colon-bypass",
        "gate-echo-bypass",
        "synthetic-head",
        "missing-ref",
        "missing-release-trigger",
    ),
)
def test_policy_scanner_rejects_workflow_gate_or_head_bypasses(
    tmp_path: Path,
    mutation: str,
) -> None:
    workflow = _required_text(CI_WORKFLOW)
    lint = _lint_job(workflow)
    if mutation == "direct-ruff":
        mutated = _with_extra_ci_run_step(workflow, "ruff check .")
    elif mutation == "direct-python-ruff":
        mutated = _with_extra_ci_run_step(workflow, "python -m ruff check .")
    elif mutation == "direct-env-ruff":
        mutated = _with_extra_ci_run_step(workflow, "env ruff check .")
    elif mutation == "direct-bash-c-ruff":
        mutated = _with_extra_ci_run_step(workflow, "bash -c 'ruff check .'")
    elif mutation == "direct-multiline-env-ruff":
        mutated = _with_extra_ci_multiline_run_step(workflow, "env ruff check .")
    elif mutation == "gate-colon-bypass":
        mutated = _with_shared_gate_bypass(workflow, "|| :")
    elif mutation == "gate-echo-bypass":
        mutated = _with_shared_gate_bypass(workflow, "|| echo ignored")
    elif mutation == "synthetic-head":
        mutated_lint = lint.replace(
            "github.event.pull_request.head.sha",
            "github.sha",
            1,
        )
        assert mutated_lint != lint, f"fixture could not apply {mutation} mutation"
        mutated = workflow.replace(lint, mutated_lint, 1)
    elif mutation == "missing-ref":
        mutated_lint = re.sub(r"(?m)^\s*ref:.*\n", "", lint, count=1)
        assert mutated_lint != lint, f"fixture could not apply {mutation} mutation"
        mutated = workflow.replace(lint, mutated_lint, 1)
    else:
        assert workflow.count("triton_v3.6") >= 2, (
            "fixture requires release branch coverage for push and pull_request"
        )
        mutated = workflow.replace("triton_v3.6", "main")
    if mutation.startswith("direct-"):
        assert f"bash {GATE_RELATIVE.as_posix()}" in _lint_job(mutated)
    if mutation.startswith("gate-"):
        assert f"bash {GATE_RELATIVE.as_posix()}" in _lint_job(mutated)
    fixture_root, manifest, output = _initialize_scanner_fixture(
        tmp_path,
        python_source="value = 1\n",
        workflow_override=mutated,
    )
    completed, payload = _run_scanner(fixture_root, manifest, output)

    assert completed.returncode != 0
    assert payload["status"] == "FAIL"
    assert any(
        finding.get("path") == WORKFLOW_RELATIVE.as_posix()
        for finding in payload["findings"]
    )
