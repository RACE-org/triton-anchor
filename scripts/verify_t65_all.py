#!/usr/bin/env python3
"""Run the complete in-scope T6.5 acceptance gate.

The runner deliberately executes every Python command with ``-S`` and an
explicit, worktree-local ``PYTHONPATH``.  The development virtualenv contains
an editable-install finder for the main checkout; allowing that finder to run
would silently validate the wrong Triton version when this command is pointed
at the v3.0/v3.3/v3.6 worktrees.

By default, only the checkout containing this script is validated.  This makes
the same file safe to commit into each v3.0/v3.3/v3.6 worktree and run there.
Use ``--all-worktrees`` for an optional aggregate run from a parent checkout,
or ``--worktree`` to select an explicit set.

The gate covers only T6.5-owned evidence: AnchorIR tests, the repository smoke
tests, API/CLI equivalence, the human-readable ``verify_t65.py`` acceptance
demo, and the staged-diff whitespace check.  The repository-wide pytest
collection (which may collect FlagGems) and an out-of-tree backend compiler are
intentionally opt-in and are reported as outside this gate.

Each target must already have its canonical ``build/lib.*/triton/_C`` artifact
and an interpreter with pytest plus the console-script runtime prerequisite.
The runner validates current ``triton_anchor`` source against that selected
native build; it does not build Triton, create a wheel, or install dependencies.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKTREE_NAMES = (
    "triton-anchor-v3.0-t65",
    "triton-anchor-v3.3-t65",
    "triton-anchor-v3.6-t65",
)


@dataclasses.dataclass
class StepResult:
    target: str
    name: str
    status: str
    returncode: int | None
    duration_seconds: float
    output_tail: str = ""
    note: str = ""


@dataclasses.dataclass
class TargetContext:
    label: str
    path: Path
    python: Path
    env: dict[str, str]
    build_python_dir: Path | None
    launcher: Path
    launcher_dir: tempfile.TemporaryDirectory
    console_script_available: bool
    converter_available: bool | None = None
    blocked: bool = False


def _paint(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    code = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "bold": "1"}[color]
    return f"\033[{code}m{text}\033[0m"


def _label(path: Path) -> str:
    name = path.name
    if name.startswith("triton-anchor-v") and name.endswith("-t65"):
        prefix = "triton-anchor-"
        suffix = "-t65"
        return name[len(prefix) : -len(suffix)]
    return "current"


def _source_triton_version(target: Path) -> str:
    """Read the expected Triton version from the worktree, not its directory."""

    source = target / "triton" / "python" / "triton" / "__init__.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(
            "cannot read source Triton version from %s: %s" % (source, error)
        )
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            return node.value.value
    raise ValueError("source Triton version is missing from %s" % source)


def _discover_targets(args: argparse.Namespace) -> list[Path]:
    if args.worktree:
        paths = [Path(value).expanduser().resolve() for value in args.worktree]
    elif args.all_worktrees:
        candidates = [SCRIPT_ROOT.parent / name for name in DEFAULT_WORKTREE_NAMES]
        paths = [path for path in candidates if path.is_dir()]
        # If the script itself lives in a versioned worktree, include that
        # checkout even when its directory is not a sibling candidate (the
        # deduplication below handles the usual sibling case).
        if SCRIPT_ROOT.name in DEFAULT_WORKTREE_NAMES:
            paths.insert(0, SCRIPT_ROOT)
    else:
        paths = [SCRIPT_ROOT]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _find_python(args: argparse.Namespace, target: Path) -> Path:
    candidates: list[Path] = []
    if args.python:
        candidates.append(Path(args.python).expanduser())
    env_python = os.environ.get("T65_PYTHON")
    if env_python:
        candidates.append(Path(env_python).expanduser())
    active_venv = os.environ.get("VIRTUAL_ENV")
    if active_venv:
        candidates.append(Path(active_venv).expanduser() / "bin" / "python")
    candidates.extend(
        [
            target / ".venv" / "bin" / "python",
            SCRIPT_ROOT / ".venv" / "bin" / "python",
            # Local git-worktree setups commonly keep one dependency venv in
            # the primary checkout and no duplicate venv in versioned
            # worktrees.  The launcher/PYTHONPATH isolation below guarantees
            # that only the target worktree's Python packages and libtriton
            # are exercised even when this shared interpreter is selected.
            SCRIPT_ROOT.parent / "triton-anchor" / ".venv" / "bin" / "python",
            Path(sys.executable),
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            # Keep the venv launcher path intact.  ``Path.resolve()`` follows
            # ``.venv/bin/python`` to ``/usr/bin/python`` and silently drops
            # the venv's site-packages; the later ``-S`` invocations would
            # then be unable to import pytest.  An absolute, non-resolved
            # path still gives subprocess a stable cwd-independent launcher
            # while preserving the interpreter environment selected by the
            # caller.
            return candidate.absolute()
    raise RuntimeError("no usable Python interpreter found")


def _site_packages(python: Path) -> Path:
    completed = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot discover site-packages: " + completed.stderr.strip())
    site = Path(completed.stdout.strip())
    if not site.is_dir():
        raise RuntimeError(f"site-packages does not exist: {site}")
    return site


def _find_build_python_dir(target: Path) -> Path | None:
    def collect(pattern: str) -> list[Path]:
        return sorted(
            {
                path.parents[2]
                for path in target.glob(pattern)
                if path.is_file()
            },
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
            reverse=True,
        )

    # setup.py's canonical Python package output is authoritative.  Optional
    # scratch builds (for example ``build/ttgpu-t65-output``) may be newer but
    # contain only a partial overlay, so selecting solely by mtime can combine
    # the wrong Python sources and native library.
    candidates = collect("build/lib.*/triton/_C/libtriton.so")
    if not candidates:
        candidates = collect("build/**/triton/_C/libtriton.so")
    return candidates[0] if candidates else None


def _library_dirs(target: Path, build_python_dir: Path | None) -> list[Path]:
    # Keep native lookup tied to the selected canonical build.  Adding every
    # ``build/**/lib`` directory lets an older/default or optional TTGPU build
    # satisfy a dependency first and can produce a deceptively mixed binary
    # even when Python import paths look correct.
    dirs: list[Path] = []
    if build_python_dir is not None:
        dirs.extend(
            [
                build_python_dir / "triton" / "_C",
                build_python_dir / "triton" / "_C" / "lib",
            ]
        )
    return [path for path in dirs if path.is_dir()]


def _make_context(args: argparse.Namespace, target: Path) -> TargetContext:
    if not target.is_dir() or not (target / "setup.py").is_file():
        raise RuntimeError(f"not a triton-anchor checkout: {target}")
    python = _find_python(args, target)
    site = _site_packages(python)
    build_python_dir = _find_build_python_dir(target)
    # Tests and adapters occasionally spawn ``sys.executable`` themselves.
    # A plain ``-S`` on the parent command does not propagate to those child
    # processes, so a tiny launcher is used as the process' executable.  The
    # launcher always starts the selected interpreter with ``-S`` and
    # PYTHONEXECUTABLE makes CPython expose the launcher path as
    # ``sys.executable`` to descendants.  This prevents an editable-install
    # finder from importing the main checkout while validating a sibling
    # Triton worktree.
    launcher_dir = tempfile.TemporaryDirectory(prefix="t65-python-")
    launcher = Path(launcher_dir.name) / "python"
    launcher.write_text(
        "#!/bin/sh\nexec %s -S \"$@\"\n"
        % shlex.quote(str(python)),
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    # Expose only the current worktree's triton_anchor source ahead of the
    # canonical build.  Putting ``target/python`` first would also shadow the
    # selected build's ``triton`` package (and libtriton.so), while putting the
    # build first can silently test a stale build/lib copy of triton_anchor.
    source_overlay = Path(launcher_dir.name) / "source"
    source_overlay.mkdir()
    (source_overlay / "triton_anchor").symlink_to(
        target / "python" / "triton_anchor",
        target_is_directory=True,
    )

    # The CLI test intentionally verifies the selected environment's
    # installed console-script shape
    # by looking next to ``sys.executable``.  Mirror the *real* installed
    # entry point beside the temporary launcher so the isolation layer does
    # not turn a packaging check into a false failure.  Do not synthesize one:
    # a missing console script must remain a genuine packaging failure.
    installed_console = python.parent / "triton-anchor-validate"
    console_script = launcher.with_name("triton-anchor-validate")
    console_available = installed_console.is_file() and os.access(installed_console, os.X_OK)
    if console_available:
        console_script.symlink_to(installed_console)
    python_entries = [
        str(path)
        for path in (
            source_overlay,
            build_python_dir,
            target / "python",
            target / "triton" / "python",
            site,
            SCRIPT_ROOT / "scripts",
        )
        if path is not None and path.is_dir()
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "LC_ALL": "C.UTF-8",
            "T65_REPOSITORY_ROOT": str(target),
            "PYTHONEXECUTABLE": str(launcher),
            "PYTHONPATH": os.pathsep.join(python_entries),
        }
    )
    library_entries = [str(path) for path in _library_dirs(target, build_python_dir)]
    if library_entries:
        old = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            library_entries + ([old] if old else [])
        )
    return TargetContext(
        label=_label(target),
        path=target,
        python=python,
        env=environment,
        build_python_dir=build_python_dir,
        launcher=launcher,
        launcher_dir=launcher_dir,
        console_script_available=console_available,
    )


def _python_command(context: TargetContext, *arguments: str) -> list[str]:
    return [str(context.launcher), *arguments]


def _tail(output: str, lines: int = 40) -> str:
    values = output.rstrip().splitlines()
    return "\n".join(values[-lines:])


def _run(
    context: TargetContext,
    name: str,
    command: Sequence[str],
    results: list[StepResult],
    *,
    timeout: int,
    skip_reason: str | None = None,
    show_output: bool = False,
) -> StepResult:
    if skip_reason is not None:
        result = StepResult(context.label, name, "SKIP", None, 0.0, note=skip_reason)
        results.append(result)
        print(f"[{context.label}] [SKIP] {name}: {skip_reason}")
        return result

    started = time.monotonic()
    pretty = " ".join(shlex.quote(str(value)) for value in command)
    print(f"[{context.label}] [RUN ] {name}: {pretty}")
    full_output = ""
    try:
        completed = subprocess.run(
            list(command),
            cwd=context.path,
            env=context.env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        output = completed.stdout or ""
        full_output = output
        status = "PASS" if completed.returncode == 0 else "FAIL"
        result = StepResult(
            context.label,
            name,
            status,
            completed.returncode,
            time.monotonic() - started,
            _tail(output),
        )
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") if isinstance(error.stdout, str) else ""
        full_output = output
        result = StepResult(
            context.label,
            name,
            "FAIL",
            124,
            time.monotonic() - started,
            _tail(output),
            note=f"timed out after {timeout}s",
        )
    except OSError as error:
        result = StepResult(
            context.label,
            name,
            "FAIL",
            127,
            time.monotonic() - started,
            note=str(error),
        )
    results.append(result)
    marker = "PASS" if result.status == "PASS" else "FAIL"
    print(
        f"[{context.label}] [{marker}] {name} "
        f"({result.duration_seconds:.1f}s, rc={result.returncode})"
    )
    if show_output and full_output:
        print(full_output.rstrip())
    elif result.status == "FAIL" and result.output_tail:
        print(result.output_tail)
    return result


def _probe_violations(
    context: TargetContext, payload: dict[str, object]
) -> list[str]:
    """Return isolation/version violations for one completed import probe."""

    violations = []
    if context.build_python_dir is None:
        violations.append("no canonical build/lib.*/triton/_C/libtriton.so found")
    expected_roots = {
        "anchor_path": context.path / "python" / "triton_anchor",
        "triton_path": (
            context.build_python_dir / "triton"
            if context.build_python_dir is not None
            else context.path / "build" / "<missing>" / "triton"
        ),
        "libtriton_path": (
            context.build_python_dir / "triton" / "_C"
            if context.build_python_dir is not None
            else context.path / "build" / "<missing>" / "triton" / "_C"
        ),
    }
    for field, expected_root in expected_roots.items():
        value = payload.get(field, "")
        if not value:
            violations.append(f"{field} is empty")
            continue
        try:
            Path(str(value)).resolve().relative_to(expected_root.resolve())
        except (OSError, ValueError):
            violations.append(
                f"{field} is outside {expected_root}: {value or '<empty>'}"
            )

    expected_version = _source_triton_version(context.path)
    reported_version = str(payload.get("triton", ""))
    if reported_version != expected_version:
        violations.append(
            "reported Triton version %r does not match source version %r"
            % (reported_version, expected_version)
        )
    return violations


def _probe(context: TargetContext, results: list[StepResult], timeout: int) -> bool:
    script = r'''
import json
from triton._C import libtriton
from triton._C.libtriton import passes
import triton, triton_anchor
from triton_anchor import ANCHOR_IR_SPEC_VERSION
print(json.dumps({
    "triton": getattr(triton, "__version__", "unknown"),
    "triton_path": str(getattr(triton, "__file__", "")),
    "anchor_path": str(getattr(triton_anchor, "__file__", "")),
    "libtriton_path": str(getattr(libtriton, "__file__", "")),
    "spec_version": ANCHOR_IR_SPEC_VERSION,
    "ttgpu_converter": bool(getattr(getattr(passes, "ttir", None), "add_convert_to_ttgpuir", None)),
}, sort_keys=True))
'''
    result = _run(
        context,
        "environment/import probe",
        _python_command(context, "-c", script),
        results,
        timeout=timeout,
        show_output=True,
    )
    if result.status != "PASS":
        context.blocked = True
        return False
    try:
        # The final line is deterministic JSON; tolerate informational output
        # from an upstream import by reading the last non-empty line.
        payload = json.loads(result.output_tail.splitlines()[-1])
        context.converter_available = bool(payload["ttgpu_converter"])
        violations = _probe_violations(context, payload)
        if violations:
            result.status = "FAIL"
            result.returncode = 1
            result.note = "; ".join(violations)
            context.blocked = True
            print(f"[{context.label}] [FAIL] environment isolation: {result.note}")
            return False
        print(
            f"[{context.label}] spec={payload['spec_version']} "
            f"TTGPU-converter={'available' if context.converter_available else 'absent (explicit skip)'}"
        )
    except (IndexError, KeyError, ValueError, json.JSONDecodeError) as error:
        context.blocked = True
        results.append(
            StepResult(
                context.label,
                "environment/import probe parsing",
                "FAIL",
                1,
                0.0,
                note=str(error),
            )
        )
        return False
    return True


def _api_cli_probe(context: TargetContext, results: list[StepResult], timeout: int) -> None:
    script = r'''
import json, os, subprocess, sys
from pathlib import Path
from triton_anchor import AnchorIRPhase, AnchorIRTrack, StructuredAnchorIRValidator
from triton_anchor.anchor_ir_rules import ANCHOR_IR_SPEC_VERSION
root = Path(os.environ["T65_REPOSITORY_ROOT"])
source = root / "python" / "triton_anchor" / "tests" / "data" / "anchor_ir" / "linalg" / "negative" / "nested_op.mlir"
text = source.read_text(encoding="utf-8")
api = StructuredAnchorIRValidator().validate_text(
    text, spec_version=ANCHOR_IR_SPEC_VERSION, track=AnchorIRTrack.LINALG,
    phase=AnchorIRPhase.PRE_HOOK, source_name=str(source),
).to_dict()
command = [sys.executable, "-S", "-m", "triton_anchor.anchor_ir_cli", str(source),
           "--spec-version", ANCHOR_IR_SPEC_VERSION, "--track", "linalg",
           "--phase", "pre_hook", "--format", "json"]
completed = subprocess.run(command, check=False, capture_output=True, text=True)
assert completed.stdout, completed.stderr
cli = json.loads(completed.stdout)
assert completed.returncode == 1, completed.returncode
assert cli == api, "API/CLI JSON report mismatch"
valid = "module { func.func @ok() { func.return } }"
completed = subprocess.run(
    [sys.executable, "-S", "-m", "triton_anchor.anchor_ir_cli", "-",
     "--spec-version", ANCHOR_IR_SPEC_VERSION, "--track", "linalg",
     "--phase", "pre_hook", "--format", "json"],
    input=valid, check=False, capture_output=True, text=True,
)
assert completed.returncode == 0, completed.stdout + completed.stderr
print("API/CLI JSON equality and valid/invalid exit codes verified")
'''
    _run(
        context,
        "Python API ↔ CLI contract",
        _python_command(context, "-c", script),
        results,
        timeout=timeout,
        show_output=True,
    )


def _run_target(context: TargetContext, args: argparse.Namespace, results: list[StepResult]) -> None:
    print(f"\n{'=' * 88}\nTARGET {context.label}: {context.path}\n{'=' * 88}")
    _run(
        context,
        "git diff --check (tracked worktree)",
        ["git", "diff", "--check", "HEAD"],
        results,
        timeout=args.timeout,
    )
    _run(
        context,
        "git diff --cached --check",
        ["git", "diff", "--cached", "--check"],
        results,
        timeout=args.timeout,
    )
    ready = _probe(context, results, args.timeout)
    if not ready:
        for name in (
            "AnchorIR pytest suite",
            "smoke pytest",
            "smoke script",
            "Python API ↔ CLI contract",
            "visual verify_t65 acceptance",
        ):
            _run(context, name, [], results, timeout=args.timeout, skip_reason="environment probe failed")
        return

    if context.converter_available:
        _run(
            context,
            "real TTIR→TTGPU converter capability",
            ["true"],
            results,
            timeout=args.timeout,
        )
    else:
        _run(
            context,
            "real TTIR→TTGPU converter capability",
            [],
            results,
            timeout=args.timeout,
            skip_reason=(
                "converter is not exposed by this Triton branch; external "
                "backend/T10.3 owns this conversion boundary"
            ),
        )

    if context.build_python_dir is None:
        artifact = _run(
            context,
            "compiled libtriton artifact",
            ["test", "-f", str(context.path / "build" / "<missing-libtriton.so>")],
            results,
            timeout=args.timeout,
        )
        artifact.note = "no worktree build/lib*/triton/_C/libtriton.so found"
    else:
        _run(
            context,
            "compiled libtriton artifact",
            [
                "test",
                "-f",
                str(context.build_python_dir / "triton" / "_C" / "libtriton.so"),
            ],
            results,
            timeout=args.timeout,
        )
        print(f"[{context.label}] compiled extension: {context.build_python_dir}")

    console_entry = _run(
        context,
        "selected-environment triton-anchor-validate prerequisite",
        ["test", "-x", str(context.python.parent / "triton-anchor-validate")],
        results,
        timeout=args.timeout,
    )
    if not context.console_script_available:
        console_entry.note = (
            "selected interpreter has no triton-anchor-validate runtime prerequisite; "
            "current setup.py metadata is checked separately by the source tests"
        )

    _run(
        context,
        "AnchorIR pytest suite",
        _python_command(
            context,
            "-m",
            "pytest",
            "python/triton_anchor/tests",
            "-q",
            "--disable-warnings",
        ),
        results,
        timeout=args.timeout,
        show_output=args.verbose,
    )
    _run(
        context,
        "smoke pytest",
        _python_command(context, "-m", "pytest", "tests/test_smoke.py", "-q", "--disable-warnings"),
        results,
        timeout=args.timeout,
        show_output=args.verbose,
    )
    _run(
        context,
        "smoke script",
        _python_command(context, "tests/test_smoke.py"),
        results,
        timeout=args.timeout,
        show_output=args.verbose,
    )
    _api_cli_probe(context, results, args.timeout)
    visual_script = context.path / "scripts" / "verify_t65.py"
    if not visual_script.is_file():
        # Compatibility for existing local worktrees created before the
        # runner was added; once merged, each branch uses its own copy.
        visual_script = SCRIPT_ROOT / "scripts" / "verify_t65.py"
    visual = _run(
        context,
        "visual verify_t65 acceptance",
        _python_command(context, str(visual_script)),
        results,
        timeout=args.timeout,
        show_output=True,
    )
    if visual.status == "PASS" and context.converter_available is False:
        visual.note = "TTIR→TTGPU converter absent; verify_t65 emitted an explicit SKIP"

    if args.include_root_pytest:
        _run(
            context,
            "optional repository-wide pytest",
            _python_command(context, "-m", "pytest", "-q", "--disable-warnings"),
            results,
            timeout=args.timeout,
            show_output=args.verbose,
        )


def _print_summary(results: Sequence[StepResult], json_path: Path | None) -> int:
    print(f"\n{'=' * 88}\nT6.5 ONE-CLICK ACCEPTANCE SUMMARY\n{'=' * 88}")
    counts = {status: 0 for status in ("PASS", "FAIL", "SKIP")}
    for result in results:
        counts[result.status] += 1
        note = f" — {result.note}" if result.note else ""
        print(
            f"[{result.status:4}] {result.target:7} {result.name} "
            f"({result.duration_seconds:.1f}s){note}"
        )
        if result.status == "FAIL" and result.output_tail:
            print("  tail:")
            print("\n".join("    " + line for line in result.output_tail.splitlines()))
    print(
        "counts: PASS={PASS} FAIL={FAIL} SKIP={SKIP}".format(**counts)
    )
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "results": [dataclasses.asdict(result) for result in results],
                    "counts": counts,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print("summary json:", json_path)
    if counts["FAIL"]:
        print("FINAL: FAIL (see aggregated step tails above)")
        return 1
    print("FINAL: PASS (explicit capability skips are non-failing)")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worktree",
        action="append",
        help="Explicit worktree path; repeat to validate multiple versions.",
    )
    parser.add_argument(
        "--all-worktrees",
        action="store_true",
        help="Also discover sibling v3.0/v3.3/v3.6 worktrees (optional aggregate mode).",
    )
    parser.add_argument(
        "--python",
        help="Python interpreter shared by worktrees (defaults to T65_PYTHON, .venv, or current Python).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-step timeout in seconds (default: 900).",
    )
    parser.add_argument(
        "--include-root-pytest",
        action="store_true",
        help="Also run repository-wide pytest; may collect out-of-scope FlagGems tests.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full step output.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        help=(
            "Write the machine-readable aggregate report to this path "
            "(recommended: build/t65-summary.json)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = _parse_args(argv)
    if args.timeout <= 0:
        print("--timeout must be positive", file=sys.stderr)
        return 2
    if args.worktree and args.all_worktrees:
        print("--worktree and --all-worktrees are mutually exclusive", file=sys.stderr)
        return 2
    targets = _discover_targets(args)
    if not targets:
        print("no matching worktree was found", file=sys.stderr)
        return 2
    results: list[StepResult] = []
    print("T6.5 one-click acceptance runner")
    print("targets:", ", ".join(f"{_label(path)}={path}" for path in targets))
    print("mode:", "aggregate" if args.all_worktrees or args.worktree else "current worktree only")
    print("scope: AnchorIR tests + smoke + API/CLI + visual acceptance + staged diff")
    print("note: T10.3 corpus runner and external backend compiler are not invoked here")

    for target in targets:
        try:
            context = _make_context(args, target)
        except (OSError, RuntimeError) as error:
            label = _label(target)
            results.append(StepResult(label, "environment setup", "FAIL", 2, 0.0, note=str(error)))
            print(f"[{label}] [FAIL] environment setup: {error}")
            continue
        _run_target(context, args, results)
    return _print_summary(results, args.summary_json)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
