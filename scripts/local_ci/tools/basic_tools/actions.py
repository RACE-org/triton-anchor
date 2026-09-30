#!/usr/bin/env python3
"""Environment probes, wheel operations and measurement handling for basic tools."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("$ " + repr(argv), flush=True)
    return subprocess.run(argv, check=True, **kwargs)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def candidate_python(context: dict[str, Any]) -> str:
    """The host-selected task wrapper, never this trusted helper interpreter."""
    return context.get("python_bin", "/task/candidate/venv/bin/python")


def remove_child(root: Path, child: Path) -> None:
    """Bound rebuild cleanup to known direct children and reject linked paths."""
    root = root.resolve(strict=True)
    if child.is_symlink() or child.parent.resolve(strict=True) != root:
        raise ValueError(f"Refusing unsafe build cleanup: {child}")
    resolved = child.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing cleanup outside build source: {child}")
    if child.is_dir():
        shutil.rmtree(child)
    elif child.exists():
        child.unlink()


def stage_root(context: dict[str, Any], tool: str) -> Path:
    return Path(
        context.get("dependency_artifacts", {}).get(
            tool, str(Path(context["artifact_dir"]) / tool)
        )
    )


def environment_fingerprint(context: dict[str, Any]) -> str:
    if context.get("environment_fingerprint"):
        return context["environment_fingerprint"]
    manifest = stage_root(context, "environment") / "environment.json"
    return (
        read_json(manifest).get("environment_fingerprint", "")
        if manifest.is_file()
        else ""
    )


def wheel_manifest(
    context: dict[str, Any], build_tool: str
) -> tuple[dict[str, Any], Path]:
    root = stage_root(context, build_tool)
    manifest = read_json(root / "wheel.json")
    wheel = Path(manifest["wheel"])
    if (
        not wheel.is_file()
        or wheel.is_symlink()
        or root.resolve() not in wheel.resolve().parents
    ):
        raise ValueError(
            "Wheel path is missing or outside the current task build artifact directory"
        )
    if digest(wheel) != manifest["sha256"]:
        raise ValueError("Wheel artifact hash mismatch; rebuild before continuing")
    return manifest, wheel


def preflight(payload: dict[str, Any]) -> None:
    context, tool = payload["context"], payload["tool_id"]
    source = Path(context["source_dir"])
    if not source.is_dir():
        raise ValueError(f"Source checkout unavailable: {source}")
    actual = run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if actual != context["target_sha"]:
        raise ValueError(f"Checkout HEAD does not match task target: {actual}")
    artifact = Path(context["artifact_dir"])
    if (
        artifact.resolve() == source.resolve()
        or source.resolve() in artifact.resolve().parents
    ):
        raise ValueError("Artifacts must be outside the candidate source checkout")
    out = artifact / tool
    for directory in (out, out / "tmp", out / "cache", out / "dump"):
        directory.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps({"task_id": context["task_id"], "target_sha": actual, "tool": tool})
    )


def environment(payload: dict[str, Any]) -> None:
    context = payload["context"]
    config = context.get("profile", {}).get("tools", {})
    missing = []
    commands = {}
    for executable in config.get("required_commands", ["git", "cmake", "ninja"]):
        found = shutil.which(executable)
        commands[executable] = found
        if not found:
            missing.append(f"executable:{executable}")
    requested_modules = config.get(
        "required_modules", ["build", "setuptools", "wheel", "pybind11"]
    )
    probe = run(
        [
            candidate_python(context),
            "-I",
            "-c",
            "import importlib.util,json,sys; "
            "print(json.dumps({'python':sys.version,'python_executable':sys.executable,"
            "'modules':{name:importlib.util.find_spec(name) is not None for name in json.loads(sys.argv[1])}}))",
            json.dumps(requested_modules),
        ],
        capture_output=True,
        text=True,
    )
    observed = json.loads(probe.stdout)
    modules = observed["modules"]
    for module, present in modules.items():
        if not present:
            missing.append(f"python-module:{module}")
    source = Path(context["source_dir"])
    # Package and test-file requirements belong to their individual tools, so a
    # frontend-only source problem does not prevent diagnosing backend builds.
    for name in config.get("required_source_files", []):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("required_source_files must stay within the checkout")
        if not (source / relative).is_file():
            missing.append(f"source:{name}")
    llvm_version = None
    if config.get("llvm_dir"):
        llvm = Path(config["llvm_dir"])
        for name in ("include", "lib", "bin"):
            if not (llvm / name).is_dir():
                missing.append(f"LLVM:{llvm / name}")
        llvm_config = (
            llvm / "bin" / ("llvm-config.exe" if os.name == "nt" else "llvm-config")
        )
        if llvm_config.is_file():
            llvm_version = run(
                [str(llvm_config), "--version"], text=True, capture_output=True
            ).stdout.strip()
        else:
            missing.append(f"LLVM:{llvm_config}")
    result = {
        "python": observed["python"],
        "python_executable": observed["python_executable"],
        "commands": commands,
        "modules": modules,
        "llvm_version": llvm_version,
        "profile_llvm_revision": context.get("profile", {}).get("llvm_revision"),
        "missing": missing,
        "target_sha": context["target_sha"],
        "task_id": context["task_id"],
    }
    observed_fingerprint = hashlib.sha256(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"target_sha", "task_id"}},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    result["observed_fingerprint"] = observed_fingerprint
    result["environment_fingerprint"] = (
        context.get("environment_fingerprint") or observed_fingerprint
    )
    write_json(
        Path(context["artifact_dir"]) / "environment" / "environment.json", result
    )
    print(json.dumps(result, ensure_ascii=False))
    if missing:
        raise RuntimeError("Environment prerequisites missing: " + ", ".join(missing))
    # A successful import discovery does not establish a consistent dependency set.
    run([candidate_python(context), "-m", "pip", "check"])


def prepare_build(payload: dict[str, Any]) -> None:
    source = Path(payload["build_source"])
    if not source.is_dir() or not any(
        (source / name).is_file() for name in ("pyproject.toml", "setup.py")
    ):
        raise ValueError(f"Build source is not an installable Python project: {source}")
    if payload["parameters"].get("build_mode", "fresh") == "fresh":
        remove_child(source, source / "build")
    remove_child(source, source / "dist")
    for child in source.glob("*.egg-info"):
        remove_child(source, child)
    out = Path(payload["context"]["artifact_dir"]) / payload["tool_id"]
    remove_child(out, out / "wheels")
    (out / "wheels").mkdir()
    # Invalidate the previous attempt before any new build starts.
    for name in ("wheel.json", "installation.json"):
        (out / name).unlink(missing_ok=True)


def record_wheel(payload: dict[str, Any]) -> None:
    context, tool = payload["context"], payload["tool_id"]
    out = Path(context["artifact_dir"]) / tool
    pattern = (
        "triton_anchor-*.whl"
        if tool == "frontend_build"
        else context["profile"]["tools"].get("backend_wheel_pattern")
    )
    if not pattern or "/" in pattern or "\\" in pattern:
        raise ValueError("A filename-only backend_wheel_pattern is required")
    wheels = list((out / "wheels").glob(pattern))
    if len(wheels) != 1:
        raise ValueError(
            f"Expected exactly one new wheel matching {pattern}; found {len(wheels)}"
        )
    wheel = wheels[0]
    if wheel.is_symlink():
        raise ValueError("Build returned a symlink instead of a wheel")
    write_json(
        out / "wheel.json",
        {
            "wheel": str(wheel.resolve()),
            "sha256": digest(wheel),
            "target_sha": context["target_sha"],
            "task_id": context["task_id"],
            "environment_fingerprint": environment_fingerprint(context),
        },
    )
    print(f"Built wheel: {wheel.name}; sha256={digest(wheel)}")


def install_wheel(payload: dict[str, Any]) -> None:
    context, tool = payload["context"], payload["tool_id"]
    build_tool = "backend_build" if tool == "backend_install" else "frontend_build"
    explicit = payload["parameters"].get("wheel")
    if explicit:
        wheel = Path(explicit).resolve(strict=True)
        if not wheel.is_file() or wheel.suffix != ".whl":
            raise ValueError("wheel must identify a wheel file")
        manifest = {"wheel": str(wheel), "sha256": digest(wheel)}
    else:
        manifest, wheel = wheel_manifest(context, build_tool)
    # Dependencies belong to the selected environment recipe. Installing the
    # candidate must not silently replace the matched LLVM/Triton/backend stack.
    run(
        [
            candidate_python(context),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheel),
        ]
    )
    write_json(
        Path(context["artifact_dir"]) / tool / "installation.json",
        {
            **manifest,
            "python_executable": candidate_python(context),
            "task_venv": context.get("task_venv"),
        },
    )


def import_identity(context: dict[str, Any]) -> dict[str, Any]:
    # Imports may print from Python or native initializers. Reserve one descriptor
    # for JSON so diagnostics cannot corrupt the import-origin report.
    program = r"""import os
fd = os.dup(1)
os.dup2(2, 1)
import hashlib, importlib.metadata as m, json, pathlib, sys, triton, triton_anchor
wheel = m.distribution('triton-anchor')
files = {pathlib.Path(wheel.locate_file(p)).resolve() for p in wheel.files or []}
imports = {}
for name in ('triton', 'triton_anchor', 'triton._C.libtriton'):
    module = sys.modules.get(name)
    if module is None or not getattr(module, '__file__', None): continue
    path = pathlib.Path(module.__file__).resolve()
    assert path in files, 'Import did not originate from the installed frontend wheel: ' + name
    imports[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
os.write(fd, (json.dumps({'distribution_version': wheel.version, 'imports': imports}) + '\n').encode())
os.close(fd)
"""
    result = run(
        [candidate_python(context), "-I", "-c", program],
        capture_output=True,
        text=True,
        cwd=Path(context["artifact_dir"]),
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return json.loads(result.stdout)


def frontend_import(payload: dict[str, Any]) -> None:
    context = payload["context"]
    path = Path(context["artifact_dir"]) / "frontend_install" / "installation.json"
    installed = read_json(path)
    installed.update(import_identity(context))
    write_json(path, installed)


def backend_discovery(payload: dict[str, Any]) -> None:
    context = payload["context"]
    expected = context.get("profile", {}).get("tools", {}).get("expected_backend")
    if not expected:
        raise ValueError("profile.tools.expected_backend is required")
    # Pass the name as argv data; no Python code interpolation.
    run(
        [
            candidate_python(context),
            "-I",
            "-c",
            "import sys; from triton.backends import backends; print(sorted(backends)); "
            "assert sys.argv[1] in backends, 'Expected backend not discovered'",
            expected,
        ]
    )
    write_json(
        Path(context["artifact_dir"]) / payload["tool_id"] / "backend_discovery.json",
        {
            "task_id": context["task_id"],
            "target_sha": context["target_sha"],
            "expected_backend": expected,
            "status": "pass",
        },
    )


def smoke_success(payload: dict[str, Any]) -> None:
    """Record successful completion of the smoke command."""
    context, tool = payload["context"], payload["tool_id"]
    write_json(
        Path(context["artifact_dir"]) / tool / "smoke_success.json",
        {
            "task_id": context["task_id"],
            "target_sha": context["target_sha"],
            "python_executable": candidate_python(context),
            "task_venv": context.get("task_venv"),
            "tool": tool,
            "status": "passed",
        },
    )


def prepare_tests(payload: dict[str, Any]) -> None:
    """Validate actual container paths and discard earlier attempt summaries."""
    source = Path(payload["test_source"]).resolve(strict=True)
    for value in payload["test_paths"]:
        selected = (source / value.split("::", 1)[0]).resolve(strict=True)
        if selected != source and source not in selected.parents:
            raise ValueError("Test path resolves outside its configured checkout")
        if not selected.is_file() and not selected.is_dir():
            raise ValueError("Test path is not a real file or directory")
    out = Path(payload["context"]["artifact_dir"]) / payload["tool_id"]
    for name in ("tests.json",):
        (out / name).unlink(missing_ok=True)


def compare_performance(payload: dict[str, Any]) -> None:
    context, tool = payload["context"], payload["tool_id"]
    out = Path(context["artifact_dir"]) / tool
    candidate = read_json(out / "candidate.json")
    validate_measurements(tool, candidate, payload["kernels"])
    metadata = candidate.setdefault("metadata", {})
    metadata.update(
        environment_fingerprint=environment_fingerprint(context),
        commit_sha=context["target_sha"],
        profile_id=context.get("profile", {}).get("id"),
        llvm_revision=context.get("profile", {}).get("llvm_revision"),
    )
    write_json(out / "candidate.json", candidate)
    baseline = context.get("performance_baselines", {}).get(tool)
    if baseline is None and context.get("variant") == "candidate":
        # Codex controls whether to measure performance.  When selected, its
        # base run is the nearest trustworthy comparison input for the
        # candidate run; the existing identity checks below still decide
        # whether the two measurements are comparable.
        path = Path(context["artifact_dir"]).parent / "base" / tool / "candidate.json"
        if path.is_file():
            try:
                before = read_json(path).get("metadata", {})
                baseline = {
                    "path": str(path),
                    "sha256": digest(path),
                    "base_sha": before.get("commit_sha"),
                    "profile_id": before.get("profile_id"),
                    "llvm_revision": before.get("llvm_revision"),
                    "environment_fingerprint": before.get("environment_fingerprint"),
                }
            except (OSError, ValueError, TypeError):
                baseline = None
    profile = context.get("profile", {})
    baseline_path = None
    reason = "baseline_missing"
    if baseline:
        expected = {
            "base_sha": context.get("base_sha"),
            "profile_id": profile.get("id"),
            "llvm_revision": profile.get("llvm_revision"),
            "environment_fingerprint": environment_fingerprint(context),
        }
        reason = next(
            (
                key + "_mismatch"
                for key, value in expected.items()
                if not value or baseline.get(key) != value
            ),
            "",
        )
        if not reason:
            try:
                path = Path(baseline["path"])
                if not path.is_file() or digest(path) != baseline.get("sha256"):
                    raise ValueError("baseline artifact digest differs")
                previous = read_json(path)
                validate_measurements(tool, previous, payload["kernels"])
                before = previous.get("metadata", {})
                if before.get("commit_sha") != context.get("base_sha"):
                    raise ValueError("baseline commit differs")
                for key in (
                    "environment_fingerprint",
                    "backend",
                    "vendor",
                    "kernels",
                    "repeat",
                    "warmup",
                    "rtol",
                    "atol",
                ):
                    if before.get(key) != metadata.get(key):
                        raise ValueError("measurement conditions differ: " + key)
                baseline_path = path
            except (OSError, KeyError, ValueError, TypeError) as exc:
                reason = "baseline_invalid: " + str(exc)
    if baseline_path is None:
        write_json(
            out / "comparison.json",
            {
                "status": "not_comparable",
                "reason": reason,
                "candidate": "candidate.json",
                "performance_only": True,
            },
        )
    else:
        script = {
            "compile_time": "compare_compile_time.py",
            "pass_profile": "compare_pass_profile.py",
            "ir_serialization": "compare_ir_serialization.py",
        }[tool]
        command = [
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).parent / "performance" / script),
            "--candidate-json",
            str(out / "candidate.json"),
            "--candidate-sha",
            context["target_sha"],
            "--base-sha",
            context.get("base_sha", ""),
            "--kernels",
            ",".join(payload["kernels"]),
            "--baseline-json",
            str(baseline_path),
            "--output-json",
            str(out / "comparison.json"),
            "--output-markdown",
            str(out / "comparison.md"),
        ]
        if tool != "compile_time":
            command += ["--output-csv", str(out / "comparison.csv")]
        run(command)
    write_json(
        out / "baseline_identity.json",
        {
            "baseline_available": baseline_path is not None,
            "baseline": baseline,
            "reason": reason,
            "profile_id": profile.get("id"),
            "environment_fingerprint": environment_fingerprint(context),
            "llvm_revision": profile.get("llvm_revision"),
            "target_sha": context["target_sha"],
            "regression_blocks_merge": False,
        },
    )


def validate_measurements(
    tool: str, candidate: dict[str, Any], kernels: list[str]
) -> None:
    if not kernels or not isinstance(candidate.get("summary"), dict):
        raise ValueError("Benchmark produced no valid candidate summary")
    for kernel in kernels:
        item = candidate["summary"].get(kernel, {})
        if tool == "compile_time":
            measurements = [item.get("compile_est", {})]
            if item.get("all_correct") is not True:
                raise ValueError(
                    f"Compile benchmark correctness failed/missing for {kernel}"
                )
        elif tool == "pass_profile":
            passes = item.get("passes", {})
            if not passes or not any(
                event.get("kind") == "pass" and event.get("kernel") == kernel
                for event in candidate.get("events", [])
            ):
                raise ValueError(f"No pass timing events collected for {kernel}")
            measurements = [value.get("wall_ms", {}) for value in passes.values()]
        else:
            if item.get("module_count", 0) < 1:
                raise ValueError(f"No IR modules collected for {kernel}")
            measurements = [
                item.get("metrics", {}).get(metric, {})
                for metric in ("serialize", "deserialize", "roundtrip")
            ]
            rows = [
                row for row in candidate.get("raw", []) if row.get("kernel") == kernel
            ]
            if not rows or not all(
                row.get("roundtrip_verified") is True for row in rows
            ):
                raise ValueError(f"IR roundtrip was not verified for {kernel}")
        for value in measurements:
            median = value.get("median_ms")
            if (
                isinstance(median, bool)
                or not isinstance(median, (int, float))
                or not math.isfinite(median)
                or median < 0
                or type(value.get("count")) is not int
                or value["count"] < 1
            ):
                raise ValueError(
                    f"Invalid/empty measured candidate timing for {kernel}"
                )


def control_plane(payload: dict[str, Any]) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from control_plane import execute

    execute(payload)


ACTIONS = {
    function.__name__: function
    for function in (
        preflight,
        environment,
        prepare_build,
        record_wheel,
        install_wheel,
        frontend_import,
        backend_discovery,
        smoke_success,
        prepare_tests,
        compare_performance,
        control_plane,
    )
}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ACTIONS:
        raise SystemExit("usage: actions.py <action> <JSON payload>")
    ACTIONS[sys.argv[1]](json.loads(sys.argv[2]))
