#!/usr/bin/env python3
"""Reusable local build and test commands, callable from a shell or Python.

Build, install, tests and smoke are separately callable stages. Callers must
satisfy dependencies first and may use their own equivalent commands.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

if __package__:
    from .variant_exec import command_environment
else:
    from variant_exec import command_environment

BACKEND_TOOLS = frozenset(
    {
        "backend_build",
        "backend_install",
        "backend_tests",
        "backend_smoke",
        "flaggems",
        "compile_time",
        "pass_profile",
        "ir_serialization",
    }
)
DEPENDENCIES = {
    "environment": [],
    "control_plane": [],
    "frontend_build": ["environment"],
    "frontend_install": ["frontend_build"],
    "frontend_tests": ["frontend_install"],
    "frontend_smoke": ["frontend_install"],
    "backend_build": ["environment", "frontend_install"],
    "backend_install": ["backend_build", "frontend_install"],
    "backend_tests": ["frontend_install", "backend_install"],
    "backend_smoke": ["frontend_install", "backend_install"],
    "flaggems": ["backend_smoke"],
    "compile_time": ["backend_smoke"],
    "pass_profile": ["backend_smoke"],
    "ir_serialization": ["backend_smoke"],
}
TOOL_IDS = tuple(DEPENDENCIES)
DEFAULT_BUILD_JOBS = 12
MINIMUM_FRONTEND = (
    "environment",
    "frontend_build",
    "frontend_install",
    "frontend_smoke",
)


def dependencies(tool_id: str, config: dict[str, Any] | None = None) -> list[str]:
    return list(DEPENDENCIES[tool_id])


def parameter_names(tool_id: str) -> set[str]:
    if tool_id in {"frontend_build", "backend_build"}:
        return {"jobs", "build_mode"}
    if tool_id in {"frontend_install", "backend_install"}:
        return {"wheel"}
    if tool_id in {"frontend_tests", "backend_tests", "control_plane"}:
        return {"paths", "keyword"}
    if tool_id == "flaggems":
        return {"mode", "ops", "categories"}
    if tool_id in {"compile_time", "pass_profile", "ir_serialization"}:
        return {"kernels", "repeat", "warmup"}
    return set()


def bounded(value: Any, name: str, low: int, high: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def path_join(root: str, *parts: str) -> str:
    # Plans run on the controller but normally describe Linux container paths.
    return (
        str(PurePosixPath(root).joinpath(*parts))
        if root.startswith("/")
        else str(Path(root).joinpath(*parts))
    )


def environment_command(
    argv: list[str], config: dict[str, Any], tools_dir: str, *, backend: bool = False
) -> list[str]:
    """Apply the same profile setup to planned stages and task-local scripts."""
    scripts = list(config.get("env_scripts", []))
    if backend:
        scripts += config.get("backend_env_scripts", [])
    for script in reversed(scripts):
        argv = [
            "bash",
            path_join(tools_dir, "basic_tools", "env_exec.sh"),
            script["path"],
            *script.get("args", []),
            "--",
            *argv,
        ]
    return argv


def test_selection(
    tool_id: str, config: dict[str, Any], parameters: dict[str, Any]
) -> list[str]:
    """Select pytest nodes only inside the trusted profile's test roots."""
    key = "frontend_test_paths" if tool_id == "frontend_tests" else "backend_test_paths"
    roots = config.get(
        key,
        ["python/triton_anchor/tests", "tests"]
        if tool_id == "frontend_tests"
        else None,
    )

    def relative(value: Any, node: bool = False) -> PurePosixPath:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 2048
            or any(c in value for c in "\\\n\r\x00")
        ):
            raise ValueError(
                "Test selections must be relative paths or pytest node IDs"
            )
        filename = value.split("::", 1)[0] if node else value
        path = PurePosixPath(filename)
        if (
            not filename
            or path.is_absolute()
            or ".." in path.parts
            or ":" in filename
            or filename.startswith("-")
        ):
            raise ValueError("Test paths must stay within their trusted roots")
        return path

    if not isinstance(roots, list) or not roots or len(roots) > 100:
        raise ValueError(f"profile.tools.{key} must configure nonempty real test paths")
    allowed = [relative(value) for value in roots]
    selected = parameters.get("paths", roots)
    if not isinstance(selected, list) or not selected or len(selected) > 100:
        raise ValueError("paths must select 1..100 test paths or pytest node IDs")
    for value in selected:
        path = relative(value, node=True)
        if not any(path == root or root in path.parents for root in allowed):
            raise ValueError(
                "Test selection is outside the profile's trusted test roots"
            )
    return list(dict.fromkeys(selected))


def normalize_context(context: dict[str, Any]) -> dict[str, Any]:
    value = dict(context)
    for key in ("source_dir", "artifact_dir"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"context.{key} is required")
    if not value.get("target_sha"):
        value["target_sha"] = subprocess.check_output(
            ["git", "-C", value["source_dir"], "rev-parse", "HEAD"], text=True
        ).strip()
    value.setdefault("task_id", "local")
    value.setdefault("triton_version", value.get("profile", {}).get("triton_version", ""))
    return value


def plan(
    tool_id: str, context: dict[str, Any], parameters: dict[str, Any] | None = None
) -> dict[str, Any]:
    if tool_id not in DEPENDENCIES:
        raise ValueError(f"Unknown basic tool: {tool_id}")
    params = dict(parameters or {})
    context = normalize_context(context)
    result: dict[str, Any] = {
        "tool_id": tool_id,
        "status": "ready",
        "reason": "",
        "commands": [],
        "artifacts": [],
        "dependencies": dependencies(
            tool_id, context.get("profile", {}).get("tools", {})
        ),
    }
    if tool_id in BACKEND_TOOLS and not context.get("profile", {}).get(
        "backend_enabled",
        bool(re.fullmatch(r"3\.0(?:\.\d+)?", context["triton_version"])),
    ):
        result.update(
            status="not_applicable", reason="所选环境未声明后端、算子及性能测试能力。"
        )
        return result
    allowed = parameter_names(tool_id)
    if set(params) - allowed:
        raise ValueError(f"Unsupported parameters: {sorted(set(params) - allowed)}")
    profile = context.get("profile", {})
    config = profile.get("tools", {})
    py = context.get("python_bin", config.get("python_bin", "/task/candidate/venv/bin/python"))
    trusted_py = context.get("trusted_python_bin", "/opt/venv/bin/python")
    root = context.get("tools_dir", str(Path(__file__).resolve().parents[1]))
    helper = path_join(root, "basic_tools", "actions.py")
    source = context["source_dir"]
    out = path_join(context["artifact_dir"], tool_id)
    env = {str(k): str(v) for k, v in config.get("env", {}).items()}
    env.update(
        ANCHOR_DIR=source,
        GITHUB_SHA=context["target_sha"],
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHON_BIN=py,
        TMPDIR=path_join(out, "tmp"),
        TRITON_CACHE_DIR=path_join(out, "cache"),
        TRITON_DUMP_DIR=path_join(out, "dump"),
        UV_LINK_MODE="copy",
    )
    if context.get("task_root"):
        env["LOCAL_CI_TASK_ROOT"] = context["task_root"]
    if context.get("task_venv"):
        env["LOCAL_CI_TASK_VENV"] = context["task_venv"]
    if config.get("llvm_dir"):
        llvm = config["llvm_dir"]
        env.update(
            LLVM_BUILD_DIR=llvm,
            LLVM_SYSPATH=llvm,
            LLVM_INCLUDE_DIRS=path_join(llvm, "include"),
            LLVM_LIBRARY_DIR=path_join(llvm, "lib"),
            LLVM_BINARY_DIR=path_join(llvm, "bin"),
        )
    if "jobs" in allowed:
        default_jobs = int(env.get("MAX_JOBS", DEFAULT_BUILD_JOBS))
        jobs = bounded(params.get("jobs", default_jobs), "jobs", 1, 64)
        if params.get("build_mode", "fresh") not in {"fresh", "incremental"}:
            raise ValueError("build_mode must be fresh or incremental")
        env.update(
            MAX_JOBS=str(jobs),
            CMAKE_BUILD_PARALLEL_LEVEL=str(jobs),
            NINJAFLAGS=f"-j{jobs}",
        )

    def add(argv: list[str], cwd: str | None = None, timeout: int = 300) -> None:
        argv = environment_command(argv, config, root, backend=tool_id in BACKEND_TOOLS)
        result["commands"].append(
            {"argv": argv, "cwd": cwd or source, "env": env.copy(), "timeout": timeout}
        )

    def action(
        name: str,
        extra: dict[str, Any] | None = None,
        timeout: int = 300,
        cwd: str | None = None,
    ) -> None:
        # Pass only the paths and environment settings used by the tool.
        action_context = {
            key: context[key]
            for key in (
                "source_dir",
                "artifact_dir",
                "variant",
                "task_id",
                "target_sha",
                "triton_version",
                "base_sha",
                "performance_baselines",
                "task_venv",
                "environment_fingerprint",
                "dependency_artifacts",
                "task_root",
            )
            if key in context
        }
        action_context["python_bin"] = py
        action_context["profile"] = {
            "id": profile.get("id"),
            "llvm_revision": profile.get("llvm_revision"),
            "tools": config,
        }
        payload = {
            "tool_id": tool_id,
            "context": action_context,
            "parameters": params,
            **(extra or {}),
        }
        add(
            [
                trusted_py,
                "-I",
                helper,
                name,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ],
            cwd,
            timeout,
        )

    action("preflight")
    if tool_id == "control_plane":
        action("control_plane", timeout=3600)
    elif tool_id == "environment":
        action("environment")
    elif tool_id in {"frontend_build", "backend_build"}:
        build_source = (
            source if tool_id == "frontend_build" else config.get("backend_dir")
        )
        if not build_source:
            raise ValueError("profile.tools.backend_dir is required")
        action("prepare_build", {"build_source": build_source})
        add(
            [
                py,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                path_join(out, "wheels"),
                build_source,
            ],
            build_source,
            7200,
        )
        action("record_wheel")
    elif tool_id in {"frontend_install", "backend_install"}:
        action("install_wheel", timeout=1200)
        if tool_id == "frontend_install":
            # -I prevents candidate source/PYTHONPATH from shadowing the installed wheel.
            action("frontend_import", cwd=out)
        else:
            action("backend_discovery", cwd=out)
    elif tool_id in {"frontend_tests", "backend_tests"}:
        test_source = (
            source if tool_id == "frontend_tests" else config.get("backend_dir")
        )
        if not test_source:
            raise ValueError("profile.tools.backend_dir is required")
        selected = test_selection(tool_id, config, params)
        keyword = params.get("keyword", "")
        if (
            not isinstance(keyword, str)
            or len(keyword) > 300
            or any(c in keyword for c in "\n\r\x00")
        ):
            raise ValueError(
                "keyword must be a pytest expression of at most 300 characters"
            )
        action("prepare_tests", {"test_source": test_source, "test_paths": selected})
        argv = [
            py,
            "-I",
            path_join(root, "basic_tools", "pytest_exec.py"),
            "--output",
            path_join(out, "tests.json"),
            "--",
            "-q",
            "-o",
            "addopts=",
            "--capture=sys",
            "--import-mode=prepend" if tool_id == "backend_tests" else "--import-mode=importlib",
            "--rootdir",
            test_source,
        ]
        if keyword:
            argv += ["-k", keyword]
        argv += [path_join(test_source, value) for value in selected]
        env["PYTEST_ADDOPTS"] = ""
        add(argv, out, 3600)
    elif tool_id == "frontend_smoke":
        add([py, "-I", path_join(source, "tests", "test_smoke.py")], out, 900)
        action("smoke_success")
    elif tool_id == "backend_smoke":
        command = config.get("backend_smoke_argv")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(s, str) for s in command)
        ):
            raise ValueError(
                "profile.tools.backend_smoke_argv must configure a real backend JIT test"
            )
        command = [py if part == "{python}" else part for part in command]
        if command[0] in {"python", "python3", config.get("python_bin", "python3")}:
            command[0] = py
        action("backend_discovery")
        add(command, config["backend_dir"], 1800)
        action("smoke_success")
    elif tool_id == "flaggems":
        mode = params.get("mode", "impact")
        if mode not in {"impact", "full"}:
            raise ValueError("FlagGems mode must be impact or full")
        for name in ("ops", "categories"):
            values = params.get(name, [])
            if not isinstance(values, list) or any(
                not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", v)
                for v in values
            ):
                raise ValueError(
                    f"{name} must be an array of operator/category identifiers"
                )
        fg = config.get("flaggems_dir")
        if not fg:
            raise ValueError("profile.tools.flaggems_dir is required")
        argv = [
            py,
            path_join(root, "basic_tools", "flaggems", "batch_test_flaggems.py"),
            "--mode",
            mode,
            "--ops",
            ",".join(params.get("ops", [])),
            "--categories",
            ",".join(params.get("categories", [])),
            "--whitelist",
            path_join(root, "basic_tools", "flaggems", "flaggems_pass_whitelist.tsv"),
            "--full-list",
            path_join(root, "basic_tools", "flaggems", "flaggems_all_ops.tsv"),
            "--flaggems-dir",
            fg,
            "--python-bin",
            py,
            "--artifact-dir",
            out,
            "--selected-output",
            path_join(out, "selected.txt"),
            "--clear-cache",
            "0",
            "--pytest-args=" + config.get("flaggems_pytest_args", "--ref cpu -vs"),
        ]
        add(argv, fg, 86400 if mode == "full" else 14400)
    else:
        kernels = params.get("kernels", ["add", "mm", "softmax", "layernorm"])
        if (
            not isinstance(kernels, list)
            or not kernels
            or not set(kernels) <= {"add", "mm", "softmax", "layernorm"}
        ):
            raise ValueError("kernels must select supported benchmark kernels")
        repeats = bounded(
            params.get("repeat", 20 if tool_id == "ir_serialization" else 3),
            "repeat",
            1,
            100,
        )
        warmup = bounded(params.get("warmup", 1), "warmup", 0, 20)
        backend = config.get("expected_backend")
        fg = config.get("flaggems_dir")
        if not backend or not fg:
            raise ValueError(
                "Performance tools require expected_backend and flaggems_dir"
            )
        script = {
            "compile_time": "compile_benchmark",
            "pass_profile": "pass_profile_benchmark",
            "ir_serialization": "ir_serialization_benchmark",
        }[tool_id]
        argv = [
            py,
            path_join(root, "basic_tools", "performance", script + ".py"),
            "--backend",
            backend,
            "--flaggems-root",
            fg,
            "--kernels",
            ",".join(kernels),
            "--repeat",
            str(repeats),
            "--warmup",
            str(warmup),
            "--output-json",
            path_join(out, "candidate.json"),
        ]
        if tool_id == "pass_profile":
            for flag, name in (
                ("--output-events-csv", "events.csv"),
                ("--output-summary-csv", "summary.csv"),
                ("--output-hotspots-markdown", "hotspots.md"),
            ):
                argv += [flag, path_join(out, name)]
        else:
            argv += ["--output-csv", path_join(out, "candidate.csv")]
        if tool_id == "ir_serialization":
            argv += [
                "--output-markdown",
                path_join(out, "serialization.md"),
                "--work-root",
                path_join(out, "work"),
            ]
        else:
            argv += [
                "--cache-root",
                path_join(out, "cache"),
                "--dump-root",
                path_join(out, "dump"),
            ]
        if config.get("vendor"):
            argv += ["--vendor", config["vendor"]]
        add(argv, fg, 7200)
        action("compare_performance", {"kernels": kernels}, timeout=120)
    result["artifacts"] = [out]
    return result


def execute(
    tool_id: str, context: dict[str, Any], parameters: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run a tool and save its ordinary result and command log."""
    context = normalize_context(context)
    spec = plan(tool_id, context, parameters)
    out = Path(context["artifact_dir"]) / tool_id
    out.mkdir(parents=True, exist_ok=True)
    reports = ("environment", "wheel", "installation", "backend_discovery", "tests",
               "control_plane", "flaggems-summary", "comparison", "smoke_success", "candidate")
    for name in (*reports, "result"):
        (out / (name + ".json")).unlink(missing_ok=True)
    started = time.monotonic()
    result = {
        "tool_id": tool_id,
        "status": spec["status"],
        "parameters": parameters or {},
        "target_sha": context["target_sha"],
        "exit_code": None,
        "artifacts": [],
    }
    if context.get("variant"):
        result.update(
            variant=context["variant"],
            llvm_hash=context.get("profile", {}).get("llvm_revision"),
            environment_fingerprint=context.get("environment_fingerprint"),
        )
    if spec["reason"]:
        result["reason"] = spec["reason"]
    if spec["status"] == "ready":
        print(
            f"{tool_id}: running; log: {out / 'command.log'}",
            file=sys.stderr, flush=True,
        )
        with (out / "command.log").open("w") as log:
            result["status"] = "pass"
            for command in spec["commands"]:
                log.write("$ " + repr(command["argv"]) + "\n")
                log.flush()
                try:
                    child = subprocess.Popen(
                        command["argv"], cwd=command["cwd"],
                        env=command_environment(context, command["env"]),
                        stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
                    try:
                        code = child.wait(timeout=command["timeout"])
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        if os.name == "posix":
                            os.killpg(child.pid, signal.SIGKILL)
                        else:
                            child.kill()
                        child.wait()
                        raise
                except subprocess.TimeoutExpired:
                    code = 124
                    result["reason"] = "Command timed out"
                except KeyboardInterrupt:
                    code = 130
                    result["reason"] = "Interrupted"
                except OSError as exc:
                    code = 127
                    result["reason"] = str(exc)
                result["exit_code"] = code
                if code:
                    result["status"] = (
                        "cancelled" if code in {130, 143, -2, -15}
                        else "infra_error" if code in {124, 126, 127} else "fail"
                    )
                    break
    details = {}
    for name in reports:
        path = out / (name + ".json")
        if path.is_file():
            try:
                value = json.loads(path.read_text())
                if name == "candidate":
                    value = {key: value[key] for key in ("metadata", "summary") if key in value}
                details[name] = value
            except (OSError, ValueError):
                pass  # The command failure and raw log remain available.
    if details:
        result["details"] = details
    if result["status"] == "pass" and details.get("control_plane", {}).get("status") == "limited":
        result.update(status="limited", reason=details["control_plane"]["summary"])
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    result["artifacts"] = [
        str(path.relative_to(out)) for path in sorted(out.iterdir())
        if path.is_file() and path.name != "result.json"
    ]
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool_id", choices=TOOL_IDS)
    parser.add_argument(
        "--context", required=True,
        help="Source, output paths and environment configuration JSON",
    )
    parser.add_argument(
        "--parameters", default="{}", help="Tool parameter JSON object"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute this plan in the current task environment",
    )
    args = parser.parse_args()
    context = json.loads(Path(args.context).read_text(encoding="utf-8-sig"))
    result = (execute if args.execute else plan)(
        args.tool_id, context, json.loads(args.parameters)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"pass", "ready", "not_applicable", "limited"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
