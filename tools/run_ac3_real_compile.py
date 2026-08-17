#!/usr/bin/env python3
"""Build and run the AC3 real-compile proof in a clean venv.

Proves the acceptance chain: BackendPlugin -> BackendPluginRegistry.select()
-> selected plugin.compiler_cls -> compiler instance -> triton.compile() ->
compile success.  Also runs an injected compile-failure mode that must return
nonzero.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import run_ac2_wheel_coexistence as build_support


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "backend_plugins"
    / "ac3_real_compile"
)


def _run(argv, *, env=None, capture=False):
    print("EXEC " + " ".join(str(item) for item in argv), flush=True)
    return subprocess.run(
        [str(item) for item in argv],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def run(output_root):
    output_root.mkdir(parents=True, exist_ok=False)
    core_dir = output_root / "core"
    plugin_dir = output_root / "plugins"
    venv_dir = output_root / "venv"
    core_dir.mkdir()
    plugin_dir.mkdir()

    build_env = os.environ.copy()
    build_env.pop("PYTHONPATH", None)
    llvm_build = WORKSPACE / "llvm-build"
    build_env.setdefault("LLVM_SYSPATH", str(llvm_build))
    build_env.setdefault("LLVM_BUILD_DIR", str(llvm_build))
    build_env.setdefault("MAX_JOBS", "4")
    # --no-build-isolation must use the interpreter that launched the runner;
    # otherwise uv may select a system Python without bdist_wheel.
    build_env["UV_PYTHON"] = sys.executable
    uv = build_support._uv_binary()

    _run(
        [
            uv,
            "build",
            "--wheel",
            "--no-build-isolation",
            "--out-dir",
            core_dir,
            ROOT,
        ],
        env=build_env,
    )
    core_wheel = build_support._single_wheel(core_dir, "triton_anchor-")

    _run([sys.executable, "-m", "venv", venv_dir], env=build_env)
    clean_python = venv_dir / "bin" / "python"
    _run([clean_python, "-m", "pip", "install", core_wheel], env=build_env)
    _run(
        [uv, "build", "--wheel", "--out-dir", plugin_dir, FIXTURE],
        env=build_env,
    )
    fixture_wheel = build_support._single_wheel(
        plugin_dir, "triton_anchor_ac3_real_compile_backend-"
    )
    _run(
        [clean_python, "-m", "pip", "install", fixture_wheel],
        env=build_env,
    )

    probe_env = build_env.copy()
    probe_env.pop("PYTHONPATH", None)
    success = _run(
        [
            clean_python,
            ROOT / "tests" / "ac3_real_compile_probe.py",
        ],
        env=probe_env,
        capture=True,
    )
    evidence = json.loads(success.stdout)

    failure = subprocess.run(
        [
            str(clean_python),
            str(ROOT / "tests" / "ac3_real_compile_probe.py"),
            "--mode",
            "fail-compile",
        ],
        cwd=ROOT,
        env=probe_env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if failure.returncode == 0:
        raise RuntimeError(
            "AC3 fail-mode probe unexpectedly succeeded: "
            + (failure.stdout or "")
        )
    evidence["fail_mode"] = {
        "expected_failure": True,
        "returncode": failure.returncode,
    }
    evidence["artifacts"] = {
        "core_wheel": str(core_wheel),
        "fixture_wheel": str(fixture_wheel),
    }

    evidence_path = output_root / "ac3-real-compile-evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"EVIDENCE {evidence_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.output_root:
        run(args.output_root.resolve())
    else:
        with tempfile.TemporaryDirectory(
            prefix="triton-anchor-ac3-real-compile."
        ) as directory:
            run(Path(directory) / "proof")


if __name__ == "__main__":
    main()
