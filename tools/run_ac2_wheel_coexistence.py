#!/usr/bin/env python3
"""Build and run the AC2 two-native-wheel proof in a clean venv."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FIXTURE_NAMES = ("coexist_alpha", "coexist_beta")
FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


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


def _uv_binary():
    located = shutil.which("uv")
    if located:
        return Path(located)
    fallback = WORKSPACE / "bootstrap-venv" / "bin" / "uv"
    if fallback.is_file():
        return fallback
    raise RuntimeError("uv is required; install it or provide it on PATH")


def _single_wheel(directory, prefix):
    matches = tuple(directory.glob(prefix + "*.whl"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {prefix} wheel in {directory}, found {len(matches)}"
        )
    return matches[0]


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
    uv = _uv_binary()

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
    core_wheel = _single_wheel(core_dir, "triton_anchor-")

    _run([sys.executable, "-m", "venv", venv_dir], env=build_env)
    clean_python = venv_dir / "bin" / "python"
    _run([clean_python, "-m", "pip", "install", core_wheel], env=build_env)
    fingerprint_result = _run(
        [
            clean_python,
            "-c",
            (
                "from triton_anchor.backends import collect_core_environment;"
                "print(collect_core_environment().core_abi_fingerprint)"
            ),
        ],
        env=build_env,
        capture=True,
    )
    fingerprint = fingerprint_result.stdout.strip()
    if FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
        raise RuntimeError(f"installed Core returned invalid ABI: {fingerprint!r}")

    fixture_env = build_env.copy()
    fixture_env["TRITON_ANCHOR_TEST_CORE_ABI_FINGERPRINT"] = fingerprint
    for fixture_name in FIXTURE_NAMES:
        _run(
            [
                uv,
                "build",
                "--wheel",
                "--out-dir",
                plugin_dir,
                ROOT
                / "tests"
                / "fixtures"
                / "backend_plugins"
                / fixture_name,
            ],
            env=fixture_env,
        )
    plugin_wheels = tuple(sorted(plugin_dir.glob("*.whl")))
    if len(plugin_wheels) != 2:
        raise RuntimeError(f"expected two plugin wheels, found {len(plugin_wheels)}")
    _run(
        [clean_python, "-m", "pip", "install", *plugin_wheels],
        env=build_env,
    )

    probe_env = build_env.copy()
    probe_env.pop("PYTHONPATH", None)
    result = _run(
        [clean_python, ROOT / "tests" / "ac2_native_coexistence_probe.py"],
        env=probe_env,
        capture=True,
    )
    evidence = json.loads(result.stdout)
    evidence["artifacts"] = {
        "core_wheel": str(core_wheel),
        "plugin_wheels": [str(path) for path in plugin_wheels],
    }
    evidence_path = output_root / "ac2-coexistence-evidence.json"
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
            prefix="triton-anchor-ac2-wheel-proof."
        ) as directory:
            run(Path(directory) / "proof")


if __name__ == "__main__":
    main()
