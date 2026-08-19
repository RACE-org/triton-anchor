#!/usr/bin/env python3
"""Build installed producer wheels and prove AC4 protocol evolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import run_ac2_wheel_coexistence as build_support


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "backend_plugins" / "ac4_protocol_1_0"
PROBE = ROOT / "tests" / "ac4_protocol_wheel_probe.py"

CASES = (
    ("1.0", "1.1", "default"),
    ("1.1", "1.0", "ignore"),
    ("1.1", "1.1", "preserved"),
    ("1.2", "1.2", "deprecated"),
    ("2.0", "2.0", "removed"),
    ("2.0", "1.2", "mismatch"),
    ("1.9", "1.9", "same_major_removal"),
)


def run(argv, *, env, capture=False):
    print("EXEC " + " ".join(str(item) for item in argv), flush=True)
    return subprocess.run(
        [str(item) for item in argv],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="triton-anchor-ac4-protocol.") as directory:
        output = Path(directory) / "proof"
        core_dir = output / "core"
        wheels_dir = output / "wheels"
        venv_dir = output / "venv"
        core_dir.mkdir(parents=True)
        wheels_dir.mkdir()
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["UV_PYTHON"] = sys.executable
        llvm_build = ROOT.parent / "llvm-build"
        env.setdefault("LLVM_SYSPATH", str(llvm_build))
        env.setdefault("LLVM_BUILD_DIR", str(llvm_build))
        env.setdefault("MAX_JOBS", "4")
        uv = build_support._uv_binary()

        run(
            [uv, "build", "--wheel", "--no-build-isolation", "--out-dir", core_dir, ROOT],
            env=env,
        )
        core_wheel = build_support._single_wheel(core_dir, "triton_anchor-")
        run([sys.executable, "-m", "venv", venv_dir], env=env)
        python = venv_dir / "bin" / "python"
        run([python, "-m", "pip", "install", core_wheel], env=env)

        evidence = []
        for producer, consumer, expected in CASES:
            case_dir = output / (producer.replace(".", "_") + "_" + consumer.replace(".", "_"))
            case_dir.mkdir()
            case_env = env.copy()
            case_env["TRITON_ANCHOR_AC4_PRODUCER_VERSION"] = producer
            case_env["TRITON_ANCHOR_AC4_MARKER_DIR"] = str(case_dir / "marker")
            run(
                [uv, "build", "--wheel", "--out-dir", case_dir, FIXTURE],
                env=case_env,
            )
            wheel = build_support._single_wheel(case_dir, "triton_anchor_ac4_protocol_backend-")
            run([python, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheel], env=case_env)
            probe_args = [
                python,
                PROBE,
                "--producer-version", producer,
                "--consumer-version", consumer,
                "--expect", expected,
            ]
            if expected == "same_major_removal":
                probe_args.extend(["--removed-protocol-field", "diagnostics"])
            result = run(
                probe_args,
                env=case_env,
                capture=True,
            )
            evidence.append(json.loads(result.stdout))

        result = {
            "case_count": len(evidence),
            "cases": evidence,
            "core_wheel": str(core_wheel),
            "status": "passed",
        }
        evidence_path = output / "ac4-protocol-evidence.json"
        evidence_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        print("EVIDENCE " + str(evidence_path))


if __name__ == "__main__":
    main()
