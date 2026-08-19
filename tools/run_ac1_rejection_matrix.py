#!/usr/bin/env python3
"""Build and execute the installed-wheel AC1 rejection matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import run_ac2_wheel_coexistence as build_support


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "backend_plugins"
W11_CASES = (
    "compatible", "incompatible", "malformed", "full_compatible",
    "bad_protocol", "bad_core", "bad_triton_commit", "bad_llvm_version",
    "bad_llvm_commit", "bad_mlir_version", "bad_mlir_commit",
)
W10_CASES = (
    "native_compatible", "native_bad_abi", "native_collision_a",
    "native_collision_b",
)
ARTIFACT_CASES = (
    "malformed_json", "coverage_error", "missing_abi", "path_escape",
    "missing_native", "hidden_native",
    "record_tamper", "bad_elf", "wrong_arch", "wrong_wheel_tag",
    "subprocess_contract", "missing_dt_needed", "unresolved_symbol",
)
LEGACY_CASES = (
    "legacy_bad_interface", "legacy_missing_compiler",
    "legacy_illegal_compiler", "legacy_missing_driver",
)


def execute(argv, env, *, capture=False):
    print("EXEC " + " ".join(str(item) for item in argv), flush=True)
    return subprocess.run(
        [str(item) for item in argv], cwd=ROOT, env=env, check=True,
        text=True, stdout=subprocess.PIPE if capture else None,
    )


def run(output_root):
    output_root.mkdir(parents=True, exist_ok=False)
    wheels = output_root / "wheels"
    projects = output_root / "legacy-projects"
    venv = output_root / "venv"
    wheels.mkdir()
    projects.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    llvm_build = ROOT.parent / "llvm-build"
    env.setdefault("LLVM_SYSPATH", str(llvm_build))
    env.setdefault("LLVM_BUILD_DIR", str(llvm_build))
    env.setdefault("MAX_JOBS", "4")
    # --no-build-isolation must use the interpreter that launched the runner;
    # otherwise uv may select a system Python without bdist_wheel.
    env["UV_PYTHON"] = sys.executable
    uv = build_support._uv_binary()

    execute([uv, "build", "--wheel", "--no-build-isolation", "--out-dir", wheels, ROOT], env)
    core = build_support._single_wheel(wheels, "triton_anchor-")
    execute([sys.executable, "-m", "venv", venv], env)
    python = venv / "bin" / "python"
    execute([python, "-m", "pip", "install", core], env)
    fingerprint = execute(
        [python, "-c", "from triton_anchor.backends import collect_core_environment as c;print(c().core_abi_fingerprint)"],
        env, capture=True,
    ).stdout.strip()
    fixture_env = env.copy()
    fixture_env["TRITON_ANCHOR_TEST_CORE_ABI_FINGERPRINT"] = fingerprint

    for case in W11_CASES + W10_CASES + ("coexist_alpha",):
        execute([uv, "build", "--wheel", "--out-dir", wheels, FIXTURES / case], fixture_env)
    for case in LEGACY_CASES:
        execute([python, ROOT / "tests/fixtures/backend_plugins/legacy/generate.py",
                 "--case", case, "--output-root", projects], env)
        execute([uv, "build", "--wheel", "--out-dir", wheels, projects / case], env)
    all_wheels = tuple(sorted(wheels.glob("*.whl")))
    execute([python, "-m", "pip", "install", *all_wheels], env)

    alpha = build_support._single_wheel(wheels, "triton_anchor_ac2_native_alpha_backend-")
    evidence = []
    for case in ARTIFACT_CASES:
        for mode in ("load", "register", "select"):
            execute([python, "-m", "pip", "install", "--force-reinstall", "--no-deps", alpha], env)
            result = execute(
                [python, ROOT / "tests/ac1_native_artifact_rejection_probe.py",
                 "--case", case, "--mode", mode], env, capture=True,
            )
            evidence.append(json.loads(result.stdout))

    w10_results = []
    for mode in ("full", "load-first", "register-first", "select-first", "compiler-first", "runtime-first"):
        result = execute(
            [python, ROOT / "tests/w10_native_wheel_probe.py", "--mode", mode],
            env,
            capture=True,
        )
        w10_results.append(json.loads(result.stdout))
    w11_full_result = json.loads(
        execute(
            [python, ROOT / "tests/w11_full_profile_probe.py"],
            env,
            capture=True,
        ).stdout
    )
    execute(
        [python, "-m", "pip", "uninstall", "-y",
         "triton-anchor-w10-native-collision-a-backend",
         "triton-anchor-w10-native-collision-b-backend",
         "triton-anchor-w10-native-compatible-backend",
         "triton-anchor-ac2-native-alpha-backend",
         "triton-anchor-w11-full-compatible-backend"],
        env,
    )
    w11_triton_result = json.loads(
        execute(
            [python, ROOT / "tests/w11_triton_version_probe.py", "--mode", "mixed"],
            env,
            capture=True,
        ).stdout
    )
    w12_results = []
    for case in LEGACY_CASES:
        prefix = case.replace("legacy_", "triton_anchor_w12_legacy_") + "_backend-"
        legacy_wheel = build_support._single_wheel(wheels, prefix)
        for mode in ("bad-register", "bad-compiler", "bad-runtime"):
            result = execute(
                [python, ROOT / "tests/w12_legacy_wheel_probe.py", "--mode", mode,
                 "--case", case, "--wheel-path", legacy_wheel,
                 "--evidence-dir", output_root / (case + "-" + mode)],
                env,
                capture=True,
            )
            w12_results.append(json.loads(result.stdout))

    result = {
        "artifact_cases": evidence,
        "artifact_case_count": len(evidence),
        "w10_results": w10_results,
        "w11_full_result": w11_full_result,
        "w11_triton_result": w11_triton_result,
        "w12_results": w12_results,
        "core_wheel": str(core),
        "fixture_wheel_count": len(all_wheels) - 1,
        "status": "passed",
    }
    path = output_root / "ac1-rejection-evidence.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("EVIDENCE " + str(path))


def main():
    with tempfile.TemporaryDirectory(prefix="triton-anchor-ac1-rejections.") as directory:
        run(Path(directory) / "proof")


if __name__ == "__main__":
    main()
