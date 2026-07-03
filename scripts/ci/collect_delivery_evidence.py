#!/usr/bin/env python3
"""Collect delivery CI evidence for triton-anchor smoke validation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:  # pragma: no cover - evidence should be best-effort.
        return f"unavailable: {exc}"


def git_command(args: list[str], repo_dir: Path | None = None) -> str:
    cwd = repo_dir or Path.cwd()
    return run_command(["git", "-C", str(cwd), "-c", f"safe.directory={cwd}", *args])


def git_info(repo_dir: Path) -> dict[str, Any]:
    exists = repo_dir.exists()
    return {
        "path": str(repo_dir),
        "exists": exists,
        "commit": git_command(["rev-parse", "HEAD"], repo_dir) if exists else "",
        "branch": git_command(["branch", "--show-current"], repo_dir) if exists else "",
        "ref": git_command(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir) if exists else "",
        "remote": git_command(["config", "--get", "remote.origin.url"], repo_dir) if exists else "",
        "status": git_command(["status", "--short"], repo_dir) if exists else "",
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def discover_backends() -> dict[str, str]:
    try:
        from triton.backends import backends

        return {
            name: f"{backend.compiler.__module__}.{backend.compiler.__name__}"
            for name, backend in backends.items()
        }
    except Exception as exc:
        return {"error": str(exc)}


def discover_adapters() -> dict[str, str]:
    try:
        from triton_anchor.adapters.registry import AdapterRegistry

        return AdapterRegistry.list_adapters()
    except Exception as exc:
        return {"error": str(exc)}


def list_artifacts(artifact_dir: Path) -> list[dict[str, Any]]:
    if not artifact_dir.exists():
        return []
    return [
        {
            "path": str(path),
            "size": path.stat().st_size,
        }
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file()
    ]


def env_value(name: str) -> str:
    return os.getenv(name, "")


def top_level_entries(path: Path | None, limit: int = 30) -> list[str]:
    if path is None or not path.exists() or not path.is_dir():
        return []
    return [entry.name for entry in sorted(path.iterdir(), key=lambda item: item.name)[:limit]]


def prebuilt_dependency_info(
    name: str,
    root_env: str,
    archive_env: str,
    url_env: str,
    sha_env: str,
    strip_env: str,
    required_env: str,
) -> dict[str, Any]:
    root_value = env_value(root_env)
    root = Path(root_value) if root_value else None
    required = env_value(required_env) or env_value("REQUIRE_PREBUILT_DEPS")
    info: dict[str, Any] = {
        "name": name,
        "root_env": root_env,
        "root": root_value,
        "exists": bool(root and root.exists()),
        "required": required,
        "archive": env_value(archive_env),
        "url": env_value(url_env),
        "sha256": env_value(sha_env),
        "strip_components": env_value(strip_env),
        "top_level_entries": top_level_entries(root),
    }

    if root is None:
        return info

    if name == "llvm":
        llvm_config = root / "bin" / "llvm-config"
        info["llvm_config"] = str(llvm_config)
        info["llvm_config_exists"] = llvm_config.exists()
        info["version"] = run_command([str(llvm_config), "--version"]) if llvm_config.exists() else ""
    elif name == "ppl":
        ppl_compile = root / "bin" / "ppl-compile"
        chip_name = env_value("TRITON_CHIP_NAME")
        chip_lib = root / "deps" / "chip" / chip_name / "lib" if chip_name else None
        runtime_lib = root / "deps" / "runtime" / "tpuv7-runtime" / "lib"
        info["ppl_compile"] = str(ppl_compile)
        info["ppl_compile_exists"] = ppl_compile.exists()
        info["ppl_compile_version"] = run_command([str(ppl_compile), "--version"]) if ppl_compile.exists() else ""
        info["chip_name"] = chip_name
        info["chip_lib"] = str(chip_lib) if chip_lib else ""
        info["chip_lib_exists"] = bool(chip_lib and chip_lib.exists())
        info["runtime_lib"] = str(runtime_lib)
        info["runtime_lib_exists"] = runtime_lib.exists()

    return info


def backend_repo_dir() -> Path | None:
    if env_value("BACKEND_PATH"):
        return Path(env_value("BACKEND_PATH"))
    if env_value("BACKEND_REPO_URL"):
        return Path(env_value("BACKEND_CLONE_DIR") or "/tmp/triton-anchor-backend")
    return None


def flaggems_repo_dir() -> Path:
    if env_value("FLAGGEMS_CLONE_DIR"):
        return Path(env_value("FLAGGEMS_CLONE_DIR"))
    return Path(env_value("WORKSPACE") or "/workspace") / "FlagGems"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=os.getenv("TRITON_ANCHOR_DELIVERY_BACKEND", "frontend-only"))
    parser.add_argument("--artifact-dir", default=os.getenv("DELIVERY_ARTIFACT_DIR", "delivery-artifacts"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    backend_dir = backend_repo_dir()
    flaggems_dir = flaggems_repo_dir()
    evidence = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "backend_profile": env_value("BACKEND_PROFILE"),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "git": {
            "commit": os.getenv("GITHUB_SHA") or git_command(["rev-parse", "HEAD"]),
            "ref": os.getenv("GITHUB_REF", ""),
            "status": git_command(["status", "--short"]),
        },
        "github": {
            "repository": env_value("GITHUB_REPOSITORY"),
            "workflow": env_value("GITHUB_WORKFLOW"),
            "run_id": env_value("GITHUB_RUN_ID"),
            "run_attempt": env_value("GITHUB_RUN_ATTEMPT"),
            "sha": env_value("GITHUB_SHA"),
            "ref": env_value("GITHUB_REF"),
        },
        "anchor_repository": git_info(Path.cwd()),
        "backend_repository": (
            {"configured": False}
            if backend_dir is None
            else {"configured": True, **git_info(backend_dir)}
        ),
        "flaggems_repository": {
            "enabled": env_value("RUN_FLAGGEMS_TESTS"),
            "configured_url": env_value("FLAGGEMS_REPO_URL"),
            "configured_ref": env_value("FLAGGEMS_REF"),
            "test_command": env_value("FLAGGEMS_TEST_COMMAND"),
            **git_info(flaggems_dir),
        },
        "prebuilt_dependencies": {
            "llvm": prebuilt_dependency_info(
                "llvm",
                "LLVM_BUILD_DIR",
                "PREBUILT_LLVM_ARCHIVE",
                "PREBUILT_LLVM_URL",
                "PREBUILT_LLVM_SHA256",
                "PREBUILT_LLVM_STRIP_COMPONENTS",
                "REQUIRE_PREBUILT_LLVM",
            ),
            "ppl": prebuilt_dependency_info(
                "ppl",
                "PPL_ROOT",
                "PREBUILT_PPL_ARCHIVE",
                "PREBUILT_PPL_URL",
                "PREBUILT_PPL_SHA256",
                "PREBUILT_PPL_STRIP_COMPONENTS",
                "REQUIRE_PREBUILT_PPL",
            ),
        },
        "environment": {
            "LLVM_BUILD_DIR": env_value("LLVM_BUILD_DIR"),
            "PPL_ROOT": env_value("PPL_ROOT"),
            "BACKEND_PROFILE": env_value("BACKEND_PROFILE"),
            "BACKEND_REPO_URL": env_value("BACKEND_REPO_URL"),
            "BACKEND_REF": env_value("BACKEND_REF"),
            "BACKEND_CLONE_DIR": env_value("BACKEND_CLONE_DIR"),
            "BACKEND_PATH": env_value("BACKEND_PATH"),
            "BACKEND_ENVSETUP": env_value("BACKEND_ENVSETUP"),
            "BACKEND_ENVSETUP_ARGS": env_value("BACKEND_ENVSETUP_ARGS"),
            "BACKEND_TEST_COMMAND": env_value("BACKEND_TEST_COMMAND"),
            "BACKEND_TORCH_VERSION": env_value("BACKEND_TORCH_VERSION"),
            "BACKEND_INSTALL_ARGS": env_value("BACKEND_INSTALL_ARGS"),
            "BACKEND_TORCH_INDEX_URL": env_value("BACKEND_TORCH_INDEX_URL"),
            "BACKEND_TORCH_TPU_WHEEL_URL": env_value("BACKEND_TORCH_TPU_WHEEL_URL"),
            "TRITON_ANCHOR_DELIVERY_BACKEND": env_value("TRITON_ANCHOR_DELIVERY_BACKEND"),
            "EXPECTED_TRITON_BACKEND": env_value("EXPECTED_TRITON_BACKEND"),
            "TRITON_CHIP_NAME": env_value("TRITON_CHIP_NAME"),
            "TRITON_TO_PPL_MODE": env_value("TRITON_TO_PPL_MODE"),
            "PPLCOMPILE_PATH": env_value("PPLCOMPILE_PATH"),
        },
        "packages": {
            "triton-anchor": package_version("triton-anchor"),
            "triton": package_version("triton"),
            "torch": package_version("torch"),
            "torch-tpu": package_version("torch-tpu"),
            "torch_tpu": package_version("torch_tpu"),
        },
        "triton_backends": discover_backends(),
        "triton_anchor_adapters": discover_adapters(),
        "artifacts": list_artifacts(artifact_dir),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote delivery evidence to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
