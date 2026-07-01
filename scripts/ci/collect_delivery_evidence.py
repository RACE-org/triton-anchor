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


def git_command(args: list[str]) -> str:
    cwd = str(Path.cwd())
    return run_command(["git", "-c", f"safe.directory={cwd}", *args])


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=os.getenv("TRITON_ANCHOR_DELIVERY_BACKEND", "frontend-only"))
    parser.add_argument("--artifact-dir", default=os.getenv("DELIVERY_ARTIFACT_DIR", "delivery-artifacts"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    evidence = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
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
        "environment": {
            "LLVM_BUILD_DIR": os.getenv("LLVM_BUILD_DIR", ""),
            "PPL_ROOT": os.getenv("PPL_ROOT", ""),
            "BACKEND_REPO_URL": os.getenv("BACKEND_REPO_URL", ""),
            "BACKEND_REF": os.getenv("BACKEND_REF", ""),
            "TRITON_ANCHOR_DELIVERY_BACKEND": os.getenv("TRITON_ANCHOR_DELIVERY_BACKEND", ""),
        },
        "packages": {
            "triton-anchor": package_version("triton-anchor"),
            "triton": package_version("triton"),
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
