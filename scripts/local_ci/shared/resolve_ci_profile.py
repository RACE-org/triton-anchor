#!/usr/bin/env python3
"""Resolve a server-owned Local CI profile from an exact LLVM revision."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


LLVM_HASH_RE = re.compile(r"[0-9a-f]{40}")


class ProfileResolutionError(ValueError):
    """Raised when an LLVM revision cannot be mapped to a trusted profile."""


def validate_llvm_hash(value: str, *, label: str = "LLVM hash") -> str:
    if not LLVM_HASH_RE.fullmatch(value):
        raise ProfileResolutionError(
            f"{label} must be a lowercase 40-character hexadecimal commit"
        )
    return value


def resolve_profile_file(*, profile_dir: str, llvm_hash: str) -> Path:
    """Return the exact server-owned profile file for an LLVM revision."""

    selected_hash = validate_llvm_hash(llvm_hash)

    if not profile_dir:
        raise ProfileResolutionError("LOCAL_CI_PROFILE_DIR is not configured")

    configured_dir = Path(profile_dir)
    if not configured_dir.is_absolute():
        raise ProfileResolutionError("LOCAL_CI_PROFILE_DIR must be an absolute path")
    try:
        resolved_dir = configured_dir.resolve(strict=True)
    except OSError as exc:
        raise ProfileResolutionError(
            f"LOCAL_CI_PROFILE_DIR is unavailable: {configured_dir}"
        ) from exc
    if not resolved_dir.is_dir():
        raise ProfileResolutionError(
            f"LOCAL_CI_PROFILE_DIR is not a directory: {resolved_dir}"
        )

    configured_file = resolved_dir / f"{selected_hash}.env"
    try:
        resolved_file = configured_file.resolve(strict=True)
    except OSError as exc:
        raise ProfileResolutionError(
            f"No server-owned Local CI profile exists for LLVM hash {selected_hash}"
        ) from exc
    if resolved_file.parent != resolved_dir or not resolved_file.is_file():
        raise ProfileResolutionError(
            f"Invalid server-owned Local CI profile: {resolved_file}"
        )
    return resolved_file


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--llvm-hash", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        profile_file = resolve_profile_file(
            profile_dir=args.profile_dir,
            llvm_hash=args.llvm_hash,
        )
    except ProfileResolutionError as exc:
        print(f"Local CI profile resolution failed: {exc}.", file=sys.stderr)
        return 2
    print(profile_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
