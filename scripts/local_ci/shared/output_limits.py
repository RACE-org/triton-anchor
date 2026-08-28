#!/usr/bin/env python3
"""Inspect and compact task-owned Local CI artifact directories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


LIMIT_EXIT_CODE = 87
ARTIFACT_MARKER = ".local-ci-artifact-root"
RUN_MARKER = ".local-ci-run-root"
OWNERSHIP_MARKERS = (ARTIFACT_MARKER, RUN_MARKER)


@dataclass(frozen=True)
class Violation:
    kind: str
    path: str
    size_bytes: int
    limit_bytes: int


def scan_tree(
    root: Path, *, max_log_bytes: int, max_file_bytes: int, max_total_bytes: int
) -> tuple[int, list[Violation]]:
    total = 0
    violations: list[Violation] = []
    if not root.exists():
        return total, violations

    for current_root, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current_root)
        directories[:] = [
            name for name in directories if not (current_path / name).is_symlink()
        ]
        for name in files:
            path = current_path / name
            if path.is_symlink():
                continue
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            total += size
            relative = path.relative_to(root).as_posix()
            limit = max_log_bytes if path.suffix == ".log" else max_file_bytes
            if size > limit:
                violations.append(
                    Violation(
                        kind="log_size_limit" if path.suffix == ".log" else "artifact_file_size_limit",
                        path=relative,
                        size_bytes=size,
                        limit_bytes=limit,
                    )
                )
    if total > max_total_bytes:
        violations.append(
            Violation(
                kind="artifact_size_limit",
                path=".",
                size_bytes=total,
                limit_bytes=max_total_bytes,
            )
        )
    return total, violations


def write_report(
    path: Path,
    *,
    root: Path,
    total_bytes: int,
    max_log_bytes: int,
    max_file_bytes: int,
    max_total_bytes: int,
    violations: list[Violation],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "triton-anchor-local-ci-output-limit/v1",
        "failure_code": (
            "log_size_limit"
            if any(item.kind == "log_size_limit" for item in violations)
            else "artifact_size_limit"
        ),
        "detected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(root),
        "total_bytes": total_bytes,
        "limits": {
            "log_bytes": max_log_bytes,
            "file_bytes": max_file_bytes,
            "total_bytes": max_total_bytes,
        },
        "violations": [asdict(item) for item in violations],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def compact_artifact_root(root: Path, report_source: Path) -> None:
    markers = [root / name for name in OWNERSHIP_MARKERS if (root / name).is_file()]
    if not root.is_dir() or not markers:
        raise ValueError(f"refusing to compact unmarked Local CI root: {root}")
    resolved_root = root.resolve()
    if resolved_root == Path(resolved_root.anchor):
        raise ValueError(f"refusing to compact filesystem root: {root}")
    report_payload = report_source.read_bytes()

    for child in root.iterdir():
        if child.name in OWNERSHIP_MARKERS:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child)
    (root / "output-limit.json").write_bytes(report_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check")
    check.add_argument("--root", required=True)
    check.add_argument("--max-log-bytes", required=True, type=int)
    check.add_argument("--max-file-bytes", required=True, type=int)
    check.add_argument("--max-total-bytes", required=True, type=int)
    check.add_argument("--report", required=True)

    compact = subparsers.add_parser("compact")
    compact.add_argument("--root", required=True)
    compact.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    report = Path(args.report)
    if args.command == "compact":
        try:
            compact_artifact_root(root, report)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0

    limits = (args.max_log_bytes, args.max_file_bytes, args.max_total_bytes)
    if any(value <= 0 for value in limits):
        print("output limits must be positive", file=sys.stderr)
        return 2
    total, violations = scan_tree(
        root,
        max_log_bytes=args.max_log_bytes,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
    )
    if not violations:
        return 0
    write_report(
        report,
        root=root,
        total_bytes=total,
        max_log_bytes=args.max_log_bytes,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
        violations=violations,
    )
    return LIMIT_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
