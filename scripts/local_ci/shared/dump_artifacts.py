#!/usr/bin/env python3
"""Preserve useful failure IR and prune Local CI-owned Triton dump state."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path


IR_SUFFIXES = {".ttir", ".linalg", ".pplir"}
TASK_DUMP_RELATIVE_DIRS = (
    Path("root/.triton/dump"),
    Path("workspace/triton-dump-dir"),
)


@dataclass(frozen=True)
class SourceFile:
    label: str
    root: Path
    path: Path
    relative_path: Path
    size: int


@dataclass(frozen=True)
class PrunePlan:
    target: Path
    files: tuple[Path, ...]
    directories: tuple[Path, ...]
    bytes: int


def safe_stage_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return safe or "unknown-stage"


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, path_text = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise ValueError(f"invalid source specification: {value!r}")
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError(f"source path must be absolute: {path}")
    return label, path


def iter_ir_files(label: str, source_root: Path) -> list[SourceFile]:
    if not source_root.exists():
        return []
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(f"source root must be a real directory: {source_root}")

    files: list[SourceFile] = []
    for current_text, directory_names, file_names in os.walk(
        source_root, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        directory_names[:] = sorted(
            name for name in directory_names if not (current / name).is_symlink()
        )
        for name in sorted(file_names):
            path = current / name
            path_stat = path.lstat()
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or path.suffix.lower() not in IR_SUFFIXES
            ):
                continue
            files.append(
                SourceFile(
                    label=label,
                    root=source_root,
                    path=path,
                    relative_path=path.relative_to(source_root),
                    size=path_stat.st_size,
                )
            )
    return files


def collect_failure_ir(args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        raise ValueError("output directory must be absolute")

    source_specs = [parse_source(value) for value in args.source]
    if len({label for label, _ in source_specs}) != len(source_specs):
        raise ValueError("source labels must be unique")

    candidates: list[SourceFile] = []
    for label, source_root in source_specs:
        try:
            output_root.resolve().relative_to(source_root.resolve())
        except ValueError:
            pass
        else:
            raise ValueError(
                f"output directory must not be inside source root: {source_root}"
            )
        candidates.extend(iter_ir_files(label, source_root))

    stage = safe_stage_name(args.stage)
    if not candidates:
        print(
            json.dumps(
                {"stage": stage, "file_count": 0, "total_bytes": 0},
                separators=(",", ":"),
            )
        )
        return 0

    stage_root = output_root / stage
    copied: list[dict[str, object]] = []
    for item in candidates:
        destination = stage_root / item.label / item.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.path, destination, follow_symlinks=False)
        copied.append(
            {
                "source": item.label,
                "source_path": item.relative_path.as_posix(),
                "artifact_path": destination.relative_to(output_root).as_posix(),
                "size_bytes": item.size,
            }
        )

    document = {
        "schema": "triton-anchor-local-ci-failure-ir/v1",
        "target_sha": args.target_sha,
        "stage": stage,
        "file_count": len(copied),
        "total_bytes": sum(int(item["size_bytes"]) for item in copied),
        "files": copied,
    }
    manifest = stage_root / "manifest.json"
    temporary_manifest = manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest)
    print(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
    return 0


def mapped_path(root: Path, relative: Path) -> Path:
    return root.joinpath(*relative.parts)


def managed_targets(root: Path) -> list[Path]:
    return [mapped_path(root, relative) for relative in TASK_DUMP_RELATIVE_DIRS]


def build_prune_plan(target: Path) -> PrunePlan | None:
    if not target.exists() and not target.is_symlink():
        return None
    target_stat = target.lstat()
    if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISDIR(target_stat.st_mode):
        raise ValueError(f"managed cleanup target must be a real directory: {target}")
    if os.path.ismount(target):
        raise ValueError(f"refusing to clean mounted directory: {target}")

    device = target_stat.st_dev
    files: list[Path] = []
    directories: list[Path] = []
    total_bytes = 0
    for current_text, directory_names, file_names in os.walk(
        target, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            child = current / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                files.append(child)
                total_bytes += child_stat.st_size
                continue
            if not stat.S_ISDIR(child_stat.st_mode):
                raise ValueError(f"unexpected directory entry type: {child}")
            if child_stat.st_dev != device or os.path.ismount(child):
                raise ValueError(f"refusing to cross nested mount: {child}")
            kept_directories.append(name)
            directories.append(child)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            child = current / name
            child_stat = child.lstat()
            if child_stat.st_dev != device and not stat.S_ISLNK(child_stat.st_mode):
                raise ValueError(f"refusing to cross filesystem boundary: {child}")
            files.append(child)
            total_bytes += child_stat.st_size
    return PrunePlan(
        target=target,
        files=tuple(files),
        directories=tuple(directories),
        bytes=total_bytes,
    )


def execute_prune_plans(plans: list[PrunePlan]) -> dict[str, int]:
    removed_files = 0
    removed_directories = 0
    removed_bytes = 0
    for plan in plans:
        for path in plan.files:
            path.unlink()
            removed_files += 1
        for path in reversed(plan.directories):
            path.rmdir()
            removed_directories += 1
        removed_bytes += plan.bytes
        if any(plan.target.iterdir()):
            raise RuntimeError(f"cleanup target is not empty: {plan.target}")
    return {
        "removed_files": removed_files,
        "removed_directories": removed_directories,
        "removed_bytes": removed_bytes,
        "cleaned_targets": len(plans),
    }


def prune(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError(f"root must be an existing absolute directory: {root}")
    root = root.resolve(strict=True)
    targets = managed_targets(root)
    plans = [plan for target in targets if (plan := build_prune_plan(target))]
    result = {"profile": args.profile, **execute_prune_plans(plans)}
    print(json.dumps(result, separators=(",", ":")))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output-dir", required=True)
    collect_parser.add_argument("--stage", required=True)
    collect_parser.add_argument("--target-sha", required=True)
    collect_parser.add_argument("--source", action="append", default=[], required=True)
    collect_parser.set_defaults(handler=collect_failure_ir)

    prune_parser = subparsers.add_parser("prune")
    prune_parser.add_argument("--profile", choices=("task-dumps",), required=True)
    prune_parser.add_argument("--root", default="/")
    prune_parser.set_defaults(handler=prune)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
