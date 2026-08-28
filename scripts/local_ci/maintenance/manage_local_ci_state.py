#!/usr/bin/env python3
"""Apply bounded retention to Local CI-owned state and ephemeral Docker objects."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA = "triton-anchor-local-ci-maintenance/v1"


@dataclass(frozen=True)
class Candidate:
    category: str
    path: str
    reason: str
    age_seconds: int
    bytes: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def directory_size(path: Path) -> int:
    if path.is_symlink():
        return path.lstat().st_size
    total = 0
    for root_text, directories, files in os.walk(path, followlinks=False):
        root = Path(root_text)
        directories[:] = [name for name in directories if not (root / name).is_symlink()]
        for name in files:
            child = root / name
            try:
                total += child.lstat().st_size
            except FileNotFoundError:
                continue
    return total


def age_seconds(path: Path, now: datetime) -> int:
    modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return max(int((now - modified).total_seconds()), 0)


def completed_status(result_path: Path) -> int | None:
    try:
        value = json.loads(result_path.read_text(encoding="utf-8"))["status"]
        return int(value)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def summary_status(summary_path: Path) -> int | None:
    try:
        lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("status:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def retention_seconds(status: int, success_days: int, failure_days: int) -> int:
    return (success_days if status == 0 else failure_days) * 86400


def iter_child_directories(root: Path) -> Iterable[Path]:
    if not root.is_dir() or root.is_symlink():
        return ()
    return tuple(
        path for path in root.iterdir() if path.is_dir() and not path.is_symlink()
    )


def collect_filesystem_candidates(
    *,
    state_dir: Path,
    artifact_roots: list[Path],
    now: datetime,
    success_days: int,
    failure_days: int,
    incomplete_days: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    statuses_by_run_id: dict[str, tuple[int, int]] = {}
    incomplete_limit = incomplete_days * 86400
    runs_root = state_dir / "runs"

    for branch_dir in iter_child_directories(runs_root):
        for run_dir in iter_child_directories(branch_dir):
            result_path = run_dir / "result.json"
            status = completed_status(result_path) if result_path.is_file() else None
            reference = result_path if status is not None else run_dir
            age = age_seconds(reference, now)
            if status is None:
                if age >= incomplete_limit:
                    candidates.append(
                        Candidate(
                            "run",
                            str(run_dir),
                            "incomplete_retention",
                            age,
                            directory_size(run_dir),
                        )
                    )
                continue
            statuses_by_run_id[run_dir.name] = (status, age)
            if age >= retention_seconds(status, success_days, failure_days):
                candidates.append(
                    Candidate(
                        "run",
                        str(run_dir),
                        "success_retention" if status == 0 else "failure_retention",
                        age,
                        directory_size(run_dir),
                    )
                )

    for runner_dir in iter_child_directories(state_dir / "runner"):
        linked = statuses_by_run_id.get(runner_dir.name)
        if linked is None:
            age = age_seconds(runner_dir, now)
            expired = age >= incomplete_limit
            reason = "unmatched_retention"
        else:
            status, age = linked
            expired = age >= retention_seconds(status, success_days, failure_days)
            reason = "success_retention" if status == 0 else "failure_retention"
        if expired:
            candidates.append(
                Candidate(
                    "runner",
                    str(runner_dir),
                    reason,
                    age,
                    directory_size(runner_dir),
                )
            )

    for artifact_root in artifact_roots:
        for artifact_dir in iter_child_directories(artifact_root):
            summary = artifact_dir / "delivery-summary.txt"
            status = summary_status(summary) if summary.is_file() else None
            reference = summary if status is not None else artifact_dir
            age = age_seconds(reference, now)
            if status is None:
                expired = age >= incomplete_limit
                reason = "incomplete_retention"
            else:
                expired = age >= retention_seconds(status, success_days, failure_days)
                reason = "success_retention" if status == 0 else "failure_retention"
            if expired:
                candidates.append(
                    Candidate(
                        "artifact",
                        str(artifact_dir),
                        reason,
                        age,
                        directory_size(artifact_dir),
                    )
                )

    for workspace in iter_child_directories(state_dir / "codex-workspaces"):
        age = age_seconds(workspace, now)
        if age >= incomplete_limit:
            candidates.append(
                Candidate(
                    "codex_workspace",
                    str(workspace),
                    "unmatched_retention",
                    age,
                    directory_size(workspace),
                )
            )
    return candidates


def parse_docker_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def docker_ids(kind: str, label: str) -> list[str]:
    command = ["docker", kind, "--filter", f"label={label}", "-q"]
    if kind == "ps":
        command.insert(2, "-a")
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def docker_inspect(kind: str, resource_id: str) -> dict[str, object]:
    result = subprocess.run(
        ["docker", kind, "inspect", resource_id],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
        raise ValueError(f"unexpected Docker inspect result for {resource_id}")
    return payload[0]


def image_container_ids(image_id: str) -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"ancestor={image_id}", "-q"],
        text=True,
        capture_output=True,
        check=True,
    )
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def collect_docker_candidates(now: datetime, grace_hours: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    grace = grace_hours * 3600
    for resource_id in docker_ids("ps", "triton-anchor.role=codex-ai"):
        metadata = docker_inspect("container", resource_id)
        state = metadata.get("State")
        if not isinstance(state, dict) or state.get("Running") is not False:
            continue
        age = max(
            int(
                (
                    now - parse_docker_time(str(metadata.get("Created", "")))
                ).total_seconds()
            ),
            0,
        )
        if age >= grace:
            candidates.append(
                Candidate("container", resource_id, "docker_orphan_grace", age, 0)
            )

    for resource_id in docker_ids("images", "triton-anchor.role=codex-ai-snapshot"):
        if image_container_ids(resource_id):
            continue
        metadata = docker_inspect("image", resource_id)
        age = max(
            int(
                (
                    now - parse_docker_time(str(metadata.get("Created", "")))
                ).total_seconds()
            ),
            0,
        )
        if age >= grace:
            candidates.append(
                Candidate("image", resource_id, "docker_orphan_grace", age, 0)
            )
    return candidates


def remove_candidate(candidate: Candidate) -> None:
    if candidate.category == "container":
        subprocess.run(["docker", "rm", "-f", candidate.path], check=True)
        return
    if candidate.category == "image":
        subprocess.run(["docker", "image", "rm", "-f", candidate.path], check=True)
        return
    path = Path(candidate.path)
    if path.is_symlink():
        raise ValueError(f"refusing to remove symlinked maintenance target: {path}")
    shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-dir",
        default=os.getenv("LOCAL_CI_STATE_DIR", "/root/projects/test/local-ci-state"),
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help="Repeat for each host-side Local CI artifact root.",
    )
    parser.add_argument(
        "--success-days",
        type=int,
        default=int(os.getenv("LOCAL_CI_SUCCESS_RETENTION_DAYS", "14")),
    )
    parser.add_argument(
        "--failure-days",
        type=int,
        default=int(os.getenv("LOCAL_CI_FAILURE_RETENTION_DAYS", "28")),
    )
    parser.add_argument(
        "--incomplete-days",
        type=int,
        default=int(os.getenv("LOCAL_CI_INCOMPLETE_RETENTION_DAYS", "7")),
    )
    parser.add_argument(
        "--docker-orphan-grace-hours",
        type=int,
        default=int(os.getenv("LOCAL_CI_DOCKER_ORPHAN_GRACE_HOURS", "72")),
    )
    parser.add_argument("--report", default="")
    parser.add_argument("--now", default="", help=argparse.SUPPRESS)
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def configured_artifact_roots(cli_roots: list[str]) -> list[Path]:
    values = list(cli_roots)
    values.extend(
        value
        for value in os.getenv("LOCAL_CI_ARTIFACT_HOST_ROOTS", "").split(os.pathsep)
        if value
    )
    return list(dict.fromkeys(Path(value).resolve() for value in values))


def main() -> int:
    args = parse_args()
    if min(args.success_days, args.failure_days, args.incomplete_days) <= 0:
        print("retention days must be positive", file=sys.stderr)
        return 2
    if args.docker_orphan_grace_hours <= 0:
        print("Docker orphan grace must be positive", file=sys.stderr)
        return 2

    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    artifact_roots = configured_artifact_roots(args.artifact_root)
    now = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else utc_now()
    report_path = (
        Path(args.report).resolve()
        if args.report
        else state_dir / "maintenance" / "latest.json"
    )
    started = utc_now()
    errors: list[str] = []
    candidates = collect_filesystem_candidates(
        state_dir=state_dir,
        artifact_roots=artifact_roots,
        now=now,
        success_days=args.success_days,
        failure_days=args.failure_days,
        incomplete_days=args.incomplete_days,
    )
    if not args.skip_docker and shutil.which("docker"):
        try:
            candidates.extend(
                collect_docker_candidates(now, args.docker_orphan_grace_hours)
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"Docker inspection failed: {exc}")

    removed: list[Candidate] = []
    if args.apply:
        for candidate in candidates:
            try:
                remove_candidate(candidate)
                removed.append(candidate)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors.append(f"{candidate.category} {candidate.path}: {exc}")

    payload = {
        "schema": SCHEMA,
        "mode": "apply" if args.apply else "dry_run",
        "status": "error" if errors else "ok",
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": {
            "success_days": args.success_days,
            "failure_days": args.failure_days,
            "incomplete_days": args.incomplete_days,
            "docker_orphan_grace_hours": args.docker_orphan_grace_hours,
        },
        "roots": {
            "state": str(state_dir),
            "artifacts": [str(path) for path in artifact_roots],
        },
        "candidate_count": len(candidates),
        "removed_count": len(removed),
        "reclaimed_bytes": sum(item.bytes for item in removed),
        "candidates": [asdict(item) for item in candidates],
        "errors": errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
