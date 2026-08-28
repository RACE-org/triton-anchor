#!/usr/bin/env python3
"""Record and publish fail-open Local CI worker health snapshots."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLLER_SCHEMA = "triton-anchor-local-ci-poller-health/v1"
ACTIVE_TASK_SCHEMA = "triton-anchor-local-ci-active-task/v1"
LAST_RESULT_SCHEMA = "triton-anchor-local-ci-last-result/v1"
SNAPSHOT_SCHEMA = "triton-anchor-local-ci-worker-health/v1"
WORKER_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
SHA_RE = re.compile(r"[0-9a-f]{40}")
TASK_STATES = {"healthy", "busy", "degraded", "offline", "unknown"}
STORAGE_MEASUREMENT_TTL_SECONDS = 1800


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def seconds_since(value: object, *, now: datetime | None = None) -> int | None:
    parsed = parse_utc(value)
    if parsed is None:
        return None
    current = now or datetime.now(timezone.utc)
    return max(0, int((current - parsed).total_seconds()))


def validate_worker_id(value: str) -> str:
    if not WORKER_ID_RE.fullmatch(value):
        raise ValueError("worker id must contain only letters, digits, dot, underscore, or hyphen")
    return value


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def process_start_ticks(pid: int) -> str:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return ""
    closing = stat_text.rfind(")")
    if closing < 0:
        return ""
    fields = stat_text[closing + 2 :].split()
    return fields[19] if len(fields) > 19 else ""


def process_matches(pid: int, expected_start_ticks: object) -> bool:
    if pid <= 0:
        return False
    actual = process_start_ticks(pid)
    return bool(actual and str(expected_start_ticks or "") == actual)


def state_path(health_dir: Path, name: str) -> Path:
    return health_dir / f"{name}.json"


def default_health_dir() -> Path:
    configured = os.getenv("LOCAL_CI_HEALTH_DIR", "")
    if configured:
        return Path(configured)
    state_dir = os.getenv("LOCAL_CI_STATE_DIR", "")
    return Path(state_dir) / "health" if state_dir else Path("./health")


@contextmanager
def state_lock(health_dir: Path, name: str) -> Iterator[None]:
    health_dir.mkdir(parents=True, exist_ok=True)
    lock_path = health_dir / f".{name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def base_poller_document(worker_id: str, pid: int) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema": POLLER_SCHEMA,
        "worker_id": validate_worker_id(worker_id),
        "pid": pid,
        "pid_start_ticks": process_start_ticks(pid),
        "started_at": now,
        "heartbeat_at": now,
        "state": "idle",
        "last_poll_started_at": "",
        "last_poll_finished_at": "",
        "last_poll_status": "not_run",
        "task_ref_count": 0,
        "last_error_code": "",
    }


def load_poller(health_dir: Path, worker_id: str, pid: int) -> dict[str, Any]:
    document = read_json(state_path(health_dir, "poller"))
    if (
        document is None
        or document.get("schema") != POLLER_SCHEMA
        or document.get("worker_id") != worker_id
        or document.get("pid") != pid
    ):
        return base_poller_document(worker_id, pid)
    return document


def poller_start(args: argparse.Namespace) -> int:
    with state_lock(args.health_dir, "poller"):
        document = base_poller_document(args.worker_id, args.pid)
        atomic_write_json(state_path(args.health_dir, "poller"), document)
    return 0


def poller_update(args: argparse.Namespace) -> int:
    with state_lock(args.health_dir, "poller"):
        document = load_poller(args.health_dir, args.worker_id, args.pid)
        now = utc_now()
        document["heartbeat_at"] = now
        if args.task_ref_count is not None:
            document["task_ref_count"] = args.task_ref_count
        if args.phase == "started":
            document["last_poll_started_at"] = now
            document["state"] = "polling"
        else:
            document["last_poll_finished_at"] = now
            document["last_poll_status"] = args.status
            document["last_error_code"] = args.error_code
            document["state"] = (
                "running"
                if state_path(args.health_dir, "active-task").is_file()
                else "idle"
            )
        atomic_write_json(state_path(args.health_dir, "poller"), document)
    return 0


def heartbeat_once(health_dir: Path, worker_id: str, parent_pid: int) -> bool:
    with state_lock(health_dir, "poller"):
        document = load_poller(health_dir, worker_id, parent_pid)
        expected_ticks = document.get("pid_start_ticks")
        if not process_matches(parent_pid, expected_ticks):
            return False
        document["heartbeat_at"] = utc_now()
        if state_path(health_dir, "active-task").is_file():
            document["state"] = "running"
        elif document.get("state") != "polling":
            document["state"] = "idle"
        atomic_write_json(state_path(health_dir, "poller"), document)
    return True


def heartbeat(args: argparse.Namespace) -> int:
    while heartbeat_once(args.health_dir, args.worker_id, args.parent_pid):
        if args.once:
            return 0
        time.sleep(args.interval)
    return 0


def task_start(args: argparse.Namespace) -> int:
    if not SHA_RE.fullmatch(args.sha):
        raise ValueError("task SHA must be a lowercase 40-character hexadecimal value")
    now = utc_now()
    document = {
        "schema": ACTIVE_TASK_SCHEMA,
        "worker_id": validate_worker_id(args.worker_id),
        "branch": args.branch,
        "sha": args.sha,
        "run_id": args.run_id,
        "profile": args.profile,
        "container": args.container,
        "execution_mode": args.execution_mode,
        "stage": args.stage,
        "started_at": now,
        "updated_at": now,
    }
    atomic_write_json(state_path(args.health_dir, "active-task"), document)
    return 0


def task_stage(args: argparse.Namespace) -> int:
    path = state_path(args.health_dir, "active-task")
    document = read_json(path)
    if document is None or document.get("schema") != ACTIVE_TASK_SCHEMA:
        return 0
    if args.run_id and document.get("run_id") != args.run_id:
        return 0
    document["stage"] = args.stage
    document["updated_at"] = utc_now()
    if args.profile:
        document["profile"] = args.profile
    if args.container:
        document["container"] = args.container
    if args.execution_mode:
        document["execution_mode"] = args.execution_mode
    atomic_write_json(path, document)
    return 0


def task_finish(args: argparse.Namespace) -> int:
    active_path = state_path(args.health_dir, "active-task")
    active = read_json(active_path) or {}
    if active and args.run_id and active.get("run_id") != args.run_id:
        return 0
    finished_at = utc_now()
    result = {
        "schema": LAST_RESULT_SCHEMA,
        "worker_id": validate_worker_id(args.worker_id),
        "branch": str(active.get("branch") or args.branch),
        "sha": str(active.get("sha") or args.sha),
        "run_id": str(active.get("run_id") or args.run_id),
        "profile": str(active.get("profile") or args.profile),
        "status": args.status,
        "exit_code": args.exit_code,
        "publish_status": args.publish_status,
        "failure_code": args.failure_code,
        "started_at": str(active.get("started_at") or ""),
        "finished_at": finished_at,
    }
    atomic_write_json(state_path(args.health_dir, "last-result"), result)
    active_path.unlink(missing_ok=True)

    poller_path = state_path(args.health_dir, "poller")
    with state_lock(args.health_dir, "poller"):
        poller = read_json(poller_path)
        if poller is not None and poller.get("schema") == POLLER_SCHEMA:
            poller["heartbeat_at"] = finished_at
            poller["state"] = "idle"
            atomic_write_json(poller_path, poller)
    return 0


def cached_storage_info(
    previous: object,
    label: str,
    configured_path: str,
    now: datetime,
) -> dict[str, Any] | None:
    if not isinstance(previous, dict):
        return None
    age = seconds_since(previous.get("measured_at"), now=now)
    directory_bytes = previous.get("directory_bytes")
    filesystem_total_bytes = previous.get("filesystem_total_bytes")
    if (
        previous.get("label") != label
        or previous.get("path") != configured_path
        or age is None
        or age > STORAGE_MEASUREMENT_TTL_SECONDS
        or not isinstance(directory_bytes, int)
        or not isinstance(filesystem_total_bytes, int)
    ):
        return None
    return {
        "label": label,
        "path": configured_path,
        "probe_path": str(previous.get("probe_path") or configured_path),
        "available": True,
        "directory_bytes": directory_bytes,
        "filesystem_total_bytes": filesystem_total_bytes,
        "directory_percent": previous.get("directory_percent"),
        "measured_at": previous.get("measured_at"),
    }


def storage_info(
    label: str,
    configured_path: str,
    previous: object,
    now: datetime,
) -> dict[str, Any]:
    if not configured_path:
        return {"label": label, "path": "", "available": False}
    path = Path(configured_path)
    if not path.exists():
        return {"label": label, "path": configured_path, "available": False}
    cached = cached_storage_info(previous, label, configured_path, now)
    if cached is not None:
        return cached
    try:
        usage = shutil.disk_usage(path)
        completed = subprocess.run(
            ["du", "-sx", "-B1", "--", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise OSError(completed.stderr.strip() or "du failed")
        directory_bytes = int(completed.stdout.split(maxsplit=1)[0])
    except (IndexError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {
            "label": label,
            "path": configured_path,
            "available": False,
            "error": str(exc)[:300],
        }
    directory_percent = (
        directory_bytes / usage.total * 100.0 if usage.total else 0.0
    )
    return {
        "label": label,
        "path": configured_path,
        "probe_path": str(path.resolve()),
        "available": True,
        "directory_bytes": directory_bytes,
        "filesystem_total_bytes": usage.total,
        "directory_percent": round(directory_percent, 2),
        "measured_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def host_artifact_path(artifact_path: str, workspace_host: str) -> str:
    if not artifact_path:
        return ""
    direct = Path(artifact_path)
    if direct.exists():
        return artifact_path
    container_workspace = os.getenv("WORKSPACE", "/workspace").rstrip("/")
    if workspace_host and artifact_path.startswith(container_workspace + "/"):
        suffix = artifact_path[len(container_workspace) + 1 :]
        return str(Path(workspace_host) / suffix)
    return artifact_path


def run_capture(command: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def docker_info(container: str, docker_bin: str) -> dict[str, Any]:
    if not container:
        return {"name": "", "available": False, "running": False}
    code, stdout, stderr = run_capture([docker_bin, "inspect", container])
    if code != 0:
        return {
            "name": container,
            "available": False,
            "running": False,
            "error": (stderr or stdout or "docker inspect failed")[:500],
        }
    try:
        values = json.loads(stdout)
        inspect = values[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return {
            "name": container,
            "available": False,
            "running": False,
            "error": "docker inspect returned invalid JSON",
        }
    state = inspect.get("State", {}) if isinstance(inspect, dict) else {}
    host_config = inspect.get("HostConfig", {}) if isinstance(inspect, dict) else {}
    nano_cpus = host_config.get("NanoCpus") or 0
    cpu_limit = float(nano_cpus) / 1_000_000_000 if isinstance(nano_cpus, int) else 0.0
    if not cpu_limit:
        quota = host_config.get("CpuQuota") or 0
        period = host_config.get("CpuPeriod") or 0
        if isinstance(quota, int) and isinstance(period, int) and quota > 0 and period > 0:
            cpu_limit = quota / period
    document: dict[str, Any] = {
        "name": container,
        "available": True,
        "running": bool(state.get("Running")),
        "status": str(state.get("Status") or "unknown"),
        "started_at": str(state.get("StartedAt") or ""),
        "oom_killed": bool(state.get("OOMKilled")),
        "restart_count": int(inspect.get("RestartCount") or 0),
        "limits": {
            "cpus": round(cpu_limit, 3) if cpu_limit else None,
            "memory_bytes": int(host_config.get("Memory") or 0) or None,
            "pids": int(host_config.get("PidsLimit") or 0) or None,
        },
        "stats": {},
    }
    if document["running"]:
        code, stdout, stderr = run_capture(
            [docker_bin, "stats", "--no-stream", "--format", "{{json .}}", container]
        )
        if code == 0:
            try:
                stats = json.loads(stdout)
            except json.JSONDecodeError:
                stats = {}
            if isinstance(stats, dict):
                document["stats"] = {
                    "cpu_percent": str(stats.get("CPUPerc") or ""),
                    "memory_usage": str(stats.get("MemUsage") or ""),
                    "memory_percent": str(stats.get("MemPerc") or ""),
                    "block_io": str(stats.get("BlockIO") or ""),
                    "network_io": str(stats.get("NetIO") or ""),
                    "pids": str(stats.get("PIDs") or ""),
                }
        else:
            document["stats_error"] = (stderr or stdout or "docker stats failed")[:500]
    return document


def snapshot(args: argparse.Namespace) -> int:
    previous_snapshot = read_json(args.output)
    poller = read_json(state_path(args.health_dir, "poller"))
    active = read_json(state_path(args.health_dir, "active-task"))
    last_result = read_json(state_path(args.health_dir, "last-result"))
    now = datetime.now(timezone.utc)

    heartbeat_age = seconds_since(poller.get("heartbeat_at"), now=now) if poller else None
    poller_alive = bool(
        poller
        and process_matches(int(poller.get("pid") or 0), poller.get("pid_start_ticks"))
    )
    heartbeat_stale = heartbeat_age is None or heartbeat_age > args.heartbeat_stale_seconds
    if active is not None:
        active = dict(active)
        active["elapsed_seconds"] = seconds_since(active.get("started_at"), now=now)

    container_name = str((active or {}).get("container") or args.container or "")
    container = docker_info(container_name, args.docker_bin)
    artifact_host = host_artifact_path(args.artifact_path, args.workspace_path)
    previous_rows = (previous_snapshot or {}).get("storage", [])
    if not isinstance(previous_rows, list):
        previous_rows = []
    previous_storage = {
        str(row.get("label")): row
        for row in previous_rows
        if isinstance(row, dict)
    }
    storage = [
        storage_info("state", args.state_path, previous_storage.get("state"), now),
        storage_info(
            "workspace", args.workspace_path, previous_storage.get("workspace"), now
        ),
        storage_info(
            "artifacts", artifact_host, previous_storage.get("artifacts"), now
        ),
    ]

    if poller is None:
        overall_state = "unknown"
    elif not poller_alive or heartbeat_stale:
        overall_state = "offline"
    elif container_name and not container.get("running"):
        overall_state = "degraded"
    elif poller.get("last_poll_status") == "error":
        overall_state = "degraded"
    elif active is not None:
        overall_state = "busy"
    else:
        overall_state = "healthy"
    if overall_state not in TASK_STATES:
        overall_state = "unknown"

    document = {
        "schema": SNAPSHOT_SCHEMA,
        "data_mode": "live",
        "worker_id": validate_worker_id(args.worker_id),
        "profile": str((active or {}).get("profile") or args.profile or "unknown"),
        "state": overall_state,
        "collected_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "poller": {
            **(poller or {}),
            "alive": poller_alive,
            "heartbeat_age_seconds": heartbeat_age,
            "heartbeat_stale": heartbeat_stale,
        },
        "active_task": active,
        "last_result": last_result,
        "container": container,
        "storage": storage,
    }
    atomic_write_json(args.output, document)
    return 0


def git_run(
    args: list[str], cwd: Path, env: dict[str, str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=check, text=True, capture_output=True
    )


def git_environment(directory: Path, token: str, username: str) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        askpass = directory / "gitee-health-askpass.sh"
        askpass.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *Username*) printf '%s\\n' \"$GITEE_USERNAME\" ;;\n"
            "  *) printf '%s\\n' \"$GITEE_TOKEN\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        env["GIT_ASKPASS"] = str(askpass)
        env["GITEE_USERNAME"] = username
        env["GITEE_TOKEN"] = token
    return env


def publish(args: argparse.Namespace) -> int:
    document = read_json(args.input)
    if document is None or document.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("input is not a Local CI worker health snapshot")
    branch_check = subprocess.run(
        ["git", "check-ref-format", f"refs/heads/{args.branch}"], check=False
    )
    if branch_check.returncode != 0:
        raise ValueError("health branch is not a valid Git ref")
    token = os.getenv("GITEE_TOKEN", "")
    username = os.getenv("GITEE_USERNAME", "")
    with tempfile.TemporaryDirectory(prefix="triton-anchor-worker-health-") as temporary:
        root = Path(temporary)
        repository = root / "repo"
        repository.mkdir()
        env = git_environment(root, token, username)
        git_run(["init", "-q"], repository, env)
        git_run(["config", "user.name", "triton-anchor-local-ci"], repository, env)
        git_run(
            ["config", "user.email", "triton-anchor-local-ci@example.invalid"],
            repository,
            env,
        )
        git_run(["checkout", "-q", "--orphan", "worker-health"], repository, env)
        shutil.copy2(args.input, repository / "worker-health.json")
        git_run(["add", "worker-health.json"], repository, env)
        git_run(
            ["commit", "-q", "-m", f"health: {document.get('worker_id', 'worker')}"],
            repository,
            env,
        )
        git_run(["remote", "add", "origin", args.repo_url], repository, env)
        push = git_run(
            ["push", "--force", "origin", f"HEAD:refs/heads/{args.branch}"],
            repository,
            env,
            check=False,
        )
        if push.returncode != 0:
            print(push.stderr or push.stdout, file=sys.stderr)
            return 1
    return 0


def add_health_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--health-dir",
        type=Path,
        default=default_health_dir(),
    )


def add_worker_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--worker-id",
        default=os.getenv("LOCAL_CI_WORKER_ID", "local-ci-worker"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("poller-start")
    add_health_dir(start)
    add_worker_id(start)
    start.add_argument("--pid", type=int, required=True)
    start.set_defaults(handler=poller_start)

    update = subparsers.add_parser("poller-update")
    add_health_dir(update)
    add_worker_id(update)
    update.add_argument("--pid", type=int, required=True)
    update.add_argument("--phase", choices=("started", "finished"), required=True)
    update.add_argument("--status", choices=("success", "error"), default="success")
    update.add_argument("--task-ref-count", type=int)
    update.add_argument("--error-code", default="")
    update.set_defaults(handler=poller_update)

    heartbeat_parser = subparsers.add_parser("heartbeat")
    add_health_dir(heartbeat_parser)
    add_worker_id(heartbeat_parser)
    heartbeat_parser.add_argument("--parent-pid", type=int, required=True)
    heartbeat_parser.add_argument("--interval", type=int, default=60)
    heartbeat_parser.add_argument("--once", action="store_true")
    heartbeat_parser.set_defaults(handler=heartbeat)

    start_task = subparsers.add_parser("task-start")
    add_health_dir(start_task)
    add_worker_id(start_task)
    start_task.add_argument("--branch", required=True)
    start_task.add_argument("--sha", required=True)
    start_task.add_argument("--run-id", required=True)
    start_task.add_argument("--profile", default="resolving")
    start_task.add_argument("--container", default="")
    start_task.add_argument("--execution-mode", default="full")
    start_task.add_argument("--stage", default="preparing")
    start_task.set_defaults(handler=task_start)

    stage_task = subparsers.add_parser("task-stage")
    add_health_dir(stage_task)
    stage_task.add_argument("--run-id", default="")
    stage_task.add_argument("--stage", required=True)
    stage_task.add_argument("--profile", default="")
    stage_task.add_argument("--container", default="")
    stage_task.add_argument("--execution-mode", default="")
    stage_task.set_defaults(handler=task_stage)

    finish_task = subparsers.add_parser("task-finish")
    add_health_dir(finish_task)
    add_worker_id(finish_task)
    finish_task.add_argument("--branch", default="")
    finish_task.add_argument("--sha", default="")
    finish_task.add_argument("--run-id", required=True)
    finish_task.add_argument("--profile", default="unknown")
    finish_task.add_argument(
        "--status", choices=("success", "failure", "error"), required=True
    )
    finish_task.add_argument("--exit-code", type=int, required=True)
    finish_task.add_argument("--publish-status", type=int, default=-1)
    finish_task.add_argument("--failure-code", default="")
    finish_task.set_defaults(handler=task_finish)

    snapshot_parser = subparsers.add_parser("snapshot")
    add_health_dir(snapshot_parser)
    add_worker_id(snapshot_parser)
    snapshot_parser.add_argument("--output", type=Path)
    snapshot_parser.add_argument(
        "--profile",
        default=os.getenv("LOCAL_CI_PROFILE_NAME")
        or os.getenv("BACKEND_PROFILE", "unknown"),
    )
    snapshot_parser.add_argument(
        "--container", default=os.getenv("LOCAL_CI_CONTAINER", "")
    )
    snapshot_parser.add_argument(
        "--state-path", default=os.getenv("LOCAL_CI_STATE_DIR", "")
    )
    snapshot_parser.add_argument(
        "--workspace-path", default=os.getenv("LOCAL_CI_WORKSPACE_HOST", "")
    )
    snapshot_parser.add_argument(
        "--artifact-path", default=os.getenv("LOCAL_CI_ARTIFACT_ROOT", "")
    )
    snapshot_parser.add_argument(
        "--heartbeat-stale-seconds",
        type=int,
        default=int(os.getenv("LOCAL_CI_HEARTBEAT_STALE_SECONDS", "180")),
    )
    snapshot_parser.add_argument("--docker-bin", default="docker")
    snapshot_parser.set_defaults(handler=snapshot)

    publish_parser = subparsers.add_parser("publish")
    add_health_dir(publish_parser)
    publish_parser.add_argument("--input", type=Path)
    publish_parser.add_argument(
        "--repo-url",
        default=os.getenv("GITEE_WORKER_HEALTH_REPO_URL", ""),
        required=False,
    )
    publish_parser.add_argument(
        "--branch",
        default=os.getenv("GITEE_WORKER_HEALTH_BRANCH", "race-org-localci"),
    )
    publish_parser.set_defaults(handler=publish)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(args, "health_dir"):
        args.health_dir = args.health_dir.resolve()
    if args.command == "snapshot" and args.output is None:
        args.output = state_path(args.health_dir, "snapshot")
    if args.command == "publish" and args.input is None:
        args.input = state_path(args.health_dir, "snapshot")
    if hasattr(args, "worker_id"):
        args.worker_id = validate_worker_id(args.worker_id)
    if args.command == "heartbeat" and args.interval <= 0:
        raise ValueError("heartbeat interval must be positive")
    if args.command == "snapshot" and args.heartbeat_stale_seconds <= 0:
        raise ValueError("heartbeat stale interval must be positive")
    if args.command == "publish" and (not args.repo_url or not args.branch):
        raise ValueError("health publish requires a repository URL and branch")
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Local CI health error: {exc}", file=sys.stderr)
        raise SystemExit(2)
