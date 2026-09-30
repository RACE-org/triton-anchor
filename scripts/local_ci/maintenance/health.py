#!/usr/bin/env python3
"""Independently collect and optionally publish worker health to Gitee."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))
from prepare.runtime import EnvironmentManager
from prepare.artifacts import atomic_json, safe_source
from prepare.control_update import read_update_request, update_request_path
from prepare.runtime_probe import runtime_status
from agent_ci.state import run_state_paths


def iso(value: float | None = None) -> str:
    return datetime.fromtimestamp(
        time.time() if value is None else value, timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect(config: dict, *, now: float | None = None, manager=None) -> dict:
    now = time.time() if now is None else now
    state = Path(config["state_dir"])
    path = state / "health/worker.json"
    worker = {}
    try:
        worker = json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    heartbeat = worker.get("heartbeat_at", 0)
    heartbeat = heartbeat if isinstance(heartbeat, (int, float)) else 0
    pid = worker.get("pid", 0)
    alive = False
    if isinstance(pid, int) and pid > 0:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            pass
    stale = not heartbeat or now - heartbeat > int(
        config.get("heartbeat_stale_seconds", 180)
    )
    tasks, recent_tasks, uploads = [], [], []
    tasks_available = (state / "runs").is_dir() and os.access(state / "runs", os.R_OK | os.X_OK)
    try:
        state_paths = run_state_paths(state)
    except OSError:
        tasks_available, state_paths = False, []
    for file in state_paths:
        try:
            record = json.loads(file.read_text())
            if not isinstance(record, dict):
                raise ValueError("Invalid local task state")
            if record.get("abandoned") and record.get("phase") not in {"publish_pending", "published"}:
                continue
            try:
                manifest = json.loads(file.with_name("task.json").read_text())
                if not isinstance(manifest, dict):
                    raise ValueError("Invalid local task manifest")
            except (OSError, ValueError):
                tasks_available = False
                manifest = {}
            budget = dict(record.get("budget") or {})
            budget.update(codex_attempts_limit=config.get("codex_attempts", 10),
                          execution_attempts_limit=config.get("execution_attempts", 3),
                          session_switches_limit=config.get("codex_session_switches", 1))
            task = {
                "task_id": record.get("task_id", file.parent.parent.name),
                "repository": manifest.get("repository"),
                "head_sha": manifest.get("head_sha", record.get("head_sha")),
                "tested_sha": manifest.get("tested_sha"),
                "pr_number": manifest.get("pr_number", 0),
                "run_id": record.get("run_id", file.parent.name),
                "stage": record.get("phase", "preparing"),
                "updated_at": record.get("updated", record.get("updated_at")),
                "last_progress_at": record.get("last_progress_at"),
                "progress_state": record.get("progress_state"),
                "budget": budget,
                "recovery": record.get("recovery") or {},
                "result_status": (record.get("detail") or {}).get("result_status"),
            }
            updated = record.get("updated", 0)
            if task["stage"] == "published":
                if isinstance(updated, (int, float)) and now - 7 * 86400 <= updated <= now:
                    recent_tasks.append(task)
            else:
                tasks.append(task)
            if task["stage"] == "publish_pending":
                delivery = record.get("delivery") or {}
                recovery = record.get("recovery") or {}
                uploads.append({
                    "task_id": task["task_id"], "run_id": task["run_id"],
                    "head_sha": task["head_sha"], "repository": task["repository"],
                    "queued_at": delivery.get("queued_at", task["updated_at"]),
                    "attempts": delivery.get("attempts", 0),
                    "next_retry_at": delivery.get("next_retry_at", recovery.get("next_retry_at")),
                    "failure_code": recovery.get("failure_code"),
                })
        except (OSError, ValueError, TypeError):
            tasks_available = False
            continue
    tasks.sort(key=lambda row: row["updated_at"] if type(row.get("updated_at")) in (int, float) else 0, reverse=True)
    recent_tasks.sort(key=lambda row: row["updated_at"], reverse=True)
    active = next((entry for entry in tasks if entry["stage"] == "running"), tasks[0] if tasks else None)
    if (active and active["stage"] == "running" and alive and not stale
            and worker.get("active_task") == active["task_id"]
            and worker.get("active_run_id", active["run_id"]) == active["run_id"]):
        for key in ("last_progress_at", "codex_alive", "codex_status"):
            if key in worker and (key != "last_progress_at" or worker[key] is not None):
                active[key] = worker[key]
    roots = {str(state)}
    storage = []
    for root in sorted(roots):
        entry = {
            "label": "state" if root == str(state) else "environment",
            "available": Path(root).exists(),
        }
        if entry["available"]:
            usage = shutil.disk_usage(root)
            entry.update(
                filesystem_total_bytes=usage.total,
                filesystem_free_bytes=usage.free,
                filesystem_used_percent=round(100 * usage.used / usage.total, 2)
                if usage.total
                else 100,
            )
        storage.append(entry)
    try:
        environments = (manager or EnvironmentManager(config, state)).health()
    except Exception as exc:
        environments = {
            "active_images": {},
            "images": [],
            "attempts": [],
            "error": type(exc).__name__,
        }
    runtime = dict(environments.get("runtime", {}))
    if config.get("runtime", {}).get("kind") == "docker-rootless":
        try:
            runtime_status(config)
            runtime.update(kind="docker-rootless", available=True, rootless=True)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            runtime.update(
                kind="docker-rootless", available=False, error=type(exc).__name__
            )
    if runtime:
        environments = {**environments, "runtime": runtime}
        if runtime.get("available") is False:
            environments.setdefault("error", "RootlessRuntimeUnavailable")
    try:
        retention = json.loads((state / "health/retention.json").read_text())
    except FileNotFoundError:
        retention = {}
    except (OSError, ValueError):
        retention = {"errors": ["unreadable"]}
    cleanup_failed = any(
        row.get("state") == "unsafe" for row in environments.get("attempts", [])
    )
    workspaces = {
        "status": "error" if cleanup_failed or retention.get("errors") else "healthy",
        "pause_intake": retention.get("pause_intake") is True,
    }
    services = []
    for name in config.get(
        "monitor_services",
        [
            "triton-anchor-local-ci.service",
            "triton-anchor-local-ci-control-update.service",
            "triton-anchor-local-ci-health.timer",
        ],
    ):
        if not isinstance(name, str) or not __import__("re").fullmatch(
            r"[A-Za-z0-9_.@-]+\.(service|timer)", name
        ):
            raise ValueError("monitor_services contains an invalid systemd unit name")
        row = {"name": name, "available": False}
        try:
            result = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    name,
                    "--property=LoadState,ActiveState,SubState,Result,Type",
                ],
                text=True,
                capture_output=True,
                timeout=5,
            )
            fields = dict(
                line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
            )
            row.update(
                available=result.returncode == 0
                and fields.get("LoadState") == "loaded",
                active_state=fields.get("ActiveState", "unknown"),
                sub_state=fields.get("SubState", "unknown"),
                result=fields.get("Result", "unknown"),
                type=fields.get("Type", "timer" if name.endswith(".timer") else "unknown"),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        services.append(row)
    control_service = next(
        (
            row
            for row in services
            if row["name"] == "triton-anchor-local-ci-control-update.service"
        ),
        {},
    )
    control_update = {
        "state": "blocked"
        if worker.get("control_update") == "blocked"
        else "idle",
        "requested_revision": None,
        "task_id": None,
        "requested_at": None,
    }
    request_path = update_request_path(config)
    if request_path.exists() or request_path.is_symlink():
        try:
            request = read_update_request(request_path)
            requested_at = request.get("requested_at")
            if (
                type(requested_at) not in (int, float)
                or not __import__("math").isfinite(requested_at)
                or requested_at < 0
            ):
                raise ValueError("Control update request time is invalid")
            state = (
                "failed"
                if control_service.get("active_state") == "failed"
                else "updating"
                if control_service.get("active_state") in {"active", "activating"}
                else "pending"
            )
            control_update = {
                "state": state,
                "requested_revision": request["revision"],
                "task_id": request["task_id"],
                "requested_at": iso(requested_at),
            }
        except (OSError, ValueError):
            control_update["state"] = "invalid"
    snapshot = {
        "schema": "triton-anchor-worker-health",
        "worker_id": config.get("worker_id", "local-ci"),
        "collected_at": iso(now),
        "state": "offline" if not alive or stale else "busy" if active else "healthy",
        "poller": {
            "alive": alive,
            "heartbeat_at": iso(heartbeat) if heartbeat else None,
            "heartbeat_stale": stale,
            "last_poll_status": "error"
            if worker.get("control_channel") == "unreachable"
            else "success" if worker.get("control_channel") == "reachable" else "unknown",
        },
        "active_task": active,
        "tasks": tasks,
        "tasks_available": tasks_available,
        "uploads_available": tasks_available,
        "recent_tasks": recent_tasks[:20],
        "thresholds": {"progress_warning_seconds": config.get("progress_warning_seconds", 1800),
                       "progress_stalled_seconds": config.get("progress_stalled_seconds", 3600),
                       "upload_seconds": 1200},
        "uploads": uploads,
        "environments": environments,
        "workspaces": workspaces,
        "storage": storage,
        "services": services,
        "control_update": control_update,
        "service_scope": "user",
        "runtime": runtime,
        "images": environments.get("images", []),
        "task_containers": environments.get("attempts", []),
    }
    return public_snapshot(snapshot)


def public_snapshot(snapshot):
    """Construct the public schema; never serialize manager/config structures."""

    def identifier(value):
        return (
            value
            if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value)
            else "unknown"
        )

    def instant(value):
        if type(value) in (int, float):
            try:
                return iso(value) if value >= 0 else None
            except (ValueError, OverflowError, OSError):
                return None
        try:
            return (
                value
                if isinstance(value, str)
                and datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo
                else None
            )
        except ValueError:
            return None

    def number(value):
        return (
            value
            if type(value) in (int, float)
            and __import__("math").isfinite(value)
            and value >= 0
            else None
        )

    def flag(value):
        return value if type(value) is bool else None

    def code(value):
        return value if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value) else None

    def sha(value):
        return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) else None

    def repository(value):
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value) else None

    def failure_code(value):
        return value if value in {
            "recovery_exhausted", "sealing_failed", "authentication", "connection", "rate_limit",
            "session_invalid", "cli_failed", "result_missing", "environment_unavailable",
            "cleanup_unconfirmed", "container_failed", "container_oom", "execution_interrupted", "disk_budget",
            "runtime_unavailable", "configuration_invalid", "publish_failed", "delivery_failed", "timeout", "control_update_failed"
        } else None

    def recovery(row):
        row = row if isinstance(row, dict) else {}
        return {
            "state": row.get("state") if row.get("state") in {
                "normal", "retry_wait", "waiting_dependency", "recovering", "recovered", "exhausted"
            } else "unknown",
            "failure_code": failure_code(row.get("failure_code")),
            "action": row.get("action") if row.get("action") in {
                "resume", "new_session", "rebuild", "wait_dependency", "retry_sealing", "retry_publish",
                "resume_codex", "new_codex_session", "rebuild_execution", "defer_wait_dependency", "no_retry",
                "wait_credentials", "wait_runtime", "publish_infra_error", "continue_sealing", "published"
            } else None,
            "next_retry_at": instant(row.get("next_retry_at")),
            "last_recovery_at": instant(row.get("last_recovery_at")),
            "outcome": row.get("outcome") if row.get("outcome") in {
                "pending", "running", "success", "failed", "recovered", "exhausted", "waiting"
            } else None,
        }

    def task(row):
        budget = row.get("budget") if isinstance(row.get("budget"), dict) else {}
        return {
            "task_id": identifier(row.get("task_id")),
            "run_id": identifier(row.get("run_id")),
            "repository": repository(row.get("repository")),
            "head_sha": sha(row.get("head_sha")),
            "tested_sha": sha(row.get("tested_sha")),
            "pr_number": number(row.get("pr_number")),
            "budget": {**{k: number(budget.get(k)) for k in (
                "codex_attempts_used", "execution_attempts_used", "session_switches",
                "codex_attempts_limit", "execution_attempts_limit", "session_switches_limit")},
                **{k: instant(budget.get(k)) for k in ("codex_deadline_at", "recovery_deadline_at")}},
            "recovery": recovery(row.get("recovery")),
            "result_status": row.get("result_status") if row.get("result_status") in {
                "pass", "fail", "infra_error", "cancelled"
            } else None,
            "stage": row.get("stage")
            if row.get("stage")
            in {"preparing", "running", "sealing", "publish_pending", "published"}
            else "unknown",
            "updated_at": instant(row.get("updated_at")),
            "last_progress_at": instant(row.get("last_progress_at")),
            "progress_state": row.get("progress_state") if row.get("progress_state") in {"normal", "delayed", "stalled_review"} else "unknown",
            "codex_alive": row.get("codex_alive")
            if type(row.get("codex_alive")) is bool
            else None,
            "codex_status": row.get("codex_status")
            if row.get("codex_status") in {
                "starting", "running", "retrying", "connection_error", "auth_error",
                "rate_limited", "session_invalid", "failed", "succeeded", "cancelled", "timeout",
            }
            else None,
        }

    poller = snapshot.get("poller", {})
    runtime = snapshot.get("runtime", {})
    environment = snapshot.get("environments", {})
    control_update = snapshot.get("control_update", {"state": "idle"})
    result = {
        "schema": "triton-anchor-worker-health",
        "worker_id": identifier(snapshot.get("worker_id")),
        "collected_at": instant(snapshot.get("collected_at")),
        "service_scope": "user",
        "state": snapshot.get("state")
        if snapshot.get("state") in {"healthy", "busy", "offline"}
        else "unknown",
        "poller": {
            "alive": flag(poller.get("alive")),
            "heartbeat_at": instant(poller.get("heartbeat_at")),
            "heartbeat_stale": flag(poller.get("heartbeat_stale")),
            "last_poll_status": poller.get("last_poll_status")
            if poller.get("last_poll_status") in {"error", "success"}
            else "unknown",
        },
        "control_update": {
            "state": control_update.get("state")
            if control_update.get("state")
            in {"idle", "pending", "updating", "failed", "invalid", "blocked"}
            else "invalid",
            "requested_revision": identifier(
                control_update.get("requested_revision")
            )
            if control_update.get("requested_revision") is not None
            else None,
            "task_id": identifier(control_update.get("task_id"))
            if control_update.get("task_id") is not None
            else None,
            "requested_at": instant(control_update.get("requested_at")),
        },
        "runtime": {
            "kind": "docker-rootless",
            "available": flag(runtime.get("available")),
            "rootless": flag(runtime.get("rootless")),
        },
        "environments": {
            "unavailable": bool(
                environment.get("error") or environment.get("unavailable")
            )
        },
        "resource_usage": [
            {k: number(r.get(k)) for k in ("cpu_percent", "memory_percent", "pids")}
            for r in environment.get(
                "resource_usage", snapshot.get("resource_usage", [])
            )
            if isinstance(r, dict)
        ],
        "resource_usage_available": environment.get(
            "resource_usage_available", snapshot.get("resource_usage_available")
        )
        is True,
        "workspaces": {
            "status": snapshot.get("workspaces", {}).get("status")
            if snapshot.get("workspaces", {}).get("status")
            in {"healthy", "error", "unreported"}
            else "unknown",
            "pause_intake": snapshot.get("workspaces", {}).get("pause_intake") is True,
        },
        "tasks": [task(r) for r in snapshot.get("tasks", []) if isinstance(r, dict)],
        "tasks_available": flag(snapshot.get("tasks_available")),
        "uploads_available": flag(snapshot.get("uploads_available")),
        "recent_tasks": [task(r) for r in snapshot.get("recent_tasks", []) if isinstance(r, dict)][:20],
        "thresholds": {k: number(snapshot.get("thresholds", {}).get(k)) for k in (
            "progress_warning_seconds", "progress_stalled_seconds", "upload_seconds")},
        "active_task": task(snapshot["active_task"])
        if isinstance(snapshot.get("active_task"), dict)
        else None,
        "uploads": [
            {
                "task_id": identifier(r.get("task_id")), "run_id": identifier(r.get("run_id")),
                "head_sha": sha(r.get("head_sha")), "repository": repository(r.get("repository")),
                "queued_at": instant(r.get("queued_at")),
                "attempts": number(r.get("attempts")),
                "next_retry_at": instant(r.get("next_retry_at")),
                "failure_code": failure_code(r.get("failure_code")),
            }
            for r in snapshot.get("uploads", [])
            if isinstance(r, dict)
        ],
        "storage": [
            {
                k: number(r.get(k))
                for k in (
                    "filesystem_total_bytes",
                    "filesystem_free_bytes",
                    "filesystem_used_percent",
                )
            }
            for r in snapshot.get("storage", [])
            if isinstance(r, dict)
        ],
        "images": [
            {
                "release_id": identifier(r.get("release_id")),
                "image_id": identifier(r.get("image_id")),
                "state": r.get("state")
                if r.get("state")
                in {"preparing", "validating", "ready", "failed", "quarantined"}
                else "unknown",
                "validated": r.get("validated") is True,
            }
            for r in snapshot.get("images", [])
            if isinstance(r, dict)
        ],
        "task_containers": [
            {"task_id": identifier(r.get("task_id")), "run_id": identifier(r.get("run_id")),
             "head_sha": sha(r.get("head_sha")),
             "repository": repository(r.get("task", {}).get("repository")) if isinstance(r.get("task"), dict)
             else repository(r.get("repository")),
             "attempt_id": identifier(r.get("attempt_id")),
             "expected_running": r.get("state") == "running" if "state" in r else flag(r.get("expected_running")),
             "available": flag(r.get("available")), "running": flag(r.get("running")),
             "status": r.get("status") if r.get("status") in {
                 "created", "running", "paused", "restarting", "removing", "exited", "dead", "missing", "removed"
             } else "unknown",
             "exit_code": number(r.get("exit_code")), "oom_killed": flag(r.get("oom_killed")),
             "finished_at": instant(r.get("finished_at")),
             **{k: number(r.get(k)) for k in ("cpu_percent", "memory_percent", "pids")}}
            for r in snapshot.get("task_containers", []) if isinstance(r, dict)
        ],
        "services": [
            {
                "name": identifier(r.get("name")),
                "available": flag(r.get("available")),
                "type": r.get("type") if r.get("type") in {
                    "simple", "exec", "forking", "oneshot", "dbus", "notify", "notify-reload", "idle", "timer"
                } else "unknown",
                "sub_state": code(r.get("sub_state")),
                "result": r.get("result") if r.get("result") in {
                    "success", "exit-code", "signal", "core-dump", "timeout", "watchdog", "start-limit-hit",
                    "resources", "protocol", "oom-kill", "exec-condition", "none"
                } else "unknown",
                "active_state": r.get("active_state")
                if r.get("active_state")
                in {"active", "inactive", "failed", "activating"}
                else "unknown",
            }
            for r in snapshot.get("services", [])
            if isinstance(r, dict)
        ],
    }
    return result


def publish_snapshot(config, document, *, branch, filename):
    """Single-writer, parentless snapshot with exact old-SHA compare-and-swap."""
    from urllib.parse import urlparse

    repository = safe_source(config.get("health_repo_url"), "health_repo_url")
    parsed = urlparse(repository)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "gitee.com"
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Health publication requires credential-free HTTPS Gitee URL")
    if not re.fullmatch(r"(?:snapshot/[A-Za-z0-9_.-]+|watchdog)", branch):
        raise ValueError("Only dedicated health snapshot refs may be replaced")
    token = os.environ.get(config.get("health_token_env", "GITEE_HEALTH_TOKEN"), "")
    if not token:
        raise ValueError("Health publishing credential is unavailable")
    with tempfile.TemporaryDirectory(prefix="local-ci-health-") as temporary:
        root = Path(temporary)
        askpass = root / "askpass.sh"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in *Username*) printf "%s\\n" "$LOCAL_CI_GIT_USER" ;; *) printf "%s\\n" "$LOCAL_CI_GIT_TOKEN" ;; esac\n'
        )
        askpass.chmod(0o700)
        env = {
            **os.environ,
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "LOCAL_CI_GIT_TOKEN": token,
            "LOCAL_CI_GIT_USER": config.get("gitee_username", "oauth2"),
        }
        checkout = root / "repo"
        checkout.mkdir()

        def git(*args):
            result = subprocess.run(
                ["git", *args],
                cwd=checkout,
                env=env,
                text=True,
                capture_output=True,
                timeout=90,
            )
            if result.returncode:
                raise RuntimeError("Gitee health snapshot operation failed")
            return result.stdout.strip()

        git("init", "-q")
        git("config", "user.name", "Local CI Health")
        git("config", "user.email", "local-ci-health@example.invalid")
        git("remote", "add", "origin", repository)
        ref = "refs/heads/" + branch
        existing = git("ls-remote", "origin", ref)
        old = existing.split()[0] if existing else ""
        atomic_json(checkout / filename, document)
        git("add", "--", filename)
        tree = git("write-tree")
        commit = git("commit-tree", tree, "-m", "Current Local CI health snapshot")
        git(
            "push",
            "--force-with-lease=" + ref + ":" + old,
            "origin",
            commit + ":" + ref,
        )


def publish(config, snapshot):
    publish_snapshot(
        config,
        public_snapshot(snapshot),
        branch="snapshot/" + config.get("worker_id", "local-ci"),
        filename="worker-health.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text())
        snapshot = collect(config)
        output = (
            Path(args.output)
            if args.output
            else Path(config["state_dir"]) / "health/worker-health.json"
        )
        atomic_json(output, snapshot)
        if args.publish:
            publish(config, snapshot)
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Local CI health collection/publishing failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
