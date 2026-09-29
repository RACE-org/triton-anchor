#!/usr/bin/env python3
"""Same-host independent watchdog: public Gitee snapshot to Gitee observation."""

from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare.artifacts import atomic_json
from maintenance.health import publish_snapshot, public_snapshot

SCHEMA = "triton-anchor-local-ci-watchdog"


def timestamp(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (AttributeError, ValueError):
        return None


def evaluate(
    document,
    previous=None,
    *,
    now=None,
    stale_seconds=1200,
    upload_seconds=1200,
    progress_seconds=1800,
    disk_free_bytes=5 * 1024**3,
    **unused,
):
    now = now or datetime.now(timezone.utc)
    previous = previous or {"active": {}, "history": []}
    observed, unknown, workers = {}, set(), []
    source_error = bool(document.get("source_error"))
    expected = document.get("expected_workers", [])
    if source_error:
        unknown.update(expected)
        unknown.update(r.get("worker_id") for r in previous.get("active", {}).values())
    snapshots = document.get("workers", [document] if document.get("worker_id") else [])
    for raw in snapshots:
        worker = public_snapshot(raw)
        workers.append(worker)
        ident = worker["worker_id"]
        collected = timestamp(worker.get("collected_at"))

        def incident(code):
            key = ident + ":" + code
            observed[key] = {"key": key, "worker_id": ident, "code": code}

        if collected is None or (now - collected).total_seconds() > stale_seconds:
            incident("snapshot_stale")
            unknown.add(ident)
            continue
        if not worker["poller"]["alive"] or worker["poller"]["heartbeat_stale"]:
            incident("poller_unavailable")
        if not worker["runtime"]["available"]:
            incident("runtime_unavailable")
        if worker["poller"]["last_poll_status"] == "error":
            incident("relay_poll_failed")
        if worker["environments"]["unavailable"]:
            incident("environment_unavailable")
        if any(row["state"] == "failed" for row in worker["images"]):
            incident("environment_update_failed")
        if any(
            (row.get("filesystem_free_bytes") or 0) < disk_free_bytes
            for row in worker["storage"]
        ):
            incident("disk_space_low")
        for row in worker["uploads"]:
            queued = timestamp(row.get("queued_at"))
            if queued is None or (now - queued).total_seconds() > upload_seconds:
                incident("delivery_pending")
        active = worker.get("active_task")
        if active and active["stage"] == "running":
            progress = timestamp(active.get("last_progress_at"))
            if progress and (now - progress).total_seconds() > progress_seconds:
                incident("task_no_progress")
    if not source_error:
        seen = {w["worker_id"] for w in workers}
        for ident in set(expected) - seen:
            unknown.add(ident)
            observed[ident + ":snapshot_stale"] = {
                "key": ident + ":snapshot_stale",
                "worker_id": ident,
                "code": "snapshot_stale",
            }
    at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    active, history = {}, list(previous.get("history", []))
    for key, row in observed.items():
        prior = previous.get("active", {}).get(key)
        active[key] = {
            **row,
            "first_detected_at": prior["first_detected_at"] if prior else at,
            "last_seen_at": at,
        }
        if not prior:
            history.append({"at": at, "key": key, "transition": "opened"})
    for key, prior in previous.get("active", {}).items():
        if key in observed:
            continue
        if prior.get("worker_id") in unknown:
            active[key] = prior
        else:
            history.append({"at": at, "key": key, "transition": "recovered"})
    return {
        "schema": SCHEMA,
        "updated_at": at,
        "source_state": "unknown" if source_error else "readable",
        "active": active,
        "history": history[-100:],
        "worker_health": workers,
        "healthy": not active and not source_error,
    }


def read_snapshot(config):
    parsed = urllib.parse.urlparse(config["health_repo_url"])
    if (
        parsed.scheme != "https"
        or parsed.hostname != "gitee.com"
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Watchdog reads a public HTTPS Gitee repository")
    repo = parsed.path.removesuffix(".git").strip("/")
    branch = "snapshot/" + config.get("worker_id", "local-ci")
    url = (
        "https://gitee.com/api/v5/repos/"
        + repo
        + "/contents/worker-health.json?"
        + urllib.parse.urlencode({"ref": branch})
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            value = json.load(response)
        if value.get("encoding") == "base64":
            value = json.loads(base64.b64decode(value["content"]))
        return {
            "workers": [value],
            "expected_workers": [config.get("worker_id", "local-ci")],
        }
    except (OSError, ValueError):
        return {
            "source_error": True,
            "expected_workers": [config.get("worker_id", "local-ci")],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    local = Path(config["state_dir"]) / "health/watchdog.json"
    previous = json.loads(local.read_text()) if local.exists() else None
    document = (
        json.loads(Path(args.input).read_text())
        if args.input
        else read_snapshot(config)
    )
    state = evaluate(
        document, previous, stale_seconds=config.get("health_stale_seconds", 1200)
    )
    atomic_json(local, state)
    if args.publish:
        try:
            publish_snapshot(config, state, branch="watchdog", filename="watchdog.json")
            local.with_name("watchdog-pending.json").unlink(missing_ok=True)
        except (OSError, ValueError, RuntimeError):
            atomic_json(local.with_name("watchdog-pending.json"), state)
            raise
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
