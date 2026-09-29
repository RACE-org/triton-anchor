"""Optional, bounded health progress for the existing result receiver."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen

POLL_SECONDS = 300
MAX_BYTES = 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,160}")
PHASES = {
    "preparing": "preparing environment", "running": "running checks",
    "sealing": "sealing result", "publish_pending": "waiting for result upload",
    "publishing": "uploading result",
}
ACTIONS = {
    "resume": "resume", "resume_codex": "resume", "resume_session": "resume",
    "new_session": "new session", "new_codex_session": "new session",
    "rebuild": "rebuild environment", "rebuild_execution": "rebuild environment",
    "wait_dependency": "waiting for dependency", "wait_credentials": "waiting for credentials",
    "retry_sealing": "retry sealing", "retry_publish": "retry upload",
    "backoff": "rate limit backoff", "wait_rate_limit": "rate limit backoff",
    "wait_runtime": "waiting for runtime", "continue_sealing": "continue sealing",
    "publish_infra_error": "publishing infrastructure failure",
}


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo else None
    except (ValueError, TypeError, OverflowError):
        return None


def read_health(config):
    parsed = urlparse(config.get("health_repo_url", ""))
    worker = config.get("worker_id", "")
    if (parsed.scheme != "https" or parsed.netloc != "gitee.com"
            or not IDENTIFIER.fullmatch(worker)):
        return None
    parts = parsed.path.strip("/").removesuffix(".git").split("/")
    if len(parts) != 2 or not all(IDENTIFIER.fullmatch(p) for p in parts):
        return None
    url = (f"https://gitee.com/api/v5/repos/{parts[0]}/{parts[1]}/contents/worker-health.json"
           f"?ref={quote('snapshot/' + worker, safe='')}")
    headers = {"Accept": "application/json"}
    token = os.environ.get(config.get("health_token_env", "GITEE_HEALTH_TOKEN")) or os.environ.get("GITEE_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    # A failed progress read must never hold up the authoritative result receiver.
    with urlopen(Request(url, headers=headers), timeout=10) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Health response exceeds budget")
    content = json.loads(raw)
    if not isinstance(content, dict) or content.get("encoding") != "base64":
        return None
    return json.loads(base64.b64decode(content["content"]))


def pending_for(row, task):
    identity = parse_qs(urlparse((row or {}).get("target_url", "")).fragment)
    return bool(row and row.get("context") == "Summary"
                and row.get("state") == "pending" and isinstance(row.get("id"), int)
                and (row.get("creator") or {}).get("login") == "github-actions[bot]"
                and identity.get("local-ci-task") == [task["task_id"]])


class ReceiverProgress:
    def __init__(self, config=None):
        if config is None:
            try:
                config = json.loads((Path(__file__).resolve().parents[1]
                                     / "prepare/config.example.json").read_text())
            except (OSError, ValueError):
                config = {}
        self.config = config
        self.next_poll = 0.0
        self.last_source = 0.0
        self.run_id = ""

    def description(self, snapshot, task, now):
        if (not isinstance(snapshot, dict) or snapshot.get("worker_id") != self.config.get("worker_id")
                or snapshot.get("tasks_available") is False):
            return None
        collected = timestamp(snapshot.get("collected_at"))
        stale = float(self.config.get("health_stale_seconds", 1200))
        if collected is None or not 0 <= now - collected <= stale or collected < self.last_source:
            return None
        self.last_source = collected
        rows = [row for row in snapshot.get("tasks", []) if isinstance(row, dict)
                and all(row.get(key) == task[key] for key in
                        ("task_id", "repository", "head_sha", "tested_sha"))]
        if len(rows) != 1:
            return None
        row = rows[0]
        run_id = row.get("run_id", "")
        updated = timestamp(row.get("updated_at"))
        if (not isinstance(run_id, str) or not IDENTIFIER.fullmatch(run_id)
                or (self.run_id and run_id < self.run_id)
                or updated is None or not 0 <= now - updated <= stale):
            return None
        self.run_id = run_id
        phase = PHASES.get(row.get("stage") or row.get("execution_phase") or row.get("phase"))
        if not phase:
            return None
        recovery = row.get("recovery") if isinstance(row.get("recovery"), dict) else {}
        budget = row.get("budget") if isinstance(row.get("budget"), dict) else {}
        parts = ["Local CI: " + phase]
        state = recovery.get("state")
        if state in {"retry_wait", "waiting_dependency", "recovering", "exhausted"}:
            parts.append(state.replace("_", " "))
            if action := ACTIONS.get(recovery.get("action")):
                parts.append(action)
        attempt = budget.get("codex_attempts_used")
        if isinstance(attempt, int) and not isinstance(attempt, bool) and 0 < attempt <= 100:
            parts.append(f"Codex attempt {attempt}/{self.config.get('codex_attempts', 10)}")
        return "; ".join(parts)[:140]

    def update(self, gh, task):
        if time.monotonic() < self.next_poll:
            return
        self.next_poll = time.monotonic() + POLL_SECONDS
        try:
            before = gh.latest_summary(task)
            if not pending_for(before, task):
                return  # Progress only updates an existing pending summary.
            snapshot = read_health(self.config)
            description = self.description(snapshot, task, datetime.now(timezone.utc).timestamp())
            if not description or description == before.get("description"):
                return
            after = gh.latest_summary(task)
            if not pending_for(after, task) or after.get("id") != before.get("id"):
                return  # A final result or another receiver won while we read health.
            gh.status(task, "pending", description, after.get("target_url", ""),
                      existing_only=True, expected_pending_id=after["id"])
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError):
            # Health is best effort and contains no authoritative execution result.
            print("Worker progress unavailable; continuing result receive")
