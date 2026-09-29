"""Task lifecycle and retryable result publication state."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .protocol import (
    ContractError,
    atomic_json,
    canonical,
    current_key,
    result_task_prefix,
    result_task_prefixes,
)


def run_state_paths(state_dir):
    """Known run layouts only; never descend into logs or task artifacts."""
    runs = Path(state_dir) / "runs"
    return sorted(
        path
        for pattern in ("*/*/state.json", "pr/*/*/*/*/state.json", "push/*/*/*/state.json")
        for path in runs.glob(pattern)
    )


def local_run_dir(state_dir, task, run_id):
    root = Path(state_dir)
    for prefix in result_task_prefixes(task):
        existing = root / prefix / run_id
        if existing.is_dir():
            return existing
    return root / result_task_prefix(task) / run_id


class Journal:
    """Host-only state. One worker process owns the lock; its threads serialize here."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.runs = self.root / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        # Several frozen tasks can share a head SHA. Index manifests, not folder names.
        self._task_runs = {}
        for path in sorted(run_state_paths(self.root), key=lambda p: p.parent.name):
            record = json.loads(path.read_text())
            self._task_runs[record["task_id"]] = path.parent
        self.guard = threading.RLock()

    @staticmethod
    def _component(value):
        if (
            not isinstance(value, str)
            or not value
            or Path(value).name != value
            or value in {".", ".."}
        ):
            raise ContractError("Invalid task/run path component")
        return value

    def run_dir(self, task_id, run_id=None):
        directory = self._task_runs.get(self._component(task_id))
        if directory is None:
            raise ContractError("Unknown task")
        if run_id and self._component(run_id) != directory.name:
            task = json.loads((directory / "task.json").read_text())
            candidate = local_run_dir(self.root, task, run_id)
            manifest = json.loads((candidate / "task.json").read_text())
            if manifest["task_id"] != task_id:
                raise ContractError("Run belongs to another task")
            return candidate
        return directory

    def has_task(self, task_id):
        return task_id in self._task_runs

    def _state(self, task_id, run_id=None):
        return json.loads((self.run_dir(task_id, run_id) / "state.json").read_text())

    def _write(self, task_id, state):
        state["updated"] = time.time()
        atomic_json(self.run_dir(task_id, state["run_id"]) / "state.json", state)

    def _new(self, task, previous=None):
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        self._component(task["task_id"])
        directory = local_run_dir(self.root, task, run_id)
        for name in ("logs", "artifacts"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "task.json", task)
        atomic_json(
            directory / "state.json",
            {
                "task_id": task["task_id"],
                "head_sha": task["head_sha"],
                "run_id": run_id,
                "previous_run_id": (previous or {}).get("run_id"),
                "phase": "preparing",
                "updated": time.time(),
                "detail": {},
                "budget": (previous or {}).get("budget", {
                    "codex_attempts_used": 0, "codex_deadline_at": None,
                    "execution_attempts_used": 0, "session_switches": 0,
                    "recovery_deadline_at": None,
                }),
                "recovery": (previous or {}).get("recovery", {"state": "normal"}),
                "last_progress_at": (previous or {}).get("last_progress_at"),
                "resume_no_progress_attempts": (previous or {}).get("resume_no_progress_attempts", 0),
                "next_session_mode": (previous or {}).get("next_session_mode", "resume_if_available"),
                "events": (previous or {}).get("events", []),
                "delivery": None,
            },
        )
        self._task_runs[task["task_id"]] = directory
        return self.task(task["task_id"])

    def register(self, task):
        with self.guard:
            if self.has_task(task["task_id"]):
                row = self.task(task["task_id"])
                if json.loads(row["manifest"]) != task:
                    raise ContractError("Immutable task manifest changed")
                return row
            return self._new(task)

    def task(self, task_id, run_id=None):
        with self.guard:
            state = self._state(task_id, run_id)
            task = json.loads((self.run_dir(task_id, run_id) / "task.json").read_text())
            return {
                "task_id": task_id,
                "head_sha": task["head_sha"],
                "subject": current_key(task),
                "manifest": canonical(task).decode(),
                "run_id": state["run_id"],
                "phase": state["phase"],
                "updated": state["updated"],
                "detail": canonical(state.get("detail", {})).decode(),
                **{key: state.get(key) for key in ("budget", "recovery", "last_progress_at")},
            }

    def tasks(self, *, active=True):
        with self.guard:
            rows = [self.task(task_id) for task_id in self._task_runs]
            return sorted(
                [r for r in rows if not active or r["phase"] != "published"],
                key=lambda r: r["updated"],
            )

    def phase(self, task_id, phase, detail=None, *, run_id=None):
        if phase not in {
            "preparing",
            "running",
            "sealing",
            "publish_pending",
            "published",
        }:
            raise ContractError("Unknown run phase: " + phase)
        with self.guard:
            state = self._state(task_id, run_id)
            state.update(phase=phase, detail=detail or {})
            self._write(task_id, state)

    def event(self, task_id, kind, detail, *, run_id=None):
        with self.guard:
            try:
                state = self._state(task_id, run_id)
            except ContractError:
                return  # Invalid/unregistered remote input has no task state.
            state["events"] = (
                state.get("events", [])
                + [{"at": time.time(), "kind": kind, "detail": detail, "run_id": state["run_id"]}]
            )[-100:]
            self._write(task_id, state)

    def queue_result(self, task_id, path, result_digest, *, run_id=None):
        with self.guard:
            state = self._state(task_id, run_id)
            saved = state.get("delivery")
            if saved and saved["digest"] != result_digest:
                raise ContractError("A sealed result cannot be rewritten")
            state["delivery"] = saved or {
                "payload_path": str(path),
                "digest": result_digest,
                "attempts": 0,
                "queued_at": time.time(),
                "published": None,
            }
            state["phase"] = (
                "published" if state["delivery"]["published"] else "publish_pending"
            )
            self._write(task_id, state)

    def delivery(self, task_id, run_id=None):
        with self.guard:
            return self._state(task_id, run_id).get("delivery")

    @staticmethod
    def result_status(delivery):
        try:
            data = Path(delivery["payload_path"]).read_bytes()
            return (
                json.loads(data).get("status")
                if hashlib.sha256(data).hexdigest() == delivery["digest"]
                else None
            )
        except (OSError, ValueError, TypeError):
            return None

    def published(self, task_id, run_id=None):
        with self.guard:
            state = self._state(task_id, run_id)
            if not state.get("delivery"):
                raise ContractError("Cannot complete delivery without a sealed result")
            state["delivery"]["published"] = (
                state["delivery"].get("published") or time.time()
            )
            state.update(
                phase="published",
                detail={
                    "completion_boundary": "gitee_upload",
                    "result_status": self.result_status(state["delivery"]),
                },
            )
            self._write(task_id, state)

    def publication_failure(self, task_id, run_id=None, *, next_retry_at=None):
        with self.guard:
            state = self._state(task_id, run_id)
            state["delivery"]["attempts"] += 1
            state["delivery"]["next_retry_at"] = next_retry_at
            self._write(task_id, state)
            return state["delivery"]["attempts"]

    def restart(self, task_id):
        """Abandon an unsealed environment after Worker restart; never reuse its installs."""
        with self.guard:
            state = self._state(task_id)
            if state.get("delivery"):
                return self.task(task_id)
            state["abandoned"] = True
            state["detail"] = {"reason": "worker_restart", "verification": "incomplete"}
            self._write(task_id, state)
            return self._new(json.loads(self.task(task_id)["manifest"]), state)

    def resume(self, task_id):
        with self.guard:
            state = self._state(task_id)
            delivery = state.get("delivery")
            if delivery and delivery["published"] is None:
                delivery["next_retry_at"] = None
                state.update(phase="publish_pending", detail={"reason": "retry_saved_upload"})
                self._write(task_id, state)
                return
            if state.get("checkpoint", {}).get("complete") and not delivery:
                if state.get("recovery", {}).get("state") == "exhausted":
                    raise ContractError("结果封存预算已耗尽，原始报告已保留；需人工处理，不自动重新测试")
                state["seal_next_retry_at"] = None
                state["recovery"] = {"state": "recovering", "action": "retry_sealing"}
                self._write(task_id, state)
                return
            if delivery and self.result_status(delivery) != "infra_error":
                raise ContractError(
                    "Only infrastructure results may be explicitly rerun"
                )
            budget = state.get("budget")
            if not budget:
                raise ContractError("旧任务无可靠恢复预算，请重新派发任务")
            if state.get("recovery", {}).get("state") == "exhausted":
                raise ContractError("任务恢复预算已耗尽，请重新派发任务")
            state["recovery"] = {**state.get("recovery", {}), "state": "recovering", "next_retry_at": None, "manual_resume": True}
            self._write(task_id, state)
            if delivery:
                self._new(json.loads(self.task(task_id)["manifest"]), state)

    def record(self, task_id, run_id=None):
        with self.guard:
            return self._state(task_id, run_id)

    def all_runs(self, *, active=True):
        with self.guard:
            rows = []
            for path in run_state_paths(self.root):
                state = json.loads(path.read_text())
                if not active or state.get("phase") != "published":
                    rows.append(self.task(state["task_id"], state["run_id"]))
            return rows

    def update(self, task_id, *, run_id=None, **fields):
        with self.guard:
            state = self._state(task_id, run_id)
            state.update(fields)
            self._write(task_id, state)
            return state

    def claim(self, task_id, counter, maximum, *, timeout=None):
        """Persist usage before starting work; restart never refunds a claimed attempt."""
        with self.guard:
            state = self._state(task_id)
            budget = state.get("budget")
            if not budget:
                raise ContractError("旧任务无可靠恢复预算，请重新派发任务")
            now = time.time()
            deadlines = [budget.get(k) for k in ("codex_deadline_at", "recovery_deadline_at")]
            if any(value is not None and value <= now for value in deadlines):
                raise ContractError("任务恢复时间预算已耗尽")
            if budget.get(counter, 0) >= maximum:
                raise ContractError("任务恢复次数预算已耗尽")
            budget[counter] = budget.get(counter, 0) + 1
            if timeout is not None and budget.get("codex_deadline_at") is None:
                budget["codex_deadline_at"] = now + timeout
            self._write(task_id, state)
            return budget.copy()

    def recover_budget(self, task_id, *, timeout=21600):
        """Migrate only explicit, contiguous start evidence; exit logs cannot date starts."""
        with self.guard:
            state = self._state(task_id)
            if state.get("budget"):
                return state["budget"]
            starts = {}
            deadlines = set()
            executions = set()
            switches = set()
            for path in run_state_paths(self.root):
                row = json.loads(path.read_text())
                if row.get("task_id") != task_id:
                    continue
                for event in row.get("events", []):
                    if event.get("kind") != "codex_start":
                        continue
                    number = event.get("detail", {}).get("attempt")
                    at = event.get("at")
                    if type(number) is int and number > 0 and type(at) in (int, float) and at > 0:
                        detail = event["detail"]
                        deadline = detail.get("deadline_at")
                        execution = detail.get("execution_attempt")
                        switched = detail.get("session_switches")
                        if (type(deadline) not in (int, float) or deadline <= at
                                or type(execution) is not int or execution < 1
                                or type(switched) is not int or switched < 0):
                            return None
                        starts[number] = min(starts.get(number, at), at)
                        deadlines.add(deadline)
                        executions.add(execution)
                        switches.add(switched)
            if not starts or set(starts) != set(range(1, max(starts)+1)) or len(deadlines) != 1:
                return None
            deadline = deadlines.pop()
            budget = dict(codex_attempts_used=max(starts), codex_deadline_at=deadline,
                          execution_attempts_used=max(executions), session_switches=max(switches),
                          recovery_deadline_at=deadline)
            state["budget"] = budget
            self._write(task_id, state)
            return budget
