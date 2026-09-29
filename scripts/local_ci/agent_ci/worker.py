#!/usr/bin/env python3
"""Poll Gitee, prepare a task for Codex, then publish its selected results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from agent_ci.codex import CodexDriver
from agent_ci.credentials import CredentialValidationError
from prepare.artifacts import EnvironmentError as RuntimeEnvironmentError
from agent_ci.delivery import MAX_RESULT_BYTES, seal_result
from agent_ci.executor import DockerExecutor
from agent_ci.policy import changed_files, minimum_checks
from agent_ci.protocol import ID, SHA, ContractError, atomic_json, is_legacy_task, validate_task, validate_result
from agent_ci.relay import GitRelay
from agent_ci.state import Journal
from agent_ci.recovery import diagnose_report, completed_tools, retry_observation, recovery_prompt, write_handoff
from prepare.control_update import (
    REQUEST_SCHEMA,
    control_request_plan,
    read_update_request,
    update_request_lock_path,
    update_request_path,
)


CONTROL_UPDATE_UNIT = "triton-anchor-local-ci-control-update.service"


def trigger_control_update(config: dict, request: dict, *, run=subprocess.run) -> str:
    """Start the exact-revision updater after the Worker releases its control lock."""
    import fcntl

    revision = request.get("revision")
    task_id = request.get("task_id")
    if (
        not isinstance(revision, str)
        or not SHA.fullmatch(revision)
        or not isinstance(task_id, str)
        or not ID.fullmatch(task_id)
    ):
        raise ValueError("Control update trigger requires an exact revision")
    request_path = update_request_path(config)
    request_lock = update_request_lock_path(config)
    request_lock.parent.mkdir(parents=True, exist_ok=True)
    with request_lock.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        requested_at = time.time()
        try:
            existing = read_update_request(request_path)
            if (
                existing["revision"] == revision
                and existing["task_id"] == task_id
                and type(existing.get("requested_at")) in (int, float)
                and existing["requested_at"] >= 0
            ):
                requested_at = existing["requested_at"]
        except (OSError, ValueError):
            pass
        atomic_json(
            request_path,
            {
                "schema": REQUEST_SCHEMA,
                "revision": revision,
                "task_id": task_id,
                "requested_at": requested_at,
            },
        )
    run(
        ["systemctl", "--user", "start", "--no-block", CONTROL_UPDATE_UNIT],
        check=True,
        timeout=30,
    )
    return CONTROL_UPDATE_UNIT


def complete_current_control_request(config: dict, current_revision: str | None) -> bool:
    """Remove a request already satisfied by this running Worker revision."""
    import fcntl

    if not isinstance(current_revision, str) or not SHA.fullmatch(current_revision):
        return False
    request_path = update_request_path(config)
    request_lock = update_request_lock_path(config)
    request_lock.parent.mkdir(parents=True, exist_ok=True)
    with request_lock.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not request_path.exists():
            return False
        try:
            request = read_update_request(request_path)
        except (OSError, ValueError):
            return False
        if request["revision"] != current_revision:
            return False
        request_path.unlink()
        return True


class ActiveTask:
    def __init__(self, task, manager):
        self.task, self.manager = task, manager
        self.cancelled = threading.Event()
        self.generation = None
        self.reason = ""
        self.releasing = False

    def cancel(self, reason):
        self.reason = reason
        self.cancelled.set()
        if self.generation is not None:
            self.manager.stop_task(self.generation)


class Worker:
    def __init__(
        self,
        config,
        *,
        relay=None,
        manager=None,
        driver=None,
        executor_factory=None,
        control_request_selector=control_request_plan,
        background=False,
    ):
        self.config, self.state_dir = config, Path(config["state_dir"])
        self.journal = Journal(self.state_dir)
        self.relay = relay or GitRelay(
            config["gitee_repo_url"],
            self.state_dir / "relay",
            allow_local=config.get("simulation", False),
        )
        if manager is None:
            from prepare.runtime import EnvironmentManager

            manager = EnvironmentManager(config, self.state_dir)
        self.manager = manager
        self.running_control_revision = getattr(
            manager, "current_control_revision", lambda: None
        )()
        self.driver = driver or CodexDriver(config, self.state_dir)
        self.executor_factory = executor_factory or DockerExecutor
        self.control_request_selector = control_request_selector
        self.control_checked_tasks = set()
        self.stop_event = threading.Event()
        self.active = None
        self.control_channel = "unknown"
        self.background = background
        self.execution_thread = None
        self.delivery_thread = None
        self.publishing = None
        self.control_lock_path = self.state_dir / "control.lock"

    def heartbeat(self, **extra):
        atomic_json(
            self.state_dir / "health/worker.json",
            {
                "schema": "triton-anchor-worker-health",
                "worker_id": self.config.get("worker_id", "local-ci"),
                "heartbeat_at": time.time(),
                "pid": os.getpid(),
                "head_sha": self.active.task["head_sha"] if self.active else None,
                "active_task": self.active.task["task_id"] if self.active else None,
                "active_run_id": self.journal.task(self.active.task["task_id"])["run_id"] if self.active and self.journal.has_task(self.active.task["task_id"]) else None,
                "control_channel": self.control_channel,
                "tasks": [
                    {k: r[k] for k in ("task_id", "run_id", "head_sha", "phase", "updated", "budget", "recovery", "last_progress_at")}
                    for r in self.journal.tasks()
                ],
                **(getattr(self.driver, "health", {}) if self.active else {}),
                **extra,
            },
        )

    def refresh_relay(self):
        try:
            self.relay.refresh()
        except Exception:
            self.control_channel = "unreachable"
            raise
        self.control_channel = "reachable"

    def queue_sealed(self, task_id, path, run_id=None):
        task = json.loads(self.journal.task(task_id, run_id)["manifest"])
        if path.stat().st_size > MAX_RESULT_BYTES:
            raise ContractError("Sealed result exceeds 2 MiB")
        result = validate_result(json.loads(path.read_bytes()), task)
        if result["run_id"] != (run_id or self.journal.task(task_id)["run_id"]):
            raise ContractError("Sealed result belongs to another run")
        self.journal.queue_result(
            task_id, path, hashlib.sha256(path.read_bytes()).hexdigest(), run_id=run_id,
        )

    def recovery(self, task_id, state, failure_code="", action="", *, delay=None, run_id=None, **extra):
        record = self.journal.record(task_id, run_id)
        budget = record.get("budget")
        now = time.time()
        if budget and action != "retry_publish" and state in {"retry_wait", "waiting_dependency", "recovering"}:
            if budget.get("recovery_deadline_at") is None:
                budget["recovery_deadline_at"] = min(
                    now + self.config.get("recovery_timeout_seconds", 21600),
                    budget.get("codex_deadline_at") or float("inf"),
                )
        previous = record.get("recovery", {})
        ongoing_states = {"retry_wait", "waiting_dependency", "recovering"}
        if not failure_code and state in ongoing_states and previous.get("state") in ongoing_states:
            failure_code = previous.get("failure_code", "")
        recovery = {
            "state": state, "failure_code": failure_code, "action": action,
            "next_retry_at": now + delay if delay is not None else None,
            "last_recovery_at": now if state == "recovered" else previous.get("last_recovery_at"),
            "outcome": {"normal": None, "recovered": "recovered", "exhausted": "failed"}.get(state, "pending"),
            "attempt": (budget or {}).get("codex_attempts_used", 0),
            "execution_attempt": (budget or {}).get("execution_attempts_used", 0),
            **extra,
        }
        self.journal.update(task_id, run_id=run_id, recovery=recovery, budget=budget)
        significant = ("state", "failure_code", "action", "outcome", "attempt", "execution_attempt")
        if state != "normal" and any(recovery.get(key) != previous.get(key) for key in significant):
            self.journal.event(task_id, "recovery", {
                **{k: recovery[k] for k in ("state", "failure_code", "action", "next_retry_at", "outcome")},
                "attempt": (budget or {}).get("codex_attempts_used", 0),
                "execution_attempt": (budget or {}).get("execution_attempts_used", 0),
            }, run_id=run_id)

    def exhausted(self, row):
        budget = row.get("budget")
        if not isinstance(budget, dict):
            return "旧任务无可靠恢复预算，请重新派发任务"
        if any(budget.get(k) is not None and budget[k] <= time.time()
               for k in ("codex_deadline_at", "recovery_deadline_at")):
            return "任务恢复时间预算已耗尽，请重新派发任务"
        return ""

    def ready(self, row):
        recovery = row.get("recovery") or {}
        if recovery.get("next_retry_at") and recovery["next_retry_at"] > time.time():
            return False
        if recovery.get("failure_code") == "authentication" and not recovery.get("manual_resume"):
            fingerprint = getattr(self.driver, "credentials_fingerprint", lambda: "")()
            record = self.journal.record(row["task_id"])
            if fingerprint == record.get("credentials_fingerprint"):
                return False
        return recovery.get("state") != "exhausted"

    @staticmethod
    def complete_report(path, policy):
        """An actual failed check is final even if the CLI's own shutdown failed."""
        return diagnose_report(path, policy)["report"]

    def previous_runs(self, task_id):
        """Follow only this frozen task's bounded execution lineage."""
        state = self.journal.record(task_id)
        previous = state.get("previous_run_id")
        seen, runs = {state["run_id"]}, []
        limit = state.get("budget", {}).get("execution_attempts_used", 0)
        while previous and previous not in seen and len(runs) < limit:
            seen.add(previous)
            try:
                directory = self.journal.run_dir(task_id, previous)
                state = self.journal.record(task_id, previous)
            except (OSError, ValueError):
                break  # Historical evidence may already have expired.
            runs.append((directory, state))
            previous = state.get("previous_run_id")
        return runs

    def seal_checkpoint(self, row):
        task_id, run_id = row["task_id"], row["run_id"]
        record = self.journal.record(task_id, run_id)
        checkpoint = record.get("checkpoint", {})
        if not checkpoint.get("complete"):
            return False
        directory = self.journal.run_dir(task_id, run_id)
        sealed = directory / "sealed/result.json"
        if sealed.exists():
            self.queue_sealed(task_id, sealed, run_id)
            return True
        attempts = record.get("sealing_attempts", 0)
        maximum = self.config.get("sealing_attempts", 3)
        if attempts >= maximum:
            recovery = record.get("recovery", {})
            if recovery.get("state") != "exhausted" or recovery.get("failure_code") != "sealing_failed":
                self.journal.update(task_id, run_id=run_id, seal_next_retry_at=None)
                self.recovery(task_id, "exhausted", "sealing_failed", "retry_sealing", run_id=run_id)
            return False  # Keep the checkpoint for explicit recovery; never rerun tests.
        retry_at = record.get("seal_next_retry_at")
        if retry_at and retry_at > time.time():
            return False
        report = checkpoint["report"]
        fallback = bool(checkpoint.get("invalid_report"))
        contract_error = record.get("sealing_contract_error")
        if contract_error is not None:
            report = {"status": "infra_error", "summary": "结果契约无效：" + contract_error}
            fallback = True
        while attempts < maximum:
            attempts += 1
            self.journal.phase(task_id, "sealing", run_id=run_id)
            self.journal.update(task_id, run_id=run_id, sealing_attempts=attempts)
            try:
                seal_result(
                    json.loads(row["manifest"]), run_id, report,
                    checkpoint.get("policy", {}), checkpoint.get("environment", {}),
                    directory, directory / "sealed", redact=self.driver.redact,
                    collect_business=not fallback,
                )
            except Exception as exc:
                invalid_report = isinstance(exc, (ValueError, TypeError)) and not fallback
                if invalid_report:
                    # Preserve the original checkpoint while sealing a minimal failure.
                    self.journal.update(task_id, run_id=run_id, sealing_contract_error=str(exc))
                    report = {"status": "infra_error", "summary": "结果契约无效：" + str(exc)}
                    fallback = True
                exhausted = attempts >= maximum
                delay = None if exhausted else 0 if invalid_report else (30, 60)[min(attempts - 1, 1)]
                self.journal.update(task_id, run_id=run_id, seal_next_retry_at=None if delay is None else time.time() + delay)
                self.recovery(task_id, "exhausted" if exhausted else "retry_wait", "sealing_failed", "retry_sealing", delay=delay, run_id=run_id)
                self.journal.event(task_id, "sealing_error", {"error": str(exc)}, run_id=run_id)
                if invalid_report and not exhausted:
                    continue
                return False
            self.queue_sealed(task_id, sealed, run_id)
            if self.journal.record(task_id, run_id).get("recovery", {}).get("failure_code") == "sealing_failed":
                self.recovery(task_id, "recovered", action="continue_sealing", run_id=run_id)
            return True
        return False

    def finish(self, row, report, *, policy=None, environment=None):
        record = self.journal.record(row["task_id"], row["run_id"])
        checkpoint = record.get("checkpoint", {})
        self.journal.update(row["task_id"], run_id=row["run_id"], checkpoint={
            "complete": True, "report": report,
            "policy": policy if policy is not None else checkpoint.get("policy", {}),
            "environment": environment if environment is not None else checkpoint.get("environment", {}),
        })
        return self.seal_checkpoint(row)

    def recover_local(self):
        """Recover every immutable outbox before any runtime/network prerequisite."""
        for row in self.journal.all_runs():
            if self.active and row["task_id"] == self.active.task["task_id"] and row["run_id"] == self.journal.task(row["task_id"])["run_id"]:
                continue
            directory = self.journal.run_dir(row["task_id"], row["run_id"])
            try:
                if (directory / "sealed/result.json").is_file():
                    self.queue_sealed(row["task_id"], directory / "sealed/result.json", row["run_id"])
                elif self.journal.record(row["task_id"], row["run_id"]).get("checkpoint", {}).get("complete"):
                    self.seal_checkpoint(row)
                if self.journal.delivery(row["task_id"], row["run_id"]):
                    self.schedule_delivery(row)
            except (OSError, ValueError) as exc:
                self.journal.event(row["task_id"], "recovery_error", {"error": str(exc)}, run_id=row["run_id"])

    def inspect_active(self, *, remote=True):
        active = self.active
        if not active:
            return
        try:
            if remote:
                valid, reason = self.relay.validity(active.task)
                if not valid:
                    active.cancel(reason)
            if not active.releasing and active.generation is not None and hasattr(self.manager, "task_health"):
                state = self.manager.task_health(active.generation)
                if state.get("available") and state.get("running") is False:
                    active.reason = "container_oom" if state.get("oom_killed") else "container_failed"
                    active.cancelled.set()
            progress = getattr(self.driver, "health", {}).get("last_progress_at")
            if progress:
                self.journal.update(active.task["task_id"], last_progress_at=progress)
            row = self.journal.task(active.task["task_id"])
            if self.exhausted(row):
                active.reason = "recovery_exhausted"
                active.cancelled.set()
            progress = row.get("last_progress_at")
            if progress and time.time() - progress >= self.config.get("progress_warning_seconds", 1800):
                level = "stalled_review" if time.time() - progress >= self.config.get("progress_stalled_seconds", 3600) else "delayed"
                self.journal.update(active.task["task_id"], progress_state=level)
            elif progress:
                self.journal.update(active.task["task_id"], progress_state="normal")
        except Exception as exc:
            self.journal.event(active.task["task_id"], "poll_error", {"error": str(exc)})

    def start_task(self, task):
        if self.active or (self.execution_thread and self.execution_thread.is_alive()):
            return
        self.active = ActiveTask(task, self.manager)
        if not self.background:
            try:
                self.process(task)
            finally:
                self.active = None
            return

        def execute():
            import fcntl
            with self.control_lock_path.open("a") as lock:
                while not self.stop_event.is_set():
                    try:
                        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        self.stop_event.wait(0.2)
                else:
                    self.active = None
                    return
                try:
                    self.process(task)
                finally:
                    self.active = None
                    fcntl.flock(lock, fcntl.LOCK_UN)
        self.execution_thread = threading.Thread(target=execute, name="local-ci-execution", daemon=True)
        self.execution_thread.start()

    def process(self, task):
        active = self.active or ActiveTask(task, self.manager)
        self.active = active
        generation = None
        row = self.journal.register(task)
        run_dir = self.journal.run_dir(task["task_id"])
        task_id = task["task_id"]
        record = self.journal.record(task_id)
        policy = record.get("checkpoint", {}).get("policy", {})
        environment = record.get("checkpoint", {}).get("environment", {})
        report = None
        wait_code = ""
        self.manager.cancel_event = active.cancelled
        self.driver.health = {}
        try:
            if record.get("detail", {}).get("started") or row["phase"] == "running":
                row = self.journal.restart(task_id)
                run_dir = self.journal.run_dir(task_id)
            self.journal.claim(task_id, "execution_attempts_used", self.config.get("execution_attempts", 3))
            previous_recovery = self.journal.record(task_id).get("recovery", {})
            if previous_recovery.get("state") not in {None, "normal"}:
                self.recovery(task_id, "recovering", previous_recovery.get("failure_code", ""), "rebuild_execution")
            self.journal.phase(task_id, "preparing", {"started": True})
            # Preserve the frozen manifest; only the runtime view resolves both LLVM sides.
            try:
                variants = self.relay.source_variants(task)
            except ContractError as exc:
                # Invalid frozen LLVM metadata is not an exhausted retry budget.
                raise ValueError("冻结源码环境元数据无效：" + str(exc)) from exc
            runtime_task = {**task, "variants": variants}
            generation = self.manager.acquire_task(runtime_task, row["run_id"])
            active.generation = generation
            if active.cancelled.is_set():
                raise InterruptedError(active.reason)
            executor = self.executor_factory(self.config, self.state_dir, generation, task, self.relay, manager=self.manager)
            checkout = executor.prepare()
            changes = changed_files(checkout, task["base_sha"], task["tested_sha"])
            if not changes and task["event_kind"] != "pull_request":
                changes = [
                    {
                        "path": "branch-validation",
                        "old_path": "branch-validation",
                        "mode": "100644",
                    }
                ]
            policy = minimum_checks(
                changes,
                backend_enabled=generation["variants"]["candidate"]["backend_enabled"],
                full=task["full"],
                event_kind=task["event_kind"],
            )
            environment = {"variants": {
                variant: {
                    **{key: runtime[key] for key in (
                        "source_sha", "triton_version", "profile", "llvm_hash",
                        "backend_enabled", "environment_fingerprint", "image_id",
                    )},
                    "backend_profile": runtime.get("env", {}).get("BACKEND_PROFILE", "")
                    if runtime["backend_enabled"] else "",
                }
                for variant, runtime in generation["variants"].items()
            }}
            environment["control_revision"] = generation.get("control_revision", self.running_control_revision)
            executor.prepare("base")
            executor.write_context(policy, changes)
            self.journal.update(task_id, checkpoint={"complete": False, "policy": policy, "environment": environment})
            self.journal.phase(task_id, "running")
            self.journal.update(task_id, last_progress_at=time.time())
            current = self.journal.record(task_id)
            no_progress = current.get("resume_no_progress_attempts", 0)
            observation = current.get("retry_observation", {})
            new_session = current.get("next_session_mode") == "new"
            previous_runs = self.previous_runs(task_id)
            while not active.cancelled.is_set():
                try:
                    if hasattr(self.driver, "check_credentials"):
                        self.driver.check_credentials()
                except (OSError, ValueError):
                    self.journal.update(task_id, credentials_fingerprint=getattr(self.driver, "credentials_fingerprint", lambda: "")())
                    wait_code = "authentication"
                    self.recovery(task_id, "waiting_dependency", wait_code, "wait_credentials")
                    break
                budget = self.journal.claim(task_id, "codex_attempts_used", self.config.get("codex_attempts", 10), timeout=self.config.get("codex_timeout_seconds", 21600))
                deadline = min(budget["codex_deadline_at"], budget.get("recovery_deadline_at") or float("inf"))
                if budget["codex_attempts_used"] > 1:
                    write_handoff(run_dir, task, environment, observation, previous_runs, self.driver.redact,
                                  rebuilt=bool(current.get("previous_run_id")))
                kwargs = dict(cancelled=active.cancelled, deadline=time.monotonic() + max(0, deadline-time.time()), recovery=recovery_prompt(observation) if budget["codex_attempts_used"] > 1 else "")
                if budget["codex_attempts_used"] > 1:
                    kwargs["recovery"] += "\n先读 /task/artifacts/recovery-context.json；核对当前环境、已有证据和遗留进程，再继续未完成工作。"
                if new_session:
                    kwargs["session_mode"] = "new"
                    new_session = False
                if budget["codex_attempts_used"] > 1:
                    self.recovery(task_id, "recovering", action="new_session" if "session_mode" in kwargs else "resume")
                self.journal.event(task_id, "codex_start", {"attempt": budget["codex_attempts_used"], "deadline_at": budget["codex_deadline_at"],
                    "execution_attempt": budget["execution_attempts_used"], "session_switches": budget["session_switches"],
                    "session_mode": kwargs.get("session_mode", "resume_if_available")})
                try:
                    outcome = self.driver.run(executor, **kwargs)
                except (FileNotFoundError, PermissionError) as exc:
                    raise ValueError("Codex启动配置不可用：" + str(exc)) from exc
                except Exception as exc:
                    outcome = {"failure_code": "authentication" if isinstance(exc, CredentialValidationError) else "cli_failed", "exit_code": -1, "reason": ""}
                    self.journal.event(task_id, "codex_error", {"error": str(exc)})
                self.journal.update(task_id, next_session_mode="resume_if_available")
                self.journal.event(task_id, "codex_exit", outcome)
                diagnosis = diagnose_report(run_dir / "artifacts/agent-result.json", policy)
                report = diagnosis["report"]
                if report is not None:
                    recovered = budget["codex_attempts_used"] > 1 or budget["execution_attempts_used"] > 1
                    self.recovery(task_id, "recovered" if recovered else "normal", action="continue_sealing")
                    break
                if outcome.get("reason") == "timeout":
                    raise TimeoutError("Codex任务时间预算已耗尽")
                if active.cancelled.is_set():
                    break
                code = outcome.get("failure_code") or "result_missing"
                observation, progressed = retry_observation(
                    diagnosis, completed_tools(run_dir / "artifacts", environment), observation,
                )
                self.journal.update(task_id, retry_observation=observation)
                if code == "authentication":
                    self.journal.update(task_id, credentials_fingerprint=getattr(self.driver, "credentials_fingerprint", lambda: "")())
                    wait_code = code
                    self.recovery(task_id, "waiting_dependency", code, "wait_credentials")
                    break
                if code not in {"rate_limit", "authentication"}:
                    no_progress = no_progress + 1 if outcome.get("session_reused") and not progressed else 0
                if code not in {"rate_limit", "authentication"} and (code == "session_invalid" or no_progress >= self.config.get("codex_resume_no_progress_attempts", 2)):
                    if budget.get("session_switches", 0) < self.config.get("codex_session_switches", 1):
                        self.journal.claim(task_id, "session_switches", self.config.get("codex_session_switches", 1))
                        new_session = True
                        no_progress = 0
                self.journal.update(task_id, resume_no_progress_attempts=no_progress,
                                    next_session_mode="new" if new_session else "resume_if_available")
                delay = min(300, 30 * 2 ** min(budget["codex_attempts_used"]-1, 4)) if code == "rate_limit" else self.config.get("retry_delay_seconds", 30)
                self.recovery(task_id, "retry_wait", code, "new_session" if new_session else "resume", delay=delay)
                active.cancelled.wait(min(delay, max(0, deadline-time.time())))
        except (ContractError, TimeoutError) as exc:
            self.recovery(task_id, "exhausted", "recovery_exhausted", "publish_infra_error")
            report = {"status": "infra_error", "summary": str(exc)}
        except ValueError as exc:
            self.recovery(task_id, "exhausted", "configuration_invalid", "publish_infra_error")
            report = {"status": "infra_error", "summary": "配置或任务契约无效：" + str(exc)}
        except RuntimeEnvironmentError as exc:
            # A healthy daemon cannot repair an invalid mount/profile/configuration.
            try:
                self.manager._daemon()
                daemon_available = True
            except Exception:
                daemon_available = False
            if daemon_available and any(word in str(exc).lower() for word in ("invalid", "requires", "must", "profile", "differs", "no rootful", "not rootless")):
                self.recovery(task_id, "exhausted", "configuration_invalid", "publish_infra_error")
                report = {"status": "infra_error", "summary": str(exc)}
            else:
                wait_code = "environment_unavailable"
                self.recovery(task_id, "waiting_dependency", wait_code, "rebuild_execution", delay=60)
                self.journal.event(task_id, "task_error", {"error": str(exc)})
        except Exception as exc:
            wait_code = "environment_unavailable"
            self.recovery(task_id, "waiting_dependency", wait_code, "rebuild_execution", delay=self.config.get("poll_interval_seconds", 60))
            self.journal.event(task_id, "task_error", {"error": str(exc)})
        finally:
            active.releasing = True
            self.manager.cancel_event = None
            if generation is not None:
                stopped = False
                try:
                    self.manager.stop_task(generation)
                    stopped = True
                    self.manager.collect_artifacts(generation)
                    report = self.complete_report(run_dir / "artifacts/agent-result.json", policy) or report
                    if report is not None:
                        self.journal.update(task_id, checkpoint={"complete": True, "report": report, "policy": policy, "environment": environment})
                    self.manager.destroy_task(generation)
                except Exception as exc:
                    self.journal.event(task_id, "cleanup_error", {"error": str(exc)})
                    if not stopped:
                        report = None
                    if report is None:
                        wait_code = "cleanup_unconfirmed"
                        self.recovery(task_id, "waiting_dependency", wait_code, "wait_runtime", delay=60)
            shutil.rmtree(run_dir / "inputs", ignore_errors=True)
        if self.stop_event.is_set():
            return
        if active.reason == "recovery_exhausted" and report is None:
            self.recovery(task_id, "exhausted", "recovery_exhausted", "publish_infra_error")
            report = {"status": "infra_error", "summary": "任务恢复时间预算已耗尽，请重新派发任务"}
        if active.cancelled.is_set() and active.reason not in {"container_failed", "container_oom", "worker_shutdown", "recovery_exhausted"}:
            report = {"status": "cancelled", "summary": active.reason}
        elif active.reason in {"container_failed", "container_oom"} and report is None:
            wait_code = active.reason
            self.recovery(task_id, "waiting_dependency", wait_code, "rebuild_execution", delay=60)
        if report is not None:
            self.finish(row, report, policy=policy, environment=environment)
        elif not wait_code:
            self.recovery(task_id, "waiting_dependency", "execution_interrupted", "rebuild_execution", delay=60)
        self.heartbeat()

    def schedule_delivery(self, row):
        box = self.journal.delivery(row["task_id"], row["run_id"])
        if not box or box.get("published") or (box.get("next_retry_at") or 0) > time.time():
            return
        if not self.background:
            self.retry_delivery(row)
            return
        if self.delivery_thread and self.delivery_thread.is_alive():
            return
        self.publishing = (row["task_id"], row["run_id"])

        def upload():
            try:
                self.retry_delivery(row)
            finally:
                self.publishing = None
        self.delivery_thread = threading.Thread(target=upload, name="local-ci-upload", daemon=True)
        self.delivery_thread.start()

    def retry_delivery(self, row):
        box = self.journal.delivery(row["task_id"], row["run_id"])
        if not box or box.get("published") or (box.get("next_retry_at") or 0) > time.time():
            return
        path = Path(box["payload_path"])
        try:
            if hashlib.sha256(path.read_bytes()).hexdigest() != box["digest"]:
                raise ContractError("Saved result changed before publication")
            task = json.loads(row["manifest"])
            digest = self.relay.publish_result(task, row["run_id"], path.parent)
            if digest != box["digest"]:
                raise ContractError("Published result differs from saved result")
            if box["attempts"]:
                self.recovery(row["task_id"], "recovered", action="published", run_id=row["run_id"])
            self.journal.published(row["task_id"], row["run_id"])
        except Exception as exc:
            attempts = box["attempts"] + 1
            delay = (60, 120, 300, 300)[min(attempts-1, 3)] if attempts < self.config.get("publish_fast_attempts", 5) else self.config.get("publish_retry_interval_seconds", 3600)
            self.journal.publication_failure(row["task_id"], row["run_id"], next_retry_at=time.time()+delay)
            self.recovery(row["task_id"], "retry_wait", "delivery_failed", "retry_publish", delay=delay, run_id=row["run_id"])
            self.journal.event(row["task_id"], "publication_error", {"error": str(exc)}, run_id=row["run_id"])

    def scan(self):
        self.recover_local()
        self.inspect_active(remote=False)
        complete_current_control_request(
            self.config, self.running_control_revision
        )
        installed_revision = getattr(
            self.manager, "current_control_revision", lambda: None
        )()
        if (
            self.running_control_revision
            and installed_revision
            and installed_revision != self.running_control_revision
        ):
            try:
                pending = read_update_request(update_request_path(self.config))
            except (OSError, ValueError):
                pending = None
            if pending and pending["revision"] == installed_revision:
                request = {
                    "revision": pending["revision"],
                    "task_id": pending["task_id"],
                }
                self.heartbeat(
                    control_revision=self.running_control_revision,
                    installed_control_revision=installed_revision,
                    requested_control_revision=pending["revision"],
                    control_update="restart_required",
                )
                return request
            self.heartbeat(
                control_revision=self.running_control_revision,
                installed_control_revision=installed_revision,
                control_update="blocked",
                error="Installed control changed without a matching update request",
            )
            return None
        runtime_blocked = False
        unsafe_tasks = set()
        if not self.active:
            # Collect a finished report before removing an orphaned environment.
            for handle in self.manager.generations().values():
                if handle["state"] == "removed":
                    continue
                try:
                    self.manager.stop_task(handle)
                    self.manager.collect_artifacts(handle)
                    if self.journal.has_task(handle["task_id"]):
                        row = self.journal.task(handle["task_id"], handle["run_id"])
                        record = self.journal.record(handle["task_id"], handle["run_id"])
                        context = record.get("checkpoint", {})
                        if not self.journal.delivery(handle["task_id"], handle["run_id"]) and context:
                            report = self.complete_report(self.journal.run_dir(handle["task_id"], handle["run_id"]) / "artifacts/agent-result.json", context.get("policy", {}))
                            if report is not None:
                                self.finish(row, report)
                    self.manager.destroy_task(handle)
                except Exception as exc:
                    runtime_blocked = True
                    unsafe_tasks.add(handle.get("task_id"))
                    self.heartbeat(runtime="cleanup_unconfirmed", error=str(exc))
            try:
                self.manager.collect_retired()
                if hasattr(self.manager, "_daemon"):
                    self.manager._daemon()
            except Exception:
                runtime_blocked = True
        from maintenance.retention import retain_local
        disk_blocked = retain_local(self.config)["pause_intake"]
        if disk_blocked:
            self.heartbeat(runtime="disk_budget_exceeded")
        for row in self.journal.tasks():
            if row["phase"] in {"publish_pending", "published"} or row["task_id"] in unsafe_tasks:
                continue
            if self.active and self.active.task["task_id"] == row["task_id"]:
                continue
            if not row.get("budget"):
                self.journal.recover_budget(row["task_id"], timeout=self.config.get("codex_timeout_seconds", 21600))
                row = self.journal.task(row["task_id"])
            reason = self.exhausted(row)
            if reason and not self.journal.record(row["task_id"]).get("checkpoint", {}).get("complete"):
                self.recovery(row["task_id"], "exhausted", "recovery_exhausted", "publish_infra_error")
                self.finish(row, {"status": "infra_error", "summary": reason})
        try:
            self.refresh_relay()
        except Exception:
            self.heartbeat()
            return None
        self.inspect_active()
        holds_control = bool(self.active or unsafe_tasks)
        waiting = []
        current_revision = self.running_control_revision
        known = [json.loads(row["manifest"]) for row in self.journal.tasks()
                 if row["phase"] not in {"publish_pending", "published"}]
        tasks = {task["task_id"]: task for task in [*known, *self.relay.tasks()]}
        self.control_checked_tasks.intersection_update(tasks)
        for task in tasks.values():
            if self.stop_event.is_set():
                break
            if is_legacy_task(task):
                continue
            try:
                validate_task(
                    task,
                    tuple(
                        self.config.get(
                            "repositories", ["RACE-org/triton-anchor"]
                        )
                    ),
                )
                try:
                    local = self.journal.task(task["task_id"])
                except ContractError:
                    local = None
                if local and local["phase"] in {"publish_pending", "published"}:
                    continue
                if local and self.journal.record(
                    task["task_id"], local["run_id"]
                ).get("checkpoint", {}).get("complete"):
                    # recover_local exclusively owns completed report sealing.
                    # Control selection must never replace that checkpoint.
                    continue
                valid, invalid_reason = self.relay.validity(task)
                if not valid:
                    if local and not (self.active and self.active.task["task_id"] == task["task_id"]):
                        self.finish(local, {"status": "cancelled", "summary": invalid_reason})
                    continue
                if local and (local.get("recovery") or {}).get("state") != "exhausted":
                    holds_control = True
                    # A journaled task has already crossed the admission
                    # boundary and must resume with this Worker snapshot.
                    self.control_checked_tasks.add(task["task_id"])
                if task["task_id"] not in self.control_checked_tasks and current_revision:
                    waiting.append(
                        {
                            "task_id": task["task_id"],
                            "captured_at": task["captured_at"],
                        }
                    )
                    continue
                if self.active or runtime_blocked or disk_blocked:
                    if (local and not self.active
                            and (local.get("recovery") or {}).get("state") != "exhausted"
                            and (local.get("recovery") or {}).get("failure_code") != "authentication"):
                        self.recovery(task["task_id"], "waiting_dependency", "disk_budget" if disk_blocked else "runtime_unavailable", "wait_dependency", delay=60)
                    continue
                if local and not self.ready(local):
                    continue
                self.start_task(task)
                if self.journal.has_task(task["task_id"]):
                    row = self.journal.task(task["task_id"])
                    if row["phase"] == "publish_pending":
                        self.schedule_delivery(row)
            except Exception as exc:
                self.journal.event(
                    task.get("task_id", "invalid"), "task_error", {"error": str(exc)}
                )
                self.heartbeat(error=str(exc))
        if waiting and not holds_control and not self.active:
            try:
                plan = self.control_request_selector(
                    self.config,
                    current_revision,
                    waiting,
                    allow_local=self.config.get("simulation", False),
                )
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self.heartbeat(
                    control_revision=current_revision,
                    control_update="blocked",
                    error=str(exc),
                )
                return None
            waiting_ids = {row["task_id"] for row in waiting}
            if not isinstance(plan, dict):
                raise ValueError("Control update selector returned an invalid plan")
            checked = set(plan.get("checked_task_ids", ()))
            request = plan.get("request")
            if (
                set(plan) != {"checked_task_ids", "request"}
                or not checked <= waiting_ids
                or (request is None and checked != waiting_ids)
                or (request is not None and (
                    not isinstance(request, dict)
                    or request.get("task_id") not in waiting_ids
                    or request.get("task_id") in checked
                    or not SHA.fullmatch(str(request.get("revision", "")))
                    or request["revision"] == current_revision
                ))
            ):
                raise ValueError("Control update selector returned an invalid plan")
            self.control_checked_tasks.update(checked)
            if request is not None:
                self.heartbeat(
                    control_revision=current_revision,
                    requested_control_revision=request["revision"],
                    control_update="required",
                )
                return request
            # The installed checkout matches the trusted branch tip.
            # Admit the newly checked tasks in this poll.
            return self.scan()
        self.heartbeat()
        return None


def scan_once(worker: Worker, control_lock, *, trigger=trigger_control_update):
    """Scan under a shared lock, then trigger any required update after unlocking."""
    import fcntl

    while True:
        try:
            fcntl.flock(control_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            # The updater holds the lock while restarting this Worker.
            if worker.stop_event.wait(1):
                return None
    try:
        request = worker.scan()
    finally:
        fcntl.flock(control_lock, fcntl.LOCK_UN)
    if request:
        trigger(worker.config, request)
    return request


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.environ.get("LOCAL_CI_CONFIG_JSON", "/opt/local-ci/config.json"),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--resume", metavar="TASK_ID")
    args = parser.parse_args(argv)
    worker = Worker(json.loads(Path(args.config).read_text()), background=not args.once)
    import fcntl

    with (worker.state_dir / "poll.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another worker owns the poll lock", file=sys.stderr)
            return 2
        if args.resume:
            worker.journal.resume(args.resume)
            return 0

        def stop(signum, frame):
            worker.stop_event.set()
            if worker.active:
                worker.active.cancel("worker_shutdown")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        with (worker.state_dir / "control.lock").open("w") as control_lock:
            while not worker.stop_event.is_set():
                try:
                    scan_once(worker, control_lock)
                except Exception as exc:
                    worker.heartbeat(error=str(exc))
                if args.once:
                    break
                worker.stop_event.wait(worker.config.get("poll_interval_seconds", 60))
            for thread in (worker.execution_thread, getattr(worker, "delivery_thread", None)):
                if thread:
                    thread.join(timeout=worker.config.get("cleanup_timeout_seconds", 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
