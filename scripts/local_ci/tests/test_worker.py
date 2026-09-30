"""Task lifecycle: execute once, stop before collecting, retry publication only."""

import fcntl
import hashlib
import json
import time
from pathlib import Path
import select
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_ci.policy import minimum_checks
from agent_ci.protocol import TASK_SCHEMA, ContractError, atomic_json, metadata_digest, task_id, validate_task
from agent_ci.worker import Worker, scan_once, trigger_control_update
from agent_ci.state import Journal, local_run_dir


def manifest(event_kind="pull_request", *, trigger_id=None):
    value = {
        "schema": TASK_SCHEMA,
        "repository": "RACE-org/triton-anchor",
        "event_kind": event_kind,
        "pr_number": 7 if event_kind == "pull_request" else 0,
        "target_branch": "main",
        "tested_sha": "a" * 40,
        "base_sha": "b" * 40,
        "head_sha": "c" * 40,
        "worker_revision_sha": "d" * 40,
        "llvm_hash": "e" * 40,
        "full": False,
        "draft": False,
        "state": "open",
        "title": "Fix behavior",
        "description": "Implementation and verification",
        "labels": [],
        "captured_at": "2026-09-11T00:00:00Z",
    }
    if trigger_id is not None:
        value["trigger_id"] = trigger_id
    value["metadata_digest"] = metadata_digest(value)
    value["task_id"] = task_id(value)
    prefix = f"ci/pr-7/{value['task_id']}" if value["pr_number"] else f"ci/branch/{value['task_id']}"
    value.update(
        task_ref=prefix + "/tested",
        base_task_ref=prefix + "/base",
        head_task_ref=prefix + "/head",
    )
    return value


def revised_manifest(revision, captured_at):
    value = manifest()
    value["worker_revision_sha"] = revision
    value["captured_at"] = captured_at
    value["task_id"] = task_id(value)
    prefix = f"ci/pr-7/{value['task_id']}"
    value.update(
        task_ref=prefix + "/tested",
        base_task_ref=prefix + "/base",
        head_task_ref=prefix + "/head",
    )
    return value


def test_idle_relay_failure_persists_until_success_and_old_codex_state_is_hidden(tmp_path):
    class Relay:
        failing = True

        def refresh(self):
            if self.failing:
                raise OSError("private relay endpoint unavailable")

    relay = Relay()
    worker = Worker(
        {"state_dir": str(tmp_path)}, relay=relay, manager=SimpleNamespace(),
        driver=SimpleNamespace(health={"codex_status": "connection_error"}),
    )
    with pytest.raises(OSError):
        worker.refresh_relay()
    for _ in range(2):
        worker.heartbeat()
        heartbeat = json.loads((tmp_path / "health/worker.json").read_text())
        assert heartbeat["control_channel"] == "unreachable"
        assert "codex_status" not in heartbeat
    relay.failing = False
    worker.refresh_relay()
    worker.heartbeat()
    assert json.loads((tmp_path / "health/worker.json").read_text())["control_channel"] == "reachable"


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("event_kind", ["pull_request", "push", "manual"])
def test_run_stops_collects_and_publishes_without_reexecuting_on_network_failure(
    tmp_path, monkeypatch, cancelled, event_kind
):
    task = manifest(event_kind)
    events = []
    calls = 0
    uploads = 0

    class Relay:
        def refresh(self):
            pass

        def validity(self, task):
            return True, ""

        def tasks(self):
            return [task]

        def source_variants(self, frozen):
            return {
                "base": {"source_sha": frozen["base_sha"], "llvm_hash": "f" * 40,
                         "triton_version": "3.0.0"},
                "candidate": {"source_sha": frozen["tested_sha"], "llvm_hash": frozen["llvm_hash"],
                              "triton_version": "3.3.0"},
            }

        def publish_result(self, task, run_id, directory):
            nonlocal uploads
            uploads += 1
            if uploads == 1:
                raise OSError("temporary relay outage")
            return hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()

    class Manager:
        def generations(self):
            return {}

        def collect_retired(self):
            pass

        def acquire_task(self, task, run_id):
            return {
                "run_id": run_id,
                "variants": {
                    variant: {**source, "profile": variant, "backend_enabled": variant == "base",
                              "env": {"BACKEND_PROFILE": "sophgo-cmodel"},
                              "environment_fingerprint": variant, "image_id": "fixture"}
                    for variant, source in task["variants"].items()
                },
                "profile": "fixture",
                "backend_enabled": False,
                "environment_fingerprint": "fixture",
                "llvm_hash": task["llvm_hash"],
                "image_id": "fixture",
                "artifacts_host": str(local_run_dir(tmp_path, task, run_id) / "artifacts"),
            }

        def stop_task(self, handle):
            events.append("stop")

        def collect_artifacts(self, handle):
            assert events[-1] == "stop"
            events.append("collect")

        def destroy_task(self, handle):
            events.append("destroy")

    class Executor:
        def __init__(self, config, state_dir, generation, task, relay, *, manager):
            self.run_dir = Path(generation["artifacts_host"]).parent

        def prepare(self, variant="candidate"):
            return tmp_path

        def write_context(self, policy, changes):
            assert ("pr_info" in policy["required_reviews"]) == bool(task["pr_number"])

    class Driver:
        def redact(self, text):
            return text

        def run(self, executor, **kwargs):
            nonlocal calls
            calls += 1
            atomic_json(
                executor.run_dir / "artifacts/agent-result.json",
                {
                    "status": "pass",
                    "summary": "Behavior verified",
                    "checks": [],
                    "reviews": [
                        {
                            "kind": kind,
                            "status": "not_applicable" if kind == "pr_info" and not task["pr_number"] else "pass",
                            "summary": "Verified",
                        }
                        for kind in ("pr_info", "architecture")
                    ],
                },
            )
            if cancelled:
                worker.active.cancel("PR closed")
            return {"exit_code": 0, "reason": ""}

    monkeypatch.setattr(
        "agent_ci.worker.changed_files", lambda *args: [{"path": "README.md"}]
    )
    monkeypatch.setattr(
        "agent_ci.worker.minimum_checks",
        lambda *args, **kwargs: {**minimum_checks(*args, **kwargs), "required_checks": []},
    )
    worker = Worker(
        {"state_dir": str(tmp_path), "simulation": True, "retry_delay_seconds": 0},
        relay=Relay(),
        manager=Manager(),
        driver=Driver(),
        executor_factory=Executor,
    )
    worker.scan()
    row = worker.journal.task(task["task_id"])
    assert row["phase"] == "publish_pending"
    queued_at = worker.journal.delivery(task["task_id"])["queued_at"]
    worker.journal.publication_failure(task["task_id"])
    assert worker.journal.delivery(task["task_id"])["queued_at"] == queued_at
    result = json.loads(
        (worker.journal.run_dir(task["task_id"]) / "sealed/result.json").read_text()
    )
    assert result["status"] == ("cancelled" if cancelled else "pass")
    assert result["task"] == task and "variants" not in task
    runtimes = result["environment"]["variants"]
    assert runtimes["base"]["llvm_hash"] == "f" * 40
    assert runtimes["candidate"]["llvm_hash"] == task["llvm_hash"]
    assert runtimes["base"]["source_sha"] == task["base_sha"]
    assert runtimes["base"]["backend_profile"] == "sophgo-cmodel"
    assert runtimes["candidate"]["backend_profile"] == ""
    assert events[-3:] == ["stop", "collect", "destroy"]
    worker.journal = Journal(tmp_path)
    assert worker.journal.register(task)["run_id"] == row["run_id"]
    directory = worker.journal.run_dir(task["task_id"])
    prefix = "runs/pr/branch-main/pr-7" if task["pr_number"] else "runs/push/branch-main"
    assert directory == tmp_path / prefix / task["head_sha"] / row["run_id"]
    # Simulate a pre-upgrade run: reload and retry its saved upload in place.
    legacy = tmp_path / "runs" / task["task_id"]
    directory.parent.rename(legacy)
    record_path = legacy / row["run_id"] / "state.json"
    record = json.loads(record_path.read_text())
    record["delivery"]["payload_path"] = str(record_path.parent / "sealed/result.json")
    atomic_json(record_path, record)
    worker.journal = Journal(tmp_path)
    worker.scan()
    assert calls == 1 and uploads == 2
    assert worker.journal.task(task["task_id"])["phase"] == "published"
    assert worker.journal.task(task["task_id"])["run_id"] == row["run_id"]
    assert worker.journal.run_dir(task["task_id"]) == record_path.parent
    worker.journal = Journal(tmp_path)
    worker.scan()
    assert calls == 1 and uploads == 2  # Published old runs remain deduplicated.


def test_sha_layout_keeps_distinct_tasks_and_restart_history(tmp_path):
    first = manifest()
    assert first["task_id"] == "d8066abb038d9f5a7e3800892d67688d71e1b3c03457c6fc74db25018f9be654"
    assert validate_task(manifest(trigger_id=""))["task_id"] == first["task_id"]
    triggered = manifest(trigger_id="12345:1")
    assert validate_task(triggered) == manifest(trigger_id="12345:1")
    assert len({first["task_id"], triggered["task_id"],
                manifest(trigger_id="12345:2")["task_id"],
                manifest(trigger_id="67890:1")["task_id"]}) == 4
    worker_selected = {**triggered, "control_policy": "worker"}
    assert task_id(worker_selected) == task_id({**worker_selected, "worker_revision_sha": "f" * 40})
    for invalid in (None, 1, "0:1", "1:0", "1", "1:1\n", "1" * 159 + ":1"):
        with pytest.raises(ContractError, match="trigger_id"):
            validate_task({**first, "trigger_id": invalid})
    with pytest.raises(ContractError, match="Runtime-only control override"):
        validate_task({**first, "_use_installed_control": True})
    second = revised_manifest("f" * 40, "2026-09-12T00:00:00Z")
    journal = Journal(tmp_path)
    row1, row2 = journal.register(first), journal.register(second)
    retriggered = journal.register(triggered)
    assert journal.run_dir(triggered["task_id"]).parent.name == first["head_sha"]
    directory1 = journal.run_dir(first["task_id"])
    directory2 = journal.run_dir(second["task_id"])
    assert directory1.parent == directory2.parent
    assert directory1 != directory2
    journal.phase(first["task_id"], "published")
    journal = Journal(tmp_path)
    assert journal.register(triggered)["run_id"] == retriggered["run_id"]
    assert journal.task(first["task_id"])["phase"] == "published"
    assert journal.task(second["task_id"])["run_id"] == row2["run_id"]
    assert journal.task(second["task_id"])["head_sha"] == first["head_sha"]
    with pytest.raises(ValueError, match="another task"):
        journal.run_dir(first["task_id"], row2["run_id"])
    journal.restart(second["task_id"])
    assert journal.run_dir(second["task_id"]) != directory2
    assert Journal(tmp_path).task(first["task_id"])["run_id"] == row1["run_id"]


@pytest.mark.parametrize("legacy", ["flat", "grouped"])
def test_new_attempt_uses_sha_but_old_run_remains_readable(tmp_path, legacy):
    task = manifest()
    journal = Journal(tmp_path)
    row = journal.register(task)
    directory = journal.run_dir(task["task_id"])
    old_parent = (tmp_path / "runs" if legacy == "flat" else directory.parent.parent) / task["task_id"]
    directory.parent.rename(old_parent)
    journal = Journal(tmp_path)
    assert journal.run_dir(task["task_id"]) == old_parent / row["run_id"]
    restarted = journal.restart(task["task_id"])
    assert journal.run_dir(task["task_id"]).parent.name == task["head_sha"]
    journal = Journal(tmp_path)
    assert journal.task(task["task_id"])["run_id"] == restarted["run_id"]
    assert journal.run_dir(task["task_id"], row["run_id"]) == old_parent / row["run_id"]


def test_heartbeat_exposes_head_without_replacing_task_identity(tmp_path):
    from types import SimpleNamespace
    task = manifest()
    worker = Worker({"state_dir": str(tmp_path), "simulation": True},
                    relay=object(), manager=object(), driver=object())
    worker.journal.register(task)
    worker.active = SimpleNamespace(task=task)
    worker.heartbeat()
    health = json.loads((tmp_path / "health/worker.json").read_text())
    assert health["head_sha"] == task["head_sha"]
    assert health["tasks"][0]["head_sha"] == task["head_sha"]
    assert health["tasks"][0]["task_id"] == task["task_id"]


def test_control_update_request_is_single_atomic_file_and_nonblocking(tmp_path):
    task = manifest()
    calls = []
    unit = trigger_control_update(
        {"state_dir": str(tmp_path)},
        {"revision": task["worker_revision_sha"], "task_id": task["task_id"]},
        run=lambda argv, **kwargs: calls.append((argv, kwargs)),
    )
    request = json.loads((tmp_path / "control-update/request.json").read_text())
    assert request["revision"] == task["worker_revision_sha"]
    assert request["task_id"] == task["task_id"]
    assert unit == "triton-anchor-local-ci-control-update.service"
    assert calls == [
        (
            [
                "systemctl",
                "--user",
                "start",
                "--no-block",
                "triton-anchor-local-ci-control-update.service",
            ],
            {"check": True, "timeout": 30},
        )
    ]

    first_requested_at = request["requested_at"]
    trigger_control_update(
        {"state_dir": str(tmp_path)},
        {"revision": task["worker_revision_sha"], "task_id": task["task_id"]},
        run=lambda argv, **kwargs: None,
    )
    repeated = json.loads((tmp_path / "control-update/request.json").read_text())
    assert repeated["requested_at"] == first_requested_at


def test_scan_releases_control_lock_before_trigger(tmp_path):
    task = manifest()
    events = []

    class FixtureWorker:
        config = {"state_dir": str(tmp_path)}

        def scan(self):
            events.append("scan")
            return {"revision": task["worker_revision_sha"], "task_id": task["task_id"]}

    with (tmp_path / "control.lock").open("w") as control_lock:
        def trigger(config, request):
            fcntl.flock(control_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            events.append("trigger-after-unlock")
            fcntl.flock(control_lock, fcntl.LOCK_UN)

        scan_once(FixtureWorker(), control_lock, trigger=trigger)

    assert events == ["scan", "trigger-after-unlock"]


def test_worker_uses_startup_revision_even_if_checkout_head_moves(tmp_path):
    task = revised_manifest("e" * 40, "2026-09-11T00:00:00Z")

    class Relay:
        def refresh(self):
            pass

        def tasks(self):
            return [task]

        def validity(self, task):
            return True, ""

    class Manager:
        revision = "d" * 40

        def generations(self):
            return {}

        def collect_retired(self):
            pass

        def current_control_revision(self):
            return self.revision

    manager = Manager()
    worker = Worker(
        {"state_dir": str(tmp_path), "simulation": True},
        relay=Relay(),
        manager=manager,
        driver=object(),
    )
    manager.revision = task["worker_revision_sha"]
    trigger_control_update(
        {"state_dir": str(tmp_path)},
        {"revision": task["worker_revision_sha"], "task_id": task["task_id"]},
        run=lambda argv, **kwargs: None,
    )
    request = worker.scan()
    assert worker.running_control_revision == "d" * 40
    assert request["revision"] == task["worker_revision_sha"]
    assert worker.journal.tasks() == []


def test_restarted_worker_clears_already_satisfied_request(tmp_path):
    revision = "d" * 40
    task = manifest()

    class Relay:
        def refresh(self):
            pass

        def tasks(self):
            return []

    class Manager:
        def generations(self):
            return {}

        def collect_retired(self):
            pass

        def current_control_revision(self):
            return revision

    trigger_control_update(
        {"state_dir": str(tmp_path)},
        {"revision": revision, "task_id": task["task_id"]},
        run=lambda argv, **kwargs: None,
    )
    worker = Worker(
        {"state_dir": str(tmp_path), "simulation": True},
        relay=Relay(),
        manager=Manager(),
        driver=object(),
    )
    assert worker.scan() is None
    assert not (tmp_path / "control-update/request.json").exists()


def test_worker_exits_on_sigterm_while_control_update_holds_lock(tmp_path):
    import fcntl

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"state_dir": str(tmp_path)}))
    program = """
import fcntl, sys, threading
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from agent_ci import worker

class IdleWorker:
    def __init__(self, config, **kwargs):
        self.execution_thread = None
        self.config = config
        self.state_dir = Path(config['state_dir'])
        self.stop_event = threading.Event()
        self.active = None
    def scan(self):
        raise AssertionError('Worker must not scan during a control update')

flock = fcntl.flock
def observed_flock(stream, operation):
    if operation & fcntl.LOCK_SH:
        print('waiting', flush=True)
    return flock(stream, operation)
fcntl.flock = observed_flock
worker.Worker = IdleWorker
raise SystemExit(worker.main(['--config', sys.argv[2]]))
"""
    with (tmp_path / "control.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        child = subprocess.Popen(
            [
                sys.executable, "-c", program,
                str(Path(__file__).resolve().parents[1]), str(config),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert select.select([child.stdout], [], [], 5)[0], (
                "Worker did not reach the control lock"
            )
            assert child.stdout.readline().strip() == "waiting"
            child.terminate()
            _, error = child.communicate(timeout=5)
            assert child.returncode == 0, error
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()


@pytest.fixture
def recovery_worker(tmp_path, monkeypatch):
    """Real journal/sealing with controlled CLI and runtime failure boundaries."""
    from types import SimpleNamespace
    task = manifest()
    calls, uploads = [], []
    relay_tasks = [task]
    credentials = ["first"]

    class Relay:
        def refresh(self):
            pass
        def validity(self, task):
            return True, "current"
        def tasks(self):
            return relay_tasks
        def source_variants(self, frozen):
            return {
                "base": {"source_sha": frozen["base_sha"], "llvm_hash": "f" * 40,
                         "triton_version": "3.0.0"},
                "candidate": {"source_sha": frozen["tested_sha"], "llvm_hash": frozen["llvm_hash"],
                              "triton_version": "3.3.0"},
            }
        def publish_result(self, task, run_id, directory):
            uploads.append((task["task_id"], run_id))
            return hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()

    class Manager:
        def generations(self):
            return {}
        def collect_retired(self):
            pass
        def acquire_task(self, task, run_id):
            variants = {
                variant: {**source, "profile": variant, "backend_enabled": variant == "base",
                          "environment_fingerprint": variant, "image_id": "fixture"}
                for variant, source in task["variants"].items()
            }
            return dict(run_id=run_id, variants=variants, profile="fixture", backend_enabled=False,
                        environment_fingerprint="fixture", llvm_hash=task["llvm_hash"],
                        image_id="fixture", artifacts_host=str(local_run_dir(tmp_path, task, run_id) / "artifacts"))
        def stop_task(self, handle):
            pass
        def collect_artifacts(self, handle):
            pass
        def destroy_task(self, handle):
            pass

    class Executor:
        def __init__(self, config, state_dir, generation, task, relay, *, manager):
            self.run_dir = Path(generation["artifacts_host"]).parent
        def prepare(self, variant="candidate"):
            return tmp_path
        def write_context(self, policy, changes):
            pass

    def write_report(executor, status="fail"):
        report = {"status": status, "summary": "actual test conclusion", "checks": [], "reviews": []}
        atomic_json(executor.run_dir / "artifacts/agent-result.json", report)
        return report

    class Driver:
        health = {}
        def redact(self, text):
            return text
        def credentials_fingerprint(self):
            return credentials[0]
        def run(self, executor, **kwargs):
            calls.append(kwargs)
            write_report(executor)
            return {"exit_code": 1, "reason": "", "failure_code": "connection"}

    monkeypatch.setattr("agent_ci.worker.changed_files", lambda *a: [{"path": "README.md"}])
    monkeypatch.setattr("agent_ci.worker.minimum_checks", lambda *a, **kw: {"required_checks": [], "required_reviews": []})
    worker = Worker(
        {"state_dir": str(tmp_path), "simulation": True, "retry_delay_seconds": 0},
        relay=Relay(),
        manager=Manager(),
        driver=Driver(),
        executor_factory=Executor,
        control_request_selector=lambda config, current, requests, **kwargs: {
            "checked_task_ids": tuple(row["task_id"] for row in requests),
            "request": None,
        },
    )
    return SimpleNamespace(worker=worker, task=task, calls=calls, uploads=uploads,
                           tasks=relay_tasks, credentials=credentials, write_report=write_report)


def test_task_uses_installed_control_regardless_of_dispatch_revision(recovery_worker):
    f = recovery_worker
    worker = f.worker
    current_revision = "f" * 40
    worker.running_control_revision = current_revision
    worker.manager.current_control_revision = lambda: current_revision
    worker.control_request_selector = lambda config, current, requests, **kwargs: {
        "checked_task_ids": (f.task["task_id"],),
        "request": None,
    }
    acquired = worker.manager.acquire_task
    runtime_tasks = []

    def observe_task(task, run_id):
        runtime_tasks.append(task)
        return acquired(task, run_id)

    worker.manager.acquire_task = observe_task
    assert worker.scan() is None
    assert len(f.calls) == 1
    assert runtime_tasks[0]["worker_revision_sha"] == "d" * 40
    frozen = json.loads(worker.journal.task(f.task["task_id"])["manifest"])
    assert "_use_installed_control" not in frozen
    assert not (worker.state_dir / "control-update/request.json").exists()
    result = json.loads(
        Path(worker.journal.delivery(f.task["task_id"])["payload_path"]).read_text()
    )
    assert result["task"]["worker_revision_sha"] == "d" * 40
    assert result["environment"]["control_revision"] == current_revision
    assert result["status"] == "fail"


def test_remote_control_tip_requests_update_without_using_task_revision(recovery_worker):
    f = recovery_worker
    worker = f.worker
    worker.running_control_revision = "c" * 40
    worker.manager.current_control_revision = lambda: worker.running_control_revision
    worker.control_request_selector = lambda config, current, requests, **kwargs: {
        "checked_task_ids": (),
        "request": {**requests[0], "revision": "f" * 40},
    }
    request = worker.scan()
    assert request == {
        "revision": "f" * 40,
        "task_id": f.task["task_id"],
        "captured_at": f.task["captured_at"],
    }
    assert request["revision"] != f.task["worker_revision_sha"]
    assert not f.calls and worker.journal.tasks() == []

    # After the updater installs the observed branch tip and restarts the
    # Worker, the same task is admitted without changing its frozen manifest.
    worker.running_control_revision = request["revision"]
    worker.manager.current_control_revision = lambda: worker.running_control_revision
    worker.control_request_selector = lambda config, current, requests, **kwargs: {
        "checked_task_ids": (f.task["task_id"],),
        "request": None,
    }
    assert worker.scan() is None
    assert len(f.calls) == 1
    result = json.loads(
        Path(worker.journal.delivery(f.task["task_id"])["payload_path"]).read_text()
    )
    assert result["task"]["worker_revision_sha"] == "d" * 40
    assert result["environment"]["control_revision"] == "f" * 40


@pytest.mark.parametrize(
    "failure",
    [
        OSError("control mirror unavailable"),
        ValueError("control branch diverged from installed checkout"),
    ],
    ids=["network", "diverged"],
)
def test_control_freshness_failure_stays_retryable(recovery_worker, failure):
    f = recovery_worker
    worker = f.worker
    worker.running_control_revision = "f" * 40
    worker.manager.current_control_revision = lambda: worker.running_control_revision

    def unavailable(*args, **kwargs):
        raise failure

    worker.control_request_selector = unavailable
    assert worker.scan() is None
    assert not f.calls and worker.journal.tasks() == []
    health = json.loads((worker.state_dir / "health/worker.json").read_text())
    assert health["control_update"] == "blocked"
    assert str(failure) in health["error"]
    assert not worker.journal.tasks()

    worker.control_request_selector = lambda config, current, requests, **kwargs: {
        "checked_task_ids": (f.task["task_id"],),
        "request": None,
    }
    assert worker.scan() is None
    assert len(f.calls) == 1


def test_failed_report_is_final_even_when_cli_exits_with_connection_error(recovery_worker):
    f = recovery_worker
    f.worker.scan()
    row = f.worker.journal.task(f.task["task_id"])
    assert len(f.calls) == 1 and row["phase"] == "published"
    assert f.worker.journal.result_status(f.worker.journal.delivery(f.task["task_id"])) == "fail"


def test_restart_preserves_task_budget_and_does_not_refund_attempts(tmp_path):
    journal = Journal(tmp_path)
    task = manifest(trigger_id="12345:2")
    journal.register(task)
    journal.claim(task["task_id"], "execution_attempts_used", 3)
    original = journal.claim(task["task_id"], "codex_attempts_used", 10, timeout=21600)
    for _ in range(2):
        journal = Journal(tmp_path)
        journal.restart(task["task_id"])
        assert json.loads(journal.task(task["task_id"])["manifest"]) == task
        budget = journal.claim(task["task_id"], "execution_attempts_used", 3)
        assert budget["codex_deadline_at"] == original["codex_deadline_at"]
        assert budget["codex_attempts_used"] == 1
    with pytest.raises(ValueError, match="次数预算"):
        journal.claim(task["task_id"], "execution_attempts_used", 3)
    journal.update(task["task_id"], budget={**budget, "codex_deadline_at": 1})
    with pytest.raises(ValueError, match="时间预算"):
        Journal(tmp_path).claim(task["task_id"], "codex_attempts_used", 10)


def test_sealed_crash_window_uploads_before_runtime_and_current_checks(recovery_worker):
    f = recovery_worker
    row = f.worker.journal.register(f.task)
    f.worker.finish(row, {"status": "fail", "summary": "saved failure"})
    digest = f.worker.journal.delivery(f.task["task_id"])["digest"]
    f.worker.journal.update(f.task["task_id"], delivery=None, phase="sealing")
    f.worker.manager._daemon = lambda: (_ for _ in ()).throw(OSError("Docker down"))
    f.worker.relay.validity = lambda task: (False, "superseded")
    f.worker.journal = Journal(f.worker.state_dir)
    f.worker.scan()
    assert not f.calls and len(f.uploads) == 1
    assert f.worker.journal.delivery(f.task["task_id"])["digest"] == digest
    assert f.worker.journal.task(f.task["task_id"])["phase"] == "published"


def test_auth_wait_requires_changed_credentials_and_shares_budget(recovery_worker):
    f = recovery_worker
    original = f.worker.driver.run
    def unauthenticated(executor, **kwargs):
        f.calls.append(kwargs)
        return {"exit_code": 1, "failure_code": "authentication"}
    f.worker.driver.run = unauthenticated
    f.worker.scan()
    f.worker.scan()
    assert len(f.calls) == 1
    row = f.worker.journal.task(f.task["task_id"])
    deadline = row["budget"]["codex_deadline_at"]
    assert row["recovery"]["state"] == "waiting_dependency"
    f.credentials[0] = "changed"
    f.worker.driver.run = original
    f.worker.scan()
    row = f.worker.journal.task(f.task["task_id"])
    assert len(f.calls) == 2 and row["budget"]["codex_attempts_used"] == 2
    assert row["budget"]["codex_deadline_at"] == deadline
    assert row["phase"] == "published"


def test_no_progress_resume_switches_session_once_without_new_budget(recovery_worker):
    f = recovery_worker
    def disconnected(executor, **kwargs):
        f.calls.append(kwargs)
        if len(f.calls) == 4:
            f.write_report(executor)
        return {"exit_code": 1, "failure_code": "connection", "session_reused": len(f.calls) > 1, "progressed": False}
    f.worker.driver.run = disconnected
    f.worker.scan()
    assert len(f.calls) == 4 and f.calls[-1]["session_mode"] == "new"
    budget = f.worker.journal.task(f.task["task_id"])["budget"]
    assert budget["codex_attempts_used"] == 4 and budget["session_switches"] == 1


def test_repeated_invalid_report_switches_despite_cli_activity(recovery_worker):
    f = recovery_worker
    def repeating(executor, **kwargs):
        f.calls.append(kwargs)
        atomic_json(executor.run_dir / "artifacts/agent-result.json", {
            "status": "pass", "summary": f"Still working {len(f.calls)}",
        })
        if len(f.calls) == 4:
            f.write_report(executor)
        return {"exit_code": 0, "session_reused": len(f.calls) > 1,
                "progressed": True, "activity_observed": True}
    f.worker.driver.run = repeating
    f.worker.scan()
    assert len(f.calls) == 4 and f.calls[-1]["session_mode"] == "new"
    assert "checks" in f.calls[1]["recovery"] and "reviews" in f.calls[1]["recovery"]
    budget = f.worker.journal.task(f.task["task_id"])["budget"]
    assert budget["codex_attempts_used"] == 4 and budget["session_switches"] == 1


def test_report_correction_finishes_without_session_switch(recovery_worker):
    f = recovery_worker
    def correcting(executor, **kwargs):
        f.calls.append(kwargs)
        if len(f.calls) == 1:
            atomic_json(executor.run_dir / "artifacts/agent-result.json", {"status": "pass"})
        else:
            assert "checks" in kwargs["recovery"]
            f.write_report(executor, "pass")
        return {"exit_code": 0, "session_reused": len(f.calls) > 1}
    f.worker.driver.run = correcting
    f.worker.scan()
    assert len(f.calls) == 2
    assert f.worker.journal.task(f.task["task_id"])["budget"]["session_switches"] == 0


def test_auth_rebuild_hands_off_evidence_without_restoring_installs(recovery_worker):
    f = recovery_worker
    first_run = []
    def recovering(executor, **kwargs):
        f.calls.append(kwargs)
        if len(f.calls) == 1:
            first_run.append(executor.run_dir)
            (executor.run_dir / "artifacts/recovery-notes.md").write_text("Dependency fixed; frontend install still required")
            return {"exit_code": 1, "failure_code": "authentication"}
        assert executor.run_dir != first_run[0]
        handoff = json.loads((executor.run_dir / "artifacts/recovery-context.json").read_text())
        assert handoff["environment_rebuilt"] is True
        previous = handoff["previous_runs"][0]
        assert previous["run_id"] == first_run[0].name and previous["installation_valid"] is False
        assert "frontend install still required" in (executor.run_dir / "artifacts" / previous["files"][0]).read_text()
        assert not (executor.run_dir / "artifacts/agent-result.json").exists()
        f.write_report(executor)
        return {"exit_code": 0}
    f.worker.driver.run = recovering
    f.worker.scan()
    deadline = f.worker.journal.task(f.task["task_id"])["budget"]["codex_deadline_at"]
    f.credentials[0] = "changed"
    f.worker.scan()
    row = f.worker.journal.task(f.task["task_id"])
    assert len(f.calls) == 2 and row["phase"] == "published"
    assert row["budget"]["codex_deadline_at"] == deadline


def test_restart_lineage_survives_reload_and_missing_history(recovery_worker):
    f = recovery_worker
    journal = f.worker.journal
    first = journal.register(f.task)
    journal.claim(f.task["task_id"], "execution_attempts_used", 3)
    second = journal.restart(f.task["task_id"])
    journal.claim(f.task["task_id"], "execution_attempts_used", 3)
    journal.restart(f.task["task_id"])
    journal.claim(f.task["task_id"], "execution_attempts_used", 3)
    f.worker.journal = Journal(f.worker.state_dir)
    previous = f.worker.previous_runs(f.task["task_id"])
    assert [path.name for path, _ in previous] == [second["run_id"], first["run_id"]]
    (previous[1][0] / "task.json").unlink()
    assert len(f.worker.previous_runs(f.task["task_id"])) == 1


def test_rate_limit_does_not_spend_pending_no_progress_switch(recovery_worker, monkeypatch):
    f = recovery_worker
    f.worker.config["codex_resume_no_progress_attempts"] = 1
    def throttled(executor, **kwargs):
        f.calls.append(kwargs)
        if len(f.calls) == 3:
            f.write_report(executor)
        return {"exit_code": 1, "failure_code": "rate_limit", "session_reused": True}
    f.worker.driver.run = throttled
    monkeypatch.setattr("agent_ci.worker.threading.Event.wait", lambda *a, **k: False)
    f.worker.scan()
    assert len(f.calls) == 3 and all("session_mode" not in call for call in f.calls)
    assert f.worker.journal.task(f.task["task_id"])["budget"]["session_switches"] == 0


def test_upload_backoff_keeps_immutable_result_until_success(recovery_worker, monkeypatch):
    f = recovery_worker
    row = f.worker.journal.register(f.task)
    f.worker.finish(row, {"status": "fail", "summary": "preserved"})
    box = f.worker.journal.delivery(f.task["task_id"])
    original = Path(box["payload_path"]).read_bytes()
    now = [time.time()]
    monkeypatch.setattr("agent_ci.worker.time.time", lambda: now[0])
    publish = f.worker.relay.publish_result
    f.worker.relay.publish_result = lambda *a: (_ for _ in ()).throw(OSError("relay outage"))
    for delay in (60, 120, 300, 300, 3600, 3600):
        f.worker.retry_delivery(row)
        box = f.worker.journal.delivery(f.task["task_id"])
        assert box["next_retry_at"] == now[0] + delay
        assert Path(box["payload_path"]).read_bytes() == original
        now[0] = box["next_retry_at"]
    f.worker.relay.publish_result = publish
    f.worker.retry_delivery(row)
    assert f.worker.journal.result_status(f.worker.journal.delivery(f.task["task_id"])) == "fail"
    assert f.worker.journal.task(f.task["task_id"])["phase"] == "published"


def test_long_execution_does_not_block_outbox_or_kill_silent_live_task(recovery_worker):
    import threading
    f = recovery_worker
    done, started = threading.Event(), threading.Event()
    def running(executor, **kwargs):
        started.set()
        assert done.wait(5)
        f.write_report(executor)
        return {"exit_code": 0}
    f.worker.driver.run = running
    f.worker.background = True
    old = revised_manifest("f" * 40, "2026-09-12T00:00:00Z")
    row = f.worker.journal.register(old)
    f.worker.finish(row, {"status": "fail", "summary": "old result"})
    try:
        f.worker.scan()
        assert started.wait(2)
        f.worker.journal.update(f.task["task_id"], last_progress_at=time.time()-3700)
        f.worker.inspect_active()
        assert f.worker.active is not None and not f.worker.active.cancelled.is_set()
        assert f.worker.journal.record(f.task["task_id"])["progress_state"] == "stalled_review"
        assert (old["task_id"], row["run_id"]) in f.uploads
        f.worker.heartbeat()
        heartbeat = json.loads((f.worker.state_dir / "health/worker.json").read_text())
        assert heartbeat["active_task"] == f.task["task_id"] and heartbeat["active_run_id"]
    finally:
        done.set()
        if f.worker.execution_thread:
            f.worker.execution_thread.join(5)



def test_recoverable_task_holds_control_revision_but_outbox_does_not(
    recovery_worker,
):
    f = recovery_worker
    worker = f.worker
    worker.journal.register(f.task)
    worker.running_control_revision = f.task["worker_revision_sha"]
    worker.manager.current_control_revision = lambda: worker.running_control_revision
    worker.journal.update(f.task["task_id"], credentials_fingerprint="first")
    worker.recovery(f.task["task_id"], "waiting_dependency", "authentication", "wait_credentials")
    newer = revised_manifest("f" * 40, "2026-09-12T00:00:00Z")
    f.tasks.append(newer)
    selected = []

    def select(config, current, waiting, **kwargs):
        selected.extend(waiting)
        return {
            "checked_task_ids": (),
            "request": {**waiting[0], "revision": "f" * 40},
        }

    worker.control_request_selector = select
    assert worker.scan() is None and not selected and not f.calls
    row = worker.journal.task(f.task["task_id"])
    worker.finish(row, {"status": "infra_error", "summary": "no more recovery"})
    worker.schedule_delivery(worker.journal.task(f.task["task_id"]))
    request = worker.scan()
    assert request["revision"] == "f" * 40
    assert selected and not f.calls


def test_old_unbudgeted_task_is_not_silently_given_a_new_execution(recovery_worker):
    f = recovery_worker
    f.worker.journal.register(f.task)
    f.worker.journal.update(f.task["task_id"], budget=None)
    f.worker.scan()
    assert not f.calls
    result = json.loads(Path(f.worker.journal.delivery(f.task["task_id"])["payload_path"]).read_text())
    assert result["status"] == "infra_error" and "重新派发" in result["summary"]


def test_budget_migration_requires_explicit_contiguous_start_evidence(tmp_path):
    journal = Journal(tmp_path)
    task = manifest()
    journal.register(task)
    journal.update(task["task_id"], budget=None)
    journal.event(task["task_id"], "codex_exit", {"exit_code": 1})
    assert journal.recover_budget(task["task_id"]) is None
    evidence = {"deadline_at": time.time()+21600, "execution_attempt": 1, "session_switches": 0}
    journal.event(task["task_id"], "codex_start", {"attempt": 1, **evidence})
    journal.event(task["task_id"], "codex_start", {"attempt": 2, **evidence})
    recovered = journal.recover_budget(task["task_id"])
    assert recovered["codex_attempts_used"] == 2
    assert recovered["codex_deadline_at"] > time.time()


def test_historical_outbox_updates_original_run_only(recovery_worker):
    f = recovery_worker
    old = f.worker.journal.register(f.task)
    f.worker.finish(old, {"status": "fail", "summary": "saved"})
    previous = f.worker.journal.record(f.task["task_id"])
    current = f.worker.journal._new(f.task, previous)
    f.worker.recover_local()
    assert f.worker.journal.task(f.task["task_id"], old["run_id"])["phase"] == "published"
    assert f.worker.journal.task(f.task["task_id"], current["run_id"])["phase"] == "preparing"
    assert f.uploads == [(f.task["task_id"], old["run_id"])]


def test_sealing_exhaustion_preserves_report_and_manual_resume_does_not_reset_budget(recovery_worker, monkeypatch):
    f = recovery_worker
    row = f.worker.journal.register(f.task)
    now = [time.time()]
    monkeypatch.setattr("agent_ci.worker.time.time", lambda: now[0])
    monkeypatch.setattr("agent_ci.worker.seal_result", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk unavailable")))
    report = {"status": "fail", "summary": "real test failure"}
    assert not f.worker.finish(row, report)
    for delay in (30, 60):
        now[0] += delay
        assert not f.worker.seal_checkpoint(row)
    now[0] += 3600
    assert not f.worker.seal_checkpoint(row)
    saved = f.worker.journal.record(f.task["task_id"])
    assert saved["checkpoint"]["report"] == report and saved["sealing_attempts"] == 3
    with pytest.raises(ValueError, match="预算已耗尽"):
        f.worker.journal.resume(f.task["task_id"])
    monkeypatch.setattr("maintenance.retention.retain_local", lambda config: {"pause_intake": True})
    f.worker.running_control_revision = "f" * 40
    # A later disk outage must not revive the exhausted sealing budget or replace
    # its terminal reason with a generic dependency wait on every scan.
    for _ in range(2):
        now[0] += 3600
        f.worker.scan()
        saved = f.worker.journal.record(f.task["task_id"])
        assert saved["recovery"]["state"] == "exhausted"
        assert saved["recovery"]["failure_code"] == "sealing_failed"
        assert saved["sealing_attempts"] == 3 and saved["checkpoint"]["report"] == report
    assert not f.calls



def test_slow_upload_does_not_block_heartbeat_or_cancellation(recovery_worker):
    import threading
    f = recovery_worker
    started, release = threading.Event(), threading.Event()
    original = f.worker.relay.publish_result
    def slow_publish(*args):
        started.set()
        assert release.wait(5)
        return original(*args)
    f.worker.relay.publish_result = slow_publish
    old = revised_manifest("f"*40, "2026-09-12T00:00:00Z")
    row = f.worker.journal.register(old)
    f.worker.finish(row, {"status": "fail", "summary": "saved"})
    f.worker.background = True
    from agent_ci.worker import ActiveTask
    f.worker.journal.register(f.task)
    active = ActiveTask(f.task, f.worker.manager)
    f.worker.active = active
    try:
        f.worker.scan()
        assert started.wait(2)
        f.worker.relay.validity = lambda task: (False, "PR closed")
        f.worker.scan()
        assert active.cancelled.is_set()
        f.worker.heartbeat()
        heartbeat = json.loads((f.worker.state_dir / "health/worker.json").read_text())
        assert time.time()-heartbeat["heartbeat_at"] < 2
        assert not f.uploads
    finally:
        release.set()
        if f.worker.delivery_thread:
            f.worker.delivery_thread.join(5)
    assert f.uploads == [(old["task_id"], row["run_id"])]


def test_relay_write_does_not_hold_control_snapshot_read_lock(tmp_path):
    import threading
    from agent_ci.relay import GitRelay
    remote = tmp_path / "remote"
    remote.mkdir()
    relay = GitRelay(str(remote), tmp_path / "relay", allow_local=True)
    started, release = threading.Event(), threading.Event()
    def git(args, **kwargs):
        if args[0] == "ls-remote":
            started.set()
            assert release.wait(5)
        stdout = b"d"*40 if args[0] == "rev-parse" else b""
        return subprocess.CompletedProcess(args, 0, stdout, b"")
    relay.git = git
    writer = threading.Thread(target=lambda: relay.write("results", {"result.json": b"{}"}))
    writer.start()
    try:
        assert started.wait(2)
        relay.refresh()
        assert relay.control_snapshot == "d"*40
    finally:
        release.set()
        writer.join(5)
    assert not writer.is_alive()


def test_offline_relay_does_not_suspend_local_execution_deadline(recovery_worker):
    from agent_ci.worker import ActiveTask
    f = recovery_worker
    f.worker.journal.register(f.task)
    budget = f.worker.journal.record(f.task["task_id"])["budget"]
    f.worker.journal.update(f.task["task_id"], budget={**budget, "codex_deadline_at": 1})
    active = ActiveTask(f.task, f.worker.manager)
    f.worker.active = active
    f.worker.relay.refresh = lambda: (_ for _ in ()).throw(OSError("network unavailable"))
    f.worker.scan()
    assert active.cancelled.is_set() and active.reason == "recovery_exhausted"
    heartbeat = json.loads((f.worker.state_dir / "health/worker.json").read_text())
    assert heartbeat["control_channel"] == "unreachable"



def test_restart_seals_failed_report_with_both_frozen_llvm_environments(recovery_worker, monkeypatch):
    f = recovery_worker
    original_task = json.loads(json.dumps(f.task))
    # Interrupt after the real execution and stable host checkpoint, before the
    # seal commit point. This must not turn the existing failure into a rerun.
    monkeypatch.setattr(f.worker, "seal_checkpoint", lambda row: False)
    f.worker.scan()
    saved = f.worker.journal.record(f.task["task_id"])
    checkpoint = saved["checkpoint"]
    assert len(f.calls) == 1 and checkpoint["complete"]
    assert checkpoint["report"]["status"] == "fail"
    assert not (f.worker.journal.run_dir(f.task["task_id"]) / "sealed/result.json").exists()
    runtimes = checkpoint["environment"]["variants"]
    assert runtimes["base"]["llvm_hash"] == "f" * 40
    assert runtimes["candidate"]["llvm_hash"] == original_task["llvm_hash"]
    assert runtimes["base"]["backend_enabled"] is True
    assert runtimes["candidate"]["backend_enabled"] is False

    def forbidden(*args, **kwargs):
        raise AssertionError("A complete checkpoint must not rebuild, retest or resolve new runtimes")
    monkeypatch.setattr(f.worker.manager, "acquire_task", forbidden)
    monkeypatch.setattr(f.worker.driver, "run", forbidden)
    monkeypatch.setattr(f.worker.relay, "source_variants", forbidden)
    restarted = Worker(f.worker.config, relay=f.worker.relay, manager=f.worker.manager,
                       driver=f.worker.driver, executor_factory=f.worker.executor_factory)
    restarted.scan()
    delivery = restarted.journal.delivery(f.task["task_id"])
    result = json.loads(Path(delivery["payload_path"]).read_text())
    assert result["status"] == "fail" and result["run_id"] == saved["run_id"]
    assert result["task"] == original_task == f.task and "variants" not in f.task
    assert result["environment"] == checkpoint["environment"]
    assert len(f.calls) == 1 and f.uploads == [(f.task["task_id"], saved["run_id"])]
    assert restarted.journal.task(f.task["task_id"])["phase"] == "published"



def test_invalid_source_metadata_is_terminal_configuration_failure(recovery_worker, monkeypatch):
    from agent_ci.protocol import ContractError
    f = recovery_worker
    def invalid_sources(task):
        raise ContractError("Task base environment identity does not match frozen source")
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid frozen source metadata must not create an environment")
    monkeypatch.setattr(f.worker.relay, "source_variants", invalid_sources)
    monkeypatch.setattr(f.worker.manager, "acquire_task", forbidden)
    f.worker.scan()
    f.worker.scan()
    saved = f.worker.journal.record(f.task["task_id"])
    result = json.loads(Path(saved["delivery"]["payload_path"]).read_text())
    assert result["status"] == "infra_error" and saved["phase"] == "published"
    assert saved["recovery"]["failure_code"] == "configuration_invalid"
    assert saved["recovery"]["state"] == "exhausted"
    assert not f.calls and len(f.uploads) == 1
