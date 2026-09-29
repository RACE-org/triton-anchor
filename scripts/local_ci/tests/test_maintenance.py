"""Retention and public health behavior, without remote attachment lifecycle."""
from datetime import datetime, timezone
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from maintenance import health
from maintenance.retention import retain_local
from maintenance.health import public_snapshot
from maintenance.watchdog import evaluate


RUN_LAYOUTS = ["", "pr/branch-main/pr-56", "push/branch-release%2F3.0"]


def make_run(root, run, phase, published, layout=""):
    path = root / "runs" / layout / ("a" * 64) / run
    (path / "artifacts").mkdir(parents=True)
    (path / "logs").mkdir()
    (path / "inputs/candidate/checkout").mkdir(parents=True)
    (path / "inputs/candidate/checkout/source.py").write_text("temporary source")
    (path / "sealed").mkdir()
    (path / "artifacts/report.json").write_text("result")
    (path / "logs/command.log").write_text("private log")
    (path / "sealed/result.json").write_text('{"verdict":"pass"}')
    (path / "state.json").write_text(
        json.dumps({"phase": phase, "delivery": {"published": published}})
    )
    return path


def test_sha_directory_health_and_retention_keep_internal_task_identity(tmp_path):
    run = make_run(tmp_path, "pending", "publish_pending", None, "push/branch-main")
    sha_parent = run.parent.with_name("b" * 40)
    run.parent.rename(sha_parent)
    run = sha_parent / run.name
    record = json.loads((run / "state.json").read_text())
    record.update(task_id="a" * 64, head_sha="b" * 40)
    (run / "state.json").write_text(json.dumps(record))
    config = {"state_dir": str(tmp_path)}
    report = retain_local(config, apply=False)
    assert report["protected"] == [{"task_id": "a" * 64, "run_id": "pending"}]
    snapshot = health.collect(
        {**config, "monitor_services": []}, manager=SimpleNamespace(health=lambda: {}),
    )
    assert snapshot["active_task"]["task_id"] == "a" * 64


@pytest.mark.parametrize("layout", RUN_LAYOUTS, ids=["legacy", "pr", "push"])
def test_retention_expires_only_published_and_preserves_summary(tmp_path, layout):
    now = 40 * 86400
    expired = make_run(tmp_path, "old", "published", 86400, layout)
    # Earlier cleanup left the input clones behind even after recording expiry.
    (expired / "retention.json").write_text("{}")
    pending = make_run(tmp_path, "pending", "publish_pending", 86400, layout)
    fresh = make_run(tmp_path, "fresh", "published", now - 86400, layout)
    report = retain_local(
        {"state_dir": str(tmp_path), "state_min_free_bytes": 0}, now=now
    )
    assert len(report["expired"]) == 1 and not report["pause_intake"]
    assert not (expired / "artifacts").exists()
    assert not (expired / "inputs").exists()
    assert (expired / "sealed/result.json").exists()
    assert (pending / "artifacts/report.json").exists()
    assert (pending / "inputs/candidate/checkout/source.py").exists()
    assert (fresh / "logs/command.log").exists()


def test_retention_rejects_linked_branch_directory(tmp_path):
    external = tmp_path / "external"
    run = make_run(external, "old", "published", 86400)
    branch = tmp_path / "runs/pr/branch-main"
    branch.mkdir(parents=True)
    (branch / "pr-56").symlink_to(external / "runs", target_is_directory=True)
    report = retain_local(
        {"state_dir": str(tmp_path), "state_min_free_bytes": 0}, now=40 * 86400
    )
    assert report["errors"] == [{"run": "old", "reason": "symlink"}]
    assert (run / "artifacts/report.json").exists()


@pytest.mark.parametrize("layout", RUN_LAYOUTS, ids=["legacy", "pr", "push"])
def test_health_reports_active_runs_in_each_layout(tmp_path, layout):
    make_run(tmp_path, "active", "running", 0, layout)
    make_run(tmp_path, "pending", "publish_pending", 0, layout)
    make_run(tmp_path, "done", "published", 1, layout)
    report = health.collect(
        {"state_dir": str(tmp_path), "monitor_services": []},
        manager=SimpleNamespace(health=lambda: {}),
    )
    assert {row["run_id"] for row in report["tasks"]} == {"active", "pending"}
    assert report["active_task"]["task_id"] == "a" * 64
    assert report["active_task"]["run_id"] == "active"
    assert report["uploads"][0]["task_id"] == "a" * 64


def test_retention_budget_pauses_without_deleting_required_pending(tmp_path):
    pending = make_run(tmp_path, "pending", "publish_pending", 0)
    report = retain_local(
        {"state_dir": str(tmp_path), "evidence_max_bytes": 1, "state_min_free_bytes": 0}
    )
    assert report["pause_intake"]
    assert (pending / "artifacts/report.json").exists()
    assert (pending / "inputs/candidate/checkout/source.py").exists()


def test_health_whitelist_removes_nested_private_configuration():
    private = "PRIVATE_SENTINEL"
    result = public_snapshot(
        {
            "worker_id": "worker-1",
            "collected_at": "2026-09-10T00:00:00Z",
            "config": private,
            "runtime": {"endpoint": private, "available": True, "env": private},
            "environments": {
                "images": [{"env": private}],
                "runtime": {"endpoint": private},
            },
            "tasks": [
                {
                    "task_id": "a" * 64,
                    "run_id": "run-1",
                    "stage": "running",
                    "config": private,
                }
            ],
            "images": [
                {
                    "release_id": "image-1",
                    "image_id": "sha256:" + "a" * 64,
                    "state": "ready",
                    "env": private,
                    "log_path": private,
                }
            ],
            "storage": [{"filesystem_free_bytes": 10, "path": private}],
        }
    )
    assert private not in json.dumps(result)
    assert result["runtime"]["available"]


def test_health_publishes_only_current_codex_status_and_original_upload_age(tmp_path):
    run = make_run(tmp_path, "active", "running", None)
    task_id = "a" * 64
    (run / "state.json").write_text(json.dumps({
        "task_id": task_id, "phase": "running", "updated": 100, "last_progress_at": 100,
    }))
    pending = make_run(tmp_path, "pending", "publish_pending", None)
    (pending / "state.json").write_text(json.dumps({
        "task_id": "b" * 64, "phase": "publish_pending", "updated": 1900, "abandoned": True,
        "delivery": {"queued_at": 100, "attempts": 20},
    }))
    heartbeat = {
        "pid": os.getpid(), "heartbeat_at": 1900, "active_task": task_id,
        "codex_status": "connection_error", "codex_alive": True,
        "last_progress_at": 100, "control_channel": "unreachable",
        "private_error": "PRIVATE_SENTINEL",
    }
    (tmp_path / "health").mkdir()
    path = tmp_path / "health/worker.json"
    path.write_text(json.dumps(heartbeat))
    config = {"state_dir": str(tmp_path), "monitor_services": []}
    manager = SimpleNamespace(health=lambda: {})
    snapshot = public_snapshot(health.collect(config, now=1900, manager=manager))
    assert snapshot["active_task"]["codex_status"] == "connection_error"
    assert snapshot["active_task"]["last_progress_at"] == health.iso(100)
    assert snapshot["uploads"][0]["queued_at"] == health.iso(100)
    assert snapshot["poller"]["last_poll_status"] == "error"
    assert "PRIVATE_SENTINEL" not in json.dumps(snapshot)
    heartbeat["last_progress_at"] = None
    path.write_text(json.dumps(heartbeat))
    assert health.collect(config, now=1900, manager=manager)["active_task"]["last_progress_at"] == health.iso(100)
    heartbeat["active_task"] = "c" * 64
    path.write_text(json.dumps(heartbeat))
    snapshot = public_snapshot(health.collect(config, now=1900, manager=manager))
    assert snapshot["active_task"]["codex_status"] is None
    assert public_snapshot({})["poller"]["last_poll_status"] == "unknown"


def test_health_reports_failed_pending_control_update(tmp_path):
    manager = SimpleNamespace(health=lambda: {"images": [], "attempts": []})
    request = tmp_path / "control-update/request.json"
    request.parent.mkdir(parents=True)
    request.write_text(
        json.dumps(
            {
                "schema": "triton-anchor-local-ci-control-update-request",
                "revision": "a" * 40,
                "task_id": "b" * 64,
                "requested_at": 10,
            }
        )
    )
    request.chmod(0o600)

    def show_service(argv, **kwargs):
        failed = argv[3] == "triton-anchor-local-ci-control-update.service"
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "LoadState=loaded\n"
                + ("ActiveState=failed\nSubState=failed\nResult=exit-code\n" if failed else "ActiveState=active\nSubState=running\nResult=success\n")
            ),
        )

    with patch.object(health.subprocess, "run", side_effect=show_service):
        snapshot = health.collect(
            {"state_dir": str(tmp_path)}, now=20, manager=manager
        )
    assert snapshot["control_update"] == {
        "state": "failed",
        "requested_revision": "a" * 40,
        "task_id": "b" * 64,
        "requested_at": "1970-01-01T00:00:10Z",
    }


def test_health_reports_worker_blocked_control_update(tmp_path):
    worker = tmp_path / "health/worker.json"
    worker.parent.mkdir(parents=True)
    worker.write_text(
        json.dumps(
            {
                "heartbeat_at": 10,
                "pid": os.getpid(),
                "control_update": "blocked",
            }
        )
    )
    manager = SimpleNamespace(health=lambda: {"images": [], "attempts": []})
    snapshot = health.collect(
        {"state_dir": str(tmp_path), "monitor_services": []},
        now=20,
        manager=manager,
    )
    assert snapshot["control_update"]["state"] == "blocked"


def snapshot(now):
    return {
        "worker_id": "worker-1",
        "collected_at": now.isoformat(),
        "state": "healthy",
        "poller": {"alive": True, "heartbeat_stale": False},
        "runtime": {"available": True, "rootless": True},
    }


def test_watchdog_unknown_does_not_clear_incident_or_claim_offline():
    now = datetime(2026, 9, 10, tzinfo=timezone.utc)
    broken = snapshot(now)
    broken["poller"]["alive"] = False
    first = evaluate({"workers": [broken]}, now=now)
    unknown = evaluate(
        {"source_error": True, "expected_workers": ["worker-1"]}, first, now=now
    )
    assert unknown["source_state"] == "unknown"
    assert unknown["active"] == first["active"]
    assert not any(r["code"] == "snapshot_stale" for r in unknown["active"].values())
    recovered = evaluate({"workers": [snapshot(now)]}, unknown, now=now)
    assert not recovered["active"]
    assert recovered["history"][-1]["transition"] == "recovered"
    assert (
        len(
            evaluate(
                {"workers": [snapshot(now)]},
                {"active": {}, "history": [{}] * 110},
                now=now,
            )["history"]
        )
        == 100
    )


def test_retention_protects_unstarted_null_delivery(tmp_path):
    run = make_run(tmp_path, "preparing", "preparing", 0)
    (run / "state.json").write_text(
        json.dumps({"phase": "preparing", "delivery": None})
    )
    report = retain_local({"state_dir": str(tmp_path), "state_min_free_bytes": 0})
    assert report["protected"] == [{"task_id": "a" * 64, "run_id": "preparing"}]
    assert not report["errors"]


def test_health_exposes_recovery_identity_without_private_checkpoint(tmp_path):
    now = 9 * 86400
    run = make_run(tmp_path, "active", "running", None)
    manifest = {"repository": "RACE-org/triton-anchor", "head_sha": "b" * 40,
                "tested_sha": "c" * 40, "pr_number": 62}
    (run / "task.json").write_text(json.dumps(manifest))
    record = {"task_id": "a" * 64, "run_id": "active", "phase": "running", "updated": now,
              "checkpoint": {"report": "PRIVATE_SENTINEL"}, "last_progress_at": now-1900,
              "budget": {"codex_attempts_used": 3, "codex_deadline_at": now+3600,
                         "execution_attempts_used": 2, "session_switches": 1},
              "recovery": {"state": "retry_wait", "failure_code": "connection", "action": "resume",
                           "next_retry_at": now+30, "session_id": "PRIVATE_SENTINEL"}}
    (run / "state.json").write_text(json.dumps(record))
    done = make_run(tmp_path, "done", "published", now-60)
    (done / "task.json").write_text(json.dumps(manifest))
    (done / "state.json").write_text(json.dumps({"task_id": "d" * 64, "phase": "published",
        "updated": now-60, "detail": {"result_status": "pass"}}))
    result = health.collect({"state_dir": str(tmp_path), "monitor_services": []}, now=now,
                            manager=SimpleNamespace(health=lambda: {}))
    task = result["active_task"]
    assert (task["repository"], task["head_sha"], task["tested_sha"]) == tuple(manifest[k] for k in ("repository", "head_sha", "tested_sha"))
    assert task["budget"]["codex_attempts_used"] == 3 and task["budget"]["codex_attempts_limit"] == 10
    assert task["recovery"]["next_retry_at"] == health.iso(now+30)
    assert result["recent_tasks"][0]["result_status"] == "pass"
    assert "PRIVATE_SENTINEL" not in json.dumps(result)
    assert public_snapshot(result) == result


def test_public_health_keeps_unknown_unknown_and_projects_container_and_oneshot_evidence():
    result = public_snapshot({"collected_at": health.iso(100), "task_containers": [{
        "task_id": "a" * 64, "run_id": "run-1", "attempt_id": "attempt-1", "state": "running",
        "available": True, "running": False, "status": "exited", "exit_code": 137, "oom_killed": True,
        "finished_at": health.iso(99), "workspace_host": "PRIVATE_SENTINEL", "cpu_percent": 3.0,
    }], "services": [{"name": "health.service", "available": True, "active_state": "inactive",
                      "type": "oneshot", "sub_state": "dead", "result": "success"}]})
    assert result["runtime"]["available"] is None and result["poller"]["alive"] is None
    container = result["task_containers"][0]
    assert container["expected_running"] and container["oom_killed"] and container["exit_code"] == 137
    assert result["services"][0]["type"] == "oneshot" and result["services"][0]["result"] == "success"
    assert "PRIVATE_SENTINEL" not in json.dumps(result)
    assert public_snapshot(result) == result

def test_broken_local_task_record_does_not_claim_an_empty_healthy_queue(tmp_path):
    run = make_run(tmp_path, "broken", "publish_pending", None)
    (run / "state.json").write_text("{interrupted or unreadable")
    snapshot = health.collect({"state_dir": str(tmp_path), "monitor_services": []}, now=100,
                              manager=SimpleNamespace(health=lambda: {}))
    assert snapshot["tasks"] == [] and snapshot["uploads"] == []
    assert snapshot["tasks_available"] is False and snapshot["uploads_available"] is False
