from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MAINTENANCE = ROOT / "maintenance/manage_local_ci_state.py"
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


def load_maintenance_module():
    spec = importlib.util.spec_from_file_location("local_ci_maintenance", MAINTENANCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MAINTENANCE_MODULE = load_maintenance_module()


def set_age(path: Path, days: int) -> None:
    timestamp = (NOW - timedelta(days=days)).timestamp()
    os.utime(path, (timestamp, timestamp))


def make_run(state: Path, run_id: str, status: int | None, days: int) -> Path:
    run = state / "runs" / "ci_push_CI_dev" / run_id
    run.mkdir(parents=True)
    (run / "local-ci.log").write_text("log\n", encoding="utf-8")
    if status is None:
        set_age(run, days)
    else:
        result = run / "result.json"
        result.write_text(json.dumps({"status": status}), encoding="utf-8")
        set_age(result, days)
    runner = state / "runner" / run_id
    runner.mkdir(parents=True)
    (runner / "poller.sh").write_text("runner\n", encoding="utf-8")
    set_age(runner, days)
    return run


def make_artifact(root: Path, name: str, status: int | None, days: int) -> Path:
    artifact = root / name
    artifact.mkdir(parents=True)
    (artifact / "payload.bin").write_bytes(b"payload")
    if status is None:
        set_age(artifact, days)
    else:
        summary = artifact / "delivery-summary.txt"
        summary.write_text(f"status: {status}\n", encoding="utf-8")
        set_age(summary, days)
    return artifact


def run_maintenance(state: Path, artifacts: Path, report: Path, apply: bool):
    command = [
        sys.executable,
        str(MAINTENANCE),
        "--state-dir",
        str(state),
        "--artifact-root",
        str(artifacts),
        "--success-days",
        "14",
        "--failure-days",
        "28",
        "--incomplete-days",
        "7",
        "--docker-orphan-grace-hours",
        "72",
        "--report",
        str(report),
        "--now",
        NOW.isoformat(),
        "--skip-docker",
    ]
    if apply:
        command.append("--apply")
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_retention_dry_run_and_apply(tmp_path: Path) -> None:
    state = tmp_path / "state"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    expired_success = make_run(state, "success-old", 0, 15)
    expired_success_boundary = make_run(state, "success-boundary", 0, 14)
    retained_success = make_run(state, "success-new", 0, 13)
    expired_failure = make_run(state, "failure-old", 1, 29)
    expired_failure_boundary = make_run(state, "failure-boundary", 1, 28)
    retained_failure = make_run(state, "failure-new", 1, 27)
    expired_incomplete = make_run(state, "incomplete-old", None, 8)
    expired_artifact = make_artifact(artifacts, "artifact-old", 0, 15)
    retained_artifact = make_artifact(artifacts, "artifact-new", 1, 27)
    report = state / "maintenance" / "latest.json"

    dry_run = run_maintenance(state, artifacts, report, apply=False)
    assert dry_run.returncode == 0, dry_run.stderr
    assert expired_success.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["removed_count"] == 0

    applied = run_maintenance(state, artifacts, report, apply=True)
    assert applied.returncode == 0, applied.stderr
    assert not expired_success.exists()
    assert not expired_success_boundary.exists()
    assert not expired_failure.exists()
    assert not expired_failure_boundary.exists()
    assert not expired_incomplete.exists()
    assert not expired_artifact.exists()
    assert not (state / "runner" / "success-old").exists()
    assert not (state / "runner" / "failure-old").exists()
    assert retained_success.exists()
    assert retained_failure.exists()
    assert retained_artifact.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["policy"]["success_days"] == 14
    assert payload["policy"]["failure_days"] == 28
    assert payload["removed_count"] == 11


def test_docker_cleanup_skips_running_containers_and_referenced_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = (NOW - timedelta(hours=73)).isoformat()

    def fake_run(command, **kwargs):
        del kwargs
        key = tuple(command)
        outputs = {
            (
                "docker",
                "ps",
                "-a",
                "--filter",
                "label=triton-anchor.role=codex-ai",
                "-q",
            ): "running\nstopped\n",
            (
                "docker",
                "container",
                "inspect",
                "running",
            ): json.dumps([{"Created": old, "State": {"Running": True}}]),
            (
                "docker",
                "container",
                "inspect",
                "stopped",
            ): json.dumps([{"Created": old, "State": {"Running": False}}]),
            (
                "docker",
                "images",
                "--filter",
                "label=triton-anchor.role=codex-ai-snapshot",
                "-q",
            ): "used-image\nfree-image\n",
            (
                "docker",
                "ps",
                "-a",
                "--filter",
                "ancestor=used-image",
                "-q",
            ): "stopped\n",
            (
                "docker",
                "ps",
                "-a",
                "--filter",
                "ancestor=free-image",
                "-q",
            ): "",
            (
                "docker",
                "image",
                "inspect",
                "free-image",
            ): json.dumps([{"Created": old}]),
        }
        return SimpleNamespace(stdout=outputs[key])

    monkeypatch.setattr(MAINTENANCE_MODULE.subprocess, "run", fake_run)

    candidates = MAINTENANCE_MODULE.collect_docker_candidates(NOW, 72)

    assert [(item.category, item.path) for item in candidates] == [
        ("container", "stopped"),
        ("image", "free-image"),
    ]
