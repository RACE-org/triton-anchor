"""The server control checkout advances only through safe Gitee-style updates."""

import fcntl
import json
import subprocess

import pytest

from prepare import control_update, deployment_config
from prepare.control_update import (
    REQUEST_SCHEMA,
    control_request_plan,
    update_control,
)
from prepare.runtime import EnvironmentManager


@pytest.fixture(autouse=True)
def isolated_server(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user-config"))
    validate = deployment_config.validate_deployment_config
    monkeypatch.setattr(EnvironmentManager, "_docker", lambda *args, **kwargs: b"")
    # These Git/installation tests run without the CI server's mounted dependencies.
    monkeypatch.setattr(deployment_config, "validate_deployment_config", lambda config: None)
    return validate


def git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def commit(root, message, content, *, config=None):
    (root / "control.txt").write_text(content)
    paths = ["control.txt"]
    if config is not None:
        source = root / deployment_config.CONFIG_SOURCE
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(config))
        paths.append(deployment_config.CONFIG_SOURCE)
    subprocess.run(["git", "add", *paths], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=root,
        check=True,
    )
    return git(root, "rev-parse", "HEAD")


def fixture_checkout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    control = tmp_path / "control"
    config = {
        "schema": "triton-anchor-local-ci-config",
        "control_root": str(control),
        "state_dir": str(tmp_path / "state"),
        "control_repo_url": str(source),
        "control_branch": "CI_dev",
        "runtime": {
            "kind": "docker-rootless",
            "endpoint": "unix:///run/user/1001/docker.sock",
            "context": "rootless",
        },
        "poll_interval_seconds": 60,
    }
    subprocess.run(["git", "init", "-q", "-b", "CI_dev"], cwd=source, check=True)
    old = commit(source, "old", "old", config=config)
    subprocess.run(["git", "clone", "-q", str(source), str(control)], check=True)
    new = commit(source, "new", "new", config={**config, "poll_interval_seconds": 30})
    (tmp_path / "local-ci.json").write_text(json.dumps(config))
    (tmp_path / "local-ci.json").chmod(0o600)
    return config, control, old, new


def test_control_checkout_fast_forwards_and_restarts_once(tmp_path, monkeypatch):
    config, control, old, new = fixture_checkout(tmp_path)
    marker = tmp_path / "state/control-update.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"revision": new}))
    restarts = []
    def docker(self, *arguments, **kwargs):
        assert arguments == ("ps", "-aq", "--filter", "label=local-ci.owner=" + self.owner,
                             "--format", '{{.Label "local-ci.kind"}}')
        return b"task\n"
    monkeypatch.setattr(EnvironmentManager, "_docker", docker)
    deferred = update_control(
        config,
        config_path=tmp_path / "local-ci.json", apply=True, allow_local=True, restart_worker=lambda: restarts.append("worker"),
        expected_revision=new,
    )
    assert deferred["state"] == "deferred-active-task"
    assert not deferred["changed"] and not deferred["restarted"]
    assert git(control, "rev-parse", "HEAD") == old
    assert not restarts
    # Environment probes no longer mount the control checkout and need not block it.
    monkeypatch.setattr(EnvironmentManager, "_docker", lambda *args, **kwargs: b"image-validation\n")
    result = update_control(
        config,
        config_path=tmp_path / "local-ci.json",
        apply=True,
        allow_local=True,
        expected_revision=new,
        restart_worker=lambda: restarts.append("worker"),
    )
    assert result["previous_revision"] == old
    assert result["revision"] == new
    assert result["changed"] and result["restarted"]
    assert result["config_changed"]
    assert git(control, "rev-parse", "HEAD") == new
    assert restarts == ["worker"]
    config_path = tmp_path / "local-ci.json"
    assert json.loads(config_path.read_text())["poll_interval_seconds"] == 30
    assert config_path.stat().st_mode & 0o777 == 0o600
    copied = config_path.stat()
    monkeypatch.setattr(
        EnvironmentManager, "_docker",
        lambda *args, **kwargs: pytest.fail("An unchanged revision needs no Docker check"),
    )
    again = update_control(
        config,
        config_path=tmp_path / "local-ci.json",
        apply=True,
        allow_local=True,
        expected_revision=new,
        restart_worker=lambda: restarts.append("worker"),
    )
    assert not again["changed"] and not again["restarted"]
    assert not again["config_changed"]
    assert config_path.stat().st_ino == copied.st_ino
    assert config_path.stat().st_mtime_ns == copied.st_mtime_ns
    assert restarts == ["worker"]


@pytest.mark.parametrize("rollback", [False, True])
def test_control_checkout_overwrites_local_changes(tmp_path, rollback):
    config, control, old, new = fixture_checkout(tmp_path)
    target = old if rollback else new
    if rollback:
        update_control(
            config, config_path=tmp_path / "local-ci.json", apply=True,
            allow_local=True, expected_revision=new, restart_worker=lambda: None,
        )
        git(tmp_path / "source", "update-ref", "refs/heads/CI_dev", old, new)
    current = git(control, "rev-parse", "HEAD")
    (control / "control.txt").write_text("staged local changes")
    git(control, "add", "control.txt")
    (control / "control.txt").write_text("unstaged local changes")
    (control / deployment_config.CONFIG_SOURCE).unlink()
    (control / "local-note.txt").write_text("unrelated untracked file")
    restarts = []
    preview = update_control(
        config, config_path=tmp_path / "local-ci.json", allow_local=True,
        expected_revision=target, restart_worker=lambda: restarts.append("worker"),
    )
    assert not preview["changed"] and not restarts
    assert git(control, "rev-parse", "HEAD") == current
    assert (control / "control.txt").read_text() == "unstaged local changes"
    result = update_control(
        config, config_path=tmp_path / "local-ci.json", apply=True,
        allow_local=True, expected_revision=target,
        restart_worker=lambda: restarts.append("worker"),
    )
    assert result["changed"] and result["restarted"]
    assert git(control, "rev-parse", "HEAD") == target
    assert (control / "control.txt").read_text() == ("old" if rollback else "new")
    assert not git(control, "diff", "HEAD", "--")
    assert (control / deployment_config.CONFIG_SOURCE).is_file()
    assert restarts == ["worker"]


def test_applied_control_update_requires_an_exact_requested_revision(tmp_path):
    config, _, _, _ = fixture_checkout(tmp_path)
    with pytest.raises(ValueError, match="exact requested revision"):
        update_control(
            config,
            config_path=tmp_path / "local-ci.json", apply=True, allow_local=True, restart_worker=lambda: None
        )


def control_candidates():
    return [
        {
            "task_id": "a" * 64,
            "captured_at": "2026-09-11T00:00:00Z",
            # Deliberately unrelated: task provenance must not select control.
            "revision": "9" * 40,
        },
        {
            "task_id": "b" * 64,
            "captured_at": "2026-09-12T00:00:00Z",
            "revision": "8" * 40,
        },
    ]


def test_new_task_requests_latest_control_branch_tip_not_task_revision(tmp_path):
    config, _, current, tip = fixture_checkout(tmp_path)
    requests = control_candidates()
    plan = control_request_plan(config, current, requests, allow_local=True)
    assert plan == {
        "checked_task_ids": (),
        "request": {
            "task_id": requests[0]["task_id"],
            "captured_at": requests[0]["captured_at"],
            "revision": tip,
        },
    }


def test_new_task_uses_current_control_or_requests_trusted_rollback(tmp_path):
    config, control, _, tip = fixture_checkout(tmp_path)
    update_control(
        config,
        config_path=tmp_path / "local-ci.json",
        apply=True,
        allow_local=True,
        expected_revision=tip,
        restart_worker=lambda: None,
    )
    requests = control_candidates()
    expected = {
        "checked_task_ids": tuple(row["task_id"] for row in requests),
        "request": None,
    }
    assert control_request_plan(config, tip, requests, allow_local=True) == expected

    ahead = commit(control, "installed ahead", "installed ahead")
    rollback = control_request_plan(config, ahead, requests, allow_local=True)
    assert rollback["checked_task_ids"] == ()
    assert rollback["request"]["revision"] == tip


def test_control_checkout_follows_multiple_trusted_rollbacks(tmp_path):
    config, control, old, first = fixture_checkout(tmp_path)
    source = tmp_path / "source"
    second = commit(
        source,
        "second forward",
        "second forward",
        config={**config, "poll_interval_seconds": 20},
    )
    third = commit(
        source,
        "third forward",
        "third forward",
        config={**config, "poll_interval_seconds": 10},
    )
    assert len({old, first, second, third}) == 4
    config_path = tmp_path / "local-ci.json"
    restarts = []
    update_control(
        config,
        config_path=config_path,
        apply=True,
        allow_local=True,
        expected_revision=third,
        restart_worker=lambda: restarts.append("forward"),
    )
    assert git(control, "rev-parse", "HEAD") == third

    # The trusted branch withdraws three commits before the next task arrives.
    subprocess.run(
        ["git", "update-ref", "refs/heads/CI_dev", old, third],
        cwd=source,
        check=True,
    )
    plan = control_request_plan(
        config, third, control_candidates(), allow_local=True
    )
    assert plan["request"]["revision"] == old

    result = update_control(
        config,
        config_path=config_path,
        apply=True,
        allow_local=True,
        expected_revision=old,
        restart_worker=lambda: restarts.append("rollback"),
    )
    assert result["previous_revision"] == third
    assert result["remote_revision"] == result["revision"] == old
    assert result["changed"] and result["restarted"]
    assert git(control, "rev-parse", "HEAD") == old
    assert json.loads(config_path.read_text())["poll_interval_seconds"] == 60
    assert restarts == ["forward", "rollback"]


def test_new_task_blocks_when_control_branch_diverged(tmp_path):
    config, control, current, _ = fixture_checkout(tmp_path)
    subprocess.run(["git", "switch", "-q", "-c", "installed-side", current], cwd=control, check=True)
    side = commit(control, "installed side", "installed side")
    with pytest.raises(ValueError, match="diverged"):
        control_request_plan(config, side, control_candidates(), allow_local=True)


def test_explicit_requested_revision_can_precede_remote_branch_tip(tmp_path):
    config, control, old, requested = fixture_checkout(tmp_path)
    remote_tip = commit(
        tmp_path / "source", "newer", "newer",
        config={**config, "poll_interval_seconds": 10},
    )
    config_path = tmp_path / "local-ci.json"
    before = config_path.read_bytes()
    preview = update_control(
        config,
        config_path=config_path,
        allow_local=True,
        expected_revision=requested,
        restart_worker=lambda: pytest.fail("Preview must not restart the Worker"),
    )
    assert preview["config_changed"]
    assert preview["config_fields"] == ["poll_interval_seconds"]
    assert config_path.read_bytes() == before
    assert git(control, "rev-parse", "HEAD") == old
    restarts = []

    def restart():
        assert git(control, "rev-parse", "HEAD") == requested
        assert json.loads(config_path.read_text())["poll_interval_seconds"] == 30
        restarts.append("worker")

    result = update_control(
        config,
        config_path=tmp_path / "local-ci.json",
        apply=True,
        allow_local=True,
        expected_revision=requested,
        restart_worker=restart,
    )
    assert result["previous_revision"] == old
    assert result["revision"] == requested
    assert result["remote_revision"] == remote_tip
    assert git(control, "rev-parse", "HEAD") == requested
    assert restarts == ["worker"]


def test_invalid_target_configuration_preserves_checkout_and_copy(
    tmp_path, monkeypatch, isolated_server
):
    config, control, old, _ = fixture_checkout(tmp_path)
    invalid = commit(
        tmp_path / "source", "invalid configuration", "invalid",
        config={**config, "resources": {"cpus": -1}},
    )
    monkeypatch.setattr(
        deployment_config, "validate_deployment_config", isolated_server
    )
    config_path = tmp_path / "local-ci.json"
    before = config_path.read_bytes()
    with pytest.raises(ValueError, match="Invalid deployment configuration"):
        update_control(
            config,
            config_path=config_path,
            apply=True,
            allow_local=True,
            expected_revision=invalid,
            restart_worker=lambda: pytest.fail("Invalid configuration must not restart"),
        )
    assert git(control, "rev-parse", "HEAD") == old
    assert config_path.read_bytes() == before


@pytest.mark.parametrize("restart_fails", [False, True])
def test_same_revision_repairs_drift_and_retries_failed_restart(
    tmp_path, monkeypatch, restart_fails
):
    config, control, _, requested = fixture_checkout(tmp_path)
    config_path = tmp_path / "local-ci.json"
    update_control(
        config, config_path=config_path, apply=True, allow_local=True,
        expected_revision=requested, restart_worker=lambda: None,
    )
    desired = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({**desired, "poll_interval_seconds": 99}))
    before = config_path.read_bytes()
    restarts = []

    def restart():
        assert json.loads(config_path.read_text()) == desired
        restarts.append("worker")
        if restart_fails and len(restarts) == 1:
            raise RuntimeError("Restart failed")

    options = dict(
        config_path=config_path, apply=True, allow_local=True,
        expected_revision=requested, restart_worker=restart,
    )
    monkeypatch.setattr(EnvironmentManager, "_docker", lambda *args, **kwargs: b"task-cleanup\n")
    deferred = update_control(config, **options)
    assert deferred["state"] == "deferred-active-task"
    assert not deferred["restarted"] and not deferred["changed"]
    assert config_path.read_bytes() == before
    assert not restarts
    monkeypatch.setattr(EnvironmentManager, "_docker", lambda *args, **kwargs: b"")
    if restart_fails:
        with pytest.raises(RuntimeError, match="Restart failed"):
            update_control(config, **options)
        assert json.loads(config_path.read_text()) == desired
    result = update_control(config, **options)
    assert not result["changed"] and result["restarted"]
    assert result["config_changed"] is not restart_fails
    assert restarts == ["worker"] * (2 if restart_fails else 1)
    assert git(control, "rev-parse", "HEAD") == requested
    # JSON formatting/key order do not constitute drift or require a rewrite.
    config_path.write_text(json.dumps(dict(reversed(list(desired.items())))))
    before = config_path.read_bytes()
    monkeypatch.setattr(
        EnvironmentManager, "_docker",
        lambda *args, **kwargs: pytest.fail("Current config needs no Docker check"),
    )
    again = update_control(config, **options)
    assert not again["changed"] and not again["config_changed"] and not again["restarted"]
    assert config_path.read_bytes() == before
    assert restarts == ["worker"] * (2 if restart_fails else 1)


def test_requested_revision_must_be_on_branch_and_rollback_must_be_current_tip(tmp_path):
    config, control, old, current = fixture_checkout(tmp_path)
    update_control(
        config,
        config_path=tmp_path / "local-ci.json",
        apply=True,
        allow_local=True,
        expected_revision=current,
        restart_worker=lambda: None,
    )
    source = tmp_path / "source"
    subprocess.run(["git", "switch", "-q", "-c", "side", old], cwd=source, check=True)
    side = commit(source, "side", "side")
    subprocess.run(["git", "switch", "-q", "CI_dev"], cwd=source, check=True)

    with pytest.raises(ValueError, match="not reachable"):
        update_control(
            config,
            config_path=tmp_path / "local-ci.json",
            apply=True,
            allow_local=True,
            expected_revision=side,
            restart_worker=lambda: None,
        )
    with pytest.raises(ValueError, match="branch tip"):
        update_control(
            config,
            config_path=tmp_path / "local-ci.json",
            apply=True,
            allow_local=True,
            expected_revision=old,
            restart_worker=lambda: None,
        )
    assert git(control, "rev-parse", "HEAD") == current


def test_active_task_defers_requested_update_without_losing_revision(tmp_path):
    config, control, old, requested = fixture_checkout(tmp_path)
    config_path = tmp_path / "local-ci.json"
    before = config_path.read_bytes()
    state = tmp_path / "state"
    state.mkdir()
    with (state / "control.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        result = update_control(
            config,
            config_path=tmp_path / "local-ci.json",
            apply=True,
            allow_local=True,
            expected_revision=requested,
            restart_worker=lambda: None,
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
    assert result["state"] == "deferred-active-task"
    assert result["revision"] == requested
    assert git(control, "rev-parse", "HEAD") == old
    assert config_path.read_bytes() == before


@pytest.mark.parametrize("completed", [False, True])
def test_request_file_is_consumed_only_after_exact_revision_update(
    tmp_path, monkeypatch, completed
):
    request_path = tmp_path / "state/control-update/request.json"
    request_path.parent.mkdir(parents=True)
    request = {
        "schema": REQUEST_SCHEMA,
        "revision": "a" * 40,
        "task_id": "b" * 64,
        "requested_at": 1,
    }
    request_path.write_text(json.dumps(request))
    request_path.chmod(0o600)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"state_dir": str(tmp_path / "state")})
    )
    monkeypatch.setattr(control_update.os, "geteuid", lambda: 1001)

    def update(config, **kwargs):
        assert kwargs["apply"] and kwargs["expected_revision"] == request["revision"]
        assert kwargs["config_path"] == config_path
        if completed:
            request_path.write_text(json.dumps({**request, "requested_at": 2}))
            request_path.chmod(0o600)
            kwargs["on_success"]()
        return {"state": "updated" if completed else "deferred-active-task",
                "revision": kwargs["expected_revision"]}

    monkeypatch.setattr(control_update, "update_control", update)
    assert control_update.main([
        "--config", str(config_path), "--request-file", str(request_path), "--apply",
    ]) == 0
    assert request_path.exists() is not completed


def test_request_file_rejects_symlink_at_configured_path(tmp_path, monkeypatch):
    state = tmp_path / "state"
    request_path = state / "control-update/request.json"
    request_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(
            {
                "schema": REQUEST_SCHEMA,
                "revision": "a" * 40,
                "task_id": "b" * 64,
                "requested_at": 1,
            }
        )
    )
    outside.chmod(0o600)
    request_path.symlink_to(outside)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"state_dir": str(state)}))
    monkeypatch.setattr(
        control_update,
        "update_control",
        lambda config, **kwargs: pytest.fail("symlink request must not be applied"),
    )
    assert (
        control_update.main(
            ["--config", str(config_path), "--request-file", str(request_path)]
        )
        == 1
    )


def test_control_update_retires_old_watchdog_before_its_single_restart(tmp_path, monkeypatch):
    from prepare import service_units
    config, control, old, new = fixture_checkout(tmp_path)
    units = service_units.user_unit_dir()
    units.mkdir(parents=True)
    for name in service_units.OBSOLETE_UNITS:
        (units / name).write_text("original unit " + name)
    keep = units / "triton-anchor-local-ci-health.timer"
    keep.write_text("keep health")
    actions = []
    original = service_units.retire_obsolete_units
    monkeypatch.setattr(control_update, "retire_obsolete_units", lambda **kwargs:
        original(**kwargs, runner=lambda args, **options: actions.append(args)))
    options = dict(config_path=tmp_path / "local-ci.json", apply=True, allow_local=True,
                   expected_revision=new, restart_worker=lambda: actions.append("restart-worker"))
    result = update_control(config, **options)
    assert result["retired_units"] == list(service_units.OBSOLETE_UNITS)
    assert actions[-1] == "restart-worker" and actions.count("restart-worker") == 1
    assert actions[-2] == ["systemctl", "--user", "daemon-reload"]
    assert keep.read_text() == "keep health"
    backup = tmp_path / "state/deploy-backups/obsolete-units"
    assert all((backup / name).read_text() == "original unit " + name for name in service_units.OBSOLETE_UNITS)
    again = update_control(config, **options)
    assert not again["retired_units"] and actions.count("restart-worker") == 1
