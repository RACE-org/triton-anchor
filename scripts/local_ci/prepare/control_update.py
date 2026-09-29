#!/usr/bin/env python3
"""Safely follow the trusted Gitee control branch along one commit history."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Callable

LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from prepare.artifacts import SHA_RE, atomic_json, safe_source, utc_now
from prepare.runtime import EnvironmentManager
from prepare.deployment_config import load_deployment_config, sync_deployment_config
from prepare.service_units import retire_obsolete_units


UPDATE_SCHEMA = "triton-anchor-local-ci-control-update"
REQUEST_SCHEMA = "triton-anchor-local-ci-control-update-request"
WORKER_SERVICE = "triton-anchor-local-ci.service"


def validate_control_source(value: object, *, allow_local: bool = False) -> str:
    source = safe_source(value, "control_repo_url")
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme == "https":
        if parsed.hostname != "gitee.com":
            raise ValueError("control_repo_url must be a credential-free Gitee HTTPS URL")
    elif not allow_local or parsed.scheme not in {"", "file"}:
        raise ValueError("Local control repositories are allowed only in tests")
    return source


def _environment(state_dir: Path) -> dict[str, str]:
    environment = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    if os.getenv("GITEE_TOKEN"):
        private = state_dir / "control-update"
        private.mkdir(parents=True, exist_ok=True)
        private.chmod(0o700)
        askpass = private / "askpass.py"
        askpass.write_text(
            "#!/usr/bin/env python3\n"
            "import os,sys\n"
            "print(os.environ.get('GITEE_USERNAME','oauth2') "
            "if 'Username' in sys.argv[1] else os.environ['GITEE_TOKEN'])\n"
        )
        askpass.chmod(0o700)
        environment["GIT_ASKPASS"] = str(askpass)
    return environment


def _git(
    root: Path,
    arguments: list[str],
    environment: dict[str, str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve()}",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *arguments,
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
    )
    if check and result.returncode:
        detail = result.stderr[-2000:]
        for name, value in environment.items():
            if any(word in name.upper() for word in ("TOKEN", "PASSWORD", "SECRET")):
                if len(value) > 3:
                    detail = detail.replace(value, "[redacted]")
        raise RuntimeError(
            f"Control git {arguments[0]} failed with exit {result.returncode}: "
            f"{detail.strip()}"
        )
    return result


def _restart_worker() -> None:
    subprocess.run(
        ["systemctl", "--user", "restart", WORKER_SERVICE],
        check=True,
        timeout=120,
    )


def _noop() -> None:
    pass


def update_request_path(config: dict) -> Path:
    return Path(config["state_dir"]) / "control-update/request.json"


def update_request_lock_path(config: dict) -> Path:
    return Path(config["state_dir"]) / "control-update/request.lock"


def read_update_request(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Control update request must be a regular file")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("Control update request must be private to the CI user")
    request = json.loads(path.read_text())
    revision = request.get("revision") if isinstance(request, dict) else None
    task_id = request.get("task_id") if isinstance(request, dict) else None
    if (
        not isinstance(request, dict)
        or request.get("schema") != REQUEST_SCHEMA
        or not isinstance(revision, str)
        or not SHA_RE.fullmatch(revision)
        or not isinstance(task_id, str)
        or not re.fullmatch(r"[0-9a-f]{64}", task_id)
    ):
        raise ValueError("Control update request is invalid")
    return request


def control_request_plan(
    config: dict,
    current_revision: str,
    requests: list[dict],
    *,
    allow_local: bool = False,
) -> dict:
    """Check the trusted branch tip when new tasks arrive.

    Task manifests identify only what source to validate.  Their dispatch
    revision is deliberately not consulted when selecting server control code.
    """
    if not isinstance(current_revision, str) or not SHA_RE.fullmatch(current_revision):
        raise ValueError("Current control revision must be an exact commit")
    if not requests:
        raise ValueError("At least one control update request is required")
    candidates = []
    for request in requests:
        task_id = request.get("task_id") if isinstance(request, dict) else None
        captured_at = request.get("captured_at") if isinstance(request, dict) else None
        if (
            not isinstance(task_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", task_id)
            or not isinstance(captured_at, str)
            or not captured_at
        ):
            raise ValueError("Control update candidate is invalid")
        candidates.append({"task_id": task_id, "captured_at": captured_at})
    root = Path(config["control_root"])
    state_dir = Path(config["state_dir"])
    source = validate_control_source(
        config.get("control_repo_url"), allow_local=allow_local
    )
    branch = config.get("control_branch", "CI_dev")
    if (
        not isinstance(branch, str)
        or not branch
        or re.search(r"[\x00-\x20~^:?*\\\[]", branch)
    ):
        raise ValueError("control_branch is not a safe Git branch name")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("control_root must be an existing dedicated directory")
    environment = _environment(state_dir)
    head = _git(root, ["rev-parse", "HEAD^{commit}"], environment).stdout.strip()
    if head != current_revision:
        raise ValueError("Worker control revision changed during request selection")
    _git(
        root,
        [
            "fetch",
            "--quiet",
            "--no-tags",
            "--force",
            source,
            f"refs/heads/{branch}:refs/local-ci/control-order",
        ],
        environment,
    )
    tip = _git(
        root, ["rev-parse", "refs/local-ci/control-order^{commit}"], environment
    ).stdout.strip()
    checked = tuple(row["task_id"] for row in candidates)
    if tip == current_revision:
        return {"checked_task_ids": checked, "request": None}
    forward = _git(
        root,
        ["merge-base", "--is-ancestor", current_revision, tip],
        environment,
        check=False,
    )
    rollback = _git(
        root,
        ["merge-base", "--is-ancestor", tip, current_revision],
        environment,
        check=False,
    )
    if forward.returncode and rollback.returncode:
        raise ValueError(
            "Configured control branch diverged from the installed trusted checkout"
        )
    trigger = min(candidates, key=lambda row: (row["captured_at"], row["task_id"]))
    return {
        "checked_task_ids": (),
        "request": {**trigger, "revision": tip},
    }

def update_control(
    config: dict,
    *,
    config_path: Path,
    apply: bool = False,
    allow_local: bool = False,
    expected_revision: str | None = None,
    restart_worker: Callable[[], None] = _restart_worker,
    on_success: Callable[[], None] = _noop,
) -> dict:
    """Replace checkout contents with a trusted related branch revision."""
    import fcntl

    root = Path(config["control_root"])
    state_dir = Path(config["state_dir"])
    source = validate_control_source(
        config.get("control_repo_url"), allow_local=allow_local
    )
    branch = config.get("control_branch", "CI_dev")
    if not isinstance(branch, str) or not branch or re.search(r"[\x00-\x20~^:?*\\\[]", branch):
        raise ValueError("control_branch is not a safe Git branch name")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("control_root must be an existing dedicated directory")
    if expected_revision is not None and (
        not isinstance(expected_revision, str)
        or not SHA_RE.fullmatch(expected_revision)
    ):
        raise ValueError("Requested control revision must be an exact commit")
    if apply and expected_revision is None:
        raise ValueError("Applied control update requires an exact requested revision")
    state_dir.mkdir(parents=True, exist_ok=True)
    environment = _environment(state_dir)
    if _git(root, ["rev-parse", "--is-inside-work-tree"], environment).stdout.strip() != "true":
        raise ValueError("control_root must be a Git checkout")
    current = _git(root, ["rev-parse", "HEAD^{commit}"], environment).stdout.strip()
    if not SHA_RE.fullmatch(current):
        raise ValueError("control_root HEAD is not an exact commit")
    remote = _git(
        root, ["ls-remote", "--exit-code", source, f"refs/heads/{branch}"], environment
    ).stdout.split()
    if len(remote) < 2 or not SHA_RE.fullmatch(remote[0]):
        raise ValueError("Configured control branch did not resolve to one commit")
    remote_revision = remote[0]
    target = expected_revision or remote_revision
    result = {
        "schema": UPDATE_SCHEMA,
        "checked_at": utc_now(),
        "branch": branch,
        "previous_revision": current,
        "revision": target,
        "remote_revision": remote_revision,
        "changed": False,
        "restarted": False,
        "applied": apply,
    }
    lock_path = state_dir / "control.lock"
    marker_path = state_dir / "control-update.json"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result["state"] = "deferred-active-task"
            return result

        # Recheck mutable local state after acquiring the deployment lock.
        current = _git(root, ["rev-parse", "HEAD^{commit}"], environment).stdout.strip()
        result["previous_revision"] = current
        if target != current or expected_revision is not None:
            _git(
                root,
                [
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    "--force",
                    source,
                    f"refs/heads/{branch}:refs/local-ci/control-update",
                ],
                environment,
            )
            fetched = _git(
                root, ["rev-parse", "refs/local-ci/control-update^{commit}"], environment
            ).stdout.strip()
            if fetched != remote_revision:
                raise RuntimeError("Control branch moved during fetch; retry the update")
            if expected_revision is not None:
                present = _git(
                    root,
                    ["cat-file", "-e", f"{target}^{{commit}}"],
                    environment,
                    check=False,
                )
                on_branch = _git(
                    root,
                    ["merge-base", "--is-ancestor", target, fetched],
                    environment,
                    check=False,
                )
                if present.returncode or on_branch.returncode:
                    raise ValueError(
                        "Requested control revision is not reachable from the configured branch"
                    )
            ancestor = _git(
                root, ["merge-base", "--is-ancestor", current, target], environment, check=False
            )
            if ancestor.returncode:
                rollback = _git(
                    root,
                    ["merge-base", "--is-ancestor", target, current],
                    environment,
                    check=False,
                )
                if rollback.returncode:
                    raise ValueError(
                        "Configured control branch diverged from the installed checkout"
                    )
                if target != fetched:
                    raise ValueError(
                        "Control rollback is allowed only to the configured branch tip"
                    )

        desired = load_deployment_config(root, target)
        # These paths/runtime settings are also embedded in installed systemd units.
        if any(desired.get(key) != config.get(key)
               for key in ("control_root", "state_dir", "python_bin", "runtime")):
            raise ValueError("Deployment paths or runtime changed; apply them with install.py")
        config_plan = sync_deployment_config(desired, config_path)
        result.update(config_changed=config_plan["changed"],
                      config_digest=config_plan["digest"],
                      config_fields=config_plan["fields"])
        if not apply:
            return result

        try:
            marker = json.loads(marker_path.read_text())
        except (FileNotFoundError, OSError, ValueError):
            marker = {}
        restart_needed = (
            target != current or config_plan["changed"]
            or marker.get("revision") != target
            or marker.get("config_digest") != config_plan["digest"]
        )
        if restart_needed:
            # A crashed Worker can release its lock before Docker stops its task.
            # Configuration-only changes must respect the same task boundary.
            manager = EnvironmentManager(config, state_dir)
            containers = manager._docker(
                "ps", "-aq", "--filter", "label=local-ci.owner=" + manager.owner,
                "--format", '{{.Label "local-ci.kind"}}', timeout=30,
            ).decode().splitlines()
            if {"task", "task-cleanup"}.intersection(containers):
                result["state"] = "deferred-active-task"
                return result
            # A failed restart must be retried even after repairing same-SHA drift.
            marker_path.unlink(missing_ok=True)
        if target != current:
            # This dedicated deployment checkout follows the trusted revision;
            # local edits and obstructing untracked files may be overwritten.
            _git(root, ["checkout", "--quiet", "--force", "--detach", target], environment)
            if _git(root, ["rev-parse", "HEAD^{commit}"], environment).stdout.strip() != target:
                raise RuntimeError("Control checkout did not reach the fetched commit")
            result["changed"] = True
        sync_deployment_config(desired, config_path, apply=True)
        result["retired_units"] = retire_obsolete_units(
            backup=state_dir / "deploy-backups/obsolete-units"
        )
        if restart_needed:
            restart_worker()
            result["restarted"] = True
            atomic_json(marker_path, {**result, "restarted_at": utc_now()})
        on_success()
        result["state"] = "updated" if result["changed"] or result["config_changed"] else "current"
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    revision = parser.add_mutually_exclusive_group()
    revision.add_argument("--expected-revision")
    revision.add_argument("--request-file")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.apply and os.geteuid() == 0:
            raise ValueError("Control update must run as the ordinary CI user")
        config = json.loads(Path(args.config).read_text())
        request_path = None
        request = None
        if args.request_file:
            request_path = Path(os.path.abspath(args.request_file))
            configured_request = Path(os.path.abspath(update_request_path(config)))
            if request_path != configured_request:
                raise ValueError("Control update request must use the configured state path")
            request = read_update_request(request_path)

        def clear_completed_request() -> None:
            if not request_path or not request:
                return
            import fcntl

            request_lock = update_request_lock_path(config)
            request_lock.parent.mkdir(parents=True, exist_ok=True)
            with request_lock.open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if request_path.exists():
                    current = read_update_request(request_path)
                    if all(
                        current.get(field) == request.get(field)
                        for field in ("task_id", "revision")
                    ):
                        request_path.unlink()

        result = update_control(
            config,
            config_path=Path(args.config),
            apply=args.apply,
            expected_revision=request["revision"] if request else args.expected_revision,
            on_success=clear_completed_request if request else _noop,
        )
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Local CI control update failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
