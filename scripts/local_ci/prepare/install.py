#!/usr/bin/env python3
"""Prepare the configured environments and start Local CI user services.

Without --apply, render the service plan. Installation keeps a backup of prior
units; --rollback restores that backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare.preflight import check_configuration
from prepare.runtime import EnvironmentManager
from prepare.runtime_probe import probe_runtime
from prepare.deployment_config import load_deployment_config, sync_deployment_config
from prepare.service_units import OBSOLETE_UNITS, retire_obsolete_units


def quoted(value: str) -> str:
    if any(character in value for character in "\n\r\x00"):
        raise ValueError("Systemd paths must not contain control characters")
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    )


def unit_path(value: str) -> str:
    if (
        not Path(value).is_absolute()
        or value != value.strip()
        or any(ord(char) < 32 or ord(char) == 127 or char in "\\*?[]" for char in value)
    ):
        raise ValueError(
            "Systemd path must be absolute, without control characters, escapes or globs"
        )
    return value.replace("%", "%%")


def render_units(
    config: dict, config_path: Path, credentials_path: Path
) -> dict[str, str]:
    python = quoted(config.get("python_bin", "/usr/bin/python3"))
    root = Path(config["control_root"]) / "scripts/local_ci"
    runtime = config.get("runtime", {})
    docker_service = runtime.get("service", "docker.service")
    if not isinstance(docker_service, str) or not __import__("re").fullmatch(
        r"[A-Za-z0-9_.@-]+\.service", docker_service
    ):
        raise ValueError("Configure the actual Rootless Docker user service")
    # These settings take path tokens, not ExecStart-style quoted arguments.
    common = (
        f"EnvironmentFile={unit_path(str(credentials_path))}\nWorkingDirectory={unit_path(config['control_root'])}\nUMask=0077\nNoNewPrivileges=yes\n"
        f"Environment={quoted('DOCKER_HOST=' + runtime.get('endpoint', ''))}\nUnsetEnvironment=DOCKER_CONTEXT DOCKER_TLS_VERIFY DOCKER_CERT_PATH\n"
    )
    worker = f"{python} {quoted(str(root / 'agent_ci/worker.py'))} --config {quoted(str(config_path))}"
    request_file = Path(config["state_dir"]) / "control-update/request.json"
    control_update = f"{python} {quoted(str(root / 'prepare/control_update.py'))} --config {quoted(str(config_path))} --request-file {quoted(str(request_file))} --apply"
    health = f"{python} {quoted(str(root / 'maintenance/health.py'))} --config {quoted(str(config_path))} --publish"
    retention = f"{python} {quoted(str(root / 'maintenance/retention.py'))} --config {quoted(str(config_path))} --apply"
    units = {
        "triton-anchor-local-ci.service": f"[Unit]\nDescription=Triton Anchor task-container Local CI worker\nAfter={docker_service}\nWants={docker_service}\n\n[Service]\nType=simple\n"
        + common
        + f"ExecStart={worker}\nRestart=always\nRestartSec=15\nTimeoutStopSec=60\nKillMode=mixed\n\n[Install]\nWantedBy=default.target\n",
        "triton-anchor-local-ci-health.service": "[Unit]\nDescription=Publish independent Local CI worker health\n\n[Service]\nType=oneshot\n"
        + common
        + f"ExecStart={health}\nTimeoutStartSec=10min\n",
        "triton-anchor-local-ci-health.timer": "[Unit]\nDescription=Refresh Local CI health independently of poller\n\n[Timer]\nOnBootSec=1min\nOnUnitActiveSec=5min\nRandomizedDelaySec=15\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n",
        "triton-anchor-local-ci-control-update.service": "[Unit]\nDescription=Update Local CI control checkout for a waiting task\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=oneshot\n"
        + common
        + f"ExecStart={control_update}\nTimeoutStartSec=10min\n",
    }
    units["triton-anchor-local-ci-retention.service"] = (
        "[Unit]\nDescription=Expire Local CI result evidence by upload age\n\n[Service]\nType=oneshot\n"
        + common
        + f"ExecStart={retention}\nTimeoutStartSec=1h\n"
    )
    units["triton-anchor-local-ci-retention.timer"] = (
        "[Unit]\nDescription=Daily Local CI result retention\n\n[Timer]\nOnBootSec=30min\nOnUnitActiveSec=1d\nPersistent=true\nRandomizedDelaySec=5min\n\n[Install]\nWantedBy=timers.target\n"
    )
    return units


def install_units(
    units: dict[str, str], destination: Path, backup: Path, *, obsolete=OBSOLETE_UNITS
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "triton-anchor-local-ci-service-install",
        "scope": "user",
        "uid": os.getuid(),
        "destination": str(destination.resolve()),
        "units": {},
    }
    if any((destination / name).is_symlink() for name in (*units, *obsolete)):
        raise ValueError("Refusing to replace symlinked systemd units")
    for name, content in units.items():
        target = destination / name
        existed = target.is_file()
        if existed:
            shutil.copy2(target, backup / name)
        manifest["units"][name] = {
            "existed": existed,
            "installed_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    for name in obsolete:
        target = destination / name
        existed = target.is_file()
        if existed:
            shutil.copy2(target, backup / name)
        manifest["units"][name] = {"existed": existed, "removed": True}
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, content in units.items():
        target = destination / name
        temporary = destination / ("." + name + ".install")
        temporary.write_text(content)
        temporary.chmod(0o644)
        os.replace(temporary, target)
    return manifest


def rollback_units(backup: Path, *, apply: bool = False) -> dict:
    manifest = json.loads((backup / "manifest.json").read_text())
    if manifest.get("scope") != "user" or not isinstance(manifest.get("units"), dict):
        raise ValueError("Invalid service backup manifest")
    if manifest.get("uid") != os.getuid():
        raise ValueError("User unit backup belongs to another account")
    destination = Path(manifest["destination"])
    for name, entry in manifest["units"].items():
        if not name.startswith("triton-anchor-local-ci") or Path(name).name != name:
            raise ValueError("Unsafe service backup entry")
        target = destination / name
        if entry.get("removed"):
            if target.exists():
                raise ValueError(
                    "An obsolete unit was recreated after installation; preserve it for manual review"
                )
            continue
        if target.is_symlink() or (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest()
            != entry["installed_sha256"]
        ):
            raise ValueError(
                "Installed unit has changed since installation; preserve it for manual review"
            )
    if apply:
        for name, entry in manifest["units"].items():
            target = destination / name
            if entry["existed"]:
                shutil.copy2(backup / name, target)
            else:
                target.unlink(missing_ok=True)
    return manifest


def load_environment(path: Path) -> None:
    """Read the private KEY=value file used by systemd without running a shell."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("Credentials environment must use KEY=value lines")
        words = shlex.split(value, comments=False)
        if len(words) > 1:
            raise ValueError("Quote environment values containing whitespace")
        os.environ[key] = words[0] if words else ""


def prepare_environments(config: dict) -> None:
    ready = check_configuration(config, runtime=False)
    if not ready["ready"]:
        raise ValueError(json.dumps(ready, ensure_ascii=False))
    manager = EnvironmentManager(config, config["state_dir"])
    manager.collect_retired()
    for branch in config["profiles"]:
        manager.rotate(branch)
    probe_runtime(config)
    ready = check_configuration(config)
    if not ready["ready"]:
        raise ValueError(json.dumps(ready, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--credentials-env")
    parser.add_argument(
        "--unit-dir",
        default=str(
            Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
            / "systemd/user"
        ),
    )
    parser.add_argument("--backup-dir")
    parser.add_argument("--render-dir")
    parser.add_argument("--rollback")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    control_lock = None
    try:
        if args.apply and os.geteuid() == 0:
            raise ValueError(
                "Install and rollback must run as the ordinary CI user, without sudo"
            )
        if args.rollback:
            manifest = rollback_units(Path(args.rollback), apply=False)
            if args.apply:
                user_units = (
                    Path(
                        os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
                    )
                    / "systemd/user"
                )
                if Path(manifest["destination"]).resolve() != user_units.resolve():
                    raise ValueError(
                        "Rollback is restricted to the current CI user's systemd/user directory"
                    )
                rollback_units(Path(args.rollback), apply=True)
                subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            print(json.dumps({"rollback": manifest, "applied": args.apply}, indent=2))
            return 0
        if not args.config or not args.credentials_env:
            parser.error("--config and --credentials-env are required")
        config_path, credentials = (
            Path(os.path.abspath(args.config)),
            Path(args.credentials_env).resolve(),
        )
        config = load_deployment_config(Path(__file__).resolve().parents[3])
        config_plan = sync_deployment_config(config, config_path)
        units = render_units(config, config_path, credentials)
        if args.render_dir:
            output = Path(args.render_dir)
            output.mkdir(parents=True, exist_ok=True)
            for name, content in units.items():
                (output / name).write_text(content)
        if args.apply:
            if (
                not credentials.is_file()
                or credentials.is_symlink()
                or credentials.stat().st_mode & 0o077
                or credentials.stat().st_uid != os.getuid()
            ):
                raise ValueError(
                    "Credentials EnvironmentFile must exist and be private (mode 600)"
                )
            user_units = (
                Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
                / "systemd/user"
            )
            if Path(args.unit_dir).resolve() != user_units.resolve():
                raise ValueError(
                    "Installation is restricted to the current CI user's systemd/user directory"
                )
            import fcntl

            previous = json.loads(config_path.read_text()) if config_path.exists() else config
            state_dir = Path(previous["state_dir"])
            state_dir.mkdir(parents=True, exist_ok=True)
            control_lock = (state_dir / "control.lock").open("a")
            try:
                fcntl.flock(control_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("An active task is using the deployment; retry installation when idle") from None
            load_environment(credentials)
            prepare_environments(config)
            backup = (
                Path(args.backup_dir)
                if args.backup_dir
                else Path(config["state_dir"])
                / "deploy-backups"
                / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
            manifest = install_units(units, Path(args.unit_dir), backup)
            retire_obsolete_units(Path(args.unit_dir), reload=False)
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            services = ["triton-anchor-local-ci.service"] + [
                name for name in units if name.endswith(".timer")
            ]
            subprocess.run(["systemctl", "--user", "enable", *services], check=True)
            sync_deployment_config(config, config_path, apply=True)
            subprocess.run(["systemctl", "--user", "restart", *services], check=True)
            print(
                json.dumps(
                    {
                        "installed": manifest,
                        "backup": str(backup),
                        "services_started": services,
                        "configuration": config_plan,
                    },
                    indent=2,
                )
            )
        else:
            print(json.dumps({"planned_units": units, "configuration": config_plan, "applied": False}, indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Local CI service installation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if control_lock is not None:
            control_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
