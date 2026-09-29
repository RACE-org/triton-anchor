"""Use the committed deployment configuration to maintain one runtime copy."""

import json
import os
from pathlib import Path
import subprocess

from prepare.artifacts import atomic_json, fingerprint


CONFIG_SOURCE = "scripts/local_ci/prepare/config.example.json"


def load_deployment_config(root: Path, revision: str = "HEAD") -> dict:
    raw = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{revision}:{CONFIG_SOURCE}"], text=True
    )
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema") != "triton-anchor-local-ci-config":
        raise ValueError("Unsupported deployment configuration")
    if Path(config.get("control_root", "")).resolve() != root.resolve():
        raise ValueError("Deployment control_root must match this control checkout")
    return config


def validate_deployment_config(config: dict) -> None:
    # preflight imports control_update; keep this import out of module startup.
    from prepare.preflight import check_configuration

    ready = check_configuration(
        config, runtime=False, require_notifications=False, verify_content=False
    )
    if not ready["ready"]:
        failures = [row["check"] for row in ready["checks"] if row["status"] != "pass"]
        raise ValueError("Invalid deployment configuration: " + ", ".join(failures))


def sync_deployment_config(config: dict, destination: Path, *, apply=False) -> dict:
    """Compare parsed JSON; publish atomically only when its content differs."""
    destination = Path(os.path.abspath(destination))
    if destination.resolve().is_relative_to(Path(config["control_root"]).resolve()):
        raise ValueError("Runtime configuration must be outside the control checkout")
    if destination.is_symlink() or (
        destination.exists()
        and (not destination.is_file() or destination.stat().st_uid != os.getuid())
    ):
        raise ValueError("Runtime configuration must be a regular file owned by the CI user")
    current = json.loads(destination.read_text()) if destination.exists() else None
    changed = current != config
    plan = {
        "path": str(destination),
        "changed": changed,
        "digest": fingerprint(config),
        "fields": sorted(key for key in set(current or {}) | set(config)
                         if (current or {}).get(key) != config.get(key)),
    }
    if changed:
        validate_deployment_config(config)
    if apply:
        if changed:
            atomic_json(destination, config)  # mkstemp keeps CI ownership and mode 0600.
        elif destination.stat().st_mode & 0o777 != 0o600:
            destination.chmod(0o600)
    return plan
