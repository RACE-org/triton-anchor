"""Retire precise obsolete user units during installation and control upgrades."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

OBSOLETE_UNITS = (
    "triton-anchor-local-ci-control-update.timer",
    "triton-anchor-local-ci-watchdog.timer",
    "triton-anchor-local-ci-watchdog.service",
)


def user_unit_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"


def retire_obsolete_units(destination=None, *, backup=None, reload=True, runner=None):
    """Stop only named retired units; retain their files before deleting them."""
    destination = Path(destination) if destination is not None else user_unit_dir()
    run = runner or subprocess.run
    if any((destination / name).is_symlink() for name in OBSOLETE_UNITS):
        raise ValueError("Refusing to retire symlinked systemd units")
    names = [name for name in OBSOLETE_UNITS if (destination / name).is_file()]
    if not names:
        return []
    if backup is not None:
        backup = Path(backup)
        backup.mkdir(parents=True, exist_ok=True)
        for name in names:
            # A retry must not overwrite the first pre-migration copy.
            if not (backup / name).exists():
                shutil.copy2(destination / name, backup / name)
    run(["systemctl", "--user", "disable", "--now", *names], check=True, timeout=60)
    for name in names:
        (destination / name).unlink()
    if reload:
        run(["systemctl", "--user", "daemon-reload"], check=True, timeout=60)
    return names
