"""Prepare a slim CI recipe; retain the previous private configuration for rollback."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from prepare.artifacts import tree_digest
from prepare.dependency_mounts import dependency_mounts
from prepare.runtime import shared_image, validate_shared_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--flaggems-source", required=True)
    parser.add_argument("--flaggems-commit", required=True)
    args = parser.parse_args()
    if os.getuid() == 0:
        raise ValueError("Run as the ordinary CI user")
    os.umask(0o077)
    path = Path(args.config).resolve()
    config = json.loads(path.read_text())
    profile = config["profiles"]["triton_v3.0"]
    source = Path(args.flaggems_source).resolve(strict=True)
    expected = args.flaggems_commit
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected or subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(source), "status", "--porcelain"]
    ):
        raise ValueError("FlagGems must be the configured clean pinned checkout")
    target = "/opt/local-ci/runtime/deps/flaggems"
    profile["mounts"] = [m for m in profile.get("mounts", []) if m["target"] != target]
    profile["mounts"].append(
        {
            "source": str(source),
            "target": target,
            "read_only": True,
            "sha256": tree_digest(source),
        }
    )
    config["image"] = args.image
    for item in config["profiles"].values():
        item.pop("image", None)
    profile["env"]["FLAGGEMS_CLONE_DIR"] = target
    profile["env"]["BACKEND_TEST_COMMAND"] = (
        '"$PYTHON_BIN" tests/test_smoke.py && "$PYTHON_BIN" tests/test_jit.py && '
        '"$PYTHON_BIN" -I /opt/local-ci/control/scripts/local_ci/prepare/profiles/slim/validate_flaggems.py'
    )
    # Dependencies must already be migrated; do not silently discard old recipes.
    for item in config["profiles"].values():
        validate_shared_profile(item)
    shared_image(config)
    dependency_mounts(config, profile, verify_content=True)
    backup = path.with_name(
        "local-ci-before-slim-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + ".json"
    )
    shutil.copyfile(path, backup)
    backup.chmod(0o600)
    temporary = path.with_name(".local-ci-slim.json")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)
    print("Private rollback configuration:", backup)
    print("Read-only FlagGems SHA:", actual)
    print("Slim foundation:", args.image)


if __name__ == "__main__":
    main()
