#!/usr/bin/env python3
"""Refresh one profile's dependencies and probe the shared runtime image."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare.runtime import EnvironmentManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text())
        branches = [
            branch
            for branch, profile in config["profiles"].items()
            if profile.get("name", branch.replace("/", "-")) == args.profile
        ]
        if len(branches) != 1:
            raise ValueError("Profile name must identify one configured target branch")
        manager = EnvironmentManager(config, config["state_dir"])
        result = manager.rotate(branches[0])
        result["collection"] = manager.collect_retired()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Local CI environment preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
