#!/usr/bin/env python3
"""Run pytest and record counts; empty or entirely skipped suites do not pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class Counts:
    def __init__(self):
        self.passed = self.failed = self.skipped = self.errors = 0

    def pytest_runtest_logreport(self, report):
        if report.failed:
            if report.when == "call":
                self.failed += 1
            else:
                self.errors += 1
        elif report.skipped:
            self.skipped += 1
        elif report.when == "call" and report.passed:
            self.passed += 1

    def pytest_collectreport(self, report):
        if report.failed:
            self.errors += 1


def execute(arguments: list[str], output: Path) -> int:
    import pytest

    counts = Counts()
    code = int(pytest.main(arguments, plugins=[counts]))
    if not code and (not counts.passed or counts.failed or counts.errors):
        code = 1
    result = {**vars(counts), "exit_code": code, "status": "pass" if code == 0 else "fail"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    return execute(arguments, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
