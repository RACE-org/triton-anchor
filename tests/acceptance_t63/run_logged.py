"""Run one acceptance command while preserving an exact timestamped transcript."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    started = dt.datetime.now(dt.timezone.utc)
    header = (
        f"started_utc: {started.isoformat()}\n"
        f"cwd: {arguments.cwd.resolve()}\n"
        f"command: {shlex.join(command)}\n"
        "--- output ---\n"
    )
    arguments.log.parent.mkdir(parents=True, exist_ok=True)
    with arguments.log.open("w", encoding="utf-8") as stream:
        stream.write(header)
        stream.flush()
        sys.stdout.write(header)
        sys.stdout.flush()
        process = subprocess.Popen(
            command,
            cwd=arguments.cwd,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
        finished = dt.datetime.now(dt.timezone.utc)
        footer = (
            "--- result ---\n"
            f"exit_code: {return_code}\n"
            f"finished_utc: {finished.isoformat()}\n"
            f"duration_seconds: {(finished - started).total_seconds():.3f}\n"
        )
        stream.write(footer)
        sys.stdout.write(footer)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
