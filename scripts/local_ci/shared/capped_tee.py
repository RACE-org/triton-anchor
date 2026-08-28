#!/usr/bin/env python3
"""Mirror stdin to stdout and a file without allowing unbounded log growth."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


LIMIT_EXIT_CODE = 86


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-bytes", required=True, type=int)
    parser.add_argument("--marker", default="")
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def write_marker(path: Path, output: Path, max_bytes: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "triton-anchor-local-ci-output-limit/v1",
        "failure_code": "log_size_limit",
        "detected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "path": str(output),
        "limit_bytes": max_bytes,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.max_bytes <= 0:
        print("--max-bytes must be positive", file=sys.stderr)
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_size = output.stat().st_size if args.append and output.exists() else 0
    remaining = max(args.max_bytes - existing_size, 0)
    mode = "ab" if args.append else "wb"
    marker = Path(args.marker) if args.marker else None
    reader = sys.stdin.buffer
    # BufferedReader.read(size) may wait for a pipe to produce all ``size``
    # bytes or close.  read1() performs at most one raw read, so short progress
    # messages are forwarded while the producer is still running.
    read_chunk = getattr(reader, "read1", reader.read)

    with output.open(mode) as stream:
        while True:
            chunk = read_chunk(min(1024 * 1024, remaining + 1))
            if not chunk:
                return 0
            if len(chunk) <= remaining:
                stream.write(chunk)
                stream.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                remaining -= len(chunk)
                continue

            if remaining:
                kept = chunk[:remaining]
                stream.write(kept)
                sys.stdout.buffer.write(kept)
            notice = (
                f"\n[local-ci] log limit reached: {args.max_bytes} bytes; "
                "terminating this stage.\n"
            ).encode("utf-8")
            stream.flush()
            sys.stdout.buffer.write(notice)
            sys.stdout.buffer.flush()
            if marker is not None:
                write_marker(marker, output, args.max_bytes)
            return LIMIT_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
