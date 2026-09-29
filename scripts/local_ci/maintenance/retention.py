#!/usr/bin/env python3
"""Expire published evidence after 30 days; preserve pending delivery and summaries."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare.artifacts import atomic_json
from agent_ci.state import run_state_paths


def _timestamp(value):
    if type(value) in (int, float):
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return 0


def _size(root):
    return sum(
        p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink()
    )


def _remove_tree(path, root):
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(
            "Evidence retention target must stay within the configured run"
        )
    if path.exists():
        shutil.rmtree(path)


def retain_local(config, *, now=None, apply=True):
    now = time.time() if now is None else now
    days = config.get("results_retention_days", 30)
    budget = config.get(
        "evidence_max_bytes", config.get("task_workspace_max_bytes", 100 * 1024**3)
    )
    if type(days) is not int or days <= 0 or type(budget) is not int or budget <= 0:
        raise ValueError(
            "Positive retention days and evidence byte budget are required"
        )
    state = Path(config["state_dir"]).resolve()
    runs = state / "runs"
    report = {
        "retention_days": days,
        "max_bytes": budget,
        "expired": [],
        "protected": [],
        "errors": [],
        "applied": apply,
    }
    for path in run_state_paths(state):
        run = path.parent
        if any(
            p.is_symlink()
            for p in (path, *path.parents)
            if p.is_relative_to(runs)
        ):
            report["errors"].append({"run": run.name, "reason": "symlink"})
            continue
        record = {}
        try:
            record = json.loads(path.read_text())
            delivery = record.get("delivery") or {}
            published = _timestamp(
                delivery.get("published", record.get("published_at"))
            )
            hold_until = _timestamp(record.get("retention_until"))
            if (
                record.get("phase") != "published"
                or not published
                or hold_until > now
            ):
                report["protected"].append(
                    {"task_id": record.get("task_id", run.parent.name), "run_id": run.name}
                )
                continue
            if now - published < days * 86400:
                continue
            if (run / "retention.json").exists() and not (run / "inputs").exists():
                continue
            marker = {
                "schema": "triton-anchor-result-retention",
                "task_id": record.get("task_id", run.parent.name),
                "run_id": run.name,
                "expired_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "reason": "retention_expired",
                "published_at": published,
            }
            if apply:
                for name in ("artifacts", "logs", "inputs"):
                    _remove_tree(run / name, run)
                # Git keeps the published summary and selected files. Retain
                # local result metadata while expiring the larger payloads.
                for name in ("artifacts", "logs", "evidence"):
                    _remove_tree(run / "sealed" / name, run)
                atomic_json(run / "retention.json", marker)
            report["expired"].append(marker)
        except (OSError, ValueError, RuntimeError) as exc:
            report["errors"].append(
                {
                    "task_id": record.get("task_id", run.parent.name),
                    "run_id": run.name,
                    "reason": type(exc).__name__,
                }
            )
    used = _size(runs) if runs.exists() else 0
    free = shutil.disk_usage(state).free if state.exists() else 0
    report.update(
        logical_bytes=used,
        state_free_bytes=free,
        pause_intake=used > budget
        or free < config.get("state_min_free_bytes", 5 * 1024**3),
    )
    if apply:
        atomic_json(state / "health/retention.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = retain_local(config, apply=args.apply)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
