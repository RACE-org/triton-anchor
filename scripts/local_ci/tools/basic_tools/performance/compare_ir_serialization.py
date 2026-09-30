#!/usr/bin/env python3
"""Compare IR serialization metrics for a candidate and cached base SHA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from common import DEFAULT_KERNELS, load_json, write_projected_csv  # noqa: E402


DEFAULT_METRICS = (
    "serialize",
    "write_text",
    "read_text",
    "deserialize",
    "roundtrip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS))
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--min-base-ms", type=float, default=0.05)
    parser.add_argument("--min-delta-ms", type=float, default=0.05)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def metric_median(document: dict[str, Any], kernel: str, metric: str) -> float:
    try:
        value = document["summary"][kernel]["metrics"][metric]["median_ms"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing {metric} median for kernel {kernel!r}") from exc
    if not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {metric} median for kernel {kernel!r}: {value!r}")
    return float(value)


def threshold_exceeded(
    baseline_ms: float,
    candidate_ms: float,
    threshold: float,
    min_base_ms: float,
    min_delta_ms: float,
) -> tuple[bool, float, float | None]:
    delta_ms = candidate_ms - baseline_ms
    if baseline_ms <= 0:
        return False, delta_ms, None
    change_ratio = delta_ms / baseline_ms
    if baseline_ms < min_base_ms or delta_ms < min_delta_ms:
        return False, delta_ms, change_ratio
    return change_ratio > threshold, delta_ms, change_ratio


def compare(
    baseline: Optional[dict[str, Any]],
    candidate: dict[str, Any],
    kernels: list[str],
    metrics: list[str],
    threshold: float,
    min_base_ms: float,
    min_delta_ms: float,
    base_sha: str,
    candidate_sha: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    if baseline is None:
        warnings.append(
            "No cached IR serialization baseline is available for base SHA "
            f"{base_sha or '<unknown>'}."
        )

    for kernel in kernels:
        for metric in metrics:
            candidate_ms = metric_median(candidate, kernel, metric)
            baseline_ms: float | None = None
            delta_ms: float | None = None
            change_ratio: float | None = None
            exceeds = False
            if baseline is not None:
                try:
                    baseline_ms = metric_median(baseline, kernel, metric)
                except ValueError as exc:
                    warnings.append(str(exc))
                else:
                    exceeds, delta_ms, change_ratio = threshold_exceeded(
                        baseline_ms,
                        candidate_ms,
                        threshold,
                        min_base_ms,
                        min_delta_ms,
                    )
                    if exceeds:
                        warnings.append(
                            f"{kernel}/{metric} slowed by {change_ratio:+.1%} "
                            f"({baseline_ms:.3f} ms -> {candidate_ms:.3f} ms), "
                            f"exceeding the {threshold:.0%} threshold."
                        )
            rows.append(
                {
                    "kernel": kernel,
                    "metric": metric,
                    "baseline_median_ms": baseline_ms,
                    "candidate_median_ms": candidate_ms,
                    "delta_ms": delta_ms,
                    "change_ratio": change_ratio,
                    "change_percent": (
                        change_ratio * 100.0 if change_ratio is not None else None
                    ),
                    "status": "warning" if exceeds else "pass",
                    "exceeds_threshold": exceeds,
                }
            )

    return {
        "status": "warning" if warnings else "pass",
        "base_sha": base_sha,
        "candidate_sha": candidate_sha,
        "threshold_ratio": threshold,
        "min_base_ms": min_base_ms,
        "min_delta_ms": min_delta_ms,
        "mode": "slowdown",
        "baseline_available": baseline is not None,
        "kernels": kernels,
        "metrics": metrics,
        "rows": rows,
        "warnings": warnings,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kernel",
        "metric",
        "baseline_median_ms",
        "candidate_median_ms",
        "delta_ms",
        "change_percent",
        "status",
        "exceeds_threshold",
    ]
    write_projected_csv(path, fieldnames, rows)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# IR serialization regression",
        "",
        f"Status: **{result['status']}**",
        f"Base SHA: `{result['base_sha'] or 'unavailable'}`",
        f"Candidate SHA: `{result['candidate_sha']}`",
        f"Slowdown threshold: `{result['threshold_ratio']:.0%}`",
        f"Minimum base time: `{result['min_base_ms']:.3f} ms`",
        f"Minimum absolute increase: `{result['min_delta_ms']:.3f} ms`",
        "",
        "| Kernel | Metric | Base median (ms) | Candidate median (ms) | Change | Result |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in result["rows"]:
        baseline = (
            "n/a"
            if row["baseline_median_ms"] is None
            else f"{row['baseline_median_ms']:.3f}"
        )
        change = "n/a" if row["change_ratio"] is None else f"{row['change_ratio']:+.1%}"
        lines.append(
            f"| {row['kernel']} | `{row['metric']}` | {baseline} | "
            f"{row['candidate_median_ms']:.3f} | {change} | {row['status']} |"
        )
    if result["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {message}" for message in result["warnings"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.threshold < 0:
        raise ValueError("--threshold must be non-negative")
    if args.min_base_ms < 0 or args.min_delta_ms < 0:
        raise ValueError("--min-base-ms and --min-delta-ms must be non-negative")
    kernels = [value.strip() for value in args.kernels.split(",") if value.strip()]
    metrics = [value.strip() for value in args.metrics.split(",") if value.strip()]
    if not kernels or not metrics:
        raise ValueError("--kernels and --metrics must not be empty")

    candidate = load_json(Path(args.candidate_json))
    baseline_path = Path(args.baseline_json) if args.baseline_json else None
    baseline = (
        load_json(baseline_path) if baseline_path and baseline_path.is_file() else None
    )
    result = compare(
        baseline,
        candidate,
        kernels,
        metrics,
        args.threshold,
        args.min_base_ms,
        args.min_delta_ms,
        args.base_sha,
        args.candidate_sha,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(Path(args.output_csv), result["rows"])
    write_markdown(Path(args.output_markdown), result)
    print(f"IR serialization comparison status: {result['status']}")
    for message in result["warnings"]:
        print(f"WARNING: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
