#!/usr/bin/env python3
"""Compare per-pass profiling medians for candidate and base commits."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS))
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--min-base-ms", type=float, default=1.0)
    parser.add_argument("--min-delta-ms", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--mode", choices=("slowdown", "symmetric"), default="slowdown")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def pass_map(document: dict[str, Any], kernel: str) -> dict[str, Any]:
    try:
        value = document["summary"][kernel]["passes"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing pass profile summary for kernel {kernel!r}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Invalid pass profile summary for kernel {kernel!r}")
    return value


def pass_median_ms(document: dict[str, Any], kernel: str, pass_name: str) -> float:
    try:
        value = pass_map(document, kernel)[pass_name]["wall_ms"]["median_ms"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing median for {kernel}/{pass_name}") from exc
    if not isinstance(value, (int, float)):
        raise ValueError(f"Invalid median for {kernel}/{pass_name}: {value!r}")
    return float(value)


def invocation_median(
    document: dict[str, Any], kernel: str, pass_name: str
) -> float | None:
    try:
        value = pass_map(document, kernel)[pass_name]["invocations"]["median_ms"]
    except (KeyError, TypeError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def threshold_exceeded(
    baseline_ms: float,
    candidate_ms: float,
    threshold: float,
    min_base_ms: float,
    min_delta_ms: float,
    mode: str,
) -> tuple[bool, float, float]:
    delta_ms = candidate_ms - baseline_ms
    if baseline_ms <= 0:
        return False, delta_ms, 0.0
    change_ratio = delta_ms / baseline_ms
    if baseline_ms < min_base_ms or abs(delta_ms) < min_delta_ms:
        return False, delta_ms, change_ratio
    if mode == "symmetric":
        return abs(change_ratio) > threshold, delta_ms, change_ratio
    return change_ratio > threshold, delta_ms, change_ratio


def compare(
    baseline: Optional[dict[str, Any]],
    candidate: dict[str, Any],
    kernels: list[str],
    threshold: float,
    min_base_ms: float,
    min_delta_ms: float,
    top_n: int,
    mode: str,
    base_sha: str,
    candidate_sha: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    hotspots: list[dict[str, Any]] = []

    if baseline is None:
        warnings.append(
            f"No cached pass-profile baseline is available for base SHA {base_sha or '<unknown>'}."
        )

    for kernel in kernels:
        candidate_passes = pass_map(candidate, kernel)
        if not candidate_passes:
            raise ValueError(f"No candidate pass measurements for kernel {kernel!r}")
        baseline_passes: dict[str, Any] = {}
        if baseline is not None:
            try:
                baseline_passes = pass_map(baseline, kernel)
            except ValueError as exc:
                warnings.append(str(exc))

        pass_names = sorted(set(candidate_passes) | set(baseline_passes))
        for pass_name in pass_names:
            candidate_ms: float | None = None
            baseline_ms: float | None = None
            delta_ms: float | None = None
            change_ratio: float | None = None
            exceeds = False
            row_status = "pass"

            if pass_name in candidate_passes:
                candidate_ms = pass_median_ms(candidate, kernel, pass_name)
            if baseline is not None and pass_name in baseline_passes:
                baseline_ms = pass_median_ms(baseline, kernel, pass_name)

            if candidate_ms is None:
                row_status = "removed"
            elif baseline_ms is None:
                row_status = "new"
            else:
                exceeds, delta_ms, change_ratio = threshold_exceeded(
                    baseline_ms,
                    candidate_ms,
                    threshold,
                    min_base_ms,
                    min_delta_ms,
                    mode,
                )
                row_status = "warning" if exceeds else "pass"
                if exceeds:
                    warnings.append(
                        f"{kernel}/{pass_name} pass time changed by {change_ratio:+.1%} "
                        f"({baseline_ms:.3f} ms -> {candidate_ms:.3f} ms), "
                        f"exceeding the {mode} threshold {threshold:.0%}."
                    )

            row = {
                "kernel": kernel,
                "pass": pass_name,
                "baseline_median_ms": baseline_ms,
                "candidate_median_ms": candidate_ms,
                "delta_ms": delta_ms,
                "change_ratio": change_ratio,
                "change_percent": change_ratio * 100.0
                if change_ratio is not None
                else None,
                "baseline_invocation_median": invocation_median(
                    baseline, kernel, pass_name
                )
                if baseline
                else None,
                "candidate_invocation_median": invocation_median(
                    candidate, kernel, pass_name
                )
                if candidate_ms is not None
                else None,
                "status": row_status,
                "exceeds_threshold": exceeds,
            }
            rows.append(row)
            if candidate_ms is not None:
                hotspots.append(row)

    hotspots.sort(key=lambda item: item["candidate_median_ms"] or 0.0, reverse=True)
    status = "warning" if warnings else "pass"
    return {
        "status": status,
        "base_sha": base_sha,
        "candidate_sha": candidate_sha,
        "threshold_ratio": threshold,
        "min_base_ms": min_base_ms,
        "min_delta_ms": min_delta_ms,
        "mode": mode,
        "baseline_available": baseline is not None,
        "kernels": kernels,
        "passes": rows,
        "hotspots": hotspots[:top_n],
        "warnings": warnings,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kernel",
        "pass",
        "baseline_median_ms",
        "candidate_median_ms",
        "delta_ms",
        "change_percent",
        "baseline_invocation_median",
        "candidate_invocation_median",
        "status",
        "exceeds_threshold",
    ]
    write_projected_csv(path, fieldnames, rows)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Pass-profile regression",
        "",
        f"Status: **{result['status']}**",
        f"Base SHA: `{result['base_sha'] or 'unavailable'}`",
        f"Candidate SHA: `{result['candidate_sha']}`",
        f"Threshold: `{result['mode']} {result['threshold_ratio']:.0%}`",
        f"Minimum base time: `{result['min_base_ms']:.3f} ms`",
        f"Minimum delta: `{result['min_delta_ms']:.3f} ms`",
        "",
        "## Candidate hotspots",
        "",
        "| Rank | Kernel | Pass | Candidate median (ms) | Change | Result |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(result["hotspots"], start=1):
        change = "n/a" if row["change_ratio"] is None else f"{row['change_ratio']:+.1%}"
        candidate = row["candidate_median_ms"]
        lines.append(
            f"| {rank} | {row['kernel']} | `{row['pass']}` | {candidate:.3f} | "
            f"{change} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Pass comparison",
            "",
            "| Kernel | Pass | Base median (ms) | Candidate median (ms) | Change | Result |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["passes"]:
        base = (
            "n/a"
            if row["baseline_median_ms"] is None
            else f"{row['baseline_median_ms']:.3f}"
        )
        candidate = (
            "n/a"
            if row["candidate_median_ms"] is None
            else f"{row['candidate_median_ms']:.3f}"
        )
        change = "n/a" if row["change_ratio"] is None else f"{row['change_ratio']:+.1%}"
        lines.append(
            f"| {row['kernel']} | `{row['pass']}` | {base} | {candidate} | {change} | {row['status']} |"
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
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    kernels = [value.strip() for value in args.kernels.split(",") if value.strip()]
    if not kernels:
        raise ValueError("--kernels must contain at least one kernel")

    candidate = load_json(Path(args.candidate_json))
    baseline_path = Path(args.baseline_json) if args.baseline_json else None
    baseline = (
        load_json(baseline_path) if baseline_path and baseline_path.is_file() else None
    )
    result = compare(
        baseline,
        candidate,
        kernels,
        args.threshold,
        args.min_base_ms,
        args.min_delta_ms,
        args.top_n,
        args.mode,
        args.base_sha,
        args.candidate_sha,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(Path(args.output_csv), result["passes"])
    write_markdown(Path(args.output_markdown), result)

    print(f"Pass-profile comparison status: {result['status']}")
    for message in result["warnings"]:
        print(f"WARNING: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
