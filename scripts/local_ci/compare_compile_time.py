#!/usr/bin/env python3
"""Compare compile-time medians for a candidate commit against a base commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_KERNELS = ("add", "mm", "softmax", "layernorm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS))
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def compile_median(document: dict[str, Any], kernel: str) -> float:
    try:
        value = document["summary"][kernel]["compile_est"]["median_ms"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing compile_est median for kernel {kernel!r}") from exc
    if not isinstance(value, (int, float)):
        raise ValueError(f"Invalid compile_est median for kernel {kernel!r}: {value!r}")
    return float(value)


def compare(
    baseline: Optional[dict[str, Any]],
    candidate: dict[str, Any],
    kernels: list[str],
    threshold: float,
    base_sha: str,
    candidate_sha: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    if baseline is None:
        warnings.append(f"No cached compile-time baseline is available for base SHA {base_sha or '<unknown>'}.")

    for kernel in kernels:
        candidate_ms = compile_median(candidate, kernel)
        base_ms: float | None = None
        change_ratio: float | None = None
        exceeds_threshold = False

        if baseline is not None:
            try:
                base_ms = compile_median(baseline, kernel)
            except ValueError as exc:
                warnings.append(str(exc))
            else:
                if base_ms <= 0:
                    warnings.append(f"Baseline median for {kernel} is not positive: {base_ms:.3f} ms.")
                else:
                    change_ratio = (candidate_ms - base_ms) / base_ms
                    exceeds_threshold = abs(change_ratio) > threshold
                    if exceeds_threshold:
                        warnings.append(
                            f"{kernel} compile time changed by {change_ratio:+.1%} "
                            f"({base_ms:.3f} ms -> {candidate_ms:.3f} ms), "
                            f"exceeding the +/-{threshold:.0%} threshold."
                        )

        rows.append(
            {
                "kernel": kernel,
                "baseline_median_ms": base_ms,
                "candidate_median_ms": candidate_ms,
                "change_ratio": change_ratio,
                "change_percent": change_ratio * 100.0 if change_ratio is not None else None,
                "exceeds_threshold": exceeds_threshold,
            }
        )

    return {
        "schema": "triton-anchor-compile-time-comparison/v1",
        "status": "warning" if warnings else "pass",
        "base_sha": base_sha,
        "candidate_sha": candidate_sha,
        "threshold_ratio": threshold,
        "baseline_available": baseline is not None,
        "kernels": rows,
        "warnings": warnings,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Compile-time regression",
        "",
        f"Status: **{result['status']}**",
        f"Base SHA: `{result['base_sha'] or 'unavailable'}`",
        f"Candidate SHA: `{result['candidate_sha']}`",
        f"Threshold: `+/-{result['threshold_ratio']:.0%}`",
        "",
        "| Kernel | Base median (ms) | Candidate median (ms) | Change | Result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in result["kernels"]:
        base = "n/a" if row["baseline_median_ms"] is None else f"{row['baseline_median_ms']:.3f}"
        change = "n/a" if row["change_ratio"] is None else f"{row['change_ratio']:+.1%}"
        state = "warning" if row["exceeds_threshold"] else "pass"
        lines.append(
            f"| {row['kernel']} | {base} | {row['candidate_median_ms']:.3f} | {change} | {state} |"
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
    kernels = [value.strip() for value in args.kernels.split(",") if value.strip()]
    if not kernels:
        raise ValueError("--kernels must contain at least one kernel")

    candidate = load_json(Path(args.candidate_json))
    baseline_path = Path(args.baseline_json) if args.baseline_json else None
    baseline = load_json(baseline_path) if baseline_path and baseline_path.is_file() else None
    result = compare(
        baseline,
        candidate,
        kernels,
        args.threshold,
        args.base_sha,
        args.candidate_sha,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_markdown), result)

    print(f"Compile-time comparison status: {result['status']}")
    for message in result["warnings"]:
        print(f"WARNING: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
