#!/usr/bin/env python3
"""Capture MLIR pass timing output for the local CI benchmark kernels.

This script intentionally runs separately from compile_benchmark.py.  Enabling
MLIR timing adds diagnostic I/O and should not contaminate the end-to-end
compile-time baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from common import (  # noqa: E402
    DEFAULT_KERNELS,
    DEFAULT_SHAPES,
    neighboring_compile_benchmark,
    summarize,
    write_projected_csv,
)

_TIMING_ROW_RE = re.compile(
    r"^\s*(?P<wall>[0-9]+(?:\.[0-9]+)?)\s*"
    r"(?P<unit>ns|us|µs|ms|s|sec|secs|seconds)?\s+"
    r"\(\s*(?P<percent>[0-9]+(?:\.[0-9]+)?)\s*%\)"
    r"(?P<name_field>.*\S)\s*$"
)
_UNIT_TO_MS = {
    "ns": 1.0e-6,
    "us": 1.0e-3,
    "µs": 1.0e-3,
    "ms": 1.0,
    "s": 1000.0,
    "sec": 1000.0,
    "secs": 1000.0,
    "seconds": 1000.0,
    None: 1000.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark per-pass MLIR timing.")
    parser.add_argument("--backend", default="sophgo")
    parser.add_argument("--vendor", default=None)
    parser.add_argument(
        "--flaggems-root",
        default=os.environ.get("FLAGGEMS_ROOT", "/workspace/FlagGems"),
    )
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    temporary_root = Path(os.environ.get("TMPDIR", "/tmp"))
    parser.add_argument(
        "--cache-root",
        default=str(temporary_root / "triton_anchor_pass_profile/cache"),
    )
    parser.add_argument(
        "--dump-root",
        default=str(temporary_root / "triton_anchor_pass_profile/dump"),
    )
    parser.add_argument("--output-json", default="pass_profile_results.json")
    parser.add_argument("--output-events-csv", default="pass_profile_events.csv")
    parser.add_argument("--output-summary-csv", default="pass_profile_summary.csv")
    parser.add_argument(
        "--output-hotspots-markdown", default="pass_profile_hotspots.md"
    )
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    parser.add_argument("--atol", type=float, default=1.0e-2)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--verbose-worker", action="store_true")
    return parser.parse_args()


def to_ms(value: str, unit: str | None) -> float:
    key = unit.lower() if isinstance(unit, str) else None
    if key not in _UNIT_TO_MS:
        key = None
    return float(value) * _UNIT_TO_MS[key]


def normalize_timing_name(raw_name: str) -> tuple[str, str, str]:
    display = raw_name.strip()
    kind = "pipeline" if re.search(r"\bPipeline$", display) else "pass"
    name = re.sub(r"\s+(Pass|Pipeline)$", "", display).strip()
    if len(name) >= 2 and name[0] in "'\"" and name[-1] == name[0]:
        name = name[1:-1]
    name = re.sub(r"\s+", " ", name).strip()
    return name or display, display, kind


def parse_timing_output(
    text: str,
    kernel: str,
    phase: str,
    run_id: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _TIMING_ROW_RE.match(line)
        if not match:
            continue

        name_field = match.group("name_field")
        leading_spaces = len(name_field) - len(name_field.lstrip())
        depth = max(0, leading_spaces // 2)
        name, display_name, kind = normalize_timing_name(name_field)
        events.append(
            {
                "kernel": kernel,
                "phase": phase,
                "run_id": run_id,
                "sequence": len(events),
                "depth": depth,
                "kind": kind,
                "name": name,
                "display_name": display_name,
                "wall_ms": to_ms(match.group("wall"), match.group("unit")),
                "percent": float(match.group("percent")),
                "raw_line": line,
            }
        )
    return events


def run_child(
    args: argparse.Namespace,
    kernel: str,
    phase: str,
    run_index: int,
    work_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache_dir = work_root / "cache" / kernel / f"{phase}_{run_index}"
    dump_dir = work_root / "dump" / kernel / f"{phase}_{run_index}"
    result_file = work_root / "results" / kernel / f"{phase}_{run_index}.json"
    log_file = work_root / "logs" / kernel / f"{phase}_{run_index}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(neighboring_compile_benchmark(Path(__file__))),
        "--worker",
        "--backend",
        args.backend,
        "--vendor",
        args.vendor or args.backend,
        "--flaggems-root",
        args.flaggems_root,
        "--worker-kernel",
        kernel,
        "--worker-phase",
        phase,
        "--worker-run-id",
        str(run_index),
        "--worker-output",
        str(result_file),
        "--worker-cache-dir",
        str(cache_dir),
        "--worker-dump-dir",
        str(dump_dir),
        "--worker-seed",
        str(20260625 + run_index),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
    ]
    if args.skip_correctness:
        cmd.append("--skip-correctness")

    env = os.environ.copy()
    flaggems_src = str(Path(args.flaggems_root) / "src")
    env["PYTHONPATH"] = flaggems_src + os.pathsep + env.get("PYTHONPATH", "")
    env["FLAGGEMS_ROOT"] = args.flaggems_root
    env["GEMS_VENDOR"] = args.vendor or args.backend
    env["TRITON_ANCHOR_PROFILE"] = "1"
    env["MLIR_ENABLE_TIMING"] = "1"

    print(f"[{kernel}] {phase} pass-profile run {run_index}: starting")
    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout or ""
    log_file.write_text(output, encoding="utf-8", errors="replace")
    if args.verbose_worker and output:
        print(output, end="" if output.endswith("\n") else "\n")

    if not result_file.exists():
        if output:
            print(output)
        raise RuntimeError(f"Worker did not write result file: {result_file}")

    result = json.loads(result_file.read_text(encoding="utf-8"))
    if completed.returncode != 0 or result.get("status") != "pass":
        if output:
            print(output)
        raise RuntimeError(
            f"Worker failed for {kernel} {phase} {run_index}: "
            f"{result.get('error', result.get('status'))}"
        )

    events = parse_timing_output(output, kernel, phase, str(run_index))
    for event in events:
        event["worker_log"] = str(log_file)
        event["compile_est_ms"] = result.get("compile_est_ms")
    print(
        f"[{kernel}] {phase} pass-profile run {run_index}: "
        f"events={len(events)}, compile_est={result['compile_est_ms']:.3f} ms, "
        f"ok={result['correctness_ok']}"
    )
    return result, events


def pass_totals_for_run(
    events: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int]]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for event in events:
        if event.get("kind") != "pass":
            continue
        name = str(event["name"])
        totals[name] += float(event["wall_ms"])
        counts[name] += 1
    return dict(totals), dict(counts)


def build_summary(
    kernels: list[str],
    run_results: list[dict[str, Any]],
    run_events: dict[tuple[str, str], list[dict[str, Any]]],
    top_n: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for kernel in kernels:
        kernel_results = [row for row in run_results if row["kernel"] == kernel]
        run_ids = [str(row["run_id"]) for row in kernel_results]
        totals_by_run: dict[str, dict[str, float]] = {}
        counts_by_run: dict[str, dict[str, int]] = {}
        pass_names: set[str] = set()

        for run_id in run_ids:
            totals, counts = pass_totals_for_run(run_events.get((kernel, run_id), []))
            totals_by_run[run_id] = totals
            counts_by_run[run_id] = counts
            pass_names.update(totals)

        pass_summary: dict[str, Any] = {}
        for name in sorted(pass_names):
            wall_values = [totals_by_run[run_id].get(name, 0.0) for run_id in run_ids]
            count_values = [
                float(counts_by_run[run_id].get(name, 0)) for run_id in run_ids
            ]
            pass_summary[name] = {
                "wall_ms": summarize(wall_values),
                "invocations": summarize(count_values),
            }

        hotspots = [
            {
                "name": name,
                "median_ms": data["wall_ms"]["median_ms"],
                "mean_ms": data["wall_ms"]["mean_ms"],
                "invocation_median": data["invocations"]["median_ms"],
            }
            for name, data in pass_summary.items()
        ]
        hotspots.sort(key=lambda item: item["median_ms"] or 0.0, reverse=True)
        total_pass_ms = [sum(totals_by_run[run_id].values()) for run_id in run_ids]
        summary[kernel] = {
            "spec": kernel_results[0].get("spec", DEFAULT_SHAPES[kernel])
            if kernel_results
            else DEFAULT_SHAPES[kernel],
            "compile_est": summarize(
                [float(row["compile_est_ms"]) for row in kernel_results]
            ),
            "total_profiled_pass_ms": summarize(total_pass_ms),
            "passes": pass_summary,
            "hotspots": hotspots[:top_n],
            "events_count": sum(
                len(run_events.get((kernel, run_id), [])) for run_id in run_ids
            ),
            "profiled_pass_count": len(pass_summary),
        }
    return summary


def write_events_csv(path: Path, events: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kernel",
        "phase",
        "run_id",
        "sequence",
        "depth",
        "kind",
        "name",
        "display_name",
        "wall_ms",
        "percent",
        "compile_est_ms",
        "worker_log",
        "raw_line",
    ]
    write_projected_csv(path, fieldnames, events)


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "kernel",
        "pass",
        "median_ms",
        "mean_ms",
        "stdev_ms",
        "min_ms",
        "max_ms",
        "run_count",
        "invocation_median",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for kernel, kernel_summary in summary.items():
            for name, data in kernel_summary["passes"].items():
                wall = data["wall_ms"]
                invocations = data["invocations"]
                writer.writerow(
                    {
                        "kernel": kernel,
                        "pass": name,
                        "median_ms": wall["median_ms"],
                        "mean_ms": wall["mean_ms"],
                        "stdev_ms": wall["stdev_ms"],
                        "min_ms": wall["min_ms"],
                        "max_ms": wall["max_ms"],
                        "run_count": wall["count"],
                        "invocation_median": invocations["median_ms"],
                    }
                )


def write_hotspots_markdown(path: Path, document: dict[str, Any]) -> None:
    lines = [
        "# Pass profile hotspots",
        "",
        f"Backend profile: `{document['metadata'].get('backend_profile') or 'unknown'}`",
        f"Commit SHA: `{document['metadata'].get('commit_sha') or 'unknown'}`",
        "",
    ]
    for kernel, kernel_summary in document["summary"].items():
        lines.extend(
            [
                f"## {kernel}",
                "",
                "| Rank | Pass | Median (ms) | Mean (ms) | Median invocations |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for rank, row in enumerate(kernel_summary["hotspots"], start=1):
            lines.append(
                f"| {rank} | `{row['name']}` | {row['median_ms']:.3f} | "
                f"{row['mean_ms']:.3f} | {row['invocation_median']:.1f} |"
            )
        if not kernel_summary["hotspots"]:
            lines.append("| n/a | No pass timing rows parsed | n/a | n/a | n/a |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_parent(args: argparse.Namespace) -> int:
    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    unknown = [k for k in kernels if k not in DEFAULT_KERNELS]
    if unknown:
        raise ValueError(f"Unknown kernels: {unknown}. Supported: {DEFAULT_KERNELS}")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    flaggems_src = Path(args.flaggems_root) / "src"
    if not flaggems_src.exists():
        raise FileNotFoundError(f"FlagGems src directory not found: {flaggems_src}")

    cache_root = Path(args.cache_root)
    session_root = cache_root.parent / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    work_root = session_root
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {args.backend}")
    print(f"FlagGems root: {args.flaggems_root}")
    print(f"Kernels: {', '.join(kernels)}")
    print(f"Repeat: {args.repeat}, warmup: {args.warmup}")
    print(f"Temporary work root: {work_root}")
    print("TRITON_ANCHOR_PROFILE=1")

    run_results: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    repeat_events: dict[tuple[str, str], list[dict[str, Any]]] = {}
    warnings: list[str] = []

    try:
        for kernel in kernels:
            for warm_idx in range(args.warmup):
                run_child(args, kernel, "warmup", warm_idx, work_root)
            for run_idx in range(args.repeat):
                result, events = run_child(args, kernel, "repeat", run_idx, work_root)
                run_results.append(result)
                all_events.extend(events)
                repeat_events[(kernel, str(run_idx))] = events
                if not any(event.get("kind") == "pass" for event in events):
                    raise RuntimeError(
                        f"No MLIR pass timing rows parsed for {kernel} repeat {run_idx}."
                    )

        summary = build_summary(kernels, run_results, repeat_events, args.top_n)
        document = {
            "metadata": {
                "backend": args.backend,
                "vendor": args.vendor or args.backend,
                "flaggems_root": args.flaggems_root,
                "kernels": kernels,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "rtol": args.rtol,
                "atol": args.atol,
                "skip_correctness": args.skip_correctness,
                "commit_sha": os.environ.get("GITHUB_SHA"),
                "backend_profile": os.environ.get("BACKEND_PROFILE"),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "work_root": str(work_root),
                "env": {
                    "TRITON_ANCHOR_PROFILE": "1",
                    "TRITON_CHIP_NAME": os.environ.get("TRITON_CHIP_NAME"),
                    "TRITON_TO_PPL_MODE": os.environ.get("TRITON_TO_PPL_MODE"),
                    "PPL_PROJECT_ROOT": os.environ.get("PPL_PROJECT_ROOT"),
                    "LLVM_BUILD_DIR": os.environ.get("LLVM_BUILD_DIR"),
                },
            },
            "summary": summary,
            "events": all_events,
            "warnings": warnings,
        }

        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )
        write_events_csv(Path(args.output_events_csv), all_events)
        write_summary_csv(Path(args.output_summary_csv), summary)
        write_hotspots_markdown(Path(args.output_hotspots_markdown), document)

        print(f"Wrote JSON: {args.output_json}")
        print(f"Wrote events CSV: {args.output_events_csv}")
        print(f"Wrote summary CSV: {args.output_summary_csv}")
        print(f"Wrote hotspots Markdown: {args.output_hotspots_markdown}")
        print("Hotspots:")
        for kernel in kernels:
            hotspots = summary[kernel]["hotspots"]
            if hotspots:
                top = hotspots[0]
                print(f"  {kernel}: {top['name']} median={top['median_ms']:.3f} ms")
            else:
                print(f"  {kernel}: no pass timing rows parsed")
        return 0
    finally:
        if not args.keep_workdirs:
            shutil.rmtree(work_root, ignore_errors=True)


def main() -> int:
    return run_parent(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
