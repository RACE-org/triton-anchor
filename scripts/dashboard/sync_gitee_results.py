#!/usr/bin/env python3
"""Normalize Gitee local-CI results into the GitHub Pages data contracts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LOCAL_CI_SHARED_DIR = Path(__file__).resolve().parents[1] / "local_ci" / "shared"
sys.path.insert(0, str(LOCAL_CI_SHARED_DIR))
from result_paths import gitee_tree_url, result_run_dir, result_task_dir  # noqa: E402


DEFAULT_PROFILE = "sophgo-cmodel"
DEFAULT_RESULTS_WEB_URL = (
    "https://gitee.com/race-org/triton-anchor-local-ci-results"
)
RUN_ID_RE = re.compile(r"^(\d{8}T\d{6}Z)-")


@dataclass(frozen=True)
class Run:
    source_branch: str
    sha: str
    run_id: str
    path: Path
    summary: dict[str, str]

    @property
    def tested_at(self) -> str:
        match = RUN_ID_RE.match(self.run_id)
        if not match:
            return ""
        value = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
        return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--full-test-source-branch", required=True)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--backend-name", default="Sophgo")
    parser.add_argument("--results-branch", default="local-ci-results")
    parser.add_argument("--results-web-url", default=DEFAULT_RESULTS_WEB_URL)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def discover_runs(results_dir: Path, source_branch: str) -> list[Run]:
    branch_dir = results_dir / result_task_dir(source_branch)
    runs: list[Run] = []
    if not branch_dir.is_dir():
        return runs
    for sha_dir in branch_dir.iterdir():
        if not sha_dir.is_dir():
            continue
        for run_dir in sha_dir.iterdir():
            if not run_dir.is_dir() or not RUN_ID_RE.match(run_dir.name):
                continue
            summary = parse_summary(run_dir / "delivery-summary.txt")
            run_source_branch = summary.get("branch") or source_branch
            runs.append(
                Run(
                    run_source_branch,
                    sha_dir.name,
                    run_dir.name,
                    run_dir,
                    summary,
                )
            )
    return sorted(runs, key=lambda run: run.run_id)


def status_code(run: Run) -> int | None:
    value = run.summary.get("status")
    if value is None:
        result = read_json(run.path / "result.json")
        value = str(result.get("status")) if result and "status" in result else None
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def normalize_stage(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"pass", "passed", "success"}:
        return "success"
    if normalized in {"warn", "warning"}:
        return "warning"
    if normalized in {"fail", "failed", "failure", "error"}:
        return "failure"
    if normalized in {"disabled", "skipped"}:
        return "disabled"
    return "unknown"


def result_url(run: Run, web_url: str, results_branch: str) -> str:
    relative = result_run_dir(run.source_branch, run.sha, run.run_id).as_posix()
    return gitee_tree_url(web_url, results_branch, relative)


def backend_document(
    runs: list[Run], backend_name: str, profile: str, web_url: str, results_branch: str
) -> dict[str, Any]:
    latest = runs[-1] if runs else None
    if latest is None:
        sophgo = {
            "id": profile,
            "name": backend_name,
            "profile": profile,
            "state": "unknown",
            "sha": "",
            "branch": "",
            "tested_at": "",
            "tests": {
                "delivery": "unknown",
                "compile_time": "unknown",
                "pass_profile": "unknown",
                "ir_serialization": "unknown",
            },
            "result_url": "",
        }
    else:
        code = status_code(latest)
        tests = {
            "delivery": "success" if code == 0 else "failure" if code is not None else "unknown",
            "compile_time": normalize_stage(latest.summary.get("compile_time_status", "")),
            "pass_profile": normalize_stage(latest.summary.get("pass_profile_status", "")),
            "ir_serialization": normalize_stage(
                latest.summary.get("ir_serialization_status", "")
            ),
        }
        if tests["delivery"] == "failure":
            overall = "failure"
        elif "warning" in tests.values():
            overall = "warning"
        elif tests["delivery"] == "success":
            overall = "success"
        else:
            overall = "unknown"
        sophgo = {
            "id": profile,
            "name": backend_name,
            "profile": profile,
            "state": overall,
            "sha": latest.sha,
            "branch": latest.source_branch,
            "tested_at": latest.tested_at,
            "tests": tests,
            "result_url": result_url(latest, web_url, results_branch),
        }

    placeholders = []
    for suffix in ("b", "c", "d"):
        placeholders.append(
            {
                "id": f"backend-{suffix}",
                "name": f"Backend {suffix.upper()}",
                "profile": "待接入",
                "state": "unknown",
                "sha": "",
                "branch": "",
                "tested_at": "",
                "tests": {
                    "delivery": "unknown",
                    "compile_time": "unknown",
                    "pass_profile": "unknown",
                    "ir_serialization": "unknown",
                },
                "result_url": "",
            }
        )
    return {
        "schema": "triton-anchor-backend-status-list/v1",
        "data_mode": "live",
        "backends": [sophgo, *placeholders],
    }


def latest_valid_run(runs: Iterable[Run], file_name: str) -> tuple[Run | None, dict[str, Any] | None]:
    for run in reversed(list(runs)):
        document = read_json(run.path / file_name)
        if document is not None:
            return run, document
    return None, None


def number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def normalize_operator_status(row: dict[str, Any]) -> str:
    raw_status = str(row.get("test_status") or "").strip().lower()
    status_map = {
        "success": "passed",
        "passed": "passed",
        "pass": "passed",
        "\u6210\u529f": "passed",
        "failure": "failed",
        "failed": "failed",
        "fail": "failed",
        "error": "failed",
        "\u5931\u8d25": "failed",
        "timeout": "timeout",
        "timed_out": "timeout",
        "\u8d85\u65f6": "timeout",
    }
    if raw_status in status_map:
        return status_map[raw_status]
    if row.get("timeout_reason"):
        return "timeout"
    exit_code = row.get("exit_code")
    if exit_code == 0:
        return "passed"
    if isinstance(exit_code, int):
        return "timeout" if exit_code == -9 else "failed"
    return "unknown"


def full_test_document(
    runs: list[Run],
    backend_name: str,
    profile: str,
    web_url: str,
    results_branch: str,
) -> tuple[dict[str, Any] | None, Run | None]:
    run, source = latest_valid_run(runs, "flaggems-summary.json")
    if run is None or source is None:
        return None, None
    source_rows = source.get("results")
    if not isinstance(source_rows, list):
        return None, None

    operators: list[dict[str, Any]] = []
    for fallback_index, source_row in enumerate(source_rows, start=1):
        if not isinstance(source_row, dict):
            continue
        name = str(source_row.get("op") or source_row.get("marker") or "").strip()
        if not name:
            continue
        status = normalize_operator_status(source_row)
        raw_stage = str(source_row.get("first_failed_stage") or "").strip()
        stage_is_success = raw_stage.lower() in {
            "",
            "pass",
            "passed",
            "success",
            "\u5168\u90e8\u901a\u8fc7",
        }
        duration_seconds = number(source_row.get("duration_seconds"))
        operators.append(
            {
                "index": int(source_row.get("index") or fallback_index),
                "name": name,
                "status": status,
                "failure_stage": None if status == "passed" or stage_is_success else raw_stage,
                "duration_ms": (
                    round(duration_seconds * 1000.0, 3)
                    if duration_seconds is not None
                    else None
                ),
                "tested_at": str(source_row.get("started_at") or "").strip(),
            }
        )

    if not operators:
        return None, None

    document = {
        "schema": "triton-anchor-full-test/v1",
        "data_mode": "live",
        "source_schema": source.get("schema", ""),
        "source_summary": source.get("summary", {}),
        "run": {
            "id": run.run_id,
            "trigger": "manual",
            "state": "completed",
            "backend": f"{backend_name} CModel",
            "profile": profile,
            "sha": run.sha,
            "branch": run.source_branch,
            "started_at": run.tested_at,
            "finished_at": "",
            "result_url": result_url(run, web_url, results_branch),
        },
        "operators": operators,
    }
    return document, run


def nested_number(document: dict[str, Any] | None, *keys: str) -> float | None:
    value: object = document
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return number(value)


def status_for_delta(delta_percent: float | None, threshold_ratio: float, valid: bool = True) -> str:
    if not valid:
        return "failure"
    if delta_percent is not None and abs(delta_percent) > threshold_ratio * 100:
        return "warning"
    return "success"


def previous_document(runs: list[Run], current: Run | None, file_name: str) -> dict[str, Any] | None:
    if current is None:
        return None
    older = [run for run in runs if run.run_id < current.run_id and run.sha != current.sha]
    _, document = latest_valid_run(older, file_name)
    return document


def candidate_and_baseline(
    runs: list[Run],
    candidate_file: str,
    baseline_file: str,
) -> tuple[Run | None, dict[str, Any] | None, dict[str, Any] | None]:
    run, candidate = latest_valid_run(runs, candidate_file)
    if run is None or candidate is None:
        return None, None, None
    baseline = read_json(run.path / baseline_file)
    if baseline is None:
        baseline = previous_document(runs, run, candidate_file)
    return run, candidate, baseline


def percent_delta(candidate_ms: float, baseline_ms: float | None) -> float | None:
    if baseline_ms in (None, 0):
        return None
    return ((candidate_ms / baseline_ms) - 1.0) * 100.0


def compile_rows(runs: list[Run], threshold_ratio: float) -> tuple[list[dict[str, Any]], Run | None]:
    run, candidate, baseline = candidate_and_baseline(
        runs, "compile-benchmark.json", "compile-benchmark-base.json"
    )
    if run is None or candidate is None:
        return [], None
    metadata = candidate.get("metadata", {})
    kernels = metadata.get("kernels", []) if isinstance(metadata, dict) else []
    if not isinstance(kernels, list):
        kernels = []
    rows: list[dict[str, Any]] = []
    for kernel_value in kernels:
        kernel = str(kernel_value)
        candidate_ms = nested_number(candidate, "summary", kernel, "compile_est", "median_ms")
        if candidate_ms is None:
            continue
        baseline_ms = nested_number(baseline, "summary", kernel, "compile_est", "median_ms")
        delta = percent_delta(candidate_ms, baseline_ms)
        correct = bool(
            candidate.get("summary", {}).get(kernel, {}).get("all_correct", True)
            if isinstance(candidate.get("summary"), dict)
            else True
        )
        rows.append(
            {
                "name": kernel,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "delta_percent": delta,
                "status": status_for_delta(delta, threshold_ratio, correct),
            }
        )
    return rows, run


def pass_rows(runs: list[Run], threshold_ratio: float) -> tuple[list[dict[str, Any]], Run | None]:
    run, candidate, baseline = candidate_and_baseline(
        runs, "pass-profile.json", "pass-profile-base.json"
    )
    if run is None or candidate is None:
        return [], None
    rows: list[dict[str, Any]] = []
    summary = candidate.get("summary", {})
    if not isinstance(summary, dict):
        return rows, run
    for kernel, kernel_data in summary.items():
        if not isinstance(kernel_data, dict):
            continue
        hotspots = kernel_data.get("hotspots", [])
        if not isinstance(hotspots, list):
            continue
        for hotspot in hotspots:
            if not isinstance(hotspot, dict):
                continue
            pass_name = str(hotspot.get("name") or "")
            if not pass_name or pass_name in {"Total", "Rest"} or pass_name.startswith("(A)"):
                continue
            candidate_ms = number(hotspot.get("median_ms"))
            if candidate_ms is None:
                continue
            baseline_ms = nested_number(
                baseline, "summary", str(kernel), "passes", pass_name, "wall_ms", "median_ms"
            )
            delta = percent_delta(candidate_ms, baseline_ms)
            rows.append(
                {
                    "name": f"{kernel} / {pass_name}",
                    "median_ms": candidate_ms,
                    "delta_percent": delta,
                    "status": status_for_delta(delta, threshold_ratio),
                }
            )
    rows.sort(key=lambda row: float(row["median_ms"]), reverse=True)
    return rows[:10], run


def ir_rows(runs: list[Run], threshold_ratio: float) -> tuple[list[dict[str, Any]], Run | None]:
    run, candidate, baseline = candidate_and_baseline(
        runs, "ir-serialization.json", "ir-serialization-base.json"
    )
    if run is None or candidate is None:
        return [], None
    summary = candidate.get("summary", {})
    if not isinstance(summary, dict):
        return [], run
    metric_names = ("serialize", "write_text", "read_text", "deserialize", "roundtrip")
    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        candidate_values = [
            nested_number(candidate, "summary", str(kernel), "metrics", metric, "median_ms")
            for kernel in summary
        ]
        candidate_values = [value for value in candidate_values if value is not None]
        if not candidate_values:
            continue
        candidate_ms = statistics.median(candidate_values)
        baseline_values = [
            nested_number(baseline, "summary", str(kernel), "metrics", metric, "median_ms")
            for kernel in summary
        ]
        baseline_values = [value for value in baseline_values if value is not None]
        baseline_ms = statistics.median(baseline_values) if baseline_values else None
        delta = percent_delta(candidate_ms, baseline_ms)
        rows.append(
            {
                "name": metric,
                "median_ms": candidate_ms,
                "baseline_ms": baseline_ms,
                "delta_percent": delta,
                "status": status_for_delta(delta, threshold_ratio),
            }
        )
    return rows, run


def performance_document(
    runs: list[Run], backend_name: str, profile: str, web_url: str, results_branch: str
) -> dict[str, Any]:
    threshold_ratio = 0.20
    compile_time, compile_run = compile_rows(runs, threshold_ratio)
    pass_profile, pass_run = pass_rows(runs, threshold_ratio)
    ir_serialization, ir_run = ir_rows(runs, threshold_ratio)
    source_runs = [run for run in (compile_run, pass_run, ir_run) if run is not None]
    newest = max(source_runs, key=lambda run: run.run_id) if source_runs else None

    def source(run: Run | None) -> dict[str, str] | None:
        if run is None:
            return None
        return {
            "sha": run.sha,
            "run_id": run.run_id,
            "result_url": result_url(run, web_url, results_branch),
        }

    return {
        "schema": "triton-anchor-performance-summary/v1",
        "data_mode": "live",
        "backend": f"{backend_name} CModel",
        "profile": profile,
        "sha": newest.sha if newest else "",
        "generated_at": newest.tested_at if newest else "",
        "sources": {
            "compile_time": source(compile_run),
            "pass_profile": source(pass_run),
            "ir_serialization": source(ir_run),
        },
        "compile_time": {
            "unit": "ms",
            "threshold": threshold_ratio,
            "kernels": compile_time,
        },
        "pass_profile": {"unit": "ms", "hotspots": pass_profile},
        "ir_serialization": {"unit": "ms", "metrics": ir_serialization},
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_full_test_csv(path: Path, document: dict[str, Any]) -> None:
    status_labels = {
        "passed": "\u901a\u8fc7",
        "failed": "\u5931\u8d25",
        "timeout": "\u8d85\u65f6",
        "unknown": "\u672a\u77e5",
    }
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "\u5e8f\u53f7",
                "\u7b97\u5b50\u540d\u79f0",
                "\u6d4b\u8bd5\u72b6\u6001",
                "\u5931\u8d25\u9636\u6bb5",
                "\u8017\u65f6(ms)",
                "\u6d4b\u8bd5\u65f6\u95f4",
            ]
        )
        for row in document["operators"]:
            writer.writerow(
                [
                    row["index"],
                    row["name"],
                    status_labels.get(row["status"], row["status"]),
                    row["failure_stage"] or "",
                    row["duration_ms"] if row["duration_ms"] is not None else "",
                    row["tested_at"],
                ]
            )


def sync_dashboard(
    results_dir: Path,
    output_dir: Path,
    source_branch: str,
    full_test_source_branch: str,
    profile: str = DEFAULT_PROFILE,
    backend_name: str = "Sophgo",
    results_branch: str = "local-ci-results",
    results_web_url: str = DEFAULT_RESULTS_WEB_URL,
) -> None:
    main_runs = discover_runs(results_dir, source_branch)
    if not main_runs:
        raise RuntimeError(f"No Gitee CI runs found for {source_branch!r}")
    full_test_runs = discover_runs(results_dir, full_test_source_branch)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "backend-status.json",
        backend_document(
            main_runs, backend_name, profile, results_web_url, results_branch
        ),
    )
    write_json(
        output_dir / "performance.json",
        performance_document(
            main_runs, backend_name, profile, results_web_url, results_branch
        ),
    )
    manifest_path = output_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest is None:
        raise RuntimeError(f"Dashboard manifest is missing or invalid: {manifest_path}")
    full_test, _ = full_test_document(
        full_test_runs, backend_name, profile, results_web_url, results_branch
    )
    if full_test is not None:
        write_json(output_dir / "full-test.json", full_test)
        write_full_test_csv(output_dir / "full-test.csv", full_test)
        full_test_mode = "live"
    else:
        full_test_mode = "mock"

    manifest["mode"] = "live" if full_test_mode == "live" else "mixed"
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    manifest["data_modes"] = {
        "full_test": full_test_mode,
        "backend_status": "live",
        "performance": "live",
    }
    write_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    sync_dashboard(
        results_dir=args.results_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        source_branch=args.source_branch,
        full_test_source_branch=args.full_test_source_branch,
        profile=args.profile,
        backend_name=args.backend_name,
        results_branch=args.results_branch,
        results_web_url=args.results_web_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
