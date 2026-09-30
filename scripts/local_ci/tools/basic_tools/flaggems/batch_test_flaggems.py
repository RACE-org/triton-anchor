#!/usr/bin/env python3
"""Run selected FlagGems operators in isolated pytest subprocesses."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from select_flaggems_tests import Entry, select_entries, write_selected


STAGES = ["Linalg生成", "MLIR生成", "C代码生成", "编译构建", "测试执行", "准确率验证"]
CSV_HEADERS = ["序号", "算子名称", "最开始失败阶段", "测试状态", "测试时间"]
PYTEST_COMPLETION_RE = re.compile(r"\b(?:PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)\b")


@dataclass(frozen=True)
class SelectedOperator:
    category: str
    op: str
    marker: str
    test_files: tuple[str, ...]


@dataclass(frozen=True)
class OperatorResult:
    index: int
    category: str
    op: str
    marker: str
    test_files: tuple[str, ...]
    first_failed_stage: str
    test_status: str
    started_at: str
    duration_seconds: float
    exit_code: int | None
    timeout_reason: str
    completed_tests: int
    timeout_extensions: int
    passed: int
    failed: int
    errors: int
    skipped: int
    log_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("impact", "sample", "full", "single"), default="impact"
    )
    parser.add_argument("--ops", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--sample-size", type=int, default=6)
    parser.add_argument("--seed", default="")
    parser.add_argument("--op", default="")
    parser.add_argument("--affected-ops", default="")
    parser.add_argument("--whitelist", required=True)
    parser.add_argument("--full-list", default="")
    parser.add_argument("--flaggems-dir", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--selected-output", default="")
    parser.add_argument("--pytest-args", default="--ref cpu -vs")
    parser.add_argument("--idle-timeout-seconds", type=int, default=300)
    parser.add_argument("--total-timeout-seconds", type=int, default=6000)
    parser.add_argument("--full-timeout-extension-seconds", type=int, default=1800)
    parser.add_argument("--full-hard-timeout-seconds", type=int, default=14400)
    parser.add_argument("--clear-cache", choices=("0", "1"), default="0")
    return parser.parse_args()


def group_selected_entries(entries: list[Entry]) -> list[SelectedOperator]:
    grouped: dict[tuple[str, str, str], list[str]] = {}
    order: list[tuple[str, str, str]] = []
    for entry in entries:
        key = (entry.category, entry.op, entry.marker)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        if entry.test_file and entry.test_file not in grouped[key]:
            grouped[key].append(entry.test_file)
    return [SelectedOperator(*key, tuple(grouped[key])) for key in order]


def safe_file_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "operator"


def snapshot_dump_dir(dump_dir: Path) -> set[str]:
    if not dump_dir.is_dir():
        return set()
    return {item.name for item in dump_dir.iterdir()}


def new_dump_dirs(dump_dir: Path, before: set[str]) -> list[Path]:
    if not dump_dir.is_dir():
        return []
    return [
        item for item in dump_dir.iterdir() if item.name not in before and item.is_dir()
    ]


def check_linalg(dump_dirs: list[Path], output: str) -> bool:
    return any(
        path.name.endswith("_linalg.mlir")
        for directory in dump_dirs
        for path in directory.rglob("*.mlir")
    )


def check_mlir(dump_dirs: list[Path], output: str) -> bool:
    if any(
        not path.name.endswith("_linalg.mlir")
        for directory in dump_dirs
        for path in directory.rglob("*.mlir")
    ):
        return True
    return bool(re.search(r"\[Success\]:\s*ppl-compile\s+\S+\.mlir", output))


def check_c_code(dump_dirs: list[Path], output: str) -> bool:
    for directory in dump_dirs:
        device_dir = directory / "device"
        if device_dir.is_dir() and any(
            path.suffix == ".c" for path in device_dir.iterdir()
        ):
            return True
    return bool(re.search(r"Building C object\s+\S+\.c\.o", output))


def check_build(output: str) -> bool:
    return bool(re.search(r"\[Success\]:\s*cmake\s+", output)) and bool(
        re.search(r"\[Success\]:\s*make install", output)
    )


def summary_count(output: str, label: str) -> int:
    matches = re.findall(
        rf"(?<![A-Za-z])([0-9]+)\s+{label}\b", output, flags=re.IGNORECASE
    )
    return int(matches[-1]) if matches else 0


def pytest_counts(output: str) -> tuple[int, int, int, int]:
    return (
        summary_count(output, "passed"),
        summary_count(output, "failed"),
        summary_count(output, "errors?"),
        summary_count(output, "skipped"),
    )


def check_execution(output: str, passed: int, exit_code: int | None) -> bool:
    fatal = re.search(
        r"ASSERT_|Obtained [0-9]+ stack frames|Traceback \(most recent call last\)|"
        r"core dumped|Aborted|Fatal Python error",
        output,
        flags=re.IGNORECASE,
    )
    return exit_code == 0 and not fatal and passed > 0


def evaluate_stages(
    dump_dirs: list[Path], output: str, exit_code: int | None
) -> tuple[dict[str, bool], tuple[int, int, int, int]]:
    passed, failed, errors, skipped = pytest_counts(output)
    checks = (
        check_linalg(dump_dirs, output),
        check_mlir(dump_dirs, output),
        check_c_code(dump_dirs, output),
        check_build(output),
        check_execution(output, passed, exit_code),
        exit_code == 0 and failed == 0 and errors == 0,
    )
    return dict(zip(STAGES, checks, strict=True)), (passed, failed, errors, skipped)


def first_failed_stage(stages: dict[str, bool]) -> str:
    for stage in STAGES:
        if not stages.get(stage, False):
            return f"{stage}失败"
    return "全部通过"


def clear_cache_dir(path: Path) -> None:
    if not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def task_cache_paths(dump_dir: Path) -> list[Path]:
    """New tool invocations may clear only their task-owned cache and dumps."""
    task_text = os.environ.get("LOCAL_CI_TASK_ROOT", "")
    if not task_text:
        raise ValueError("Cache cleanup requires an explicit task root")
    task = Path(task_text).resolve(strict=True)
    cache_text = os.environ.get("TRITON_CACHE_DIR", "")
    if not cache_text:
        raise ValueError("Task FlagGems requires an owned TRITON_CACHE_DIR")
    paths = [Path(cache_text), dump_dir]
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_absolute()
            or path.resolve() == task
            or not path.resolve().is_relative_to(task)
        ):
            raise ValueError(f"FlagGems cache/dump escaped task root: {path}")
    return paths


def count_observed_completed_tests(output: str) -> int:
    completed = 0
    for line in output.splitlines():
        marker = PYTEST_COMPLETION_RE.search(line)
        node_separator = line.find("::")
        if marker and node_separator >= 0 and node_separator < marker.start():
            completed += 1
    return completed


def observed_completed_tests(log_path: Path) -> int:
    try:
        output = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return count_observed_completed_tests(output)


def next_full_soft_deadline(
    current_deadline: float,
    hard_deadline: float,
    extension_seconds: int,
    completed_tests: int,
    progress_checkpoint: int,
) -> float | None:
    if completed_tests <= progress_checkpoint or current_deadline >= hard_deadline:
        return None
    return min(current_deadline + extension_seconds, hard_deadline)


def build_pytest_command(
    selected: SelectedOperator, python_bin: str, pytest_args: str, flaggems_dir: Path
) -> list[str]:
    root = flaggems_dir.resolve()
    test_files = []
    for entry in selected.test_files or ("tests",):
        filename, separator, node = entry.partition("::")
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise ValueError("FlagGems test path must exist inside the source checkout")
        test_files.append(str(path) + (separator + node if separator else ""))
    return [
        python_bin,
        "-u",
        "-m",
        "pytest",
        "-s",
        *test_files,
        "-m",
        selected.marker,
        *shlex.split(pytest_args),
    ]


def run_operator(
    selected: SelectedOperator,
    index: int,
    args: argparse.Namespace,
    flaggems_dir: Path,
    dump_dir: Path,
    log_dir: Path,
) -> OperatorResult:
    started = datetime.now()
    started_monotonic = time.monotonic()
    log_name = f"{index:03d}-{safe_file_part(selected.op)}.log"
    log_path = log_dir / log_name
    # Upstream conftest writes result.json in cwd; the source may be a read-only mount.
    work_dir = (log_dir / f"{index:03d}-{safe_file_part(selected.op)}-work").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    command = build_pytest_command(
        selected, args.python_bin, args.pytest_args, flaggems_dir
    )
    command += ["-o", "cache_dir=" + str(work_dir / ".pytest_cache")]
    soft_deadline = float(args.total_timeout_seconds)
    hard_deadline = float(args.full_hard_timeout_seconds)
    progress_checkpoint = 0
    timeout_extensions = 0

    if args.clear_cache == "1":
        for cache_path in task_cache_paths(dump_dir):
            clear_cache_dir(cache_path)

    before = snapshot_dump_dir(dump_dir)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["FLAGGEMS_ROOT"] = str(flaggems_dir)
    environment["TRITON_DUMP_DIR"] = str(dump_dir)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["FLAGGEMS_CACHE_DIR"] = str(work_dir / ".flaggems")
    timeout_reason = ""

    print(
        f"[{index}] {selected.op}: marker={selected.marker}, "
        f"files={','.join(selected.test_files) or 'auto'}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        stream.write(f"command: {shlex.join(command)}\n")
        if args.mode == "full":
            stream.write(
                "timeout_policy: "
                f"idle={args.idle_timeout_seconds}s, "
                f"soft={args.total_timeout_seconds}s, "
                f"extension={args.full_timeout_extension_seconds}s, "
                f"hard={args.full_hard_timeout_seconds}s\n"
            )
        else:
            stream.write(
                "timeout_policy: "
                f"idle={args.idle_timeout_seconds}s, "
                f"strict_total={args.total_timeout_seconds}s\n"
            )
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=str(work_dir),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        atexit.register(terminate_process_group, process)
        previous_handlers = {}

        def cancel_operator(signum, _frame):
            terminate_process_group(process)
            process.wait()
            raise SystemExit(128 + signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, cancel_operator)
        last_size = log_path.stat().st_size
        last_activity = time.monotonic()
        while process.poll() is None:
            time.sleep(1)
            now = time.monotonic()
            if process.poll() is not None:
                break
            try:
                current_size = log_path.stat().st_size
            except OSError:
                current_size = last_size
            if current_size > last_size:
                last_size = current_size
                last_activity = now
            if (
                args.idle_timeout_seconds > 0
                and now - last_activity > args.idle_timeout_seconds
            ):
                timeout_reason = "idle"
                terminate_process_group(process)
                break
            elapsed = now - started_monotonic
            if args.total_timeout_seconds > 0:
                if args.mode != "full" and elapsed > args.total_timeout_seconds:
                    timeout_reason = "total"
                    terminate_process_group(process)
                    break
                if args.mode == "full" and elapsed > hard_deadline:
                    timeout_reason = "hard"
                    terminate_process_group(process)
                    break
                if args.mode == "full" and elapsed > soft_deadline:
                    completed_tests = observed_completed_tests(log_path)
                    next_deadline = next_full_soft_deadline(
                        soft_deadline,
                        hard_deadline,
                        args.full_timeout_extension_seconds,
                        completed_tests,
                        progress_checkpoint,
                    )
                    if next_deadline is None:
                        timeout_reason = "soft_no_progress"
                        terminate_process_group(process)
                        break
                    timeout_extensions += 1
                    extension_message = (
                        f"[{index}] {selected.op}: extending full timeout "
                        f"from {soft_deadline:.0f}s to {next_deadline:.0f}s; "
                        f"completed_tests={completed_tests}"
                    )
                    print(extension_message, flush=True)
                    stream.write(f"local-ci: {extension_message}\n")
                    stream.flush()
                    progress_checkpoint = completed_tests
                    soft_deadline = next_deadline
        exit_code = process.wait()
        atexit.unregister(terminate_process_group)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    output = log_path.read_text(encoding="utf-8", errors="replace")
    completed_tests = count_observed_completed_tests(output)
    dump_dirs = new_dump_dirs(dump_dir, before)
    stages, counts = evaluate_stages(dump_dirs, output, exit_code)
    passed, failed, errors, skipped = counts
    if timeout_reason:
        failed_stage = "超时"
        test_status = "超时"
    else:
        failed_stage = first_failed_stage(stages)
        test_status = "成功" if failed_stage == "全部通过" else "失败"

    duration = time.monotonic() - started_monotonic
    result = OperatorResult(
        index=index,
        category=selected.category,
        op=selected.op,
        marker=selected.marker,
        test_files=selected.test_files,
        first_failed_stage=failed_stage,
        test_status=test_status,
        started_at=started.strftime("%H:%M:%S"),
        duration_seconds=round(duration, 3),
        exit_code=exit_code,
        timeout_reason=timeout_reason,
        completed_tests=completed_tests,
        timeout_extensions=timeout_extensions,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        log_file=f"flaggems/{log_name}",
    )
    print(
        f"[{index}] {selected.op}: {test_status}, {failed_stage}, "
        f"exit={exit_code}, duration={duration:.1f}s, "
        f"completed_tests={completed_tests}, timeout_reason={timeout_reason or 'none'}, "
        f"extensions={timeout_extensions}, log={log_path}",
        flush=True,
    )
    if test_status != "成功":
        tail = output.splitlines()[-40:]
        print(f"--- {selected.op} failure tail ---", flush=True)
        print("\n".join(tail), flush=True)
        print(f"--- end {selected.op} failure tail ---", flush=True)
    return result


def write_reports(
    artifact_dir: Path, results: list[OperatorResult], args: argparse.Namespace
) -> None:
    csv_path = artifact_dir / "flaggems-summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "序号": result.index,
                    "算子名称": result.op,
                    "最开始失败阶段": result.first_failed_stage,
                    "测试状态": result.test_status,
                    "测试时间": result.started_at,
                }
            )

    passed = sum(result.test_status == "成功" for result in results)
    failed = sum(result.test_status == "失败" for result in results)
    timed_out = sum(result.test_status == "超时" for result in results)
    document = {
        "schema": "triton-anchor-local-ci/flaggems-v1",
        "mode": args.mode,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "affected_ops": [
            op.strip()
            for op in getattr(args, "affected_ops", "").split(",")
            if op.strip()
        ],
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "timed_out": timed_out,
            "status": "pass" if results and failed == 0 and timed_out == 0 else "fail",
        },
        "results": [asdict(result) for result in results],
    }
    (artifact_dir / "flaggems-summary.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# FlagGems test summary",
        "",
        f"- Mode: `{args.mode}`",
        f"- Total: {len(results)}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        f"- Timed out: {timed_out}",
        "",
        "| # | Operator | Category | Marker | First failed stage | Status | Completed tests | Timeout reason | Extensions | Duration (s) | Log |",
        "| ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.index} | {result.op} | {result.category} | {result.marker} | "
            f"{result.first_failed_stage} | {result.test_status} | "
            f"{result.completed_tests} | {result.timeout_reason or '-'} | "
            f"{result.timeout_extensions} | {result.duration_seconds:.3f} | "
            f"[{Path(result.log_file).name}]({result.log_file}) |"
        )
    (artifact_dir / "flaggems-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.mode == "full" and args.total_timeout_seconds > 0:
        if args.full_timeout_extension_seconds <= 0:
            raise ValueError("--full-timeout-extension-seconds must be positive")
        if args.full_hard_timeout_seconds < args.total_timeout_seconds:
            raise ValueError(
                "--full-hard-timeout-seconds must be at least --total-timeout-seconds"
            )
    flaggems_dir = Path(args.flaggems_dir).resolve()
    if not (flaggems_dir / "tests").is_dir():
        raise ValueError(
            f"FlagGems tests directory does not exist: {flaggems_dir / 'tests'}"
        )

    artifact_dir = Path(args.artifact_dir).resolve()
    log_dir = artifact_dir / "flaggems"
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(
        os.getenv("TRITON_DUMP_DIR", "/workspace/triton-dump-dir")
    ).resolve()
    protected_paths = {Path("/"), Path("/workspace"), Path.home().resolve()}
    if dump_dir in protected_paths:
        raise ValueError(f"Refusing to use unsafe TRITON_DUMP_DIR: {dump_dir}")
    dump_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_cache == "1":
        for cache_path in task_cache_paths(dump_dir):
            clear_cache_dir(cache_path)

    selected_entries = select_entries(args)
    selected = group_selected_entries(selected_entries)
    if not selected:
        raise ValueError("FlagGems selection must contain at least one operator")
    write_selected(
        args.selected_output,
        selected_entries,
        args,
        shlex.join([sys.executable, *sys.argv]),
    )
    print(f"FlagGems dir: {flaggems_dir}")
    print(f"Dump dir: {dump_dir}")
    print(f"Selected operators: {len(selected)}")

    results: list[OperatorResult] = []
    write_reports(artifact_dir, results, args)
    for index, operator in enumerate(selected, start=1):
        results.append(
            run_operator(operator, index, args, flaggems_dir, dump_dir, log_dir)
        )
        write_reports(artifact_dir, results, args)

    failed = [result for result in results if result.test_status != "成功"]
    print(
        f"FlagGems summary: {len(results) - len(failed)} passed, "
        f"{len(failed)} failed or timed out; CSV: {artifact_dir / 'flaggems-summary.csv'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FlagGems batch runner failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
