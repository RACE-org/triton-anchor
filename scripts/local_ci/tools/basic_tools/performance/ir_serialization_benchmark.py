#!/usr/bin/env python3
"""Measure TTIR text serialization and file-based deserialization overhead.

The benchmark first compiles each configured FlagGems kernel once with the
existing compile_benchmark worker. It then locates the real ``.ttir`` cache
artifacts produced by Triton and measures only the following operations:

* ``serialize``: convert parsed MLIR modules to canonical text with ``str``;
* ``write_text``: write that UTF-8 text to disk;
* ``read_text``: read the generated files as UTF-8 text;
* ``deserialize``: parse the files with ``ir.parse_mlir_module``;
* ``roundtrip``: serialize, write, and deserialize.

``deserialize`` includes the parser's own file I/O and module clone. The
reported ``parse_estimate`` subtracts a separately measured raw read and is
therefore diagnostic rather than an exact parser-only measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from common import (  # noqa: E402
    DEFAULT_KERNELS,
    neighboring_compile_benchmark,
    summarize,
    write_projected_csv,
)

METRICS = (
    "serialize",
    "write_text",
    "read_text",
    "deserialize",
    "parse_estimate",
    "roundtrip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure TTIR serialization/deserialization overhead."
    )
    parser.add_argument("--backend", default="sophgo")
    parser.add_argument("--vendor", default=None)
    parser.add_argument(
        "--flaggems-root",
        default=os.environ.get("FLAGGEMS_ROOT", "/workspace/FlagGems"),
    )
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS))
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--work-root",
        default=str(
            Path(os.environ.get("TMPDIR", "/tmp")) / "triton_anchor_ir_serialization"
        ),
    )
    parser.add_argument("--output-json", default="ir_serialization_results.json")
    parser.add_argument("--output-csv", default="ir_serialization_results.csv")
    parser.add_argument("--output-markdown", default="ir_serialization_summary.md")
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument("--verbose-worker", action="store_true")
    return parser.parse_args()


def generate_ttir(
    args: argparse.Namespace,
    kernel: str,
    kernel_root: Path,
) -> tuple[list[Path], dict[str, Any]]:
    cache_dir = kernel_root / "cache"
    dump_dir = kernel_root / "dump"
    result_file = kernel_root / "compile-worker.json"
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
        "ir-serialization",
        "--worker-run-id",
        "0",
        "--worker-output",
        str(result_file),
        "--worker-cache-dir",
        str(cache_dir),
        "--worker-dump-dir",
        str(dump_dir),
        "--worker-seed",
        "20260625",
    ]
    env = os.environ.copy()
    flaggems_src = str(Path(args.flaggems_root) / "src")
    env["PYTHONPATH"] = flaggems_src + os.pathsep + env.get("PYTHONPATH", "")
    env["FLAGGEMS_ROOT"] = args.flaggems_root
    env["GEMS_VENDOR"] = args.vendor or args.backend

    print(f"[{kernel}] generating TTIR cache artifacts")
    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    worker_output = completed.stdout or ""
    (kernel_root / "compile-worker.log").write_text(
        worker_output, encoding="utf-8", errors="replace"
    )
    if args.verbose_worker and worker_output:
        print(worker_output, end="" if worker_output.endswith("\n") else "\n")
    if not result_file.is_file():
        raise RuntimeError(
            f"Compile worker did not write {result_file}; output:\n{worker_output}"
        )
    worker_result = json.loads(result_file.read_text(encoding="utf-8-sig"))
    if completed.returncode != 0 or worker_result.get("status") != "pass":
        raise RuntimeError(
            f"Compile worker failed for {kernel}: "
            f"{worker_result.get('error', worker_result.get('status'))}\n"
            f"{worker_output}"
        )

    ttir_files = sorted(path for path in cache_dir.rglob("*.ttir") if path.is_file())
    if not ttir_files:
        cached_names = sorted(
            str(path.relative_to(cache_dir))
            for path in cache_dir.rglob("*")
            if path.is_file()
        )
        raise RuntimeError(
            f"No .ttir cache artifact was generated for {kernel}. "
            f"Cached files: {cached_names[:30]}"
        )
    print(
        f"[{kernel}] generated {len(ttir_files)} TTIR module(s), "
        f"{sum(path.stat().st_size for path in ttir_files)} bytes"
    )
    return ttir_files, worker_result


def create_context() -> Any:
    from triton._C.libtriton import ir

    context = ir.context()
    ir.load_dialects(context)
    try:
        from triton._C.libtriton import anchor
    except ImportError:
        anchor = None
    if anchor is not None:
        anchor.load_dialects(context)
    return context


def elapsed_ms(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter_ns()
    value = fn()
    end = time.perf_counter_ns()
    return (end - start) / 1_000_000.0, value


def measure_once(
    modules: list[Any],
    context: Any,
    output_files: list[Path],
    kernel: str,
    phase: str,
    run_id: int,
) -> dict[str, Any]:
    from triton._C.libtriton import ir

    if not modules or len(output_files) != len(modules):
        raise RuntimeError("IR round-trip requires a nonempty module/output pair list")

    serialize_ms, texts = elapsed_ms(lambda: [str(module) for module in modules])

    def write_all() -> None:
        for path, text in zip(output_files, texts):
            path.write_text(text, encoding="utf-8")

    write_text_ms, _ = elapsed_ms(write_all)

    def read_all() -> list[str]:
        return [path.read_text(encoding="utf-8") for path in output_files]

    read_text_ms, read_texts = elapsed_ms(read_all)

    def parse_all() -> list[Any]:
        return [ir.parse_mlir_module(str(path), context) for path in output_files]

    deserialize_ms, parsed_modules = elapsed_ms(parse_all)
    if len(parsed_modules) != len(modules) or len(read_texts) != len(modules):
        raise RuntimeError("IR round-trip changed the module count")
    # Check correctness outside the timed parser/serialization intervals.
    for original, written, parsed in zip(texts, read_texts, parsed_modules):
        if original != written:
            raise RuntimeError("IR round-trip changed serialized file contents")
        if not parsed.verify():
            raise RuntimeError("IR round-trip produced an invalid MLIR module")
        if original.strip() != str(parsed).strip():
            raise RuntimeError("IR round-trip changed canonical module contents")

    parse_estimate_ms = max(0.0, deserialize_ms - read_text_ms)
    roundtrip_ms = serialize_ms + write_text_ms + deserialize_ms
    return {
        "kernel": kernel,
        "phase": phase,
        "run_id": run_id,
        "module_count": len(modules),
        "ir_bytes": sum(len(text.encode("utf-8")) for text in texts),
        "serialize_ms": serialize_ms,
        "write_text_ms": write_text_ms,
        "read_text_ms": read_text_ms,
        "deserialize_ms": deserialize_ms,
        "parse_estimate_ms": parse_estimate_ms,
        "roundtrip_ms": roundtrip_ms,
        "roundtrip_verified": True,
    }


def benchmark_kernel(
    args: argparse.Namespace,
    kernel: str,
    work_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kernel_root = work_root / kernel
    kernel_root.mkdir(parents=True, exist_ok=True)
    ttir_files, worker_result = generate_ttir(args, kernel, kernel_root)
    context = create_context()
    from triton._C.libtriton import ir

    modules = [ir.parse_mlir_module(str(path), context) for path in ttir_files]
    roundtrip_dir = kernel_root / "roundtrip"
    roundtrip_dir.mkdir(parents=True, exist_ok=True)
    output_files = [
        roundtrip_dir / f"module-{index}.ttir" for index in range(len(modules))
    ]

    for warmup_id in range(args.warmup):
        measure_once(
            modules,
            context,
            output_files,
            kernel,
            "warmup",
            warmup_id,
        )

    rows = [
        measure_once(
            modules,
            context,
            output_files,
            kernel,
            "repeat",
            run_id,
        )
        for run_id in range(args.repeat)
    ]
    for row in rows:
        print(
            f"[{kernel}] run {row['run_id']}: serialize={row['serialize_ms']:.3f} ms, "
            f"deserialize={row['deserialize_ms']:.3f} ms, "
            f"roundtrip={row['roundtrip_ms']:.3f} ms"
        )
    return rows, {
        "spec": worker_result.get("spec", {}),
        "source_files": [str(path.relative_to(kernel_root)) for path in ttir_files],
        "module_count": len(modules),
        "ir_bytes": rows[0]["ir_bytes"] if rows else 0,
    }


def build_summary(
    kernels: list[str],
    rows: list[dict[str, Any]],
    kernel_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for kernel in kernels:
        kernel_rows = [row for row in rows if row["kernel"] == kernel]
        metadata = kernel_metadata[kernel]
        summary[kernel] = {
            **metadata,
            "metrics": {
                metric: summarize([float(row[f"{metric}_ms"]) for row in kernel_rows])
                for metric in METRICS
            },
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kernel",
        "phase",
        "run_id",
        "module_count",
        "ir_bytes",
        *(f"{metric}_ms" for metric in METRICS),
    ]
    write_projected_csv(path, fieldnames, rows)


def write_markdown(path: Path, document: dict[str, Any]) -> None:
    metadata = document["metadata"]
    lines = [
        "# IR serialization profile",
        "",
        f"Commit SHA: `{metadata.get('commit_sha') or 'unknown'}`",
        f"Backend profile: `{metadata.get('backend_profile') or 'unknown'}`",
        "",
        "`deserialize` includes file read, MLIR parse, and module clone. "
        "`parse estimate` is deserialize minus a separately measured raw read.",
        "",
        "| Kernel | Modules | IR bytes | Serialize median (ms) | Write median (ms) | "
        "Read median (ms) | Deserialize median (ms) | Parse estimate (ms) | Round-trip (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kernel, data in document["summary"].items():
        metrics = data["metrics"]
        lines.append(
            f"| {kernel} | {data['module_count']} | {data['ir_bytes']} | "
            f"{metrics['serialize']['median_ms']:.3f} | "
            f"{metrics['write_text']['median_ms']:.3f} | "
            f"{metrics['read_text']['median_ms']:.3f} | "
            f"{metrics['deserialize']['median_ms']:.3f} | "
            f"{metrics['parse_estimate']['median_ms']:.3f} | "
            f"{metrics['roundtrip']['median_ms']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    kernels = [value.strip() for value in args.kernels.split(",") if value.strip()]
    unknown = [kernel for kernel in kernels if kernel not in DEFAULT_KERNELS]
    if unknown:
        raise ValueError(f"Unknown kernels: {unknown}. Supported: {DEFAULT_KERNELS}")
    if not kernels:
        raise ValueError("--kernels must contain at least one kernel")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if not (Path(args.flaggems_root) / "src").is_dir():
        raise FileNotFoundError(
            f"FlagGems src directory not found: {Path(args.flaggems_root) / 'src'}"
        )

    base_work_root = Path(args.work_root)
    work_root = base_work_root / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    work_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    kernel_metadata: dict[str, dict[str, Any]] = {}
    try:
        print(f"Backend: {args.backend}")
        print(f"FlagGems root: {args.flaggems_root}")
        print(f"Kernels: {', '.join(kernels)}")
        print(f"Repeat: {args.repeat}, warmup: {args.warmup}")
        print(f"Temporary work root: {work_root}")
        for kernel in kernels:
            rows, metadata = benchmark_kernel(args, kernel, work_root)
            all_rows.extend(rows)
            kernel_metadata[kernel] = metadata

        document = {
            "metadata": {
                "backend": args.backend,
                "vendor": args.vendor or args.backend,
                "backend_profile": os.environ.get("BACKEND_PROFILE"),
                "commit_sha": os.environ.get("GITHUB_SHA"),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "flaggems_root": args.flaggems_root,
                "kernels": kernels,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "measurement_boundary": {
                    "serialize": "str(module)",
                    "write_text": "Path.write_text(UTF-8)",
                    "read_text": "Path.read_text(UTF-8)",
                    "deserialize": "ir.parse_mlir_module(file, context); includes file read, parse, and clone",
                    "parse_estimate": "max(0, deserialize - raw read); diagnostic only",
                    "roundtrip": "serialize + write_text + deserialize",
                },
            },
            "summary": build_summary(kernels, all_rows, kernel_metadata),
            "raw": all_rows,
        }
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_csv(Path(args.output_csv), all_rows)
        write_markdown(Path(args.output_markdown), document)
        print(f"Wrote JSON: {args.output_json}")
        print(f"Wrote CSV: {args.output_csv}")
        print(f"Wrote Markdown: {args.output_markdown}")
        return 0
    finally:
        if not args.keep_workdirs:
            shutil.rmtree(work_root, ignore_errors=True)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
