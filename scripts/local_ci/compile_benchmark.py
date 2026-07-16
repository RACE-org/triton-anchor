#!/usr/bin/env python3
"""Compile-time benchmark for FlagGems kernels through a Triton backend.

This script is intentionally independent from pytest.  It benchmarks a fixed
set of public FlagGems/PyTorch entry points:

  add        -> torch.add
  mm         -> torch.mm
  softmax    -> torch.softmax
  layernorm  -> torch.layer_norm

The parent process starts a fresh worker process for every measured repeat so
that Triton in-memory JIT caches cannot make a later "cold" run warm.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable


DEFAULT_KERNELS = ("add", "mm", "softmax", "layernorm")
DEFAULT_SHAPES = {
    "add": {"shape": [1024, 1024], "dtype": "float32"},
    "mm": {"m": 256, "n": 256, "k": 256, "dtype": "float32"},
    "softmax": {"shape": [128, 1024], "dim": -1, "dtype": "float32"},
    "layernorm": {
        "shape": [128, 1024],
        "normalized_shape": [1024],
        "dtype": "float32",
        "eps": 1.0e-5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FlagGems end-to-end compile time."
    )
    parser.add_argument("--backend", default="sophgo")
    parser.add_argument("--vendor", default=None)
    parser.add_argument("--flaggems-root", default=os.environ.get("FLAGGEMS_ROOT", "/workspace/FlagGems"))
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cache-root", default="/tmp/triton_anchor_compile_bench/cache")
    parser.add_argument("--dump-root", default="/tmp/triton_anchor_compile_bench/dump")
    parser.add_argument("--output-json", default="compile_benchmark_results.json")
    parser.add_argument("--output-csv", default="compile_benchmark_results.csv")
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    parser.add_argument("--atol", type=float, default=1.0e-2)
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--verbose-worker", action="store_true")

    # Internal worker mode.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-kernel", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-cache-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-dump-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-phase", default="repeat", help=argparse.SUPPRESS)

    return parser.parse_args()


def ensure_flaggems_path(flaggems_root: str) -> None:
    src = Path(flaggems_root) / "src"
    if not src.exists():
        raise FileNotFoundError(f"FlagGems src directory not found: {src}")
    sys.path.insert(0, str(src))


def import_backend(backend: str) -> None:
    if backend == "sophgo":
        plugin = __import__("triton_sophgo")
        from triton.backends import Backend, backends
        from triton.runtime.driver import driver

        # In some source-tree runs the Python module can be importable while
        # package entry_points are not visible to importlib.metadata.  Triton's
        # active driver discovery only looks at triton.backends.backends, so we
        # explicitly register the backend when entry_points did not do it.
        if "sophgo" not in backends:
            backends["sophgo"] = Backend(
                compiler=getattr(plugin, "compiler_cls"),
                driver=getattr(plugin, "driver_cls"),
            )
            driver.reset_active()
    else:
        # Other out-of-tree backends should be discoverable through entry_points
        # or imported by their environment setup before this script runs.
        return


def dtype_from_name(torch_mod: Any, name: str) -> Any:
    mapping = {
        "float32": torch_mod.float32,
        "fp32": torch_mod.float32,
        "float16": torch_mod.float16,
        "fp16": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
        "bf16": torch_mod.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def synchronize(torch_mod: Any, flag_gems_mod: Any) -> None:
    device = getattr(flag_gems_mod, "device", None)
    try:
        if device == "cuda":
            torch_mod.cuda.synchronize()
        elif device == "tpu" and hasattr(torch_mod, "tpu"):
            torch_mod.tpu.synchronize()
    except Exception:
        # Some emulator/device interfaces are synchronous or do not expose a
        # public synchronize method.  Timing still includes the Python call.
        pass


def to_device(cpu_tensor: Any, flag_gems_mod: Any) -> Any:
    return cpu_tensor.to(flag_gems_mod.device)


def build_case(kernel: str, torch_mod: Any, flag_gems_mod: Any) -> tuple[Callable[[], Any], Any, dict[str, Any]]:
    dtype_name = DEFAULT_SHAPES[kernel]["dtype"]
    dtype = dtype_from_name(torch_mod, dtype_name)

    if kernel == "add":
        shape = tuple(DEFAULT_SHAPES[kernel]["shape"])
        a_cpu = torch_mod.randn(shape, dtype=dtype)
        b_cpu = torch_mod.randn(shape, dtype=dtype)
        a = to_device(a_cpu, flag_gems_mod)
        b = to_device(b_cpu, flag_gems_mod)
        ref = torch_mod.add(a_cpu, b_cpu)

        def run() -> Any:
            with flag_gems_mod.use_gems():
                return torch_mod.add(a, b)

        spec = {"shape": list(shape), "dtype": dtype_name}
        return run, ref, spec

    if kernel == "mm":
        m = DEFAULT_SHAPES[kernel]["m"]
        n = DEFAULT_SHAPES[kernel]["n"]
        k = DEFAULT_SHAPES[kernel]["k"]
        a_cpu = torch_mod.randn((m, k), dtype=dtype)
        b_cpu = torch_mod.randn((k, n), dtype=dtype)
        a = to_device(a_cpu, flag_gems_mod)
        b = to_device(b_cpu, flag_gems_mod)
        ref = torch_mod.mm(a_cpu, b_cpu)

        def run() -> Any:
            with flag_gems_mod.use_gems():
                return torch_mod.mm(a, b)

        spec = {"a_shape": [m, k], "b_shape": [k, n], "dtype": dtype_name}
        return run, ref, spec

    if kernel == "softmax":
        shape = tuple(DEFAULT_SHAPES[kernel]["shape"])
        dim = DEFAULT_SHAPES[kernel]["dim"]
        x_cpu = torch_mod.randn(shape, dtype=dtype)
        x = to_device(x_cpu, flag_gems_mod)
        ref = torch_mod.softmax(x_cpu, dim=dim)

        def run() -> Any:
            with flag_gems_mod.use_gems():
                return torch_mod.softmax(x, dim=dim)

        spec = {"shape": list(shape), "dim": dim, "dtype": dtype_name}
        return run, ref, spec

    if kernel == "layernorm":
        shape = tuple(DEFAULT_SHAPES[kernel]["shape"])
        normalized_shape = tuple(DEFAULT_SHAPES[kernel]["normalized_shape"])
        eps = DEFAULT_SHAPES[kernel]["eps"]
        x_cpu = torch_mod.randn(shape, dtype=dtype)
        weight_cpu = torch_mod.randn(normalized_shape, dtype=dtype)
        bias_cpu = torch_mod.randn(normalized_shape, dtype=dtype)
        x = to_device(x_cpu, flag_gems_mod)
        weight = to_device(weight_cpu, flag_gems_mod)
        bias = to_device(bias_cpu, flag_gems_mod)
        ref = torch_mod.layer_norm(x_cpu, normalized_shape, weight_cpu, bias_cpu, eps)

        def run() -> Any:
            with flag_gems_mod.use_gems():
                return torch_mod.layer_norm(x, normalized_shape, weight, bias, eps)

        spec = {
            "shape": list(shape),
            "normalized_shape": list(normalized_shape),
            "dtype": dtype_name,
            "eps": eps,
        }
        return run, ref, spec

    raise ValueError(f"Unknown kernel: {kernel}")


def time_call(fn: Callable[[], Any], torch_mod: Any, flag_gems_mod: Any) -> tuple[float, Any]:
    start = time.perf_counter()
    out = fn()
    synchronize(torch_mod, flag_gems_mod)
    end = time.perf_counter()
    return (end - start) * 1000.0, out


def check_correctness(result: Any, reference: Any, torch_mod: Any, rtol: float, atol: float) -> tuple[bool, float]:
    result_cpu = result.detach().cpu()
    ref_cpu = reference.detach().cpu()
    ok = bool(torch_mod.allclose(result_cpu, ref_cpu, rtol=rtol, atol=atol))
    max_abs_diff = float((result_cpu - ref_cpu).abs().max().item())
    return ok, max_abs_diff


def run_worker(args: argparse.Namespace) -> int:
    if not args.worker_kernel or not args.worker_output:
        raise ValueError("Worker mode requires --worker-kernel and --worker-output")

    if args.worker_cache_dir:
        os.environ["TRITON_CACHE_DIR"] = args.worker_cache_dir
        Path(args.worker_cache_dir).mkdir(parents=True, exist_ok=True)
    if args.worker_dump_dir:
        os.environ["TRITON_DUMP_DIR"] = args.worker_dump_dir
        Path(args.worker_dump_dir).mkdir(parents=True, exist_ok=True)

    vendor = args.vendor or args.backend
    os.environ["GEMS_VENDOR"] = vendor
    os.environ["FLAGGEMS_ROOT"] = args.flaggems_root

    ensure_flaggems_path(args.flaggems_root)
    import_backend(args.backend)

    import torch
    import flag_gems

    torch.manual_seed(args.worker_seed)
    run, reference, spec = build_case(args.worker_kernel, torch, flag_gems)

    try:
        cold_ms, cold_out = time_call(run, torch, flag_gems)
        warm_ms, warm_out = time_call(run, torch, flag_gems)
        correctness_ok = True
        max_abs_diff = 0.0
        if not args.skip_correctness:
            correctness_ok, max_abs_diff = check_correctness(
                cold_out, reference, torch, args.rtol, args.atol
            )
        result = {
            "backend": args.backend,
            "vendor": vendor,
            "kernel": args.worker_kernel,
            "phase": args.worker_phase,
            "run_id": args.worker_run_id,
            "spec": spec,
            "cold_ms": cold_ms,
            "warm_ms": warm_ms,
            "compile_est_ms": cold_ms - warm_ms,
            "correctness_ok": correctness_ok,
            "max_abs_diff": max_abs_diff,
            "device": getattr(flag_gems, "device", None),
            "flag_gems_file": getattr(flag_gems, "__file__", None),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "triton_dump_dir": os.environ.get("TRITON_DUMP_DIR"),
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", None),
            "status": "pass" if correctness_ok else "fail",
        }
    except Exception as exc:
        result = {
            "backend": args.backend,
            "vendor": vendor,
            "kernel": args.worker_kernel,
            "phase": args.worker_phase,
            "run_id": args.worker_run_id,
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    output = Path(args.worker_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if result["status"] == "pass" else 1


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "stdev_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def run_child(args: argparse.Namespace, kernel: str, phase: str, run_index: int, work_root: Path) -> dict[str, Any]:
    cache_dir = work_root / "cache" / kernel / f"{phase}_{run_index}"
    dump_dir = work_root / "dump" / kernel / f"{phase}_{run_index}"
    result_file = work_root / "results" / kernel / f"{phase}_{run_index}.json"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
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

    print(f"[{kernel}] {phase} run {run_index}: starting")
    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=None if args.verbose_worker else subprocess.PIPE,
        stderr=None if args.verbose_worker else subprocess.STDOUT,
    )

    if not result_file.exists():
        if completed.stdout:
            print(completed.stdout)
        raise RuntimeError(f"Worker did not write result file: {result_file}")

    result = json.loads(result_file.read_text(encoding="utf-8"))
    if completed.returncode != 0 or result.get("status") != "pass":
        if completed.stdout:
            print(completed.stdout)
        raise RuntimeError(
            f"Worker failed for {kernel} {phase} {run_index}: "
            f"{result.get('error', result.get('status'))}"
        )

    print(
        f"[{kernel}] {phase} run {run_index}: "
        f"cold={result['cold_ms']:.3f} ms, "
        f"warm={result['warm_ms']:.3f} ms, "
        f"compile_est={result['compile_est_ms']:.3f} ms, "
        f"ok={result['correctness_ok']}"
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "backend",
        "vendor",
        "kernel",
        "run_id",
        "cold_ms",
        "warm_ms",
        "compile_est_ms",
        "correctness_ok",
        "max_abs_diff",
        "status",
        "spec",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "backend": row.get("backend"),
                    "vendor": row.get("vendor"),
                    "kernel": row.get("kernel"),
                    "run_id": row.get("run_id"),
                    "cold_ms": row.get("cold_ms"),
                    "warm_ms": row.get("warm_ms"),
                    "compile_est_ms": row.get("compile_est_ms"),
                    "correctness_ok": row.get("correctness_ok"),
                    "max_abs_diff": row.get("max_abs_diff"),
                    "status": row.get("status"),
                    "spec": json.dumps(row.get("spec", {}), sort_keys=True),
                }
            )


def run_parent(args: argparse.Namespace) -> int:
    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    unknown = [k for k in kernels if k not in DEFAULT_KERNELS]
    if unknown:
        raise ValueError(f"Unknown kernels: {unknown}. Supported: {DEFAULT_KERNELS}")

    flaggems_src = Path(args.flaggems_root) / "src"
    if not flaggems_src.exists():
        raise FileNotFoundError(f"FlagGems src directory not found: {flaggems_src}")

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    cache_root = Path(args.cache_root)
    dump_root = Path(args.dump_root)
    session_root = cache_root.parent / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    work_root = session_root
    cache_root = work_root / "cache"
    dump_root = work_root / "dump"
    cache_root.mkdir(parents=True, exist_ok=True)
    dump_root.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {args.backend}")
    print(f"FlagGems root: {args.flaggems_root}")
    print(f"Kernels: {', '.join(kernels)}")
    print(f"Repeat: {args.repeat}, warmup: {args.warmup}")
    print(f"Temporary work root: {work_root}")

    raw_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    try:
        for kernel in kernels:
            for warm_idx in range(args.warmup):
                try:
                    run_child(args, kernel, "warmup", warm_idx, work_root)
                except Exception as exc:
                    failures.append(f"{kernel} warmup {warm_idx}: {exc}")
                    raise

            for run_idx in range(args.repeat):
                result = run_child(args, kernel, "repeat", run_idx, work_root)
                raw_rows.append(result)

        summary: dict[str, Any] = {}
        for kernel in kernels:
            rows = [r for r in raw_rows if r["kernel"] == kernel]
            summary[kernel] = {
                "cold": summarize([r["cold_ms"] for r in rows]),
                "warm": summarize([r["warm_ms"] for r in rows]),
                "compile_est": summarize([r["compile_est_ms"] for r in rows]),
                "all_correct": all(bool(r["correctness_ok"]) for r in rows),
                "spec": rows[0].get("spec", {}) if rows else DEFAULT_SHAPES[kernel],
            }

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
                "cache_root": str(cache_root),
                "dump_root": str(dump_root),
                "env": {
                    "TRITON_CHIP_NAME": os.environ.get("TRITON_CHIP_NAME"),
                    "TRITON_TO_PPL_MODE": os.environ.get("TRITON_TO_PPL_MODE"),
                    "PPL_PROJECT_ROOT": os.environ.get("PPL_PROJECT_ROOT"),
                    "LLVM_BUILD_DIR": os.environ.get("LLVM_BUILD_DIR"),
                    "TRITON_DUMP_DIR": os.environ.get("TRITON_DUMP_DIR"),
                },
            },
            "summary": summary,
            "raw": raw_rows,
            "failures": failures,
        }

        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        write_csv(output_csv, raw_rows)

        print(f"Wrote JSON: {output_json}")
        print(f"Wrote CSV: {output_csv}")
        print("Summary:")
        for kernel in kernels:
            comp = summary[kernel]["compile_est"]
            print(
                f"  {kernel}: compile_est median={comp['median_ms']:.3f} ms, "
                f"mean={comp['mean_ms']:.3f} ms, stdev={comp['stdev_ms']:.3f} ms"
            )
        return 0
    finally:
        if not args.keep_workdirs:
            shutil.rmtree(work_root, ignore_errors=True)


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
