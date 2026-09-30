"""Command line interface for Triton Anchor pass diagnostics."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import traceback
from ast import literal_eval
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .diagnostics import (
    MLIRDiagnosticLocation,
    PassDiagnostic,
    PassDiagnosticResult,
    extract_mlir_location,
)


@dataclass
class InputDiagnosticResult:
    """Summary for failures that happen before any compiler pass runs."""

    ok: bool
    stage: str
    pipeline: str
    output_dir: Path
    diagnostic_path: Path
    error: str
    input_path: Path | None = None
    python_target: str | None = None
    diagnostic_text: str | None = None
    mlir_location: MLIRDiagnosticLocation | None = None
    traceback: str | None = None
    summary_path: Path | None = None


class InputDiagnosticError(Exception):
    """A diagnosable input/pre-pass failure."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        diagnostic_text: str = "",
        input_path: Path | None = None,
        python_target: str | None = None,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.diagnostic_text = diagnostic_text
        self.input_path = input_path
        self.python_target = python_target
        self.original = original


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        mod = _load_input_module(args)
    except FileNotFoundError:
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2
    except InputDiagnosticError as exc:
        result = _write_input_diagnostic(exc, args)
        _print_input_diagnostic(result, quiet=args.quiet)
        return 2
    except ImportError as exc:
        print(
            "error: triton._C.libtriton is not available; build or install the "
            "Triton Anchor C++ extension before running pass diagnostics.",
            file=sys.stderr,
        )
        print(f"detail: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: failed to prepare input module: {exc}", file=sys.stderr)
        return 2

    diagnostic = PassDiagnostic(
        output_dir=args.output_dir,
        enable_debug=not args.no_debug,
        save_success_snapshots=not args.no_success_snapshots,
        write_summary_json=not args.no_summary_json,
    )

    try:
        if args.pipeline == "ttir":
            result = diagnostic.diagnose_ttir(mod)
        elif args.pipeline == "triton-linalg":
            result = diagnostic.diagnose_triton_linalg(mod)
        elif args.pipeline == "sophgo-pplir":
            result = diagnostic.diagnose_sophgo_pplir(mod)
        else:
            print(f"error: unsupported pipeline: {args.pipeline}", file=sys.stderr)
            return 2
    except ImportError as exc:
        print(
            "error: triton._C.libtriton is not available; build or install the "
            "Triton Anchor C++ extension before running pass diagnostics.",
            file=sys.stderr,
        )
        print(f"detail: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: failed to run diagnostics: {exc}", file=sys.stderr)
        return 1

    _print_result(result, quiet=args.quiet)
    return 0 if result.ok else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triton-anchor-diagnose",
        description=(
            "Run Triton Anchor compiler pipelines one pass at a time and "
            "report the first failing pass."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Input MLIR/TTIR file to diagnose.",
    )
    parser.add_argument(
        "--python",
        dest="python_target",
        metavar="MODULE:FUNCTION",
        help=(
            "Generate TTIR in memory from a Python @triton.jit function and "
            "diagnose it without requiring a pre-dumped IR file."
        ),
    )
    parser.add_argument(
        "--signature",
        default=None,
        help=(
            "Kernel signature for --python mode. Use comma-separated argument "
            "types, e.g. '*fp32,*fp32,*fp32,i32'."
        ),
    )
    parser.add_argument(
        "--constant",
        action="append",
        default=[],
        metavar="NAME_OR_INDEX=VALUE",
        help=(
            "Compile-time constant for --python mode. Can be repeated. "
            "Example: --constant BLOCK=256 or --constant 4=256."
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=("ttir", "triton-linalg", "sophgo-pplir"),
        default="ttir",
        help="Compiler pipeline to diagnose. Default: ttir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for IR snapshots and summary.json.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not enable MLIR PassManager debug mode.",
    )
    parser.add_argument(
        "--no-success-snapshots",
        action="store_true",
        help="Only save before snapshots, not after snapshots for successful passes.",
    )
    parser.add_argument(
        "--no-summary-json",
        action="store_true",
        help="Do not write summary.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the final status line.",
    )
    return parser


def _load_input_module(args):
    if args.python_target:
        if args.input is not None:
            raise ValueError("file input and --python are mutually exclusive")
        if args.pipeline != "ttir":
            raise ValueError("--python mode currently supports --pipeline ttir only")
        return _make_ttir_from_python(
            args.python_target,
            signature=args.signature,
            constants=args.constant,
        )

    if args.input is None:
        raise ValueError("provide an input IR file or --python MODULE:FUNCTION")
    return _load_mlir_module(args.input)


def _load_mlir_module(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    from triton._C.libtriton import anchor, ir

    ctx = ir.context()
    ir.load_dialects(ctx)
    anchor.load_dialects(ctx)
    try:
        mod, _ = _capture_stderr(lambda: ir.parse_mlir_module(str(input_path), ctx))
    except _CapturedInputError as exc:
        original = exc.original
        raise InputDiagnosticError(
            "input-parse",
            str(original),
            diagnostic_text=exc.diagnostic_text,
            input_path=input_path,
            original=original,
        ) from original
    mod.context = ctx
    return mod


def _make_ttir_from_python(
    python_target: str,
    *,
    signature: str | None,
    constants: list[str],
):
    if ":" not in python_target:
        raise ValueError("--python must be in MODULE:FUNCTION format")
    if not signature:
        raise ValueError("--signature is required when using --python")

    import triton
    from triton._C.libtriton import anchor, ir

    try:
        module_name, function_name = python_target.split(":", 1)
        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)

        parsed_signature = _parse_signature(signature)
        parsed_constants = _parse_constants(constants, fn=fn)

        ctx = ir.context()
        ir.load_dialects(ctx)
        anchor.load_dialects(ctx)
        src = triton.compiler.ASTSource(
            fn=fn,
            signature=parsed_signature,
            constants=parsed_constants,
        )
        mod, _ = _capture_stderr(
            lambda: src.make_ir(
                options=_MinimalTritonOptions(),
                codegen_fns=None,
                context=ctx,
            )
        )
    except _CapturedInputError as exc:
        original = exc.original
        raise InputDiagnosticError(
            "python-frontend",
            str(original),
            diagnostic_text=exc.diagnostic_text,
            python_target=python_target,
            original=original,
        ) from original
    except Exception as exc:
        raise InputDiagnosticError(
            "python-frontend",
            str(exc),
            python_target=python_target,
            original=exc,
        ) from exc
    mod.context = ctx
    return mod


def _parse_signature(signature: str) -> dict[int, str]:
    values = [item.strip() for item in signature.split(",")]
    values = [item for item in values if item]
    if not values:
        raise ValueError("--signature cannot be empty")
    return {index: value for index, value in enumerate(values)}


def _parse_constants(constants: list[str], *, fn: Any) -> dict[int, Any]:
    parsed: dict[int, Any] = {}
    arg_names = _get_function_arg_names(fn)
    for item in constants:
        if "=" not in item:
            raise ValueError(f"invalid --constant value: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --constant key: {item}")
        value = _parse_constant_value(raw_value.strip())
        index = _constant_key_to_index(key, arg_names)
        parsed[index] = value
    return parsed


def _get_function_arg_names(fn: Any) -> list[str]:
    wrapped = getattr(fn, "fn", fn)
    code = getattr(wrapped, "__code__", None)
    if code is None:
        return []
    return list(code.co_varnames[: code.co_argcount])


def _constant_key_to_index(key: str, arg_names: list[str]) -> int:
    if key.isdigit():
        return int(key)
    if key in arg_names:
        return arg_names.index(key)
    raise ValueError(
        f"constant key '{key}' is not an argument name or index; "
        f"available args: {', '.join(arg_names) if arg_names else '<unknown>'}"
    )


def _parse_constant_value(value: str) -> Any:
    try:
        return literal_eval(value)
    except (SyntaxError, ValueError):
        return value


class _MinimalTritonOptions:
    def __init__(self) -> None:
        self.num_warps = 4
        self.num_stages = 3
        self.num_ctas = 1
        self.cluster_dims = (1, 1, 1)
        self.ptx_version = None
        self.enable_fp_fusion = True
        self.supported_fp8_dtypes = ()
        self.deprecated_fp8_dtypes = ()
        self.allowed_dot_input_precisions = ("ieee", "tf32", "tf32x3")
        self.allow_fp8e4nv = False
        self.max_num_imprecise_acc_default = False
        self.debug = False


class _CapturedInputError(Exception):
    def __init__(self, original: Exception, diagnostic_text: str) -> None:
        super().__init__(str(original))
        self.original = original
        self.diagnostic_text = diagnostic_text


def _capture_stderr(fn):
    saved_stderr_fd = os.dup(2)
    original: Exception | None = None
    result = None
    with tempfile.TemporaryFile(mode="w+b") as captured:
        try:
            os.dup2(captured.fileno(), 2)
            try:
                result = fn()
            except Exception as exc:
                original = exc
            finally:
                os.dup2(saved_stderr_fd, 2)
        finally:
            os.close(saved_stderr_fd)

        captured.seek(0)
        diagnostic_text = captured.read().decode("utf-8", errors="replace")

    if original is not None:
        raise _CapturedInputError(original, diagnostic_text) from original
    return result, diagnostic_text


def _write_input_diagnostic(
    exc: InputDiagnosticError,
    args: argparse.Namespace,
) -> InputDiagnosticResult:
    output_dir = Path(
        args.output_dir or tempfile.mkdtemp(prefix="triton-anchor-diagnose-")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    original = exc.original or exc
    tb = "".join(
        traceback.format_exception(type(original), original, original.__traceback__)
    )
    input_text, location_path = _read_input_text_for_diagnostic(exc.input_path)
    mlir_location = extract_mlir_location(
        exc.message,
        exc.diagnostic_text,
        input_text,
        location_path,
    )

    diagnostic_path = output_dir / f"{exc.stage}.diagnostic.txt"
    result = InputDiagnosticResult(
        ok=False,
        stage=exc.stage,
        pipeline=args.pipeline,
        output_dir=output_dir,
        diagnostic_path=diagnostic_path,
        error=exc.message,
        input_path=exc.input_path,
        python_target=exc.python_target,
        diagnostic_text=exc.diagnostic_text or None,
        mlir_location=mlir_location,
        traceback=tb,
    )
    _write_input_diagnostic_file(result)
    if not args.no_summary_json:
        summary_path = output_dir / "summary.json"
        result.summary_path = summary_path
        summary_path.write_text(
            json.dumps(_to_jsonable(asdict(result)), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result


def _read_input_text_for_diagnostic(input_path: Path | None) -> tuple[str, Path]:
    if input_path is None:
        return "", Path("<unknown>")
    try:
        return input_path.read_text(encoding="utf-8"), input_path
    except OSError:
        return "", input_path


def _write_input_diagnostic_file(result: InputDiagnosticResult) -> None:
    parts = [
        f"stage: {result.stage}",
        f"pipeline: {result.pipeline}",
        f"error: {result.error}",
    ]
    if result.input_path is not None:
        parts.append(f"input: {result.input_path}")
    if result.python_target is not None:
        parts.append(f"python_target: {result.python_target}")
    if result.mlir_location is not None:
        location = result.mlir_location
        parts.extend(
            [
                "location:",
                f"  raw: {location.raw}",
                f"  file: {location.file}",
                f"  line: {location.line}",
                f"  column: {location.column}",
                f"  operation: {location.operation}",
                f"  ir_line: {location.ir_line}",
                f"  ir_snippet: {location.ir_snippet}",
            ]
        )
    if result.diagnostic_text:
        parts.extend(["captured diagnostics:", result.diagnostic_text.rstrip()])
    if result.traceback:
        parts.extend(["traceback:", result.traceback.rstrip()])
    result.diagnostic_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _print_input_diagnostic(
    result: InputDiagnosticResult,
    *,
    quiet: bool = False,
) -> None:
    if result.stage == "python-frontend":
        print("FAILED: python-frontend failed before TTIR generation.")
    elif result.stage == "input-parse":
        print("FAILED: input-parse failed before pass diagnostics.")
    else:
        print(f"FAILED: {result.stage} failed before pass diagnostics.")
    if quiet:
        return

    if result.input_path is not None:
        print(f"input: {result.input_path}")
    if result.python_target is not None:
        print(f"python target: {result.python_target}")
    print(f"diagnostic detail: {result.diagnostic_path}")
    if result.mlir_location is not None:
        location = result.mlir_location
        if location.line is not None and location.column is not None:
            if location.file:
                print(f"location: {location.file}:{location.line}:{location.column}")
            else:
                print(f"location: {location.line}:{location.column}")
        elif location.raw and location.raw != location.operation:
            print(f"location: {location.raw}")
        else:
            print("location: <not available>")
        if location.operation:
            print(f"operation: {location.operation}")
        if location.ir_line is not None:
            print(f"ir line: {location.ir_line}: {location.ir_snippet}")
    if result.error:
        print(f"error: {result.error}")
    print(f"diagnostic output: {result.output_dir}")
    if result.summary_path is not None:
        print(f"summary: {result.summary_path}")


def _print_result(result: PassDiagnosticResult, *, quiet: bool = False) -> None:
    if result.ok:
        print(
            f"OK: pipeline {result.pipeline} completed, "
            f"{result.executed_passes}/{result.total_passes} passes executed."
        )
        if quiet:
            return
        print(f"total duration: {result.total_duration_ms:.2f} ms")
        print(f"input IR: {result.input_ir_bytes} bytes")
        print(f"output IR: {result.output_ir_bytes} bytes")
        if result.peak_rss_bytes > 0:
            print(f"peak RSS: {result.peak_rss_bytes / (1024 * 1024):.2f} MB")
        slowest = result.slowest_pass
        if slowest is not None:
            print(
                f"slowest pass: #{slowest.index} {slowest.name} "
                f"({slowest.duration_ms:.2f} ms)"
            )
        print(f"diagnostic output: {result.output_dir}")
        if result.summary_path is not None:
            print(f"summary: {result.summary_path}")
        return

    print(
        f"FAILED: pipeline {result.pipeline} failed at pass "
        f"{result.failed_index}/{result.total_passes}: {result.failed_pass}"
    )
    if quiet:
        return

    failed = result.failed_record
    if failed is not None:
        print(f"before IR: {failed.before_ir}")
        if failed.diagnostic_path is not None:
            print(f"diagnostic detail: {failed.diagnostic_path}")
        print(f"pass duration: {failed.duration_ms:.2f} ms")
        print(f"before IR size: {failed.before_ir_bytes} bytes")
    print(f"total duration (up to failure): {result.total_duration_ms:.2f} ms")
    if result.peak_rss_bytes > 0:
        print(f"peak RSS: {result.peak_rss_bytes / (1024 * 1024):.2f} MB")
    if result.mlir_location is not None:
        location = result.mlir_location
        if location.line is not None and location.column is not None:
            if location.file:
                print(f"location: {location.file}:{location.line}:{location.column}")
            else:
                print(f"location: {location.line}:{location.column}")
        elif location.raw and location.raw != location.operation:
            print(f"location: {location.raw}")
        else:
            print("location: <not available>")
        if location.operation:
            print(f"operation: {location.operation}")
        if location.ir_line is not None:
            print(f"ir line: {location.ir_line}: {location.ir_snippet}")
    if result.error:
        print(f"error: {result.error}")
    print(f"diagnostic output: {result.output_dir}")
    if result.summary_path is not None:
        print(f"summary: {result.summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
