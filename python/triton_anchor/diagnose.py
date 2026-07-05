"""Command line interface for Triton Anchor pass diagnostics."""

from __future__ import annotations

import argparse
import importlib
import sys
from ast import literal_eval
from pathlib import Path
from typing import Any

from .diagnostics import PassDiagnostic, PassDiagnosticResult


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        mod = _load_input_module(args)
    except FileNotFoundError:
        print(f"error: input file not found: {args.input}", file=sys.stderr)
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
        choices=("ttir", "triton-linalg"),
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
    mod = ir.parse_mlir_module(str(input_path), ctx)
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
    mod = src.make_ir(options=_MinimalTritonOptions(), codegen_fns=None, context=ctx)
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


def _print_result(result: PassDiagnosticResult, *, quiet: bool = False) -> None:
    if result.ok:
        print(
            f"OK: pipeline {result.pipeline} completed, "
            f"{result.executed_passes}/{result.total_passes} passes executed."
        )
        if quiet:
            return
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
