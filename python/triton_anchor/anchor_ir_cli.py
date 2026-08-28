"""Command-line interface for structured AnchorIR validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import _anchor_ir_text_isolation as text_isolation
from .anchor_ir_rules import validate_policy_request
from .anchor_ir_schema import (
    AnchorIRValidationReport,
    format_anchor_ir_validation_report,
)
from .anchor_ir_validator import StructuredAnchorIRValidator

EXIT_VALID = 0
EXIT_INVALID = 1
EXIT_USAGE = 2


class _AnchorIRInputTooLargeError(ValueError):
    """Internal signal used to preserve structured CLI/API diagnostics."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triton-anchor-validate",
        description="Validate MLIR against a versioned AnchorIR contract.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="MLIR input file, or '-' / omitted to read stdin.",
    )
    parser.add_argument(
        "--spec-version",
        default=None,
        help="Full AnchorIR specification version, for example anchor-ir/1.1.0.",
    )
    parser.add_argument(
        "--track",
        default=None,
        metavar="TRACK",
        help="AnchorIR output Track: linalg or triton_gpu.",
    )
    parser.add_argument(
        "--phase",
        default=None,
        metavar="PHASE",
        help="Validation phase around the backend Hook: pre_hook or post_hook.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--allow-extension",
        action="append",
        default=[],
        metavar="DIALECT",
        help=(
            "Declare one backend extension dialect for post_hook validation; "
            "repeat for multiple dialects."
        ),
    )
    return parser


def _read_limited_bytes(stream) -> bytes:
    data = stream.read(text_isolation.MAX_ANCHOR_IR_TEXT_BYTES + 1)
    if len(data) > text_isolation.MAX_ANCHOR_IR_TEXT_BYTES:
        raise _AnchorIRInputTooLargeError()
    return data


def _read_limited_text(stream) -> str:
    chunks = []
    byte_count = 0
    while True:
        chunk = stream.read(64 * 1024)
        if not chunk:
            return "".join(chunks)
        byte_count += len(chunk.encode("utf-8", errors="strict"))
        if byte_count > text_isolation.MAX_ANCHOR_IR_TEXT_BYTES:
            raise _AnchorIRInputTooLargeError()
        chunks.append(chunk)


def _read_input(input_name: str) -> tuple[str, str]:
    if input_name == "-":
        stream = getattr(sys.stdin, "buffer", None)
        if stream is None:
            return _read_limited_text(sys.stdin), "<stdin>"
        return _read_limited_bytes(stream).decode("utf-8", errors="strict"), "<stdin>"
    path = Path(input_name)
    with path.open("rb") as stream:
        return _read_limited_bytes(stream).decode("utf-8", errors="strict"), str(path)


def _input_limit_report(args) -> AnchorIRValidationReport:
    request = validate_policy_request(
        spec_version=args.spec_version,
        track=args.track,
        phase=args.phase,
    )
    if not request.valid:
        return request
    policy = StructuredAnchorIRValidator._resolve_policy(
        spec_version=args.spec_version,
        track=args.track,
        phase=args.phase,
        extension_dialects=args.allow_extension,
    )
    raw_report = text_isolation.text_input_limit_report(
        policy.to_dict(),
        text_isolation.MAX_ANCHOR_IR_TEXT_BYTES + 1,
    )
    if raw_report is None:
        raise AssertionError("AnchorIR input-limit report was unexpectedly absent")
    return AnchorIRValidationReport.from_dict(raw_report)


def _write_output(output: str) -> None:
    encoded = (output + "\n").encode("utf-8", errors="strict")
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        sys.stdout.write(encoded.decode("utf-8", errors="strict"))
        return
    stream.write(encoded)


def format_text_report(report: AnchorIRValidationReport) -> str:
    """Render a deterministic human-readable view of a structured Report."""

    return format_anchor_ir_validation_report(report)


def _format_json_report(report: AnchorIRValidationReport) -> str:
    return json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI and return its stable process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        ir_text, source_name = _read_input(args.input)
    except _AnchorIRInputTooLargeError:
        try:
            report = _input_limit_report(args)
        except (TypeError, ValueError, RuntimeError) as error:
            parser.exit(EXIT_USAGE, "%s: error: %s\n" % (parser.prog, error))
    except (OSError, UnicodeError) as error:
        parser.error("cannot read input %r: %s" % (args.input, error))
    else:
        try:
            report = StructuredAnchorIRValidator().validate_text(
                ir_text,
                spec_version=args.spec_version,
                track=args.track,
                phase=args.phase,
                source_name=source_name,
                extension_dialects=args.allow_extension,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            parser.exit(EXIT_USAGE, "%s: error: %s\n" % (parser.prog, error))
    if args.format == "json":
        output = _format_json_report(report)
    else:
        output = format_text_report(report)
    _write_output(output)
    return EXIT_VALID if report.valid else EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
