"""Crash containment for untrusted text parsed by the pinned MLIR build."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import threading
import weakref
from contextlib import contextmanager
from typing import Any, Optional

_PARSER_WORKER_TIMEOUT_SECONDS = 60
_PARSER_WORKER_PROTOCOL = "anchor-ir-text-worker/1.0.0"
_WORKER_ERROR_DETAIL_LIMIT = 512
# Text is accepted from both library callers and the CLI.  Keep one hard,
# version-independent byte ceiling before serializing a worker request or
# handing a string to an in-process MLIR parser.
MAX_ANCHOR_IR_TEXT_BYTES = 16 * 1024 * 1024
# ``source_name`` is copied into every source-location diagnostic. Bound it
# independently so tiny IR cannot multiply attacker-controlled metadata.
MAX_ANCHOR_IR_SOURCE_NAME_BYTES = 64 * 1024

_EXPLICIT_CONTEXT_LOCKS_GUARD = threading.Lock()
_EXPLICIT_CONTEXT_LOCKS = weakref.WeakKeyDictionary()


@contextmanager
def lock_explicit_anchor_ir_context(context):
    """Serialize dialect loading and native text work for one context."""

    if context is None:
        raise TypeError("an explicit MLIR context is required")
    with _EXPLICIT_CONTEXT_LOCKS_GUARD:
        context_lock = _EXPLICIT_CONTEXT_LOCKS.get(context)
        if context_lock is None:
            context_lock = threading.RLock()
            _EXPLICIT_CONTEXT_LOCKS[context] = context_lock
    with context_lock:
        yield


# These spellings enter custom parsers with known unchecked casts or indexing
# in the pinned Triton/MLIR revision.  The normal public path isolates every
# default-context text request; this detector additionally protects the
# explicit-context compatibility path for the known crash classes.
_UNSAFE_DENSE_CUSTOM_ELEMENT = re.compile(
    r"""
    (?<![A-Za-z0-9_.$#!@%^])
    dense(?![A-Za-z0-9_.$])\s*<\s*
    (?!")
    (?=(?:true|false|[-+]?(?:\d|\.\d)|\[|\())
    .*?
    >\s*:\s*
    (?:tensor|vector)\s*<
    [^,\n>]*x\s*!
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)
_UNSAFE_TRITON_GPU_CUSTOM_PARSER = re.compile(
    r"\#triton_gpu\.(?:slice|amd_mfma)\s*<",
    re.IGNORECASE,
)


def requires_parser_isolation(ir_text: str) -> bool:
    """Return whether parsing must be protected from a native assertion."""

    # The unsafe spelling can also occur as ordinary user data in a StringAttr
    # or a ``//`` comment.  Mask those lexical regions before applying the
    # conservative grammar detector, otherwise valid IR is rejected merely
    # because its text mentions an invalid dense literal.
    masked = list(ir_text)
    in_string = False
    escaped = False
    in_line_comment = False
    index = 0
    while index < len(ir_text):
        character = ir_text[index]
        if in_line_comment:
            if character in "\r\n":
                in_line_comment = False
            else:
                masked[index] = " "
            index += 1
            continue
        if in_string:
            if character not in "\r\n":
                masked[index] = " "
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            masked[index] = " "
            index += 1
            continue
        if character == "/" and index + 1 < len(ir_text) and ir_text[index + 1] == "/":
            in_line_comment = True
            masked[index] = " "
            masked[index + 1] = " "
            index += 2
            continue
        index += 1

    masked_text = "".join(masked)
    return (
        _UNSAFE_DENSE_CUSTOM_ELEMENT.search(masked_text) is not None
        or _UNSAFE_TRITON_GPU_CUSTOM_PARSER.search(masked_text) is not None
    )


def utf8_byte_length(value: str, field_name: str) -> int:
    """Validate public text and return its exact UTF-8 byte length."""

    if not isinstance(value, str):
        raise TypeError("%s must be a str" % field_name)
    try:
        return len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise ValueError(
            "%s must be valid Unicode encodable as UTF-8" % field_name
        ) from error


def require_utf8_text(value: str, field_name: str) -> None:
    """Fail before native bindings for text outside the public UTF-8 contract."""

    utf8_byte_length(value, field_name)


def validate_source_name(value: str) -> None:
    """Validate and bound source metadata before worker or native parsing."""

    utf8_length = utf8_byte_length(value, "source_name")
    if utf8_length > MAX_ANCHOR_IR_SOURCE_NAME_BYTES:
        raise ValueError(
            "source_name exceeds the %d-byte UTF-8 limit"
            % MAX_ANCHOR_IR_SOURCE_NAME_BYTES
        )


def _parser_crash_report(
    policy: dict[str, Any],
    detail: str = "isolated MLIR parser terminated unexpectedly",
) -> dict[str, Any]:
    diagnostic = policy["parse_failure_diagnostic"]
    return {
        "valid": False,
        "spec_version": policy["spec_version"],
        "track": policy["track"],
        "phase": policy["phase"],
        "diagnostics": [
            {
                "code": diagnostic["code"],
                "severity": "error",
                "message": (diagnostic["message"] + ": " + detail),
                "hint": diagnostic["hint"],
                "spec_version": policy["spec_version"],
                "track": policy["track"],
                "phase": policy["phase"],
                "object_kind": "module",
                "object_name": "builtin.module",
                "operation_path": "",
                "object_path": "",
                "location": None,
            }
        ],
    }


def text_input_limit_report(
    policy: dict[str, Any],
    utf8_length: int,
) -> Optional[dict[str, Any]]:
    """Return the structured failure for a text request above the hard cap."""

    if utf8_length <= MAX_ANCHOR_IR_TEXT_BYTES:
        return None
    return _parser_crash_report(
        policy,
        "AnchorIR text exceeds the %d-byte input limit" % MAX_ANCHOR_IR_TEXT_BYTES,
    )


def _limited_result(
    action: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    if action == "normalize":
        return {
            "validation_report": report,
            "normalized_text": None,
        }
    return report


def _summarize_worker_error(stderr: str) -> str:
    """Return one bounded, terminal-safe line from worker stderr."""

    printable = "".join(
        character if character.isprintable() else " " for character in stderr
    )
    compact = " ".join(printable.split())
    if len(compact) > _WORKER_ERROR_DETAIL_LIMIT:
        return compact[:_WORKER_ERROR_DETAIL_LIMIT] + "..."
    return compact


def run_isolated_native_text(
    action: str,
    ir_text: str,
    policy: dict[str, Any],
    source_name: str,
    *,
    timeout_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """Run one native text request and convert a process abort to a Report."""

    if action not in {"validate", "normalize"}:
        raise ValueError(f"unsupported isolated AnchorIR text action {action!r}")
    worker_timeout = (
        _PARSER_WORKER_TIMEOUT_SECONDS
        if timeout_seconds is None
        else timeout_seconds
    )
    if (
        isinstance(worker_timeout, bool)
        or not isinstance(worker_timeout, (int, float))
    ):
        raise TypeError("timeout_seconds must be a positive finite number")
    worker_timeout = float(worker_timeout)
    if not math.isfinite(worker_timeout) or worker_timeout <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    ir_text_length = utf8_byte_length(ir_text, "ir_text")
    validate_source_name(source_name)
    limit_report = text_input_limit_report(policy, ir_text_length)
    if limit_report is not None:
        return _limited_result(action, limit_report)
    request = {
        "protocol_version": _PARSER_WORKER_PROTOCOL,
        "action": action,
        "ir_text": ir_text,
        "policy": policy,
        "source_name": source_name,
    }
    encoded_request = json.dumps(request, ensure_ascii=False)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "triton_anchor._anchor_ir_text_worker"],
            input=encoded_request,
            text=True,
            encoding="utf-8",
            errors="strict",
            capture_output=True,
            check=False,
            timeout=worker_timeout,
        )
    except subprocess.TimeoutExpired:
        report = _parser_crash_report(
            policy,
            "isolated MLIR parser exceeded the %g-second safety timeout"
            % worker_timeout,
        )
        return _limited_result(action, report)
    except OSError as error:
        # Process-creation failures are infrastructure errors, not malformed
        # user IR.  Let the API raise and let the CLI map RuntimeError to its
        # documented usage/infrastructure exit code 2.
        raise RuntimeError("failed to start isolated AnchorIR parser") from error
    except UnicodeError as error:
        raise RuntimeError(
            "isolated AnchorIR parser violated the UTF-8 worker protocol"
        ) from error
    if completed.returncode < 0:
        report = _parser_crash_report(
            policy,
            "isolated MLIR parser terminated by signal %d" % (-completed.returncode),
        )
        return _limited_result(action, report)
    if completed.returncode > 0:
        detail = _summarize_worker_error(completed.stderr)
        message = "isolated AnchorIR parser exited with status %d" % (
            completed.returncode,
        )
        if detail:
            message += ": " + detail
        raise RuntimeError(message)

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("isolated AnchorIR parser returned invalid JSON") from error
    if not isinstance(response, dict):
        raise TypeError("isolated AnchorIR parser returned an invalid response")
    if response.get("protocol_version") != _PARSER_WORKER_PROTOCOL:
        raise RuntimeError(
            "isolated AnchorIR parser returned an incompatible protocol version"
        )
    if "worker_error" in response:
        raise RuntimeError(
            f"isolated AnchorIR parser failed: {response['worker_error']}"
        )
    if set(response) != {"protocol_version", "result"} or not isinstance(
        response["result"], dict
    ):
        raise RuntimeError("isolated AnchorIR parser returned an invalid result")
    return response["result"]
