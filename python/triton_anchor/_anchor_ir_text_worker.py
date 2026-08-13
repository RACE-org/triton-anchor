"""Private native-parser worker used by the public crash-contained text API."""

from __future__ import annotations

import json
import sys

_PARSER_WORKER_PROTOCOL = "anchor-ir-text-worker/1.0.0"


def main() -> int:
    try:
        request = json.loads(
            sys.stdin.buffer.read().decode("utf-8", errors="strict")
        )
        if not isinstance(request, dict):
            raise TypeError("request must be a JSON object")
        if request.get("protocol_version") != _PARSER_WORKER_PROTOCOL:
            raise ValueError("unsupported parser worker protocol")
        action = request["action"]
        ir_text = request["ir_text"]
        policy = request["policy"]
        source_name = request["source_name"]
        if action not in {"validate", "normalize"}:
            raise ValueError("unsupported action")
        if not isinstance(ir_text, str) or not isinstance(policy, dict):
            raise TypeError("invalid worker request fields")
        if not isinstance(source_name, str):
            raise TypeError("source_name must be a string")

        from triton._C.libtriton import anchor, ir

        context = ir.context()
        ir.load_dialects(context)
        anchor.load_dialects(context)
        if action == "validate":
            result = anchor.validate_anchor_ir_text(
                ir_text,
                context,
                policy,
                source_name,
            )
        else:
            result = anchor.normalize_anchor_ir_text(
                ir_text,
                context,
                policy,
                source_name,
            )
        response = {
            "protocol_version": _PARSER_WORKER_PROTOCOL,
            "result": result,
        }
    except Exception as error:  # noqa: BLE001 - serialize worker failures.
        response = {
            "protocol_version": _PARSER_WORKER_PROTOCOL,
            "worker_error": f"{type(error).__name__}: {error}",
        }

    encoded_response = (
        json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8", errors="strict")
    sys.stdout.buffer.write(encoded_response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
