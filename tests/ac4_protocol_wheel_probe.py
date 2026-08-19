"""Installed-wheel AC4 producer/consumer protocol evolution probe."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from dataclasses import replace


DISTRIBUTION = "triton-anchor-ac4-protocol-backend"
ENTRY_POINT = "ac4_protocol"
MODULE = "ac4_protocol_backend"


def _marker_path() -> Path:
    value = os.environ.get("TRITON_ANCHOR_AC4_MARKER_DIR")
    assert value, "TRITON_ANCHOR_AC4_MARKER_DIR is required"
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path / "diagnostics.called"


def _load_counter(entry_point):
    original = metadata.EntryPoint.load
    calls = {entry_point.name: 0}

    def counted_load(candidate):
        if candidate.name in calls:
            calls[candidate.name] += 1
        return original(candidate)

    metadata.EntryPoint.load = counted_load
    return original, calls


def _installed_manifest(distribution):
    from triton_anchor.backends import load_distribution_manifest

    document = load_distribution_manifest(distribution)
    assert document is not None
    plugin = document.get_by_entry_point(ENTRY_POINT)
    return plugin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumer-version", required=True)
    parser.add_argument("--producer-version", required=True)
    parser.add_argument(
        "--expect",
        choices=(
            "default",
            "ignore",
            "preserved",
            "deprecated",
            "removed",
            "mismatch",
            "same_major_removal",
        ),
        required=True,
    )
    parser.add_argument("--removed-protocol-field", default=None)
    args = parser.parse_args()
    assert not os.environ.get("PYTHONPATH")

    distribution = metadata.distribution(DISTRIBUTION)
    entry_point = next(
        ep
        for ep in distribution.entry_points
        if ep.group == "triton.backends" and ep.name == ENTRY_POINT
    )
    plugin = _installed_manifest(distribution)
    assert plugin.producer_protocol_version == args.producer_version
    assert entry_point.value == MODULE

    from triton_anchor.backends import (
        BackendPluginProtocolError,
        BackendPluginRegistry,
        PluginCompatibilityStatus,
        PluginLifecycleState,
        collect_core_environment,
    )

    base_environment = collect_core_environment()
    environment = replace(
        base_environment,
        backend_protocol_version=args.consumer_version,
    )
    original_load, load_calls = _load_counter(entry_point)
    marker = _marker_path()
    if marker.exists():
        marker.unlink()
    registry = BackendPluginRegistry(
        distribution_provider=lambda: (distribution,),
        environment_provider=lambda: environment,
        removed_protocol_fields=(
            (args.removed_protocol_field,)
            if args.removed_protocol_field
            else ()
        ),
    )
    error = None
    try:
        record = registry.discover()[0]
        if args.expect in {"mismatch", "same_major_removal"}:
            try:
                registry.validate(record.record_id)
            except BackendPluginProtocolError as caught:
                error = caught
            else:
                raise AssertionError("protocol rejection was accepted")
            rejected = registry.inspect(record.record_id)
            assert rejected.state is PluginLifecycleState.REJECTED
            assert (
                rejected.compatibility_status
                is PluginCompatibilityStatus.INCOMPATIBLE
            )
            payload = error.to_dict()
            assert payload["actual"] == args.producer_version
            if args.expect == "same_major_removal":
                assert payload["diagnostics"][0]["code"] == (
                    "backend_plugin_protocol_field_removal_forbidden"
                )
            assert load_calls[ENTRY_POINT] == 0
            assert MODULE not in sys.modules
            assert not marker.exists()
            result = {
                "case": args.expect,
                "consumer_protocol_version": args.consumer_version,
                "producer_protocol_version": args.producer_version,
                "entry_point_load_calls": load_calls,
                "error": payload,
                "module_imported": MODULE in sys.modules,
                "state": rejected.state.value,
            }
        else:
            validated = registry.validate(record.record_id)
            assert validated.state is PluginLifecycleState.VALIDATED
            registered = registry.register(record.record_id)
            assert registered.state is PluginLifecycleState.REGISTERED
            diagnostics = registry.diagnostics(record.record_id)
            assert diagnostics["protocol_field_diagnostics"] == (
                [] if args.expect != "deprecated" else diagnostics["protocol_field_diagnostics"]
            )
            if args.expect == "default":
                assert diagnostics["plugin_diagnostics"] == {}
                assert not marker.exists()
            elif args.expect == "ignore":
                assert diagnostics["plugin_diagnostics"] is None
                assert not marker.exists()
            elif args.expect == "preserved":
                assert diagnostics["plugin_diagnostics"] == {"healthy": True}
                assert not diagnostics["protocol_field_diagnostics"]
                assert marker.exists()
            elif args.expect == "deprecated":
                assert diagnostics["plugin_diagnostics"] == {"healthy": True}
                assert len(diagnostics["protocol_field_diagnostics"]) == 1
                warning = diagnostics["protocol_field_diagnostics"][0]
                assert warning["code"] == "backend_plugin_protocol_field_deprecated"
                assert warning["severity"] == "warning"
                assert marker.exists()
            elif args.expect == "removed":
                assert diagnostics["plugin_diagnostics"] is None
                assert not marker.exists()
            else:
                raise AssertionError(args.expect)
            assert load_calls[ENTRY_POINT] == 1
            result = {
                "case": args.expect,
                "consumer_protocol_version": args.consumer_version,
                "producer_protocol_version": args.producer_version,
                "entry_point_load_calls": load_calls,
                "error": None,
                "module_imported": MODULE in sys.modules,
                "plugin_diagnostics": diagnostics["plugin_diagnostics"],
                "protocol_field_diagnostics": diagnostics["protocol_field_diagnostics"],
                "state": registered.state.value,
            }
    finally:
        metadata.EntryPoint.load = original_load
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
