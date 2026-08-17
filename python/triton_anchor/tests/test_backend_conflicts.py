from types import SimpleNamespace

import pytest

from triton_anchor.backends.conflicts import (
    ConflictKind,
    ConflictSeverity,
    detect_conflicts,
    detect_static_conflicts,
)
from triton_anchor.backends.errors import BackendPluginConflictError
from triton_anchor.backends.protocol import (
    PluginIsolationMode,
    PluginLifecycleState,
)


def _record(
    record_id,
    *,
    plugin_id=None,
    entry_point=None,
    targets=(),
    state=PluginLifecycleState.DISCOVERED,
    native_libraries=(),
    abi_fingerprint=None,
    native_artifacts=(),
):
    manifest = None
    if plugin_id is not None:
        manifest = SimpleNamespace(
            targets=targets,
            native_libraries=native_libraries,
            abi_fingerprint=abi_fingerprint,
            isolation_mode=(
                PluginIsolationMode.NATIVE_IN_PROCESS
                if native_libraries
                else PluginIsolationMode.PYTHON_ONLY
            ),
        )
    return SimpleNamespace(
        record_id=record_id,
        plugin_id=plugin_id,
        entry_point_name=entry_point,
        manifest=manifest,
        state=state,
        compatibility_report=(
            SimpleNamespace(native_artifacts=native_artifacts)
            if native_artifacts
            else None
        ),
    )


def _by_kind(report):
    return {conflict.kind: conflict for conflict in report.conflicts}


def test_distinct_plugin_identity_and_targets_have_no_conflicts():
    report = detect_conflicts(
        [
            _record(
                "record-b",
                plugin_id="vendor.b",
                entry_point="backend_b",
                targets=("target-b",),
            ),
            _record(
                "record-a",
                plugin_id="vendor.a",
                entry_point="backend_a",
                targets=("target-a",),
            ),
        ]
    )

    assert report.conflicts == ()
    assert report.ok
    assert not report.has_fatal
    assert not report.requires_selection
    assert report.to_dict() == {
        "ok": True,
        "has_fatal": False,
        "requires_selection": False,
        "conflicts": [],
    }


def test_duplicate_plugin_id_is_fatal_and_structured():
    report = detect_conflicts(
        [
            _record(
                "z-record",
                plugin_id="vendor.same",
                entry_point="backend_z",
            ),
            _record(
                "a-record",
                plugin_id="vendor.same",
                entry_point="backend_a",
            ),
        ]
    )
    conflict = report.conflicts[0]

    assert conflict.kind is ConflictKind.DUPLICATE_PLUGIN_ID
    assert conflict.severity is ConflictSeverity.FATAL
    assert conflict.claim == "vendor.same"
    assert conflict.record_ids == ("a-record", "z-record")
    assert conflict.plugin_ids == ("vendor.same",)
    assert conflict.entry_point_names == ("backend_a", "backend_z")
    assert report.has_fatal
    assert not report.ok

    with pytest.raises(BackendPluginConflictError) as raised:
        report.raise_for_fatal()
    assert raised.value.field == "plugin_id"
    assert raised.value.actual == "a-record, z-record"


def test_duplicate_entry_point_name_is_fatal():
    report = detect_conflicts(
        [
            _record(
                "record-a",
                plugin_id="vendor.a",
                entry_point="same_backend",
            ),
            _record(
                "record-b",
                plugin_id="vendor.b",
                entry_point="same_backend",
            ),
        ]
    )
    conflict = report.conflicts[0]

    assert conflict.kind is ConflictKind.DUPLICATE_ENTRY_POINT
    assert conflict.severity is ConflictSeverity.FATAL
    assert conflict.claim == "same_backend"
    assert conflict.record_ids == ("record-a", "record-b")


def test_target_overlap_requires_selection_but_is_not_fatal():
    report = detect_conflicts(
        [
            _record(
                "record-a",
                plugin_id="vendor.a",
                entry_point="backend_a",
                targets=("cuda", "shared"),
            ),
            _record(
                "record-b",
                plugin_id="vendor.b",
                entry_point="backend_b",
                targets=("shared", "xpu"),
            ),
        ]
    )
    conflict = report.conflicts[0]

    assert conflict.kind is ConflictKind.TARGET_OVERLAP
    assert conflict.severity is ConflictSeverity.REQUIRES_SELECTION
    assert conflict.claim == "shared"
    assert report.ok
    assert not report.has_fatal
    assert report.requires_selection
    assert report.fatal_conflicts == ()
    assert report.selection_conflicts == (conflict,)
    report.raise_for_fatal()


def test_multiple_active_records_are_fatal():
    report = detect_conflicts(
        [
            _record(
                "record-a",
                plugin_id="vendor.a",
                entry_point="backend_a",
                targets=("target-a",),
                state=PluginLifecycleState.ACTIVE,
            ),
            _record(
                "record-b",
                plugin_id="vendor.b",
                entry_point="backend_b",
                targets=("target-b",),
                state="active",
            ),
            _record(
                "record-c",
                plugin_id="vendor.c",
                entry_point="backend_c",
                targets=("target-c",),
                state=PluginLifecycleState.REGISTERED,
            ),
        ]
    )
    conflict = report.conflicts[0]

    assert conflict.kind is ConflictKind.MULTIPLE_ACTIVE
    assert conflict.severity is ConflictSeverity.FATAL
    assert conflict.claim == "active"
    assert conflict.record_ids == ("record-a", "record-b")


def test_all_findings_and_serialization_are_deterministic():
    records = [
        _record(
            "record-z",
            plugin_id="vendor.same",
            entry_point="same_backend",
            targets=("z-target", "shared"),
            state=PluginLifecycleState.ACTIVE,
        ),
        _record(
            "record-a",
            plugin_id="vendor.same",
            entry_point="same_backend",
            targets=("shared", "z-target"),
            state=PluginLifecycleState.ACTIVE,
        ),
    ]

    forward = detect_conflicts(records)
    reverse = detect_static_conflicts(reversed(records))

    assert forward.to_dict() == reverse.to_dict()
    assert [conflict.kind for conflict in forward.conflicts] == [
        ConflictKind.DUPLICATE_PLUGIN_ID,
        ConflictKind.DUPLICATE_ENTRY_POINT,
        ConflictKind.MULTIPLE_ACTIVE,
        ConflictKind.TARGET_OVERLAP,
        ConflictKind.TARGET_OVERLAP,
    ]
    assert {
        conflict.claim
        for conflict in forward.conflicts
        if conflict.kind is ConflictKind.TARGET_OVERLAP
    } == {"shared", "z-target"}
    serialized = forward.to_dict()
    assert serialized["conflicts"][0]["kind"] == "duplicate_plugin_id"
    assert serialized["conflicts"][0]["severity"] == "fatal"


def test_unverified_native_declarations_are_not_trusted_as_binary_facts():
    report = detect_conflicts(
        [
            _record(
                "record-a",
                plugin_id="vendor.a",
                entry_point="backend_a",
                targets=("target-a",),
                native_libraries=("libbackend.so",),
                abi_fingerprint="abi-a",
            ),
            _record(
                "record-b",
                plugin_id="vendor.b",
                entry_point="backend_b",
                targets=("target-b",),
                native_libraries=("libbackend.so",),
                abi_fingerprint="abi-b",
            ),
        ]
    )

    assert report.conflicts == ()


def test_verified_native_soname_and_export_conflicts_are_fatal():
    first_artifact = SimpleNamespace(
        identity="libvendor_collision.so",
        exported_symbols=("vendor_collision", "vendor_first"),
    )
    second_artifact = SimpleNamespace(
        identity="libvendor_collision.so",
        exported_symbols=("vendor_collision", "vendor_second"),
    )
    report = detect_conflicts(
        [
            _record(
                "record-b",
                plugin_id="vendor.b",
                entry_point="backend_b",
                targets=("target-b",),
                native_libraries=("vendor_b/libbackend.so",),
                native_artifacts=(second_artifact,),
            ),
            _record(
                "record-a",
                plugin_id="vendor.a",
                entry_point="backend_a",
                targets=("target-a",),
                native_libraries=("vendor_a/libbackend.so",),
                native_artifacts=(first_artifact,),
            ),
        ]
    )

    assert [conflict.kind for conflict in report.conflicts] == [
        ConflictKind.DUPLICATE_NATIVE_IDENTITY,
        ConflictKind.DUPLICATE_EXPORTED_SYMBOL,
    ]
    assert report.conflicts[0].claim == "libvendor_collision.so"
    assert report.conflicts[1].claim == "vendor_collision"
    assert all(
        conflict.severity is ConflictSeverity.FATAL
        for conflict in report.conflicts
    )
    with pytest.raises(BackendPluginConflictError) as caught:
        report.raise_for_fatal()
    assert caught.value.field == "native_libraries.SONAME"


def test_distinct_verified_native_artifacts_can_coexist():
    report = detect_conflicts(
        [
            _record(
                "record-a",
                plugin_id="vendor.a",
                entry_point="backend_a",
                targets=("target-a",),
                native_libraries=("vendor_a/libbackend.so",),
                native_artifacts=(
                    SimpleNamespace(
                        identity="libvendor_a.so",
                        exported_symbols=("vendor_a_entry",),
                    ),
                ),
            ),
            _record(
                "record-b",
                plugin_id="vendor.b",
                entry_point="backend_b",
                targets=("target-b",),
                native_libraries=("vendor_b/libbackend.so",),
                native_artifacts=(
                    SimpleNamespace(
                        identity="libvendor_b.so",
                        exported_symbols=("vendor_b_entry",),
                    ),
                ),
            ),
        ]
    )

    assert report.conflicts == ()


def test_legacy_record_without_manifest_participates_in_entry_point_conflicts():
    report = detect_conflicts(
        [
            _record("legacy-a", entry_point="same_backend"),
            _record("legacy-b", entry_point="same_backend"),
        ]
    )

    assert report.conflicts[0].kind is ConflictKind.DUPLICATE_ENTRY_POINT


def test_duplicate_record_id_is_malformed_input():
    with pytest.raises(ValueError, match="unique record_id"):
        detect_conflicts(
            [
                _record(
                    "same-record",
                    plugin_id="vendor.a",
                    entry_point="backend_a",
                ),
                _record(
                    "same-record",
                    plugin_id="vendor.b",
                    entry_point="backend_b",
                ),
            ]
        )


def test_record_id_is_required():
    with pytest.raises(TypeError, match="record_id"):
        detect_conflicts([SimpleNamespace()])
