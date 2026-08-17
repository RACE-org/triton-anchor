"""Contract tests for Backend Plugin Protocol 1.0."""

import pytest

from triton_anchor.backends import (
    BACKEND_PLUGIN_PROTOCOL_VERSION,
    BackendPlugin,
    BackendPluginBase,
    BackendPluginCompatibilityError,
    BackendPluginInterfaceError,
    LegacyBackendPluginShim,
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
    can_transition,
)


class DummyCompiler:
    pass


class DummyDriver:
    pass


class StructuralPlugin:
    compiler_cls = DummyCompiler
    driver_cls = DummyDriver


class BasePlugin(BackendPluginBase):
    compiler_cls = DummyCompiler
    driver_cls = DummyDriver


class ClassEntryPointPlugin:
    compiler_cls = DummyCompiler
    driver_cls = DummyDriver


class CountingPlugin:
    def __init__(self):
        self.reads = {"compiler_cls": 0, "driver_cls": 0}

    @property
    def compiler_cls(self):
        self.reads["compiler_cls"] += 1
        return DummyCompiler

    @property
    def driver_cls(self):
        self.reads["driver_cls"] += 1
        return DummyDriver


def test_protocol_version_is_frozen_at_1_0():
    assert BACKEND_PLUGIN_PROTOCOL_VERSION == "1.0"


def test_minimal_runtime_interface_is_structural():
    assert isinstance(StructuralPlugin(), BackendPlugin)
    assert isinstance(BasePlugin(), BackendPlugin)


def test_optional_hooks_have_safe_defaults():
    plugin = BasePlugin()
    assert plugin.initialize({}) is None
    assert plugin.shutdown() is None
    assert plugin.diagnostics() == {}


def test_manifest_lifecycle_follows_validation_before_load():
    assert can_transition(
        PluginLifecycleState.DISCOVERED, PluginLifecycleState.VALIDATED
    )
    assert can_transition(
        PluginLifecycleState.VALIDATED, PluginLifecycleState.LOADED
    )
    assert not can_transition(
        PluginLifecycleState.DISCOVERED, PluginLifecycleState.REGISTERED
    )


def test_legacy_lifecycle_may_load_without_claiming_validation():
    assert can_transition(
        PluginLifecycleState.DISCOVERED,
        PluginLifecycleState.LOADED,
        PluginSource.LEGACY,
    )
    assert not can_transition(
        PluginLifecycleState.DISCOVERED,
        PluginLifecycleState.LOADED,
        PluginSource.MANIFEST,
    )


def test_rejected_is_terminal():
    for state in PluginLifecycleState:
        assert not can_transition(PluginLifecycleState.REJECTED, state)


def test_legacy_shim_preserves_original_classes():
    loaded = StructuralPlugin()
    shim = LegacyBackendPluginShim.from_loaded_object("legacy", loaded)

    assert shim.compiler_cls is DummyCompiler
    assert shim.driver_cls is DummyDriver
    assert shim.plugin_object is loaded
    assert shim.source is PluginSource.LEGACY
    assert (
        shim.compatibility_status
        is PluginCompatibilityStatus.LEGACY_UNVERIFIED
    )


def test_legacy_shim_matches_existing_class_entry_point_behavior():
    shim = LegacyBackendPluginShim.from_loaded_object(
        "legacy-class", ClassEntryPointPlugin
    )
    assert isinstance(shim.plugin_object, ClassEntryPointPlugin)
    assert shim.compiler_cls is DummyCompiler
    assert shim.driver_cls is DummyDriver


def test_legacy_shim_reads_runtime_fields_once():
    plugin = CountingPlugin()
    shim = LegacyBackendPluginShim.from_loaded_object("counting", plugin)

    assert shim.compiler_cls is DummyCompiler
    assert shim.driver_cls is DummyDriver
    assert plugin.reads == {"compiler_cls": 1, "driver_cls": 1}


@pytest.mark.parametrize(
    "plugin, missing",
    [
        (object(), ("compiler_cls", "driver_cls")),
        (type("OnlyCompiler", (), {"compiler_cls": DummyCompiler})(), ("driver_cls",)),
        (type("OnlyDriver", (), {"driver_cls": DummyDriver})(), ("compiler_cls",)),
        (
            type(
                "FalsyCompiler",
                (),
                {"compiler_cls": 0, "driver_cls": DummyDriver},
            )(),
            ("compiler_cls",),
        ),
    ],
)
def test_legacy_shim_rejects_incomplete_runtime_interface(plugin, missing):
    with pytest.raises(BackendPluginInterfaceError) as caught:
        LegacyBackendPluginShim.from_loaded_object("broken", plugin)

    assert caught.value.missing_fields == missing
    assert caught.value.entry_point == "broken"
    assert caught.value.to_dict()["code"] == "backend_plugin_interface_error"


@pytest.mark.parametrize(
    "invalid_fields,expected_invalid_fields,expected_field",
    [
        (("compiler_cls",), ("compiler_cls",), "compiler_cls"),
        (("driver_cls",), ("driver_cls",), "driver_cls"),
        (
            ("driver_cls", "compiler_cls", "driver_cls"),
            ("compiler_cls", "driver_cls"),
            "compiler_cls,driver_cls",
        ),
    ],
)
def test_interface_error_reports_exact_invalid_fields(
    invalid_fields,
    expected_invalid_fields,
    expected_field,
):
    error = BackendPluginInterfaceError(invalid_fields=invalid_fields)

    assert error.invalid_fields == expected_invalid_fields
    assert error.field == expected_field
    assert error.code == "backend_plugin_interface_error"
    assert error.expected == "class objects"
    assert error.actual
    assert error.remediation


def test_triton_version_error_is_structured_for_registry_use():
    error = BackendPluginCompatibilityError(
        "Triton version",
        ">=3.2,<3.3",
        "3.1.0",
        plugin_id="vendor.mock",
        entry_point="mock",
    )

    assert str(error) == "Incompatible Triton version: expected >=3.2,<3.3, got 3.1.0"
    assert error.to_dict() == {
        "code": "backend_plugin_compatibility_error",
        "message": (
            "Incompatible Triton version: expected >=3.2,<3.3, got 3.1.0"
        ),
        "plugin_id": "vendor.mock",
        "entry_point": "mock",
        "detail": "expected=>=3.2,<3.3; actual=3.1.0",
        "field": "Triton version",
        "dimension": "Triton version",
        "expected": ">=3.2,<3.3",
        "actual": "3.1.0",
        "remediation": (
            "Install a backend compatible with the current Triton version, "
            "or use a matching triton-anchor environment."
        ),
    }
