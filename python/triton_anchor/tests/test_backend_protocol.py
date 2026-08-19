"""Contract tests for Backend Plugin Protocol 1.0."""

import pytest

from triton_anchor.backends import (
    BACKEND_PLUGIN_PROTOCOL_VERSION,
    DIAGNOSTICS_FIELD_POLICY,
    BackendPlugin,
    BackendPluginBase,
    BackendPluginCompatibilityError,
    BackendPluginInterfaceError,
    BackendPluginProtocolError,
    LegacyBackendPluginShim,
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
    ProtocolFieldStatus,
    can_transition,
    consume_protocol_field,
    evaluate_protocol_field_removal,
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


class ProtocolFieldProducer:
    def __init__(self, value=None, *, fail_on_read=False):
        self.value = value
        self.fail_on_read = fail_on_read
        self.reads = 0

    @property
    def diagnostics(self):
        self.reads += 1
        if self.fail_on_read:
            raise AssertionError("protocol field must not be read")
        return self.value


def test_protocol_version_is_frozen_at_1_0():
    assert BACKEND_PLUGIN_PROTOCOL_VERSION == "1.0"


@pytest.mark.parametrize(
    (
        "producer_version",
        "consumer_version",
        "field_is_read",
        "expected_status",
        "expected_value",
        "has_deprecation_diagnostic",
    ),
    [
        (
            "1.0",
            "1.1",
            False,
            ProtocolFieldStatus.COMPATIBLE_DEFAULT,
            {},
            False,
        ),
        (
            "1.1",
            "1.0",
            False,
            ProtocolFieldStatus.COMPATIBLE_IGNORED,
            None,
            False,
        ),
        (
            "1.1",
            "1.1",
            True,
            ProtocolFieldStatus.COMPATIBLE_PRESERVED,
            {"healthy": True},
            False,
        ),
        (
            "1.2",
            "1.2",
            True,
            ProtocolFieldStatus.ACCEPTED_WITH_DEPRECATION_DIAGNOSTIC,
            {"healthy": True},
            True,
        ),
        (
            "2.0",
            "2.0",
            False,
            ProtocolFieldStatus.COMPATIBLE,
            None,
            False,
        ),
    ],
    ids=(
        "older-producer-default",
        "older-consumer-ignore",
        "same-minor-preserve",
        "deprecated-warning",
        "removed-major-ignore",
    ),
)
def test_protocol_field_helper_version_matrix(
    producer_version,
    consumer_version,
    field_is_read,
    expected_status,
    expected_value,
    has_deprecation_diagnostic,
):
    """Characterize the standalone helper, not Registry/Manifest AC4 wiring."""
    producer = ProtocolFieldProducer(
        lambda: {"healthy": True},
        fail_on_read=not field_is_read,
    )

    result = consume_protocol_field(
        producer,
        producer_protocol_version=producer_version,
        consumer_protocol_version=consumer_version,
    )

    assert result.field == "diagnostics"
    assert result.status is expected_status
    assert result.value == expected_value
    assert result.error is None
    assert producer.reads == int(field_is_read)
    if has_deprecation_diagnostic:
        assert len(result.diagnostics) == 1
        diagnostic = result.diagnostics[0]
        assert diagnostic.code == "backend_plugin_protocol_field_deprecated"
        assert diagnostic.severity == "warning"
        assert diagnostic.field == "diagnostics"
        assert diagnostic.introduced_in == DIAGNOSTICS_FIELD_POLICY.introduced_in
        assert diagnostic.deprecated_in == DIAGNOSTICS_FIELD_POLICY.deprecated_in
        assert diagnostic.removed_in == DIAGNOSTICS_FIELD_POLICY.removed_in
        assert diagnostic.producer_protocol_version == producer_version
        assert diagnostic.consumer_protocol_version == consumer_version
    else:
        assert result.diagnostics == ()


def test_protocol_field_helper_missing_current_field_uses_fresh_default():
    """Characterize helper defaulting without claiming end-to-end AC4."""
    first = consume_protocol_field(
        object(),
        producer_protocol_version="1.1",
        consumer_protocol_version="1.1",
    )
    second = consume_protocol_field(
        object(),
        producer_protocol_version="1.1",
        consumer_protocol_version="1.1",
    )

    assert first.status is ProtocolFieldStatus.COMPATIBLE_DEFAULT
    assert second.status is ProtocolFieldStatus.COMPATIBLE_DEFAULT
    assert first.value == second.value == {}
    assert first.value is not second.value
    assert first.diagnostics == second.diagnostics == ()
    assert first.error is second.error is None


@pytest.mark.parametrize(
    "producer_version,consumer_version,expected",
    [
        ("1.2", "2.0", ">=2.0,<3.0"),
        ("2.0", "1.2", ">=1.0,<2.0"),
    ],
    ids=("older-producer", "newer-producer"),
)
def test_protocol_field_helper_major_mismatch_is_explicit_and_read_free(
    producer_version,
    consumer_version,
    expected,
):
    """Characterize helper major rejection without claiming wheel compatibility."""
    producer = ProtocolFieldProducer(fail_on_read=True)

    result = consume_protocol_field(
        producer,
        producer_protocol_version=producer_version,
        consumer_protocol_version=consumer_version,
    )

    assert result.status is ProtocolFieldStatus.EXPLICIT_PROTOCOL_INCOMPATIBILITY
    assert result.value is None
    assert result.diagnostics == ()
    assert isinstance(result.error, BackendPluginProtocolError)
    assert result.error.field == "backend_protocol"
    assert result.error.expected == expected
    assert result.error.actual == producer_version
    assert producer.reads == 0


@pytest.mark.parametrize(
    "candidate_version,expected_status,diagnostic_code",
    [
        (
            "1.9",
            ProtocolFieldStatus.FORBIDDEN,
            "backend_plugin_protocol_field_removal_forbidden",
        ),
        ("2.0", ProtocolFieldStatus.COMPATIBLE, None),
    ],
    ids=("same-major-forbidden", "next-major-compatible"),
)
def test_protocol_field_helper_removal_policy_matrix(
    candidate_version,
    expected_status,
    diagnostic_code,
):
    """Characterize removal policy only; this is not the excluded AC4 matrix."""
    result = evaluate_protocol_field_removal(candidate_version)

    assert result.field == "diagnostics"
    assert result.status is expected_status
    assert result.value is None
    assert result.error is None
    if diagnostic_code is None:
        assert result.diagnostics == ()
    else:
        assert len(result.diagnostics) == 1
        diagnostic = result.diagnostics[0]
        assert diagnostic.code == diagnostic_code
        assert diagnostic.severity == "error"
        assert diagnostic.field == "diagnostics"
        assert diagnostic.consumer_protocol_version == candidate_version


def test_protocol_field_helper_rejects_malformed_versions():
    """Malformed exact versions are helper input errors, not AC4 evidence."""
    with pytest.raises(ValueError, match="Invalid Backend Plugin Protocol version"):
        consume_protocol_field(
            object(),
            producer_protocol_version="not-a-version",
            consumer_protocol_version="1.1",
        )
    with pytest.raises(ValueError, match="Invalid Backend Plugin Protocol version"):
        consume_protocol_field(
            object(),
            producer_protocol_version="1.1",
            consumer_protocol_version="not-a-version",
        )
    with pytest.raises(ValueError, match="Invalid Backend Plugin Protocol version"):
        evaluate_protocol_field_removal("not-a-version")


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
