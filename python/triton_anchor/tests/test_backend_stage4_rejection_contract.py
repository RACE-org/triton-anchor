"""Characterize rejection behavior before the Stage 4 extraction."""

from dataclasses import FrozenInstanceError, replace

import pytest

import triton_anchor.backends.registry as registry_module
from triton_anchor.backends._registry_rejections import plan_record_rejection
from triton_anchor.backends import (
    BackendPluginCompatibilityError,
    BackendPluginConflictError,
    BackendPluginError,
    BackendPluginProtocolError,
    BackendPluginRegistry,
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
)
from triton_anchor.tests.test_backend_registry import (
    FakeDistribution,
    RuntimePlugin,
    SUPPORTED_TAG,
    make_registry,
    manifest,
    plugin_record,
)


_COMPATIBILITY_ERROR_JSON_FIELDS = {
    "code",
    "message",
    "plugin_id",
    "entry_point",
    "detail",
    "field",
    "expected",
    "actual",
    "remediation",
    "dimension",
}


class EqualDistribution(FakeDistribution):
    """Distinct distribution objects that deliberately compare equal."""

    def __eq__(self, other):
        return isinstance(other, FakeDistribution)


def _entry_point_load_count(*distributions):
    return sum(
        entry_point.load_calls
        for distribution in distributions
        for entry_point in distribution.entry_points
    )


def test_record_rejection_uses_exact_snapshot_and_appends_error_identity(
    tmp_path,
):
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record()),
    )
    registry = make_registry((distribution,))
    discovered = registry.discover()[0]
    previous_error = BackendPluginError("previous rejection diagnostic")
    rejection_error = BackendPluginError("current rejection diagnostic")
    exact_snapshot = replace(
        discovered,
        errors=(previous_error,),
        selected_targets=("snapshot-only",),
    )

    # This private wrapper is the exact mutation seam Stage 4 will delegate;
    # using a non-canonical snapshot detects an accidental apply-time re-read.
    rejected = registry._reject(
        exact_snapshot,
        rejection_error,
        PluginCompatibilityStatus.COMPATIBLE,
    )

    current = registry.inspect(discovered.record_id)
    assert rejected is current
    assert current.state is PluginLifecycleState.REJECTED
    assert current.compatibility_status is PluginCompatibilityStatus.COMPATIBLE
    assert current.selected_targets == ("snapshot-only",)
    assert current.errors[0] is previous_error
    assert current.errors[1] is rejection_error
    assert current.error is previous_error
    assert _entry_point_load_count(distribution) == 0


def test_record_rejection_plan_is_frozen_and_preserves_exact_inputs(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record()),
    )
    registry = make_registry((distribution,))
    record = registry.discover()[0]
    error = BackendPluginError("planned rejection")

    plan = plan_record_rejection(
        record,
        error,
        PluginCompatibilityStatus.NOT_CHECKED,
    )

    assert plan.record_id == record.record_id
    assert plan.record is record
    assert plan.error is error
    assert plan.compatibility_status is PluginCompatibilityStatus.NOT_CHECKED
    with pytest.raises(FrozenInstanceError):
        plan.record_id = "changed"
    assert registry.inspect(record.record_id) is record
    assert record.state is PluginLifecycleState.DISCOVERED
    assert _entry_point_load_count(distribution) == 0


def test_distribution_scope_uses_identity_and_specializes_discovered_manifest_errors(
    tmp_path,
):
    scoped = EqualDistribution(
        tmp_path / "scoped",
        name="scoped-backend",
        manifest=manifest(
            plugin_record("first"),
            plugin_record("second"),
            plugin_record("validated"),
            plugin_record("legacy-shaped"),
        ),
        entry_points=(
            ("first", RuntimePlugin()),
            ("second", RuntimePlugin()),
            ("validated", RuntimePlugin()),
            ("legacy-shaped", RuntimePlugin()),
        ),
    )
    equal_but_distinct = EqualDistribution(
        tmp_path / "equal",
        name="equal-backend",
        manifest=manifest(plugin_record("other")),
        entry_points=(("other", RuntimePlugin()),),
    )
    assert scoped == equal_but_distinct
    assert scoped is not equal_but_distinct

    registry = make_registry((equal_but_distinct, scoped))
    records = {
        record.entry_point_name: record for record in registry.discover()
    }
    validated = registry.validate(records["validated"].record_id)
    assert validated.state is PluginLifecycleState.VALIDATED

    # A real distribution cannot mix Manifest and Legacy records.  This private
    # setup freezes the source predicate of the existing scope applicator.
    with registry._lock:
        registry._replace(
            replace(
                records["legacy-shaped"],
                source=PluginSource.LEGACY,
                manifest=None,
                compatibility_status=(
                    PluginCompatibilityStatus.LEGACY_UNVERIFIED
                ),
            )
        )

    original_error = BackendPluginCompatibilityError(
        "wheel platform tag",
        "a supported tag",
        "cp39-cp39-win_amd64",
        plugin_id="origin.plugin",
        entry_point="origin-entry-point",
    )
    # The private scope wrapper is the seam under characterization.  Invoke it
    # under the same Registry RLock held by every production caller.
    with registry._lock:
        registry._reject_manifest_scope(
            original_error,
            PluginCompatibilityStatus.INCOMPATIBLE,
            distribution=scoped,
            specialize_error=True,
        )

    current = {
        record.entry_point_name: record for record in registry.list()
    }
    first = current["first"]
    second = current["second"]
    assert first.state is PluginLifecycleState.REJECTED
    assert second.state is PluginLifecycleState.REJECTED
    assert first.compatibility_status is PluginCompatibilityStatus.INCOMPATIBLE
    assert second.compatibility_status is PluginCompatibilityStatus.INCOMPATIBLE
    assert first.error is not original_error
    assert second.error is not original_error
    assert first.error is not second.error

    for record in (first, second):
        payload = record.error.to_dict()
        assert set(payload) == _COMPATIBILITY_ERROR_JSON_FIELDS
        assert payload["code"] == "backend_plugin_compatibility_error"
        assert payload["plugin_id"] == "vendor." + record.entry_point_name
        assert payload["entry_point"] == record.entry_point_name
        assert payload["field"] == "wheel platform tag"
        assert payload["expected"]
        assert payload["actual"]
        assert payload["remediation"]

    assert current["validated"].state is PluginLifecycleState.VALIDATED
    assert current["validated"].errors == ()
    assert current["legacy-shaped"].state is PluginLifecycleState.DISCOVERED
    assert current["legacy-shaped"].source is PluginSource.LEGACY
    assert current["legacy-shaped"].errors == ()
    assert current["other"].state is PluginLifecycleState.DISCOVERED
    assert current["other"].errors == ()
    assert _entry_point_load_count(scoped, equal_but_distinct) == 0


def test_process_environment_failure_rejects_only_discovered_manifests_once(
    tmp_path,
):
    manifest_distribution = FakeDistribution(
        tmp_path / "manifest",
        name="manifest-backend",
        manifest=manifest(
            plugin_record("trigger"),
            plugin_record("other"),
            plugin_record("validated"),
            plugin_record("rejected"),
        ),
        entry_points=(
            ("trigger", RuntimePlugin()),
            ("other", RuntimePlugin()),
            ("validated", RuntimePlugin()),
            ("rejected", RuntimePlugin()),
        ),
    )
    legacy_distribution = FakeDistribution(
        tmp_path / "legacy",
        name="legacy-backend",
        manifest=None,
        entry_points=(("legacy", RuntimePlugin()),),
    )
    provider_calls = []

    def broken_environment_provider():
        provider_calls.append(True)
        raise RuntimeError("environment unavailable")

    registry = BackendPluginRegistry(
        distribution_provider=lambda: (
            manifest_distribution,
            legacy_distribution,
        ),
        environment_provider=broken_environment_provider,
        supported_tags=(SUPPORTED_TAG,),
        preflight_profile="full",
    )
    records = {
        record.entry_point_name: record for record in registry.discover()
    }
    preexisting_error = BackendPluginError("pre-existing rejection")
    # These private setup writes create otherwise unreachable mixed lifecycle
    # states before the process-scoped public validate() path is exercised.
    with registry._lock:
        registry._replace(
            replace(
                records["validated"],
                state=PluginLifecycleState.VALIDATED,
                compatibility_status=PluginCompatibilityStatus.COMPATIBLE,
            )
        )
        registry._reject(
            records["rejected"],
            preexisting_error,
            PluginCompatibilityStatus.NOT_CHECKED,
        )

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.validate(records["trigger"].record_id)

    current = {
        record.entry_point_name: record for record in registry.list()
    }
    assert caught.value.dimension == "Core environment metadata"
    assert caught.value.plugin_id is None
    assert caught.value.entry_point is None
    assert current["trigger"].error is caught.value
    assert current["other"].error is caught.value
    assert current["trigger"].compatibility_status is (
        PluginCompatibilityStatus.INCOMPATIBLE
    )
    assert current["other"].compatibility_status is (
        PluginCompatibilityStatus.INCOMPATIBLE
    )
    assert current["validated"].state is PluginLifecycleState.VALIDATED
    assert current["validated"].errors == ()
    assert current["rejected"].state is PluginLifecycleState.REJECTED
    assert current["rejected"].error is preexisting_error
    assert current["legacy"].source is PluginSource.LEGACY
    assert current["legacy"].state is PluginLifecycleState.DISCOVERED
    assert current["legacy"].compatibility_status is (
        PluginCompatibilityStatus.LEGACY_UNVERIFIED
    )
    assert current["legacy"].errors == ()

    diagnostics = registry.diagnostics()
    assert diagnostics["registry_errors"] == [caught.value.to_dict()]
    with pytest.raises(BackendPluginCompatibilityError) as repeated:
        registry.validate(records["other"].record_id)
    assert repeated.value is caught.value
    registry.validate()
    assert provider_calls == [True]
    assert _entry_point_load_count(
        manifest_distribution,
        legacy_distribution,
    ) == 0


def test_multi_conflict_keeps_only_first_kind_error_and_preserves_status(
    tmp_path,
):
    first = FakeDistribution(
        tmp_path / "first",
        name="first-backend",
        manifest=manifest(
            plugin_record("same", plugin_id="vendor.duplicate")
        ),
        entry_points=(("same", RuntimePlugin()),),
    )
    second = FakeDistribution(
        tmp_path / "second",
        name="second-backend",
        manifest=manifest(
            plugin_record("same", plugin_id="vendor.duplicate")
        ),
        entry_points=(("same", RuntimePlugin()),),
    )
    registry = make_registry((second, first))

    with pytest.raises(BackendPluginConflictError) as caught:
        registry.select("same", environment={})

    records = registry.list()
    assert all(
        record.state is PluginLifecycleState.REJECTED for record in records
    )
    assert all(
        record.compatibility_status is PluginCompatibilityStatus.COMPATIBLE
        for record in records
    )
    assert all(len(record.errors) == 1 for record in records)
    assert all(record.error.field == "plugin_id" for record in records)
    assert records[0].error is records[1].error
    assert caught.value is not records[0].error
    assert caught.value.to_dict() == records[0].error.to_dict()

    with pytest.raises(BackendPluginConflictError):
        registry.select("same", environment={})
    assert all(len(record.errors) == 1 for record in registry.list())
    assert _entry_point_load_count(first, second) == 0


def test_validate_strict_finishes_snapshot_then_raises_first_record_error(
    tmp_path,
):
    first = FakeDistribution(
        tmp_path / "first",
        name="a-backend",
        manifest=manifest(
            plugin_record("alpha", backend_protocol=">=2.0,<3.0")
        ),
        entry_points=(("alpha", RuntimePlugin()),),
    )
    compatible = FakeDistribution(
        tmp_path / "compatible",
        name="b-backend",
        manifest=manifest(plugin_record("beta")),
        entry_points=(("beta", RuntimePlugin()),),
    )
    last = FakeDistribution(
        tmp_path / "last",
        name="c-backend",
        manifest=manifest(
            plugin_record(
                "gamma",
                requires_triton={"version": ">=9.0"},
            )
        ),
        entry_points=(("gamma", RuntimePlugin()),),
    )
    registry = make_registry((last, compatible, first))

    with pytest.raises(BackendPluginProtocolError) as caught:
        registry.validate(strict=True)

    records = {
        record.entry_point_name: record for record in registry.list()
    }
    assert caught.value is records["alpha"].error
    assert records["alpha"].state is PluginLifecycleState.REJECTED
    assert records["alpha"].error.field == "backend_protocol"
    assert records["beta"].state is PluginLifecycleState.VALIDATED
    assert records["gamma"].state is PluginLifecycleState.REJECTED
    assert records["gamma"].error.field == "Triton version"
    assert _entry_point_load_count(first, compatible, last) == 0


def test_facade_preserves_generic_and_unexpected_validator_statuses(
    tmp_path,
    monkeypatch,
):
    generic_distribution = FakeDistribution(
        tmp_path / "generic",
        name="generic-backend",
        manifest=manifest(plugin_record("generic")),
        entry_points=(("generic", RuntimePlugin()),),
    )
    generic_registry = make_registry((generic_distribution,))
    generic_record = generic_registry.discover()[0]
    generic_error = BackendPluginError("metadata validator stopped")

    def raise_generic(*args, **kwargs):
        raise generic_error

    monkeypatch.setattr(
        registry_module,
        "validate_backend_plugin",
        raise_generic,
    )
    with pytest.raises(BackendPluginError) as generic_caught:
        generic_registry.validate(generic_record.record_id)

    generic_rejected = generic_registry.inspect(generic_record.record_id)
    assert generic_caught.value is generic_error
    assert generic_rejected.error is generic_error
    assert generic_rejected.state is PluginLifecycleState.REJECTED
    assert generic_rejected.compatibility_status is (
        PluginCompatibilityStatus.NOT_CHECKED
    )
    assert _entry_point_load_count(generic_distribution) == 0

    unexpected_distribution = FakeDistribution(
        tmp_path / "unexpected",
        name="unexpected-backend",
        manifest=manifest(plugin_record("unexpected")),
        entry_points=(("unexpected", RuntimePlugin()),),
    )
    unexpected_registry = make_registry((unexpected_distribution,))
    unexpected_record = unexpected_registry.discover()[0]

    def raise_unexpected(*args, **kwargs):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(
        registry_module,
        "validate_backend_plugin",
        raise_unexpected,
    )
    with pytest.raises(BackendPluginCompatibilityError) as unexpected_caught:
        unexpected_registry.validate(unexpected_record.record_id)

    unexpected_rejected = unexpected_registry.inspect(
        unexpected_record.record_id
    )
    assert unexpected_caught.value is unexpected_rejected.error
    assert unexpected_caught.value.dimension == "pre-load validation"
    assert unexpected_caught.value.plugin_id == "vendor.unexpected"
    assert unexpected_caught.value.entry_point == "unexpected"
    assert unexpected_caught.value.actual == "<error: validator exploded>"
    assert unexpected_rejected.state is PluginLifecycleState.REJECTED
    assert unexpected_rejected.compatibility_status is (
        PluginCompatibilityStatus.INCOMPATIBLE
    )
    assert _entry_point_load_count(unexpected_distribution) == 0
