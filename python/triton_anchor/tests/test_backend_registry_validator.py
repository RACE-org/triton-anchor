"""Focused contracts for the Registry's pure validation planner."""

from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace

import pytest

from triton_anchor.backends._registry_preflight import (
    evaluate_record_preflight,
)
from triton_anchor.backends._registry_validator import (
    ValidationRejectionScope,
    plan_process_environment_failure,
    plan_record_validation,
)
from triton_anchor.backends.errors import (
    BackendPluginCapabilityError,
    BackendPluginCompatibilityError,
    BackendPluginError,
    BackendPluginProtocolError,
)
from triton_anchor.backends.protocol import (
    PluginCompatibilityStatus,
    PluginIsolationMode,
)


@dataclass(frozen=True)
class ManifestStub:
    isolation_mode: PluginIsolationMode = PluginIsolationMode.PYTHON_ONLY


@dataclass(frozen=True)
class RecordStub:
    record_id: str = "vendor-backend:mock"
    entry_point_name: str = "mock"
    distribution: object = object()
    manifest: ManifestStub = ManifestStub()

    @property
    def plugin_id(self):
        return "vendor.mock"


def environment():
    return SimpleNamespace(core_abi_fingerprint="environment-fingerprint")


def test_success_plan_preserves_validator_order_arguments_and_reports():
    calls = []
    compatibility_report = object()
    capability_report = object()
    record = RecordStub()

    def compatibility_validator(manifest, current_environment, **kwargs):
        calls.append(
            (
                "compatibility",
                manifest,
                current_environment,
                kwargs,
            )
        )
        return compatibility_report

    def capability_validator(manifest, **kwargs):
        calls.append(("capability", manifest, kwargs))
        return capability_report

    plan = plan_record_validation(
        record,
        environment(),
        preflight_profile="full",
        core_abi_fingerprint=None,
        supported_tags=None,
        core_capabilities=("core.ir.v1",),
        compatibility_validator=compatibility_validator,
        capability_validator=capability_validator,
    )

    assert [call[0] for call in calls] == ["compatibility", "capability"]
    assert calls[0][3] == {
        "distribution": record.distribution,
        "core_abi_fingerprint": "environment-fingerprint",
        "supported_tags": None,
    }
    assert calls[1][2] == {"core_provided": ("core.ir.v1",)}
    assert plan.accepted
    assert plan.rejection_scope is None
    assert plan.compatibility_report is compatibility_report
    assert plan.capability_report is capability_report
    assert plan.compatibility_status is PluginCompatibilityStatus.COMPATIBLE
    assert plan.first_error is None


def test_protocol_failure_is_record_scoped_and_short_circuits_capabilities():
    error = BackendPluginProtocolError(">=1.0,<2.0", "2.0")
    capability_calls = []

    def compatibility_validator(*args, **kwargs):
        raise error

    def capability_validator(*args, **kwargs):
        capability_calls.append(True)
        return object()

    plan = plan_record_validation(
        RecordStub(),
        environment(),
        preflight_profile="full",
        core_abi_fingerprint=None,
        supported_tags=None,
        core_capabilities=(),
        compatibility_validator=compatibility_validator,
        capability_validator=capability_validator,
    )

    assert plan.reject_record
    assert not plan.reject_distribution
    assert plan.first_error is error
    assert plan.errors == (error,)
    assert plan.compatibility_status is PluginCompatibilityStatus.INCOMPATIBLE
    assert capability_calls == []


def test_wheel_platform_failure_preserves_distribution_scope_and_identity():
    distribution = object()
    record = RecordStub(distribution=distribution)
    error = BackendPluginCompatibilityError(
        "wheel platform tag",
        "a supported tag",
        "cp39-cp39-win_amd64",
    )

    def compatibility_validator(*args, **kwargs):
        raise error

    plan = plan_record_validation(
        record,
        environment(),
        preflight_profile="full",
        core_abi_fingerprint="explicit-fingerprint",
        supported_tags=(),
        core_capabilities=(),
        compatibility_validator=compatibility_validator,
    )

    assert plan.rejection_scope is ValidationRejectionScope.DISTRIBUTION
    assert plan.reject_distribution
    assert plan.distribution is distribution
    assert plan.specialize_error
    assert plan.first_error is error


def test_process_environment_failure_is_explicit_and_plan_is_frozen():
    error = BackendPluginCompatibilityError(
        "Core environment metadata",
        "readable CoreEnvironment",
        "<error: unavailable>",
    )
    plan = plan_process_environment_failure(RecordStub(), error)

    assert plan.rejection_scope is ValidationRejectionScope.PROCESS
    assert plan.reject_process
    assert plan.first_error is error
    assert plan.compatibility_status is PluginCompatibilityStatus.INCOMPATIBLE
    with pytest.raises(FrozenInstanceError):
        plan.specialize_error = True


def test_capability_failure_is_record_scoped_after_compatibility_succeeds():
    compatibility_report = object()
    error = BackendPluginCapabilityError(
        ("core.ir.v1",),
        scope="plugin",
    )

    def capability_validator(*args, **kwargs):
        raise error

    plan = plan_record_validation(
        RecordStub(),
        environment(),
        preflight_profile="full",
        core_abi_fingerprint=None,
        supported_tags=None,
        core_capabilities=(),
        compatibility_validator=lambda *args, **kwargs: compatibility_report,
        capability_validator=capability_validator,
    )

    assert plan.reject_record
    assert plan.first_error is error
    assert plan.compatibility_report is None
    assert plan.capability_report is None
    assert plan.compatibility_status is PluginCompatibilityStatus.INCOMPATIBLE


def test_generic_and_unexpected_errors_keep_their_historical_statuses():
    generic_error = BackendPluginError("metadata validator stopped")

    def raise_generic(*args, **kwargs):
        raise generic_error

    generic_plan = plan_record_validation(
        RecordStub(),
        environment(),
        preflight_profile="full",
        core_abi_fingerprint=None,
        supported_tags=None,
        core_capabilities=(),
        compatibility_validator=raise_generic,
    )
    assert generic_plan.first_error is generic_error
    assert (
        generic_plan.compatibility_status
        is PluginCompatibilityStatus.NOT_CHECKED
    )

    def raise_unexpected(*args, **kwargs):
        raise RuntimeError("validator exploded")

    unexpected_plan = plan_record_validation(
        RecordStub(),
        environment(),
        preflight_profile="full",
        core_abi_fingerprint=None,
        supported_tags=None,
        core_capabilities=(),
        compatibility_validator=raise_unexpected,
    )
    assert unexpected_plan.reject_record
    assert isinstance(
        unexpected_plan.first_error,
        BackendPluginCompatibilityError,
    )
    assert unexpected_plan.first_error.dimension == "pre-load validation"
    assert unexpected_plan.first_error.actual == "<error: validator exploded>"
    assert (
        unexpected_plan.compatibility_status
        is PluginCompatibilityStatus.INCOMPATIBLE
    )


def test_triton_profile_rejects_subprocess_before_any_validator_call():
    record = RecordStub(
        manifest=ManifestStub(isolation_mode=PluginIsolationMode.SUBPROCESS)
    )
    calls = []

    def forbidden_validator(*args, **kwargs):
        calls.append(True)
        return object()

    plan = evaluate_record_preflight(
        record,
        environment(),
        preflight_profile="triton_version",
        core_abi_fingerprint=None,
        supported_tags=None,
        core_capabilities=(),
        compatibility_validator=forbidden_validator,
        triton_version_validator=forbidden_validator,
        capability_validator=forbidden_validator,
    )

    assert plan.reject_record
    assert plan.first_error.dimension == (
        "isolation mode for triton_version profile"
    )
    assert calls == []
