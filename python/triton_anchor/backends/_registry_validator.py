"""Pure, import-free validation planning for the Registry facade.

The functions in this module call only metadata validators supplied by the
facade.  They do not load plugins, acquire locks, or mutate Registry state.
The owning facade applies the returned plan in its established record order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Protocol, Tuple

from packaging.tags import Tag

from .capabilities import CapabilityReport, validate_plugin_capabilities
from .compatibility import (
    CompatibilityReport,
    validate_backend_plugin,
    validate_triton_version_requirement,
)
from .environment import CoreEnvironment
from .errors import (
    BackendPluginCapabilityError,
    BackendPluginCompatibilityError,
    BackendPluginError,
    BackendPluginProtocolError,
)
from .manifest import BackendPluginManifest
from .protocol import PluginCompatibilityStatus, PluginIsolationMode


class ValidationRecord(Protocol):
    """The stable record projection required by metadata validation."""

    record_id: str
    entry_point_name: str
    distribution: Any
    manifest: BackendPluginManifest

    @property
    def plugin_id(self) -> Optional[str]: ...


class ValidationRejectionScope(str, Enum):
    """Registry records affected when the facade applies a failed plan."""

    RECORD = "record"
    DISTRIBUTION = "distribution"
    PROCESS = "process"


@dataclass(frozen=True)
class ValidationOutcome:
    """Immutable reports, status, and ordered errors from validation."""

    compatibility_report: Optional[CompatibilityReport] = None
    capability_report: Optional[CapabilityReport] = None
    errors: Tuple[BackendPluginError, ...] = ()
    compatibility_status: PluginCompatibilityStatus = (
        PluginCompatibilityStatus.COMPATIBLE
    )

    @property
    def first_error(self) -> Optional[BackendPluginError]:
        """Return the stable first error observed by the validator chain."""
        return self.errors[0] if self.errors else None

    @property
    def error(self) -> Optional[BackendPluginError]:
        """Compatibility alias used by the existing preflight facade."""
        return self.first_error


@dataclass(frozen=True)
class ValidationPlan:
    """A side-effect-free instruction for applying one validation outcome.

    ``distribution`` retains the exact object used by discovery because the
    facade's existing distribution-scope rejection compares object identity.
    It is deliberately excluded from equality and repr normalization.
    """

    record_id: Optional[str]
    outcome: ValidationOutcome
    rejection_scope: Optional[ValidationRejectionScope] = None
    distribution: Any = field(default=None, repr=False, compare=False)
    specialize_error: bool = False

    @property
    def first_error(self) -> Optional[BackendPluginError]:
        return self.outcome.first_error

    @property
    def error(self) -> Optional[BackendPluginError]:
        """Compatibility alias used by ``BackendPluginRegistry``."""
        return self.first_error

    @property
    def errors(self) -> Tuple[BackendPluginError, ...]:
        return self.outcome.errors

    @property
    def compatibility_report(self) -> Optional[CompatibilityReport]:
        return self.outcome.compatibility_report

    @property
    def capability_report(self) -> Optional[CapabilityReport]:
        return self.outcome.capability_report

    @property
    def compatibility_status(self) -> PluginCompatibilityStatus:
        return self.outcome.compatibility_status

    @property
    def accepted(self) -> bool:
        return self.rejection_scope is None and self.first_error is None

    @property
    def reject_record(self) -> bool:
        return self.rejection_scope is ValidationRejectionScope.RECORD

    @property
    def reject_distribution(self) -> bool:
        return self.rejection_scope is ValidationRejectionScope.DISTRIBUTION

    @property
    def reject_process(self) -> bool:
        return self.rejection_scope is ValidationRejectionScope.PROCESS


def _rejected_plan(
    record: ValidationRecord,
    error: BackendPluginError,
    compatibility_status: PluginCompatibilityStatus,
    *,
    rejection_scope: ValidationRejectionScope = (
        ValidationRejectionScope.RECORD
    ),
    specialize_error: bool = False,
) -> ValidationPlan:
    return ValidationPlan(
        record_id=record.record_id,
        outcome=ValidationOutcome(
            errors=(error,),
            compatibility_status=compatibility_status,
        ),
        rejection_scope=rejection_scope,
        distribution=(
            record.distribution
            if rejection_scope is ValidationRejectionScope.DISTRIBUTION
            else None
        ),
        specialize_error=specialize_error,
    )


def plan_process_environment_failure(
    record: ValidationRecord,
    error: BackendPluginError,
) -> ValidationPlan:
    """Plan fail-closed rejection for a process-wide environment failure."""
    return _rejected_plan(
        record,
        error,
        PluginCompatibilityStatus.INCOMPATIBLE,
        rejection_scope=ValidationRejectionScope.PROCESS,
    )


def plan_record_validation(
    record: ValidationRecord,
    environment: CoreEnvironment,
    *,
    preflight_profile: str,
    core_abi_fingerprint: Optional[str],
    supported_tags: Optional[Tuple[Tag, ...]],
    core_capabilities: Iterable[str],
    compatibility_validator: Callable[..., CompatibilityReport] = (
        validate_backend_plugin
    ),
    triton_version_validator: Callable[..., CompatibilityReport] = (
        validate_triton_version_requirement
    ),
    capability_validator: Callable[..., CapabilityReport] = (
        validate_plugin_capabilities
    ),
) -> ValidationPlan:
    """Run the existing ordered validator chain and return an apply plan."""
    try:
        if preflight_profile == "triton_version":
            if (
                record.manifest.isolation_mode
                is not PluginIsolationMode.PYTHON_ONLY
            ):
                raise BackendPluginCompatibilityError(
                    "isolation mode for triton_version profile",
                    PluginIsolationMode.PYTHON_ONLY.value,
                    record.manifest.isolation_mode.value,
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                    remediation=(
                        "The staged W6-W8 profile only admits python_only "
                        "plugins. Keep native_in_process and subprocess "
                        "plugins disabled until their ABI or IR-contract "
                        "validation is enabled."
                    ),
                )
            compatibility_report = triton_version_validator(
                record.manifest,
                environment,
            )
        else:
            compatibility_report = compatibility_validator(
                record.manifest,
                environment,
                distribution=record.distribution,
                core_abi_fingerprint=(
                    core_abi_fingerprint
                    if core_abi_fingerprint is not None
                    else environment.core_abi_fingerprint
                ),
                supported_tags=supported_tags,
            )
        capability_report = capability_validator(
            record.manifest,
            core_provided=core_capabilities,
        )
    except BackendPluginCapabilityError as exc:
        return _rejected_plan(
            record,
            exc,
            PluginCompatibilityStatus.INCOMPATIBLE,
        )
    except (
        BackendPluginCompatibilityError,
        BackendPluginProtocolError,
    ) as exc:
        reject_distribution = (
            isinstance(exc, BackendPluginCompatibilityError)
            and exc.dimension.startswith("wheel platform")
        )
        return _rejected_plan(
            record,
            exc,
            PluginCompatibilityStatus.INCOMPATIBLE,
            rejection_scope=(
                ValidationRejectionScope.DISTRIBUTION
                if reject_distribution
                else ValidationRejectionScope.RECORD
            ),
            specialize_error=reject_distribution,
        )
    except BackendPluginError as exc:
        return _rejected_plan(
            record,
            exc,
            PluginCompatibilityStatus.NOT_CHECKED,
        )
    except Exception as exc:
        error = BackendPluginCompatibilityError(
            "pre-load validation",
            "all compatibility checks complete without an internal error",
            f"<error: {exc}>",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            remediation=(
                "Repair the backend distribution metadata or report this "
                "validator failure; the plugin was not imported."
            ),
        )
        return _rejected_plan(
            record,
            error,
            PluginCompatibilityStatus.INCOMPATIBLE,
        )

    return ValidationPlan(
        record_id=record.record_id,
        outcome=ValidationOutcome(
            compatibility_report=compatibility_report,
            capability_report=capability_report,
        ),
    )
