"""Single-record preflight evaluation for the Registry facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol, Tuple

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


class PreflightRecord(Protocol):
    """The stable subset needed by preflight evaluation."""

    record_id: str
    entry_point_name: str
    distribution: Any
    manifest: BackendPluginManifest

    @property
    def plugin_id(self) -> Optional[str]: ...


@dataclass(frozen=True)
class PreflightOutcome:
    """Validation result; the Registry applies all state changes."""

    compatibility_report: Optional[CompatibilityReport] = None
    capability_report: Optional[CapabilityReport] = None
    error: Optional[BackendPluginError] = None
    compatibility_status: PluginCompatibilityStatus = (
        PluginCompatibilityStatus.COMPATIBLE
    )
    reject_distribution: bool = False
    specialize_error: bool = False


def evaluate_record_preflight(
    record: PreflightRecord,
    environment: CoreEnvironment,
    *,
    preflight_profile: str,
    core_abi_fingerprint: Optional[str],
    supported_tags: Optional[Tuple[Tag, ...]],
    core_capabilities: Iterable[str],
) -> PreflightOutcome:
    """Evaluate compatibility without mutating Registry state."""
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
            report = validate_triton_version_requirement(
                record.manifest,
                environment,
            )
        else:
            report = validate_backend_plugin(
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
        capability_report = validate_plugin_capabilities(
            record.manifest,
            core_provided=core_capabilities,
        )
    except BackendPluginCapabilityError as exc:
        return PreflightOutcome(
            error=exc,
            compatibility_status=PluginCompatibilityStatus.INCOMPATIBLE,
        )
    except (
        BackendPluginCompatibilityError,
        BackendPluginProtocolError,
    ) as exc:
        reject_distribution = (
            isinstance(exc, BackendPluginCompatibilityError)
            and exc.dimension.startswith("wheel platform")
        )
        return PreflightOutcome(
            error=exc,
            compatibility_status=PluginCompatibilityStatus.INCOMPATIBLE,
            reject_distribution=reject_distribution,
            specialize_error=reject_distribution,
        )
    except BackendPluginError as exc:
        return PreflightOutcome(
            error=exc,
            compatibility_status=PluginCompatibilityStatus.NOT_CHECKED,
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
        return PreflightOutcome(
            error=error,
            compatibility_status=PluginCompatibilityStatus.INCOMPATIBLE,
        )

    return PreflightOutcome(
        compatibility_report=report,
        capability_report=capability_report,
    )
