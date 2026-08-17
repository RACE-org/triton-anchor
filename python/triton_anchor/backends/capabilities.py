"""Deterministic, import-free capability negotiation for backend plugins.

Capability names are opaque identifiers.  This module deliberately performs
only exact string-set comparisons: it does not infer aliases, versions, or
relationships between capability names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from .errors import BackendPluginCapabilityError
from .manifest import BackendPluginManifest


def _capability_tuple(values: Iterable[str], field_name: str) -> Tuple[str, ...]:
    """Validate and deterministically order one capability declaration."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be an iterable of capability strings")

    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{field_name} must contain only non-empty strings")
    if any(value != value.strip() for value in result):
        raise ValueError(
            f"{field_name} capability names cannot have surrounding whitespace"
        )
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} cannot contain duplicate capability names")
    return tuple(sorted(result))


@dataclass(frozen=True)
class CapabilityReport:
    """Complete result of one capability negotiation.

    ``plugin_required`` is checked only against capabilities provided by Core.
    Kernel requirements are checked against the union of Core and plugin
    capabilities.
    """

    core_provided: Tuple[str, ...]
    plugin_provided: Tuple[str, ...]
    plugin_required: Tuple[str, ...]
    kernel_required: Tuple[str, ...]
    kernel_available: Tuple[str, ...]
    missing_for_plugin: Tuple[str, ...]
    missing_for_kernel: Tuple[str, ...]
    plugin_id: Optional[str] = None
    entry_point: Optional[str] = None

    @property
    def compatible(self) -> bool:
        return not self.missing_for_plugin and not self.missing_for_kernel

    @property
    def missing(self) -> Tuple[str, ...]:
        """Return the deterministic union of all missing capabilities."""
        return tuple(
            sorted(set(self.missing_for_plugin).union(self.missing_for_kernel))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "entry_point": self.entry_point,
            "compatible": self.compatible,
            "core_provided": list(self.core_provided),
            "plugin_provided": list(self.plugin_provided),
            "plugin_required": list(self.plugin_required),
            "kernel_required": list(self.kernel_required),
            "kernel_available": list(self.kernel_available),
            "missing": list(self.missing),
            "missing_for_plugin": list(self.missing_for_plugin),
            "missing_for_kernel": list(self.missing_for_kernel),
        }


def evaluate_capabilities(
    *,
    core_provided: Iterable[str],
    plugin_provided: Iterable[str],
    plugin_required: Iterable[str] = (),
    kernel_required: Iterable[str] = (),
    plugin_id: Optional[str] = None,
    entry_point: Optional[str] = None,
) -> CapabilityReport:
    """Evaluate the W6 capability subset rules without importing a plugin."""
    core = _capability_tuple(core_provided, "core_provided")
    provided = _capability_tuple(plugin_provided, "plugin_provided")
    required_by_plugin = _capability_tuple(plugin_required, "plugin_required")
    required_by_kernel = _capability_tuple(kernel_required, "kernel_required")

    core_set = set(core)
    kernel_available = tuple(sorted(core_set.union(provided)))
    missing_for_plugin = tuple(
        sorted(set(required_by_plugin).difference(core_set))
    )
    missing_for_kernel = tuple(
        sorted(set(required_by_kernel).difference(kernel_available))
    )
    return CapabilityReport(
        core_provided=core,
        plugin_provided=provided,
        plugin_required=required_by_plugin,
        kernel_required=required_by_kernel,
        kernel_available=kernel_available,
        missing_for_plugin=missing_for_plugin,
        missing_for_kernel=missing_for_kernel,
        plugin_id=plugin_id,
        entry_point=entry_point,
    )


def _capability_error(report: CapabilityReport) -> BackendPluginCapabilityError:
    if report.missing_for_plugin and report.missing_for_kernel:
        scope = "plugin_and_kernel"
        # Core capabilities are the capabilities available to both scopes.
        available = report.core_provided
    elif report.missing_for_plugin:
        scope = "plugin"
        available = report.core_provided
    else:
        scope = "kernel"
        available = report.kernel_available
    return BackendPluginCapabilityError(
        report.missing,
        scope=scope,
        available_capabilities=available,
        missing_plugin_capabilities=report.missing_for_plugin,
        missing_kernel_capabilities=report.missing_for_kernel,
        plugin_id=report.plugin_id,
        entry_point=report.entry_point,
    )


def validate_capabilities(
    *,
    core_provided: Iterable[str],
    plugin_provided: Iterable[str],
    plugin_required: Iterable[str] = (),
    kernel_required: Iterable[str] = (),
    plugin_id: Optional[str] = None,
    entry_point: Optional[str] = None,
) -> CapabilityReport:
    """Return a report or raise a structured error listing missing names."""
    report = evaluate_capabilities(
        core_provided=core_provided,
        plugin_provided=plugin_provided,
        plugin_required=plugin_required,
        kernel_required=kernel_required,
        plugin_id=plugin_id,
        entry_point=entry_point,
    )
    if not report.compatible:
        raise _capability_error(report)
    return report


def evaluate_plugin_capabilities(
    plugin: BackendPluginManifest,
    *,
    core_provided: Iterable[str],
    kernel_required: Iterable[str] = (),
) -> CapabilityReport:
    """Evaluate capability requirements read from one static Manifest record."""
    return evaluate_capabilities(
        core_provided=core_provided,
        plugin_provided=plugin.capabilities,
        plugin_required=plugin.requires_capabilities,
        kernel_required=kernel_required,
        plugin_id=plugin.plugin_id,
        entry_point=plugin.entry_point,
    )


def validate_plugin_capabilities(
    plugin: BackendPluginManifest,
    *,
    core_provided: Iterable[str],
    kernel_required: Iterable[str] = (),
) -> CapabilityReport:
    """Validate one Manifest record and preserve its identity in diagnostics."""
    report = evaluate_plugin_capabilities(
        plugin,
        core_provided=core_provided,
        kernel_required=kernel_required,
    )
    if not report.compatible:
        raise _capability_error(report)
    return report
