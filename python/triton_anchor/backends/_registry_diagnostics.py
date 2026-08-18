"""Stateless projection and normalization helpers for Registry diagnostics.

The Registry facade owns locking, discovery, record lookup, plugin callback
execution, Mapping normalization boundaries, and the timing of every state
snapshot supplied to these helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from .errors import BackendPluginLifecycleError


def diagnostics_attribute_failure(
    message: str,
    *,
    plugin_id: Optional[str],
    entry_point: str,
    actual: str,
) -> Dict[str, Any]:
    """Project a diagnostics attribute lookup failure into its stable shape."""
    return {
        "error": BackendPluginLifecycleError(
            message,
            plugin_id=plugin_id,
            entry_point=entry_point,
            field="diagnostics",
            expected="a callable hook or no hook",
            actual=actual,
            remediation=(
                "Fix the diagnostics attribute so inspection does not raise."
            ),
        ).to_dict()
    }


def diagnostics_callback_failure(
    message: str,
    *,
    plugin_id: Optional[str],
    entry_point: str,
    actual: str,
) -> Dict[str, Any]:
    """Project a diagnostics callback failure into its stable shape."""
    return {
        "error": BackendPluginLifecycleError(
            message,
            plugin_id=plugin_id,
            entry_point=entry_point,
            field="diagnostics",
            expected="a diagnostic result",
            actual=actual,
            remediation="Fix diagnostics() so it is read-only.",
        ).to_dict()
    }


def normalize_diagnostics_value(value: Any) -> Any:
    """Shallow-copy Mapping results and preserve other values by identity."""
    return dict(value) if isinstance(value, Mapping) else value


def project_registry_diagnostics(
    *,
    preflight_profile: str,
    core_abi_fingerprint: object,
    registry_errors: Sequence[Mapping[str, Any]],
    conflicts: Mapping[str, Any],
    selections: Mapping[str, Any],
    plugins: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Assemble already-observed Registry diagnostics without side effects."""
    return {
        "preflight_profile": preflight_profile,
        "core_abi_fingerprint": core_abi_fingerprint,
        "registry_errors": registry_errors,
        "conflicts": conflicts,
        "selections": selections,
        "plugins": plugins,
    }


__all__ = [
    "diagnostics_attribute_failure",
    "diagnostics_callback_failure",
    "normalize_diagnostics_value",
    "project_registry_diagnostics",
]
