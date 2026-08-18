"""Load and lifecycle helpers used within Registry-owned lock boundaries.

These helpers never acquire Registry locks themselves.  The facade retains its
established RLock while calling plugin-related helpers and owns state changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict as _Dict, Optional, Protocol, Tuple

from .errors import BackendPluginLifecycleError
from .protocol import PluginLifecycleState, PluginSource, can_transition


class LifecycleRecord(Protocol):
    entry_point_name: str
    source: Optional[PluginSource]
    state: PluginLifecycleState

    @property
    def plugin_id(self) -> Optional[str]: ...


@dataclass(frozen=True)
class RuntimeInterfaceInspection:
    """Stable result of reading the required runtime pair once each."""

    compiler_cls: Any
    driver_cls: Any
    missing_fields: Tuple[str, ...]
    invalid_fields: Tuple[str, ...]
    field_errors: Tuple[Tuple[str, str], ...]


def load_plugin_object(entry_point: Any) -> Any:
    """Load one entry point and instantiate class-based plugin factories."""
    loaded = entry_point.load()
    return loaded() if isinstance(loaded, type) else loaded


def inspect_runtime_interfaces(plugin: Any) -> RuntimeInterfaceInspection:
    """Read compiler/driver attributes in their established order."""
    values: _Dict[str, Any] = {}
    field_errors: _Dict[str, str] = {}
    for name in ("compiler_cls", "driver_cls"):
        try:
            values[name] = getattr(plugin, name, None)
        except Exception as exc:
            field_errors[name] = str(exc)
    missing = tuple(name for name, value in values.items() if not value)
    invalid = tuple(
        name
        for name, value in values.items()
        if value is not None and not isinstance(value, type)
    )
    return RuntimeInterfaceInspection(
        compiler_cls=values.get("compiler_cls"),
        driver_cls=values.get("driver_cls"),
        missing_fields=missing,
        invalid_fields=invalid,
        field_errors=tuple(field_errors.items()),
    )


def best_effort_shutdown(plugin: Any) -> bool:
    """Run shutdown after failed initialization, suppressing cleanup errors."""
    try:
        shutdown = getattr(plugin, "shutdown", None)
    except Exception:
        shutdown = None
    if callable(shutdown):
        try:
            shutdown()
            return True
        except Exception:
            pass
    return False


def ensure_transition(
    record: LifecycleRecord,
    target: PluginLifecycleState,
    *,
    transition_allowed: Callable[
        [PluginLifecycleState, PluginLifecycleState, PluginSource], bool
    ] = can_transition,
) -> None:
    """Validate a lifecycle edge without mutating its record."""
    source = record.source or PluginSource.MANIFEST
    if not transition_allowed(record.state, target, source):
        raise BackendPluginLifecycleError(
            f"Invalid backend plugin lifecycle transition: "
            f"{record.state.value} -> {target.value}",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="state",
            expected=target.value,
            actual=record.state.value,
            remediation=(
                "Use Registry validate/load/register operations in order; "
                "do not mutate plugin state directly."
            ),
        )
