"""Compatibility shim for pre-Manifest backend plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import BackendPluginInterfaceError
from .protocol import PluginCompatibilityStatus, PluginSource


@dataclass(frozen=True)
class LegacyBackendPluginShim:
    """Expose an old ``compiler_cls``/``driver_cls`` plugin without mutation.

    This shim intentionally does not synthesize a manifest or claim version or
    ABI compatibility.  It preserves the original class objects exactly and
    marks the plugin as ``LEGACY_UNVERIFIED``.
    """

    entry_point_name: str
    compiler_cls: type
    driver_cls: type
    plugin_object: Any
    source: PluginSource = field(default=PluginSource.LEGACY, init=False)
    compatibility_status: PluginCompatibilityStatus = field(
        default=PluginCompatibilityStatus.LEGACY_UNVERIFIED, init=False
    )

    @classmethod
    def from_loaded_object(
        cls, entry_point_name: str, loaded_object: Any
    ) -> "LegacyBackendPluginShim":
        """Adapt an object already loaded by the existing entry-point path.

        Class entry points are instantiated to match the current
        ``triton.backends._discover_backends`` behavior.  Loading the entry
        point itself remains the caller's responsibility.
        """
        plugin = loaded_object() if isinstance(loaded_object, type) else loaded_object
        runtime_fields = {}
        field_errors = {}
        for field_name in ("compiler_cls", "driver_cls"):
            try:
                runtime_fields[field_name] = getattr(plugin, field_name, None)
            except Exception as exc:
                field_errors[field_name] = str(exc)
        missing_fields = [
            field for field, value in runtime_fields.items() if not value
        ]
        invalid_fields = [
            field
            for field, value in runtime_fields.items()
            if value is not None and not isinstance(value, type)
        ]
        if missing_fields or invalid_fields or field_errors:
            raise BackendPluginInterfaceError(
                missing_fields,
                invalid_fields=invalid_fields,
                field_errors=field_errors,
                entry_point=entry_point_name,
            )

        return cls(
            entry_point_name=entry_point_name,
            compiler_cls=runtime_fields["compiler_cls"],
            driver_cls=runtime_fields["driver_cls"],
            plugin_object=plugin,
        )
