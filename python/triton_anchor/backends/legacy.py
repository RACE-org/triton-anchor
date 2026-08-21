"""Compatibility shim for pre-Manifest backend plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import BackendPluginInterfaceError
from .protocol import PluginCompatibilityStatus, PluginSource


@dataclass(frozen=True)
class LegacyRuntimePair:
    """Exact runtime classes produced by a version-owned Legacy adapter.

    Core deliberately has no knowledge of Triton's backend ABCs or package
    layout.  A registered integration materializer imports/interprets the
    entry-point root and returns this small, immutable result.  The Registry
    subsequently applies every registered runtime-pair validator (including
    the version-specific F6 contract) before publishing either class.
    """

    compiler_cls: type
    driver_cls: type

    def __post_init__(self) -> None:
        if not isinstance(self.compiler_cls, type):
            raise TypeError("Legacy runtime compiler_cls must be a class")
        if not isinstance(self.driver_cls, type):
            raise TypeError("Legacy runtime driver_cls must be a class")


@dataclass(frozen=True)
class LegacyRuntimePairMaterializationContext:
    """Stable metadata and the raw entry-point root given to an adapter."""

    record_id: str
    entry_point_name: str
    entry_point_value: str
    loaded_root: Any = field(repr=False, compare=False)


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
    def from_runtime_pair(
        cls,
        entry_point_name: str,
        runtime_pair: LegacyRuntimePair,
        loaded_root: Any,
    ) -> "LegacyBackendPluginShim":
        """Bind an adapter-produced pair to the exact loaded entry point."""
        if type(runtime_pair) is not LegacyRuntimePair:
            raise TypeError("runtime_pair must be an exact LegacyRuntimePair")
        return cls(
            entry_point_name=entry_point_name,
            compiler_cls=runtime_pair.compiler_cls,
            driver_cls=runtime_pair.driver_cls,
            plugin_object=loaded_root,
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
        missing_fields = [field for field, value in runtime_fields.items() if not value]
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
