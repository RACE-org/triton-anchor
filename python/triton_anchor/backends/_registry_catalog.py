"""Lock-free catalog operations for backend Registry records.

The public Registry facade owns locking, discovery-provider calls, and the
decision to mark discovery complete.  ``RegistryCatalog`` only normalizes
discovery identities, accepts already-produced discovery results into
``RegistryState``, and answers import-free record queries.

The record factory is supplied by the facade so this internal module does not
reverse-import :mod:`triton_anchor.backends.registry` or relocate the public
``BackendPluginRecord`` type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Container, Optional, Tuple

from packaging.utils import canonicalize_name

from ._registry_state import RegistryState
from .conflicts import ConflictReport, detect_conflicts
from .errors import (
    BackendPluginConflictError,
    BackendPluginError,
    BackendPluginSelectionError,
)
from .manifest import BackendPluginManifest
from .protocol import (
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
)


@dataclass(frozen=True)
class DiscoveryRecordMetadata:
    """Normalized, immutable identity metadata for one discovered record."""

    entry_point_name: str
    entry_point_value: str
    distribution_name: Optional[str]
    distribution_version: Optional[str]


def distribution_identity(
    distribution: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Read a distribution identity without importing plugin code."""
    try:
        metadata = getattr(distribution, "metadata", None)
    except Exception:
        metadata = None
    name = None
    if metadata is not None:
        try:
            name = metadata.get("Name")
        except Exception:
            name = None
    if name is None:
        try:
            name = getattr(distribution, "name", None)
        except Exception:
            name = None
    try:
        version = getattr(distribution, "version", None)
    except Exception:
        version = None

    try:
        name_text = str(name) if name is not None else None
    except Exception:
        name_text = None
    try:
        version_text = str(version) if version is not None else None
    except Exception:
        version_text = None
    return name_text, version_text


def entry_point_name(entry_point: Any) -> str:
    """Read and normalize an entry-point name defensively."""
    try:
        value = getattr(entry_point, "name", "")
        return str(value) if value is not None else ""
    except Exception:
        return ""


def entry_point_value(entry_point: Any) -> str:
    """Read and normalize an entry-point value defensively."""
    try:
        value = getattr(entry_point, "value", None)
        return str(value) if value is not None else ""
    except Exception:
        return ""


def record_metadata(
    entry_point: Any,
    distribution: Any,
) -> DiscoveryRecordMetadata:
    """Build the immutable discovery identity used by record construction."""
    distribution_name, distribution_version = distribution_identity(
        distribution
    )
    return DiscoveryRecordMetadata(
        entry_point_name=entry_point_name(entry_point),
        entry_point_value=entry_point_value(entry_point),
        distribution_name=distribution_name,
        distribution_version=distribution_version,
    )


def allocate_record_id(
    existing_ids: Container[str],
    distribution_name: Optional[str],
    entry_point: str,
) -> str:
    """Allocate the historical deterministic Registry record identity."""
    distribution_key = canonicalize_name(
        distribution_name or "unknown-distribution"
    )
    base = f"{distribution_key}:{entry_point}"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}#{suffix}"
        suffix += 1
    return candidate


RecordFactory = Callable[..., Any]


class RegistryCatalog:
    """Import-free record construction and query seam for one Registry.

    This object deliberately owns neither a lock nor discovery completion.
    The facade must call it under the same lock boundaries it uses today.
    """

    def __init__(
        self,
        state: RegistryState,
        *,
        record_factory: RecordFactory,
    ) -> None:
        self._state = state
        self._record_factory = record_factory

    def allocate_record_id(
        self,
        distribution_name: Optional[str],
        entry_point: str,
    ) -> str:
        """Allocate an ID against the current state without copying it."""
        return allocate_record_id(
            self._state,
            distribution_name,
            entry_point,
        )

    def accept_discovery_result(
        self,
        *,
        entry_point: Any,
        distribution: Any,
        source: Optional[PluginSource],
        manifest: Optional[BackendPluginManifest],
        state: PluginLifecycleState,
        compatibility_status: PluginCompatibilityStatus,
        error: Optional[BackendPluginError] = None,
    ) -> Any:
        """Construct and insert one metadata-only discovery result."""
        metadata = record_metadata(entry_point, distribution)
        record = self._record_factory(
            record_id=self.allocate_record_id(
                metadata.distribution_name,
                metadata.entry_point_name,
            ),
            entry_point_name=metadata.entry_point_name,
            entry_point_value=metadata.entry_point_value,
            distribution_name=metadata.distribution_name,
            distribution_version=metadata.distribution_version,
            source=source,
            state=state,
            compatibility_status=compatibility_status,
            entry_point=entry_point,
            distribution=distribution,
            manifest=manifest,
            errors=(error,) if error is not None else (),
        )
        return self._state.insert_record(record)

    def list_snapshot(self) -> Tuple[Any, ...]:
        """Return the stable insertion-ordered record snapshot."""
        return self._state.records_snapshot()

    def resolve(self, identifier: str) -> Any:
        """Resolve an exact record ID or one unique Registry key."""
        if self._state.contains_record(identifier):
            return self._state.get_record(identifier)
        matches = tuple(
            record
            for record in self._state.records_snapshot()
            if record.registry_key == identifier
        )
        if not matches:
            raise BackendPluginSelectionError(
                f"Unknown backend plugin record '{identifier}'",
                field="registry_key",
                expected="an existing record_id or unique registry_key",
                actual=identifier,
                remediation=(
                    "Call registry.list() and use one of the reported "
                    "record_id or registry_key values."
                ),
            )
        if len(matches) > 1:
            raise BackendPluginConflictError(
                f"Backend plugin key '{identifier}' is ambiguous across "
                "records: "
                + ", ".join(record.record_id for record in matches),
                plugin_id=identifier,
                field="registry_key",
                expected="a unique plugin identity",
                actual=", ".join(record.record_id for record in matches),
                remediation=(
                    "Use a record_id for inspection now; W7 will report and "
                    "govern the underlying identity conflict."
                ),
            )
        return matches[0]

    def inspect_snapshot(self, identifier: str) -> Any:
        """Return the immutable record selected by ``identifier``."""
        return self.resolve(identifier)

    def conflict_view(self) -> ConflictReport:
        """Project the current record snapshot into a deterministic report."""
        return detect_conflicts(self._state.records_snapshot())


__all__ = [
    "DiscoveryRecordMetadata",
    "RegistryCatalog",
    "allocate_record_id",
    "distribution_identity",
    "entry_point_name",
    "entry_point_value",
    "record_metadata",
]
