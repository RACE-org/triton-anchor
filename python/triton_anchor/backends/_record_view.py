"""Immutable, import-free views over backend plugin registry records.

Selection and conflict analysis consume the same Registry records, but their
input contracts are intentionally different.  Selection requires complete,
validated metadata and reports structured selection errors.  Conflict
analysis accepts partially discovered records and only requires a stable
``record_id``.  Keeping separate projectors preserves those error boundaries
while giving both callers explicit, read-only input models.

This module is internal.  Registry construction remains the authority for the
public ``BackendPluginRecord`` type and does not depend on these projections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from .errors import BackendPluginSelectionError
from .protocol import PluginIsolationMode, PluginLifecycleState


@dataclass(frozen=True)
class BackendPluginRecordIdentityView:
    """Stable identity fields shared by import-free record projections."""

    record_id: str
    plugin_id: Optional[str]
    entry_point_name: Optional[str]


@dataclass(frozen=True)
class SelectionRecordView(BackendPluginRecordIdentityView):
    """Complete immutable metadata required by deterministic selection."""

    entry_point_name: str
    record: Any = field(repr=False, compare=False)
    registry_key: str
    manifest: Any = field(repr=False, compare=False)
    is_legacy: bool
    state: Optional[str]
    compatibility_status: Optional[str]
    priority: int

    @property
    def sort_key(self) -> Tuple[str, str, str, str]:
        return selection_record_sort_key(self)

    @property
    def selector_keys(self) -> Tuple[str, ...]:
        return selection_selector_keys(self)


@dataclass(frozen=True)
class ConflictRecordView(BackendPluginRecordIdentityView):
    """Immutable claims consumed by deterministic conflict analysis."""

    targets: Tuple[str, ...]
    active: bool
    native_identities: Tuple[str, ...]
    exported_symbols: Tuple[str, ...]


def normalize_enum_string(value: Any) -> Optional[str]:
    """Return the string value of an enum-like object, preserving ``None``."""
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return raw if isinstance(raw, str) else str(raw)


def normalize_optional_string(value: Any) -> Optional[str]:
    """Keep non-empty strings and normalize every other value to ``None``."""
    return value if isinstance(value, str) and value else None


def selection_record_sort_key(
    view: SelectionRecordView,
) -> Tuple[str, str, str, str]:
    """Return the frozen W8 ordering key for selection records."""
    return (
        view.record_id,
        view.registry_key,
        view.plugin_id or "",
        view.entry_point_name,
    )


def conflict_record_sort_key(view: ConflictRecordView) -> str:
    """Return the frozen W7 ordering key for conflict records."""
    return view.record_id


def selection_selector_keys(view: SelectionRecordView) -> Tuple[str, ...]:
    """Return exact, stable selector aliases for one selection record."""
    values = {view.record_id, view.registry_key}
    if not view.is_legacy and view.plugin_id is not None:
        values.add(view.plugin_id)
    return tuple(sorted(values))


def _selection_record_string(
    record: Any,
    field_name: str,
) -> Optional[str]:
    try:
        value = getattr(record, field_name, None)
    except Exception as exc:
        raise BackendPluginSelectionError(
            f"Backend record field '{field_name}' is unreadable: {exc}",
            field=field_name,
            expected="readable record metadata",
            actual=f"<error: {exc}>",
            remediation="Repair the discovered backend record before selection.",
        ) from exc
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BackendPluginSelectionError(
            f"Backend record field '{field_name}' must be a non-empty string",
            field=field_name,
            expected="a non-empty string",
            actual=repr(value),
            remediation="Repair the discovered backend record before selection.",
        )
    return value


def project_selection_record(record: Any) -> SelectionRecordView:
    """Project one record using the frozen W8 validation/error contract."""
    record_id = _selection_record_string(record, "record_id")
    if record_id is None:
        raise BackendPluginSelectionError(
            "Every backend selection record must have a record_id",
            field="record_id",
            expected="a stable non-empty record identifier",
            actual="<missing>",
            remediation="Pass records created by BackendPluginRegistry.discover().",
        )

    try:
        manifest = getattr(record, "manifest", None)
        source = normalize_enum_string(getattr(record, "source", None))
        state = normalize_enum_string(getattr(record, "state", None))
        compatibility_status = normalize_enum_string(
            getattr(record, "compatibility_status", None)
        )
    except Exception as exc:
        raise BackendPluginSelectionError(
            f"Backend record '{record_id}' metadata is unreadable: {exc}",
            field="record",
            expected="readable discovery and validation metadata",
            actual=f"<error: {exc}>",
            remediation="Repair or rediscover the backend record before selection.",
        ) from exc

    is_legacy = manifest is None and source == "legacy"
    plugin_id = (
        getattr(manifest, "plugin_id", None)
        if manifest is not None
        else _selection_record_string(record, "plugin_id")
    )
    if plugin_id is not None and (
        not isinstance(plugin_id, str) or not plugin_id
    ):
        raise BackendPluginSelectionError(
            f"Backend record '{record_id}' has an invalid plugin_id",
            field="plugin_id",
            expected="a non-empty string or null for Legacy",
            actual=repr(plugin_id),
            remediation="Repair the backend Manifest before selection.",
        )

    registry_key = _selection_record_string(record, "registry_key")
    if registry_key is None:
        registry_key = plugin_id or record_id

    entry_point_name = _selection_record_string(record, "entry_point_name")
    if entry_point_name is None and manifest is not None:
        entry_point_name = getattr(manifest, "entry_point", None)
    if not isinstance(entry_point_name, str) or not entry_point_name:
        raise BackendPluginSelectionError(
            f"Backend record '{record_id}' has no entry-point name",
            field="entry_point_name",
            expected="a non-empty triton.backends entry-point name",
            actual=repr(entry_point_name),
            remediation="Rediscover the backend from valid package metadata.",
        )

    priority = getattr(manifest, "priority", 0) if manifest is not None else 0
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise BackendPluginSelectionError(
            f"Backend record '{record_id}' has an invalid priority",
            plugin_id=plugin_id,
            entry_point=entry_point_name,
            field="priority",
            expected="an integer",
            actual=repr(priority),
            remediation="Set Manifest priority to an integer.",
        )

    return SelectionRecordView(
        record=record,
        record_id=record_id,
        registry_key=registry_key,
        plugin_id=plugin_id,
        entry_point_name=entry_point_name,
        manifest=manifest,
        is_legacy=is_legacy,
        state=state,
        compatibility_status=compatibility_status,
        priority=priority,
    )


def project_conflict_record(record: Any) -> ConflictRecordView:
    """Project one record using the frozen W7 permissive input contract."""
    record_id = normalize_optional_string(getattr(record, "record_id", None))
    if record_id is None:
        raise TypeError(
            "Conflict analysis requires each record to expose a non-empty "
            "string record_id"
        )

    plugin_id = normalize_optional_string(getattr(record, "plugin_id", None))
    entry_point_name = normalize_optional_string(
        getattr(record, "entry_point_name", None)
    )

    manifest = getattr(record, "manifest", None)
    raw_targets = (
        getattr(manifest, "targets", ()) if manifest is not None else ()
    )
    try:
        targets = tuple(
            sorted(
                {
                    target
                    for target in raw_targets
                    if isinstance(target, str) and target
                }
            )
        )
    except TypeError as exc:
        raise TypeError(
            f"Record '{record_id}' manifest.targets must be iterable"
        ) from exc

    state = getattr(record, "state", None)
    active = (
        state is PluginLifecycleState.ACTIVE
        or state == PluginLifecycleState.ACTIVE.value
    )

    native_identities = ()
    exported_symbols = ()
    isolation_mode = getattr(manifest, "isolation_mode", None)
    if (
        isolation_mode is PluginIsolationMode.NATIVE_IN_PROCESS
        or isolation_mode == PluginIsolationMode.NATIVE_IN_PROCESS.value
    ):
        report = getattr(record, "compatibility_report", None)
        artifacts = getattr(report, "native_artifacts", ()) if report else ()
        try:
            native_identities = tuple(
                sorted(
                    {
                        identity
                        for identity in (
                            getattr(artifact, "identity", None)
                            for artifact in artifacts
                        )
                        if isinstance(identity, str) and identity
                    }
                )
            )
            exported_symbols = tuple(
                sorted(
                    {
                        symbol
                        for artifact in artifacts
                        for symbol in getattr(
                            artifact, "exported_symbols", ()
                        )
                        if isinstance(symbol, str) and symbol
                    }
                )
            )
        except TypeError as exc:
            raise TypeError(
                "Validated native artifact reports must expose iterable "
                "exported_symbols"
            ) from exc

    return ConflictRecordView(
        record_id=record_id,
        plugin_id=plugin_id,
        entry_point_name=entry_point_name,
        targets=targets,
        active=active,
        native_identities=native_identities,
        exported_symbols=exported_symbols,
    )


__all__ = [
    "BackendPluginRecordIdentityView",
    "ConflictRecordView",
    "SelectionRecordView",
    "conflict_record_sort_key",
    "normalize_enum_string",
    "normalize_optional_string",
    "project_conflict_record",
    "project_selection_record",
    "selection_record_sort_key",
    "selection_selector_keys",
]
