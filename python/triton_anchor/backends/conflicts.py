"""Deterministic, import-free conflict analysis for backend plugin records.

The analyzer deliberately operates on the records supplied by its caller.  It
does not load plugins, mutate registry state, or decide which lifecycle states
are eligible for selection.  This keeps W7 useful both immediately after
discovery and after a caller has filtered records by Triton compatibility.

Version/platform/ABI mismatches are per-plugin compatibility failures.  For
validated ``native_in_process`` records, however, this module also compares the
verified binary reports produced before import.  Duplicate SONAMEs and dynamic
exports are fatal because both plugins may later coexist in the host process.
A shared target remains non-fatal by itself: W8 may resolve it through an
explicit choice or a deterministic priority policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .errors import BackendPluginConflictError
from .protocol import PluginIsolationMode, PluginLifecycleState


class ConflictKind(str, Enum):
    """Static claims that can conflict within one candidate set."""

    DUPLICATE_PLUGIN_ID = "duplicate_plugin_id"
    DUPLICATE_ENTRY_POINT = "duplicate_entry_point"
    TARGET_OVERLAP = "target_overlap"
    MULTIPLE_ACTIVE = "multiple_active"
    DUPLICATE_NATIVE_IDENTITY = "duplicate_native_identity"
    DUPLICATE_EXPORTED_SYMBOL = "duplicate_exported_symbol"


class ConflictSeverity(str, Enum):
    """Whether selection can resolve a conflict."""

    FATAL = "fatal"
    REQUIRES_SELECTION = "requires_selection"


@dataclass(frozen=True)
class Conflict:
    """One deterministic description of a conflicting claim."""

    kind: ConflictKind
    severity: ConflictSeverity
    claim: str
    record_ids: Tuple[str, ...]
    plugin_ids: Tuple[str, ...]
    entry_point_names: Tuple[str, ...]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a stable, JSON-compatible representation."""
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "claim": self.claim,
            "record_ids": list(self.record_ids),
            "plugin_ids": list(self.plugin_ids),
            "entry_point_names": list(self.entry_point_names),
            "message": self.message,
        }

    def to_error(self) -> BackendPluginConflictError:
        """Convert this finding to the registry's structured error type."""
        field, expected, remediation = _diagnostic_contract(self)
        return BackendPluginConflictError(
            self.message,
            detail=(
                f"kind={self.kind.value}; severity={self.severity.value}; "
                f"claim={self.claim}; records={','.join(self.record_ids)}"
            ),
            field=field,
            expected=expected,
            actual=", ".join(self.record_ids),
            remediation=remediation,
        )


@dataclass(frozen=True)
class ConflictReport:
    """All conflicts found for one caller-defined candidate set."""

    conflicts: Tuple[Conflict, ...] = ()

    @property
    def has_fatal(self) -> bool:
        return any(
            conflict.severity is ConflictSeverity.FATAL
            for conflict in self.conflicts
        )

    @property
    def requires_selection(self) -> bool:
        return any(
            conflict.severity is ConflictSeverity.REQUIRES_SELECTION
            for conflict in self.conflicts
        )

    @property
    def ok(self) -> bool:
        """Return whether the report contains no fatal conflict."""
        return not self.has_fatal

    @property
    def fatal_conflicts(self) -> Tuple[Conflict, ...]:
        return tuple(
            conflict
            for conflict in self.conflicts
            if conflict.severity is ConflictSeverity.FATAL
        )

    @property
    def selection_conflicts(self) -> Tuple[Conflict, ...]:
        return tuple(
            conflict
            for conflict in self.conflicts
            if conflict.severity is ConflictSeverity.REQUIRES_SELECTION
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a stable, JSON-compatible representation."""
        return {
            "ok": self.ok,
            "has_fatal": self.has_fatal,
            "requires_selection": self.requires_selection,
            "conflicts": [
                conflict.to_dict() for conflict in self.conflicts
            ],
        }

    def raise_for_fatal(self) -> None:
        """Raise the first deterministic fatal finding, if one exists."""
        if self.fatal_conflicts:
            raise self.fatal_conflicts[0].to_error()


@dataclass(frozen=True)
class _RecordView:
    """Safe immutable projection of a BackendPluginRecord-like object."""

    record_id: str
    plugin_id: Optional[str]
    entry_point_name: Optional[str]
    targets: Tuple[str, ...]
    active: bool
    native_identities: Tuple[str, ...]
    exported_symbols: Tuple[str, ...]


_KIND_ORDER = {
    ConflictKind.DUPLICATE_PLUGIN_ID: 0,
    ConflictKind.DUPLICATE_ENTRY_POINT: 1,
    ConflictKind.DUPLICATE_NATIVE_IDENTITY: 2,
    ConflictKind.DUPLICATE_EXPORTED_SYMBOL: 3,
    ConflictKind.MULTIPLE_ACTIVE: 4,
    ConflictKind.TARGET_OVERLAP: 5,
}


def _optional_string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _project_record(record: Any) -> _RecordView:
    record_id = _optional_string(getattr(record, "record_id", None))
    if record_id is None:
        raise TypeError(
            "Conflict analysis requires each record to expose a non-empty "
            "string record_id"
        )

    plugin_id = _optional_string(getattr(record, "plugin_id", None))
    entry_point_name = _optional_string(
        getattr(record, "entry_point_name", None)
    )

    manifest = getattr(record, "manifest", None)
    raw_targets = getattr(manifest, "targets", ()) if manifest is not None else ()
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
    return _RecordView(
        record_id=record_id,
        plugin_id=plugin_id,
        entry_point_name=entry_point_name,
        targets=targets,
        active=active,
        native_identities=native_identities,
        exported_symbols=exported_symbols,
    )


def _group_by(
    records: Iterable[_RecordView],
    attribute: str,
) -> Mapping[str, List[_RecordView]]:
    grouped: Dict[str, List[_RecordView]] = {}
    for record in records:
        value = getattr(record, attribute)
        if value is not None:
            grouped.setdefault(value, []).append(record)
    return grouped


def _record_metadata(
    records: Iterable[_RecordView],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    records = tuple(records)
    return (
        tuple(sorted(record.record_id for record in records)),
        tuple(
            sorted(
                {
                    record.plugin_id
                    for record in records
                    if record.plugin_id is not None
                }
            )
        ),
        tuple(
            sorted(
                {
                    record.entry_point_name
                    for record in records
                    if record.entry_point_name is not None
                }
            )
        ),
    )


def _new_conflict(
    kind: ConflictKind,
    severity: ConflictSeverity,
    claim: str,
    records: Iterable[_RecordView],
) -> Conflict:
    record_ids, plugin_ids, entry_point_names = _record_metadata(records)
    if kind is ConflictKind.DUPLICATE_PLUGIN_ID:
        message = (
            f"Multiple backend plugin records claim plugin_id '{claim}': "
            + ", ".join(record_ids)
        )
    elif kind is ConflictKind.DUPLICATE_ENTRY_POINT:
        message = (
            f"Multiple backend plugin records claim entry point '{claim}': "
            + ", ".join(record_ids)
        )
    elif kind is ConflictKind.TARGET_OVERLAP:
        message = (
            f"Target '{claim}' has multiple backend candidates and requires "
            "selection: "
            + ", ".join(record_ids)
        )
    elif kind is ConflictKind.DUPLICATE_NATIVE_IDENTITY:
        message = (
            "Multiple in-process backend native libraries claim SONAME "
            "'{}': ".format(claim)
            + ", ".join(record_ids)
        )
    elif kind is ConflictKind.DUPLICATE_EXPORTED_SYMBOL:
        message = (
            "Multiple in-process backend native libraries export symbol "
            "'{}': ".format(claim)
            + ", ".join(record_ids)
        )
    else:
        message = (
            "Multiple backend plugin records are ACTIVE: "
            + ", ".join(record_ids)
        )
    return Conflict(
        kind=kind,
        severity=severity,
        claim=claim,
        record_ids=record_ids,
        plugin_ids=plugin_ids,
        entry_point_names=entry_point_names,
        message=message,
    )


def _diagnostic_contract(
    conflict: Conflict,
) -> Tuple[str, str, str]:
    if conflict.kind is ConflictKind.DUPLICATE_PLUGIN_ID:
        return (
            "plugin_id",
            "claimed by at most one backend plugin record",
            "Give every backend plugin a globally unique plugin_id.",
        )
    if conflict.kind is ConflictKind.DUPLICATE_ENTRY_POINT:
        return (
            "entry_point",
            "claimed by at most one backend plugin record",
            "Give every backend plugin a unique triton.backends entry point.",
        )
    if conflict.kind is ConflictKind.MULTIPLE_ACTIVE:
        return (
            "state",
            "at most one ACTIVE backend plugin record",
            "Deactivate the current backend before activating another one.",
        )
    if conflict.kind is ConflictKind.DUPLICATE_NATIVE_IDENTITY:
        return (
            "native_libraries.SONAME",
            "a unique SONAME across in-process backend plugins",
            (
                "Give each backend shared library a vendor-qualified SONAME "
                "and rebuild its wheel."
            ),
        )
    if conflict.kind is ConflictKind.DUPLICATE_EXPORTED_SYMBOL:
        return (
            "native_libraries.exported_symbols",
            "no overlapping dynamic export across in-process backend plugins",
            (
                "Hide private symbols and expose a vendor-qualified, minimal "
                "plugin API from each shared library."
            ),
        )
    return (
        "targets",
        "one selected backend for the requested target",
        "Select a backend explicitly or apply the deterministic W8 policy.",
    )


def detect_conflicts(records: Iterable[Any]) -> ConflictReport:
    """Detect W7 identity, target and active-state conflicts.

    The result is independent of input enumeration order.  Repeating the same
    ``record_id`` is rejected as malformed input rather than reported as a
    plugin conflict because record IDs are the identities used in diagnostics.
    """
    views = tuple(sorted((_project_record(record) for record in records),
                         key=lambda record: record.record_id))
    record_ids = [record.record_id for record in views]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "Conflict analysis requires unique record_id values"
        )

    conflicts: List[Conflict] = []

    for plugin_id, claimants in sorted(
        _group_by(views, "plugin_id").items()
    ):
        if len(claimants) > 1:
            conflicts.append(
                _new_conflict(
                    ConflictKind.DUPLICATE_PLUGIN_ID,
                    ConflictSeverity.FATAL,
                    plugin_id,
                    claimants,
                )
            )

    for entry_point, claimants in sorted(
        _group_by(views, "entry_point_name").items()
    ):
        if len(claimants) > 1:
            conflicts.append(
                _new_conflict(
                    ConflictKind.DUPLICATE_ENTRY_POINT,
                    ConflictSeverity.FATAL,
                    entry_point,
                    claimants,
                )
            )

    active = tuple(record for record in views if record.active)
    if len(active) > 1:
        conflicts.append(
            _new_conflict(
                ConflictKind.MULTIPLE_ACTIVE,
                ConflictSeverity.FATAL,
                PluginLifecycleState.ACTIVE.value,
                active,
            )
        )

    native_identities: Dict[str, List[_RecordView]] = {}
    exported_symbols: Dict[str, List[_RecordView]] = {}
    for record in views:
        for identity in record.native_identities:
            native_identities.setdefault(identity, []).append(record)
        for symbol in record.exported_symbols:
            exported_symbols.setdefault(symbol, []).append(record)
    for identity, claimants in sorted(native_identities.items()):
        if len(claimants) > 1:
            conflicts.append(
                _new_conflict(
                    ConflictKind.DUPLICATE_NATIVE_IDENTITY,
                    ConflictSeverity.FATAL,
                    identity,
                    claimants,
                )
            )
    for symbol, claimants in sorted(exported_symbols.items()):
        if len(claimants) > 1:
            conflicts.append(
                _new_conflict(
                    ConflictKind.DUPLICATE_EXPORTED_SYMBOL,
                    ConflictSeverity.FATAL,
                    symbol,
                    claimants,
                )
            )

    targets: Dict[str, List[_RecordView]] = {}
    for record in views:
        for target in record.targets:
            targets.setdefault(target, []).append(record)
    for target, claimants in sorted(targets.items()):
        if len(claimants) > 1:
            conflicts.append(
                _new_conflict(
                    ConflictKind.TARGET_OVERLAP,
                    ConflictSeverity.REQUIRES_SELECTION,
                    target,
                    claimants,
                )
            )

    conflicts.sort(
        key=lambda conflict: (
            _KIND_ORDER[conflict.kind],
            conflict.claim,
            conflict.record_ids,
        )
    )
    return ConflictReport(tuple(conflicts))


def detect_static_conflicts(records: Iterable[Any]) -> ConflictReport:
    """Explicitly named alias for :func:`detect_conflicts`."""
    return detect_conflicts(records)


__all__ = [
    "Conflict",
    "ConflictKind",
    "ConflictReport",
    "ConflictSeverity",
    "detect_conflicts",
    "detect_static_conflicts",
]
