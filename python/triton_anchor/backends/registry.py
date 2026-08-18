"""Stateful, import-gated registry for out-of-tree backend plugins."""

from __future__ import annotations

import importlib.metadata
import os
import threading
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from packaging.tags import Tag
from packaging.utils import canonicalize_name

from .capabilities import (
    CapabilityReport,
    evaluate_capabilities,
    validate_plugin_capabilities,
)
from ._registry_discovery import (
    BACKEND_ENTRY_POINT_GROUP as _BACKEND_ENTRY_POINT_GROUP,
    allocate_record_id as _allocate_record_id,
    copy_manifest_error as _copy_manifest_error,
    discover_distributions as _discover_distributions,
    distribution_identity as _distribution_identity,
    entry_point_name as _entry_point_name,
    entry_point_value as _entry_point_value,
    record_metadata as _record_metadata,
    source_hint as _source_hint,
)
from ._registry_lifecycle import (
    best_effort_shutdown as _best_effort_shutdown,
    ensure_transition as _ensure_transition,
    inspect_runtime_interfaces as _inspect_runtime_interfaces,
    load_plugin_object as _load_plugin_object,
)
from ._registry_preflight import evaluate_record_preflight
from .compatibility import (
    CompatibilityReport,
    validate_backend_plugin,
    validate_triton_version_requirement,
)
from .conflicts import ConflictReport, detect_conflicts
from .environment import CoreEnvironment, collect_core_environment
from .errors import (
    BackendPluginCompatibilityError,
    BackendPluginConflictError,
    BackendPluginDiscoveryError,
    BackendPluginError,
    BackendPluginInterfaceError,
    BackendPluginLifecycleError,
    BackendPluginLoadError,
    BackendPluginManifestError,
    BackendPluginSelectionError,
)
from .manifest import BackendPluginManifest, load_distribution_manifest
from .protocol import (
    PluginCompatibilityStatus,
    PluginIsolationMode,
    PluginLifecycleState,
    PluginSource,
    can_transition,
)
from .selection import SelectionDecision, select_backend


_PREFLIGHT_PROFILES = {"triton_version", "full"}


@dataclass(frozen=True)
class BackendPluginRecord:
    """One entry point and all state that must stay paired with it."""

    record_id: str
    entry_point_name: str
    entry_point_value: str
    distribution_name: Optional[str]
    distribution_version: Optional[str]
    source: Optional[PluginSource]
    state: PluginLifecycleState
    compatibility_status: PluginCompatibilityStatus
    entry_point: Any = field(repr=False, compare=False)
    distribution: Any = field(repr=False, compare=False)
    manifest: Optional[BackendPluginManifest] = None
    compatibility_report: Optional[CompatibilityReport] = None
    capability_report: Optional[CapabilityReport] = None
    errors: Tuple[BackendPluginError, ...] = ()
    plugin_object: Any = field(default=None, repr=False, compare=False)
    compiler_cls: Optional[type] = field(default=None, repr=False, compare=False)
    driver_cls: Optional[type] = field(default=None, repr=False, compare=False)
    initialized: bool = False
    shutdown_called: bool = False
    selected_targets: Tuple[str, ...] = ()

    @property
    def plugin_id(self) -> Optional[str]:
        return self.manifest.plugin_id if self.manifest is not None else None

    @property
    def registry_key(self) -> str:
        if self.plugin_id is not None:
            return self.plugin_id
        if self.source is PluginSource.LEGACY:
            distribution = canonicalize_name(
                self.distribution_name or "unknown-distribution"
            )
            return f"legacy:{distribution}:{self.entry_point_name}"
        return self.record_id

    @property
    def error(self) -> Optional[BackendPluginError]:
        return self.errors[0] if self.errors else None

    def to_dict(self) -> Dict[str, Any]:
        manifest = None
        if self.manifest is not None:
            manifest = {
                "plugin_id": self.manifest.plugin_id,
                "entry_point": self.manifest.entry_point,
                "backend_protocol": self.manifest.backend_protocol,
                "targets": list(self.manifest.targets),
                "capabilities": list(self.manifest.capabilities),
                "requires_capabilities": list(
                    self.manifest.requires_capabilities
                ),
                "isolation_mode": self.manifest.isolation_mode.value,
                "priority": self.manifest.priority,
            }
        return {
            "record_id": self.record_id,
            "registry_key": self.registry_key,
            "plugin_id": self.plugin_id,
            "entry_point": self.entry_point_name,
            "entry_point_value": self.entry_point_value,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "source": self.source.value if self.source is not None else None,
            "state": self.state.value,
            "compatibility_status": self.compatibility_status.value,
            "manifest": manifest,
            "compatibility_report": (
                self.compatibility_report.to_dict()
                if self.compatibility_report is not None
                else None
            ),
            "capability_report": (
                self.capability_report.to_dict()
                if self.capability_report is not None
                else None
            ),
            "errors": [error.to_dict() for error in self.errors],
            "loaded": self.plugin_object is not None,
            "registered": (
                self.state
                in {
                    PluginLifecycleState.REGISTERED,
                    PluginLifecycleState.SELECTED,
                    PluginLifecycleState.ACTIVE,
                }
            ),
            "compiler_cls": _qualified_name(self.compiler_cls),
            "driver_cls": _qualified_name(self.driver_cls),
            "initialized": self.initialized,
            "shutdown_called": self.shutdown_called,
            "selected_targets": list(self.selected_targets),
        }


def _qualified_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        module = getattr(value, "__module__", None)
        qualname = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", None),
        )
    except Exception:
        return "<unreadable>"
    if module and qualname:
        return f"{module}.{qualname}"
    try:
        return repr(value)
    except Exception:
        return "<unreadable>"


class BackendPluginRegistry:
    """Discover, inspect, validate, load, and register backend plugins.

    Discovery is metadata-only.  Manifest plugins cannot call ``ep.load()``
    until every W5 compatibility check has moved their record to VALIDATED.
    """

    def __init__(
        self,
        *,
        distribution_provider: Optional[Callable[[], Iterable[Any]]] = None,
        environment_provider: Optional[Callable[[], CoreEnvironment]] = None,
        core_abi_fingerprint: Optional[str] = None,
        supported_tags: Optional[Iterable[Tag]] = None,
        core_capabilities: Iterable[str] = (),
        preflight_profile: str = "full",
    ) -> None:
        self._distribution_provider = (
            distribution_provider or importlib.metadata.distributions
        )
        self._environment_provider = (
            environment_provider or collect_core_environment
        )
        self._core_abi_fingerprint = core_abi_fingerprint
        self._supported_tags = (
            tuple(supported_tags) if supported_tags is not None else None
        )
        self._core_capabilities = evaluate_capabilities(
            core_provided=core_capabilities,
            plugin_provided=(),
        ).core_provided
        if preflight_profile not in _PREFLIGHT_PROFILES:
            raise ValueError(
                "preflight_profile must be 'triton_version' or 'full'"
            )
        self._preflight_profile = preflight_profile
        self._records: Dict[str, BackendPluginRecord] = {}
        self._registry_errors: Tuple[BackendPluginError, ...] = ()
        self._environment: Optional[CoreEnvironment] = None
        self._environment_error: Optional[BackendPluginError] = None
        self._discovered = False
        self._loading: Set[str] = set()
        self._registering: Set[str] = set()
        self._selections: Dict[str, SelectionDecision] = {}
        self._reset_hooks: list = []
        self._resetting = False
        self._generation = 0
        self._lock = threading.RLock()

    def _ensure_not_resetting(self, operation: str) -> None:
        if self._resetting:
            raise BackendPluginLifecycleError(
                "Backend Registry operation is not allowed during reset",
                field=operation,
                expected="reset to complete before lifecycle mutation",
                actual="reset in progress",
                remediation=(
                    "Do not discover, validate, load, register, select, or "
                    "activate plugins from shutdown/reset hooks."
                ),
            )

    def register_reset_hook(self, callback: Callable[[], None]) -> None:
        """Register an idempotent W9 cache-reset callback."""
        if not callable(callback):
            raise TypeError("reset hook must be callable")
        with self._lock:
            self._ensure_not_resetting("register_reset_hook")
            if callback not in self._reset_hooks:
                self._reset_hooks.append(callback)

    @property
    def generation(self) -> int:
        """Return the Registry epoch used to invalidate W9 adapter snapshots."""
        with self._lock:
            return self._generation

    def _get_environment(self) -> CoreEnvironment:
        if self._environment is not None:
            return self._environment
        if self._environment_error is not None:
            raise self._environment_error
        try:
            environment = self._environment_provider()
        except Exception as exc:
            error = BackendPluginCompatibilityError(
                "Core environment metadata",
                "readable CoreEnvironment",
                f"<error: {exc}>",
                remediation=(
                    "Repair or rebuild triton-anchor so its generated build "
                    "metadata can be read before validating plugins."
                ),
            )
            self._environment_error = error
            self._record_registry_error(error)
            raise error from exc
        if not isinstance(environment, CoreEnvironment):
            error = BackendPluginCompatibilityError(
                "Core environment metadata",
                "CoreEnvironment",
                type(environment).__name__,
                remediation="Return a CoreEnvironment from environment_provider.",
            )
            self._environment_error = error
            self._record_registry_error(error)
            raise error
        self._environment = environment
        return environment

    def _record_registry_error(self, error: BackendPluginError) -> None:
        if all(existing is not error for existing in self._registry_errors):
            self._registry_errors += (error,)

    def _allocate_record_id(self, distribution_name: Optional[str], name: str) -> str:
        return _allocate_record_id(self._records, distribution_name, name)

    def _store_record(
        self,
        *,
        entry_point: Any,
        distribution: Any,
        source: Optional[PluginSource],
        manifest: Optional[BackendPluginManifest],
        state: PluginLifecycleState,
        compatibility_status: PluginCompatibilityStatus,
        error: Optional[BackendPluginError] = None,
    ) -> BackendPluginRecord:
        metadata = _record_metadata(entry_point, distribution)
        record = BackendPluginRecord(
            record_id=self._allocate_record_id(
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
        self._records[record.record_id] = record
        return record

    def discover(self, *, strict: bool = False) -> Tuple[BackendPluginRecord, ...]:
        """Discover backend metadata without importing backend code."""
        with self._lock:
            self._ensure_not_resetting("discover")
            if self._discovered:
                records = self.list()
                if strict:
                    self._raise_first_discovery_error(records)
                return records

            try:
                distributions = tuple(self._distribution_provider())
            except Exception as exc:
                error = BackendPluginDiscoveryError(
                    f"Unable to enumerate installed distributions: {exc}",
                    field="installed_distributions",
                    expected="an iterable of distribution metadata",
                    actual=f"<error: {exc}>",
                    remediation=(
                        "Repair the Python package metadata environment before "
                        "discovering backend plugins."
                    ),
                )
                self._registry_errors = (error,)
                raise error from exc

            _discover_distributions(
                distributions,
                store_record=self._store_record,
                record_registry_error=self._record_registry_error,
                load_manifest=load_distribution_manifest,
            )
            self._discovered = True
            records = self.list()
            if strict:
                self._raise_first_discovery_error(records)
            return records

    def _raise_first_discovery_error(
        self, records: Tuple[BackendPluginRecord, ...]
    ) -> None:
        if self._registry_errors:
            raise self._registry_errors[0]
        for record in records:
            if record.state is PluginLifecycleState.REJECTED and record.error:
                raise record.error

    def list(self) -> Tuple[BackendPluginRecord, ...]:
        """Return immutable record snapshots without validation or loading."""
        with self._lock:
            if not self._discovered:
                self.discover()
            return tuple(self._records.values())

    def list_plugins(self) -> Tuple[BackendPluginRecord, ...]:
        return self.list()

    def _resolve(self, identifier: str) -> BackendPluginRecord:
        if identifier in self._records:
            return self._records[identifier]
        matches = tuple(
            record
            for record in self._records.values()
            if record.registry_key == identifier
        )
        if not matches:
            raise BackendPluginSelectionError(
                f"Unknown backend plugin record '{identifier}'",
                field="registry_key",
                expected="an existing record_id or unique registry_key",
                actual=identifier,
                remediation=(
                    "Call registry.list() and use one of the reported record_id "
                    "or registry_key values."
                ),
            )
        if len(matches) > 1:
            raise BackendPluginConflictError(
                f"Backend plugin key '{identifier}' is ambiguous across records: "
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

    def inspect(self, identifier: str) -> BackendPluginRecord:
        """Return one record without validating or importing it."""
        with self._lock:
            self.discover()
            return self._resolve(identifier)

    def conflicts(self) -> ConflictReport:
        """Return deterministic W7 static conflicts without loading plugins."""
        with self._lock:
            self.discover()
            return detect_conflicts(self._records.values())

    def _replace(self, record: BackendPluginRecord) -> BackendPluginRecord:
        self._records[record.record_id] = record
        for target, decision in tuple(self._selections.items()):
            if decision.record_id == record.record_id:
                self._selections[target] = replace(
                    decision,
                    record=record,
                )
        return record

    def _reject(
        self,
        record: BackendPluginRecord,
        error: BackendPluginError,
        compatibility_status: PluginCompatibilityStatus,
    ) -> BackendPluginRecord:
        return self._replace(
            replace(
                record,
                state=PluginLifecycleState.REJECTED,
                compatibility_status=compatibility_status,
                errors=record.errors + (error,),
            )
        )

    def _reject_conflicted(
        self,
        conflict,
        error: BackendPluginError,
    ) -> None:
        """Mark every record involved in one fatal conflict REJECTED."""
        for conflicted_id in conflict.record_ids:
            conflicted = self._records.get(conflicted_id)
            if (
                conflicted is not None
                and conflicted.state is not PluginLifecycleState.REJECTED
            ):
                self._reject(
                    conflicted,
                    error,
                    conflicted.compatibility_status,
                )

    def _reject_all_fatal_conflicts(self, records) -> None:
        """Mark every record in any fatal static conflict REJECTED."""
        for conflict in detect_conflicts(records).fatal_conflicts:
            self._reject_conflicted(conflict, conflict.to_error())

    def _reject_manifest_scope(
        self,
        error: BackendPluginError,
        compatibility_status: PluginCompatibilityStatus,
        *,
        distribution: Any = None,
        specialize_error: bool = False,
    ) -> None:
        """Fail closed for one shared validation scope without importing."""
        for candidate in tuple(self._records.values()):
            if (
                candidate.source is not PluginSource.MANIFEST
                or candidate.state is not PluginLifecycleState.DISCOVERED
                or (
                    distribution is not None
                    and candidate.distribution is not distribution
                )
            ):
                continue
            candidate_error = error
            if (
                specialize_error
                and isinstance(error, BackendPluginCompatibilityError)
            ):
                candidate_error = BackendPluginCompatibilityError(
                    error.dimension,
                    error.expected,
                    error.actual,
                    plugin_id=candidate.plugin_id,
                    entry_point=candidate.entry_point_name,
                    remediation=error.remediation,
                )
            self._reject(
                candidate,
                candidate_error,
                compatibility_status,
            )

    def _validate_record(
        self, record: BackendPluginRecord
    ) -> BackendPluginRecord:
        if record.state is PluginLifecycleState.REJECTED:
            return record
        if record.source is PluginSource.LEGACY:
            return record
        if record.state is not PluginLifecycleState.DISCOVERED:
            return record
        if record.manifest is None:
            error = BackendPluginManifestError(
                "Manifest plugin record has no parsed Manifest",
                entry_point=record.entry_point_name,
                field="manifest",
                expected="a parsed BackendPluginManifest",
                actual="<missing>",
                remediation="Repair the backend Manifest and rediscover it.",
            )
            return self._reject(
                record, error, PluginCompatibilityStatus.NOT_CHECKED
            )

        try:
            environment = self._get_environment()
        except BackendPluginError as exc:
            # CoreEnvironment is process-wide.  Preserve it as a Registry-level
            # diagnostic while ensuring no Manifest record can bypass the gate.
            self._reject_manifest_scope(
                exc,
                PluginCompatibilityStatus.INCOMPATIBLE,
            )
            return self._records[record.record_id]
        outcome = evaluate_record_preflight(
            record,
            environment,
            preflight_profile=self._preflight_profile,
            core_abi_fingerprint=self._core_abi_fingerprint,
            supported_tags=self._supported_tags,
            core_capabilities=self._core_capabilities,
            compatibility_validator=validate_backend_plugin,
            triton_version_validator=validate_triton_version_requirement,
            capability_validator=validate_plugin_capabilities,
        )
        if outcome.error is not None:
            if outcome.reject_distribution:
                self._reject_manifest_scope(
                    outcome.error,
                    outcome.compatibility_status,
                    distribution=record.distribution,
                    specialize_error=outcome.specialize_error,
                )
                return self._records[record.record_id]
            return self._reject(
                record,
                outcome.error,
                outcome.compatibility_status,
            )

        if not can_transition(
            record.state, PluginLifecycleState.VALIDATED, PluginSource.MANIFEST
        ):
            raise BackendPluginLifecycleError(
                f"Cannot validate backend plugin from state {record.state.value}",
                plugin_id=record.plugin_id,
                entry_point=record.entry_point_name,
            )
        return self._replace(
            replace(
                record,
                state=PluginLifecycleState.VALIDATED,
                compatibility_status=PluginCompatibilityStatus.COMPATIBLE,
                compatibility_report=outcome.compatibility_report,
                capability_report=outcome.capability_report,
            )
        )

    def validate(
        self,
        identifier: Optional[str] = None,
        *,
        strict: bool = False,
    ) -> Any:
        """Validate one record, or all records with per-plugin isolation."""
        with self._lock:
            self._ensure_not_resetting("validate")
            self.discover()
            if identifier is not None:
                record = self._validate_record(self._resolve(identifier))
                if record.state is PluginLifecycleState.REJECTED and record.error:
                    raise record.error
                return record

            for record in tuple(self._records.values()):
                self._validate_record(record)
            records = self.list()
            if strict:
                for record in records:
                    if (
                        record.state is PluginLifecycleState.REJECTED
                        and record.error is not None
                    ):
                        raise record.error
            return records

    def _pre_import_fatal_conflict_gate(
        self,
        record: BackendPluginRecord,
    ) -> BackendPluginRecord:
        """Reject fatal static conflicts involving one record before import."""
        if (
            record.source is not PluginSource.MANIFEST
            or record.state
            in {
                PluginLifecycleState.REJECTED,
                PluginLifecycleState.LOADED,
                PluginLifecycleState.REGISTERED,
                PluginLifecycleState.SELECTED,
                PluginLifecycleState.ACTIVE,
            }
        ):
            return record

        record = self._validate_record(record)
        if record.state is PluginLifecycleState.REJECTED:
            return record

        if (
            record.manifest is not None
            and record.manifest.isolation_mode
            is PluginIsolationMode.NATIVE_IN_PROCESS
        ):
            for candidate in tuple(self._records.values()):
                if (
                    candidate.record_id == record.record_id
                    or candidate.source is not PluginSource.MANIFEST
                    or candidate.state is not PluginLifecycleState.DISCOVERED
                    or candidate.manifest is None
                    or candidate.manifest.isolation_mode
                    is not PluginIsolationMode.NATIVE_IN_PROCESS
                ):
                    continue
                self._validate_record(candidate)

        record = self._records[record.record_id]
        for conflict in detect_conflicts(
            self._records.values()
        ).fatal_conflicts:
            if record.record_id in conflict.record_ids:
                error = conflict.to_error()
                self._reject_conflicted(conflict, error)
                raise error
        return record

    def _transition(
        self,
        record: BackendPluginRecord,
        target: PluginLifecycleState,
    ) -> None:
        _ensure_transition(record, target, transition_allowed=can_transition)

    def load(self, identifier: str) -> BackendPluginRecord:
        """Import one plugin only after its applicable pre-load gate passes."""
        with self._lock:
            self._ensure_not_resetting("load")
            self.discover()
            record = self._resolve(identifier)
            record = self._pre_import_fatal_conflict_gate(record)

            if record.state is PluginLifecycleState.REJECTED:
                if record.error is not None:
                    raise record.error
                raise BackendPluginLoadError(
                    "Rejected backend plugin cannot be loaded",
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                )
            if record.state in {
                PluginLifecycleState.LOADED,
                PluginLifecycleState.REGISTERED,
                PluginLifecycleState.SELECTED,
                PluginLifecycleState.ACTIVE,
            }:
                return record

            if record.record_id in self._loading:
                raise BackendPluginLifecycleError(
                    "Recursive backend plugin load is not allowed",
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                    field="load",
                    expected="one non-reentrant load operation",
                    actual="recursive load",
                    remediation=(
                        "Do not call Registry load/register for the same plugin "
                        "from its entry-point loader or constructor."
                    ),
                )
            self._transition(record, PluginLifecycleState.LOADED)
            self._loading.add(record.record_id)
            try:
                try:
                    plugin = _load_plugin_object(record.entry_point)
                except Exception as exc:
                    error = BackendPluginLoadError(
                        f"Failed to load backend plugin "
                        f"'{record.registry_key}': {exc}",
                        plugin_id=record.plugin_id,
                        entry_point=record.entry_point_name,
                        field="entry_point.load",
                        expected="a loadable backend plugin object",
                        actual=f"<error: {exc}>",
                        remediation=(
                            "Fix the backend Python package or native "
                            "dependencies and reinstall it."
                        ),
                    )
                    current = self._records.get(record.record_id, record)
                    self._reject(
                        current,
                        error,
                        current.compatibility_status,
                    )
                    raise error from exc

                current = self._records.get(record.record_id)
                if (
                    current is None
                    or current.state is PluginLifecycleState.REJECTED
                ):
                    error = (
                        current.error
                        if current is not None and current.error is not None
                        else BackendPluginLifecycleError(
                            "Backend plugin state changed while loading",
                            plugin_id=record.plugin_id,
                            entry_point=record.entry_point_name,
                            field="state",
                            expected=record.state.value,
                            actual=(
                                current.state.value
                                if current is not None
                                else "<record removed>"
                            ),
                            remediation=(
                                "Do not reset or mutate the Registry from "
                                "plugin load callbacks."
                            ),
                        )
                    )
                    raise error

                return self._replace(
                    replace(
                        current,
                        state=PluginLifecycleState.LOADED,
                        plugin_object=plugin,
                    )
                )
            finally:
                self._loading.discard(record.record_id)

    def register(
        self,
        identifier: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> BackendPluginRecord:
        """Check the runtime pair, initialize once, and register one plugin."""
        with self._lock:
            self._ensure_not_resetting("register")
            record = self.load(identifier)
            if record.state in {
                PluginLifecycleState.REGISTERED,
                PluginLifecycleState.SELECTED,
                PluginLifecycleState.ACTIVE,
            }:
                return record
            if record.record_id in self._registering:
                raise BackendPluginLifecycleError(
                    "Recursive backend plugin registration is not allowed",
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                    field="register",
                    expected="one non-reentrant registration operation",
                    actual="recursive registration",
                    remediation=(
                        "Do not call Registry register for the same plugin from "
                        "runtime attributes or initialize()."
                    ),
                )
            self._transition(record, PluginLifecycleState.REGISTERED)
            self._registering.add(record.record_id)
            try:
                plugin = record.plugin_object
                interface = _inspect_runtime_interfaces(plugin)
                if (
                    interface.missing_fields
                    or interface.invalid_fields
                    or interface.field_errors
                ):
                    error = BackendPluginInterfaceError(
                        interface.missing_fields,
                        invalid_fields=interface.invalid_fields,
                        field_errors=dict(interface.field_errors),
                        plugin_id=record.plugin_id,
                        entry_point=record.entry_point_name,
                    )
                    self._reject(record, error, record.compatibility_status)
                    raise error

                compiler_cls = interface.compiler_cls
                driver_cls = interface.driver_cls
                initialized = False
                if record.source is PluginSource.MANIFEST:
                    try:
                        initializer = getattr(plugin, "initialize", None)
                    except Exception as exc:
                        error = BackendPluginLifecycleError(
                            f"Unable to inspect backend initialize hook: {exc}",
                            plugin_id=record.plugin_id,
                            entry_point=record.entry_point_name,
                            field="initialize",
                            expected="a callable hook or no hook",
                            actual=f"<error: {exc}>",
                            remediation=(
                                "Fix the initialize attribute so it is safely "
                                "readable and callable."
                            ),
                        )
                        self._reject(record, error, record.compatibility_status)
                        raise error from exc
                    if initializer is not None and not callable(initializer):
                        error = BackendPluginLifecycleError(
                            "Backend plugin initialize hook is not callable",
                            plugin_id=record.plugin_id,
                            entry_point=record.entry_point_name,
                            field="initialize",
                            expected="a callable hook or no hook",
                            actual=type(initializer).__name__,
                            remediation=(
                                "Expose initialize(context) as a callable or "
                                "remove the optional hook."
                            ),
                        )
                        self._reject(record, error, record.compatibility_status)
                        raise error
                    if callable(initializer):
                        try:
                            initialization_context = dict(
                                context
                                if context is not None
                                else {
                                    "environment": self._get_environment(),
                                    "manifest": record.manifest,
                                    "record_id": record.record_id,
                                }
                            )
                            initializer(initialization_context)
                            initialized = True
                        except Exception as exc:
                            shutdown_called = _best_effort_shutdown(plugin)
                            current = replace(
                                self._records.get(record.record_id, record),
                                shutdown_called=shutdown_called,
                            )
                            self._replace(current)
                            error = BackendPluginLifecycleError(
                                f"Backend plugin initialize() failed: {exc}",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="initialize",
                                expected="successful initialization",
                                actual=f"<error: {exc}>",
                                remediation=(
                                    "Fix initialize(), release partial "
                                    "resources, and reinstall the backend."
                                ),
                            )
                            self._reject(
                                current,
                                error,
                                current.compatibility_status,
                            )
                            raise error from exc

                current = self._records.get(record.record_id)
                if (
                    current is None
                    or current.state is PluginLifecycleState.REJECTED
                ):
                    error = (
                        current.error
                        if current is not None and current.error is not None
                        else BackendPluginLifecycleError(
                            "Backend plugin state changed while registering",
                            plugin_id=record.plugin_id,
                            entry_point=record.entry_point_name,
                            field="state",
                            expected=PluginLifecycleState.LOADED.value,
                            actual=(
                                current.state.value
                                if current is not None
                                else "<record removed>"
                            ),
                            remediation=(
                                "Do not reset or mutate the Registry from "
                                "plugin lifecycle hooks."
                            ),
                        )
                    )
                    raise error

                return self._replace(
                    replace(
                        current,
                        state=PluginLifecycleState.REGISTERED,
                        compiler_cls=compiler_cls,
                        driver_cls=driver_cls,
                        initialized=initialized,
                    )
                )
            finally:
                self._registering.discard(record.record_id)

    def select(
        self,
        target: Any,
        *,
        kernel_required_capabilities: Iterable[str] = (),
        explicit_selector: Optional[str] = None,
        environment: Optional[Mapping[str, str]] = None,
    ) -> SelectionDecision:
        """Validate, choose, load, register, and mark one backend selected.

        Selection itself is static and import-free.  Only the winning record
        reaches ``register()``; losing candidates are never loaded.
        """
        with self._lock:
            self._ensure_not_resetting("select")
            records = self.validate()
            # Fatal conflicts must reject the involved records before the
            # static selection algorithm raises (selection itself is not
            # changed); this also covers compiler/runtime entry paths that
            # resolve through select().
            self._reject_all_fatal_conflicts(records)
            decision = select_backend(
                records,
                target=target,
                kernel_required_capabilities=(
                    kernel_required_capabilities
                ),
                core_provided_capabilities=self._core_capabilities,
                explicit_selector=explicit_selector,
                environment=(
                    os.environ if environment is None else environment
                ),
            )

            target_name = decision.target
            previous_decision = self._selections.get(target_name)
            if (
                previous_decision is not None
                and previous_decision.record_id != decision.record_id
            ):
                previous = self._records.get(previous_decision.record_id)
                if (
                    previous is not None
                    and previous.state is PluginLifecycleState.ACTIVE
                ):
                    raise BackendPluginLifecycleError(
                        "Cannot switch an ACTIVE backend selection",
                        plugin_id=previous.plugin_id,
                        entry_point=previous.entry_point_name,
                        field="active selection",
                        expected=previous.record_id,
                        actual=decision.record_id,
                        remediation=(
                            "Reset the Registry-backed runtime driver before "
                            "selecting a different plugin for this target."
                        ),
                    )

            selected = self.register(decision.record_id)
            if (
                previous_decision is not None
                and previous_decision.record_id != selected.record_id
            ):
                previous = self._records.get(previous_decision.record_id)
                if previous is not None:
                    remaining_targets = tuple(
                        item
                        for item in previous.selected_targets
                        if item != target_name
                    )
                    previous_state = previous.state
                    if (
                        previous_state
                        in {
                            PluginLifecycleState.SELECTED,
                            PluginLifecycleState.ACTIVE,
                        }
                        and not remaining_targets
                    ):
                        self._transition(
                            previous,
                            PluginLifecycleState.REGISTERED,
                        )
                        previous_state = PluginLifecycleState.REGISTERED
                    self._replace(
                        replace(
                            previous,
                            state=previous_state,
                            selected_targets=remaining_targets,
                        )
                    )

            selected = self._records[selected.record_id]
            selected_targets = tuple(
                sorted(set(selected.selected_targets).union({target_name}))
            )
            selected_state = selected.state
            if selected_state is PluginLifecycleState.REGISTERED:
                self._transition(
                    selected,
                    PluginLifecycleState.SELECTED,
                )
                selected_state = PluginLifecycleState.SELECTED
            elif selected_state not in {
                PluginLifecycleState.SELECTED,
                PluginLifecycleState.ACTIVE,
            }:
                raise BackendPluginLifecycleError(
                    "Selected backend is not registered",
                    plugin_id=selected.plugin_id,
                    entry_point=selected.entry_point_name,
                    field="state",
                    expected="registered, selected, or active",
                    actual=selected_state.value,
                    remediation=(
                        "Use Registry.select() so validation, loading, and "
                        "registration complete before state selection."
                    ),
                )

            selected = self._replace(
                replace(
                    selected,
                    state=selected_state,
                    selected_targets=selected_targets,
                )
            )
            decision = replace(decision, record=selected)
            self._selections[target_name] = decision
            self._generation += 1
            return decision

    def get_selection(self, target: str) -> Optional[SelectionDecision]:
        """Return the cached selection for one target without loading."""
        with self._lock:
            return self._selections.get(target)

    def activate(self, identifier: str) -> BackendPluginRecord:
        """Mark one selected runtime pair active without reloading it.

        W9 calls this only after the selected driver class has been
        instantiated successfully.  Activation is process-wide: a second
        active Registry record is a conflict rather than an implicit switch.
        """
        with self._lock:
            self._ensure_not_resetting("activate")
            self.discover()
            record = self._resolve(identifier)
            other_active = tuple(
                candidate
                for candidate in self._records.values()
                if (
                    candidate.record_id != record.record_id
                    and candidate.state is PluginLifecycleState.ACTIVE
                )
            )
            if other_active:
                active_ids = tuple(
                    sorted(candidate.record_id for candidate in other_active)
                )
                raise BackendPluginConflictError(
                    "Cannot activate more than one backend plugin: "
                    + ", ".join(active_ids + (record.record_id,)),
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                    field="active",
                    expected="one ACTIVE backend plugin",
                    actual=", ".join(active_ids + (record.record_id,)),
                    remediation=(
                        "Keep the current active driver or explicitly reset "
                        "runtime state before activating another backend."
                    ),
                )

            if record.state is PluginLifecycleState.ACTIVE:
                return record
            if record.state is not PluginLifecycleState.SELECTED:
                raise BackendPluginLifecycleError(
                    "Backend plugin must be selected before activation",
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                    field="state",
                    expected=PluginLifecycleState.SELECTED.value,
                    actual=record.state.value,
                    remediation=(
                        "Resolve the backend through Registry.select() before "
                        "constructing and activating its runtime driver."
                    ),
                )

            self._transition(record, PluginLifecycleState.ACTIVE)
            return self._replace(
                replace(record, state=PluginLifecycleState.ACTIVE)
            )

    def diagnostics(self, identifier: Optional[str] = None) -> Any:
        """Return static state and optional loaded-plugin diagnostics."""
        with self._lock:
            self.discover()
            records = (
                (self._resolve(identifier),)
                if identifier is not None
                else tuple(self._records.values())
            )
            results = []
            for record in records:
                result = record.to_dict()
                result["plugin_diagnostics"] = None
                try:
                    diagnostics = (
                        getattr(record.plugin_object, "diagnostics", None)
                        if record.plugin_object is not None
                        else None
                    )
                except Exception as exc:
                    diagnostics = None
                    result["plugin_diagnostics"] = {
                        "error": BackendPluginLifecycleError(
                            f"Unable to inspect backend diagnostics hook: {exc}",
                            plugin_id=record.plugin_id,
                            entry_point=record.entry_point_name,
                            field="diagnostics",
                            expected="a callable hook or no hook",
                            actual=f"<error: {exc}>",
                            remediation=(
                                "Fix the diagnostics attribute so inspection "
                                "does not raise."
                            ),
                        ).to_dict()
                    }
                if callable(diagnostics):
                    try:
                        value = diagnostics()
                        result["plugin_diagnostics"] = (
                            dict(value) if isinstance(value, Mapping) else value
                        )
                    except Exception as exc:
                        result["plugin_diagnostics"] = {
                            "error": BackendPluginLifecycleError(
                                f"Backend plugin diagnostics() failed: {exc}",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="diagnostics",
                                expected="a diagnostic result",
                                actual=f"<error: {exc}>",
                                remediation="Fix diagnostics() so it is read-only.",
                            ).to_dict()
                        }
                results.append(result)
            if identifier is not None:
                return results[0]
            return {
                "preflight_profile": self._preflight_profile,
                "core_abi_fingerprint": (
                    self._environment.core_abi_fingerprint
                    if self._environment is not None
                    else None
                ),
                "registry_errors": [
                    error.to_dict() for error in self._registry_errors
                ],
                "conflicts": self.conflicts().to_dict(),
                "selections": {
                    target: decision.to_dict()
                    for target, decision in sorted(self._selections.items())
                },
                "plugins": results,
            }

    def reset(self) -> Tuple[BackendPluginError, ...]:
        """Best-effort shutdown and clear state; Python modules stay imported."""
        shutdown_errors = []
        reset_hooks: Tuple[Callable[[], None], ...] = ()
        shutdown_records: Tuple[BackendPluginRecord, ...] = ()
        reset_started = False
        try:
            with self._lock:
                self._ensure_not_resetting("reset")
                if self._loading or self._registering:
                    raise BackendPluginLifecycleError(
                        "Cannot reset the Registry during plugin load or "
                        "registration",
                        field="reset",
                        expected="no in-progress plugin lifecycle operation",
                        actual=(
                            "loading="
                            + ",".join(sorted(self._loading))
                            + "; registering="
                            + ",".join(sorted(self._registering))
                        ),
                        remediation=(
                            "Complete the current load/register callback before "
                            "resetting the Registry."
                        ),
                )
                self._resetting = True
                reset_started = True
                shutdown_records = tuple(
                    record
                    for record in self._records.values()
                    if (
                        record.source is PluginSource.MANIFEST
                        and record.plugin_object is not None
                        and not record.shutdown_called
                    )
                )
                reset_hooks = tuple(self._reset_hooks)
                self._records.clear()
                self._registry_errors = ()
                self._environment = None
                self._environment_error = None
                self._discovered = False
                self._loading.clear()
                self._registering.clear()
                self._selections.clear()
                self._generation += 1

            # W9 hooks may hold their own lazy-cache locks.  Run them after
            # releasing the Registry lock to avoid Registry/LazyProxy lock
            # inversion; _resetting still rejects lifecycle re-entry.
            for callback in reset_hooks:
                try:
                    callback()
                except BackendPluginError as exc:
                    shutdown_errors.append(exc)
                except Exception as exc:
                    shutdown_errors.append(
                        BackendPluginLifecycleError(
                            f"Backend reset hook failed: {exc}",
                            field="reset_hook",
                            expected="successful cache cleanup",
                            actual=f"<error: {exc}>",
                            remediation=(
                                "Fix the W9 adapter reset hook so it only "
                                "clears local compiler/driver caches."
                                ),
                            )
                        )

            # User shutdown hooks may acquire plugin-owned locks or attempt
            # Registry re-entry.  They also run outside the Registry lock;
            # _resetting makes any lifecycle re-entry fail immediately.
            for record in shutdown_records:
                try:
                    shutdown = getattr(
                        record.plugin_object,
                        "shutdown",
                        None,
                    )
                except Exception as exc:
                    shutdown_errors.append(
                        BackendPluginLifecycleError(
                            "Unable to inspect backend shutdown hook: "
                            f"{exc}",
                            plugin_id=record.plugin_id,
                            entry_point=record.entry_point_name,
                            field="shutdown",
                            expected="a callable hook or no hook",
                            actual=f"<error: {exc}>",
                            remediation=(
                                "Fix the shutdown attribute so reset can "
                                "inspect it safely."
                            ),
                        )
                    )
                    shutdown = None
                if callable(shutdown):
                    try:
                        shutdown()
                    except BackendPluginError as exc:
                        shutdown_errors.append(exc)
                    except Exception as exc:
                        shutdown_errors.append(
                            BackendPluginLifecycleError(
                                "Backend plugin shutdown() failed: "
                                f"{exc}",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="shutdown",
                                expected="successful cleanup",
                                actual=f"<error: {exc}>",
                                remediation=(
                                    "Fix shutdown() so registry reset can "
                                    "release plugin-owned resources."
                                ),
                            )
                        )
            return tuple(shutdown_errors)
        finally:
            if reset_started:
                with self._lock:
                    self._resetting = False


backend_plugin_registry = BackendPluginRegistry()


def get_backend_plugin_registry() -> BackendPluginRegistry:
    """Return the process-wide Registry without triggering discovery."""
    return backend_plugin_registry
