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
from ._registry_catalog import RegistryCatalog as _RegistryCatalog
from ._registry_discovery import (
    BACKEND_ENTRY_POINT_GROUP as _BACKEND_ENTRY_POINT_GROUP,
    copy_manifest_error as _copy_manifest_error,
    discover_distributions as _discover_distributions,
    distribution_identity as _distribution_identity,
    entry_point_name as _entry_point_name,
    entry_point_value as _entry_point_value,
    source_hint as _source_hint,
)
from ._registry_lifecycle import (
    best_effort_shutdown as _best_effort_shutdown,
    ensure_transition as _ensure_transition,
    inspect_runtime_interfaces as _inspect_runtime_interfaces,
    load_plugin_object as _load_plugin_object,
)
from ._registry_lifecycle_selection import (
    plan_activation as _plan_activation,
    plan_load_precheck as _plan_load_precheck,
    plan_previous_selection_release as _plan_previous_selection_release,
    plan_register_precheck as _plan_register_precheck,
    plan_selection_switch as _plan_selection_switch,
    plan_winner_selection as _plan_winner_selection,
)
from ._registry_preflight import evaluate_record_preflight
from ._registry_selection_state import (
    SelectionState as _SelectionState,
    materialize_selection_decision as _materialize_selection_decision,
    selection_state_from_decision as _selection_state_from_decision,
)
from ._registry_state import RegistryState as _RegistryState
from ._registry_validator import (
    ValidationPlan as _ValidationPlan,
    plan_process_environment_failure as _plan_process_environment_failure,
    plan_record_validation as _plan_record_validation,
)
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
        self._state = _RegistryState()
        self._catalog = _RegistryCatalog(
            self._state,
            record_factory=lambda **fields: BackendPluginRecord(**fields),
        )
        self._lock = threading.RLock()

    def _ensure_not_resetting(self, operation: str) -> None:
        if self._state.resetting:
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
            self._state.add_reset_hook(callback)

    @property
    def generation(self) -> int:
        """Return the Registry epoch used to invalidate W9 adapter snapshots."""
        with self._lock:
            return self._state.generation

    def _get_environment(self) -> CoreEnvironment:
        if self._state.environment is not None:
            return self._state.environment
        if self._state.environment_error is not None:
            raise self._state.environment_error
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
            self._state.cache_environment_error(error)
            self._record_registry_error(error)
            raise error from exc
        if not isinstance(environment, CoreEnvironment):
            error = BackendPluginCompatibilityError(
                "Core environment metadata",
                "CoreEnvironment",
                type(environment).__name__,
                remediation="Return a CoreEnvironment from environment_provider.",
            )
            self._state.cache_environment_error(error)
            self._record_registry_error(error)
            raise error
        self._state.cache_environment(environment)
        return environment

    def _record_registry_error(self, error: BackendPluginError) -> None:
        self._state.append_registry_error(error)

    def _allocate_record_id(self, distribution_name: Optional[str], name: str) -> str:
        return self._catalog.allocate_record_id(distribution_name, name)

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
        return self._catalog.accept_discovery_result(
            entry_point=entry_point,
            distribution=distribution,
            source=source,
            manifest=manifest,
            state=state,
            compatibility_status=compatibility_status,
            error=error,
        )

    def discover(self, *, strict: bool = False) -> Tuple[BackendPluginRecord, ...]:
        """Discover backend metadata without importing backend code."""
        with self._lock:
            self._ensure_not_resetting("discover")
            if self._state.discovered:
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
                self._state.replace_registry_errors((error,))
                raise error from exc

            _discover_distributions(
                distributions,
                store_record=self._store_record,
                record_registry_error=self._record_registry_error,
                load_manifest=load_distribution_manifest,
            )
            self._state.mark_discovered()
            records = self.list()
            if strict:
                self._raise_first_discovery_error(records)
            return records

    def _raise_first_discovery_error(
        self, records: Tuple[BackendPluginRecord, ...]
    ) -> None:
        registry_errors = self._state.registry_errors_snapshot()
        if registry_errors:
            raise registry_errors[0]
        for record in records:
            if record.state is PluginLifecycleState.REJECTED and record.error:
                raise record.error

    def list(self) -> Tuple[BackendPluginRecord, ...]:
        """Return immutable record snapshots without validation or loading."""
        with self._lock:
            if not self._state.discovered:
                self.discover()
            return self._catalog.list_snapshot()

    def list_plugins(self) -> Tuple[BackendPluginRecord, ...]:
        return self.list()

    def _resolve(self, identifier: str) -> BackendPluginRecord:
        return self._catalog.resolve(identifier)

    def inspect(self, identifier: str) -> BackendPluginRecord:
        """Return one record without validating or importing it."""
        with self._lock:
            self.discover()
            return self._resolve(identifier)

    def conflicts(self) -> ConflictReport:
        """Return deterministic W7 static conflicts without loading plugins."""
        with self._lock:
            self.discover()
            return self._catalog.conflict_view()

    def _replace(self, record: BackendPluginRecord) -> BackendPluginRecord:
        return self._state.replace_record(record)

    def _materialize_selection(
        self,
        selection: _SelectionState,
    ) -> SelectionDecision:
        return _materialize_selection_decision(
            selection,
            self._state.get_record(selection.record_id),
        )

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
            conflicted = self._state.get_record(conflicted_id)
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
        for candidate in self._state.records_snapshot():
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

    def _apply_validation_rejection(
        self,
        record: BackendPluginRecord,
        plan: _ValidationPlan,
    ) -> BackendPluginRecord:
        error = plan.error
        if error is None:
            return record
        if plan.reject_process:
            self._reject_manifest_scope(
                error,
                plan.compatibility_status,
            )
            return self._state.get_record(record.record_id)
        if plan.reject_distribution:
            self._reject_manifest_scope(
                error,
                plan.compatibility_status,
                distribution=plan.distribution,
                specialize_error=plan.specialize_error,
            )
            return self._state.get_record(record.record_id)
        return self._reject(
            record,
            error,
            plan.compatibility_status,
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
            plan = _plan_process_environment_failure(record, exc)
            return self._apply_validation_rejection(record, plan)
        plan = _plan_record_validation(
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
        if plan.error is not None:
            return self._apply_validation_rejection(record, plan)

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
                compatibility_report=plan.compatibility_report,
                capability_report=plan.capability_report,
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

            for record in self._state.records_snapshot():
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
            for candidate in self._state.records_snapshot():
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

        record = self._state.get_record(record.record_id)
        for conflict in detect_conflicts(
            self._state.records_snapshot()
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
            plan = _plan_load_precheck(
                record,
                is_loading=self._state.is_loading(record.record_id),
                transition_validator=self._transition,
            )
            if plan.return_existing:
                return record
            self._state.begin_loading(record.record_id)
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
                    current = self._state.get_record(record.record_id)
                    if current is None:
                        current = record
                    self._reject(
                        current,
                        error,
                        current.compatibility_status,
                    )
                    raise error from exc

                current = self._state.get_record(record.record_id)
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
                self._state.end_loading(record.record_id)

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
            plan = _plan_register_precheck(
                record,
                is_registering=self._state.is_registering(record.record_id),
                transition_validator=self._transition,
            )
            if plan.return_existing:
                return record
            self._state.begin_registering(record.record_id)
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
                            current = self._state.get_record(record.record_id)
                            if current is None:
                                current = record
                            current = replace(
                                current, shutdown_called=shutdown_called
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

                current = self._state.get_record(record.record_id)
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
                self._state.end_registering(record.record_id)

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
            previous_selection = self._state.get_selection_state(target_name)
            previous_record_id = (
                previous_selection.record_id
                if previous_selection is not None
                else None
            )
            previous = (
                self._state.get_record(previous_record_id)
                if previous_record_id is not None
                and previous_record_id != decision.record_id
                else None
            )
            switch_plan = _plan_selection_switch(
                target=target_name,
                winner_record_id=decision.record_id,
                previous_record_id=previous_record_id,
                previous_record=previous,
            )

            selected = self.register(decision.record_id)
            if switch_plan.changes_winner:
                previous = self._state.get_record(
                    switch_plan.previous_record_id
                )
                release_plan = _plan_previous_selection_release(
                    target=target_name,
                    winner_record_id=selected.record_id,
                    previous_record=previous,
                    transition_validator=self._transition,
                )
                if release_plan is not None:
                    self._replace(
                        replace(
                            previous,
                            state=release_plan.state,
                            selected_targets=release_plan.selected_targets,
                        )
                    )

            selected = self._state.get_record(selected.record_id)
            winner_plan = _plan_winner_selection(
                target=target_name,
                winner=selected,
                transition_validator=self._transition,
            )
            selected = self._replace(
                replace(
                    selected,
                    state=winner_plan.state,
                    selected_targets=winner_plan.selected_targets,
                )
            )
            decision = replace(decision, record=selected)
            selection = _selection_state_from_decision(decision)
            self._state.set_selection_state(target_name, selection)
            if winner_plan.increment_generation:
                self._state.increment_generation()
            return self._materialize_selection(selection)

    def get_selection(self, target: str) -> Optional[SelectionDecision]:
        """Return the cached selection for one target without loading."""
        with self._lock:
            selection = self._state.get_selection_state(target)
            return (
                None
                if selection is None
                else self._materialize_selection(selection)
            )

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
            plan = _plan_activation(
                record,
                self._state.records_snapshot(),
                transition_validator=self._transition,
            )
            if plan.return_existing:
                return record
            return self._replace(
                replace(record, state=plan.transition_to)
            )

    def diagnostics(self, identifier: Optional[str] = None) -> Any:
        """Return static state and optional loaded-plugin diagnostics."""
        with self._lock:
            self.discover()
            records = (
                (self._resolve(identifier),)
                if identifier is not None
                else self._state.records_snapshot()
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
                    self._state.environment.core_abi_fingerprint
                    if self._state.environment is not None
                    else None
                ),
                "registry_errors": [
                    error.to_dict()
                    for error in self._state.registry_errors_snapshot()
                ],
                "conflicts": self.conflicts().to_dict(),
                "selections": {
                    target: self._materialize_selection(selection).to_dict()
                    for target, selection in sorted(
                        self._state.selection_states_snapshot()
                    )
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
                loading = self._state.loading_snapshot()
                registering = self._state.registering_snapshot()
                if loading or registering:
                    raise BackendPluginLifecycleError(
                        "Cannot reset the Registry during plugin load or "
                        "registration",
                        field="reset",
                        expected="no in-progress plugin lifecycle operation",
                        actual=(
                            "loading="
                            + ",".join(sorted(loading))
                            + "; registering="
                            + ",".join(sorted(registering))
                        ),
                        remediation=(
                            "Complete the current load/register callback before "
                            "resetting the Registry."
                        ),
                )
                self._state.begin_reset()
                reset_started = True
                shutdown_records = tuple(
                    record
                    for record in self._state.records_snapshot()
                    if (
                        record.source is PluginSource.MANIFEST
                        and record.plugin_object is not None
                        and not record.shutdown_called
                    )
                )
                reset_hooks = self._state.reset_hooks_snapshot()
                self._state.clear_for_reset()
                self._state.increment_generation()

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
                    self._state.end_reset()


backend_plugin_registry = BackendPluginRegistry()


def get_backend_plugin_registry() -> BackendPluginRegistry:
    """Return the process-wide Registry without triggering discovery."""
    return backend_plugin_registry
