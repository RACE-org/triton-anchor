"""Stateful, import-gated registry for out-of-tree backend plugins."""

from __future__ import annotations

import importlib.metadata
import inspect
import os
import threading
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from types import CoroutineType, MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
)

from packaging.tags import Tag
from packaging.utils import canonicalize_name

from .capabilities import (
    CapabilityReport,
    _capability_error,
    evaluate_capabilities,
    evaluate_plugin_capabilities,
)
from .compatibility import (
    CompatibilityReport,
    _evaluate_backend_plugin_compatibility,
    _sort_backend_plugin_errors,
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
from .manifest import (
    MANIFEST_FILENAME,
    BackendPluginManifest,
    load_distribution_manifest,
    operational_record_manifest_error,
    stable_manifest_actual,
)
from .interfaces import (
    BackendPluginInterfaceIssue,
    RuntimeInterfaceMember,
    RuntimePairValidationContext,
    RuntimePairValidationResult,
)
from .native import NativeInspectionReport, inspect_native_artifacts
from .protocol import (
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
    can_transition,
)
from .selection import SelectionDecision, select_backend


_BACKEND_ENTRY_POINT_GROUP = "triton.backends"
_PREFLIGHT_PROFILES = {"triton_version", "full"}
_UNSET = object()
_MISSING = object()


def _stable_exception_name(error: BaseException) -> str:
    error_type = type(error)
    try:
        module = getattr(error_type, "__module__", None)
        qualname = getattr(error_type, "__qualname__", error_type.__name__)
    except BaseException:
        return "builtins.BaseException"
    return f"{module}.{qualname}" if module else qualname


def _stable_exception_actual(error: BaseException) -> str:
    return f"<error: {_stable_exception_name(error)}>"


def _stable_type_name(value: Any) -> str:
    """Return a deterministic runtime type name without calling ``repr``."""
    value_type = type(value)
    try:
        module = getattr(value_type, "__module__", None)
        qualname = getattr(value_type, "__qualname__", value_type.__name__)
    except BaseException:
        return "<unreadable runtime type>"
    return f"{module}.{qualname}" if module else qualname


def _close_unawaited(value: Any) -> None:
    """Best-effort close a native coroutine without running or awaiting it."""
    if inspect.iscoroutine(value):
        try:
            value.close()
        except BaseException:
            # The synchronous contract error remains primary. A plugin-owned
            # coroutine must not strand a Registry attempt or stop reset.
            pass


def _run_runtime_pair_validators(
    validators: Tuple[Tuple[str, Callable[[Any], Any]], ...],
    context: RuntimePairValidationContext,
) -> Tuple[
    Tuple[str, ...],
    Tuple[RuntimeInterfaceMember, ...],
    Tuple[BackendPluginInterfaceIssue, ...],
]:
    """Invoke trusted integration validators without masking programming bugs."""
    contract_ids = []
    required_surface = []
    issues = []
    for name, validator in validators:
        result = validator(context)
        if type(result) is CoroutineType:
            _close_unawaited(result)
        if type(result) is not RuntimePairValidationResult:
            raise TypeError(
                "runtime pair validator must return an exact "
                "RuntimePairValidationResult"
            )
        if result.contract_id != name:
            raise TypeError(
                "runtime pair validator result contract_id must match its "
                "registration name"
            )
        if type(result.required_surface) is not tuple or any(
            type(member) is not RuntimeInterfaceMember
            for member in result.required_surface
        ):
            raise TypeError("runtime pair validator returned an invalid surface")
        if type(result.issues) is not tuple or any(
            type(issue) is not BackendPluginInterfaceIssue
            for issue in result.issues
        ):
            raise TypeError("runtime pair validator returned invalid issues")
        contract_ids.append(name)
        required_surface.extend(result.required_surface)
        issues.extend(result.issues)
    return tuple(contract_ids), tuple(required_surface), tuple(issues)


def _freeze_json_value(value: Any) -> Any:
    """Freeze JSON-like extension values in an initialize context snapshot."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _manifest_snapshot(
    manifest: Optional[BackendPluginManifest],
) -> Optional[BackendPluginManifest]:
    if manifest is None:
        return None
    return replace(
        manifest,
        requires_triton=replace(
            manifest.requires_triton,
            extensions=_freeze_json_value(manifest.requires_triton.extensions),
        ),
        extensions=_freeze_json_value(manifest.extensions),
    )


def _error_identity(error: BackendPluginError) -> Tuple[Any, ...]:
    """Return a deterministic identity used to suppress repeated findings."""
    return (
        error.code,
        error.plugin_id,
        error.entry_point,
        error.field,
        getattr(error, "dimension", None),
        error.expected,
        error.actual,
        error.detail,
        getattr(error, "conflict_kind", None),
        getattr(error, "claim", None),
        tuple(getattr(error, "related_plugin_ids", ())),
        tuple(getattr(error, "related_record_ids", ())),
    )


@dataclass(frozen=True)
class _LifecycleWait:
    """One live thread-to-thread lifecycle dependency."""

    owner_thread: int
    operation: str
    record_id: str
    operation_token: object = field(repr=False, compare=False)
    wait_token: object = field(repr=False, compare=False)


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
        if type(self.manifest) is BackendPluginManifest:
            plugin_id = self.manifest.plugin_id
            return plugin_id if type(plugin_id) is str else None
        return None

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
            if type(self.manifest) is not BackendPluginManifest:
                manifest = {
                    "type": stable_manifest_actual(self.manifest),
                }
            else:
                isolation_mode = self.manifest.isolation_mode
                manifest = {
                    "plugin_id": _manifest_diagnostic_scalar(
                        self.manifest.plugin_id
                    ),
                    "entry_point": _manifest_diagnostic_scalar(
                        self.manifest.entry_point
                    ),
                    "backend_protocol": _manifest_diagnostic_scalar(
                        self.manifest.backend_protocol
                    ),
                    "targets": _manifest_diagnostic_sequence(
                        self.manifest.targets
                    ),
                    "capabilities": _manifest_diagnostic_sequence(
                        self.manifest.capabilities
                    ),
                    "requires_capabilities": _manifest_diagnostic_sequence(
                        self.manifest.requires_capabilities
                    ),
                    "isolation_mode": stable_manifest_actual(
                        isolation_mode
                    ),
                    "priority": _manifest_diagnostic_scalar(
                        self.manifest.priority
                    ),
                }
        return {
            "record_id": self.record_id,
            "registry_key": self.registry_key,
            "plugin_id": self.plugin_id,
            "entry_point": self.entry_point_name,
            "entry_point_value": self.entry_point_value,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "source": _source_diagnostic_value(self.source),
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


class _BackendPluginWaitCycleError(BackendPluginLifecycleError):
    """Private marker preserving a wait-cycle diagnostic through hooks."""

    def __init__(
        self,
        record: BackendPluginRecord,
        cycle_actual: str,
    ) -> None:
        self.cycle_actual = cycle_actual
        super().__init__(
            "Backend plugin lifecycle wait cycle detected",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="lifecycle.wait_cycle",
            expected="an acyclic load/register wait graph",
            actual=cycle_actual,
            remediation=(
                "Remove synchronous circular load/register dependencies from "
                "entry-point loaders, constructors, and initialize hooks."
            ),
        )

    def for_record(
        self,
        record: BackendPluginRecord,
    ) -> _BackendPluginWaitCycleError:
        return type(self)(record, self.cycle_actual)


def _manifest_diagnostic_scalar(value: Any) -> Any:
    """Keep exact JSON scalars; safely describe forged runtime objects."""
    if value is None or type(value) in {str, int, float, bool}:
        return value
    return stable_manifest_actual(value)


def _source_diagnostic_value(value: Any) -> Optional[str]:
    """Render a forged source without invoking ``repr`` or ``.value``."""
    if value is None:
        return None
    if type(value) is PluginSource:
        return value.value
    if type(value) is str:
        return value
    return stable_manifest_actual(value)


def _manifest_diagnostic_sequence(value: Any) -> Any:
    """Serialize an exact string tuple without invoking forged iterables."""
    if type(value) is tuple and all(type(item) is str for item in value):
        return list(value)
    return stable_manifest_actual(value)


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


def _distribution_identity(
    distribution: Any,
) -> Tuple[Optional[str], Optional[str]]:
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


def _entry_point_name(entry_point: Any) -> str:
    try:
        value = getattr(entry_point, "name", "")
        return str(value) if value is not None else ""
    except Exception:
        return ""


def _entry_point_value(entry_point: Any) -> str:
    try:
        value = getattr(entry_point, "value", None)
        return str(value) if value is not None else ""
    except Exception:
        return ""


def _source_hint(distribution: Any) -> Optional[PluginSource]:
    try:
        files = getattr(distribution, "files", None)
    except Exception:
        return None
    if files is None:
        return None
    try:
        if any(
            PurePosixPath(str(item)).name == MANIFEST_FILENAME
            for item in files
        ):
            return PluginSource.MANIFEST
    except Exception:
        return None
    return PluginSource.LEGACY


def _copy_manifest_error(
    error: BackendPluginError,
    *,
    entry_point: str,
    plugin_id: Optional[str] = None,
) -> BackendPluginManifestError:
    return BackendPluginManifestError(
        str(error),
        plugin_id=plugin_id if plugin_id is not None else error.plugin_id,
        entry_point=entry_point,
        detail=error.detail,
        field=error.field,
        expected=error.expected,
        actual=error.actual,
        remediation=error.remediation
        or "Fix the installed backend Manifest and reinstall its wheel.",
    )


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
        # In-flight values are ``(lifecycle_epoch, owner_thread, token)``.
        # The identity token prevents an old completion from removing a newer
        # attempt for the same deterministic record ID.
        self._loading: Dict[str, Tuple[int, int, object]] = {}
        self._registering: Dict[str, Tuple[int, int, object]] = {}
        self._diagnosing: Dict[str, Tuple[int, int, object]] = {}
        self._waiting_on: Dict[int, _LifecycleWait] = {}
        self._cleanup_stack: list[BackendPluginRecord] = []
        self._reset_operation_errors: list[BackendPluginError] = []
        self._selections: Dict[str, SelectionDecision] = {}
        self._reset_hooks: list = []
        self._runtime_pair_validators: Dict[str, Callable[[Any], Any]] = {}
        self._resetting = False
        self._lifecycle_epoch = 0
        self._generation = 0
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

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

    def register_runtime_pair_validator(
        self,
        name: str,
        validator: Callable[[RuntimePairValidationContext], Any],
    ) -> None:
        """Register one trusted, version-neutral pre-initialize validator.

        Registration is identity-idempotent and reset-stable.  A new contract
        cannot be added after a runtime pair has entered registration or been
        published, because doing so would leave an already-visible plugin
        outside the new trust boundary.
        """
        if type(name) is not str or not name.strip():
            raise TypeError("runtime pair validator name must be non-empty")
        if not callable(validator):
            raise TypeError("runtime pair validator must be callable")
        with self._condition:
            existing = self._runtime_pair_validators.get(name)
            if existing is validator:
                return
            if existing is not None:
                raise BackendPluginLifecycleError(
                    "Runtime pair validator name is already registered",
                    field="runtime_pair_validator",
                    expected="the original validator object for this name",
                    actual="a different validator object",
                    remediation=(
                        "Reuse the process-owned validator instance; never "
                        "replace a live runtime interface contract."
                    ),
                )
            self._ensure_not_resetting("register_runtime_pair_validator")
            published = tuple(
                sorted(
                    record.record_id
                    for record in self._records.values()
                    if record.state
                    in {
                        PluginLifecycleState.REGISTERED,
                        PluginLifecycleState.SELECTED,
                        PluginLifecycleState.ACTIVE,
                    }
                )
            )
            if self._registering or published:
                raise BackendPluginLifecycleError(
                    "Runtime pair validator registration is too late",
                    field="runtime_pair_validator",
                    expected=(
                        "contract registration before runtime pair validation "
                        "or publication"
                    ),
                    actual=(
                        "registration in progress"
                        if self._registering
                        else "runtime pair already published"
                    ),
                    remediation=(
                        "Register the Triton runtime interface contract during "
                        "adapter initialization, before selecting plugins."
                    ),
                )
            self._runtime_pair_validators[name] = validator

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
                f"<error: {_stable_exception_name(exc)}>",
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
        distribution_key = canonicalize_name(
            distribution_name or "unknown-distribution"
        )
        base = f"{distribution_key}:{name}"
        candidate = base
        suffix = 2
        while candidate in self._records:
            candidate = f"{base}#{suffix}"
            suffix += 1
        return candidate

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
        distribution_name, distribution_version = _distribution_identity(
            distribution
        )
        name = _entry_point_name(entry_point)
        record = BackendPluginRecord(
            record_id=self._allocate_record_id(distribution_name, name),
            entry_point_name=name,
            entry_point_value=_entry_point_value(entry_point),
            distribution_name=distribution_name,
            distribution_version=distribution_version,
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

            candidates = []
            for distribution in distributions:
                try:
                    entry_points = tuple(
                        entry_point
                        for entry_point in (
                            getattr(distribution, "entry_points", ()) or ()
                        )
                        if getattr(entry_point, "group", None)
                        == _BACKEND_ENTRY_POINT_GROUP
                    )
                except Exception as exc:
                    name, _ = _distribution_identity(distribution)
                    error = BackendPluginDiscoveryError(
                        "Unable to enumerate distribution entry points: "
                        f"{name or '<unknown>'}: {exc}",
                        field="distribution.entry_points",
                        expected="readable entry-point metadata",
                        actual=f"<error: {exc}>",
                        remediation=(
                            "Reinstall the affected distribution with valid "
                            "entry-point metadata."
                        ),
                    )
                    self._record_registry_error(error)
                    continue
                if not entry_points:
                    continue
                name, version = _distribution_identity(distribution)
                if not name:
                    error = BackendPluginDiscoveryError(
                        "Backend distribution has no readable package name",
                        field="distribution.metadata.Name",
                        expected="a non-empty installed distribution name",
                        actual="<unavailable>",
                        remediation=(
                            "Reinstall the affected backend with valid "
                            "distribution metadata."
                        ),
                    )
                    self._record_registry_error(error)
                    source = _source_hint(distribution)
                    for entry_point in entry_points:
                        self._store_record(
                            entry_point=entry_point,
                            distribution=distribution,
                            source=source,
                            manifest=None,
                            state=PluginLifecycleState.REJECTED,
                            compatibility_status=(
                                PluginCompatibilityStatus.NOT_CHECKED
                            ),
                            error=error,
                        )
                    continue
                if any(not _entry_point_name(ep) for ep in entry_points):
                    error = BackendPluginDiscoveryError(
                        f"Backend distribution '{name}' has an unreadable "
                        "entry-point name",
                        field="entry_point.name",
                        expected="a non-empty entry-point name",
                        actual="<unavailable>",
                        remediation=(
                            "Reinstall the affected backend with valid "
                            "entry-point metadata."
                        ),
                    )
                    self._record_registry_error(error)
                    source = _source_hint(distribution)
                    for entry_point in entry_points:
                        self._store_record(
                            entry_point=entry_point,
                            distribution=distribution,
                            source=source,
                            manifest=None,
                            state=PluginLifecycleState.REJECTED,
                            compatibility_status=(
                                PluginCompatibilityStatus.NOT_CHECKED
                            ),
                            error=error,
                        )
                    continue
                candidates.append(
                    (
                        (
                            canonicalize_name(name or "unknown-distribution"),
                            str(version or ""),
                            tuple(
                                sorted(
                                    (
                                        _entry_point_name(ep),
                                        _entry_point_value(ep),
                                    )
                                    for ep in entry_points
                                )
                            ),
                        ),
                        distribution,
                        entry_points,
                    )
                )

            for _, distribution, entry_points in sorted(
                candidates, key=lambda item: item[0]
            ):
                ordered_entry_points = tuple(
                    sorted(
                        entry_points,
                        key=lambda ep: (
                            _entry_point_name(ep),
                            _entry_point_value(ep),
                        )
                    )
                )
                try:
                    document = load_distribution_manifest(distribution)
                except BackendPluginError as exc:
                    source = _source_hint(distribution)
                    for entry_point in ordered_entry_points:
                        error = _copy_manifest_error(
                            exc,
                            entry_point=_entry_point_name(entry_point),
                        )
                        self._store_record(
                            entry_point=entry_point,
                            distribution=distribution,
                            source=source,
                            manifest=None,
                            state=PluginLifecycleState.REJECTED,
                            compatibility_status=(
                                PluginCompatibilityStatus.NOT_CHECKED
                            ),
                            error=error,
                        )
                    continue
                except (AttributeError, OSError, TypeError, ValueError) as exc:
                    source = _source_hint(distribution)
                    unexpected = BackendPluginManifestError(
                        "Unable to read distribution Manifest metadata",
                        field="distribution_manifest",
                        expected="readable, valid static Manifest metadata",
                        actual=_stable_exception_actual(exc),
                        remediation=(
                            "Repair or reinstall the affected backend "
                            "distribution; discovery did not import plugin code."
                        ),
                    )
                    for entry_point in ordered_entry_points:
                        self._store_record(
                            entry_point=entry_point,
                            distribution=distribution,
                            source=source,
                            manifest=None,
                            state=PluginLifecycleState.REJECTED,
                            compatibility_status=(
                                PluginCompatibilityStatus.NOT_CHECKED
                            ),
                            error=_copy_manifest_error(
                                unexpected,
                                entry_point=_entry_point_name(entry_point),
                            ),
                        )
                    continue

                if document is None:
                    for entry_point in ordered_entry_points:
                        self._store_record(
                            entry_point=entry_point,
                            distribution=distribution,
                            source=PluginSource.LEGACY,
                            manifest=None,
                            state=PluginLifecycleState.DISCOVERED,
                            compatibility_status=(
                                PluginCompatibilityStatus.LEGACY_UNVERIFIED
                            ),
                        )
                    continue

                manifests = {
                    plugin.entry_point: plugin for plugin in document.plugins
                }
                for entry_point in ordered_entry_points:
                    manifest = manifests[_entry_point_name(entry_point)]
                    self._store_record(
                        entry_point=entry_point,
                        distribution=distribution,
                        source=PluginSource.MANIFEST,
                        manifest=manifest,
                        state=PluginLifecycleState.DISCOVERED,
                        compatibility_status=PluginCompatibilityStatus.NOT_CHECKED,
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

    def _resolve(
        self,
        identifier: str,
        *,
        reject_conflicts: bool = False,
    ) -> BackendPluginRecord:
        if identifier in self._records:
            return self._records[identifier]
        matches = tuple(
            record
            for record in self._records.values()
            if (
                record.registry_key == identifier
                or (
                    isinstance(record.error, BackendPluginManifestError)
                    and record.error.plugin_id == identifier
                )
            )
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
        if reject_conflicts:
            unsupported_matches = []
            for record in matches:
                error = operational_record_manifest_error(record)
                if error is not None:
                    rejected = self._reject_non_operational_manifest(
                        record,
                        error,
                    )
                    unsupported_matches.append(
                        (rejected.record_id, rejected.error or error)
                    )
            if unsupported_matches:
                raise sorted(
                    unsupported_matches,
                    key=lambda item: item[0],
                )[0][1]
        if len(matches) > 1:
            if reject_conflicts:
                match_ids = {record.record_id for record in matches}
                operational_records = tuple(
                    record
                    for record in self._records.values()
                    if operational_record_manifest_error(record) is None
                )
                related_conflicts = tuple(
                    conflict
                    for conflict in detect_conflicts(
                        operational_records
                    ).fatal_conflicts
                    if match_ids.intersection(conflict.record_ids)
                )
                for conflict in related_conflicts:
                    self._reject_conflicted(conflict)
                if related_conflicts:
                    raise related_conflicts[0].to_error()
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
        return self._reject_many(record, (error,), compatibility_status)

    def _reject_many(
        self,
        record: BackendPluginRecord,
        errors: Iterable[BackendPluginError],
        compatibility_status: PluginCompatibilityStatus,
        *,
        compatibility_report: Any = _UNSET,
        capability_report: Any = _UNSET,
    ) -> BackendPluginRecord:
        """Atomically reject a record with all new deterministic failures."""
        combined = list(record.errors)
        identities = {_error_identity(error) for error in combined}
        for error in errors:
            identity = _error_identity(error)
            if identity not in identities:
                combined.append(error)
                identities.add(identity)
        replacements = {
            "state": PluginLifecycleState.REJECTED,
            "compatibility_status": compatibility_status,
            "errors": tuple(combined),
        }
        if compatibility_report is not _UNSET:
            replacements["compatibility_report"] = compatibility_report
        if capability_report is not _UNSET:
            replacements["capability_report"] = capability_report
        return self._replace(replace(record, **replacements))

    def _reject_non_operational_manifest(
        self,
        record: BackendPluginRecord,
        error: BackendPluginManifestError,
    ) -> BackendPluginRecord:
        """Revoke every forged runtime fact without calling plugin code.

        Unsupported isolation is structural and must remain the primary error
        even if a caller manually forged an old rejection, compatibility
        report, lifecycle state, runtime object, or cached selection.
        """
        selected = any(
            decision.record_id == record.record_id
            for decision in self._selections.values()
        )
        published = selected or record.state in {
            PluginLifecycleState.SELECTED,
            PluginLifecycleState.ACTIVE,
        } or bool(record.selected_targets)
        rejected = replace(
            record,
            state=PluginLifecycleState.REJECTED,
            compatibility_status=PluginCompatibilityStatus.NOT_CHECKED,
            compatibility_report=None,
            capability_report=None,
            errors=(error,),
            plugin_object=None,
            compiler_cls=None,
            driver_cls=None,
            initialized=False,
            shutdown_called=False,
            selected_targets=(),
        )
        self._records[record.record_id] = rejected
        self._cleanup_stack = [
            candidate
            for candidate in self._cleanup_stack
            if candidate.record_id != record.record_id
        ]
        for target, decision in tuple(self._selections.items()):
            if decision.record_id == record.record_id:
                del self._selections[target]
        if published:
            self._generation += 1
        return rejected

    def _guard_operational_record(
        self, record: BackendPluginRecord
    ) -> BackendPluginRecord:
        """Apply the Protocol 1.0 shape gate before any lifecycle shortcut."""
        error = operational_record_manifest_error(record)
        if error is None:
            return record
        return self._reject_non_operational_manifest(record, error)

    def _reject_conflicted(
        self,
        conflict,
    ) -> None:
        """Mark every record involved in one fatal conflict REJECTED."""
        for conflicted_id in conflict.record_ids:
            conflicted = self._records.get(conflicted_id)
            if conflicted is None:
                continue
            related_record_ids = tuple(
                record_id
                for record_id in conflict.record_ids
                if record_id != conflicted.record_id
            )
            related_plugin_ids = tuple(
                plugin_id
                for plugin_id in (
                    self._records[record_id].plugin_id
                    for record_id in related_record_ids
                    if record_id in self._records
                )
                if plugin_id is not None
            )
            error = conflict.to_error(
                plugin_id=conflicted.plugin_id,
                entry_point=conflicted.entry_point_name,
                related_plugin_ids=related_plugin_ids,
                related_record_ids=related_record_ids,
            )
            self._reject_many(
                conflicted,
                (error,),
                conflicted.compatibility_status,
            )

    def _reject_all_fatal_conflicts(self, records) -> None:
        """Mark every record in any fatal static conflict REJECTED."""
        for conflict in detect_conflicts(records).fatal_conflicts:
            self._reject_conflicted(conflict)

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
        # This precedes every state/source/profile shortcut.  A manually
        # forged VALIDATED/ACTIVE/REJECTED record is still not authorization.
        record = self._guard_operational_record(record)
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
                specialize_error=True,
            )
            return self._records[record.record_id]

        errors = []
        report = None
        if self._preflight_profile == "triton_version":
            native_report = NativeInspectionReport()
            try:
                native_report = inspect_native_artifacts(
                    record.manifest, record.distribution
                )
            except BackendPluginError as error:
                errors.append(error)
            try:
                report = validate_triton_version_requirement(
                    record.manifest,
                    environment,
                )
            except BackendPluginError as error:
                errors.append(error)
            else:
                report = replace(
                    report,
                    native_artifacts=native_report.artifacts,
                )
        else:
            evaluation = _evaluate_backend_plugin_compatibility(
                record.manifest,
                environment,
                distribution=record.distribution,
                core_abi_fingerprint=(
                    self._core_abi_fingerprint
                    if self._core_abi_fingerprint is not None
                    else environment.core_abi_fingerprint
                ),
                supported_tags=self._supported_tags,
            )
            report = evaluation.report
            errors.extend(evaluation.errors)

        capability_report = evaluate_plugin_capabilities(
            record.manifest,
            core_provided=self._core_capabilities,
        )
        if not capability_report.compatible:
            errors.append(_capability_error(capability_report))

        ordered_errors = _sort_backend_plugin_errors(errors)
        if ordered_errors:
            if report is not None and report.compatible:
                report = replace(report, compatible=False)
            compatibility_status = (
                PluginCompatibilityStatus.NOT_CHECKED
                if any(
                    isinstance(error, BackendPluginManifestError)
                    for error in ordered_errors
                )
                else PluginCompatibilityStatus.INCOMPATIBLE
            )
            return self._reject_many(
                record,
                ordered_errors,
                compatibility_status,
                compatibility_report=report,
                capability_report=capability_report,
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
                compatibility_report=report,
                capability_report=capability_report,
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
                record = self._validate_record(
                    self._resolve(identifier, reject_conflicts=True)
                )
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
        record = self._guard_operational_record(record)
        if record.state is PluginLifecycleState.REJECTED:
            return record
        if record.source is not PluginSource.MANIFEST or record.state in {
            PluginLifecycleState.LOADED,
            PluginLifecycleState.REGISTERED,
            PluginLifecycleState.SELECTED,
            PluginLifecycleState.ACTIVE,
        }:
            return record

        if record.state is not PluginLifecycleState.REJECTED:
            record = self._validate_record(record)

        operational_records = tuple(
            candidate
            for candidate in self._records.values()
            if operational_record_manifest_error(candidate) is None
        )
        for conflict in detect_conflicts(operational_records).fatal_conflicts:
            if record.record_id in conflict.record_ids:
                self._reject_conflicted(conflict)
        return self._records[record.record_id]

    def _transition(
        self,
        record: BackendPluginRecord,
        target: PluginLifecycleState,
    ) -> None:
        source = record.source or PluginSource.MANIFEST
        if not can_transition(record.state, target, source):
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

    @staticmethod
    def _operation_owned_by(
        operations: Mapping[str, Tuple[int, int, object]],
        thread_id: int,
    ) -> bool:
        return any(
            owner == thread_id
            for _epoch, owner, _token in operations.values()
        )

    def _active_wait_edge(
        self,
        waiter_thread: int,
    ) -> Optional[_LifecycleWait]:
        """Return one wait edge only while its target token is still live."""
        edge = self._waiting_on.get(waiter_thread)
        if edge is None:
            return None
        operations = (
            self._loading
            if edge.operation == "load"
            else self._registering
        )
        target = operations.get(edge.record_id)
        if (
            target is not None
            and target[1] == edge.owner_thread
            and target[2] is edge.operation_token
        ):
            return edge
        if self._waiting_on.get(waiter_thread) is edge:
            del self._waiting_on[waiter_thread]
        return None

    @staticmethod
    def _wait_cycle_actual(edges: Iterable[_LifecycleWait]) -> str:
        items = sorted(
            {(edge.record_id, edge.operation) for edge in edges}
        )
        return ", ".join(
            f"{operation}({record_id})" for record_id, operation in items
        )

    def _wait_for_lifecycle_operation(
        self,
        *,
        record: BackendPluginRecord,
        operation: str,
        operation_token: object,
        owner_thread: int,
        waiter_thread: int,
    ) -> None:
        """Wait once, rejecting only a real load/register wait cycle.

        The Registry Condition must be held by the caller. The edge's
        ``finally`` is deliberately BaseException-safe so reset and operation
        owners cannot be stranded by an interrupted waiter.
        """
        wait_token = object()
        edge = _LifecycleWait(
            owner_thread=owner_thread,
            operation=operation,
            record_id=record.record_id,
            operation_token=operation_token,
            wait_token=wait_token,
        )
        cycle_edges = [edge]
        cursor = owner_thread
        seen = set()
        while True:
            if cursor == waiter_thread or cursor in seen:
                raise _BackendPluginWaitCycleError(
                    record,
                    self._wait_cycle_actual(cycle_edges),
                )
            seen.add(cursor)
            owner_edge = self._active_wait_edge(cursor)
            if owner_edge is None:
                break
            cycle_edges.append(owner_edge)
            cursor = owner_edge.owner_thread

        self._waiting_on[waiter_thread] = edge
        try:
            self._condition.wait()
        finally:
            current = self._waiting_on.get(waiter_thread)
            if current is not None and current.wait_token is wait_token:
                del self._waiting_on[waiter_thread]

    def _finish_operation(
        self,
        operations: Dict[str, Tuple[int, int, object]],
        record_id: str,
        token: object,
    ) -> None:
        operation = operations.get(record_id)
        if operation is not None and operation[2] is token:
            del operations[record_id]
        for waiter_thread, edge in tuple(self._waiting_on.items()):
            if edge.operation_token is token:
                del self._waiting_on[waiter_thread]
        self._condition.notify_all()

    @staticmethod
    def _stale_operation_error(
        record: BackendPluginRecord,
        operation: str,
        expected_epoch: int,
        actual_epoch: int,
    ) -> BackendPluginLifecycleError:
        return BackendPluginLifecycleError(
            f"Backend plugin {operation} result belongs to a stale Registry generation",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="generation",
            expected=str(expected_epoch),
            actual=str(actual_epoch),
            remediation=(
                f"Retry {operation} after Registry reset has completed; stale "
                "plugin instances are never published."
            ),
        )

    @staticmethod
    def _hook_error(
        record: BackendPluginRecord,
        hook_name: str,
        message: str,
        *,
        expected: str,
        actual: str,
        remediation: str,
    ) -> BackendPluginLifecycleError:
        return BackendPluginLifecycleError(
            message,
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field=hook_name,
            expected=expected,
            actual=actual,
            remediation=remediation,
        )

    def _read_hook(
        self,
        record: BackendPluginRecord,
        hook_name: str,
    ) -> Tuple[Any, Optional[BackendPluginLifecycleError]]:
        try:
            statically_present = (
                inspect.getattr_static(
                    record.plugin_object, hook_name, _MISSING
                )
                is not _MISSING
            )
        except Exception as exc:
            return _MISSING, self._hook_error(
                record,
                hook_name,
                f"Unable to inspect backend {hook_name} hook",
                expected="a callable hook or no hook",
                actual=_stable_exception_actual(exc),
                remediation=(
                    f"Fix the {hook_name} attribute so it is safely readable."
                ),
            )
        try:
            hook = (
                getattr(record.plugin_object, hook_name)
                if statically_present
                else getattr(record.plugin_object, hook_name, _MISSING)
            )
        except Exception as exc:
            return _MISSING, self._hook_error(
                record,
                hook_name,
                f"Unable to inspect backend {hook_name} hook",
                expected="a callable hook or no hook",
                actual=_stable_exception_actual(exc),
                remediation=(
                    f"Fix the {hook_name} attribute so it is safely readable."
                ),
            )
        if hook is _MISSING:
            return _MISSING, None
        if not callable(hook):
            return hook, self._hook_error(
                record,
                hook_name,
                f"Backend plugin {hook_name} hook is not callable",
                expected="a callable hook or no hook",
                actual=_stable_type_name(hook),
                remediation=(
                    f"Expose {hook_name} as a callable or remove the optional hook."
                ),
            )
        return hook, None

    def _bind_hook(
        self,
        record: BackendPluginRecord,
        hook_name: str,
        hook: Callable[..., Any],
        args: Tuple[Any, ...],
    ) -> Optional[BackendPluginLifecycleError]:
        expected = (
            "initialize(context)"
            if hook_name == "initialize"
            else f"{hook_name}()"
        )
        try:
            signature = inspect.signature(hook)
        except (TypeError, ValueError):
            # Some extension/builtin callables expose no inspectable
            # signature. The real invocation remains authoritative.
            return None
        except Exception as exc:
            return self._hook_error(
                record,
                hook_name,
                f"Unable to inspect backend {hook_name} signature",
                expected=expected,
                actual=_stable_exception_actual(exc),
                remediation=(
                    f"Expose an introspectable valid {hook_name} signature."
                ),
            )
        try:
            signature.bind(*args)
        except TypeError:
            return self._hook_error(
                record,
                hook_name,
                f"Backend plugin {hook_name} hook has an incompatible signature",
                expected=expected,
                actual=f"callable does not accept {expected}",
                remediation=(
                    f"Fix the {hook_name} signature to match Protocol 1.0."
                ),
            )
        return None

    def _call_shutdown(
        self,
        record: BackendPluginRecord,
    ) -> Tuple[bool, Optional[BackendPluginLifecycleError]]:
        """Run one best-effort synchronous shutdown outside the Registry lock."""
        hook, error = self._read_hook(record, "shutdown")
        if error is not None:
            return False, error
        if hook is _MISSING:
            return False, None
        error = self._bind_hook(record, "shutdown", hook, ())
        if error is not None:
            return False, error
        try:
            value = hook()
        except Exception as exc:
            return True, self._hook_error(
                record,
                "shutdown",
                "Backend plugin shutdown() failed",
                expected="successful cleanup returning None",
                actual=_stable_exception_actual(exc),
                remediation=(
                    "Fix shutdown() so Registry reset can release resources."
                ),
            )
        if inspect.isawaitable(value):
            actual = _stable_type_name(value)
            _close_unawaited(value)
            return True, self._hook_error(
                record,
                "shutdown",
                "Backend plugin shutdown() returned an awaitable",
                expected="None from a synchronous shutdown() hook",
                actual=actual,
                remediation=(
                    "Implement shutdown() as a synchronous hook returning None."
                ),
            )
        if value is not None:
            return True, self._hook_error(
                record,
                "shutdown",
                "Backend plugin shutdown() returned a non-None value",
                expected="None",
                actual=_stable_type_name(value),
                remediation="Return None from shutdown().",
            )
        return True, None

    def load(self, identifier: str) -> BackendPluginRecord:
        """Import one plugin only after its applicable pre-load gate passes."""
        thread_id = threading.get_ident()
        while True:
            with self._condition:
                self._ensure_not_resetting("load")
                self.discover()
                record = self._resolve(identifier, reject_conflicts=True)
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

                operation = self._loading.get(record.record_id)
                if operation is not None:
                    wait_epoch, owner, existing_token = operation
                    if owner == thread_id:
                        raise BackendPluginLifecycleError(
                            "Recursive backend plugin load is not allowed",
                            plugin_id=record.plugin_id,
                            entry_point=record.entry_point_name,
                            field="load",
                            expected="one non-reentrant load operation",
                            actual="recursive load",
                            remediation=(
                                "Do not call Registry load/register for the same "
                                "plugin from its entry-point loader or constructor."
                            ),
                        )
                    self._wait_for_lifecycle_operation(
                        record=record,
                        operation="load",
                        operation_token=existing_token,
                        owner_thread=owner,
                        waiter_thread=thread_id,
                    )
                    if self._lifecycle_epoch != wait_epoch:
                        raise self._stale_operation_error(
                            record, "load", wait_epoch, self._lifecycle_epoch
                        )
                    continue

                self._transition(record, PluginLifecycleState.LOADED)
                epoch = self._lifecycle_epoch
                token = object()
                self._loading[record.record_id] = (
                    epoch,
                    thread_id,
                    token,
                )

            operation_finished = False
            try:
                # Entry-point loading and plugin construction are
                # plugin-controlled and must run outside the Registry lock.
                load_exception: Optional[Exception] = None
                try:
                    loaded = record.entry_point.load()
                    plugin = loaded() if isinstance(loaded, type) else loaded
                except _BackendPluginWaitCycleError as exc:
                    load_exception = exc.for_record(record)
                    plugin = None
                except Exception as exc:
                    load_exception = exc
                    plugin = None

                with self._condition:
                    stale = (
                        epoch != self._lifecycle_epoch
                        or record.record_id not in self._records
                    )
                    if stale:
                        error: Optional[BackendPluginError] = (
                            self._stale_operation_error(
                                record,
                                "load",
                                epoch,
                                self._lifecycle_epoch,
                            )
                        )
                    elif load_exception is not None:
                        error = (
                            load_exception
                            if isinstance(
                                load_exception,
                                _BackendPluginWaitCycleError,
                            )
                            else BackendPluginLoadError(
                                f"Failed to load backend plugin "
                                f"'{record.registry_key}'",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="entry_point.load",
                                expected="a loadable backend plugin object",
                                actual=_stable_exception_actual(
                                    load_exception
                                ),
                                remediation=(
                                    "Fix the backend Python package or native "
                                    "dependencies and reinstall it."
                                ),
                            )
                        )
                        current = self._records[record.record_id]
                        self._reject(
                            current,
                            error,
                            current.compatibility_status,
                        )
                    else:
                        current = self._records[record.record_id]
                        if current.state is PluginLifecycleState.REJECTED:
                            error = current.error or BackendPluginLifecycleError(
                                "Backend plugin state changed while loading",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="state",
                                expected=record.state.value,
                                actual=current.state.value,
                                remediation=(
                                    "Retry after resolving the recorded failure."
                                ),
                            )
                        else:
                            result = self._replace(
                                replace(
                                    current,
                                    state=PluginLifecycleState.LOADED,
                                    plugin_object=plugin,
                                )
                            )
                            error = None
                    self._finish_operation(
                        self._loading, record.record_id, token
                    )
                    operation_finished = True

                if error is not None:
                    if (
                        load_exception is not None
                        and error is not load_exception
                    ):
                        raise error from load_exception
                    raise error
                return result
            finally:
                # Never strand reset/concurrent waiters if plugin code raises
                # BaseException or error normalization itself fails.
                if not operation_finished:
                    with self._condition:
                        self._finish_operation(
                            self._loading, record.record_id, token
                        )

    def register(
        self,
        identifier: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> BackendPluginRecord:
        """Check the runtime pair, initialize once, and register one plugin."""
        loaded_record = self.load(identifier)
        record_id = loaded_record.record_id
        thread_id = threading.get_ident()

        while True:
            with self._condition:
                self._ensure_not_resetting("register")
                record = self._records.get(record_id)
                if record is None:
                    raise self._stale_operation_error(
                        loaded_record,
                        "register",
                        max(0, self._lifecycle_epoch - 1),
                        self._lifecycle_epoch,
                    )
                if record.state in {
                    PluginLifecycleState.REGISTERED,
                    PluginLifecycleState.SELECTED,
                    PluginLifecycleState.ACTIVE,
                }:
                    return record
                if record.state is PluginLifecycleState.REJECTED:
                    if record.error is not None:
                        raise record.error
                    raise BackendPluginLifecycleError(
                        "Rejected backend plugin cannot be registered",
                        plugin_id=record.plugin_id,
                        entry_point=record.entry_point_name,
                        field="state",
                        expected=PluginLifecycleState.LOADED.value,
                        actual=record.state.value,
                        remediation=(
                            "Reset and rediscover after resolving the recorded "
                            "lifecycle failure."
                        ),
                    )

                operation = self._registering.get(record_id)
                if operation is not None:
                    wait_epoch, owner, existing_token = operation
                    if owner == thread_id:
                        raise BackendPluginLifecycleError(
                            "Recursive backend plugin registration is not allowed",
                            plugin_id=record.plugin_id,
                            entry_point=record.entry_point_name,
                            field="register",
                            expected="one non-reentrant registration operation",
                            actual="recursive registration",
                            remediation=(
                                "Do not call Registry register for the same plugin "
                                "from runtime attributes or initialize()."
                            ),
                        )
                    self._wait_for_lifecycle_operation(
                        record=record,
                        operation="register",
                        operation_token=existing_token,
                        owner_thread=owner,
                        waiter_thread=thread_id,
                    )
                    if self._lifecycle_epoch != wait_epoch:
                        raise self._stale_operation_error(
                            record,
                            "register",
                            wait_epoch,
                            self._lifecycle_epoch,
                        )
                    continue

                if context is not None:
                    error = BackendPluginLifecycleError(
                        "Custom initialize context is not supported by Protocol 1.0",
                        plugin_id=record.plugin_id,
                        entry_point=record.entry_point_name,
                        field="initialize.context",
                        expected=(
                            "Registry-owned read-only Mapping with keys "
                            "environment, manifest, record_id"
                        ),
                        actual="caller-supplied context",
                        remediation=(
                            "Consume the fixed Registry context in initialize()."
                        ),
                    )
                    self._reject(record, error, record.compatibility_status)
                    raise error

                self._transition(record, PluginLifecycleState.REGISTERED)
                epoch = self._lifecycle_epoch
                initialization_context = MappingProxyType(
                    {
                        "environment": self._get_environment(),
                        "manifest": _manifest_snapshot(record.manifest),
                        "record_id": record.record_id,
                    }
                )
                token = object()
                self._registering[record_id] = (
                    epoch,
                    thread_id,
                    token,
                )
                runtime_pair_validators = tuple(
                    sorted(self._runtime_pair_validators.items())
                )

            operation_finished = False
            try:
                # Runtime descriptors, signature inspection, and lifecycle
                # hooks are plugin-controlled; none run under Registry locks.
                values: Dict[str, Any] = {}
                field_errors: Dict[str, str] = {}
                for name in ("compiler_cls", "driver_cls"):
                    try:
                        values[name] = getattr(
                            record.plugin_object, name, _MISSING
                        )
                    except Exception as exc:
                        values[name] = _MISSING
                        field_errors[name] = _stable_exception_actual(exc)
                missing = tuple(
                    name
                    for name, value in values.items()
                    if value is _MISSING or value is None
                )
                invalid = tuple(
                    name
                    for name, value in values.items()
                    if value is not _MISSING
                    and value is not None
                    and not isinstance(value, type)
                )

                primary_error: Optional[BackendPluginError] = None
                cleanup_error: Optional[BackendPluginLifecycleError] = None
                cleanup_called = False
                initialized = False
                initializer_invoked = False
                if missing or invalid or field_errors:
                    primary_error = BackendPluginInterfaceError(
                        missing,
                        invalid_fields=invalid,
                        field_errors=field_errors,
                        plugin_id=record.plugin_id,
                        entry_point=record.entry_point_name,
                    )
                elif runtime_pair_validators:
                    validation_context = RuntimePairValidationContext(
                        compiler_cls=values["compiler_cls"],
                        driver_cls=values["driver_cls"],
                    )
                    (
                        contract_ids,
                        required_surface,
                        interface_issues,
                    ) = _run_runtime_pair_validators(
                        runtime_pair_validators,
                        validation_context,
                    )
                    if interface_issues:
                        primary_error = BackendPluginInterfaceError(
                            contract_ids=contract_ids,
                            required_surface=required_surface,
                            interface_issues=interface_issues,
                            plugin_id=record.plugin_id,
                            entry_point=record.entry_point_name,
                        )

                # A reset that wins during static validation must prevent the
                # old attempt from entering initialize().  Check the operation
                # token and epoch at the new trust boundary, then release a
                # stale attempt before any plugin lifecycle hook is read.
                validation_stale = False
                if primary_error is None:
                    with self._condition:
                        current_operation = self._registering.get(record_id)
                        validation_stale = (
                            epoch != self._lifecycle_epoch
                            or record_id not in self._records
                            or current_operation is None
                            or current_operation[2] is not token
                        )
                        if validation_stale:
                            stale_error = self._stale_operation_error(
                                record,
                                "register",
                                epoch,
                                self._lifecycle_epoch,
                            )
                            self._finish_operation(
                                self._registering, record_id, token
                            )
                            operation_finished = True
                    if validation_stale:
                        raise stale_error

                if (
                    primary_error is None
                    and record.source is PluginSource.MANIFEST
                ):
                    initializer, primary_error = self._read_hook(
                        record, "initialize"
                    )
                    if primary_error is None and initializer is not _MISSING:
                        primary_error = self._bind_hook(
                            record,
                            "initialize",
                            initializer,
                            (initialization_context,),
                        )
                    if primary_error is None and initializer is not _MISSING:
                        initializer_invoked = True
                        try:
                            initialize_result = initializer(
                                initialization_context
                            )
                        except _BackendPluginWaitCycleError as exc:
                            primary_error = exc.for_record(record)
                        except Exception as exc:
                            primary_error = self._hook_error(
                                record,
                                "initialize",
                                "Backend plugin initialize() failed",
                                expected=(
                                    "successful initialization returning None"
                                ),
                                actual=_stable_exception_actual(exc),
                                remediation=(
                                    "Fix initialize(), release partial resources, "
                                    "and reinstall the backend."
                                ),
                            )
                        else:
                            if inspect.isawaitable(initialize_result):
                                actual = _stable_type_name(initialize_result)
                                _close_unawaited(initialize_result)
                                primary_error = self._hook_error(
                                    record,
                                    "initialize",
                                    "Backend plugin initialize() returned an awaitable",
                                    expected=(
                                        "None from a synchronous "
                                        "initialize(context) hook"
                                    ),
                                    actual=actual,
                                    remediation=(
                                        "Implement initialize(context) "
                                        "synchronously and return None."
                                    ),
                                )
                            elif initialize_result is not None:
                                primary_error = self._hook_error(
                                    record,
                                    "initialize",
                                    "Backend plugin initialize() returned a "
                                    "non-None value",
                                    expected="None",
                                    actual=_stable_type_name(initialize_result),
                                    remediation=(
                                        "Return None from initialize(context)."
                                    ),
                                )
                            else:
                                initialized = True
                    if primary_error is not None and initializer_invoked:
                        cleanup_called, cleanup_error = self._call_shutdown(
                            record
                        )

                with self._condition:
                    stale = (
                        epoch != self._lifecycle_epoch
                        or record_id not in self._records
                    )
                    if not stale:
                        # Validity and publication are one atomic Registry
                        # step. Reset either wins before this lock acquisition,
                        # or observes the published cleanup-stack entry.
                        current = self._records[record_id]
                        if primary_error is not None:
                            errors: Tuple[BackendPluginError, ...] = (
                                (primary_error, cleanup_error)
                                if cleanup_error is not None
                                else (primary_error,)
                            )
                            current = self._replace(
                                replace(
                                    current,
                                    shutdown_called=cleanup_called,
                                )
                            )
                            self._reject_many(
                                current,
                                errors,
                                current.compatibility_status,
                            )
                            error = primary_error
                        else:
                            result = self._replace(
                                replace(
                                    current,
                                    state=PluginLifecycleState.REGISTERED,
                                    compiler_cls=values["compiler_cls"],
                                    driver_cls=values["driver_cls"],
                                    initialized=initialized,
                                )
                            )
                            if result.source is PluginSource.MANIFEST:
                                self._cleanup_stack.append(result)
                            error = None
                        self._finish_operation(
                            self._registering, record_id, token
                        )
                        operation_finished = True

                if not stale:
                    if error is not None:
                        raise error
                    return result

                # Reset invalidated an unpublished successful initializer. The
                # owner performs its one cleanup attempt before waking reset.
                if initializer_invoked and primary_error is None:
                    cleanup_called, cleanup_error = self._call_shutdown(record)

                with self._condition:
                    # An actual initialize contract failure remains the
                    # primary throwing result even if reset won the race.
                    error = primary_error or self._stale_operation_error(
                        record,
                        "register",
                        epoch,
                        self._lifecycle_epoch,
                    )
                    if cleanup_error is not None and self._resetting:
                        self._reset_operation_errors.append(cleanup_error)
                    self._finish_operation(
                        self._registering, record_id, token
                    )
                    operation_finished = True

                raise error
            finally:
                # A plugin BaseException or an error-normalization bug must not
                # strand concurrent callers or make reset wait forever.
                if not operation_finished:
                    with self._condition:
                        self._finish_operation(
                            self._registering, record_id, token
                        )

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
        with self._condition:
            self._ensure_not_resetting("select")
            records = self.validate()
            operational_records = [
                record
                for record in records
                if operational_record_manifest_error(record) is None
            ]

            selection_environment = (
                os.environ if environment is None else environment
            )
            # Fatal conflicts must reject the involved records before the
            # static selection algorithm raises (selection itself is not
            # changed); this also covers compiler/runtime entry paths that
            # resolve through select().
            self._reject_all_fatal_conflicts(
                tuple(
                    record
                    for record in operational_records
                )
            )
            decision = select_backend(
                records,
                target=target,
                kernel_required_capabilities=(
                    kernel_required_capabilities
                ),
                core_provided_capabilities=self._core_capabilities,
                explicit_selector=explicit_selector,
                environment=selection_environment,
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

            selection_epoch = self._lifecycle_epoch

        # register() may run entry-point, descriptor, and initialize code.
        # None of that plugin-controlled work may inherit the selection lock.
        selected = self.register(decision.record_id)

        with self._condition:
            if selection_epoch != self._lifecycle_epoch:
                raise self._stale_operation_error(
                    decision.record,
                    "select",
                    selection_epoch,
                    self._lifecycle_epoch,
                )
            self._ensure_not_resetting("select")
            selected = self._records.get(selected.record_id)
            if selected is None:
                raise self._stale_operation_error(
                    decision.record,
                    "select",
                    selection_epoch,
                    self._lifecycle_epoch,
                )
            # Another selector may have committed while register() ran, so
            # repeat the ACTIVE-switch gate against current state.
            previous_decision = self._selections.get(target_name)
            if (
                previous_decision is not None
                and previous_decision.record_id != selected.record_id
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
                        actual=selected.record_id,
                        remediation=(
                            "Reset the Registry-backed runtime driver before "
                            "selecting a different plugin for this target."
                        ),
                    )

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
            record = self._resolve(identifier, reject_conflicts=True)
            record = self._guard_operational_record(record)
            if record.state is PluginLifecycleState.REJECTED:
                if record.error is not None:
                    raise record.error
                raise BackendPluginLifecycleError(
                    "Rejected backend plugin cannot be activated",
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                )
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

    def _diagnostics_value(self, record: BackendPluginRecord) -> Dict[str, Any]:
        """Call and snapshot one optional diagnostics hook outside the lock."""
        hook, error = self._read_hook(record, "diagnostics")
        if error is not None:
            return {"error": error.to_dict()}
        if hook is _MISSING:
            return {}
        error = self._bind_hook(record, "diagnostics", hook, ())
        if error is not None:
            return {"error": error.to_dict()}
        try:
            value = hook()
        except Exception as exc:
            error = self._hook_error(
                record,
                "diagnostics",
                "Backend plugin diagnostics() failed",
                expected="a Mapping result from a synchronous hook",
                actual=_stable_exception_actual(exc),
                remediation=(
                    "Fix diagnostics() so it is synchronous and read-only."
                ),
            )
            return {"error": error.to_dict()}
        if inspect.isawaitable(value):
            actual = _stable_type_name(value)
            _close_unawaited(value)
            error = self._hook_error(
                record,
                "diagnostics",
                "Backend plugin diagnostics() returned an awaitable",
                expected="Mapping from a synchronous diagnostics() hook",
                actual=actual,
                remediation="Implement diagnostics() synchronously.",
            )
            return {"error": error.to_dict()}
        if not isinstance(value, Mapping):
            error = self._hook_error(
                record,
                "diagnostics",
                "Backend plugin diagnostics() returned a non-Mapping value",
                expected="Mapping",
                actual=_stable_type_name(value),
                remediation="Return a Mapping from diagnostics().",
            )
            return {"error": error.to_dict()}
        try:
            return dict(value)
        except Exception as exc:
            error = self._hook_error(
                record,
                "diagnostics",
                "Unable to snapshot backend diagnostics Mapping",
                expected="a Mapping that can be copied with dict()",
                actual=_stable_exception_actual(exc),
                remediation="Return a stable Mapping from diagnostics().",
            )
            return {"error": error.to_dict()}

    def diagnostics(self, identifier: Optional[str] = None) -> Any:
        """Return a static snapshot plus optional loaded-plugin diagnostics."""
        thread_id = threading.get_ident()
        prepared: list[Tuple[BackendPluginRecord, Dict[str, Any]]] = []
        operations: list[Tuple[BackendPluginRecord, object]] = []
        with self._condition:
            self._ensure_not_resetting("diagnostics")
            self.discover()
            records = (
                (self._resolve(identifier),)
                if identifier is not None
                else tuple(
                    sorted(
                        self._records.values(),
                        key=lambda record: (
                            record.registry_key,
                            record.record_id,
                        ),
                    )
                )
            )
            try:
                for record in records:
                    record = self._guard_operational_record(record)
                    result = record.to_dict()
                    # Unloaded records remain import-free and distinguish "not
                    # available yet" from a loaded plugin with no optional hook.
                    result["plugin_diagnostics"] = None
                    if record.plugin_object is not None:
                        operation = self._diagnosing.get(record.record_id)
                        if operation is not None:
                            same_thread = operation[1] == thread_id
                            error = BackendPluginLifecycleError(
                                (
                                    "Recursive backend plugin diagnostics is not allowed"
                                    if same_thread
                                    else "Backend plugin diagnostics is already in progress"
                                ),
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="diagnostics",
                                expected="one non-reentrant diagnostics operation",
                                actual=(
                                    "recursive diagnostics"
                                    if same_thread
                                    else "diagnostics already in progress"
                                ),
                                remediation=(
                                    "Do not call diagnostics for the same plugin "
                                    "from inside its diagnostics hook."
                                ),
                            )
                            result["plugin_diagnostics"] = {
                                "error": error.to_dict()
                            }
                        else:
                            token = object()
                            self._diagnosing[record.record_id] = (
                                self._lifecycle_epoch,
                                thread_id,
                                token,
                            )
                            operations.append((record, token))
                    prepared.append((record, result))

                # Snapshot Registry-owned data while holding the lock. Calling
                # plugin hooks below cannot mutate this report mid-construction.
                operational_conflict_records = tuple(
                    record
                    for record in self._records.values()
                    if operational_record_manifest_error(record) is None
                )
                registry_payload = {
                    "preflight_profile": self._preflight_profile,
                    "core_abi_fingerprint": (
                        self._environment.core_abi_fingerprint
                        if self._environment is not None
                        else None
                    ),
                    "registry_errors": [
                        error.to_dict() for error in self._registry_errors
                    ],
                    "conflicts": detect_conflicts(
                        operational_conflict_records
                    ).to_dict(),
                    "selections": {
                        target: decision.to_dict()
                        for target, decision in sorted(self._selections.items())
                    },
                }
            except BaseException:
                # Do not retain tokens if Registry snapshot construction itself
                # fails after one or more operations have been claimed.
                for record, token in operations:
                    self._finish_operation(
                        self._diagnosing, record.record_id, token
                    )
                raise

        by_record_id = {
            record.record_id: result for record, result in prepared
        }
        try:
            for record, token in operations:
                try:
                    by_record_id[record.record_id][
                        "plugin_diagnostics"
                    ] = self._diagnostics_value(record)
                finally:
                    with self._condition:
                        self._finish_operation(
                            self._diagnosing, record.record_id, token
                        )
        finally:
            # A batch claims every loaded record before invoking the first
            # hook. If an early hook raises BaseException, release the tokens
            # for all later records as well.
            with self._condition:
                for record, token in operations:
                    self._finish_operation(
                        self._diagnosing, record.record_id, token
                    )

        results = [result for _record, result in prepared]
        if identifier is not None:
            return results[0]
        registry_payload["plugins"] = results
        return registry_payload

    def reset(self) -> Tuple[BackendPluginError, ...]:
        """Invalidate state, then clean up initialized plugins exactly once."""
        cleanup_errors: list[Tuple[int, BackendPluginError]] = []
        reset_hooks: Tuple[Callable[[], None], ...] = ()
        shutdown_records: Tuple[BackendPluginRecord, ...] = ()
        reset_started = False
        try:
            with self._condition:
                self._ensure_not_resetting("reset")
                thread_id = threading.get_ident()
                if any(
                    self._operation_owned_by(operations, thread_id)
                    for operations in (
                        self._loading,
                        self._registering,
                        self._diagnosing,
                    )
                ):
                    raise BackendPluginLifecycleError(
                        "Cannot reset the Registry re-entrantly from a plugin hook",
                        field="reset",
                        expected="reset from outside plugin lifecycle hooks",
                        actual=(
                            "current thread owns an in-progress plugin operation"
                        ),
                        remediation=(
                            "Return from the plugin hook before resetting the "
                            "Registry."
                        ),
                    )

                self._resetting = True
                reset_started = True
                self._lifecycle_epoch += 1
                # Public adapter snapshots use selection/reset generation;
                # lifecycle attempts use the separate epoch above.
                self._generation += 1
                self._reset_operation_errors = []

                # Invalidate public state before waiting. Owners can complete
                # only against the advanced epoch and cannot publish stale
                # plugin instances after this point.
                self._records.clear()
                self._registry_errors = ()
                self._environment = None
                self._environment_error = None
                self._discovered = False
                self._selections.clear()
                self._condition.notify_all()

                while self._loading or self._registering or self._diagnosing:
                    self._condition.wait()
                # Every live wait edge targets one of the now-empty owner
                # ledgers. Clear any stale edge before reset becomes visible.
                self._waiting_on.clear()

                shutdown_records = tuple(
                    record
                    for record in reversed(self._cleanup_stack)
                    if operational_record_manifest_error(record) is None
                )
                self._cleanup_stack.clear()
                reset_hooks = tuple(self._reset_hooks)
                cleanup_errors.extend(
                    (0, error) for error in self._reset_operation_errors
                )
                self._reset_operation_errors = []

            # Adapter hooks may own independent cache locks. Run them after
            # releasing the Registry lock; _resetting rejects re-entry.
            for callback in reset_hooks:
                try:
                    callback()
                except BackendPluginError as exc:
                    cleanup_errors.append((0, exc))
                except Exception as exc:
                    cleanup_errors.append(
                        (
                            0,
                            BackendPluginLifecycleError(
                                "Backend reset hook failed",
                                field="reset_hook",
                                expected="successful cache cleanup",
                                actual=_stable_exception_actual(exc),
                                remediation=(
                                    "Fix the W9 adapter reset hook so it only "
                                    "clears local compiler/driver caches."
                                ),
                            ),
                        )
                    )

            # Successful Manifest registrations are cleaned in reverse order.
            # One failure never prevents cleanup of the remaining plugins.
            for record in shutdown_records:
                _called, error = self._call_shutdown(record)
                if error is not None:
                    cleanup_errors.append((1, error))

            return tuple(
                error
                for _rank, error in sorted(
                    cleanup_errors,
                    key=lambda item: (
                        item[1].plugin_id or "",
                        item[0],
                        item[1].entry_point or "",
                        item[1].code,
                        item[1].field or "",
                        item[1].actual or "",
                    ),
                )
            )
        finally:
            if reset_started:
                with self._condition:
                    self._resetting = False
                    self._condition.notify_all()


backend_plugin_registry = BackendPluginRegistry()


def get_backend_plugin_registry() -> BackendPluginRegistry:
    """Return the process-wide Registry without triggering discovery."""
    return backend_plugin_registry
