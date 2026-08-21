"""Stateful, import-gated registry for out-of-tree backend plugins."""

from __future__ import annotations

import importlib.metadata
import inspect
import os
import threading
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from types import MappingProxyType
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
    BackendPluginInterfaceIssue,
    BackendPluginLifecycleError,
    BackendPluginLoadError,
    BackendPluginManifestError,
    BackendPluginNoCandidateError,
    BackendPluginSelectionError,
)
from .manifest import (
    MANIFEST_FILENAME,
    BackendPluginManifest,
    load_distribution_manifest,
    validate_plugin_isolation,
)
from .native import inspect_native_artifacts
from .protocol import (
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
    can_transition,
)
from .selection import (
    SelectionDecision,
    SelectionMethod,
    _capability_names,
    _target_name,
    select_backend,
)


_BACKEND_ENTRY_POINT_GROUP = "triton.backends"
_PREFLIGHT_PROFILES = {"triton_version", "full"}
_UNSET = object()
_MISSING = object()
_TYPE_MRO_DESCRIPTOR = type.__dict__["__mro__"]
_LEGACY_LEASE_FACTORY = object()


def _is_class_object(value: Any) -> bool:
    """Recognize a class without consulting a spoofable ``__class__``."""
    try:
        mro = _TYPE_MRO_DESCRIPTOR.__get__(value, type(value))
    except (AttributeError, TypeError):
        return False
    return type(mro) is tuple and bool(mro) and mro[0] is value


def _stable_type_name(value: Any) -> str:
    """Return a deterministic runtime type name without calling ``repr``."""
    value_type = type(value)
    module = getattr(value_type, "__module__", None)
    qualname = getattr(value_type, "__qualname__", value_type.__name__)
    return f"{module}.{qualname}" if module else qualname


def _close_unawaited(value: Any) -> None:
    """Avoid coroutine warnings while preserving the synchronous contract."""
    if inspect.iscoroutine(value):
        value.close()


def _freeze_json_value(value: Any) -> Any:
    """Freeze the JSON-like extension values stored in Manifest snapshots."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_json_value(item) for item in value)
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
        tuple(
            (
                issue.field,
                issue.owner,
                issue.member,
                issue.problem,
                issue.expected_kind,
                issue.actual_kind,
                issue.expected_signature,
                issue.actual_signature,
                issue.remediation,
            )
            for issue in getattr(error, "interface_issues", ())
        ),
    )


@dataclass(frozen=True)
class RuntimePairValidationContext:
    """Immutable class pair presented to a named interface validator."""

    record_id: str
    plugin_id: Optional[str]
    entry_point: str
    compiler_cls: type
    driver_cls: type


RuntimePairValidator = Callable[
    [RuntimePairValidationContext],
    Optional[Iterable[BackendPluginInterfaceIssue]],
]


def _runtime_pair_validator_failure(
    contract_id: str,
    actual_kind: str,
    problem: str,
) -> BackendPluginInterfaceIssue:
    """Create a stable fail-closed finding without exception text or repr."""
    return BackendPluginInterfaceIssue(
        field="compiler_cls",
        owner="RuntimePairValidator",
        member=contract_id,
        expected_kind="interface_issues",
        actual_kind=actual_kind,
        problem=problem,
        remediation=(
            "Fix the named runtime-pair validator so it returns only "
            "BackendPluginInterfaceIssue values or None."
        ),
    )


def _is_well_formed_interface_issue(
    issue: BackendPluginInterfaceIssue,
) -> bool:
    required = (
        issue.field,
        issue.owner,
        issue.member,
        issue.expected_kind,
        issue.actual_kind,
        issue.problem,
        issue.remediation,
    )
    return (
        type(issue.field) is str
        and issue.field in {"compiler_cls", "driver_cls"}
        and all(type(value) is str and bool(value.strip()) for value in required)
        and (
            issue.expected_signature is None
            or (
                type(issue.expected_signature) is str
                and bool(issue.expected_signature.strip())
            )
        )
        and (
            issue.actual_signature is None
            or (
                type(issue.actual_signature) is str
                and bool(issue.actual_signature.strip())
            )
        )
    )


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
        result = {
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
        if self.source is PluginSource.LEGACY:
            result["legacy_compatibility"] = {
                "static_compatibility_proven": False,
                "missing_static_proofs": [
                    "protocol",
                    "core",
                    "triton",
                    "llvm",
                    "capabilities",
                    "native_abi",
                    "priority",
                    "targets",
                ],
                "remediation": (
                    "Migrate this backend to Backend Plugin Manifest Schema "
                    "1.0 so compatibility can be proven before import."
                ),
            }
        return result


class LegacyRecordLease:
    """Opaque authority for conditionally publishing one Legacy runtime pair.

    A lease is issued only after the exact Legacy record has loaded,
    registered, and passed every current runtime-pair validator.  The Registry
    keeps ownership of its mutable record snapshot; callers may inspect the
    snapshot but cannot transfer a lease between Registry instances or reset
    epochs.
    """

    __slots__ = (
        "_record",
        "_registry_identity",
        "_lifecycle_epoch",
        "_token",
        "_plugin_object",
        "_compiler_cls",
        "_driver_cls",
    )

    def __init__(
        self,
        record: BackendPluginRecord,
        *,
        registry_identity: object,
        lifecycle_epoch: int,
        token: object,
        factory: object,
    ) -> None:
        if factory is not _LEGACY_LEASE_FACTORY:
            raise TypeError(
                "LegacyRecordLease values are issued by BackendPluginRegistry"
            )
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_registry_identity", registry_identity)
        object.__setattr__(self, "_lifecycle_epoch", lifecycle_epoch)
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_plugin_object", record.plugin_object)
        object.__setattr__(self, "_compiler_cls", record.compiler_cls)
        object.__setattr__(self, "_driver_cls", record.driver_cls)

    @property
    def record(self) -> BackendPluginRecord:
        """Return the current immutable record snapshot for this lease."""
        return self._record

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("LegacyRecordLease is immutable")

    def _refresh(
        self,
        record: BackendPluginRecord,
        *,
        factory: object,
    ) -> None:
        """Advance the snapshot after an atomic Registry-owned replacement."""
        if factory is not _LEGACY_LEASE_FACTORY:
            raise TypeError("LegacyRecordLease snapshots are Registry-owned")
        object.__setattr__(self, "_record", record)


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
        # The token prevents an old completion from deleting a newer operation.
        self._loading: Dict[str, Tuple[int, int, object]] = {}
        self._registering: Dict[str, Tuple[int, int, object]] = {}
        self._diagnosing: Dict[str, Tuple[int, int, object]] = {}
        self._waiting_for: Dict[int, int] = {}
        self._cleanup_stack: list[BackendPluginRecord] = []
        self._reset_operation_errors: list[BackendPluginError] = []
        self._selections: Dict[str, SelectionDecision] = {}
        self._legacy_lease_identity = object()
        self._legacy_leases: Dict[str, LegacyRecordLease] = {}
        self._legacy_lease_tokens: Dict[str, object] = {}
        self._legacy_publication_owner: Optional[int] = None
        self._reset_invalidation_hooks: list = []
        self._reset_hooks: list = []
        self._runtime_pair_validators: Dict[str, RuntimePairValidator] = {}
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

    def register_reset_invalidation_hook(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Register a trusted cache invalidator for the reset commit point.

        Unlike ordinary shutdown/reset hooks, invalidators run while the
        Registry condition is held, after the lifecycle epoch advances and
        before records and selections are cleared.  They must perform only a
        bounded in-memory cache mutation and must never call Registry or
        plugin code.  This ordering prevents a stale adapter mapping from
        becoming observable after its owning Registry record is gone.
        """
        if not callable(callback):
            raise TypeError("reset invalidation hook must be callable")
        with self._condition:
            self._ensure_not_resetting("register_reset_invalidation_hook")
            if callback not in self._reset_invalidation_hooks:
                self._reset_invalidation_hooks.append(callback)

    def register_runtime_pair_validator(
        self,
        contract_id: str,
        callback: RuntimePairValidator,
    ) -> None:
        """Attach one named pre-initialize runtime-pair validator.

        Registering the identical callback again is idempotent. Replacing a
        name requires an empty Registry (normally immediately after reset).
        Attaching a new name is only safe outside load/pair-validation races
        and before a record has been published. Validators are process
        contracts and intentionally survive :meth:`reset`.
        """
        if not isinstance(contract_id, str) or not contract_id.strip():
            raise ValueError("runtime-pair validator contract_id must be non-empty")
        if not callable(callback):
            raise TypeError("runtime-pair validator must be callable")

        with self._condition:
            self._ensure_not_resetting("register_runtime_pair_validator")
            current = self._runtime_pair_validators.get(contract_id, _MISSING)
            if current is callback:
                return

            lifecycle_operations = {
                **self._loading,
                **self._registering,
            }
            if lifecycle_operations:
                record_id = sorted(lifecycle_operations)[0]
                record = self._records.get(record_id)
                raise BackendPluginLifecycleError(
                    "Runtime-pair validator registration raced plugin load or "
                    "interface validation",
                    plugin_id=record.plugin_id if record is not None else None,
                    entry_point=(
                        record.entry_point_name if record is not None else None
                    ),
                    field="runtime_pair_validator",
                    expected=(
                        "validator registration before plugin load and "
                        "runtime-pair validation"
                    ),
                    actual="plugin lifecycle operation in progress",
                    remediation=(
                        "Reset the Registry, then register the validator before "
                        "loading or selecting plugins."
                    ),
                )

            if current is not _MISSING and self._records:
                record = sorted(
                    self._records.values(), key=lambda item: item.record_id
                )[0]
                raise BackendPluginLifecycleError(
                    "Runtime-pair validator cannot be replaced after plugin "
                    "discovery",
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                    field="runtime_pair_validator",
                    expected=(
                        "an empty Registry before replacing a named validator"
                    ),
                    actual=record.state.value,
                    remediation=(
                        "Reset the Registry, then replace the validator before "
                        "discovering or loading plugins."
                    ),
                )

            published_states = {
                PluginLifecycleState.REGISTERED,
                PluginLifecycleState.SELECTED,
                PluginLifecycleState.ACTIVE,
            }
            published = tuple(
                sorted(
                    (
                        record
                        for record in self._records.values()
                        if record.initialized or record.state in published_states
                    ),
                    key=lambda record: record.record_id,
                )
            )
            if published:
                record = published[0]
                raise BackendPluginLifecycleError(
                    "Runtime-pair validator cannot be attached after plugin "
                    "publication",
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                    field="runtime_pair_validator",
                    expected="validator registration before plugin publication",
                    actual=record.state.value,
                    remediation=(
                        "Reset the Registry, then register the validator before "
                        "loading or selecting plugins."
                    ),
                )

            self._runtime_pair_validators[contract_id] = callback

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
                except Exception as exc:
                    source = _source_hint(distribution)
                    unexpected = BackendPluginManifestError(
                        "Unexpected error while reading distribution Manifest: "
                        f"{exc}",
                        field="distribution_manifest",
                        expected="readable, valid static Manifest metadata",
                        actual=f"<error: {exc}>",
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

    def list_legacy_records(self) -> Tuple[BackendPluginRecord, ...]:
        """Return deterministic Legacy metadata without importing plugins."""
        with self._lock:
            self._ensure_not_resetting("list_legacy_records")
            self.discover()
            return tuple(
                sorted(
                    (
                        record
                        for record in self._records.values()
                        if record.source is PluginSource.LEGACY
                    ),
                    key=lambda record: record.record_id,
                )
            )

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
            if reject_conflicts:
                match_ids = {record.record_id for record in matches}
                related_conflicts = tuple(
                    conflict
                    for conflict in detect_conflicts(
                        self._records.values()
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
        lease = self._legacy_leases.get(record.record_id)
        if lease is not None:
            lease._refresh(record, factory=_LEGACY_LEASE_FACTORY)
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
        if record.manifest is not None:
            try:
                validate_plugin_isolation(record.manifest)
            except BackendPluginManifestError as error:
                remaining = tuple(
                    existing
                    for existing in record.errors
                    if _error_identity(existing) != _error_identity(error)
                )
                rejected = self._replace(
                    replace(
                        record,
                        state=PluginLifecycleState.REJECTED,
                        compatibility_status=(
                            PluginCompatibilityStatus.NOT_CHECKED
                        ),
                        errors=(error,) + remaining,
                        plugin_object=None,
                        compiler_cls=None,
                        driver_cls=None,
                        initialized=False,
                        selected_targets=(),
                        compatibility_report=None,
                        capability_report=None,
                    )
                )
                invalidated = tuple(
                    target
                    for target, decision in self._selections.items()
                    if decision.record_id == record.record_id
                )
                for target in invalidated:
                    del self._selections[target]
                self._cleanup_stack[:] = [
                    cleanup_record
                    for cleanup_record in self._cleanup_stack
                    if cleanup_record.record_id != record.record_id
                ]
                if invalidated:
                    self._generation += 1
                return rejected
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

        errors = []
        report = None
        if self._preflight_profile == "triton_version":
            try:
                inspect_native_artifacts(
                    record.manifest,
                    record.distribution,
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
            compatibility_status = (
                PluginCompatibilityStatus.NOT_CHECKED
                if isinstance(ordered_errors[0], BackendPluginManifestError)
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
        record = self._validate_record(record)
        if record.state is PluginLifecycleState.REJECTED:
            return record
        if record.state in {
            PluginLifecycleState.LOADED,
            PluginLifecycleState.REGISTERED,
            PluginLifecycleState.SELECTED,
            PluginLifecycleState.ACTIVE,
        }:
            return record

        for conflict in detect_conflicts(self._records.values()).fatal_conflicts:
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
        return any(owner == thread_id for _epoch, owner, _token in operations.values())

    def _finish_operation(
        self,
        operations: Dict[str, Tuple[int, int, object]],
        record_id: str,
        token: object,
    ) -> None:
        operation = operations.get(record_id)
        if operation is not None and operation[2] is token:
            del operations[record_id]
        self._condition.notify_all()

    def _wait_for_lifecycle_owner(
        self,
        record: BackendPluginRecord,
        operation: str,
        waiter_thread: int,
        owner_thread: int,
    ) -> None:
        """Wait once unless doing so would close a lifecycle wait cycle."""
        cursor = owner_thread
        visited = set()
        while cursor not in visited:
            if cursor == waiter_thread:
                raise BackendPluginLifecycleError(
                    "Backend plugin lifecycle re-entry would deadlock",
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                    field=operation,
                    expected="an acyclic lifecycle dependency graph",
                    actual="cross-plugin lifecycle wait cycle",
                    remediation=(
                        "Do not synchronously load or register mutually "
                        "dependent plugins from entry-point or lifecycle hooks."
                    ),
                )
            visited.add(cursor)
            next_owner = self._waiting_for.get(cursor)
            if next_owner is None:
                break
            cursor = next_owner

        try:
            self._waiting_for[waiter_thread] = owner_thread
            self._condition.wait()
        finally:
            if self._waiting_for.get(waiter_thread) == owner_thread:
                del self._waiting_for[waiter_thread]

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

    def _check_expected_epoch(
        self,
        expected_epoch: Optional[int],
        record: Optional[BackendPluginRecord],
        operation: str,
    ) -> None:
        """Reject an internal operation that belongs to an older lifecycle."""
        if expected_epoch is None or expected_epoch == self._lifecycle_epoch:
            return
        if record is not None:
            raise self._stale_operation_error(
                record,
                operation,
                expected_epoch,
                self._lifecycle_epoch,
            )
        raise BackendPluginLifecycleError(
            f"Backend plugin {operation} belongs to a stale Registry generation",
            field="generation",
            expected=str(expected_epoch),
            actual=str(self._lifecycle_epoch),
            remediation=(
                f"Retry {operation} after Registry reset has completed; stale "
                "operations never rediscover or publish plugins."
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
                f"Unable to inspect backend {hook_name} hook: {exc}",
                expected="a callable hook or no hook",
                actual=f"<error: {exc}>",
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
                f"Unable to inspect backend {hook_name} hook: {exc}",
                expected="a callable hook or no hook",
                actual=f"<error: {exc}>",
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
        try:
            signature = inspect.signature(hook)
        except (TypeError, ValueError):
            # Some extension/builtin callables expose no inspectable signature.
            return None
        except Exception as exc:
            return self._hook_error(
                record,
                hook_name,
                f"Unable to inspect backend {hook_name} signature: {exc}",
                expected=(
                    "initialize(context)"
                    if hook_name == "initialize"
                    else f"{hook_name}()"
                ),
                actual=f"<error: {exc}>",
                remediation=f"Expose an introspectable valid {hook_name} signature.",
            )
        try:
            signature.bind(*args)
        except TypeError as exc:
            return self._hook_error(
                record,
                hook_name,
                f"Backend plugin {hook_name} hook has an incompatible signature",
                expected=(
                    "initialize(context)"
                    if hook_name == "initialize"
                    else f"{hook_name}()"
                ),
                actual=str(exc),
                remediation=f"Fix the {hook_name} signature to match Protocol 1.0.",
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
                f"Backend plugin shutdown() failed: {exc}",
                expected="successful cleanup returning None",
                actual=f"<error: {exc}>",
                remediation="Fix shutdown() so Registry reset can release resources.",
            )
        if inspect.isawaitable(value):
            _close_unawaited(value)
            return True, self._hook_error(
                record,
                "shutdown",
                "Backend plugin shutdown() returned an awaitable",
                expected="None from a synchronous shutdown() hook",
                actual=_stable_type_name(value),
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

    def load(
        self,
        identifier: str,
        *,
        _expected_epoch: Optional[int] = None,
        _expected_record: Optional[BackendPluginRecord] = None,
    ) -> BackendPluginRecord:
        """Import one plugin only after its applicable pre-load gate passes."""
        thread_id = threading.get_ident()
        while True:
            token: Any = _MISSING
            operation_finished = False
            try:
                with self._condition:
                    self._check_expected_epoch(
                        _expected_epoch,
                        _expected_record,
                        "load",
                    )
                    self._ensure_not_resetting("load")
                    self.discover()
                    record = self._resolve(
                        identifier,
                        reject_conflicts=True,
                    )
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
                        wait_epoch, owner, _existing_token = operation
                        if owner == thread_id:
                            raise BackendPluginLifecycleError(
                                "Recursive backend plugin load is not allowed",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="load",
                                expected="one non-reentrant load operation",
                                actual="recursive load",
                                remediation=(
                                    "Do not call Registry load/register for the "
                                    "same plugin from its entry-point loader or "
                                    "constructor."
                                ),
                            )
                        self._wait_for_lifecycle_owner(
                            record,
                            "load",
                            thread_id,
                            owner,
                        )
                        if self._lifecycle_epoch != wait_epoch:
                            raise self._stale_operation_error(
                                record,
                                "load",
                                wait_epoch,
                                self._lifecycle_epoch,
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

                # Entry-point loading and plugin construction are
                # plugin-controlled and must never run under the Registry lock.
                load_exception: Optional[Exception] = None
                try:
                    loaded = record.entry_point.load()
                    plugin = loaded() if isinstance(loaded, type) else loaded
                except Exception as exc:
                    load_exception = exc
                    plugin = None

                with self._condition:
                    stale = (
                        epoch != self._lifecycle_epoch
                        or record.record_id not in self._records
                    )
                    if stale:
                        error: BackendPluginError = self._stale_operation_error(
                            record, "load", epoch, self._lifecycle_epoch
                        )
                    elif load_exception is not None:
                        error = BackendPluginLoadError(
                            f"Failed to load backend plugin "
                            f"'{record.registry_key}': {load_exception}",
                            plugin_id=record.plugin_id,
                            entry_point=record.entry_point_name,
                            field="entry_point.load",
                            expected="a loadable backend plugin object",
                            actual=f"<error: {load_exception}>",
                            remediation=(
                                "Fix the backend Python package or native "
                                "dependencies and reinstall it."
                            ),
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
                        self._loading,
                        record.record_id,
                        token,
                    )
                    operation_finished = True

                if error is not None:
                    if load_exception is not None:
                        raise error from load_exception
                    raise error
                return result
            finally:
                # Programming errors and BaseException must not strand an
                # owner token and make reset/waiters block forever.
                if token is not _MISSING and not operation_finished:
                    with self._condition:
                        self._finish_operation(
                            self._loading,
                            record.record_id,
                            token,
                        )

    @staticmethod
    def _run_runtime_pair_validators(
        context: RuntimePairValidationContext,
        validators: Tuple[Tuple[str, RuntimePairValidator], ...],
    ) -> Optional[BackendPluginInterfaceError]:
        """Run a lock-free validator snapshot and aggregate its findings."""
        issues: list[BackendPluginInterfaceIssue] = []
        missing_fields: list[str] = []
        invalid_fields: list[str] = []
        field_error_messages: Dict[str, set[str]] = {}
        runtime_fields = {"compiler_cls", "driver_cls"}

        def absorb_interface_error(
            contract_id: str,
            error: BackendPluginInterfaceError,
        ) -> None:
            absorbed = False
            invalid_component = False
            for issue in error.interface_issues:
                if (
                    type(issue) is BackendPluginInterfaceIssue
                    and _is_well_formed_interface_issue(issue)
                ):
                    issues.append(issue)
                    absorbed = True
                else:
                    invalid_component = True
            for field_name in error.missing_fields:
                if type(field_name) is str and field_name in runtime_fields:
                    missing_fields.append(field_name)
                    absorbed = True
                else:
                    invalid_component = True
            for field_name in error.invalid_fields:
                if type(field_name) is str and field_name in runtime_fields:
                    invalid_fields.append(field_name)
                    absorbed = True
                else:
                    invalid_component = True
            for field_name, message in error.field_errors.items():
                if (
                    type(field_name) is str
                    and field_name in runtime_fields
                    and type(message) is str
                    and bool(message.strip())
                ):
                    field_error_messages.setdefault(field_name, set()).add(
                        message
                    )
                    absorbed = True
                else:
                    invalid_component = True
            if invalid_component:
                issues.append(
                    _runtime_pair_validator_failure(
                        contract_id,
                        "backend_plugin_interface_error",
                        "invalid_result",
                    )
                )
            if not absorbed and not invalid_component:
                issues.append(
                    _runtime_pair_validator_failure(
                        contract_id,
                        "backend_plugin_interface_error",
                        "validator_rejection",
                    )
                )

        for contract_id, callback in validators:
            try:
                result = callback(context)
            except BackendPluginInterfaceError as exc:
                absorb_interface_error(contract_id, exc)
                continue
            except Exception as exc:
                issues.append(
                    _runtime_pair_validator_failure(
                        contract_id,
                        _stable_type_name(exc),
                        "validator_failure",
                    )
                )
                continue

            if result is None:
                continue
            _close_unawaited(result)
            try:
                iterator = iter(result)
            except BackendPluginInterfaceError as exc:
                absorb_interface_error(contract_id, exc)
                continue
            except Exception:
                issues.append(
                    _runtime_pair_validator_failure(
                        contract_id,
                        _stable_type_name(result),
                        "invalid_result",
                    )
                )
                continue

            while True:
                try:
                    issue = next(iterator)
                except StopIteration:
                    break
                except BackendPluginInterfaceError as exc:
                    absorb_interface_error(contract_id, exc)
                    break
                except Exception as exc:
                    issues.append(
                        _runtime_pair_validator_failure(
                            contract_id,
                            _stable_type_name(exc),
                            "validator_failure",
                        )
                    )
                    break
                if type(issue) is not BackendPluginInterfaceIssue:
                    issues.append(
                        _runtime_pair_validator_failure(
                            contract_id,
                            _stable_type_name(issue),
                            "invalid_result",
                        )
                    )
                    break
                if not _is_well_formed_interface_issue(issue):
                    issues.append(
                        _runtime_pair_validator_failure(
                            contract_id,
                            _stable_type_name(issue),
                            "invalid_result",
                        )
                    )
                    break
                issues.append(issue)

        if not (
            issues
            or missing_fields
            or invalid_fields
            or field_error_messages
        ):
            return None
        field_errors = {
            field_name: " | ".join(sorted(messages))
            for field_name, messages in sorted(field_error_messages.items())
        }
        return BackendPluginInterfaceError(
            missing_fields,
            invalid_fields=invalid_fields,
            field_errors=field_errors,
            interface_issues=issues,
            plugin_id=context.plugin_id,
            entry_point=context.entry_point,
        )

    def register(
        self,
        identifier: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        _expected_epoch: Optional[int] = None,
        _expected_record: Optional[BackendPluginRecord] = None,
    ) -> BackendPluginRecord:
        """Check the runtime pair, initialize once, and register one plugin."""
        with self._condition:
            operation_epoch = (
                self._lifecycle_epoch
                if _expected_epoch is None
                else _expected_epoch
            )
            self._check_expected_epoch(
                operation_epoch,
                _expected_record,
                "register",
            )
            self._ensure_not_resetting("register")
        loaded_record = self.load(
            identifier,
            _expected_epoch=operation_epoch,
            _expected_record=_expected_record,
        )
        record_id = loaded_record.record_id
        thread_id = threading.get_ident()

        while True:
            token: Any = _MISSING
            operation_finished = False
            initializer_invoked = False
            cleanup_attempted = False
            try:
                with self._condition:
                    self._check_expected_epoch(
                        operation_epoch,
                        _expected_record or loaded_record,
                        "register",
                    )
                    self._ensure_not_resetting("register")
                    record = self._records.get(record_id)
                    if record is None:
                        raise self._stale_operation_error(
                            loaded_record,
                            "register",
                            operation_epoch,
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
                        )

                    operation = self._registering.get(record_id)
                    if operation is not None:
                        wait_epoch, owner, _existing_token = operation
                        if owner == thread_id:
                            raise BackendPluginLifecycleError(
                                "Recursive backend plugin registration is not "
                                "allowed",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="register",
                                expected=(
                                    "one non-reentrant registration operation"
                                ),
                                actual="recursive registration",
                                remediation=(
                                    "Do not call Registry register for the same "
                                    "plugin from runtime attributes or initialize()."
                                ),
                            )
                        self._wait_for_lifecycle_owner(
                            record,
                            "register",
                            thread_id,
                            owner,
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
                            "Custom initialize context is not supported by "
                            "Protocol 1.0",
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
                    token = object()
                    self._registering[record_id] = (
                        epoch,
                        thread_id,
                        token,
                    )
                    runtime_pair_validators = tuple(
                        sorted(self._runtime_pair_validators.items())
                    )

                # Runtime attributes, signature introspection, and hooks are
                # plugin-controlled and therefore run outside the Registry lock.
                values: Dict[str, Any] = {}
                field_errors: Dict[str, str] = {}
                for name in ("compiler_cls", "driver_cls"):
                    try:
                        values[name] = getattr(
                            record.plugin_object,
                            name,
                            _MISSING,
                        )
                    except Exception as exc:
                        values[name] = _MISSING
                        field_errors[name] = str(exc)
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
                    and not _is_class_object(value)
                )

                primary_error: Optional[BackendPluginError] = None
                cleanup_error: Optional[BackendPluginLifecycleError] = None
                cleanup_called = False
                initialized = False
                initialization_context: Optional[Mapping[str, Any]] = None
                if missing or invalid or field_errors:
                    primary_error = BackendPluginInterfaceError(
                        missing,
                        invalid_fields=invalid,
                        field_errors=field_errors,
                        plugin_id=record.plugin_id,
                        entry_point=record.entry_point_name,
                    )
                else:
                    validation_context = RuntimePairValidationContext(
                        record_id=record.record_id,
                        plugin_id=record.plugin_id,
                        entry_point=record.entry_point_name,
                        compiler_cls=values["compiler_cls"],
                        driver_cls=values["driver_cls"],
                    )
                    primary_error = self._run_runtime_pair_validators(
                        validation_context,
                        runtime_pair_validators,
                    )

                if (
                    primary_error is None
                    and record.source is PluginSource.MANIFEST
                ):
                    # Materialize the lifecycle context only after every
                    # runtime-pair validator passes.  Besides preserving the
                    # documented boundary, this prevents a rejected plugin's
                    # global ABC mutations from being reached by snapshot
                    # helpers before interface validation.
                    with self._condition:
                        current = self._records.get(record_id)
                        if (
                            epoch != self._lifecycle_epoch
                            or current is None
                        ):
                            primary_error = self._stale_operation_error(
                                record,
                                "register",
                                epoch,
                                self._lifecycle_epoch,
                            )
                        elif current.state is PluginLifecycleState.REJECTED:
                            primary_error = current.error or (
                                BackendPluginLifecycleError(
                                    "Backend plugin state changed before "
                                    "initialization",
                                    plugin_id=record.plugin_id,
                                    entry_point=record.entry_point_name,
                                    field="state",
                                    expected=PluginLifecycleState.LOADED.value,
                                    actual=current.state.value,
                                    remediation=(
                                        "Resolve the recorded validation or "
                                        "conflict error before registering."
                                    ),
                                )
                            )
                        elif current.state is not PluginLifecycleState.LOADED:
                            primary_error = BackendPluginLifecycleError(
                                "Backend plugin state changed before "
                                "initialization",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="state",
                                expected=PluginLifecycleState.LOADED.value,
                                actual=current.state.value,
                                remediation=(
                                    "Retry registration through the Registry."
                                ),
                            )
                        else:
                            initialization_context = MappingProxyType(
                                {
                                    "environment": self._get_environment(),
                                    "manifest": _manifest_snapshot(
                                        record.manifest
                                    ),
                                    "record_id": record.record_id,
                                }
                            )
                    if primary_error is not None:
                        initializer = _MISSING
                    else:
                        initializer, primary_error = self._read_hook(
                            record,
                            "initialize",
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
                        except Exception as exc:
                            primary_error = self._hook_error(
                                record,
                                "initialize",
                                f"Backend plugin initialize() failed: {exc}",
                                expected="successful initialization returning None",
                                actual=f"<error: {exc}>",
                                remediation=(
                                    "Fix initialize(), release partial resources, "
                                    "and reinstall the backend."
                                ),
                            )
                        else:
                            if inspect.isawaitable(initialize_result):
                                _close_unawaited(initialize_result)
                                primary_error = self._hook_error(
                                    record,
                                    "initialize",
                                    "Backend plugin initialize() returned an awaitable",
                                    expected=(
                                        "None from a synchronous "
                                        "initialize(context) hook"
                                    ),
                                    actual=_stable_type_name(initialize_result),
                                    remediation=(
                                        "Implement initialize(context) synchronously "
                                        "and return None."
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
                        cleanup_attempted = True
                        cleanup_called, cleanup_error = self._call_shutdown(
                            record
                        )

                deferred_cleanup_needed = False
                with self._condition:
                    stale = (
                        epoch != self._lifecycle_epoch
                        or record_id not in self._records
                    )
                    if stale:
                        error = self._stale_operation_error(
                            record,
                            "register",
                            epoch,
                            self._lifecycle_epoch,
                        )
                        # A successful Manifest registration invalidated before
                        # publication still owns the same shutdown obligation as
                        # a published registration.  Keep the owner token until
                        # that lock-free cleanup completes, so reset waits for it.
                        deferred_cleanup_needed = (
                            primary_error is None
                            and record.source is PluginSource.MANIFEST
                        )
                        if not deferred_cleanup_needed:
                            if (
                                cleanup_error is not None
                                and self._resetting
                            ):
                                self._reset_operation_errors.append(
                                    cleanup_error
                                )
                            self._finish_operation(
                                self._registering,
                                record_id,
                                token,
                            )
                            operation_finished = True
                    else:
                        current = self._records[record_id]
                        if primary_error is not None:
                            errors = (
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
                        elif current.state is PluginLifecycleState.REJECTED:
                            error = current.error or BackendPluginLifecycleError(
                                "Backend plugin state changed while registering",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="state",
                                expected=PluginLifecycleState.LOADED.value,
                                actual=current.state.value,
                                remediation=(
                                    "Resolve the recorded conflict or validation "
                                    "failure before registering the plugin."
                                ),
                            )
                            deferred_cleanup_needed = (
                                record.source is PluginSource.MANIFEST
                            )
                        elif current.state is not PluginLifecycleState.LOADED:
                            error = BackendPluginLifecycleError(
                                "Backend plugin state changed while registering",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="state",
                                expected=PluginLifecycleState.LOADED.value,
                                actual=current.state.value,
                                remediation=(
                                    "Retry registration through the Registry; "
                                    "concurrent lifecycle publication is rejected."
                                ),
                            )
                            deferred_cleanup_needed = (
                                record.source is PluginSource.MANIFEST
                            )
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
                        if not deferred_cleanup_needed:
                            self._finish_operation(
                                self._registering,
                                record_id,
                                token,
                            )
                            operation_finished = True

                if deferred_cleanup_needed:
                    cleanup_attempted = True
                    cleanup_called, cleanup_error = self._call_shutdown(record)
                    with self._condition:
                        current = self._records.get(record_id)
                        if (
                            current is not None
                            and epoch == self._lifecycle_epoch
                            and current.state is PluginLifecycleState.REJECTED
                        ):
                            current = self._replace(
                                replace(
                                    current,
                                    shutdown_called=cleanup_called,
                                )
                            )
                            if cleanup_error is not None:
                                self._reject_many(
                                    current,
                                    (cleanup_error,),
                                    current.compatibility_status,
                                )
                        elif cleanup_error is not None and self._resetting:
                            self._reset_operation_errors.append(cleanup_error)
                        self._finish_operation(
                            self._registering,
                            record_id,
                            token,
                        )
                        operation_finished = True

                if error is not None:
                    raise error
                return result
            finally:
                if token is not _MISSING and not operation_finished:
                    try:
                        if initializer_invoked and not cleanup_attempted:
                            cleanup_attempted = True
                            self._call_shutdown(record)
                    finally:
                        # Never leave an owner token behind, even when a
                        # programming error or BaseException escapes.
                        with self._condition:
                            self._finish_operation(
                                self._registering,
                                record_id,
                                token,
                            )

    @staticmethod
    def _legacy_target_name(target: Any) -> str:
        # This may execute a target object's ``backend`` property.  Every
        # caller must invoke it before acquiring the Registry lock; the
        # returned exact ``str`` is inert inside selection dictionaries.
        return _target_name(target)

    @staticmethod
    def _legacy_capability_names(values: Iterable[str]) -> Tuple[str, ...]:
        # Iterable materialization can execute caller code.  Like target
        # normalization, keep it outside the Registry lock and retain only
        # exact built-in strings for the locked selection phase.
        return _capability_names(values, "kernel_required_capabilities")

    def _validate_legacy_lease(
        self,
        lease: LegacyRecordLease,
        operation: str,
    ) -> BackendPluginRecord:
        """Validate Registry, reset-epoch, record, and runtime-pair identity."""
        if type(lease) is not LegacyRecordLease:
            raise BackendPluginLifecycleError(
                f"Backend plugin {operation} requires a Registry-issued "
                "Legacy record lease",
                field="legacy_lease",
                expected="LegacyRecordLease from this Registry",
                actual=_stable_type_name(lease),
                remediation=(
                    "Call materialize_legacy() for an exact record in the "
                    "current Registry generation."
                ),
            )

        record = lease.record
        if lease._registry_identity is not self._legacy_lease_identity:
            raise BackendPluginLifecycleError(
                f"Backend plugin {operation} received a lease from another "
                "Registry",
                plugin_id=record.plugin_id,
                entry_point=record.entry_point_name,
                field="legacy_lease.registry",
                expected="the current BackendPluginRegistry",
                actual="a different BackendPluginRegistry",
                remediation="Materialize the Legacy record in this Registry.",
            )
        if lease._lifecycle_epoch != self._lifecycle_epoch:
            raise BackendPluginLifecycleError(
                f"Backend plugin {operation} lease belongs to a stale "
                "Registry lifecycle epoch",
                plugin_id=record.plugin_id,
                entry_point=record.entry_point_name,
                field="registry lifecycle_epoch",
                expected=str(lease._lifecycle_epoch),
                actual=str(self._lifecycle_epoch),
                remediation=(
                    f"Retry {operation} after Registry reset has completed; "
                    "stale Legacy runtime pairs are never published."
                ),
            )

        current = self._records.get(record.record_id)
        canonical = self._legacy_leases.get(record.record_id)
        if (
            canonical is not lease
            or self._legacy_lease_tokens.get(record.record_id) is not lease._token
            or current is None
            or current is not record
        ):
            raise BackendPluginLifecycleError(
                f"Backend plugin {operation} received a stale or replaced "
                "Legacy record lease",
                plugin_id=record.plugin_id,
                entry_point=record.entry_point_name,
                field="legacy_lease.record",
                expected="the current Registry record identity",
                actual="missing or replaced record identity",
                remediation=(
                    "Discard the lease and materialize the current Legacy "
                    "record again."
                ),
            )
        if (
            current.source is not PluginSource.LEGACY
            or current.compatibility_status
            is not PluginCompatibilityStatus.LEGACY_UNVERIFIED
        ):
            raise BackendPluginLifecycleError(
                f"Backend plugin {operation} lease no longer identifies an "
                "unverified Legacy record",
                plugin_id=current.plugin_id,
                entry_point=current.entry_point_name,
                field="legacy_lease.source",
                expected="source=legacy; compatibility=legacy_unverified",
                actual=(
                    f"source={getattr(current.source, 'value', None)}; "
                    "compatibility="
                    f"{current.compatibility_status.value}"
                ),
                remediation="Rediscover and materialize an exact Legacy record.",
            )
        if (
            current.plugin_object is not lease._plugin_object
            or current.compiler_cls is not lease._compiler_cls
            or current.driver_cls is not lease._driver_cls
        ):
            raise BackendPluginLifecycleError(
                f"Backend plugin {operation} runtime pair differs from its "
                "materialized Legacy lease",
                entry_point=current.entry_point_name,
                field="legacy_lease.runtime_pair",
                expected="the exact materialized plugin/compiler/driver identities",
                actual="one or more runtime identities changed",
                remediation=(
                    "Reject mutable Legacy runtime-pair publication and retry "
                    "after Registry reset."
                ),
            )
        if current.state is PluginLifecycleState.REJECTED:
            if current.error is not None:
                raise current.error
            raise BackendPluginLifecycleError(
                f"Rejected Legacy backend cannot perform {operation}",
                entry_point=current.entry_point_name,
                field="state",
                expected="registered, selected, or active",
                actual=current.state.value,
                remediation="Resolve the recorded Legacy failure first.",
            )
        if current.state not in {
            PluginLifecycleState.REGISTERED,
            PluginLifecycleState.SELECTED,
            PluginLifecycleState.ACTIVE,
        }:
            raise BackendPluginLifecycleError(
                f"Legacy backend must be registered before {operation}",
                entry_point=current.entry_point_name,
                field="state",
                expected="registered, selected, or active",
                actual=current.state.value,
                remediation="Materialize the exact Legacy record first.",
            )
        return current

    def materialize_legacy(self, identifier: str) -> LegacyRecordLease:
        """Load/register one exact Legacy record and issue an epoch lease.

        ``identifier`` accepts only a record ID or a unique Registry key.  In
        particular, an entry-point name is not an implicit Legacy selector.
        Plugin import, runtime-pair inspection, and F6 validators occur in
        :meth:`register`; none of them run while this method holds the lock.
        """
        if not isinstance(identifier, str) or not identifier:
            raise BackendPluginSelectionError(
                "Legacy materialization identifier must be a non-empty string",
                field="registry_key",
                expected="an exact record_id or unique registry_key",
                actual=repr(identifier),
                remediation="Use a key returned by list_legacy_records().",
            )
        with self._condition:
            self._ensure_not_resetting("materialize_legacy")
            self.discover()
            record = self._resolve(identifier, reject_conflicts=True)
            if record.source is not PluginSource.LEGACY:
                raise BackendPluginSelectionError(
                    f"Backend record '{identifier}' is not a Legacy record",
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                    field="source",
                    expected=PluginSource.LEGACY.value,
                    actual=(
                        record.source.value
                        if record.source is not None
                        else "<unknown>"
                    ),
                    remediation=(
                        "Use Registry.select() for Manifest plugins or choose "
                        "an exact key returned by list_legacy_records()."
                    ),
                )
            materialization_epoch = self._lifecycle_epoch
            expected_record = record

        materialized = self.register(
            record.record_id,
            _expected_epoch=materialization_epoch,
            _expected_record=expected_record,
        )

        with self._condition:
            self._check_expected_epoch(
                materialization_epoch,
                expected_record,
                "materialize_legacy",
            )
            self._ensure_not_resetting("materialize_legacy")
            current = self._records.get(materialized.record_id)
            if current is None:
                raise self._stale_operation_error(
                    expected_record,
                    "materialize_legacy",
                    materialization_epoch,
                    self._lifecycle_epoch,
                )
            if (
                current.source is not PluginSource.LEGACY
                or current.compatibility_status
                is not PluginCompatibilityStatus.LEGACY_UNVERIFIED
            ):
                raise BackendPluginLifecycleError(
                    "Materialized record no longer has Legacy compatibility state",
                    plugin_id=current.plugin_id,
                    entry_point=current.entry_point_name,
                    field="source,compatibility_status",
                    expected="legacy,legacy_unverified",
                    actual=(
                        f"{getattr(current.source, 'value', None)},"
                        f"{current.compatibility_status.value}"
                    ),
                    remediation="Reset and rediscover the exact Legacy record.",
                )
            lease = self._legacy_leases.get(current.record_id)
            if lease is None:
                token = object()
                lease = LegacyRecordLease(
                    current,
                    registry_identity=self._legacy_lease_identity,
                    lifecycle_epoch=materialization_epoch,
                    token=token,
                    factory=_LEGACY_LEASE_FACTORY,
                )
                self._legacy_leases[current.record_id] = lease
                self._legacy_lease_tokens[current.record_id] = token
            else:
                lease._refresh(current, factory=_LEGACY_LEASE_FACTORY)
            self._validate_legacy_lease(lease, "materialize_legacy")
            return lease

    def select_materialized_legacy(
        self,
        target: Any,
        *,
        lease: LegacyRecordLease,
        kernel_required_capabilities: Iterable[str] = (),
    ) -> SelectionDecision:
        """Conditionally publish one already materialized Legacy record.

        A different record already selected for the target is never replaced.
        Repeating the same record/target decision is idempotent and does not
        advance the public generation.
        """
        target_name = self._legacy_target_name(target)
        required = self._legacy_capability_names(
            kernel_required_capabilities
        )
        with self._condition:
            self._ensure_not_resetting("select_materialized_legacy")
            current = self._validate_legacy_lease(
                lease, "select_materialized_legacy"
            )
            # Reuse W8's target/capability input validation.  The exact
            # selector keeps Legacy out of automatic Manifest ranking.
            decision = select_backend(
                (current,),
                target=target_name,
                kernel_required_capabilities=required,
                core_provided_capabilities=self._core_capabilities,
                explicit_selector=current.record_id,
                environment={},
            )
            decision = replace(
                decision,
                method=SelectionMethod.LEGACY_FALLBACK,
                selector=None,
                candidate_record_ids=(current.record_id,),
                record=current,
            )
            target_name = decision.target
            previous = self._selections.get(target_name)
            if previous is not None:
                if previous.record_id == current.record_id:
                    if target_name not in current.selected_targets:
                        raise BackendPluginLifecycleError(
                            "Legacy selection cache and record targets disagree",
                            entry_point=current.entry_point_name,
                            field="selected_targets",
                            expected=target_name,
                            actual=", ".join(current.selected_targets) or "<empty>",
                            remediation="Reset the Registry before retrying selection.",
                        )
                    return previous
                raise BackendPluginConflictError(
                    f"Target '{target_name}' is already bound to backend "
                    f"'{previous.record_id}', not Legacy record "
                    f"'{current.record_id}'",
                    entry_point=current.entry_point_name,
                    conflict_kind="selected_target",
                    claim=target_name,
                    related_plugin_ids=(
                        tuple(
                            plugin_id
                            for plugin_id in (previous.plugin_id, current.plugin_id)
                            if plugin_id is not None
                        )
                    ),
                    related_record_ids=(previous.record_id, current.record_id),
                    field="targets",
                    expected=previous.record_id,
                    actual=current.record_id,
                    remediation=(
                        "Keep the current target binding or reset it before "
                        "selecting another backend record."
                    ),
                )

            selected_state = current.state
            if selected_state is PluginLifecycleState.REGISTERED:
                self._transition(current, PluginLifecycleState.SELECTED)
                selected_state = PluginLifecycleState.SELECTED
            selected_targets = tuple(
                sorted(set(current.selected_targets).union({target_name}))
            )
            selected = self._replace(
                replace(
                    current,
                    state=selected_state,
                    selected_targets=selected_targets,
                )
            )
            decision = replace(decision, record=selected)
            self._selections[target_name] = decision
            self._generation += 1
            return decision

    def commit_materialized_legacy(
        self,
        target: Any,
        *,
        lease: LegacyRecordLease,
        publisher: Callable[[SelectionDecision, BackendPluginRecord], Any],
        activate: bool = False,
        kernel_required_capabilities: Iterable[str] = (),
    ) -> Tuple[SelectionDecision, BackendPluginRecord, Any]:
        """Atomically select and publish one leased Legacy runtime pair.

        ``publisher`` is a trusted integration callback that may only mutate
        an adapter-owned in-memory cache.  It runs under the Registry
        condition after the provisional Core state is installed, so reset
        cannot clear Core ownership between selection and public publication.
        If publication fails, the exact prior record and selection snapshots
        are restored before the error escapes.  With ``activate=True`` the
        global ACTIVE conflict check and transition are part of the same
        transaction.
        """
        if not callable(publisher):
            raise TypeError("Legacy publisher must be callable")
        target_name = self._legacy_target_name(target)
        required = self._legacy_capability_names(
            kernel_required_capabilities
        )

        with self._condition:
            self._ensure_not_resetting("commit_materialized_legacy")
            current = self._validate_legacy_lease(
                lease, "commit_materialized_legacy"
            )
            candidate = select_backend(
                (current,),
                target=target_name,
                kernel_required_capabilities=required,
                core_provided_capabilities=self._core_capabilities,
                explicit_selector=current.record_id,
                environment={},
            )
            candidate = replace(
                candidate,
                method=SelectionMethod.LEGACY_FALLBACK,
                selector=None,
                candidate_record_ids=(current.record_id,),
                record=current,
            )

            previous_decision = self._selections.get(target_name)
            if (
                previous_decision is not None
                and previous_decision.record_id != current.record_id
            ):
                raise BackendPluginConflictError(
                    f"Target '{target_name}' is already bound to backend "
                    f"'{previous_decision.record_id}', not Legacy record "
                    f"'{current.record_id}'",
                    entry_point=current.entry_point_name,
                    conflict_kind="selected_target",
                    claim=target_name,
                    related_plugin_ids=tuple(
                        plugin_id
                        for plugin_id in (
                            previous_decision.plugin_id,
                            current.plugin_id,
                        )
                        if plugin_id is not None
                    ),
                    related_record_ids=(
                        previous_decision.record_id,
                        current.record_id,
                    ),
                    field="targets",
                    expected=previous_decision.record_id,
                    actual=current.record_id,
                    remediation=(
                        "Keep the current target binding or reset it before "
                        "selecting another backend record."
                    ),
                )

            if activate:
                other_active = tuple(
                    record
                    for record in self._records.values()
                    if (
                        record.record_id != current.record_id
                        and record.state is PluginLifecycleState.ACTIVE
                    )
                )
                if other_active:
                    active_ids = tuple(
                        sorted(record.record_id for record in other_active)
                    )
                    raise BackendPluginConflictError(
                        "Cannot activate more than one backend plugin: "
                        + ", ".join(active_ids + (current.record_id,)),
                        entry_point=current.entry_point_name,
                        field="active",
                        expected="one ACTIVE backend plugin",
                        actual=", ".join(active_ids + (current.record_id,)),
                        related_record_ids=active_ids + (current.record_id,),
                        remediation=(
                            "Keep the current active driver or reset runtime "
                            "state before activating another backend."
                        ),
                    )

            previous_record = current
            previous_record_decisions = {
                name: decision
                for name, decision in self._selections.items()
                if decision.record_id == current.record_id
            }
            selected_targets = tuple(
                sorted(set(current.selected_targets).union({target_name}))
            )
            state = current.state
            if activate:
                if state is PluginLifecycleState.REGISTERED:
                    self._transition(current, PluginLifecycleState.SELECTED)
                    intermediate = replace(
                        current,
                        state=PluginLifecycleState.SELECTED,
                    )
                    self._transition(intermediate, PluginLifecycleState.ACTIVE)
                elif state is PluginLifecycleState.SELECTED:
                    self._transition(current, PluginLifecycleState.ACTIVE)
                state = PluginLifecycleState.ACTIVE
            elif state is PluginLifecycleState.REGISTERED:
                self._transition(current, PluginLifecycleState.SELECTED)
                state = PluginLifecycleState.SELECTED
            if (
                state is current.state
                and selected_targets == current.selected_targets
            ):
                selected = current
            else:
                selected = self._replace(
                    replace(
                        current,
                        state=state,
                        selected_targets=selected_targets,
                    )
                )
            if previous_decision is None:
                decision = replace(candidate, record=selected)
                self._selections[target_name] = decision
            else:
                decision = self._selections[target_name]

            publication_epoch = self._lifecycle_epoch
            self._legacy_publication_owner = threading.get_ident()
            try:
                publication = publisher(decision, selected)
                if (
                    publication_epoch != self._lifecycle_epoch
                    or self._records.get(selected.record_id) is not selected
                    or self._selections.get(target_name) is not decision
                ):
                    raise BackendPluginLifecycleError(
                        "Legacy publication changed Registry ownership re-entrantly",
                        plugin_id=selected.plugin_id,
                        entry_point=selected.entry_point_name,
                        field="legacy publication",
                        expected="the exact provisional record and selection",
                        actual="reset or lifecycle mutation during publication",
                        remediation=(
                            "Publisher callbacks must only mutate their bounded "
                            "adapter cache and must not re-enter Registry."
                        ),
                    )
            except BaseException:
                # Restore exact identities as well as values.  Conditional
                # cleanup APIs use object identity to reject ABA operations.
                if (
                    publication_epoch == self._lifecycle_epoch
                    and self._records.get(selected.record_id) is selected
                    and self._selections.get(target_name) is decision
                ):
                    self._records[previous_record.record_id] = previous_record
                    lease._refresh(
                        previous_record,
                        factory=_LEGACY_LEASE_FACTORY,
                    )
                    for name, stored in tuple(self._selections.items()):
                        if stored.record_id == previous_record.record_id:
                            self._selections.pop(name, None)
                    self._selections.update(previous_record_decisions)
                raise
            finally:
                self._legacy_publication_owner = None

            if previous_decision is None:
                self._generation += 1
            return decision, selected, publication

    def select_and_activate_materialized_legacy(
        self,
        target: Any,
        *,
        lease: LegacyRecordLease,
        publisher: Optional[
            Callable[[SelectionDecision, BackendPluginRecord], Any]
        ] = None,
    ) -> BackendPluginRecord:
        """Atomically bind and activate one leased Legacy pair.

        The optional publisher is reserved for a trusted integration cache;
        normal Core callers can omit it and observe only Registry state.
        """
        callback = publisher or (lambda _decision, _record: None)
        _decision, record, _publication = self.commit_materialized_legacy(
            target,
            lease=lease,
            publisher=callback,
            activate=True,
        )
        return record

    def reject_materialized_legacy(
        self,
        lease: LegacyRecordLease,
        error: BackendPluginError,
        *,
        unpublisher: Optional[
            Callable[[BackendPluginRecord], Any]
        ] = None,
    ) -> BackendPluginRecord:
        """Reject one probed Legacy pair and atomically remove its selections."""
        if not isinstance(error, BackendPluginError):
            raise TypeError("Legacy rejection requires a BackendPluginError")
        if unpublisher is not None and not callable(unpublisher):
            raise TypeError("Legacy unpublisher must be callable")
        with self._condition:
            self._ensure_not_resetting("reject_materialized_legacy")
            current = self._validate_legacy_lease(
                lease, "reject_materialized_legacy"
            )
            # Adapter publication follows the Registry -> adapter lock order
            # also used by reset and commit_materialized_legacy().  Removing
            # the bounded public cache first prevents a concurrently visible
            # mapping from outliving the record that owns it; the Registry
            # condition prevents another governed publisher from racing this
            # rejection.
            if unpublisher is not None:
                self._legacy_publication_owner = threading.get_ident()
                try:
                    unpublisher(current)
                finally:
                    self._legacy_publication_owner = None
            invalidated = tuple(
                target
                for target, decision in self._selections.items()
                if decision.record_id == current.record_id
            )
            for target in invalidated:
                del self._selections[target]
            current = self._replace(replace(current, selected_targets=()))
            rejected = self._reject(
                current,
                error,
                PluginCompatibilityStatus.LEGACY_UNVERIFIED,
            )
            if invalidated:
                self._generation += 1
            return rejected

    def commit_selection_if_current(
        self,
        decision: SelectionDecision,
        *,
        publisher: Callable[[SelectionDecision, BackendPluginRecord], Any],
        activate: bool = False,
    ) -> Tuple[SelectionDecision, BackendPluginRecord, Any]:
        """Publish one exact existing selection at a bounded adapter boundary.

        Plugin probes and constructors run before this method.  The callback
        may only mutate an adapter-owned in-memory cache.  Exact decision and
        record identities close reset/reselection ABA windows, while optional
        activation and publication share the Registry critical section.
        """
        if type(decision) is not SelectionDecision:
            raise TypeError("selection commit requires a SelectionDecision")
        if not callable(publisher):
            raise TypeError("selection publisher must be callable")
        for field_name, value in (
            ("selection.target", decision.target),
            ("selection.record_id", decision.record_id),
        ):
            if type(value) is not str or not value:
                raise BackendPluginLifecycleError(
                    "Selection commit identity is not an inert string",
                    field=field_name,
                    expected="an exact non-empty built-in string",
                    actual=_stable_type_name(value),
                    remediation="Use an unmodified Registry-issued decision.",
                )

        with self._condition:
            self._ensure_not_resetting("commit_selection_if_current")
            stored = self._selections.get(decision.target)
            if (
                stored is None
                or stored.ownership_token is not decision.ownership_token
            ):
                raise BackendPluginLifecycleError(
                    "Backend selection changed before adapter publication",
                    plugin_id=decision.plugin_id,
                    entry_point=decision.entry_point_name,
                    field="selection",
                    expected=(
                        f"current decision {decision.target} -> "
                        f"{decision.record_id}"
                    ),
                    actual=(
                        "<missing>"
                        if stored is None
                        else f"current decision {stored.target} -> {stored.record_id}"
                    ),
                    remediation=(
                        "Retry consumption using the current Registry selection."
                    ),
                )
            current = self._records.get(decision.record_id)
            if current is None or stored.record is not current:
                raise BackendPluginLifecycleError(
                    "Backend selection record changed before adapter publication",
                    plugin_id=decision.plugin_id,
                    entry_point=decision.entry_point_name,
                    field="selection.record",
                    expected="the exact current Registry record",
                    actual="missing or replaced record identity",
                    remediation=(
                        "Retry consumption after the concurrent lifecycle change."
                    ),
                )
            if decision.target not in current.selected_targets:
                raise BackendPluginLifecycleError(
                    "Backend record no longer owns the selected target",
                    plugin_id=current.plugin_id,
                    entry_point=current.entry_point_name,
                    field="selected_targets",
                    expected=decision.target,
                    actual=", ".join(current.selected_targets) or "<none>",
                    remediation="Retry from the current Registry selection.",
                )

            previous_record = current
            previous_decisions = {
                target: item
                for target, item in self._selections.items()
                if item.record_id == current.record_id
            }
            if activate and current.state is not PluginLifecycleState.ACTIVE:
                other_active = tuple(
                    record
                    for record in self._records.values()
                    if (
                        record.record_id != current.record_id
                        and record.state is PluginLifecycleState.ACTIVE
                    )
                )
                if other_active:
                    active_ids = tuple(
                        sorted(record.record_id for record in other_active)
                    )
                    raise BackendPluginConflictError(
                        "Cannot activate more than one backend plugin: "
                        + ", ".join(active_ids + (current.record_id,)),
                        plugin_id=current.plugin_id,
                        entry_point=current.entry_point_name,
                        field="active",
                        expected="one ACTIVE backend plugin",
                        actual=", ".join(active_ids + (current.record_id,)),
                        related_record_ids=active_ids + (current.record_id,),
                        remediation=(
                            "Reset runtime state before activating another backend."
                        ),
                    )
                if current.state is PluginLifecycleState.SELECTED:
                    self._transition(current, PluginLifecycleState.ACTIVE)
                    current = self._replace(
                        replace(current, state=PluginLifecycleState.ACTIVE)
                    )
                    stored = self._selections[decision.target]
                elif current.state is not PluginLifecycleState.ACTIVE:
                    raise BackendPluginLifecycleError(
                        "Backend selection is not ready for activation",
                        plugin_id=current.plugin_id,
                        entry_point=current.entry_point_name,
                        field="state",
                        expected="selected or active",
                        actual=current.state.value,
                        remediation="Select the backend before runtime activation.",
                    )

            publication_epoch = self._lifecycle_epoch
            self._legacy_publication_owner = threading.get_ident()
            try:
                publication = publisher(stored, current)
                if (
                    publication_epoch != self._lifecycle_epoch
                    or self._records.get(current.record_id) is not current
                    or self._selections.get(stored.target) is not stored
                ):
                    raise BackendPluginLifecycleError(
                        "Backend publication changed Registry ownership re-entrantly",
                        plugin_id=current.plugin_id,
                        entry_point=current.entry_point_name,
                        field="backend publication",
                        expected="the exact current record and selection",
                        actual="reset or lifecycle mutation during publication",
                        remediation=(
                            "Publisher callbacks must only mutate their bounded "
                            "adapter cache."
                        ),
                    )
            except BaseException:
                if (
                    publication_epoch == self._lifecycle_epoch
                    and self._records.get(current.record_id) is current
                    and self._selections.get(stored.target) is stored
                ):
                    self._records[previous_record.record_id] = previous_record
                    lease = self._legacy_leases.get(previous_record.record_id)
                    if lease is not None:
                        lease._refresh(
                            previous_record,
                            factory=_LEGACY_LEASE_FACTORY,
                        )
                    for target, item in tuple(self._selections.items()):
                        if item.record_id == previous_record.record_id:
                            self._selections.pop(target, None)
                    self._selections.update(previous_decisions)
                raise
            finally:
                self._legacy_publication_owner = None
            return stored, current, publication

    def activate_materialized_legacy(
        self,
        lease: LegacyRecordLease,
        target: Any,
    ) -> BackendPluginRecord:
        """Activate the leased pair only for its exact current target binding."""
        target_name = self._legacy_target_name(target)
        with self._condition:
            self._ensure_not_resetting("activate_materialized_legacy")
            current = self._validate_legacy_lease(
                lease, "activate_materialized_legacy"
            )
            decision = self._selections.get(target_name)
            if decision is None or decision.record_id != current.record_id:
                raise BackendPluginLifecycleError(
                    "Legacy backend must own the target selection before activation",
                    entry_point=current.entry_point_name,
                    field="selection",
                    expected=f"{target_name} -> {current.record_id}",
                    actual=(
                        "<missing>"
                        if decision is None
                        else f"{target_name} -> {decision.record_id}"
                    ),
                    remediation=(
                        "Call select_materialized_legacy() with the same lease "
                        "and target before activation."
                    ),
                )
            return self.activate(current.record_id)

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
        target_name = _target_name(target)
        required = _capability_names(
            kernel_required_capabilities,
            "kernel_required_capabilities",
        )
        with self._lock:
            self._ensure_not_resetting("select")
            records = self.validate()
            # Fatal conflicts must reject the involved records before the
            # static selection algorithm raises (selection itself is not
            # changed); this also covers compiler/runtime entry paths that
            # resolve through select().
            self._reject_all_fatal_conflicts(records)
            try:
                decision = select_backend(
                    records,
                    target=target_name,
                    kernel_required_capabilities=required,
                    core_provided_capabilities=self._core_capabilities,
                    explicit_selector=explicit_selector,
                    environment=(
                        os.environ if environment is None else environment
                    ),
                )
            except BackendPluginNoCandidateError:
                # NoCandidate is an authority to enter the weaker Legacy
                # bridge.  It is only safe when discovery itself was complete;
                # a distribution-level metadata failure has no record/targets
                # with which the pure selector could prove irrelevance.
                if self._registry_errors:
                    raise self._registry_errors[0]
                raise

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

        # register() may run entry-point, attribute, and initialize code.
        selected = self.register(
            decision.record_id,
            _expected_epoch=selection_epoch,
            _expected_record=decision.record,
        )

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
            previous_decision = self._selections.get(decision.target)
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
        if type(target) is not str or not target:
            raise BackendPluginSelectionError(
                "Selection lookup target must be an inert string",
                field="target",
                expected="an exact non-empty built-in string",
                actual=_stable_type_name(target),
                remediation="Normalize the target before Registry lookup.",
            )
        with self._lock:
            return self._selections.get(target)

    def release_selections_if_current(
        self,
        decisions: Iterable[SelectionDecision],
    ) -> Tuple[BackendPluginRecord, ...]:
        """Atomically release an exact batch of conditional selections.

        Every stored decision and record identity is validated before any
        mutation.  Consequently a reset, activation, or newer selection makes
        the entire release fail closed instead of partially removing state.
        Records selected for multiple targets are replaced only once.
        """
        try:
            decision_tuple = tuple(decisions)
        except TypeError as exc:
            raise TypeError("selection release decisions must be iterable") from exc
        if any(type(decision) is not SelectionDecision for decision in decision_tuple):
            raise TypeError("selection release requires SelectionDecision values")
        for decision in decision_tuple:
            for field_name, value in (
                ("selection.target", decision.target),
                ("selection.record_id", decision.record_id),
            ):
                if type(value) is not str or not value:
                    raise BackendPluginLifecycleError(
                        "Selection release identity is not an inert string",
                        field=field_name,
                        expected="an exact non-empty built-in string",
                        actual=_stable_type_name(value),
                        remediation="Use an unmodified Registry-issued decision.",
                    )
        targets = tuple(decision.target for decision in decision_tuple)
        if len(targets) != len(set(targets)):
            raise ValueError("selection release targets must be unique")
        if not decision_tuple:
            return ()

        with self._condition:
            self._ensure_not_resetting("release_selections_if_current")
            records: Dict[str, BackendPluginRecord] = {}
            released_targets: Dict[str, set[str]] = {}
            for decision in decision_tuple:
                current_decision = self._selections.get(decision.target)
                if (
                    current_decision is None
                    or current_decision.ownership_token
                    is not decision.ownership_token
                ):
                    raise BackendPluginLifecycleError(
                        "Conditional backend selection is stale or has been replaced",
                        plugin_id=decision.plugin_id,
                        entry_point=decision.entry_point_name,
                        field="selection",
                        expected=(
                            f"current decision {decision.target} -> "
                            f"{decision.record_id}"
                        ),
                        actual=(
                            "<missing>"
                            if current_decision is None
                            else (
                                f"current decision {current_decision.target} -> "
                                f"{current_decision.record_id}"
                            )
                        ),
                        remediation=(
                            "Do not release decisions after reset or replacement; "
                            "leave newer selections untouched."
                        ),
                    )
                record = self._records.get(decision.record_id)
                if record is None or current_decision.record is not record:
                    raise BackendPluginLifecycleError(
                        "Conditional backend selection record is stale",
                        plugin_id=decision.plugin_id,
                        entry_point=decision.entry_point_name,
                        field="selection.record",
                        expected="the exact current Registry record",
                        actual="missing or replaced record identity",
                        remediation=(
                            "Do not release an old decision after another "
                            "lifecycle operation replaced its record snapshot."
                        ),
                    )
                if record.state is PluginLifecycleState.ACTIVE:
                    raise BackendPluginLifecycleError(
                        "An ACTIVE backend selection cannot be conditionally released",
                        plugin_id=record.plugin_id,
                        entry_point=record.entry_point_name,
                        field="state",
                        expected=PluginLifecycleState.SELECTED.value,
                        actual=record.state.value,
                        remediation="Reset the active runtime driver before release.",
                    )
                if record.state not in {
                    PluginLifecycleState.SELECTED,
                    PluginLifecycleState.REGISTERED,
                }:
                    raise BackendPluginLifecycleError(
                        "Conditional selection record is not releasable",
                        plugin_id=record.plugin_id,
                        entry_point=record.entry_point_name,
                        field="state",
                        expected="selected or registered",
                        actual=record.state.value,
                        remediation="Retry from a fresh Registry selection.",
                    )
                records[record.record_id] = record
                released_targets.setdefault(record.record_id, set()).add(
                    decision.target
                )

            # All authority checks succeeded.  Remove the batch before record
            # replacement so _replace() updates only selections that remain.
            for decision in decision_tuple:
                del self._selections[decision.target]

            released_records = []
            for record_id in sorted(records):
                record = records[record_id]
                removed = released_targets[record_id]
                remaining_targets = tuple(
                    target
                    for target in record.selected_targets
                    if target not in removed
                )
                state = record.state
                if (
                    state is PluginLifecycleState.SELECTED
                    and not remaining_targets
                ):
                    self._transition(record, PluginLifecycleState.REGISTERED)
                    state = PluginLifecycleState.REGISTERED
                released_records.append(
                    self._replace(
                        replace(
                            record,
                            state=state,
                            selected_targets=remaining_targets,
                        )
                    )
                )
            self._generation += 1
            return tuple(released_records)

    def release_selection_if_current(
        self,
        decision: SelectionDecision,
    ) -> BackendPluginRecord:
        """Singular convenience wrapper for conditional selection release."""
        return self.release_selections_if_current((decision,))[0]

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
            record = self._validate_record(record)
            if record.state is PluginLifecycleState.REJECTED:
                if record.error is not None:
                    raise record.error
                raise BackendPluginLifecycleError(
                    "Rejected backend plugin cannot be activated",
                    plugin_id=record.plugin_id,
                    entry_point=record.entry_point_name,
                    field="state",
                    expected=PluginLifecycleState.SELECTED.value,
                    actual=record.state.value,
                    remediation="Resolve the recorded validation failure first.",
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
                f"Backend plugin diagnostics() failed: {exc}",
                expected="a Mapping result from a synchronous hook",
                actual=f"<error: {exc}>",
                remediation="Fix diagnostics() so it is synchronous and read-only.",
            )
            return {"error": error.to_dict()}
        if inspect.isawaitable(value):
            _close_unawaited(value)
            error = self._hook_error(
                record,
                "diagnostics",
                "Backend plugin diagnostics() returned an awaitable",
                expected="Mapping from a synchronous diagnostics() hook",
                actual=_stable_type_name(value),
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
                f"Unable to snapshot backend diagnostics Mapping: {exc}",
                expected="a Mapping that can be copied with dict()",
                actual=f"<error: {exc}>",
                remediation="Return a stable Mapping from diagnostics().",
            )
            return {"error": error.to_dict()}

    def diagnostics(self, identifier: Optional[str] = None) -> Any:
        """Return static state and optional loaded-plugin diagnostics."""
        thread_id = threading.get_ident()
        operations: list[Tuple[BackendPluginRecord, object]] = []
        try:
            with self._condition:
                self._ensure_not_resetting("diagnostics")
                self.discover()
                records = (
                    (self._resolve(identifier),)
                    if identifier is not None
                    else tuple(
                        sorted(
                            self._records.values(),
                            key=lambda item: item.registry_key,
                        )
                    )
                )
                prepared = []
                for record in records:
                    result = record.to_dict()
                    result["plugin_diagnostics"] = None
                    if record.plugin_object is not None:
                        registration = self._registering.get(record.record_id)
                        operation = self._diagnosing.get(record.record_id)
                        if registration is not None:
                            error = BackendPluginLifecycleError(
                                "Backend plugin diagnostics are unavailable "
                                "during registration",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="diagnostics",
                                expected=(
                                    "registration to complete before diagnostics"
                                ),
                                actual="initialize in progress",
                                remediation=(
                                    "Retry diagnostics after register() completes; "
                                    "partially initialized objects are never probed."
                                ),
                            )
                            result["plugin_diagnostics"] = {
                                "error": error.to_dict()
                            }
                        elif record.state not in {
                            PluginLifecycleState.REGISTERED,
                            PluginLifecycleState.SELECTED,
                            PluginLifecycleState.ACTIVE,
                        }:
                            # Merely importing a plugin does not authorize its
                            # optional runtime hook.  Keep the static record
                            # visible without probing a half-initialized or
                            # rejected object.
                            pass
                        elif operation is not None:
                            error = BackendPluginLifecycleError(
                                "Recursive backend plugin diagnostics is not "
                                "allowed",
                                plugin_id=record.plugin_id,
                                entry_point=record.entry_point_name,
                                field="diagnostics",
                                expected="one non-reentrant diagnostics operation",
                                actual=(
                                    "recursive diagnostics"
                                    if operation[1] == thread_id
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
                            operations.append((record, token))
                            self._diagnosing[record.record_id] = (
                                self._lifecycle_epoch,
                                thread_id,
                                token,
                            )
                    prepared.append((record, result))

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
                        self._records.values()
                    ).to_dict(),
                    "selections": {
                        target: decision.to_dict()
                        for target, decision in sorted(
                            self._selections.items()
                        )
                    },
                }

            by_record_id = {
                record.record_id: result for record, result in prepared
            }
            for record, token in operations:
                by_record_id[record.record_id][
                    "plugin_diagnostics"
                ] = self._diagnostics_value(record)
                with self._condition:
                    self._finish_operation(
                        self._diagnosing,
                        record.record_id,
                        token,
                    )

            results = [result for _record, result in prepared]
            if identifier is not None:
                return results[0]
            registry_payload["plugins"] = results
            return registry_payload
        finally:
            # Preparation and hook execution both run after tokens may exist.
            # Any programming error or BaseException must release every token
            # so reset and later diagnostics cannot block forever.
            with self._condition:
                for record, token in operations:
                    self._finish_operation(
                        self._diagnosing,
                        record.record_id,
                        token,
                    )

    def reset(self) -> Tuple[BackendPluginError, ...]:
        """Best-effort shutdown and clear state; Python modules stay imported."""
        cleanup_errors: list[Tuple[int, BackendPluginError]] = []
        reset_hooks: Tuple[Callable[[], None], ...] = ()
        shutdown_records: Tuple[BackendPluginRecord, ...] = ()
        reset_started = False
        fatal_reset_error: Optional[BaseException] = None
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
                        actual="current thread owns an in-progress plugin operation",
                        remediation=(
                            "Return from the plugin hook before resetting the Registry."
                        ),
                    )
                if self._legacy_publication_owner == thread_id:
                    raise BackendPluginLifecycleError(
                        "Cannot reset the Registry re-entrantly from a Legacy "
                        "publication callback",
                        field="reset",
                        expected="reset outside adapter publication",
                        actual="current thread owns Legacy publication",
                        remediation=(
                            "Return from the bounded publication callback before "
                            "resetting the Registry."
                        ),
                    )
                reset_started = True
                self._resetting = True
                self._lifecycle_epoch += 1
                self._generation += 1
                self._reset_operation_errors = []

                # Adapter-owned publication caches are invalidated at the
                # same commit boundary as the lifecycle epoch.  These trusted
                # callbacks are deliberately constrained to bounded memory
                # mutation and run before Core ownership is cleared, so a
                # stale public mapping is never observable with no owning
                # Registry record.
                for callback in tuple(self._reset_invalidation_hooks):
                    try:
                        callback()
                    except BackendPluginError as exc:
                        cleanup_errors.append((0, exc))
                    except Exception as exc:
                        cleanup_errors.append(
                            (
                                0,
                                BackendPluginLifecycleError(
                                    f"Backend reset invalidation hook failed: {exc}",
                                    field="reset_invalidation_hook",
                                    expected="successful in-memory cache invalidation",
                                    actual=f"<error: {exc}>",
                                    remediation=(
                                        "Fix the adapter invalidation hook so it "
                                        "does not call plugin or Registry code."
                                    ),
                                ),
                            )
                        )
                    except BaseException as exc:
                        # SystemExit/KeyboardInterrupt must not strand a half
                        # reset Registry after the lifecycle epoch advanced.
                        # Preserve the first fatal signal, finish the atomic
                        # Core/cache invalidation and cleanup, then re-raise it.
                        if fatal_reset_error is None:
                            fatal_reset_error = exc

                # Invalidate public state first. Owners complete against the
                # advanced epoch and can therefore never publish stale results.
                self._records.clear()
                self._registry_errors = ()
                self._environment = None
                self._environment_error = None
                self._discovered = False
                self._selections.clear()
                self._legacy_leases.clear()
                self._legacy_lease_tokens.clear()
                self._condition.notify_all()

                while self._loading or self._registering or self._diagnosing:
                    self._condition.wait()
                self._waiting_for.clear()

                shutdown_records = tuple(reversed(self._cleanup_stack))
                self._cleanup_stack.clear()
                reset_hooks = tuple(self._reset_hooks)
                cleanup_errors.extend(
                    (0, error) for error in self._reset_operation_errors
                )
                self._reset_operation_errors = []

            # W9 hooks may hold their own lazy-cache locks.  Run them after
            # releasing the Registry lock to avoid Registry/LazyProxy lock
            # inversion; _resetting still rejects lifecycle re-entry.
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
                            f"Backend reset hook failed: {exc}",
                            field="reset_hook",
                            expected="successful cache cleanup",
                            actual=f"<error: {exc}>",
                            remediation=(
                                "Fix the W9 adapter reset hook so it only "
                                "clears local compiler/driver caches."
                                ),
                            ),
                        )
                    )
                except BaseException as exc:
                    if fatal_reset_error is None:
                        fatal_reset_error = exc

            # User shutdown hooks may acquire plugin-owned locks or attempt
            # Registry re-entry.  They also run outside the Registry lock;
            # _resetting makes any lifecycle re-entry fail immediately.
            for record in shutdown_records:
                try:
                    _called, error = self._call_shutdown(record)
                except BaseException as exc:
                    if fatal_reset_error is None:
                        fatal_reset_error = exc
                    continue
                if error is not None:
                    cleanup_errors.append((1, error))

            result = tuple(
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
            if fatal_reset_error is not None:
                raise fatal_reset_error
            return result
        finally:
            if reset_started:
                with self._condition:
                    self._resetting = False
                    self._condition.notify_all()


backend_plugin_registry = BackendPluginRegistry()


def get_backend_plugin_registry() -> BackendPluginRegistry:
    """Return the process-wide Registry without triggering discovery."""
    return backend_plugin_registry
