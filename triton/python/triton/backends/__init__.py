import os
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Optional, Tuple

from .driver import DriverBase
from .compiler import BaseBackend
from .interface import (
    TRITON_RUNTIME_INTERFACE_CONTRACT,
    _type_mro,
    validate_triton_runtime_pair,
)


@dataclass(frozen=True)
class Backend:
    compiler: Optional[type] = None
    driver: Optional[type] = None
    record_id: Optional[str] = None
    plugin_id: Optional[str] = None
    entry_point_name: Optional[str] = None


@dataclass(frozen=True)
class _DriverBackendResolution:
    manifest_candidates: Tuple[Backend, ...]
    manifest_decisions: Tuple[Any, ...] = ()
    manual_candidates: Tuple[Backend, ...] = ()
    selected_legacy_decisions: Tuple[Any, ...] = ()
    selected_legacy_backends: Tuple[Backend, ...] = ()
    requested_legacy_record_ids: Tuple[str, ...] = ()
    speculative_decisions: Tuple[Any, ...] = ()
    speculative_backends: Tuple[Backend, ...] = ()
    speculative_publications: Tuple[
        Tuple[Backend, Optional[Backend]], ...
    ] = ()
    preexisting_decisions: Tuple[Any, ...] = ()
    forced_selection: bool = False
    manifest_error: Optional[BaseException] = None
    cache_epoch: int = 0

    @property
    def allows_legacy_fallback(self) -> bool:
        return not self.forced_selection and not self.preexisting_decisions

    @property
    def candidates(self) -> Tuple[Backend, ...]:
        """Compatibility view used by public ``get_driver_backends``."""
        if self.manifest_candidates:
            return self.manifest_candidates
        if self.selected_legacy_backends:
            return self.selected_legacy_backends
        return self.manual_candidates


@dataclass(frozen=True)
class _CompilerBackendResolution:
    backend: Backend
    decision: Any = None
    speculative_decision: Any = None
    speculative_backend: Optional[Backend] = None
    previous_backend: Optional[Backend] = None


@dataclass(frozen=True)
class _LegacyCandidate:
    backend: Backend
    lease: Any = None

    @property
    def record(self):
        return None if self.lease is None else self.lease.record

    @property
    def identity(self) -> str:
        record = self.record
        if record is not None:
            return record.record_id
        return "manual:" + str(self.backend.entry_point_name or "<unnamed>")


# Keep this public object stable: existing code may import it by reference or
# add a manually managed Legacy backend.  Registry-backed entries are added
# only after selection.
backends: Dict[str, Backend] = {}

# Registry reset and public-map publication cross two independently locked
# components.  Keep this lock local and hold it only for small in-memory cache
# operations; plugin callbacks and Registry calls must always run without it.
_backend_cache_lock = threading.RLock()
_backend_cache_epoch = 0
_legacy_target_owners: Dict[str, Backend] = {}
_LEGACY_COMPILER_FALLBACK_AUTHORITY = object()
_LEGACY_RUNTIME_FALLBACK_AUTHORITY = object()


def _registry_api():
    # Lazy import avoids making the Triton package depend on plugin loading at
    # module import time.  triton_anchor.backends itself performs no ep.load().
    import triton_anchor.backends as api

    return api


def _registry():
    api = _registry_api()
    registry = api.get_backend_plugin_registry()
    try:
        registry.register_runtime_pair_validator(
            TRITON_RUNTIME_INTERFACE_CONTRACT,
            validate_triton_runtime_pair,
        )
    except api.BackendPluginLifecycleError as error:
        if (
            error.field != "runtime_pair_validator"
            or error.actual not in {"registered", "selected", "active"}
        ):
            raise
        # A forged/unsupported record may already claim a published state in
        # a newly attached Registry.  Replay the version-independent manifest
        # guards first; they purge such records without loading plugin code.
        # A supported Python record remains published, so the retry still
        # fails closed and requires reset before Triton's contract can attach.
        registry.validate()
        registry.register_runtime_pair_validator(
            TRITON_RUNTIME_INTERFACE_CONTRACT,
            validate_triton_runtime_pair,
        )
    return registry


def _enum_value(value):
    return getattr(value, "value", value)


def _is_selected_record(record) -> bool:
    return (
        _enum_value(record.state) in {"selected", "active"}
        and bool(record.selected_targets)
    )


def _backend_matches_record(backend: Any, record: Any) -> bool:
    """Match a cache value to the complete immutable Registry owner."""
    return (
        type(backend) is Backend
        and backend.record_id == record.record_id
        and backend.compiler is record.compiler_cls
        and backend.driver is record.driver_cls
        and backend.plugin_id == record.plugin_id
        and backend.entry_point_name == record.entry_point_name
    )


def _same_backend_pair(left: Any, right: Any) -> bool:
    return (
        type(left) is Backend
        and type(right) is Backend
        and left.compiler is right.compiler
        and left.driver is right.driver
        and left.record_id == right.record_id
        and left.entry_point_name == right.entry_point_name
    )


def _target_owner_conflict(target_name: str, expected: Any, actual: Any):
    return _registry_api().BackendPluginConflictError(
        f"Target '{target_name}' already belongs to another Legacy pair",
        plugin_id=getattr(expected, "plugin_id", None),
        entry_point=getattr(expected, "entry_point_name", None),
        conflict_kind="selected_target",
        claim=target_name,
        field="compiler_cls,driver_cls",
        expected=(
            getattr(actual, "record_id", None)
            or getattr(actual, "entry_point_name", None)
            or "existing Legacy pair"
        ),
        actual=(
            getattr(expected, "record_id", None)
            or getattr(expected, "entry_point_name", None)
            or "new Legacy pair"
        ),
        remediation=(
            "Reset backend state or use the same Legacy compiler/driver pair "
            "for both compiler-first and runtime-first consumption."
        ),
    )


def _prune_registry_backends(registry) -> None:
    stale = []
    with _backend_cache_lock:
        cached = tuple(backends.items())
    for name, backend in cached:
        # Foreign/manual values are validated by _manual_backends() after the
        # cache lock is released.  Never execute their attribute hooks while
        # pruning Core-owned exact Backend publications.
        if type(name) is not str or type(backend) is not Backend:
            continue
        record_id = backend.record_id
        if record_id is None:
            continue
        try:
            record = registry.inspect(record_id)
        except Exception:
            stale.append((name, backend))
            continue
        if not _is_selected_record(record) or not _backend_matches_record(
            backend, record
        ):
            stale.append((name, backend))
    with _backend_cache_lock:
        for name, expected_backend in stale:
            if backends.get(name) is expected_backend:
                backends.pop(name, None)


def _drop_cached_backend(expected_backend: Optional[Backend]) -> None:
    """Remove only the exact publication owned by one failed operation."""
    if expected_backend is None:
        return
    with _backend_cache_lock:
        for name, backend in tuple(backends.items()):
            if backend is expected_backend:
                backends.pop(name, None)


def _cached_backend_for_record(record) -> Optional[Backend]:
    with _backend_cache_lock:
        candidate = backends.get(record.entry_point_name)
        if _backend_matches_record(candidate, record):
            return candidate
    return None


def _unpublish_legacy_record(record) -> None:
    """Remove only adapter state owned by one exact Legacy runtime pair."""
    expected = Backend(
        compiler=record.compiler_cls,
        driver=record.driver_cls,
        record_id=record.record_id,
        plugin_id=record.plugin_id,
        entry_point_name=record.entry_point_name,
    )
    with _backend_cache_lock:
        for name, backend in tuple(backends.items()):
            if _backend_matches_record(backend, record):
                backends.pop(name, None)
        for target, owner in tuple(_legacy_target_owners.items()):
            if _same_backend_pair(owner, expected):
                _legacy_target_owners.pop(target, None)


def _cache_epoch() -> int:
    with _backend_cache_lock:
        return _backend_cache_epoch


def _stale_legacy_operation(record, expected_epoch):
    return _registry_api().BackendPluginLifecycleError(
        "Legacy backend consumption was invalidated by Registry reset",
        plugin_id=record.plugin_id,
        entry_point=record.entry_point_name,
        field="registry reset",
        expected=str(expected_epoch),
        actual=str(_cache_epoch()),
        remediation=(
            "Retry backend resolution after Registry reset completes."
        ),
    )


def _legacy_mapping_conflict(record):
    return _registry_api().BackendPluginConflictError(
        "Legacy backend public mapping key is already owned: "
        + record.entry_point_name,
        plugin_id=record.plugin_id,
        entry_point=record.entry_point_name,
        field="backends",
        expected=(
            "an unused entry-point name or the same Registry record"
        ),
        actual=record.entry_point_name,
        related_record_ids=(record.record_id,),
        remediation=(
            "Remove the manually managed mapping or migrate the Legacy "
            "backend to a unique Manifest identity."
        ),
    )


def _ensure_legacy_mapping_available(record, expected_epoch) -> None:
    with _backend_cache_lock:
        if _backend_cache_epoch != expected_epoch:
            raise _stale_legacy_operation(record, expected_epoch)
        current = backends.get(record.entry_point_name)
        if current is not None and not _backend_matches_record(current, record):
            raise _legacy_mapping_conflict(record)


def _publish_legacy_mapping(
    decision,
    record,
    expected_epoch,
) -> Backend:
    """Publish one Core-owned Legacy pair at Core's atomic commit point."""
    backend = Backend(
        compiler=record.compiler_cls,
        driver=record.driver_cls,
        record_id=record.record_id,
        plugin_id=record.plugin_id,
        entry_point_name=record.entry_point_name,
    )
    with _backend_cache_lock:
        if _backend_cache_epoch != expected_epoch:
            raise _stale_legacy_operation(record, expected_epoch)
        owner = _legacy_target_owners.get(decision.target)
        if owner is not None and not _same_backend_pair(owner, backend):
            raise _target_owner_conflict(decision.target, backend, owner)
        current = backends.get(record.entry_point_name)
        if current is not None and not _backend_matches_record(current, record):
            raise _legacy_mapping_conflict(record)
        # A fresh immutable Backend object is the publication ownership
        # token.  Even an idempotent re-publication replaces the old token so
        # cleanup from an earlier operation cannot delete a newer mapping.
        backends[record.entry_point_name] = backend
    return backend


def _backend_for_decision(decision) -> Backend:
    """Resolve an exact Registry decision without publishing adapter state."""
    registry = _registry()
    try:
        record = registry.inspect(decision.record_id)
    except Exception as exc:
        raise _registry_api().BackendPluginLifecycleError(
            "Backend selection was invalidated before consumption",
            plugin_id=decision.plugin_id,
            entry_point=decision.entry_point_name,
            field="registry generation",
            expected="the exact selected record",
            actual="selection record unavailable",
            remediation=(
                "Retry backend resolution after the concurrent Registry "
                "reset or selection change completes."
            )
        ) from exc
    try:
        # Cached decisions are operational inputs too.  Registry validation
        # rechecks even selected/active records, rejects hand-crafted isolation,
        # and removes the stale selection before any runtime class is exposed.
        record = registry.validate(record.record_id)
    except Exception:
        _prune_registry_backends(registry)
        raise
    if (
        not _is_selected_record(record)
        or decision.target not in record.selected_targets
    ):
        raise _registry_api().BackendPluginLifecycleError(
            "Backend selection changed before consumption",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="selected_targets",
            expected=decision.target,
            actual=", ".join(record.selected_targets) or "<none>",
            remediation=(
                "Retry backend resolution using the current Registry "
                "selection."
            )
        )
    current_decision = registry.get_selection(decision.target)
    if (
        current_decision is None
        or current_decision.ownership_token is not decision.ownership_token
        or current_decision.record is not record
    ):
        raise _registry_api().BackendPluginLifecycleError(
            "Backend selection changed before consumption",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="selection",
            expected=f"{decision.target} -> {decision.record_id}",
            actual=(
                "<missing>"
                if current_decision is None
                else f"{current_decision.target} -> {current_decision.record_id}"
            ),
            remediation="Retry using the exact current Registry selection.",
        )
    return Backend(
        compiler=record.compiler_cls,
        driver=record.driver_cls,
        record_id=record.record_id,
        plugin_id=record.plugin_id,
        entry_point_name=record.entry_point_name,
    )


def _publish_selected_mapping(decision, record, expected_epoch: int) -> Backend:
    """Bounded Registry -> adapter publication callback for any selected pair."""
    backend = Backend(
        compiler=record.compiler_cls,
        driver=record.driver_cls,
        record_id=record.record_id,
        plugin_id=record.plugin_id,
        entry_point_name=record.entry_point_name,
    )
    with _backend_cache_lock:
        if _backend_cache_epoch != expected_epoch:
            raise _stale_legacy_operation(record, expected_epoch)
        owner = _legacy_target_owners.get(decision.target)
        if owner is not None and not _same_backend_pair(owner, backend):
            raise _target_owner_conflict(decision.target, backend, owner)
        current = backends.get(record.entry_point_name)
        if current is not None and not _backend_matches_record(current, record):
            if _enum_value(record.source) == "legacy":
                raise _legacy_mapping_conflict(record)
            raise _registry_api().BackendPluginConflictError(
                "Backend public mapping key is already owned: "
                + record.entry_point_name,
                plugin_id=record.plugin_id,
                entry_point=record.entry_point_name,
                field="backends",
                expected="an unused entry-point name",
                actual=record.entry_point_name,
                related_record_ids=(record.record_id,),
                remediation=(
                    "Remove the manually managed mapping or choose a unique "
                    "backend entry-point name."
                ),
            )
        backends[record.entry_point_name] = backend
    return backend


def _cache_decision(decision) -> Backend:
    """Atomically publish one exact decision into Triton's public mapping."""
    registry = _registry()
    _backend_for_decision(decision)
    expected_epoch = _cache_epoch()
    _stored, _record, publication = registry.commit_selection_if_current(
        decision,
        publisher=lambda current, record: _publish_selected_mapping(
            current, record, expected_epoch
        ),
    )
    return publication


def _selection_error(message, *, field, expected, actual, remediation):
    return _registry_api().BackendPluginSelectionError(
        message,
        field=field,
        expected=expected,
        actual=actual,
        remediation=remediation,
    )


def _selector_record(records, selector):
    matches = []
    for record in records:
        keys = {record.record_id, record.registry_key}
        if record.manifest is not None and record.plugin_id is not None:
            keys.add(record.plugin_id)
        if selector in keys:
            matches.append(record)
    if not matches:
        raise _selection_error(
            f"No backend plugin matches selector '{selector}'",
            field="backend_selector",
            expected="a Manifest plugin_id/record key or a Legacy record key",
            actual=selector,
            remediation=(
                "Use a selector shown by BackendPluginRegistry diagnostics."
            ),
        )
    if len(matches) != 1:
        record_ids = tuple(sorted(record.record_id for record in matches))
        raise _selection_error(
            f"Backend selector '{selector}' is ambiguous: "
            + ", ".join(record_ids),
            field="backend_selector",
            expected="one backend record",
            actual=", ".join(record_ids),
            remediation="Select one backend by its unique record_id.",
        )
    return matches[0]


def _bootstrap_target(record) -> str:
    if record.manifest is not None:
        targets = tuple(sorted(record.manifest.targets))
        if targets:
            return targets[0]
    # Legacy records have no static target declaration.  This value is only a
    # selection cache key; W8 still requires an exact Legacy record selector.
    return record.entry_point_name


def _target_name(target) -> str:
    # Normalize before any adapter/Registry lock and retain only an exact
    # built-in string.  This prevents caller-controlled hash/equality hooks
    # from running inside selection or cache dictionaries.
    from triton_anchor.backends.selection import _target_name as normalize

    return normalize(target)


def _manual_backends() -> Tuple[Backend, ...]:
    result = []
    with _backend_cache_lock:
        cached = tuple(backends.items())
    for name, backend in cached:
        if type(name) is not str or not name or name != name.strip():
            raise _selection_error(
                "Manual backend mapping key is not a stable string",
                field="backends",
                expected="an exact non-empty string key",
                actual=type(name).__name__,
                remediation="Register manual backends under a stable string name.",
            )
        if type(backend) is not Backend:
            raise _selection_error(
                "Manual backend mapping value is not an immutable Backend",
                field="backends",
                expected="an exact triton.backends.Backend value",
                actual=type(backend).__name__,
                remediation=(
                    "Store an exact Backend(compiler=..., driver=...) value."
                ),
            )
        if backend.record_id is not None:
            continue
        entry_point_name = backend.entry_point_name
        if entry_point_name is None:
            entry_point_name = name
        elif type(entry_point_name) is not str or not entry_point_name:
            raise _selection_error(
                "Manual backend entry-point identity is invalid",
                field="entry_point_name",
                expected="an exact non-empty string",
                actual=type(entry_point_name).__name__,
                remediation="Use the public mapping key as the entry-point name.",
            )
        # Copy the exact frozen dataclass after the lock is released.  All
        # later F6 checks and callbacks consume this immutable snapshot rather
        # than re-reading a mutable/malicious mapping object.
        result.append(
            Backend(
                compiler=backend.compiler,
                driver=backend.driver,
                plugin_id=backend.plugin_id,
                entry_point_name=entry_point_name,
            )
        )
    return tuple(sorted(result, key=lambda item: item.entry_point_name))


def _reset_backend_cache() -> None:
    global _backend_cache_epoch
    with _backend_cache_lock:
        _backend_cache_epoch += 1
        _legacy_target_owners.clear()
        for name, backend in tuple(backends.items()):
            if (
                type(name) is str
                and type(backend) is Backend
                and backend.record_id is not None
            ):
                backends.pop(name, None)


def register_backend_reset_hook(callback) -> None:
    """Coordinate Registry reset with a Triton-side lazy cache."""
    _registry().register_reset_hook(callback)


def _discover_backends():
    """Discover static backend metadata without importing plugin code."""
    _registry().discover()
    return backends


def _legacy_capability_error(required: Iterable[str]):
    required = tuple(sorted(set(required)))
    return _registry_api().BackendPluginCapabilityError(
        required,
        scope="kernel",
        available_capabilities=(),
        missing_kernel_capabilities=required,
    )


def _legacy_callback_error(record, field, expected, error):
    return _registry_api().BackendPluginLifecycleError(
        f"Legacy backend callback '{field}' failed: {error}",
        plugin_id=record.plugin_id,
        entry_point=record.entry_point_name,
        field=field,
        expected=expected,
        actual=f"<error: {error}>",
        remediation=(
            "Fix the Legacy backend callback and migrate the backend to "
            "Manifest Schema 1.0 for static compatibility checks."
        ),
    )


def _call_legacy_registry(record, callback):
    """Translate Core's generic stale generation into Triton reset terms."""
    api = _registry_api()
    try:
        return callback()
    except api.BackendPluginLifecycleError as error:
        if error.field != "generation":
            raise
        raise api.BackendPluginLifecycleError(
            "Legacy backend operation was invalidated by Registry reset",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="registry lifecycle_epoch",
            expected=error.expected,
            actual=error.actual,
            remediation=(
                "Discard the stale Legacy lease and retry after Registry "
                "reset completes."
            ),
        ) from error


def _reject_legacy_failure(registry, lease, error) -> None:
    """Conditionally reject this lease, preferring a stale-lease failure."""
    try:
        _call_legacy_registry(
            lease.record,
            lambda: registry.reject_materialized_legacy(
                lease,
                error,
                unpublisher=_unpublish_legacy_record,
            ),
        )
    except Exception:
        raise
    raise error


def _materialize_legacy_leases(
    registry,
    *,
    record_ids: Optional[Tuple[str, ...]] = None,
) -> Tuple[Any, ...]:
    allowed = None if record_ids is None else set(record_ids)
    records = tuple(
        record
        for record in sorted(
            registry.list_legacy_records(), key=lambda item: item.record_id
        )
        if allowed is None or record.record_id in allowed
    )
    # Materialize every deterministic candidate before invoking any target or
    # hardware predicate.  A load/F6 failure is authoritative and cannot be
    # hidden by a later Legacy record that happens to return True.
    return tuple(
        _call_legacy_registry(
            record,
            lambda record=record: registry.materialize_legacy(
                record.record_id
            ),
        )
        for record in records
    )


def _legacy_candidate_from_lease(lease) -> _LegacyCandidate:
    record = lease.record
    return _LegacyCandidate(
        backend=Backend(
            compiler=record.compiler_cls,
            driver=record.driver_cls,
            record_id=record.record_id,
            plugin_id=record.plugin_id,
            entry_point_name=record.entry_point_name,
        ),
        lease=lease,
    )


def _validate_manual_runtime_pair(backend: Backend) -> None:
    """Apply the frozen F6 surface to one immutable manual pair snapshot."""
    invalid_fields = tuple(
        field
        for field, candidate in (
            ("compiler_cls", backend.compiler),
            ("driver_cls", backend.driver),
        )
        if not (
            (candidate_mro := _type_mro(candidate))
            and candidate_mro[0] is candidate
        )
    )
    if invalid_fields:
        raise _registry_api().BackendPluginInterfaceError(
            invalid_fields=invalid_fields,
            entry_point=backend.entry_point_name,
        )
    context = _registry_api().RuntimePairValidationContext(
        record_id="manual:" + str(backend.entry_point_name or "<unnamed>"),
        plugin_id=backend.plugin_id,
        entry_point=backend.entry_point_name or "<unnamed>",
        compiler_cls=backend.compiler,
        driver_cls=backend.driver,
    )
    issues = validate_triton_runtime_pair(context)
    if issues:
        raise _registry_api().BackendPluginInterfaceError(
            interface_issues=issues,
            plugin_id=backend.plugin_id,
            entry_point=backend.entry_point_name,
        )


def _legacy_candidates(
    registry,
    *,
    manual_candidates: Optional[Tuple[Backend, ...]] = None,
    selected_record_ids: Tuple[str, ...] = (),
) -> Tuple[_LegacyCandidate, ...]:
    allowed = set(selected_record_ids)
    legacy_records = tuple(
        record
        for record in sorted(
            registry.list_legacy_records(), key=lambda item: item.record_id
        )
        if not allowed or record.record_id in allowed
    )
    manual_backends = (
        ()
        if selected_record_ids
        else (
            _manual_backends()
            if manual_candidates is None
            else manual_candidates
        )
    )

    # Public identity collisions are fully knowable from metadata.  Reject
    # them before importing either Legacy entry point or running F6/plugin
    # callbacks, and never let a Registry record overwrite a manual mapping.
    registry_by_name = {
        record.entry_point_name: record for record in legacy_records
    }
    manual_by_name = {
        backend.entry_point_name: backend for backend in manual_backends
    }
    colliding_names = tuple(
        sorted(set(registry_by_name).intersection(manual_by_name))
    )
    if colliding_names:
        name = colliding_names[0]
        record = registry_by_name[name]
        raise _registry_api().BackendPluginConflictError(
            "Legacy backend public identity is ambiguous: " + name,
            entry_point=name,
            field="entry_point",
            expected="one Legacy candidate per public entry-point name",
            actual="manual mapping and Registry record",
            related_record_ids=(record.record_id, "manual:" + name),
            remediation=(
                "Remove the duplicate manual/entry-point registration or "
                "migrate each backend to a unique Manifest identity."
            ),
        )

    # Manual entries do not pass through Registry.register(), so apply the
    # identical frozen Triton 3.3 surface gate explicitly and before any
    # Legacy import or target/hardware callback.
    for backend in manual_backends:
        _validate_manual_runtime_pair(backend)

    leases = _materialize_legacy_leases(
        registry,
        record_ids=tuple(record.record_id for record in legacy_records),
    )
    registry_candidates = tuple(
        _legacy_candidate_from_lease(lease) for lease in leases
    )
    manual = (
        ()
        if selected_record_ids
        else tuple(
            _LegacyCandidate(backend=backend)
            for backend in manual_backends
        )
    )

    by_entry_point: Dict[str, list[_LegacyCandidate]] = {}
    for candidate in registry_candidates + manual:
        name = candidate.backend.entry_point_name or "<unnamed>"
        by_entry_point.setdefault(name, []).append(candidate)
    collisions = tuple(
        (name, tuple(items))
        for name, items in sorted(by_entry_point.items())
        if len(items) > 1
    )
    if collisions:
        name, items = collisions[0]
        raise _registry_api().BackendPluginConflictError(
            "Legacy backend public identity is ambiguous: " + name,
            field="entry_point",
            expected="one Legacy candidate per public entry-point name",
            actual=str(len(items)),
            related_record_ids=tuple(
                sorted(candidate.identity for candidate in items)
            ),
            remediation=(
                "Remove the duplicate manual/entry-point registration or "
                "migrate each backend to a unique Manifest identity."
            ),
        )
    return tuple(
        sorted(
            registry_candidates + manual,
            key=lambda candidate: candidate.identity,
        )
    )


def _normalize_required_capabilities(values: Iterable[str]) -> Tuple[str, ...]:
    # Reuse W8's authoritative public-input contract before any Legacy
    # materialization.  In particular, strings are not treated as iterables,
    # duplicates are not silently collapsed, and malformed values remain a
    # structured selection error with zero plugin imports.
    from triton_anchor.backends.selection import _capability_names

    return _capability_names(values, "kernel_required_capabilities")


def _legacy_fallback_denied(kind: str, record_ids: Iterable[str]):
    ids = tuple(sorted(record_ids))
    return _selection_error(
        f"Legacy {kind} fallback is not allowed while Manifest candidates exist",
        field="legacy_fallback",
        expected=f"no applicable Manifest {kind} candidate",
        actual=", ".join(ids) or "<unknown Manifest candidate>",
        remediation=(
            "Use the Manifest-governed backend, or remove/repair it before "
            "requesting the Legacy compatibility bridge."
        ),
    )


def _authorize_direct_compiler_legacy_fallback(
    registry,
    target,
    required: Tuple[str, ...],
) -> None:
    api = _registry_api()
    records = registry.validate()
    rejected = tuple(
        sorted(
            (record.record_id, record.error)
            for record in records
            if record.error is not None and _enum_value(record.source) != "legacy"
        )
    )
    if rejected:
        raise rejected[0][1]
    try:
        decision = api.select_backend(
            records,
            target=target,
            kernel_required_capabilities=required,
            core_provided_capabilities=registry._core_capabilities,
            environment={},
        )
    except api.BackendPluginNoCandidateError:
        return
    raise _legacy_fallback_denied("compiler", (decision.record_id,))


def _authorize_direct_runtime_legacy_fallback(registry) -> None:
    records = registry.validate()
    rejected = tuple(
        sorted(
            (record.record_id, record.error)
            for record in records
            if record.error is not None and _enum_value(record.source) != "legacy"
        )
    )
    if rejected:
        raise rejected[0][1]
    candidates = tuple(
        record.record_id
        for record in records
        if (
            record.manifest is not None
            and _enum_value(record.compatibility_status) == "compatible"
            and _enum_value(record.state)
            in {"validated", "loaded", "registered", "selected", "active"}
        )
    )
    if candidates:
        raise _legacy_fallback_denied("runtime", candidates)


def _candidate_callback_error(candidate, field, expected, error):
    record = candidate.record
    return _registry_api().BackendPluginLifecycleError(
        f"Legacy backend callback '{field}' failed: {error}",
        plugin_id=(record.plugin_id if record is not None else candidate.backend.plugin_id),
        entry_point=candidate.backend.entry_point_name,
        field=field,
        expected=expected,
        actual=f"<error: {error}>",
        remediation=(
            "Fix the Legacy backend callback and migrate the backend to "
            "Manifest Schema 1.0 for static compatibility checks."
        ),
    )


def _raise_candidate_failure(registry, candidate, error):
    if candidate.lease is not None:
        _reject_legacy_failure(registry, candidate.lease, error)
    raise error


def _claim_manual_legacy_target(
    registry,
    backend: Backend,
    target,
    expected_epoch: int,
) -> None:
    """Bind a manual Legacy pair so compiler/runtime cannot split sources."""
    target_name = _target_name(target)
    selected = registry.get_selection(target_name)
    if selected is not None:
        try:
            selected_record = registry.inspect(selected.record_id)
            actual = Backend(
                compiler=selected_record.compiler_cls,
                driver=selected_record.driver_cls,
                record_id=selected_record.record_id,
                plugin_id=selected_record.plugin_id,
                entry_point_name=selected_record.entry_point_name,
            )
        except Exception:
            actual = Backend(record_id=selected.record_id)
        raise _target_owner_conflict(target_name, backend, actual)

    with _backend_cache_lock:
        if _backend_cache_epoch != expected_epoch:
            raise _registry_api().BackendPluginLifecycleError(
                "Manual Legacy target ownership was invalidated by reset",
                entry_point=backend.entry_point_name,
                field="registry reset",
                expected=str(expected_epoch),
                actual=str(_backend_cache_epoch),
                remediation="Retry Legacy resolution after reset completes.",
            )
        current = _legacy_target_owners.get(target_name)
        if current is not None and not _same_backend_pair(current, backend):
            raise _target_owner_conflict(target_name, backend, current)
        _legacy_target_owners[target_name] = backend

    # Core selection commits check the same owner while publishing.  This
    # second read closes the opposite interleaving where Core selected just
    # before the manual owner acquired the adapter lock.
    selected = registry.get_selection(target_name)
    if selected is not None:
        with _backend_cache_lock:
            if _legacy_target_owners.get(target_name) is backend:
                _legacy_target_owners.pop(target_name, None)
        try:
            selected_record = registry.inspect(selected.record_id)
            actual = Backend(
                compiler=selected_record.compiler_cls,
                driver=selected_record.driver_cls,
                record_id=selected_record.record_id,
                plugin_id=selected_record.plugin_id,
                entry_point_name=selected_record.entry_point_name,
            )
        except Exception:
            actual = Backend(record_id=selected.record_id)
        raise _target_owner_conflict(target_name, backend, actual)

    # Final linearization check closes reset between the second Registry read
    # and return.  A stale operation never succeeds after its owner entry was
    # cleared by the reset invalidator.
    with _backend_cache_lock:
        if (
            _backend_cache_epoch != expected_epoch
            or _legacy_target_owners.get(target_name) is not backend
        ):
            if _legacy_target_owners.get(target_name) is backend:
                _legacy_target_owners.pop(target_name, None)
            raise _registry_api().BackendPluginLifecycleError(
                "Manual Legacy target ownership was invalidated by reset",
                entry_point=backend.entry_point_name,
                field="registry reset",
                expected=str(expected_epoch),
                actual=str(_backend_cache_epoch),
                remediation="Retry Legacy resolution after reset completes.",
            )


def _legacy_no_compiler_match(target, candidates):
    return _selection_error(
        "No Legacy backend compiler supports the requested target",
        field="compiler_cls.supports_target",
        expected="one compatible Legacy backend compiler",
        actual="0",
        remediation=(
            "Install a Manifest backend for this target or keep exactly one "
            "Legacy backend whose compiler supports it."
        ),
    )


def _legacy_compiler_conflict(matches):
    return _registry_api().BackendPluginConflictError(
        "Multiple Legacy backend compilers support the requested target: "
        + ", ".join(candidate.identity for candidate in matches),
        field="compiler_cls.supports_target",
        expected="one compatible Legacy backend compiler",
        actual=str(len(matches)),
        related_record_ids=tuple(candidate.identity for candidate in matches),
        remediation=(
            "Migrate the candidates to Manifest Schema 1.0 and assign "
            "deterministic target ownership and priority."
        ),
    )


def _resolve_legacy_compiler_for_target(
    target,
    *,
    kernel_required_capabilities: Iterable[str] = (),
    authority=None,
    selected_record_ids: Tuple[str, ...] = (),
):
    """Construct and conditionally publish one no-Manifest compiler pair."""
    required = _normalize_required_capabilities(kernel_required_capabilities)
    if required:
        # Legacy metadata cannot prove kernel capabilities.  Fail before even
        # materializing a record, including when a Legacy choice is cached.
        raise _legacy_capability_error(required)

    registry = _registry()
    if authority is not _LEGACY_COMPILER_FALLBACK_AUTHORITY:
        _authorize_direct_compiler_legacy_fallback(registry, target, required)
    expected_epoch = _cache_epoch()
    candidates = _legacy_candidates(
        registry,
        selected_record_ids=selected_record_ids,
    )
    matches = []
    for candidate in candidates:
        supports_target = getattr(candidate.backend.compiler, "supports_target", None)
        try:
            if not callable(supports_target):
                raise TypeError("supports_target is not callable")
            supported = bool(supports_target(target))
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, _registry_api().BackendPluginError)
                else _candidate_callback_error(
                    candidate,
                    "compiler_cls.supports_target",
                    "a successful boolean target predicate",
                    exc,
                )
            )
            _raise_candidate_failure(registry, candidate, error)
        if supported:
            matches.append(candidate)

    if not matches:
        raise _legacy_no_compiler_match(target, candidates)
    if len(matches) != 1:
        raise _legacy_compiler_conflict(matches)

    candidate = matches[0]
    record = candidate.record
    if record is not None:
        _ensure_legacy_mapping_available(record, expected_epoch)
    try:
        compiler = candidate.backend.compiler(target)
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, _registry_api().BackendPluginError)
            else _candidate_callback_error(
                candidate,
                "compiler_cls.__init__",
                "a successfully constructed compiler",
                exc,
            )
        )
        _raise_candidate_failure(registry, candidate, error)

    if candidate.lease is None:
        _claim_manual_legacy_target(
            registry,
            candidate.backend,
            target,
            expected_epoch,
        )
        return compiler

    # Constructor success is the last plugin-controlled step.  Only now may
    # the Registry selection and public Triton mapping become observable.
    _ensure_legacy_mapping_available(record, expected_epoch)
    _decision, _selected, _backend = _call_legacy_registry(
        record,
        lambda: registry.commit_materialized_legacy(
            target,
            lease=candidate.lease,
            publisher=lambda decision, selected: _publish_legacy_mapping(
                decision,
                selected,
                expected_epoch,
            ),
            kernel_required_capabilities=required,
        ),
    )
    return compiler


def resolve_legacy_compiler_for_target(
    target,
    *,
    kernel_required_capabilities: Iterable[str] = (),
):
    """Resolve Legacy only after independently proving Manifest absence."""
    return _resolve_legacy_compiler_for_target(
        target,
        kernel_required_capabilities=kernel_required_capabilities,
    )


def _legacy_no_active_driver():
    return _selection_error(
        "No Legacy backend driver is active",
        field="driver_cls.is_active",
        expected="one active Legacy backend driver",
        actual="0",
        remediation=(
            "Install a Manifest backend or configure exactly one Legacy "
            "driver to report active."
        ),
    )


def _legacy_active_driver_conflict(matches):
    return _registry_api().BackendPluginConflictError(
        "Multiple Legacy backend drivers are active: "
        + ", ".join(candidate.identity for candidate in matches),
        field="driver_cls.is_active",
        expected="one active Legacy backend driver",
        actual=str(len(matches)),
        related_record_ids=tuple(candidate.identity for candidate in matches),
        remediation=(
            "Migrate the candidates to Manifest Schema 1.0 or configure "
            "hardware so exactly one Legacy driver reports active."
        ),
    )


def _resolve_active_legacy_driver(
    *,
    authority=None,
    manual_candidates: Optional[Tuple[Backend, ...]] = None,
    selected_record_ids: Tuple[str, ...] = (),
):
    """Construct, pair-check, select, and activate one Legacy runtime."""
    registry = _registry()
    if authority is not _LEGACY_RUNTIME_FALLBACK_AUTHORITY:
        _authorize_direct_runtime_legacy_fallback(registry)
    expected_epoch = _cache_epoch()
    candidates = _legacy_candidates(
        registry,
        manual_candidates=manual_candidates,
        selected_record_ids=selected_record_ids,
    )
    matches = []
    for candidate in candidates:
        is_active = getattr(candidate.backend.driver, "is_active", None)
        try:
            if not callable(is_active):
                raise TypeError("is_active is not callable")
            active = bool(is_active())
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, _registry_api().BackendPluginError)
                else _candidate_callback_error(
                    candidate,
                    "driver_cls.is_active",
                    "a successful boolean active probe",
                    exc,
                )
            )
            _raise_candidate_failure(registry, candidate, error)
        if active:
            matches.append(candidate)

    if not matches:
        raise _legacy_no_active_driver()
    if len(matches) != 1:
        raise _legacy_active_driver_conflict(matches)

    candidate = matches[0]
    record = candidate.record
    if record is not None:
        _ensure_legacy_mapping_available(record, expected_epoch)
    try:
        driver = candidate.backend.driver()
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, _registry_api().BackendPluginError)
            else _candidate_callback_error(
                candidate,
                "driver_cls.__init__",
                "a successfully constructed driver",
                exc,
            )
        )
        _raise_candidate_failure(registry, candidate, error)

    try:
        get_current_target = getattr(driver, "get_current_target", None)
        if not callable(get_current_target):
            raise TypeError("get_current_target is not callable")
        target = get_current_target()
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, _registry_api().BackendPluginError)
            else _candidate_callback_error(
                candidate,
                "driver_cls.get_current_target",
                "a readable current target",
                exc,
            )
        )
        _raise_candidate_failure(registry, candidate, error)

    supports_target = getattr(candidate.backend.compiler, "supports_target", None)
    try:
        if not callable(supports_target):
            raise TypeError("supports_target is not callable")
        supported = bool(supports_target(target))
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, _registry_api().BackendPluginError)
            else _candidate_callback_error(
                candidate,
                "compiler_cls.supports_target",
                "a successful boolean target predicate",
                exc,
            )
        )
        _raise_candidate_failure(registry, candidate, error)
    if not supported:
        target_name = _target_name(target)
        raise _selection_error(
            "Active Legacy driver target is not supported by its paired "
            "compiler",
            field="compiler_cls.supports_target",
            expected="True for the active driver's current target",
            actual=str(target_name),
            remediation=(
                "Return a current target supported by the compiler from the "
                "same Legacy entry-point record."
            ),
        )

    if candidate.lease is None:
        _claim_manual_legacy_target(
            registry,
            candidate.backend,
            target,
            expected_epoch,
        )
        return driver

    _ensure_legacy_mapping_available(record, expected_epoch)
    _active = _call_legacy_registry(
        record,
        lambda: registry.select_and_activate_materialized_legacy(
            target,
            lease=candidate.lease,
            publisher=lambda decision, selected: _publish_legacy_mapping(
                decision,
                selected,
                expected_epoch,
            ),
        ),
    )
    return driver


def resolve_active_legacy_driver():
    """Resolve Legacy runtime only after proving Manifest absence."""
    return _resolve_active_legacy_driver()


def _get_backend_resolution(
    target,
    *,
    allow_manual_legacy: bool = True,
    publish: bool = True,
) -> _CompilerBackendResolution:
    """Return one compiler backend plus exact auto-publication provenance."""
    api = _registry_api()
    registry = _registry()
    target_name = _target_name(target)
    existing = (
        registry.get_selection(target_name)
        if isinstance(target_name, str) and target_name
        else None
    )
    environment_selector = os.environ.get(api.BACKEND_SELECTOR_ENV)
    if (
        existing is not None
        and (
            _enum_value(existing.record.state) == "active"
            or _enum_value(existing.method) == "python_explicit"
            or not environment_selector
            or existing.selector == environment_selector
        )
    ):
        backend = (
            _cache_decision(existing)
            if publish
            else _backend_for_decision(existing)
        )
        return _CompilerBackendResolution(
            backend,
            decision=existing,
        )

    selection_before = registry.get_selection(target_name)
    try:
        decision = registry.select(target)
    except api.BackendPluginNoCandidateError:
        records = registry.list()
        rejected_manifest = tuple(
            sorted(
                (record.record_id, record.error)
                for record in records
                if (
                    record.error is not None
                    and _enum_value(record.source) != "legacy"
                )
            )
        )
        if rejected_manifest:
            raise rejected_manifest[0][1]
        if not allow_manual_legacy:
            # ``make_backend`` owns the governed Legacy consumption path.  It
            # must combine manual and Registry Legacy candidates exactly once
            # and wrap every callback failure structurally, rather than probe
            # manual mappings here and then probe the winner a second time.
            raise
        # Installed no-Manifest entry points belong to the governed bridge.
        # Do not let an identically named manual mapping silently shadow the
        # record; the bridge will report the public-key conflict atomically.
        if any(_enum_value(record.source) == "legacy" for record in records):
            raise
        manual_backends = _manual_backends()
        for backend in manual_backends:
            _validate_manual_runtime_pair(backend)
        candidates = tuple(
            backend
            for backend in manual_backends
            if (
                callable(
                    getattr(backend.compiler, "supports_target", None)
                )
                and backend.compiler.supports_target(target)
            )
        )
        manifest_claims_target = any(
            record.manifest is not None
            and target_name in record.manifest.targets
            for record in records
        )
        candidate_entry_points = {
            backend.entry_point_name
            for backend in candidates
            if backend.entry_point_name is not None
        }
        unreadable_related = any(
            record.manifest is None
            and _enum_value(record.source) != "legacy"
            and record.entry_point_name in candidate_entry_points
            for record in records
        )
        if manifest_claims_target or unreadable_related:
            raise
        if len(candidates) == 1:
            return _CompilerBackendResolution(candidates[0])
        if len(candidates) > 1:
            raise _selection_error(
                "Manual Legacy backend selection is ambiguous",
                field="compiler_cls.supports_target",
                expected="one compatible manually registered backend",
                actual=str(len(candidates)),
                remediation=(
                    "Migrate the backend to a Manifest or keep only one "
                    "manual Legacy backend for this target."
                ),
            )
        raise
    except api.BackendPluginError:
        _prune_registry_backends(registry)
        raise
    try:
        backend = (
            _cache_decision(decision)
            if publish
            else _backend_for_decision(decision)
        )
    except BaseException:
        if selection_before is None:
            try:
                registry.release_selection_if_current(decision)
            except api.BackendPluginLifecycleError:
                pass
        _prune_registry_backends(registry)
        raise
    current = registry.get_selection(target_name)
    speculative = (
        current
        if selection_before is None
        and current is not None
        and current.ownership_token is decision.ownership_token
        else None
    )
    return _CompilerBackendResolution(
        backend,
        decision=decision,
        speculative_decision=(None if publish else speculative),
    )


def get_backend(target) -> Backend:
    """Return the Registry-selected compiler/driver pair for one target."""
    return _get_backend_resolution(target).backend


def _rollback_driver_resolution_build(
    registry,
    fresh_decisions: Iterable[Any],
) -> None:
    decisions = tuple(fresh_decisions)
    for decision in decisions:
        try:
            registry.release_selection_if_current(decision)
        except _registry_api().BackendPluginLifecycleError:
            # A newer selection/reset owns this target.  Conditional token
            # failure proves it is not this build's state and must be left.
            continue


def _get_driver_backend_resolution() -> _DriverBackendResolution:
    """Resolve Manifest/manual candidates and retain conditional provenance."""
    api = _registry_api()
    registry = _registry()
    cache_epoch = _cache_epoch()
    try:
        records = registry.validate()
    except Exception:
        _prune_registry_backends(registry)
        raise
    _prune_registry_backends(registry)
    environment = dict(os.environ)
    selector = environment.get(api.BACKEND_SELECTOR_ENV)
    if selector == "":
        selector = None

    cached = []
    for record in records:
        for target in record.selected_targets:
            decision = registry.get_selection(target)
            if decision is not None:
                cached.append(decision)
    cached = list(
        {
            decision.target: decision
            for decision in sorted(cached, key=lambda item: item.target)
        }.values()
    )
    python_explicit = tuple(
        decision
        for decision in cached
        if (
            _enum_value(decision.method) == "python_explicit"
            and not decision.is_legacy
        )
    )
    cached_legacy = tuple(
        decision for decision in cached if bool(decision.is_legacy)
    )

    decisions = list(python_explicit)
    requested_legacy_record_ids = [
        decision.record_id for decision in cached_legacy
    ]
    preexisting_manifest = [
        decision for decision in cached if not decision.is_legacy
    ]
    fresh_by_target = {}

    def remember_fresh(target, selected):
        fresh_by_target[target] = selected
        # Adding another target to the same record refreshes every stored
        # SelectionDecision via Registry._replace().  Refresh only identities
        # created by this build, immediately after our own serialized select.
        for owned_target, owned in tuple(fresh_by_target.items()):
            current = registry.get_selection(owned_target)
            if (
                current is not None
                and current.ownership_token is owned.ownership_token
            ):
                fresh_by_target[owned_target] = current

    def select_for_runtime(target):
        before = registry.get_selection(target)
        selected = registry.select(target, environment=environment)
        if before is None:
            remember_fresh(target, selected)
        return selected

    if python_explicit:
        pass
    elif selector is not None:
        record = _selector_record(records, selector)
        if _enum_value(record.source) == "legacy":
            # The selector constrains the governed Legacy candidate set; it
            # is not permission to publish before is_active/constructor/
            # target probes succeed.
            requested_legacy_record_ids.append(record.record_id)
        else:
            decisions.append(select_for_runtime(_bootstrap_target(record)))
    else:
        targets = tuple(
            sorted(
                {
                    target
                    for record in records
                    if (
                        record.manifest is not None
                        and _enum_value(record.compatibility_status)
                        == "compatible"
                        and _enum_value(record.state)
                        in {
                            "validated",
                            "loaded",
                            "registered",
                            "selected",
                            "active",
                        }
                    )
                    for target in record.manifest.targets
                }
            )
        )
        for target in targets:
            existing = registry.get_selection(target)
            if existing is not None:
                if existing.is_legacy:
                    requested_legacy_record_ids.append(existing.record_id)
                else:
                    decisions.append(existing)
            else:
                try:
                    decisions.append(select_for_runtime(target))
                except BaseException:
                    _rollback_driver_resolution_build(
                        registry, tuple(fresh_by_target.values())
                    )
                    raise

    selected = []
    selected_ids = set()
    current_decisions = []
    try:
        for decision in decisions:
            current = registry.get_selection(decision.target)
            if (
                current is None
                or current.ownership_token is not decision.ownership_token
            ):
                raise _registry_api().BackendPluginLifecycleError(
                    "Runtime backend selection changed during resolution",
                    plugin_id=decision.plugin_id,
                    entry_point=decision.entry_point_name,
                    field="selection",
                    expected=f"{decision.target} -> {decision.record_id}",
                    actual=(
                        "<missing>"
                        if current is None
                        else f"{current.target} -> {current.record_id}"
                    ),
                    remediation="Retry runtime resolution from current state.",
                )
            current_decisions.append(current)
            if current.record_id in selected_ids:
                continue
            selected.append(_backend_for_decision(current))
            selected_ids.add(current.record_id)
    except BaseException:
        _rollback_driver_resolution_build(
            registry, tuple(fresh_by_target.values())
        )
        raise

    forced_selection = (
        bool(python_explicit) or selector is not None
    ) and bool(selected)
    manual = _manual_backends()
    rejected = tuple(
        sorted(
            (
                record.record_id,
                record.error,
            )
            for record in records
            if (
                record.error is not None
                and _enum_value(record.source) != "legacy"
            )
        )
    )
    manifest_error = rejected[0][1] if rejected else None
    manual_entry_points = {
        backend.entry_point_name
        for backend in manual
        if backend.entry_point_name is not None
    }
    unreadable = tuple(
        (
            record.record_id,
            record.error,
        )
        for record in records
        if (
            record.error is not None
            and record.manifest is None
            and _enum_value(record.source) != "legacy"
            and record.entry_point_name in manual_entry_points
        )
    )
    if unreadable:
        manifest_error = sorted(unreadable)[0][1]
    if (
        not manual
        and not selected
        and not requested_legacy_record_ids
        and manifest_error is None
    ):
        any_errors = tuple(
            sorted(
                (
                    record.record_id,
                    record.error,
                )
                for record in records
                if record.error is not None
            )
        )
        if any_errors:
            manifest_error = any_errors[0][1]

    return _DriverBackendResolution(
        manifest_candidates=tuple(selected),
        manifest_decisions=tuple(current_decisions),
        manual_candidates=manual,
        requested_legacy_record_ids=tuple(
            sorted(set(requested_legacy_record_ids))
        ),
        speculative_decisions=tuple(
            sorted(fresh_by_target.values(), key=lambda item: item.target)
        ),
        preexisting_decisions=tuple(
            sorted(preexisting_manifest, key=lambda item: item.target)
        ),
        forced_selection=forced_selection,
        manifest_error=manifest_error,
        cache_epoch=cache_epoch,
    )


def _release_speculative_driver_selections(
    resolution: _DriverBackendResolution,
) -> None:
    """Unconditionally release only this probe's exact Manifest decisions."""
    registry = _registry()
    if len(resolution.speculative_decisions) > 1 and not hasattr(
        registry, "release_selections_if_current"
    ):
        raise _registry_api().BackendPluginLifecycleError(
            "Multiple speculative Manifest selections cannot be released "
            "atomically",
            field="selection",
            expected="an atomic conditional selection release",
            actual=str(len(resolution.speculative_decisions)),
            remediation=(
                "Retry with one runtime target or use a Registry that "
                "supports atomic conditional batch release."
            ),
        )

    if resolution.speculative_decisions:
        release_many = getattr(
            registry, "release_selections_if_current", None
        )
        if callable(release_many):
            release_many(resolution.speculative_decisions)
        else:
            registry.release_selection_if_current(
                resolution.speculative_decisions[0]
            )

    if _cache_epoch() != resolution.cache_epoch:
        raise _registry_api().BackendPluginLifecycleError(
            "Runtime backend probing was invalidated by Registry reset",
            field="registry reset",
            expected=str(resolution.cache_epoch),
            actual=str(_cache_epoch()),
            remediation="Retry runtime backend resolution after reset.",
        )


def _authorize_legacy_driver_fallback(
    resolution: _DriverBackendResolution,
) -> None:
    """Clean speculative Manifest state, then authorize Legacy fallback."""
    _release_speculative_driver_selections(resolution)
    if not resolution.allows_legacy_fallback:
        raise _selection_error(
            "The existing or explicitly selected backend driver is inactive",
            field="driver_cls.is_active",
            expected="one active selected backend driver",
            actual="0",
            remediation=(
                "Reset the existing selection or choose a backend whose "
                "driver reports active."
            ),
        )
    if resolution.requested_legacy_record_ids:
        # An exact environment selector is an explicit Legacy opt-in, not an
        # automatic security downgrade.  It still remains lazy and passes F6
        # plus every runtime probe before atomic publication.
        return
    if resolution.manifest_error is not None:
        raise resolution.manifest_error


def get_driver_backends() -> Tuple[Backend, ...]:
    """Return deterministic Registry winners for runtime active probing."""
    resolution = _get_driver_backend_resolution()
    if (
        not resolution.manifest_candidates
        and resolution.manifest_error is not None
        and not resolution.requested_legacy_record_ids
    ):
        raise resolution.manifest_error
    if resolution.requested_legacy_record_ids:
        # Exact Legacy selectors remain lazy; only the runtime consumer may
        # run is_active/constructor/target probes and atomically publish them.
        return ()
    return resolution.candidates


def _commit_manifest_driver(
    resolution: _DriverBackendResolution,
    backend: Backend,
    target,
) -> Backend:
    """Atomically activate/map the exact Manifest decision after all probes."""
    target_name = _target_name(target)
    matches = tuple(
        decision
        for decision in resolution.manifest_decisions
        if (
            decision.record_id == backend.record_id
            and decision.target == target_name
        )
    )
    if len(matches) != 1:
        raise _registry_api().BackendPluginLifecycleError(
            "Active Manifest driver target is not its exact Registry selection",
            plugin_id=backend.plugin_id,
            entry_point=backend.entry_point_name,
            field="selection",
            expected=f"{target_name} -> {backend.record_id}",
            actual=str(len(matches)),
            remediation=(
                "Return a target selected for the same Manifest record."
            ),
        )
    decision = matches[0]
    registry = _registry()
    _stored, _record, publication = registry.commit_selection_if_current(
        decision,
        activate=True,
        publisher=lambda current, record: _publish_selected_mapping(
            current, record, resolution.cache_epoch
        ),
    )
    return publication


def _release_compiler_resolution(
    resolution: _CompilerBackendResolution,
) -> None:
    """Conditionally roll back only this compiler attempt's auto selection."""
    decision = resolution.speculative_decision
    if decision is not None:
        _registry().release_selection_if_current(decision)


def _selected_compiler_callback_error(backend, field, expected, error):
    return _registry_api().BackendPluginLifecycleError(
        f"Selected backend compiler callback '{field}' failed: {error}",
        plugin_id=backend.plugin_id,
        entry_point=backend.entry_point_name,
        field=field,
        expected=expected,
        actual=f"<error: {error}>",
        remediation=(
            "Fix the selected compiler callback so Manifest publication can "
            "complete without leaving partial selection state."
        ),
    )


def _environment_selected_legacy_record(registry):
    selector = os.environ.get(_registry_api().BACKEND_SELECTOR_ENV)
    if not selector:
        return None
    record = _selector_record(registry.validate(), selector)
    if _enum_value(record.source) == "legacy":
        return record
    return None


def make_backend(target):
    """Instantiate the selected compiler while preserving target validation."""
    api = _registry_api()
    registry = _registry()
    explicit_legacy = _environment_selected_legacy_record(registry)
    if explicit_legacy is not None:
        return _resolve_legacy_compiler_for_target(
            target,
            authority=_LEGACY_COMPILER_FALLBACK_AUTHORITY,
            selected_record_ids=(explicit_legacy.record_id,),
        )
    target_name = _target_name(target)
    existing = registry.get_selection(target_name)
    if existing is not None and bool(existing.is_legacy):
        # A Core explicit Legacy selection is an exact opt-in, not permission
        # to expose its mapping before target/constructor probes complete.
        return _resolve_legacy_compiler_for_target(
            target,
            authority=_LEGACY_COMPILER_FALLBACK_AUTHORITY,
            selected_record_ids=(existing.record_id,),
        )
    try:
        resolution = _get_backend_resolution(
            target,
            allow_manual_legacy=False,
            publish=False,
        )
    except api.BackendPluginNoCandidateError:
        return _resolve_legacy_compiler_for_target(
            target,
            authority=_LEGACY_COMPILER_FALLBACK_AUTHORITY,
        )
    backend = resolution.backend
    try:
        compiler_cls = backend.compiler
        supports_target = getattr(compiler_cls, "supports_target", None)
        if not callable(supports_target):
            raise _selection_error(
                "Selected backend compiler has no callable supports_target()",
                field="compiler_cls.supports_target",
                expected="a callable target predicate",
                actual=repr(supports_target),
                remediation=(
                    "Implement supports_target(target) on the selected compiler."
                ),
            )
        try:
            supported = bool(supports_target(target))
        except api.BackendPluginError:
            raise
        except Exception as exc:
            raise _selected_compiler_callback_error(
                backend,
                "compiler_cls.supports_target",
                "a successful boolean target predicate",
                exc,
            ) from exc
        if not supported:
            target_name = _target_name(target)
            raise _selection_error(
                "Selected backend compiler rejected its declared target",
                field="compiler_cls.supports_target",
                expected=str(target_name),
                actual="False",
                remediation=(
                    "Align Manifest targets with compiler_cls.supports_target()."
                ),
            )
        try:
            compiler = compiler_cls(target)
        except api.BackendPluginError:
            raise
        except Exception as exc:
            raise _selected_compiler_callback_error(
                backend,
                "compiler_cls.__init__",
                "a successfully constructed compiler",
                exc,
            ) from exc
        _cache_decision(resolution.decision)
        return compiler
    except BaseException:
        _release_compiler_resolution(resolution)
        raise


def activate_backend(backend: Backend, *, target=None):
    """Activate the Registry record paired with an instantiated driver."""
    record_id = getattr(backend, "record_id", None)
    if record_id is None:
        # Existing manually managed Legacy entries retain their old behavior.
        if target is None:
            return None
        target_name = _target_name(target)
        registry = _registry()
        records = registry.validate()
        malformed_same_entry_point = tuple(
            sorted(
                (record.record_id, record.error)
                for record in records
                if (
                    record.error is not None
                    and record.manifest is None
                    and _enum_value(record.source) != "legacy"
                    and record.entry_point_name
                    == backend.entry_point_name
                )
            )
        )
        if malformed_same_entry_point:
            raise malformed_same_entry_point[0][1]
        matching = tuple(
            record
            for record in records
            if (
                record.manifest is not None
                and target_name in record.manifest.targets
            )
        )
        rejected = tuple(
            sorted(
                (record.record_id, record.error)
                for record in matching
                if record.error is not None
            )
        )
        if rejected:
            raise rejected[0][1]
        if matching:
            selected = registry.get_selection(target_name)
            raise _selection_error(
                "Manual Legacy driver overlaps a Manifest-governed target",
                field="targets",
                expected=(
                    selected.record_id
                    if selected is not None
                    else "a Registry-selected Manifest backend"
                ),
                actual="manual Legacy backend",
                remediation=(
                    "Use the Manifest winner for this target, or remove the "
                    "overlapping Manifest before using the Legacy channel."
                ),
            )
        return None
    if target is not None:
        selected = get_backend(target)
        if selected.record_id != record_id:
            target_name = getattr(target, "backend", target)
            raise _selection_error(
                "Active driver and compiler resolve to different plugins",
                field="compiler_cls,driver_cls",
                expected=record_id,
                actual=selected.record_id,
                remediation=(
                    "Make the active driver's target resolve to the same "
                    "Manifest plugin for compiler and runtime."
                ),
            )
    try:
        return _registry().activate(record_id)
    except Exception:
        _drop_cached_backend(backend)
        raise


_registry().register_reset_invalidation_hook(_reset_backend_cache)
_discover_backends()


__all__ = [
    "Backend",
    "BaseBackend",
    "DriverBase",
    "activate_backend",
    "backends",
    "get_backend",
    "get_driver_backends",
    "make_backend",
    "register_backend_reset_hook",
    "resolve_active_legacy_driver",
    "resolve_legacy_compiler_for_target",
]
