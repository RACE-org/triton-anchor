from __future__ import annotations

import threading
from dataclasses import dataclass

from triton_anchor.backends import (
    BackendPluginConflictError,
    BackendPluginLifecycleError,
    BackendPluginNoCandidateError,
    BackendPluginSelectionError,
)

from ..backends import (
    DriverBase,
    _backend_registry_generation,
    _backend_registry_lifecycle_epoch,
    _cache_decision,
    _construct_legacy_driver,
    _legacy_driver_target,
    _prepare_driver_backend_proposals,
    _probe_compiler,
    _probe_driver_active,
    _require_legacy_resolution_lifecycle_epoch,
    _resolve_active_legacy_driver_after_manifest_miss,
    _run_legacy_resolution,
    _safe_class_name,
    activate_explicit_driver,
    register_backend_reset_hook,
)


@dataclass(frozen=True)
class _DriverResolution:
    driver: DriverBase
    registry_generation: int | None
    registry_lifecycle_epoch: int | None


class _LegacyRuntimeFallback(Exception):
    def __init__(
        self,
        generation,
        lifecycle_epoch,
        record_ids=(),
        no_active_error=None,
    ):
        super().__init__("governed Legacy runtime fallback")
        self.generation = generation
        self.lifecycle_epoch = lifecycle_epoch
        self.record_ids = tuple(record_ids)
        self.no_active_error = no_active_error


def _create_driver_attempt(reset_guard=None) -> _DriverResolution:
    proposals = _prepare_driver_backend_proposals()
    lease_generations = {
        lease.generation for proposal in proposals for lease in proposal.leases
    }
    lease_epochs = {
        lease.lifecycle_epoch for proposal in proposals for lease in proposal.leases
    }
    if len(lease_generations) > 1 or len(lease_epochs) > 1:
        raise BackendPluginLifecycleError(
            "Runtime proposals crossed a Registry generation",
            field="registry generation",
            expected="one generation for the complete runtime scan",
            actual=", ".join(str(value) for value in sorted(lease_generations)),
            remediation="Retry runtime resolution after Registry state stabilizes.",
        )
    resolution_generation = (
        next(iter(lease_generations))
        if lease_generations
        else _backend_registry_generation()
    )
    resolution_lifecycle_epoch = (
        next(iter(lease_epochs))
        if lease_epochs
        else _backend_registry_lifecycle_epoch()
    )
    if reset_guard is not None:
        reset_guard()
    if not proposals:
        raise _LegacyRuntimeFallback(
            resolution_generation,
            resolution_lifecycle_epoch,
        )
    actives = []
    for proposal in proposals:
        backend = proposal.backend
        _require_legacy_resolution_lifecycle_epoch(
            _registry_for_runtime(),
            resolution_lifecycle_epoch,
            backend=backend,
        )
        if _probe_driver_active(backend):
            actives.append(proposal)
        if reset_guard is not None:
            reset_guard()
        _require_legacy_resolution_lifecycle_epoch(
            _registry_for_runtime(),
            resolution_lifecycle_epoch,
            backend=backend,
        )

    if not actives:
        no_active_error = BackendPluginSelectionError(
            "No Registry-selected backend driver is active",
            field="driver_cls.is_active",
            expected="one active backend driver",
            actual="0",
            remediation=(
                "Install or select a backend whose driver reports active, "
                "or set TRITON_ANCHOR_BACKEND explicitly."
            ),
        )
        manifest_proposals = tuple(
            proposal for proposal in proposals if proposal.leases
        )
        if manifest_proposals and not any(
            proposal.forced for proposal in manifest_proposals
        ):
            if reset_guard is not None:
                reset_guard()
            if _backend_registry_generation() != resolution_generation:
                raise no_active_error
            raise _LegacyRuntimeFallback(
                resolution_generation,
                resolution_lifecycle_epoch,
                (),
                no_active_error,
            )
        raise no_active_error
    if len(actives) > 1:
        record_ids = tuple(
            sorted(
                proposal.backend.record_id
                or "manual:" + (proposal.backend.entry_point_name or "<unknown>")
                for proposal in actives
            )
        )
        raise BackendPluginConflictError(
            f"{len(actives)} Registry-selected backend drivers are active",
            field="driver_cls.is_active",
            expected="one active backend driver",
            actual=", ".join(
                sorted(
                    _safe_class_name(proposal.backend.driver) for proposal in actives
                )
            ),
            conflict_kind="active_runtime_driver",
            related_record_ids=record_ids,
            remediation=(
                "Use TRITON_ANCHOR_BACKEND or hardware configuration so only "
                "one selected driver reports active."
            ),
        )

    proposal = actives[0]
    backend = proposal.backend
    if reset_guard is not None:
        reset_guard()
    active_driver = _construct_legacy_driver(backend)
    if reset_guard is not None:
        reset_guard()
    _require_legacy_resolution_lifecycle_epoch(
        _registry_for_runtime(),
        resolution_lifecycle_epoch,
        backend=backend,
    )
    target = _legacy_driver_target(backend, active_driver)
    if reset_guard is not None:
        reset_guard()
    _require_legacy_resolution_lifecycle_epoch(
        _registry_for_runtime(),
        resolution_lifecycle_epoch,
        backend=backend,
    )

    if not proposal.leases:
        # Manual mappings are outside Registry governance but may only be
        # reached when no Manifest or governed Legacy record exists.
        from ..backends import activate_backend

        if not _probe_compiler(backend, target):
            raise BackendPluginSelectionError(
                "Manual driver and compiler disagree on the current target",
                field="compiler_cls.supports_target",
                expected=target.backend,
                actual="False",
                remediation="Keep the manual compiler/driver pair consistent.",
            )
        if reset_guard is not None:
            reset_guard()
        _require_legacy_resolution_lifecycle_epoch(
            _registry_for_runtime(),
            resolution_lifecycle_epoch,
            backend=backend,
        )
        activate_backend(backend, target=target)
        return _DriverResolution(
            active_driver,
            resolution_generation,
            resolution_lifecycle_epoch,
        )

    target_leases = tuple(
        lease for lease in proposal.leases if lease.target == target.backend
    )
    if len(target_leases) != 1:
        raise BackendPluginSelectionError(
            "Manifest driver returned a target outside its exact record",
            plugin_id=backend.plugin_id,
            entry_point=backend.entry_point_name,
            field="driver_cls.get_current_target",
            expected=", ".join(lease.target for lease in proposal.leases),
            actual=target.backend,
            remediation=(
                "Return a declared GPUTarget owned by the same Manifest record."
            ),
        )
    if not _probe_compiler(backend, target):
        raise BackendPluginSelectionError(
            "Manifest driver and compiler disagree on the current target",
            plugin_id=backend.plugin_id,
            entry_point=backend.entry_point_name,
            field="compiler_cls.supports_target",
            expected=target.backend,
            actual="False",
            remediation="Keep driver and compiler target support consistent.",
        )
    if reset_guard is not None:
        reset_guard()
    registry = _registry_for_runtime()
    _require_legacy_resolution_lifecycle_epoch(
        registry,
        resolution_lifecycle_epoch,
        backend=backend,
    )
    decision = registry.commit_runtime_selection(
        target_leases[0],
        activate=True,
    )
    published = _cache_decision(decision)
    return _DriverResolution(
        active_driver,
        published.registry_generation,
        published.registry_lifecycle_epoch,
    )


def _create_driver(reset_guard=None) -> _DriverResolution:
    try:
        return _run_legacy_resolution(
            "manifest-runtime",
            lambda: _create_driver_attempt(reset_guard),
        )
    except _LegacyRuntimeFallback as fallback:
        try:
            active_driver, backend = _resolve_active_legacy_driver_after_manifest_miss(
                expected_generation=fallback.generation,
                expected_lifecycle_epoch=fallback.lifecycle_epoch,
                inactive_manifest_record_ids=fallback.record_ids,
                return_backend=True,
            )
        except BackendPluginNoCandidateError:
            if fallback.no_active_error is not None:
                raise fallback.no_active_error
            raise
        return _DriverResolution(
            active_driver,
            backend.registry_generation,
            backend.registry_lifecycle_epoch,
        )


def _registry_for_runtime():
    from ..backends import _registry

    return _registry()


class DriverConfig:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._default: DriverBase | None = None
        self._active: DriverBase | None = None
        self._default_initializing = False
        self._default_owner_thread: int | None = None
        self._default_attempt = 0
        self._default_failures: dict[int, BaseException] = {}
        self._default_waiters: dict[int, int] = {}
        self._default_registry_generation: int | None = None
        self._active_registry_generation: int | None = None
        self._default_registry_lifecycle_epoch: int | None = None
        self._active_registry_lifecycle_epoch: int | None = None
        self._reset_epoch = 0

    @property
    def default(self) -> DriverBase:
        waited_attempt = None
        while True:
            with self._condition:
                if self._default is not None:
                    if (
                        self._default_registry_lifecycle_epoch is None
                        or _backend_registry_lifecycle_epoch()
                        == self._default_registry_lifecycle_epoch
                    ):
                        return self._default
                    self._default = None
                    self._active = None
                    self._default_registry_generation = None
                    self._active_registry_generation = None
                    self._default_registry_lifecycle_epoch = None
                    self._active_registry_lifecycle_epoch = None
                if self._default_initializing:
                    if self._default_owner_thread == threading.get_ident():
                        raise BackendPluginLifecycleError(
                            "Driver default initialization is re-entrant",
                            field="driver.default",
                            expected="non-recursive runtime initialization",
                            actual="the initializer re-entered driver.default",
                            remediation=(
                                "Return from is_active(), constructors, and "
                                "get_current_target() before resolving driver.default."
                            ),
                        )
                    waited_attempt = self._default_attempt
                    self._default_waiters[waited_attempt] = (
                        self._default_waiters.get(waited_attempt, 0) + 1
                    )
                    while (
                        self._default_initializing
                        and self._default_attempt == waited_attempt
                        and self._default is None
                    ):
                        self._condition.wait()
                    failure = self._default_failures.get(waited_attempt)
                    remaining = self._default_waiters[waited_attempt] - 1
                    if remaining:
                        self._default_waiters[waited_attempt] = remaining
                    else:
                        del self._default_waiters[waited_attempt]
                        self._default_failures.pop(waited_attempt, None)
                    if failure is not None:
                        raise failure
                    if self._default is not None:
                        if (
                            self._default_registry_lifecycle_epoch is None
                            or _backend_registry_lifecycle_epoch()
                            == self._default_registry_lifecycle_epoch
                        ):
                            return self._default
                        self._default = None
                        self._active = None
                        self._default_registry_generation = None
                        self._active_registry_generation = None
                        self._default_registry_lifecycle_epoch = None
                        self._active_registry_lifecycle_epoch = None
                    continue
                reset_epoch = self._reset_epoch
                self._default_attempt += 1
                attempt = self._default_attempt
                self._default_initializing = True
                self._default_owner_thread = threading.get_ident()

            # Backend construction invokes plugin code. Run it without the
            # config lock so Registry.reset() can invalidate this attempt
            # without waiting on a plugin-owned lock.
            try:
                resolution = _create_driver(
                    lambda expected=reset_epoch: self._check_reset_epoch(expected)
                )
            except BaseException as exc:
                with self._condition:
                    if (
                        self._default_initializing
                        and self._default_attempt == attempt
                        and self._default_owner_thread == threading.get_ident()
                    ):
                        self._default_initializing = False
                        self._default_owner_thread = None
                        if self._default_waiters.get(attempt, 0):
                            self._default_failures[attempt] = exc
                    self._condition.notify_all()
                raise

            with self._condition:
                lifecycle_matches = (
                    resolution.registry_lifecycle_epoch is None
                    or _backend_registry_lifecycle_epoch()
                    == resolution.registry_lifecycle_epoch
                )
                if (
                    self._reset_epoch == reset_epoch
                    and self._default is None
                    and lifecycle_matches
                ):
                    self._default = resolution.driver
                    self._default_registry_generation = resolution.registry_generation
                    self._default_registry_lifecycle_epoch = (
                        resolution.registry_lifecycle_epoch
                    )
                    self._default_initializing = False
                    self._default_owner_thread = None
                    self._condition.notify_all()
                    return resolution.driver
                self._default_initializing = False
                self._default_owner_thread = None
                self._condition.notify_all()

    @property
    def active(self) -> DriverBase:
        while True:
            with self._lock:
                if self._active is not None:
                    if (
                        self._active_registry_lifecycle_epoch is None
                        or _backend_registry_lifecycle_epoch()
                        == self._active_registry_lifecycle_epoch
                    ):
                        return self._active
                    self._active = None
                    self._active_registry_generation = None
                    self._active_registry_lifecycle_epoch = None
                reset_epoch = self._reset_epoch

            default = self.default
            with self._lock:
                if self._active is not None:
                    if (
                        self._active_registry_lifecycle_epoch is None
                        or _backend_registry_lifecycle_epoch()
                        == self._active_registry_lifecycle_epoch
                    ):
                        return self._active
                    self._active = None
                    self._active_registry_generation = None
                    self._active_registry_lifecycle_epoch = None
                if self._reset_epoch == reset_epoch and self._default is default:
                    if (
                        self._default_registry_lifecycle_epoch is not None
                        and _backend_registry_lifecycle_epoch()
                        != self._default_registry_lifecycle_epoch
                    ):
                        self._default = None
                        self._default_registry_generation = None
                        self._default_registry_lifecycle_epoch = None
                        continue
                    self._active = default
                    self._active_registry_generation = self._default_registry_generation
                    self._active_registry_lifecycle_epoch = (
                        self._default_registry_lifecycle_epoch
                    )
                    return default

    def _check_reset_epoch(self, expected, backend=None) -> None:
        with self._lock:
            actual = self._reset_epoch
        if actual != expected:
            raise BackendPluginLifecycleError(
                "Explicit backend activation was invalidated by Registry reset",
                plugin_id=getattr(backend, "plugin_id", None),
                entry_point=getattr(backend, "entry_point_name", None),
                field="registry reset",
                expected=str(expected),
                actual=str(actual),
                remediation=(
                    "Retry driver.set_active() after Registry reset completes."
                ),
            )

    def set_active(self, driver: DriverBase) -> None:
        with self._lock:
            if self._default is not None and driver is self._default:
                if (
                    self._default_registry_lifecycle_epoch is None
                    or _backend_registry_lifecycle_epoch()
                    == self._default_registry_lifecycle_epoch
                ):
                    self._active = self._default
                    self._active_registry_generation = self._default_registry_generation
                    self._active_registry_lifecycle_epoch = (
                        self._default_registry_lifecycle_epoch
                    )
                    return
                self._default = None
                self._active = None
                self._default_registry_generation = None
                self._active_registry_generation = None
                self._default_registry_lifecycle_epoch = None
                self._active_registry_lifecycle_epoch = None
            reset_epoch = self._reset_epoch
        registry_lifecycle_epoch = _backend_registry_lifecycle_epoch()
        registry_generation = _backend_registry_generation()
        if _backend_registry_lifecycle_epoch() != registry_lifecycle_epoch:
            raise BackendPluginLifecycleError(
                "Explicit backend activation crossed a Registry reset boundary",
                field="registry lifecycle epoch",
                expected=str(registry_lifecycle_epoch),
                actual=str(_backend_registry_lifecycle_epoch()),
                remediation="Retry driver.set_active() after reset completes.",
            )
        self._check_reset_epoch(reset_epoch)

        # The adapter first matches and F6-validates the exact recorded class;
        # only then may it call get_current_target() or another plugin hook.
        backend = activate_explicit_driver(
            driver,
            expected_generation=registry_generation,
            expected_lifecycle_epoch=registry_lifecycle_epoch,
        )
        self._check_reset_epoch(reset_epoch, backend)
        with self._lock:
            lifecycle_matches = (
                backend.registry_lifecycle_epoch is None
                or _backend_registry_lifecycle_epoch()
                == backend.registry_lifecycle_epoch
            )
            if self._reset_epoch != reset_epoch or not lifecycle_matches:
                actual = self._reset_epoch
                actual_lifecycle_epoch = _backend_registry_lifecycle_epoch()
            else:
                self._active = driver
                self._active_registry_generation = backend.registry_generation
                self._active_registry_lifecycle_epoch = backend.registry_lifecycle_epoch
                return
        lifecycle_changed = not lifecycle_matches
        raise BackendPluginLifecycleError(
            "Explicit backend activation was invalidated by Registry reset",
            plugin_id=backend.plugin_id,
            entry_point=backend.entry_point_name,
            field=(
                "registry lifecycle epoch" if lifecycle_changed else "registry reset"
            ),
            expected=str(
                backend.registry_lifecycle_epoch if lifecycle_changed else reset_epoch
            ),
            actual=str(actual_lifecycle_epoch if lifecycle_changed else actual),
            remediation=("Retry driver.set_active() after Registry reset completes."),
        )

    def reset_active(self) -> None:
        while True:
            with self._lock:
                reset_epoch = self._reset_epoch
            default = self.default
            lifecycle_mismatch = None
            with self._lock:
                if self._reset_epoch == reset_epoch and self._default is default:
                    expected_lifecycle_epoch = self._default_registry_lifecycle_epoch
                    actual_lifecycle_epoch = _backend_registry_lifecycle_epoch()
                    if (
                        expected_lifecycle_epoch is not None
                        and actual_lifecycle_epoch != expected_lifecycle_epoch
                    ):
                        self._default = None
                        self._active = None
                        self._default_registry_generation = None
                        self._active_registry_generation = None
                        self._default_registry_lifecycle_epoch = None
                        self._active_registry_lifecycle_epoch = None
                        lifecycle_mismatch = (
                            expected_lifecycle_epoch,
                            actual_lifecycle_epoch,
                        )
                    else:
                        self._active = default
                        self._active_registry_generation = (
                            self._default_registry_generation
                        )
                        self._active_registry_lifecycle_epoch = (
                            self._default_registry_lifecycle_epoch
                        )
                        return
            if lifecycle_mismatch is not None:
                expected_lifecycle_epoch, actual_lifecycle_epoch = lifecycle_mismatch
                raise BackendPluginLifecycleError(
                    "Driver reset_active crossed a Registry reset boundary",
                    field="registry lifecycle epoch",
                    expected=str(expected_lifecycle_epoch),
                    actual=str(actual_lifecycle_epoch),
                    remediation=(
                        "Retry driver.reset_active() after Registry reset completes."
                    ),
                )

    def _reset_registry_state(self) -> None:
        with self._condition:
            self._reset_epoch += 1
            self._default = None
            self._active = None
            self._default_registry_generation = None
            self._active_registry_generation = None
            self._default_registry_lifecycle_epoch = None
            self._active_registry_lifecycle_epoch = None
            self._condition.notify_all()


driver = DriverConfig()
register_backend_reset_hook(driver._reset_registry_state)
