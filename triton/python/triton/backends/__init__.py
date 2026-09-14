import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from .driver import DriverBase
from .compiler import BaseBackend


@dataclass(frozen=True)
class Backend:
    compiler: Optional[type] = None
    driver: Optional[type] = None
    record_id: Optional[str] = None
    plugin_id: Optional[str] = None
    entry_point_name: Optional[str] = None


# Keep this public object stable: existing code may import it by reference or
# add a manually managed Legacy backend.  Registry-backed entries are added
# only after selection.
backends: Dict[str, Backend] = {}
_LEGACY_BRIDGE_PLUGIN_ID = "__triton_legacy_entry_point__"
_legacy_bridge_errors: Dict[str, Exception] = {}


def _registry_api():
    # Lazy import avoids making the Triton package depend on plugin loading at
    # module import time.  triton_anchor.backends itself performs no ep.load().
    import triton_anchor.backends as api

    return api


def _registry():
    return _registry_api().get_backend_plugin_registry()


def _enum_value(value):
    return getattr(value, "value", value)


def _is_selected_record(record) -> bool:
    return (
        _enum_value(record.state) in {"selected", "active"}
        and bool(record.selected_targets)
    )


def _prune_registry_backends(registry) -> None:
    for name, backend in tuple(backends.items()):
        record_id = getattr(backend, "record_id", None)
        if record_id is None:
            continue
        try:
            record = registry.inspect(record_id)
        except Exception:
            backends.pop(name, None)
            continue
        if not _is_selected_record(record):
            backends.pop(name, None)


def _is_legacy_bridge_backend(backend) -> bool:
    return (
        isinstance(backend, Backend)
        and backend.record_id is None
        and backend.plugin_id == _LEGACY_BRIDGE_PLUGIN_ID
    )


def _interface_error(
    *,
    entry_point: str,
    invalid_fields=(),
    field_errors=None,
):
    return _registry_api().BackendPluginInterfaceError(
        invalid_fields=invalid_fields,
        field_errors=field_errors or {},
        entry_point=entry_point,
    )


def _abstract_methods(cls) -> Tuple[str, ...]:
    return tuple(sorted(getattr(cls, "__abstractmethods__", ()) or ()))


def _validate_legacy_runtime_pair(record) -> None:
    """Validate the old compiler/driver pair before public publication."""
    field_errors = {}
    compiler_cls = record.compiler_cls
    driver_cls = record.driver_cls
    if not isinstance(compiler_cls, type):
        field_errors["compiler_cls"] = "not a class object"
    else:
        try:
            if not issubclass(compiler_cls, BaseBackend):
                field_errors["compiler_cls"] = "not a BaseBackend subclass"
        except TypeError:
            field_errors["compiler_cls"] = "not a BaseBackend subclass"
        missing = _abstract_methods(compiler_cls)
        if missing:
            field_errors["compiler_cls"] = (
                "abstract methods are not implemented: "
                + ", ".join(missing)
            )
        supports_target = getattr(compiler_cls, "supports_target", None)
        if not callable(supports_target):
            field_errors["compiler_cls.supports_target"] = "not callable"

    if not isinstance(driver_cls, type):
        field_errors["driver_cls"] = "not a class object"
    else:
        try:
            if not issubclass(driver_cls, DriverBase):
                field_errors["driver_cls"] = "not a DriverBase subclass"
        except TypeError:
            field_errors["driver_cls"] = "not a DriverBase subclass"
        missing = _abstract_methods(driver_cls)
        if missing:
            field_errors["driver_cls"] = (
                "abstract methods are not implemented: "
                + ", ".join(missing)
            )
        is_active = getattr(driver_cls, "is_active", None)
        if not callable(is_active):
            field_errors["driver_cls.is_active"] = "not callable"
        get_current_target = getattr(driver_cls, "get_current_target", None)
        if not callable(get_current_target):
            field_errors["driver_cls.get_current_target"] = "not callable"

    if field_errors:
        raise _interface_error(
            entry_point=record.entry_point_name,
            field_errors=field_errors,
        )


def _legacy_records(registry) -> Tuple[object, ...]:
    return tuple(
        record
        for record in registry.list()
        if (
            record.manifest is None
            and _enum_value(record.source) == "legacy"
            and _enum_value(record.state) != "rejected"
        )
    )


def _publish_legacy_backend(record) -> Backend:
    _validate_legacy_runtime_pair(record)
    backend = Backend(
        compiler=record.compiler_cls,
        driver=record.driver_cls,
        record_id=None,
        plugin_id=_LEGACY_BRIDGE_PLUGIN_ID,
        entry_point_name=record.entry_point_name,
    )
    current = backends.get(record.entry_point_name)
    if current is not None:
        if (
            (
                _is_legacy_bridge_backend(current)
                or current.record_id == record.record_id
            )
            and current.compiler is backend.compiler
            and current.driver is backend.driver
        ):
            return current
        raise _registry_api().BackendPluginConflictError(
            "Legacy backend public mapping key is already owned: "
            + record.entry_point_name,
            entry_point=record.entry_point_name,
            field="backends",
            expected="an unused entry-point name or the same Legacy backend",
            actual=record.entry_point_name,
            remediation=(
                "Remove the conflicting manual mapping or migrate the backend "
                "to a Manifest with a unique entry-point name."
            ),
        )
    backends[record.entry_point_name] = backend
    return backend


def _materialize_legacy_backends(registry) -> Tuple[Backend, ...]:
    """Load only no-Manifest legacy entry points for the public API bridge."""
    published = []
    generation = registry.generation
    for record in _legacy_records(registry):
        if registry.generation != generation:
            return ()
        if record.record_id in _legacy_bridge_errors:
            continue
        try:
            registered = registry.register(record.record_id)
            if registry.generation != generation:
                return ()
            backend = _publish_legacy_backend(registered)
            if registry.generation != generation:
                # A reset may have completed while plugin code was running.
                # Remove only this stale object, never a newer publication.
                if backends.get(record.entry_point_name) is backend:
                    backends.pop(record.entry_point_name, None)
                return ()
            published.append(backend)
        except Exception as error:
            if registry.generation != generation:
                return ()
            # Legacy discovery historically warned and continued per plugin.
            # Keep failed entries out of the public mapping and retain their
            # diagnostic for explicit selection and environments with no peers.
            _legacy_bridge_errors[record.record_id] = error
            print(
                "Warning: Failed to load out-of-tree backend "
                f"'{record.entry_point_name}': {error}",
                file=sys.stderr,
            )
    return tuple(published)


def _cache_decision(decision) -> Backend:
    if decision.record_id in _legacy_bridge_errors:
        raise _legacy_bridge_errors[decision.record_id]
    registry = _registry()
    generation = registry.generation
    try:
        record = registry.inspect(decision.record_id)
    except Exception as exc:
        raise _registry_api().BackendPluginLifecycleError(
            "Backend selection was invalidated before consumption",
            plugin_id=decision.plugin_id,
            entry_point=decision.entry_point_name,
            field="registry generation",
            expected=str(generation),
            actual="selection record unavailable",
            remediation=(
                "Retry backend resolution after the concurrent Registry "
                "reset or selection change completes."
            ),
        ) from exc
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
            ),
        )
    backend = Backend(
        compiler=record.compiler_cls,
        driver=record.driver_cls,
        record_id=record.record_id,
        plugin_id=record.plugin_id,
        entry_point_name=record.entry_point_name,
    )
    backends[record.entry_point_name] = backend
    _prune_registry_backends(registry)
    if registry.generation != generation:
        cached = backends.get(record.entry_point_name)
        if (
            cached is not None
            and getattr(cached, "record_id", None) == record.record_id
        ):
            backends.pop(record.entry_point_name, None)
        raise _registry_api().BackendPluginLifecycleError(
            "Backend selection was invalidated during consumption",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="registry generation",
            expected=str(generation),
            actual=str(registry.generation),
            remediation=(
                "Retry backend resolution after the concurrent Registry "
                "reset or selection change completes."
            ),
        )
    return backend


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
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        return target.get("backend")
    return getattr(target, "backend", None)


def _manual_backends() -> Tuple[Backend, ...]:
    # A previously selected Legacy record remains available for implicit
    # probing after an environment selector is cleared. Use its Legacy view
    # here so Manifest ownership checks still apply to the actual target.
    legacy_ids = {record.record_id for record in _legacy_records(_registry())}
    result = []
    for name, backend in sorted(backends.items()):
        record_id = getattr(backend, "record_id", None)
        if record_id is not None:
            if record_id not in legacy_ids:
                continue
            backend = replace(
                backend, record_id=None, plugin_id=_LEGACY_BRIDGE_PLUGIN_ID
            )
        if (
            isinstance(backend, Backend)
            and backend.entry_point_name is None
        ):
            backend = replace(backend, entry_point_name=name)
        result.append(backend)
    return tuple(result)


def _reset_backend_cache() -> None:
    _legacy_bridge_errors.clear()
    for name, backend in tuple(backends.items()):
        if (
            getattr(backend, "record_id", None) is not None
            or _is_legacy_bridge_backend(backend)
        ):
            backends.pop(name, None)


def register_backend_reset_hook(callback) -> None:
    """Coordinate Registry reset with a Triton-side lazy cache."""
    _registry().register_reset_hook(callback)


def _discover_backends():
    """Discover Registry metadata and publish no-Manifest legacy entries."""
    registry = _registry()
    registry.discover()
    _materialize_legacy_backends(registry)
    return backends


def get_backend(target) -> Backend:
    """Return the Registry-selected compiler/driver pair for one target."""
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
        return _cache_decision(existing)

    try:
        decision = registry.select(target)
    except api.BackendPluginSelectionError:
        # An explicit selector is authoritative, including an invalid one.
        if environment_selector:
            raise
        records = registry.list()
        manifest_claims_target = any(
            record.manifest is not None
            and target_name in record.manifest.targets
            for record in records
        )
        if manifest_claims_target:
            raise
        _materialize_legacy_backends(registry)
        records = registry.list()
        candidates = []
        for backend in _manual_backends():
            supports_target = getattr(backend.compiler, "supports_target", None)
            if not callable(supports_target):
                continue
            try:
                supported = supports_target(target)
            except api.BackendPluginError:
                raise
            except Exception as exc:
                raise api.BackendPluginLifecycleError(
                    f"Legacy backend compiler target probe failed: {exc}",
                    plugin_id=backend.plugin_id,
                    entry_point=backend.entry_point_name,
                    field="compiler_cls.supports_target",
                    expected="a successful boolean target predicate",
                    actual=f"<error: {exc}>",
                    remediation=(
                        "Fix supports_target(target), or migrate the backend "
                        "to a Manifest so static target ownership is explicit."
                    ),
                ) from exc
            if supported:
                candidates.append(backend)
        candidates = tuple(candidates)
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
        if unreadable_related:
            raise
        if len(candidates) == 1:
            return candidates[0]
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
    return _cache_decision(decision)


def get_driver_backends() -> Tuple[Backend, ...]:
    """Return deterministic Registry winners for runtime active probing.

    Without an explicit selector, one winner is selected per declared
    Manifest target.  This preserves existing ``driver_cls.is_active()``
    hardware probing while ensuring same-target losers are never imported.
    """
    api = _registry_api()
    registry = _registry()
    records = registry.validate()
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
    python_explicit = tuple(
        decision
        for decision in cached
        if _enum_value(decision.method) == "python_explicit"
    )

    decisions = list(python_explicit)
    if python_explicit:
        pass
    elif selector is not None:
        record = _selector_record(records, selector)
        decisions.append(
            registry.select(
                _bootstrap_target(record),
                environment=environment,
            )
        )
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
            decisions.append(
                existing
                if existing is not None
                else registry.select(target, environment=environment)
            )

    selected = []
    selected_ids = set()
    for decision in decisions:
        if decision.record_id in selected_ids:
            continue
        selected.append(_cache_decision(decision))
        selected_ids.add(decision.record_id)

    forced_selection = bool(python_explicit) or selector is not None
    if selected and forced_selection:
        return tuple(selected)

    # No-Manifest plugins have no static target ownership. Include them in
    # runtime probing alongside Manifest winners; the actual driver target
    # determines whether a Legacy candidate overlaps a governed target.
    _materialize_legacy_backends(registry)
    manual = _manual_backends()
    if selected or manual:
        return tuple(selected) + manual

    errors = {
        record.record_id: record.error
        for record in registry.list()
        if record.error is not None
    }
    errors.update(_legacy_bridge_errors)
    if errors:
        raise errors[sorted(errors)[0]]
    return ()


def make_backend(target):
    """Instantiate the selected compiler while preserving target validation."""
    backend = get_backend(target)
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
    if not supports_target(target):
        target_name = getattr(target, "backend", target)
        raise _selection_error(
            "Selected backend compiler rejected its declared target",
            field="compiler_cls.supports_target",
            expected=str(target_name),
            actual="False",
            remediation=(
                "Align Manifest targets with compiler_cls.supports_target()."
            ),
        )
    return compiler_cls(target)


def _has_manifest_records() -> bool:
    return any(
        _enum_value(record.source) != "legacy"
        for record in _registry().list()
    )


def _legacy_target_error(backend: Backend, target):
    """Return a Legacy ownership error without activating a driver."""
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
        return malformed_same_entry_point[0][1]
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
    selected = registry.get_selection(target_name)
    if rejected and selected is None:
        return rejected[0][1]
    if matching:
        return _selection_error(
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


def activate_backend(backend: Backend, *, target=None):
    """Activate the Registry record paired with an instantiated driver."""
    record_id = getattr(backend, "record_id", None)
    if record_id is None:
        # Existing manually managed Legacy entries retain their old behavior.
        if target is None:
            return None
        error = _legacy_target_error(backend, target)
        if error is not None:
            raise error
        return None
    if target is not None:
        selected = get_backend(target)
        if selected.record_id != record_id:
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
    return _registry().activate(record_id)


_registry().register_reset_hook(_reset_backend_cache)
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
]
