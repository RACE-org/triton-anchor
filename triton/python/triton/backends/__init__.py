import importlib
import os
import threading
import types
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple, Type

from .compiler import BaseBackend, GPUTarget
from .driver import DriverBase


@dataclass(frozen=True)
class Backend:
    compiler: Optional[Type[BaseBackend]] = None
    driver: Optional[Type[DriverBase]] = None
    record_id: Optional[str] = None
    plugin_id: Optional[str] = None
    entry_point_name: Optional[str] = None
    registry_generation: Optional[int] = None
    registry_lifecycle_epoch: Optional[int] = None


@dataclass(frozen=True)
class _RuntimeBackendProposal:
    """Private uncommitted Manifest runtime pair and its target leases."""

    backend: Backend
    leases: Tuple[object, ...]
    forced: bool = False


@dataclass(frozen=True)
class _ExplicitDriverMatch:
    backend: Backend
    source: str
    lease: object = None


# Keep this public object stable: existing code may import it by reference or
# add a manually managed Legacy backend.  Registry-backed entries are added
# only after selection.
backends: Dict[str, Backend] = {}

_RUNTIME_INTERFACE_CONTRACT_ID = "triton.backends.abstract-runtime-pair"
_LEGACY_RUNTIME_PAIR_MATERIALIZER_ID = "triton.backends.v36-legacy-package-root"
# Preserve the process-owned validator across importlib.reload().  A Registry
# deliberately treats a different object under the same name as a conflicting
# late contract rather than silently replacing its trust boundary.
_runtime_interface_lock = globals().get("_runtime_interface_lock", threading.Lock())
_runtime_pair_validator = globals().get("_runtime_pair_validator")
_legacy_materializer_lock = globals().get("_legacy_materializer_lock", threading.Lock())
_legacy_runtime_pair_materializer = globals().get("_legacy_runtime_pair_materializer")
_backend_cache_lock = globals().get("_backend_cache_lock", threading.RLock())
_legacy_resolution_local = globals().get("_legacy_resolution_local", threading.local())

_TYPE_DICT_GETSET = type.__dict__["__dict__"]
_TYPE_MRO_GETSET = type.__dict__["__mro__"]
_TYPE_MODULE_GETSET = type.__dict__["__module__"]
_TYPE_QUALNAME_GETSET = type.__dict__["__qualname__"]
_MODULE_DICT_GETSET = types.ModuleType.__dict__["__dict__"]


def _registry_api():
    # Lazy import avoids making the Triton package depend on plugin loading at
    # module import time.  triton_anchor.backends itself performs no ep.load().
    import triton_anchor.backends as api

    return api


def _registry():
    return _registry_api().get_backend_plugin_registry()


def _get_runtime_pair_validator():
    global _runtime_pair_validator
    with _runtime_interface_lock:
        if _runtime_pair_validator is None:
            api = _registry_api()
            _runtime_pair_validator = api.AbstractRuntimePairValidator(
                contract_id=_RUNTIME_INTERFACE_CONTRACT_ID,
                surfaces=(
                    api.RuntimeInterfaceSurface(
                        field="compiler_cls",
                        abstract_base=BaseBackend,
                    ),
                    api.RuntimeInterfaceSurface(
                        field="driver_cls",
                        abstract_base=DriverBase,
                    ),
                ),
            )
        return _runtime_pair_validator


def register_runtime_interface_contract(registry=None) -> None:
    """Register this checkout's dynamically derived Triton runtime surface."""
    target_registry = _registry() if registry is None else registry
    target_registry.register_runtime_pair_validator(
        _RUNTIME_INTERFACE_CONTRACT_ID,
        _get_runtime_pair_validator(),
    )


def _stable_exception_actual(exc: BaseException) -> str:
    return _safe_class_name(type(exc))


def _safe_class_name(cls: type) -> str:
    try:
        module = _TYPE_MODULE_GETSET.__get__(cls, type(cls))
        qualname = _TYPE_QUALNAME_GETSET.__get__(cls, type(cls))
    except (TypeError, AttributeError):
        value_type = type(cls)
        try:
            value_module = _TYPE_MODULE_GETSET.__get__(value_type, type(value_type))
            value_qualname = _TYPE_QUALNAME_GETSET.__get__(value_type, type(value_type))
        except (TypeError, AttributeError):
            return "<invalid class identity>"
        if type(value_module) is str and type(value_qualname) is str:
            return f"<non-class {value_module}.{value_qualname}>"
        return "<invalid class identity>"
    if type(module) is not str or type(qualname) is not str:
        return "<invalid class identity>"
    return f"{module}.{qualname}"


def _module_runtime_candidates(module, abstract_base, *, field, entry_point):
    api = _registry_api()
    try:
        namespace = _MODULE_DICT_GETSET.__get__(module, type(module))
    except (TypeError, AttributeError) as exc:
        raise api.BackendPluginInterfaceError(
            (),
            field_errors={
                field: "module namespace is not statically readable: "
                + _stable_exception_actual(exc)
            },
            entry_point=entry_point,
        ) from exc
    module_name = namespace.get("__name__")
    if type(module_name) is not str or not module_name:
        raise api.BackendPluginInterfaceError(
            (),
            field_errors={field: "module has no stable exact-string __name__"},
            entry_point=entry_point,
        )

    inherited = []
    structural = []
    reexported_structural = []
    for value in tuple(namespace.values()):
        if not isinstance(value, type) or value is abstract_base:
            continue
        try:
            mro = _TYPE_MRO_GETSET.__get__(value, type(value))
            owner = _TYPE_MODULE_GETSET.__get__(value, type(value))
        except (TypeError, AttributeError):
            continue
        # Match upstream v3.6 first: an ABC-derived class may be re-exported
        # from an implementation submodule.  F6 safely determines whether it
        # is concrete.  Structural fallback is restricted to classes owned by
        # this module so imported helper classes do not become candidates.
        if abstract_base in mro:
            inherited.append(value)
        if owner == module_name:
            structural.append(value)
        else:
            reexported_structural.append(value)
    inherited.sort(key=_safe_class_name)
    structural = [
        candidate
        for candidate in structural
        if all(candidate is not inherited_item for inherited_item in inherited)
    ]
    structural.sort(key=_safe_class_name)
    candidates = (
        tuple((candidate, 0) for candidate in inherited)
        + tuple((candidate, 1) for candidate in structural)
        + tuple(
            (candidate, 2)
            for candidate in sorted(
                (
                    candidate
                    for candidate in reexported_structural
                    if all(
                        candidate is not inherited_item for inherited_item in inherited
                    )
                ),
                key=_safe_class_name,
            )
        )
    )
    if not candidates:
        raise api.BackendPluginInterfaceError(
            (field,),
            field_errors={
                field: (
                    "expected a Triton 3.6 ABC-derived or structurally "
                    f"complete class in {module_name}; found none"
                )
            },
            entry_point=entry_point,
        )
    return candidates


def _materialize_v36_legacy_runtime_pair(context):
    api = _registry_api()
    if type(context) is not api.LegacyRuntimePairMaterializationContext:
        raise TypeError(
            "context must be an exact LegacyRuntimePairMaterializationContext"
        )
    package_root = context.entry_point_value
    if (
        type(package_root) is not str
        or not package_root
        or package_root != package_root.strip()
        or ":" in package_root
    ):
        raise api.BackendPluginInterfaceError(
            (),
            field_errors={
                "entry_point.value": (
                    "Triton 3.6 Legacy entry points must name a package root"
                )
            },
            entry_point=context.entry_point_name,
        )
    try:
        compiler_module = importlib.import_module(package_root + ".compiler")
        driver_module = importlib.import_module(package_root + ".driver")
    except ImportError as exc:
        raise api.BackendPluginLoadError(
            "Unable to import Triton 3.6 Legacy compiler/driver modules",
            entry_point=context.entry_point_name,
            field="legacy_module_import",
            expected=(f"{package_root}.compiler and {package_root}.driver"),
            actual=_stable_exception_actual(exc),
            remediation=(
                "Install the complete Legacy package root, or migrate the "
                "backend to a Protocol 1.0 Manifest."
            ),
        ) from exc
    compiler_candidates = _module_runtime_candidates(
        compiler_module,
        BaseBackend,
        field="compiler_cls",
        entry_point=context.entry_point_name,
    )
    driver_candidates = _module_runtime_candidates(
        driver_module,
        DriverBase,
        field="driver_cls",
        entry_point=context.entry_point_name,
    )
    pairs = []
    validator = _get_runtime_pair_validator()
    for compiler_cls, compiler_rank in compiler_candidates:
        for driver_cls, driver_rank in driver_candidates:
            result = validator(
                api.RuntimePairValidationContext(
                    compiler_cls=compiler_cls,
                    driver_cls=driver_cls,
                )
            )
            if not result.issues:
                pairs.append((compiler_rank + driver_rank, compiler_cls, driver_cls))
    if pairs:
        best_rank = min(rank for rank, _compiler, _driver in pairs)
        best_pairs = tuple(pair for pair in pairs if pair[0] == best_rank)
    else:
        best_pairs = ()
    if len(best_pairs) == 1:
        _rank, compiler_cls, driver_cls = best_pairs[0]
    elif not pairs and (len(compiler_candidates) == 1 and len(driver_candidates) == 1):
        # Preserve the exact single pair so Registry F6 produces its full,
        # version-derived interface diagnostics and rejects before probing.
        compiler_cls = compiler_candidates[0][0]
        driver_cls = driver_candidates[0][0]
    else:
        raise api.BackendPluginInterfaceError(
            (),
            field_errors={
                "compiler_cls,driver_cls": (
                    "expected one F6-valid runtime pair; found "
                    f"{len(best_pairs)} best-ranked pairs across "
                    f"{len(compiler_candidates)} compiler "
                    f"and {len(driver_candidates)} driver candidates"
                )
            },
            entry_point=context.entry_point_name,
        )
    return api.LegacyRuntimePair(compiler_cls=compiler_cls, driver_cls=driver_cls)


def _get_legacy_runtime_pair_materializer():
    global _legacy_runtime_pair_materializer
    with _legacy_materializer_lock:
        if _legacy_runtime_pair_materializer is None:
            _legacy_runtime_pair_materializer = _materialize_v36_legacy_runtime_pair
        return _legacy_runtime_pair_materializer


def register_legacy_runtime_pair_materializer(registry=None) -> None:
    """Register the v3.6 package-root Legacy materialization convention."""
    target_registry = _registry() if registry is None else registry
    target_registry.register_legacy_runtime_pair_materializer(
        _LEGACY_RUNTIME_PAIR_MATERIALIZER_ID,
        _get_legacy_runtime_pair_materializer(),
    )


def _enum_value(value):
    return getattr(value, "value", value)


def _is_selected_record(record) -> bool:
    operational_manifest = (
        _registry_api().operational_record_manifest_error(record) is None
    )
    return (
        operational_manifest
        and _enum_value(record.state) in {"selected", "active"}
        and bool(record.selected_targets)
    )


def _prune_registry_backends(registry) -> None:
    api = _registry_api()
    with _backend_cache_lock:
        snapshot = tuple(backends.items())
    removals = []
    for name, backend in snapshot:
        if type(backend) is not Backend or backend.record_id is None:
            continue
        try:
            record = registry.validate(backend.record_id)
        except api.BackendPluginError:
            removals.append((name, backend))
            continue
        if not _is_selected_record(record):
            removals.append((name, backend))
    with _backend_cache_lock:
        for name, backend in removals:
            if backends.get(name) is backend:
                backends.pop(name, None)


def _cache_decision(decision) -> Backend:
    registry = _registry()
    api = _registry_api()
    generation = registry.generation
    lifecycle_epoch = registry.lifecycle_epoch
    try:
        record = registry.validate(decision.record_id)
    except api.BackendPluginManifestError:
        raise
    except api.BackendPluginError as exc:
        raise api.BackendPluginLifecycleError(
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
    if record.manifest is not None:
        api.require_operational_manifest(record.manifest)
    if (
        not _is_selected_record(record)
        or decision.target not in record.selected_targets
    ):
        raise api.BackendPluginLifecycleError(
            "Backend selection changed before consumption",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="selected_targets",
            expected=decision.target,
            actual=", ".join(record.selected_targets) or "<none>",
            remediation=(
                "Retry backend resolution using the current Registry selection."
            ),
        )
    backend = Backend(
        compiler=record.compiler_cls,
        driver=record.driver_cls,
        record_id=record.record_id,
        plugin_id=record.plugin_id,
        entry_point_name=record.entry_point_name,
        registry_generation=generation,
        registry_lifecycle_epoch=lifecycle_epoch,
    )
    with _backend_cache_lock:
        backends[record.entry_point_name] = backend
    _prune_registry_backends(registry)
    current_generation = registry.generation
    if current_generation != generation:
        current_lifecycle_epoch = registry.lifecycle_epoch
        current_decision = registry.get_selection(decision.target)
        if (
            current_lifecycle_epoch == lifecycle_epoch
            and current_decision is not None
            and current_decision.record_id == record.record_id
            and current_decision.record.compiler_cls is backend.compiler
            and current_decision.record.driver_cls is backend.driver
        ):
            return replace(
                backend,
                registry_generation=current_generation,
            )
        with _backend_cache_lock:
            cached = backends.get(record.entry_point_name)
            if cached is backend:
                backends.pop(record.entry_point_name, None)
        raise api.BackendPluginLifecycleError(
            "Backend selection was invalidated during consumption",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="registry generation",
            expected=str(generation),
            actual=str(current_generation),
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
        error = getattr(record, "error", None)
        error_plugin_id = getattr(error, "plugin_id", None)
        if isinstance(error_plugin_id, str) and error_plugin_id:
            keys.add(error_plugin_id)
        if selector in keys:
            matches.append(record)
    if not matches:
        raise _selection_error(
            f"No backend plugin matches selector '{selector}'",
            field="backend_selector",
            expected="a Manifest plugin_id/record key or a Legacy record key",
            actual=selector,
            remediation=("Use a selector shown by BackendPluginRegistry diagnostics."),
        )
    unsupported = []
    for record in matches:
        error = _registry_api().operational_record_manifest_error(record)
        if error is not None:
            unsupported.append((record.record_id, error))
    unsupported.sort(key=lambda item: item[0])
    if unsupported:
        raise unsupported[0][1]
    if len(matches) != 1:
        record_ids = tuple(sorted(record.record_id for record in matches))
        raise _selection_error(
            f"Backend selector '{selector}' is ambiguous: " + ", ".join(record_ids),
            field="backend_selector",
            expected="one backend record",
            actual=", ".join(record_ids),
            remediation="Select one backend by its unique record_id.",
        )
    record = matches[0]
    if record.manifest is not None:
        _registry_api().require_operational_manifest(record.manifest)
    return record


def _bootstrap_target(record) -> str:
    if record.manifest is not None:
        _registry_api().require_operational_manifest(record.manifest)
        targets = tuple(sorted(record.manifest.targets))
        if targets:
            return targets[0]
    # Legacy records have no static target declaration.  This value is only a
    # selection cache key; W8 still requires an exact Legacy record selector.
    return record.entry_point_name


def _unsupported_record_claims_target(record, target_name) -> bool:
    manifest = getattr(record, "manifest", None)
    if manifest is not None:
        if type(manifest) is not _registry_api().BackendPluginManifest:
            return True
        targets = manifest.targets
        if type(targets) is not tuple or any(
            type(candidate) is not str for candidate in targets
        ):
            return True
        return type(target_name) is str and target_name in targets
    entry_point_name = getattr(record, "entry_point_name", None)
    return type(entry_point_name) is str and entry_point_name == target_name


def _operational_record_claims_target(record, target_name) -> bool:
    api = _registry_api()
    if api.operational_record_manifest_error(record) is not None:
        return False
    manifest = getattr(record, "manifest", None)
    if type(manifest) is not api.BackendPluginManifest:
        return False
    targets = manifest.targets
    if type(targets) is not tuple or any(
        type(candidate) is not str for candidate in targets
    ):
        return False
    return type(target_name) is str and target_name in targets


def _target_name(target) -> str:
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        return target.get("backend")
    return getattr(target, "backend", None)


def _manual_backends() -> Tuple[Backend, ...]:
    result = []
    with _backend_cache_lock:
        snapshot = tuple(backends.items())
    stable_items = tuple(
        sorted(
            ((name, backend) for name, backend in snapshot if type(name) is str),
            key=lambda item: item[0],
        )
    )
    for name, backend in stable_items:
        if type(backend) is not Backend:
            continue
        if backend.record_id is not None:
            continue
        if backend.entry_point_name is None:
            backend = replace(backend, entry_point_name=name)
        result.append(backend)
    return tuple(result)


def _reset_backend_cache() -> None:
    with _backend_cache_lock:
        for name, backend in tuple(backends.items()):
            if type(backend) is Backend and backend.record_id is not None:
                backends.pop(name, None)


def register_backend_reset_hook(callback) -> None:
    """Coordinate Registry reset with a Triton-side lazy cache."""
    _registry().register_reset_hook(callback)


def _discover_backends():
    """Discover static backend metadata without importing plugin code."""
    _registry().discover()
    return backends


def _kernel_capabilities(values) -> Tuple[str, ...]:
    api = _registry_api()
    if isinstance(values, (str, bytes)):
        raise api.BackendPluginSelectionError(
            "Kernel capability requirements must not be a string",
            field="kernel_required_capabilities",
            expected="an iterable of unique non-empty strings",
            actual=type(values).__name__,
            remediation="Pass capability names in a tuple or list.",
        )
    try:
        capabilities = tuple(values)
    except TypeError as exc:
        raise api.BackendPluginSelectionError(
            "Kernel capability requirements must be iterable",
            field="kernel_required_capabilities",
            expected="an iterable of unique non-empty strings",
            actual=type(values).__name__,
            remediation="Pass an empty tuple when Legacy fallback is allowed.",
        ) from exc
    if any(
        type(value) is not str or not value or value != value.strip()
        for value in capabilities
    ) or len(capabilities) != len(set(capabilities)):
        raise api.BackendPluginSelectionError(
            "Kernel capability requirements are invalid",
            field="kernel_required_capabilities",
            expected="unique non-empty strings without surrounding whitespace",
            actual=type(values).__name__,
            remediation="Normalize capability names before backend resolution.",
        )
    return tuple(sorted(capabilities))


def _legacy_capability_error(capabilities):
    return _registry_api().BackendPluginSelectionError(
        "Legacy backend fallback cannot satisfy kernel capabilities",
        field="kernel_required_capabilities",
        expected="no capability requirements for LEGACY_UNVERIFIED fallback",
        actual=", ".join(capabilities),
        remediation=(
            "Install a Manifest backend declaring these capabilities, or "
            "remove the requirements before selecting Legacy."
        ),
    )


def _run_legacy_resolution(operation, callback):
    api = _registry_api()
    stack = getattr(_legacy_resolution_local, "stack", ())
    if stack:
        raise api.BackendPluginLifecycleError(
            "Recursive Legacy backend resolution is not allowed",
            field="legacy_resolution",
            expected="one non-reentrant compiler/runtime resolution per thread",
            actual=operation,
            remediation=(
                "Do not call Triton backend resolution recursively from the "
                "same plugin callback thread. Cross-thread callers are "
                "coordinated by Registry leases and atomic publication."
            ),
        )
    _legacy_resolution_local.stack = (operation,)
    try:
        return callback()
    finally:
        _legacy_resolution_local.stack = stack


def _legacy_records(registry, *, selector=None):
    records = registry.list()
    if selector is not None:
        selected = _selector_record(records, selector)
        if _enum_value(selected.source) != "legacy":
            return ()
        return (selected,)
    return tuple(
        sorted(
            (
                record
                for record in records
                if _enum_value(record.source) == "legacy"
                and _enum_value(record.state)
                in {
                    "discovered",
                    "loaded",
                    "registered",
                    "selected",
                    "active",
                    "rejected",
                }
            ),
            key=lambda record: record.record_id,
        )
    )


def _published_legacy_records(registry):
    api = _registry_api()
    records = tuple(
        sorted(
            (
                record
                for record in registry.list()
                if _enum_value(record.source) == "legacy"
                and _enum_value(record.state) in {"selected", "active"}
                and record.selected_targets
            ),
            key=lambda record: record.record_id,
        )
    )
    if len(records) > 1:
        record_ids = tuple(record.record_id for record in records)
        raise api.BackendPluginConflictError(
            "Compiler and runtime cannot bind different Legacy records: "
            + ", ".join(record_ids),
            field="compiler_cls,driver_cls",
            expected="one process-wide governed Legacy runtime pair",
            actual=", ".join(record_ids),
            conflict_kind="legacy_runtime_selection",
            related_record_ids=record_ids,
            remediation=(
                "Reset Registry runtime state, then select one exact Legacy "
                "record for both compiler and driver consumption."
            ),
        )
    return records


def _published_governed_records(registry):
    return tuple(
        sorted(
            (
                record
                for record in registry.list()
                if _enum_value(record.state) in {"selected", "active"}
                and record.selected_targets
            ),
            key=lambda record: record.record_id,
        )
    )


def _backend_from_lease(lease) -> Backend:
    return Backend(
        compiler=lease.compiler_cls,
        driver=lease.driver_cls,
        record_id=lease.record_id,
        plugin_id=None,
        entry_point_name=lease.entry_point_name,
        registry_generation=lease.generation,
        registry_lifecycle_epoch=lease.lifecycle_epoch,
    )


def _runtime_backend_proposal(leases, *, forced=False):
    leases = tuple(sorted(leases, key=lambda lease: lease.target))
    if not leases:
        raise ValueError("runtime backend proposal requires at least one lease")
    first = leases[0]
    if any(
        lease.record_id != first.record_id
        or lease.compiler_cls is not first.compiler_cls
        or lease.driver_cls is not first.driver_cls
        for lease in leases[1:]
    ):
        raise _registry_api().BackendPluginLifecycleError(
            "Manifest runtime proposal mixed different records",
            field="compiler_cls,driver_cls",
            expected=first.record_id,
            actual=", ".join(sorted({lease.record_id for lease in leases})),
            remediation="Prepare one exact Manifest record per runtime proposal.",
        )
    return _RuntimeBackendProposal(
        backend=Backend(
            compiler=first.compiler_cls,
            driver=first.driver_cls,
            record_id=first.record_id,
            plugin_id=first.plugin_id,
            entry_point_name=first.entry_point_name,
        ),
        leases=leases,
        forced=forced,
    )


def _probe_error(backend, field, message, expected, exc):
    api = _registry_api()
    if isinstance(exc, api.BackendPluginError) and not isinstance(
        exc,
        api.BackendPluginNoCandidateError,
    ):
        return exc
    return api.BackendPluginLifecycleError(
        message,
        plugin_id=backend.plugin_id,
        entry_point=backend.entry_point_name,
        field=field,
        expected=expected,
        actual=_stable_exception_actual(exc),
        remediation=(
            f"Fix {field} without changing Registry state, then reset and retry."
        ),
    )


def _probe_compiler(backend, target) -> bool:
    try:
        supports_target = getattr(backend.compiler, "supports_target", None)
    except Exception as exc:
        raise _probe_error(
            backend,
            "compiler_cls.supports_target",
            "Legacy compiler target predicate lookup failed",
            "a callable target predicate after F6 validation",
            exc,
        ) from exc
    if not callable(supports_target):
        raise _selection_error(
            "Legacy compiler has no callable supports_target()",
            field="compiler_cls.supports_target",
            expected="a callable target predicate after F6 validation",
            actual=type(supports_target).__name__,
            remediation="Implement the current BaseBackend abstract surface.",
        )
    try:
        result = supports_target(target)
    except Exception as exc:
        raise _probe_error(
            backend,
            "compiler_cls.supports_target",
            "Legacy compiler target probe failed",
            "a boolean target predicate result",
            exc,
        ) from exc
    if type(result) is not bool:
        raise _selection_error(
            "Legacy compiler target probe returned a non-boolean value",
            field="compiler_cls.supports_target",
            expected="bool",
            actual=type(result).__name__,
            remediation="Return True or False from supports_target(target).",
        )
    return result


def _construct_legacy_compiler(backend, target):
    try:
        compiler = backend.compiler(target)
    except Exception as exc:
        raise _probe_error(
            backend,
            "compiler_cls.__init__",
            "Legacy compiler construction failed",
            "a successfully constructed compiler",
            exc,
        ) from exc
    if type(compiler) is not backend.compiler:
        raise _selection_error(
            "Legacy compiler constructor returned a foreign runtime object",
            field="compiler_cls.__new__",
            expected=_safe_class_name(backend.compiler),
            actual=_safe_class_name(type(compiler)),
            remediation="Return an exact instance of the F6-validated class.",
        )
    return compiler


def _legacy_compiler_resolution(
    target,
    *,
    kernel_required_capabilities=(),
    selector=None,
    construct=False,
):
    api = _registry_api()
    capabilities = _kernel_capabilities(kernel_required_capabilities)
    if capabilities:
        raise _legacy_capability_error(capabilities)
    registry = _registry()

    def resolve():
        operation_lifecycle_epoch = registry.lifecycle_epoch
        existing = registry.get_selection(_target_name(target))
        if existing is not None:
            if selector is not None:
                selected = _selector_record(registry.list(), selector)
                if selected.record_id != existing.record_id:
                    raise api.BackendPluginConflictError(
                        "Explicit Legacy selector conflicts with the existing "
                        "target selection",
                        field="backend_selector",
                        expected=existing.record_id,
                        actual=selected.record_id,
                        conflict_kind="legacy_runtime_selection",
                        related_record_ids=(
                            existing.record_id,
                            selected.record_id,
                        ),
                        remediation=(
                            "Reset before selecting a different Legacy record."
                        ),
                    )
            backend = _cache_decision(existing)
            if not _probe_compiler(backend, target):
                raise _selection_error(
                    "Selected Legacy compiler no longer supports the target",
                    field="compiler_cls.supports_target",
                    expected=str(_target_name(target)),
                    actual="False",
                    remediation="Reset before changing Legacy target support.",
                )
            _require_legacy_resolution_lifecycle_epoch(
                registry,
                operation_lifecycle_epoch,
                backend=backend,
            )
            if not construct:
                return backend
            compiler = _construct_legacy_compiler(backend, target)
            _require_legacy_resolution_lifecycle_epoch(
                registry,
                operation_lifecycle_epoch,
                backend=backend,
            )
            return compiler

        published = _published_legacy_records(registry)
        if published:
            records = published
            if selector is not None:
                selected = _selector_record(registry.list(), selector)
                if selected.record_id != records[0].record_id:
                    raise api.BackendPluginConflictError(
                        "Explicit Legacy selector conflicts with the "
                        "published runtime pair",
                        field="backend_selector",
                        expected=records[0].record_id,
                        actual=selected.record_id,
                        conflict_kind="legacy_runtime_selection",
                        related_record_ids=(
                            records[0].record_id,
                            selected.record_id,
                        ),
                        remediation=(
                            "Reset before selecting a different Legacy record."
                        ),
                    )
        else:
            records = _legacy_records(registry, selector=selector)
        if not records:
            raise api.BackendPluginNoCandidateError(
                "No governed Legacy backend record is available",
                field="compiler_cls.supports_target",
                expected="one F6-valid Legacy compiler supporting the target",
                actual="0",
                remediation=(
                    "Install a Triton 3.6 Legacy distribution or a compatible "
                    "Manifest backend."
                ),
            )
        matches = []
        failures = []
        for record in records:
            try:
                _require_legacy_resolution_lifecycle_epoch(
                    registry,
                    operation_lifecycle_epoch,
                )
                lease = registry.prepare_legacy_selection(record.record_id, target)
                backend = _backend_from_lease(lease)
                if lease.lifecycle_epoch != operation_lifecycle_epoch:
                    raise api.BackendPluginLifecycleError(
                        "Legacy compiler lease crossed a Registry reset boundary",
                        entry_point=backend.entry_point_name,
                        field="registry lifecycle epoch",
                        expected=str(operation_lifecycle_epoch),
                        actual=str(lease.lifecycle_epoch),
                        remediation="Retry the complete compiler candidate scan.",
                    )
                if _probe_compiler(backend, target):
                    matches.append((lease, backend))
                _require_legacy_resolution_lifecycle_epoch(
                    registry,
                    operation_lifecycle_epoch,
                    backend=backend,
                )
            except api.BackendPluginError as exc:
                failures.append((record.record_id, exc))
        failures.sort(key=lambda item: item[0])
        if failures:
            raise failures[0][1]
        if not matches:
            raise api.BackendPluginNoCandidateError(
                "No Legacy compiler supports the requested target",
                field="compiler_cls.supports_target",
                expected="one True result",
                actual="0",
                remediation="Install a Legacy backend supporting this GPUTarget.",
            )
        if len(matches) != 1:
            record_ids = tuple(sorted(lease.record_id for lease, _ in matches))
            raise api.BackendPluginConflictError(
                "Multiple Legacy compilers support the requested target: "
                + ", ".join(record_ids),
                field="compiler_cls.supports_target",
                expected="one matching Legacy record",
                actual=", ".join(record_ids),
                conflict_kind="legacy_runtime_selection",
                related_record_ids=record_ids,
                remediation=(
                    "Remove the overlap or select one exact Legacy record with "
                    f"{api.BACKEND_SELECTOR_ENV}."
                ),
            )
        lease, backend = matches[0]
        _require_legacy_resolution_lifecycle_epoch(
            registry,
            operation_lifecycle_epoch,
            backend=backend,
        )
        compiler = _construct_legacy_compiler(backend, target) if construct else None
        _require_legacy_resolution_lifecycle_epoch(
            registry,
            operation_lifecycle_epoch,
            backend=backend,
        )
        decision = registry.commit_legacy_selection(lease)
        published = _cache_decision(decision)
        return compiler if construct else published

    return _run_legacy_resolution("compiler", resolve)


def resolve_legacy_compiler_for_target(
    target,
    *,
    kernel_required_capabilities=(),
) -> Backend:
    """Resolve a compiler with Manifest-first governed Legacy fallback."""
    api = _registry_api()
    capabilities = _kernel_capabilities(kernel_required_capabilities)
    try:
        return _selected_backend(target, capabilities)
    except api.BackendPluginNoCandidateError as manifest_miss:
        if capabilities:
            raise _legacy_capability_error(capabilities) from manifest_miss
        return _legacy_compiler_resolution(target)


def _probe_driver_active(backend) -> bool:
    try:
        is_active = getattr(backend.driver, "is_active", None)
    except Exception as exc:
        raise _probe_error(
            backend,
            "driver_cls.is_active",
            "Legacy driver active predicate lookup failed",
            "a callable active-device predicate after F6 validation",
            exc,
        ) from exc
    if not callable(is_active):
        raise _selection_error(
            "Legacy driver has no callable is_active()",
            field="driver_cls.is_active",
            expected="a callable active-device predicate after F6 validation",
            actual=type(is_active).__name__,
            remediation="Implement the current DriverBase abstract surface.",
        )
    try:
        result = is_active()
    except Exception as exc:
        raise _probe_error(
            backend,
            "driver_cls.is_active",
            "Legacy driver active probe failed",
            "a boolean active-device predicate result",
            exc,
        ) from exc
    if type(result) is not bool:
        raise _selection_error(
            "Legacy driver active probe returned a non-boolean value",
            field="driver_cls.is_active",
            expected="bool",
            actual=type(result).__name__,
            remediation="Return True or False from driver_cls.is_active().",
        )
    return result


def _construct_legacy_driver(backend):
    try:
        driver = backend.driver()
    except Exception as exc:
        raise _probe_error(
            backend,
            "driver_cls.__init__",
            "Legacy driver construction failed",
            "a successfully constructed driver",
            exc,
        ) from exc
    if type(driver) is not backend.driver:
        raise _selection_error(
            "Legacy driver constructor returned a foreign runtime object",
            field="driver_cls.__new__",
            expected=_safe_class_name(backend.driver),
            actual=_safe_class_name(type(driver)),
            remediation="Return an exact instance of the F6-validated class.",
        )
    return driver


def _legacy_driver_target(backend, driver):
    try:
        get_current_target = getattr(driver, "get_current_target", None)
    except Exception as exc:
        raise _probe_error(
            backend,
            "driver_cls.get_current_target",
            "Legacy driver current-target accessor lookup failed",
            "a callable target accessor after F6 validation",
            exc,
        ) from exc
    if not callable(get_current_target):
        raise _selection_error(
            "Legacy driver has no callable get_current_target()",
            field="driver_cls.get_current_target",
            expected="a callable target accessor after F6 validation",
            actual=type(get_current_target).__name__,
            remediation="Implement the current DriverBase abstract surface.",
        )
    try:
        target = get_current_target()
    except Exception as exc:
        raise _probe_error(
            backend,
            "driver_cls.get_current_target",
            "Legacy driver current-target lookup failed",
            "the current Triton 3.6 GPUTarget",
            exc,
        ) from exc
    if type(target) is not GPUTarget:
        raise _selection_error(
            "Legacy driver returned an invalid current target",
            field="driver_cls.get_current_target",
            expected="an exact Triton 3.6 GPUTarget",
            actual=type(target).__name__,
            remediation=(
                "Return triton.backends.compiler.GPUTarget from the Legacy "
                "driver's get_current_target()."
            ),
        )
    return target


def _legacy_resolution_stale_error(backend, lease, registry):
    return _registry_api().BackendPluginLifecycleError(
        "Legacy probe result was invalidated before publication",
        plugin_id=backend.plugin_id,
        entry_point=backend.entry_point_name,
        field="registry generation",
        expected=str(lease.generation),
        actual=str(registry.generation),
        remediation=(
            "Discard the in-flight plugin result and retry after Registry "
            "reset or selection completes."
        ),
    )


def _require_legacy_resolution_generation(
    registry,
    expected_generation,
    *,
    backend=None,
):
    actual_generation = registry.generation
    if actual_generation != expected_generation:
        raise _registry_api().BackendPluginLifecycleError(
            "Legacy resolution crossed a Registry generation boundary",
            plugin_id=getattr(backend, "plugin_id", None),
            entry_point=getattr(backend, "entry_point_name", None),
            field="registry generation",
            expected=str(expected_generation),
            actual=str(actual_generation),
            remediation=(
                "Discard the entire in-flight candidate scan and retry after "
                "Registry reset or selection completes."
            ),
        )


def _require_legacy_resolution_lifecycle_epoch(
    registry,
    expected_lifecycle_epoch,
    *,
    backend=None,
):
    actual_lifecycle_epoch = registry.lifecycle_epoch
    if actual_lifecycle_epoch != expected_lifecycle_epoch:
        raise _registry_api().BackendPluginLifecycleError(
            "Backend resolution crossed a Registry reset boundary",
            plugin_id=getattr(backend, "plugin_id", None),
            entry_point=getattr(backend, "entry_point_name", None),
            field="registry lifecycle epoch",
            expected=str(expected_lifecycle_epoch),
            actual=str(actual_lifecycle_epoch),
            remediation=(
                "Discard the in-flight plugin result and retry after Registry "
                "reset completes."
            ),
        )


def _legacy_runtime_resolution(
    *,
    kernel_required_capabilities=(),
    expected_generation=None,
    expected_lifecycle_epoch=None,
    inactive_manifest_record_ids=(),
    return_backend=False,
):
    api = _registry_api()
    capabilities = _kernel_capabilities(kernel_required_capabilities)
    if capabilities:
        raise _legacy_capability_error(capabilities)
    registry = _registry()

    def resolve():
        operation_lifecycle_epoch = (
            registry.lifecycle_epoch
            if expected_lifecycle_epoch is None
            else expected_lifecycle_epoch
        )
        _require_legacy_resolution_lifecycle_epoch(
            registry,
            operation_lifecycle_epoch,
        )
        if (
            expected_generation is not None
            and registry.generation != expected_generation
        ):
            raise api.BackendPluginLifecycleError(
                "Manifest runtime fallback lost its selection snapshot",
                field="registry generation",
                expected=str(expected_generation),
                actual=str(registry.generation),
                remediation=(
                    "Retry runtime resolution so any newly published Manifest "
                    "selection remains authoritative."
                ),
            )
        manifest_failures = tuple(
            sorted(
                (
                    (record.record_id, record.error)
                    for record in registry.list()
                    if (
                        _enum_value(record.source) != "legacy"
                        and record.error is not None
                    )
                ),
                key=lambda item: item[0],
            )
        )
        if manifest_failures:
            # Runtime has no target until a driver is constructed.  A rejected
            # Manifest may therefore own the eventual hardware target; its
            # scope cannot safely be proven irrelevant before importing
            # Legacy, so fallback is deliberately fail-closed.
            raise manifest_failures[0][1]
        selector = os.environ.get(api.BACKEND_SELECTOR_ENV)
        if selector == "":
            selector = None
        published = _published_legacy_records(registry)
        if published:
            records = published
            if selector is not None:
                selected = _selector_record(registry.list(), selector)
                if selected.record_id != records[0].record_id:
                    raise api.BackendPluginConflictError(
                        "Explicit Legacy selector conflicts with the "
                        "published compiler/runtime pair",
                        field="backend_selector",
                        expected=records[0].record_id,
                        actual=selected.record_id,
                        conflict_kind="legacy_runtime_selection",
                        related_record_ids=(
                            records[0].record_id,
                            selected.record_id,
                        ),
                        remediation=(
                            "Reset before selecting a different Legacy record."
                        ),
                    )
        else:
            records = _legacy_records(registry, selector=selector)
        if not records:
            raise api.BackendPluginNoCandidateError(
                "No governed Legacy backend driver is available",
                field="driver_cls.is_active",
                expected="one F6-valid active Legacy driver",
                actual="0",
                remediation=(
                    "Install a Triton 3.6 Legacy distribution or a compatible "
                    "Manifest backend."
                ),
            )

        matches = []
        failures = []
        for record in records:
            try:
                _require_legacy_resolution_lifecycle_epoch(
                    registry,
                    operation_lifecycle_epoch,
                )
                published_before = _published_governed_records(registry)
                if published_before and any(
                    owner.record_id != record.record_id for owner in published_before
                ):
                    owner_ids = tuple(owner.record_id for owner in published_before)
                    raise api.BackendPluginConflictError(
                        "A compiler selection was published during Legacy runtime probing",
                        field="compiler_cls,driver_cls",
                        expected="reuse the published exact runtime record",
                        actual=", ".join(owner_ids),
                        conflict_kind="legacy_runtime_selection",
                        related_record_ids=owner_ids,
                        remediation="Retry runtime resolution using the published record.",
                    )
                lease = registry.prepare_legacy_selection(
                    record.record_id,
                    record.entry_point_name,
                )
                backend = _backend_from_lease(lease)
                if lease.lifecycle_epoch != operation_lifecycle_epoch:
                    raise _registry_api().BackendPluginLifecycleError(
                        "Legacy runtime lease crossed a Registry reset boundary",
                        entry_point=backend.entry_point_name,
                        field="registry lifecycle epoch",
                        expected=str(operation_lifecycle_epoch),
                        actual=str(lease.lifecycle_epoch),
                        remediation="Retry the complete runtime candidate scan.",
                    )
                active = _probe_driver_active(backend)
                if active:
                    matches.append((lease, backend))
                _require_legacy_resolution_lifecycle_epoch(
                    registry,
                    operation_lifecycle_epoch,
                    backend=backend,
                )
                published_after = _published_governed_records(registry)
                if published_after:
                    owner_ids = tuple(owner.record_id for owner in published_after)
                    if any(
                        owner.record_id != record.record_id for owner in published_after
                    ):
                        raise api.BackendPluginConflictError(
                            "A compiler selection was published during Legacy runtime probing",
                            field="compiler_cls,driver_cls",
                            expected="reuse the published exact runtime record",
                            actual=", ".join(owner_ids),
                            conflict_kind="legacy_runtime_selection",
                            related_record_ids=owner_ids,
                            remediation="Retry runtime resolution using the published record.",
                        )
                    if not active:
                        raise api.BackendPluginSelectionError(
                            "The published Legacy compiler record has no active driver",
                            entry_point=backend.entry_point_name,
                            field="driver_cls.is_active",
                            expected="True from the published exact runtime pair",
                            actual="False",
                            remediation=(
                                "Reset before changing the process-wide Legacy "
                                "runtime pair."
                            ),
                        )
                    break
            except api.BackendPluginError as exc:
                if _published_governed_records(registry):
                    raise
                failures.append((record.record_id, exc))
        failures.sort(key=lambda item: item[0])
        if failures:
            raise failures[0][1]
        if not matches:
            raise api.BackendPluginSelectionError(
                "No governed Legacy backend driver is active",
                field="driver_cls.is_active",
                expected="one True result",
                actual="0",
                remediation=(
                    "Install or select the Legacy backend for the active device."
                ),
            )
        if len(matches) != 1:
            record_ids = tuple(sorted(lease.record_id for lease, _ in matches))
            raise api.BackendPluginConflictError(
                "Multiple governed Legacy backend drivers are active: "
                + ", ".join(record_ids),
                field="driver_cls.is_active",
                expected="one active Legacy record",
                actual=", ".join(record_ids),
                conflict_kind="legacy_runtime_selection",
                related_record_ids=record_ids,
                remediation=(
                    "Use the exact Legacy selector or configure hardware so "
                    "only one driver reports active."
                ),
            )

        lease, backend = matches[0]
        _require_legacy_resolution_lifecycle_epoch(
            registry,
            operation_lifecycle_epoch,
            backend=backend,
        )
        driver = _construct_legacy_driver(backend)
        _require_legacy_resolution_lifecycle_epoch(
            registry,
            operation_lifecycle_epoch,
            backend=backend,
        )
        target = _legacy_driver_target(backend, driver)
        _require_legacy_resolution_lifecycle_epoch(
            registry,
            operation_lifecycle_epoch,
            backend=backend,
        )
        if not _probe_compiler(backend, target):
            raise api.BackendPluginSelectionError(
                "Active Legacy driver and compiler disagree on the target",
                field="compiler_cls.supports_target",
                expected=target.backend,
                actual="False",
                remediation=(
                    "Return a current GPUTarget supported by the compiler from "
                    "the same Legacy entry point."
                ),
            )
        decision = registry.commit_legacy_selection(
            lease,
            target=target,
            activate=True,
            inactive_manifest_record_ids=inactive_manifest_record_ids,
        )
        published_backend = _cache_decision(decision)
        if (
            published_backend.compiler is not backend.compiler
            or published_backend.driver is not backend.driver
        ):
            raise api.BackendPluginLifecycleError(
                "Published Legacy runtime pair changed after probing",
                entry_point=backend.entry_point_name,
                field="compiler_cls,driver_cls",
                expected=lease.record_id,
                actual=published_backend.record_id,
                remediation="Reset and retry exact-record Legacy resolution.",
            )
        return (driver, published_backend) if return_backend else driver

    return _run_legacy_resolution("runtime", resolve)


def _resolve_active_legacy_driver_after_manifest_miss(
    *,
    kernel_required_capabilities=(),
    expected_generation=None,
    expected_lifecycle_epoch=None,
    inactive_manifest_record_ids=(),
    return_backend=False,
):
    return _legacy_runtime_resolution(
        kernel_required_capabilities=kernel_required_capabilities,
        expected_generation=expected_generation,
        expected_lifecycle_epoch=expected_lifecycle_epoch,
        inactive_manifest_record_ids=inactive_manifest_record_ids,
        return_backend=return_backend,
    )


def _backend_registry_generation() -> int:
    return _registry().generation


def _backend_registry_lifecycle_epoch() -> int:
    return _registry().lifecycle_epoch


def _driver_matches_runtime_class(driver, driver_cls) -> bool:
    return type(driver) is driver_cls


def _legacy_explicit_driver_resolution(driver, target):
    api = _registry_api()
    if type(target) is not GPUTarget:
        raise api.BackendPluginSelectionError(
            "Explicit Legacy driver returned an invalid current target",
            field="driver.get_current_target",
            expected="an exact Triton 3.6 GPUTarget",
            actual=type(target).__name__,
            remediation="Return the current Triton 3.6 GPUTarget.",
        )
    registry = _registry()

    def resolve():
        operation_generation = registry.generation
        selector = os.environ.get(api.BACKEND_SELECTOR_ENV)
        if selector == "":
            selector = None
        published = _published_legacy_records(registry)
        records = (
            published if published else _legacy_records(registry, selector=selector)
        )
        if not records:
            raise api.BackendPluginNoCandidateError(
                "No governed Legacy record matches the explicit driver",
                field="driver_cls",
                expected="one F6-valid Legacy runtime pair",
                actual="0",
                remediation="Install or select the driver's Legacy distribution.",
            )
        matches = []
        failures = []
        for record in records:
            try:
                _require_legacy_resolution_generation(
                    registry,
                    operation_generation,
                )
                lease = registry.prepare_legacy_selection(record.record_id, target)
                backend = _backend_from_lease(lease)
                if lease.generation != operation_generation:
                    raise api.BackendPluginLifecycleError(
                        "Explicit Legacy driver crossed a Registry generation",
                        entry_point=backend.entry_point_name,
                        field="registry generation",
                        expected=str(operation_generation),
                        actual=str(lease.generation),
                        remediation="Retry explicit activation after reset.",
                    )
                if _driver_matches_runtime_class(driver, backend.driver) and (
                    _probe_compiler(backend, target)
                ):
                    matches.append((lease, backend))
                _require_legacy_resolution_generation(
                    registry,
                    operation_generation,
                    backend=backend,
                )
            except api.BackendPluginError as exc:
                failures.append((record.record_id, exc))
        failures.sort(key=lambda item: item[0])
        if failures:
            raise failures[0][1]
        if not matches:
            raise api.BackendPluginSelectionError(
                "Explicit driver does not match a target-compatible Legacy pair",
                field="compiler_cls,driver_cls",
                expected="one exact-record runtime pair",
                actual="0",
                remediation=(
                    "Activate the driver class supplied by the Legacy record "
                    "whose compiler supports its current target."
                ),
            )
        if len(matches) != 1:
            record_ids = tuple(sorted(lease.record_id for lease, _ in matches))
            raise api.BackendPluginConflictError(
                "Explicit driver matches multiple Legacy records: "
                + ", ".join(record_ids),
                field="driver_cls",
                expected="one exact Legacy record",
                actual=", ".join(record_ids),
                conflict_kind="legacy_runtime_selection",
                related_record_ids=record_ids,
                remediation="Remove duplicate Legacy entry-point registrations.",
            )
        lease, backend = matches[0]
        decision = registry.commit_legacy_selection(
            lease,
            target=target,
            activate=True,
        )
        return _cache_decision(decision)

    return _run_legacy_resolution("explicit-runtime", resolve)


def _prepare_explicit_driver_match(
    driver,
    *,
    expected_generation,
) -> _ExplicitDriverMatch:
    """F6-gate exact Registry classes before reading a supplied driver."""
    api = _registry_api()
    registry = _registry()
    records = registry.validate()
    manifest_failures = tuple(
        sorted(
            (record.record_id, record.error)
            for record in records
            if record.error is not None and _enum_value(record.source) != "legacy"
        )
    )
    if manifest_failures:
        raise manifest_failures[0][1]
    selector = os.environ.get(api.BACKEND_SELECTOR_ENV)
    if selector == "":
        selector = None
    published = tuple(
        record
        for record in records
        if _enum_value(record.state) in {"selected", "active"}
        and record.selected_targets
    )
    operation_generation = expected_generation
    _require_legacy_resolution_generation(
        registry,
        operation_generation,
    )

    def match_records(candidates):
        stage_matches = []
        stage_failures = []
        for record in candidates:
            try:
                _require_legacy_resolution_generation(
                    registry,
                    operation_generation,
                )
                if _enum_value(record.source) == "legacy":
                    lease = registry.prepare_legacy_selection(
                        record.record_id,
                        record.entry_point_name,
                    )
                    backend = _backend_from_lease(lease)
                    source = "legacy"
                else:
                    target = record.manifest.targets[0]
                    lease = registry.prepare_runtime_selection(
                        target,
                        explicit_selector=record.record_id,
                        environment=dict(os.environ),
                    )
                    backend = Backend(
                        compiler=lease.compiler_cls,
                        driver=lease.driver_cls,
                        record_id=lease.record_id,
                        plugin_id=lease.plugin_id,
                        entry_point_name=lease.entry_point_name,
                    )
                    source = "manifest"
                if lease.generation != operation_generation:
                    raise api.BackendPluginLifecycleError(
                        "Explicit driver preparation crossed a Registry generation",
                        plugin_id=backend.plugin_id,
                        entry_point=backend.entry_point_name,
                        field="registry generation",
                        expected=str(operation_generation),
                        actual=str(lease.generation),
                        remediation="Retry explicit activation after reset.",
                    )
                if _driver_matches_runtime_class(driver, backend.driver):
                    stage_matches.append(
                        _ExplicitDriverMatch(
                            backend=backend,
                            source=source,
                            lease=lease,
                        )
                    )
            except api.BackendPluginError as exc:
                stage_failures.append((record.record_id, exc))
        if stage_failures:
            raise sorted(stage_failures, key=lambda item: item[0])[0][1]
        return stage_matches

    if selector is not None:
        selected_record = _selector_record(records, selector)
        matches = match_records((selected_record,))
    elif published:
        matches = match_records(
            tuple(sorted(published, key=lambda record: record.record_id))
        )
    else:
        # Manifest is a strict first stage.  A matching Manifest driver must
        # never import Legacy siblings, and an invalid Manifest pair is not
        # permission to downgrade.
        manifest_records = tuple(
            sorted(
                (
                    record
                    for record in records
                    if _enum_value(record.source) != "legacy"
                ),
                key=lambda record: record.record_id,
            )
        )
        matches = match_records(manifest_records)
        if not matches:
            legacy_records = tuple(
                sorted(
                    (
                        record
                        for record in records
                        if _enum_value(record.source) == "legacy"
                    ),
                    key=lambda record: record.record_id,
                )
            )
            matches = match_records(legacy_records)

    if not matches and selector is None and not published and not records:
        for backend in _manual_backends():
            if type(backend) is Backend and _driver_matches_runtime_class(
                driver, backend.driver
            ):
                matches.append(
                    _ExplicitDriverMatch(
                        backend=backend,
                        source="manual",
                    )
                )
    if len(matches) != 1:
        raise api.BackendPluginSelectionError(
            "Explicit driver does not identify one F6-validated runtime pair",
            field="driver_cls",
            expected="one exact recorded driver class",
            actual=str(len(matches)),
            remediation=(
                "Use the exact driver class from one Registry record or one "
                "unambiguous manual backend."
            ),
        )
    return matches[0]


def activate_explicit_driver(
    driver,
    *,
    expected_generation=None,
    expected_lifecycle_epoch=None,
) -> Backend:
    """F6-gate, probe, and atomically publish one supplied driver."""
    api = _registry_api()
    registry = _registry()
    operation_generation = (
        registry.generation if expected_generation is None else expected_generation
    )
    operation_lifecycle_epoch = (
        registry.lifecycle_epoch
        if expected_lifecycle_epoch is None
        else expected_lifecycle_epoch
    )
    if registry.lifecycle_epoch != operation_lifecycle_epoch:
        raise api.BackendPluginLifecycleError(
            "Explicit driver activation crossed a Registry reset boundary",
            field="registry lifecycle epoch",
            expected=str(operation_lifecycle_epoch),
            actual=str(registry.lifecycle_epoch),
            remediation="Retry explicit activation after reset completes.",
        )
    match = _prepare_explicit_driver_match(
        driver,
        expected_generation=operation_generation,
    )
    backend = match.backend
    _require_legacy_resolution_generation(
        registry,
        operation_generation,
        backend=backend,
    )
    target = _legacy_driver_target(backend, driver)
    _require_legacy_resolution_generation(
        registry,
        operation_generation,
        backend=backend,
    )

    if match.source == "manifest":
        lease = match.lease
        if lease.target != target.backend:
            lease = registry.prepare_runtime_selection(
                target,
                explicit_selector=backend.record_id,
                environment=dict(os.environ),
            )
        if lease.generation != operation_generation:
            raise api.BackendPluginLifecycleError(
                "Explicit Manifest driver crossed a Registry generation",
                plugin_id=backend.plugin_id,
                entry_point=backend.entry_point_name,
                field="registry generation",
                expected=str(operation_generation),
                actual=str(lease.generation),
                remediation="Retry explicit activation after reset.",
            )
        if not _probe_compiler(backend, target):
            raise api.BackendPluginSelectionError(
                "Explicit Manifest driver and compiler disagree on the target",
                plugin_id=backend.plugin_id,
                entry_point=backend.entry_point_name,
                field="compiler_cls.supports_target",
                expected=target.backend,
                actual="False",
                remediation="Use the exact same-record runtime pair.",
            )
        decision = registry.commit_runtime_selection(lease, activate=True)
        resolved = _cache_decision(decision)
        if resolved.registry_lifecycle_epoch != operation_lifecycle_epoch:
            raise api.BackendPluginLifecycleError(
                "Explicit Manifest activation crossed a Registry reset boundary",
                plugin_id=resolved.plugin_id,
                entry_point=resolved.entry_point_name,
                field="registry lifecycle epoch",
                expected=str(operation_lifecycle_epoch),
                actual=str(resolved.registry_lifecycle_epoch),
                remediation="Retry explicit activation after reset completes.",
            )
        return resolved

    if match.source == "legacy":
        selector = os.environ.get(api.BACKEND_SELECTOR_ENV)
        selected_by_environment = False
        if selector:
            selected_by_environment = (
                _selector_record(registry.list(), selector).record_id
                == backend.record_id
            )
        if not selected_by_environment:
            try:
                manifest_lease = registry.prepare_runtime_selection(
                    target,
                    environment=dict(os.environ),
                )
            except api.BackendPluginNoCandidateError:
                pass
            else:
                raise api.BackendPluginSelectionError(
                    "Explicit Legacy driver would bypass a Manifest winner",
                    plugin_id=manifest_lease.plugin_id,
                    entry_point=manifest_lease.entry_point_name,
                    field="compiler_cls,driver_cls",
                    expected=manifest_lease.record_id,
                    actual=backend.record_id,
                    remediation="Use the Manifest runtime pair for this target.",
                )
        if not _probe_compiler(backend, target):
            raise api.BackendPluginSelectionError(
                "Explicit Legacy driver and compiler disagree on the target",
                field="compiler_cls.supports_target",
                expected=target.backend,
                actual="False",
                remediation="Use the exact same-record Legacy runtime pair.",
            )
        decision = registry.commit_legacy_selection(
            match.lease,
            target=target,
            activate=True,
        )
        resolved = _cache_decision(decision)
        if resolved.registry_lifecycle_epoch != operation_lifecycle_epoch:
            raise api.BackendPluginLifecycleError(
                "Explicit Legacy activation crossed a Registry reset boundary",
                plugin_id=resolved.plugin_id,
                entry_point=resolved.entry_point_name,
                field="registry lifecycle epoch",
                expected=str(operation_lifecycle_epoch),
                actual=str(resolved.registry_lifecycle_epoch),
                remediation="Retry explicit activation after reset completes.",
            )
        return resolved

    # Manual compatibility is last and cannot overlap any governed target.
    if not _probe_compiler(backend, target):
        raise api.BackendPluginSelectionError(
            "Manual driver and compiler disagree on the target",
            field="compiler_cls.supports_target",
            expected=target.backend,
            actual="False",
            remediation="Use a consistent manual runtime pair.",
        )
    _require_legacy_resolution_generation(
        registry,
        operation_generation,
        backend=backend,
    )
    activate_backend(backend, target=target)
    if registry.lifecycle_epoch != operation_lifecycle_epoch:
        raise api.BackendPluginLifecycleError(
            "Manual activation crossed a Registry reset boundary",
            entry_point=backend.entry_point_name,
            field="registry lifecycle epoch",
            expected=str(operation_lifecycle_epoch),
            actual=str(registry.lifecycle_epoch),
            remediation="Retry explicit activation after reset completes.",
        )
    return replace(
        backend,
        registry_generation=operation_generation,
        registry_lifecycle_epoch=operation_lifecycle_epoch,
    )


def resolve_active_legacy_driver(*, kernel_required_capabilities=()):
    """Resolve Legacy only after proving no Manifest driver is available."""
    candidates = get_driver_backends()
    if candidates:
        raise _selection_error(
            "Manifest driver candidates must be consumed before Legacy",
            field="driver_cls.is_active",
            expected="no applicable Manifest driver candidates",
            actual=str(len(candidates)),
            remediation=(
                "Use triton.runtime.driver so Manifest active probes run "
                "before governed Legacy fallback."
            ),
        )
    return _resolve_active_legacy_driver_after_manifest_miss(
        kernel_required_capabilities=kernel_required_capabilities,
    )


def _selected_backend(target, capabilities):
    api = _registry_api()
    registry = _registry()
    target_name = _target_name(target)
    existing = (
        registry.get_selection(target_name)
        if isinstance(target_name, str) and target_name
        else None
    )
    environment_selector = os.environ.get(api.BACKEND_SELECTOR_ENV)
    if environment_selector == "":
        environment_selector = None
    selected_record = None
    if environment_selector is not None:
        selected_record = _selector_record(
            registry.list(),
            environment_selector,
        )
    if (
        existing is not None
        and selected_record is not None
        and selected_record.record_id != existing.record_id
        and _enum_value(existing.record.state) == "active"
    ):
        raise api.BackendPluginConflictError(
            "Explicit backend selector conflicts with the active runtime pair",
            field="backend_selector",
            expected=existing.record_id,
            actual=selected_record.record_id,
            conflict_kind="active_backend",
            related_record_ids=(existing.record_id, selected_record.record_id),
            remediation="Reset runtime state before switching backend records.",
        )
    if existing is not None and (
        selected_record is None or selected_record.record_id == existing.record_id
    ):
        if existing.is_legacy and capabilities:
            raise _legacy_capability_error(capabilities)
        if not existing.is_legacy and capabilities:
            report = existing.record.capability_report
            if existing.record.manifest is None or report is None:
                raise api.BackendPluginLifecycleError(
                    "Selected Manifest backend lacks capability evidence",
                    plugin_id=existing.plugin_id,
                    entry_point=existing.entry_point_name,
                    field="capability_report",
                    expected="static Manifest capability evidence",
                    actual="missing",
                    remediation="Reset and revalidate the Manifest backend.",
                )
            api.validate_plugin_capabilities(
                existing.record.manifest,
                core_provided=report.core_provided,
                kernel_required=capabilities,
            )
        return _cache_decision(existing)

    if environment_selector is not None:
        if _enum_value(selected_record.source) == "legacy":
            return _legacy_compiler_resolution(
                target,
                kernel_required_capabilities=capabilities,
                selector=environment_selector,
            )
    decision = registry.select(
        target,
        kernel_required_capabilities=capabilities,
    )
    return _cache_decision(decision)


def _prepare_manifest_compiler_backend(target, capabilities):
    """Prepare, but do not publish, the exact Manifest compiler winner."""
    api = _registry_api()
    registry = _registry()
    target_name = _target_name(target)
    selector = os.environ.get(api.BACKEND_SELECTOR_ENV)
    if selector == "":
        selector = None
    existing = registry.get_selection(target_name)
    explicit_selector = None
    if existing is not None and selector is None:
        if existing.is_legacy:
            raise api.BackendPluginNoCandidateError(
                "Target is already governed by a Legacy runtime pair",
                entry_point=existing.entry_point_name,
                field="source",
                expected="reuse the exact Legacy selection",
                actual="legacy",
                remediation="Resolve the existing Legacy record through its probe path.",
            )
        explicit_selector = existing.record_id
    lease = registry.prepare_runtime_selection(
        target,
        kernel_required_capabilities=capabilities,
        explicit_selector=explicit_selector,
        environment=dict(os.environ),
    )
    return lease, Backend(
        compiler=lease.compiler_cls,
        driver=lease.driver_cls,
        record_id=lease.record_id,
        plugin_id=lease.plugin_id,
        entry_point_name=lease.entry_point_name,
        registry_generation=lease.generation,
        registry_lifecycle_epoch=lease.lifecycle_epoch,
    )


def _manual_backend_for_target(target):
    candidates = tuple(
        backend for backend in _manual_backends() if _probe_compiler(backend, target)
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise _selection_error(
            "Manual Legacy backend selection is ambiguous",
            field="compiler_cls.supports_target",
            expected="one compatible manually registered backend",
            actual=str(len(candidates)),
            remediation=(
                "Migrate the backend to a Manifest or keep only one manual "
                "Legacy backend for this target."
            ),
        )
    return None


def get_backend(
    target,
    *,
    kernel_required_capabilities=(),
) -> Backend:
    """Return the Manifest-first compiler/driver pair for one target."""
    api = _registry_api()
    registry = _registry()
    capabilities = _kernel_capabilities(kernel_required_capabilities)
    try:
        return _selected_backend(target, capabilities)
    except api.BackendPluginNoCandidateError as manifest_miss:
        if capabilities:
            raise _legacy_capability_error(capabilities) from manifest_miss
        try:
            return _legacy_compiler_resolution(target)
        except api.BackendPluginNoCandidateError as legacy_miss:
            if registry.list():
                raise legacy_miss
            manual = _manual_backend_for_target(target)
            if manual is not None:
                return manual
            raise manifest_miss


def _prepare_driver_backend_proposals() -> Tuple[_RuntimeBackendProposal, ...]:
    """Prepare deterministic Manifest pairs without publishing selections."""
    api = _registry_api()
    registry = _registry()
    _prune_registry_backends(registry)
    records = registry.validate()
    environment = dict(os.environ)
    selector = environment.get(api.BACKEND_SELECTOR_ENV)
    if selector == "":
        selector = None

    rejected = tuple(
        sorted(
            (record.record_id, record.error)
            for record in records
            if record.error is not None and _enum_value(record.source) != "legacy"
        )
    )
    if rejected:
        # Runtime does not know its target until after construction.  Any
        # rejected Manifest could own that target, so probing Legacy/manual
        # code first would be an unsafe downgrade.
        raise rejected[0][1]

    published_legacy = _published_legacy_records(registry)
    if published_legacy:
        if selector is not None:
            selected = _selector_record(records, selector)
            if selected.record_id != published_legacy[0].record_id:
                raise api.BackendPluginConflictError(
                    "Explicit selector conflicts with the governed Legacy runtime",
                    field="backend_selector",
                    expected=published_legacy[0].record_id,
                    actual=selected.record_id,
                    conflict_kind="legacy_runtime_selection",
                    related_record_ids=(
                        published_legacy[0].record_id,
                        selected.record_id,
                    ),
                    remediation="Reset before switching runtime records.",
                )
        return ()

    selector_record = None
    forced_existing = ()
    if selector is not None:
        selector_record = _selector_record(records, selector)
        if _enum_value(selector_record.source) == "legacy":
            return ()
        targets = tuple(selector_record.manifest.targets)
    else:
        seen_decisions = {}
        for record in records:
            for selected_target in record.selected_targets:
                decision = registry.get_selection(selected_target)
                if decision is not None and not decision.is_legacy:
                    seen_decisions[decision.record_id] = decision
        forced_existing = tuple(
            seen_decisions[record_id] for record_id in sorted(seen_decisions)
        )
        targets = tuple(
            sorted(
                {
                    target
                    for record in (
                        tuple(decision.record for decision in forced_existing)
                        if forced_existing
                        else records
                    )
                    if (
                        record.manifest is not None
                        and api.operational_record_manifest_error(record) is None
                        and _enum_value(record.compatibility_status) == "compatible"
                    )
                    for target in record.manifest.targets
                }
            )
        )

    grouped = {}
    forced = {}
    for target in targets:
        existing = registry.get_selection(target)
        if existing is not None and existing.is_legacy:
            return ()
        exact_record_id = None
        target_forced = selector_record is not None
        if forced_existing:
            owners = tuple(
                decision
                for decision in forced_existing
                if target in decision.record.manifest.targets
            )
            if len(owners) != 1:
                raise api.BackendPluginConflictError(
                    "Forced Manifest runtime selections overlap",
                    field="targets",
                    expected="one forced record per target",
                    actual=", ".join(decision.record_id for decision in owners) or "0",
                    conflict_kind="runtime_selection",
                    remediation="Reset conflicting explicit runtime selections.",
                )
            exact_record_id = owners[0].record_id
            target_forced = True
        elif existing is not None and selector_record is None:
            exact_record_id = existing.record_id
            target_forced = (
                _enum_value(existing.method) in {"python_explicit", "environment"}
                or _enum_value(existing.record.state) == "active"
            )
        lease = registry.prepare_runtime_selection(
            target,
            explicit_selector=exact_record_id,
            environment=environment,
        )
        grouped.setdefault(lease.record_id, []).append(lease)
        forced[lease.record_id] = forced.get(lease.record_id, False) or target_forced

    proposals = tuple(
        _runtime_backend_proposal(
            grouped[record_id],
            forced=forced[record_id],
        )
        for record_id in sorted(grouped)
    )
    if proposals:
        return proposals

    if _legacy_records(registry):
        return ()
    manual = _manual_backends()
    return tuple(
        _RuntimeBackendProposal(
            backend=backend,
            leases=(),
            forced=False,
        )
        for backend in manual
    )


def get_driver_backends() -> Tuple[Backend, ...]:
    """Return F6-validated runtime pairs without early publication."""
    return tuple(proposal.backend for proposal in _prepare_driver_backend_proposals())


def make_backend(target, *, kernel_required_capabilities=()):
    """Instantiate a Manifest-first compiler without early Legacy publication."""
    api = _registry_api()
    capabilities = _kernel_capabilities(kernel_required_capabilities)
    selector = os.environ.get(api.BACKEND_SELECTOR_ENV)
    if selector == "":
        selector = None
    if selector is not None:
        selected_record = _selector_record(_registry().list(), selector)
        if _enum_value(selected_record.source) == "legacy":
            return _legacy_compiler_resolution(
                target,
                kernel_required_capabilities=capabilities,
                selector=selector,
                construct=True,
            )

    registry = _registry()
    existing = registry.get_selection(_target_name(target))
    if existing is not None and existing.is_legacy:
        return _legacy_compiler_resolution(
            target,
            kernel_required_capabilities=capabilities,
            selector=selector,
            construct=True,
        )

    def resolve_manifest_compiler():
        lease, backend = _prepare_manifest_compiler_backend(
            target,
            capabilities,
        )
        if not _probe_compiler(backend, target):
            raise _selection_error(
                "Selected backend compiler rejected its declared target",
                field="compiler_cls.supports_target",
                expected=str(_target_name(target)),
                actual="False",
                remediation=(
                    "Align Manifest targets with compiler_cls.supports_target()."
                ),
            )
        _require_legacy_resolution_lifecycle_epoch(
            registry,
            lease.lifecycle_epoch,
            backend=backend,
        )
        compiler = _construct_legacy_compiler(backend, target)
        _require_legacy_resolution_lifecycle_epoch(
            registry,
            lease.lifecycle_epoch,
            backend=backend,
        )
        decision = registry.commit_runtime_selection(lease)
        published = _cache_decision(decision)
        if (
            published.compiler is not backend.compiler
            or published.driver is not backend.driver
        ):
            raise api.BackendPluginLifecycleError(
                "Published Manifest runtime pair changed after compiler probing",
                plugin_id=backend.plugin_id,
                entry_point=backend.entry_point_name,
                field="compiler_cls,driver_cls",
                expected=backend.record_id,
                actual=published.record_id,
                remediation="Reset and retry exact-record compiler resolution.",
            )
        return compiler

    try:
        return _run_legacy_resolution(
            "manifest-compiler",
            resolve_manifest_compiler,
        )
    except api.BackendPluginNoCandidateError as manifest_miss:
        if capabilities:
            raise _legacy_capability_error(capabilities) from manifest_miss
        try:
            return _legacy_compiler_resolution(
                target,
                kernel_required_capabilities=capabilities,
                construct=True,
            )
        except api.BackendPluginNoCandidateError as legacy_miss:
            if registry.list():
                raise legacy_miss
            manual = _manual_backend_for_target(target)
            if manual is None:
                raise legacy_miss
            return _construct_legacy_compiler(manual, target)


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
        api = _registry_api()
        published_governed = tuple(
            sorted(
                (
                    record
                    for record in records
                    if _enum_value(record.state) in {"selected", "active"}
                    and record.selected_targets
                ),
                key=lambda record: record.record_id,
            )
        )
        if published_governed:
            record_ids = tuple(record.record_id for record in published_governed)
            raise _selection_error(
                "Manual Legacy driver overlaps a governed runtime pair",
                field="compiler_cls,driver_cls",
                expected=", ".join(record_ids),
                actual="manual Legacy backend",
                remediation=(
                    "Reuse the published Registry runtime pair or reset before "
                    "activating a manual backend."
                ),
            )
        existing_selection = registry.get_selection(target_name)
        if existing_selection is not None:
            raise _selection_error(
                "Manual Legacy driver overlaps a governed target selection",
                field="compiler_cls,driver_cls",
                expected=existing_selection.record_id,
                actual="manual Legacy backend",
                remediation=(
                    "Reuse the selected Registry runtime pair or reset before "
                    "activating a manual backend."
                ),
            )
        operational_matching = tuple(
            record
            for record in records
            if (
                record.manifest is not None
                and api.operational_record_manifest_error(record) is None
                and type(record.manifest.targets) is tuple
                and all(type(candidate) is str for candidate in record.manifest.targets)
                and target_name in record.manifest.targets
            )
        )
        malformed_same_entry_point = tuple(
            sorted(
                (record.record_id, record.error)
                for record in records
                if (
                    record.error is not None
                    and record.manifest is None
                    and api.operational_record_manifest_error(record) is None
                    and _enum_value(record.source) != "legacy"
                    and record.entry_point_name == backend.entry_point_name
                )
            )
        )
        if malformed_same_entry_point:
            raise malformed_same_entry_point[0][1]
        rejected = tuple(
            sorted(
                (record.record_id, record.error)
                for record in operational_matching
                if record.error is not None
            )
        )
        if rejected:
            raise rejected[0][1]
        if operational_matching:
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
        unsupported_related = []
        for record in records:
            error = api.operational_record_manifest_error(record)
            if error is None:
                continue
            target_related = _unsupported_record_claims_target(record, target_name) or (
                record.manifest is None
                and record.entry_point_name == backend.entry_point_name
            )
            if not target_related:
                continue
            unsupported_related.append((record.record_id, error))
        unsupported_related.sort(key=lambda item: item[0])
        if unsupported_related:
            raise unsupported_related[0][1]
        return None
    registry = _registry()
    record = registry.validate(record_id)
    if record.manifest is not None:
        _registry_api().require_operational_manifest(record.manifest)
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
    return registry.activate(record_id)


if __name__ == "triton.backends":
    register_runtime_interface_contract()
    register_legacy_runtime_pair_materializer()
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
    "register_legacy_runtime_pair_materializer",
    "register_runtime_interface_contract",
    "resolve_active_legacy_driver",
    "resolve_legacy_compiler_for_target",
]
