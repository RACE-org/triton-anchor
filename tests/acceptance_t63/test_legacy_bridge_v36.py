"""Triton 3.6 governed lazy-Legacy compatibility acceptance tests.

The Legacy layout in this file is the real Triton 3.6 convention: an entry
point value names a package root and the compiler/driver classes live in its
``.compiler`` and ``.driver`` modules.  The root deliberately exposes no
``compiler_cls`` or ``driver_cls`` aliases.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import inspect
import json
import sys
import threading
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import triton_anchor.backends as backend_api
from packaging.tags import Tag
from triton_anchor.backends import (
    BackendPluginCompatibilityError,
    BackendPluginConflictError,
    BackendPluginInterfaceError,
    BackendPluginLifecycleError,
    BackendPluginRegistry,
    BackendPluginSelectionError,
    CoreEnvironment,
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRITON_PACKAGE = REPOSITORY_ROOT / "triton/python/triton"
BASELINE_PATH = Path(__file__).with_name("triton_v36_legacy_baseline.json")
TRITON_COMMIT = "6cc4505027d7b39fe18a44a7f89085b8babb7400"
LLVM_COMMIT = "a992f29451b9e140424f35ac5e20177db4afbdc0"


def _environment() -> CoreEnvironment:
    return CoreEnvironment(
        core_version="0.2.0",
        build_info_generated=True,
        backend_protocol_version="1.0",
        manifest_schema_version="1.0",
        triton_version="3.6.0",
        vendored_triton_commit=TRITON_COMMIT,
        expected_llvm_project_commit=LLVM_COMMIT,
        actual_llvm_version_raw="22.0.0git",
        actual_llvm_version="22.0.0",
        actual_llvm_version_suffix="git",
        actual_llvm_commit=LLVM_COMMIT,
        actual_mlir_version_raw="22.0.0git",
        actual_mlir_version="22.0.0",
        actual_mlir_version_suffix="git",
        actual_mlir_commit=LLVM_COMMIT,
        cxx_standard="17",
        cxx_compiler_id="GNU",
        cxx_compiler_version="13.3.0",
        cxx11_abi="1",
        build_type="Release",
        ttgpu=True,
        built_python_version="3.12.3",
        built_python_soabi="cpython-fixture",
        built_platform="fixture-platform",
        core_abi_fingerprint_schema="triton-anchor-core-abi-v1",
        core_library_sha256="sha256:" + "c" * 64,
        core_abi_fingerprint="sha256:" + "a" * 64,
        runtime_python_version="3.12.3",
        runtime_python_implementation="CPython",
        runtime_python_soabi="cpython-fixture",
        runtime_platform="fixture-platform",
        runtime_system="Linux",
        runtime_machine="x86_64",
    )


def _package(name: str, path: Path, *, version: str | None = None):
    package = types.ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(path)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = package.__path__
    if version is not None:
        package.__version__ = version
    return package


class _EntryPoint:
    group = "triton.backends"

    def __init__(self, name: str, value: str, loaded_root: Any) -> None:
        self.name = name
        self.value = value
        self.loaded_root = loaded_root
        self.load_calls = 0
        self.dist = None

    def load(self) -> Any:
        self.load_calls += 1
        return self.loaded_root


class _LegacyDistribution:
    def __init__(self, name: str, entry_point: _EntryPoint) -> None:
        self.metadata = {"Name": name}
        self.name = name
        self.version = "1.0.0"
        self.entry_points = (entry_point,)
        self.files: tuple[str, ...] = ()
        entry_point.dist = self

    def read_text(self, filename: str) -> str | None:
        if filename == "WHEEL":
            return "Wheel-Version: 1.0\nTag: py3-none-any\n"
        return None


class _ManifestDistribution(_LegacyDistribution):
    def __init__(
        self,
        name: str,
        entry_point: _EntryPoint,
        manifest_path: Path,
    ) -> None:
        super().__init__(name, entry_point)
        self.files = ("fixture/triton_anchor_backend.json",)
        self._manifest_path = manifest_path

    def locate_file(self, _file: Any) -> Path:
        return self._manifest_path


@dataclass
class _Behavior:
    calls: Counter[str]
    seen_targets: list[Any]
    supports: bool = True
    active: bool = True
    supports_error: BaseException | None = None
    active_error: BaseException | None = None
    compiler_constructor_error: BaseException | None = None
    driver_constructor_error: BaseException | None = None
    current_target_error: BaseException | None = None
    current_target: Any = None
    supports_entered: threading.Event | None = None
    supports_release: threading.Event | None = None
    active_entered: threading.Event | None = None
    active_release: threading.Event | None = None


@dataclass
class _LegacyFixture:
    name: str
    package_root: str
    entry_point: _EntryPoint
    distribution: _LegacyDistribution
    compiler_cls: type
    driver_cls: type
    behavior: _Behavior


@dataclass
class _ManifestFixture:
    name: str
    entry_point: _EntryPoint
    distribution: _ManifestDistribution
    behavior: _Behavior


def _generic_implementation(member: str):
    def implementation(*_args: Any, **_kwargs: Any) -> Any:
        if member == "hash":
            return "t63-f5-v36"
        if member == "parse_options":
            return {}
        if member == "get_module_map":
            return {}
        if member == "get_benchmarker":
            return lambda _call, *, quantiles, **_kwargs: tuple(0.0 for _ in quantiles)
        return None

    return implementation


def _implementation_for(
    descriptor: Any,
    member: str,
    implementation: Any,
) -> Any:
    if isinstance(descriptor, property):
        return property(implementation)
    if isinstance(descriptor, classmethod):
        return classmethod(implementation)
    if isinstance(descriptor, staticmethod):
        return staticmethod(implementation)
    return implementation


def _compiler_class(
    contract: type,
    module_name: str,
    behavior: _Behavior,
    *,
    label: str,
    omitted: frozenset[str] = frozenset(),
) -> type:
    namespace: dict[str, Any] = {"__module__": module_name}
    for member in sorted(contract.__abstractmethods__):
        if member in omitted:
            continue
        descriptor = inspect.getattr_static(contract, member)
        if member == "supports_target":

            def supports_target(target: Any) -> bool:
                behavior.calls["supports_target"] += 1
                behavior.seen_targets.append(target)
                if behavior.supports_entered is not None:
                    behavior.supports_entered.set()
                if behavior.supports_release is not None:
                    assert behavior.supports_release.wait(5)
                if behavior.supports_error is not None:
                    raise behavior.supports_error
                return behavior.supports

            implementation = supports_target
        else:
            implementation = _generic_implementation(member)
        namespace[member] = _implementation_for(descriptor, member, implementation)

    def constructor(self: Any, target: Any) -> None:
        behavior.calls["compiler_constructor"] += 1
        behavior.seen_targets.append(target)
        if behavior.compiler_constructor_error is not None:
            raise behavior.compiler_constructor_error
        self.target = target

    namespace["__init__"] = constructor
    return type(f"{label}Compiler", (contract,), namespace)


def _driver_class(
    contract: type,
    module_name: str,
    behavior: _Behavior,
    *,
    label: str,
    omitted: frozenset[str] = frozenset(),
) -> type:
    namespace: dict[str, Any] = {"__module__": module_name}
    for member in sorted(contract.__abstractmethods__):
        if member in omitted:
            continue
        descriptor = inspect.getattr_static(contract, member)
        if member == "is_active":

            def is_active(_cls: type) -> bool:
                behavior.calls["is_active"] += 1
                if behavior.active_entered is not None:
                    behavior.active_entered.set()
                if behavior.active_release is not None:
                    assert behavior.active_release.wait(5)
                if behavior.active_error is not None:
                    raise behavior.active_error
                return behavior.active

            implementation = is_active
        elif member == "get_current_target":

            def get_current_target(_self: Any) -> Any:
                behavior.calls["get_current_target"] += 1
                if behavior.current_target_error is not None:
                    raise behavior.current_target_error
                return behavior.current_target

            implementation = get_current_target
        else:
            implementation = _generic_implementation(member)
        namespace[member] = _implementation_for(descriptor, member, implementation)

    def constructor(self: Any) -> None:
        behavior.calls["driver_constructor"] += 1
        if behavior.driver_constructor_error is not None:
            raise behavior.driver_constructor_error

    namespace["__init__"] = constructor
    return type(f"{label}Driver", (contract,), namespace)


@dataclass
class _Harness:
    registry: BackendPluginRegistry
    distributions: list[Any]
    backends: types.ModuleType
    runtime_driver: types.ModuleType
    module_names: list[str]
    root: Path
    sequence: int = 0

    def legacy(
        self,
        label: str,
        *,
        target_name: str = "legacyv36",
        supports: bool = True,
        active: bool = True,
        omit_compiler: str | None = None,
        omit_driver: str | None = None,
    ) -> _LegacyFixture:
        self.sequence += 1
        package_root = f"t63_f5_v36_{label}_{self.sequence}"
        root_module = _package(package_root, self.root / package_root)
        compiler_name = package_root + ".compiler"
        driver_name = package_root + ".driver"
        behavior = _Behavior(Counter(), [], supports=supports, active=active)
        behavior.current_target = self.backends.compiler.GPUTarget(
            target_name, f"{label}-arch", 32
        )
        compiler_cls = _compiler_class(
            self.backends.BaseBackend,
            compiler_name,
            behavior,
            label=label.title(),
            omitted=frozenset({omit_compiler}) if omit_compiler else frozenset(),
        )
        driver_cls = _driver_class(
            self.backends.DriverBase,
            driver_name,
            behavior,
            label=label.title(),
            omitted=frozenset({omit_driver}) if omit_driver else frozenset(),
        )
        compiler_module = types.ModuleType(compiler_name)
        compiler_module.__package__ = package_root
        compiler_module.LegacyCompiler = compiler_cls
        driver_module = types.ModuleType(driver_name)
        driver_module.__package__ = package_root
        driver_module.LegacyDriver = driver_cls
        # This is intentionally the historical package-root layout.  The root
        # has no compiler_cls/driver_cls compatibility aliases.
        assert not hasattr(root_module, "compiler_cls")
        assert not hasattr(root_module, "driver_cls")
        sys.modules[package_root] = root_module
        sys.modules[compiler_name] = compiler_module
        sys.modules[driver_name] = driver_module
        self.module_names.extend((package_root, compiler_name, driver_name))
        entry_point = _EntryPoint(label, package_root, root_module)
        distribution = _LegacyDistribution(f"t63-f5-{label}", entry_point)
        return _LegacyFixture(
            label,
            package_root,
            entry_point,
            distribution,
            compiler_cls,
            driver_cls,
            behavior,
        )

    def manifest(
        self,
        label: str,
        *,
        target_name: str = "legacyv36",
        compatible: bool = True,
    ) -> _ManifestFixture:
        self.sequence += 1
        behavior = _Behavior(Counter(), [])
        behavior.current_target = self.backends.compiler.GPUTarget(
            target_name, f"{label}-arch", 32
        )
        compiler_cls = _compiler_class(
            self.backends.BaseBackend,
            __name__,
            behavior,
            label=label.title(),
        )
        driver_cls = _driver_class(
            self.backends.DriverBase,
            __name__,
            behavior,
            label=label.title(),
        )

        class Plugin:
            def initialize(self, _context: Any) -> None:
                behavior.calls["initialize"] += 1

        plugin = Plugin()
        plugin.compiler_cls = compiler_cls
        plugin.driver_cls = driver_cls
        entry_point = _EntryPoint(label, f"t63_manifest_{label}:plugin", plugin)
        directory = self.root / f"manifest-{label}-{self.sequence}"
        directory.mkdir()
        manifest_path = directory / "triton_anchor_backend.json"
        requires = ">=3.6,<3.7" if compatible else ">=3.5,<3.6"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "plugins": [
                        {
                            "plugin_id": f"acceptance.f5.{label}",
                            "entry_point": label,
                            "backend_protocol": ">=1.0,<2.0",
                            "requires_core": ">=0.2,<0.3",
                            "requires_triton": {
                                "version": requires,
                                "commit": TRITON_COMMIT,
                            },
                            "targets": [target_name],
                            "capabilities": ["acceptance.f5"],
                            "isolation_mode": "python_only",
                            "priority": 100,
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        distribution = _ManifestDistribution(
            f"t63-f5-manifest-{label}", entry_point, manifest_path
        )
        return _ManifestFixture(label, entry_point, distribution, behavior)

    def install(self, *fixtures: Any) -> None:
        assert self.registry.reset() == ()
        self.distributions[:] = [fixture.distribution for fixture in fixtures]


@pytest.fixture
def f5_runtime(tmp_path: Path):
    registry_module = importlib.import_module("triton_anchor.backends.registry")
    previous_registry = registry_module.backend_plugin_registry
    previous_api_registry = backend_api.backend_plugin_registry
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "triton" or name.startswith("triton.")
    }
    for name in tuple(previous_modules):
        sys.modules.pop(name, None)

    distributions: list[Any] = []
    module_names: list[str] = []
    registry = BackendPluginRegistry(
        distribution_provider=lambda: tuple(distributions),
        environment_provider=_environment,
        supported_tags=(Tag("py3", "none", "any"),),
        preflight_profile="triton_version",
    )
    registry_module.backend_plugin_registry = registry
    backend_api.backend_plugin_registry = registry
    sys.modules["triton"] = _package("triton", TRITON_PACKAGE, version="3.6.0")

    try:
        backends = importlib.import_module("triton.backends")
        sys.modules["triton.runtime"] = _package(
            "triton.runtime", TRITON_PACKAGE / "runtime"
        )
        runtime_driver = importlib.import_module("triton.runtime.driver")
        yield _Harness(
            registry,
            distributions,
            backends,
            runtime_driver,
            module_names,
            tmp_path,
        )
    finally:
        registry.reset()
        registry_module.backend_plugin_registry = previous_registry
        backend_api.backend_plugin_registry = previous_api_registry
        for name in module_names:
            sys.modules.pop(name, None)
        for name in tuple(sys.modules):
            if name == "triton" or name.startswith("triton."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


def _record(harness: _Harness, fixture: _LegacyFixture):
    return next(
        record
        for record in harness.registry.list()
        if record.entry_point_name == fixture.name
    )


def test_upstream_v36_legacy_oracle_is_frozen() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["upstream_commit"] == ("9ceba3e222fd2c9af9cbaf6a440beb23911c197b")
    assert baseline["fixture"]["entry_point_value"] == ("t63_v36_legacy_oracle")
    assert baseline["fixture"]["has_manifest"] is False
    assert baseline["observed"]["mapping_key"] == "baseline-legacy"
    assert baseline["observed"]["compiler_driver_same_package_root"] is True
    assert baseline["observed"]["make_backend_target_identity_preserved"] is True


def test_core_exposes_version_neutral_legacy_bridge_primitives() -> None:
    required = (
        "LegacyRuntimePair",
        "LegacyRuntimePairMaterializationContext",
        "LegacySelectionLease",
        "BackendPluginNoCandidateError",
    )
    assert not [name for name in required if getattr(backend_api, name, None) is None]
    assert callable(
        getattr(
            BackendPluginRegistry, "register_legacy_runtime_pair_materializer", None
        )
    )
    assert callable(getattr(BackendPluginRegistry, "prepare_legacy_selection", None))
    assert callable(getattr(BackendPluginRegistry, "commit_legacy_selection", None))


def test_discovery_and_diagnostics_remain_metadata_only(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("metadata")
    f5_runtime.install(legacy)
    assert f5_runtime.backends._discover_backends() == {}
    records = f5_runtime.registry.discover(strict=True)
    assert len(records) == 1
    record = records[0]
    assert record.source is PluginSource.LEGACY
    assert record.state is PluginLifecycleState.DISCOVERED
    assert record.compatibility_status is PluginCompatibilityStatus.LEGACY_UNVERIFIED
    f5_runtime.registry.list()
    f5_runtime.registry.inspect(record.record_id)
    f5_runtime.registry.conflicts()
    f5_runtime.registry.validate()
    f5_runtime.registry.diagnostics()
    assert legacy.entry_point.load_calls == 0
    assert f5_runtime.backends.backends == {}


def test_compiler_first_lazily_materializes_exact_v36_package_root(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("compiler")
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "compiler-arch", 32)
    compiler = f5_runtime.backends.make_backend(target)
    record = _record(f5_runtime, legacy)
    decision = f5_runtime.registry.get_selection("legacyv36")
    assert type(compiler) is legacy.compiler_cls
    assert compiler.target is target
    assert legacy.entry_point.load_calls == 1
    assert legacy.behavior.seen_targets.count(target) >= 1
    assert record.state is PluginLifecycleState.SELECTED
    assert record.compatibility_status is PluginCompatibilityStatus.LEGACY_UNVERIFIED
    assert decision is not None and decision.record_id == record.record_id
    public = f5_runtime.backends.backends[legacy.name]
    assert public.record_id == record.record_id
    assert public.compiler is legacy.compiler_cls
    assert public.driver is legacy.driver_cls


def test_runtime_first_publishes_driver_compiler_and_selection_from_one_record(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("runtime")
    f5_runtime.install(legacy)
    driver = f5_runtime.runtime_driver.driver.default
    record = _record(f5_runtime, legacy)
    decision = f5_runtime.registry.get_selection("legacyv36")
    assert type(driver) is legacy.driver_cls
    assert f5_runtime.runtime_driver.driver.active is driver
    assert decision is not None and decision.record_id == record.record_id
    assert record.state is PluginLifecycleState.ACTIVE
    assert record.compatibility_status is PluginCompatibilityStatus.LEGACY_UNVERIFIED
    public = f5_runtime.backends.backends[legacy.name]
    assert public.record_id == record.record_id
    assert public.compiler is legacy.compiler_cls
    assert public.driver is legacy.driver_cls
    assert legacy.behavior.calls["is_active"] == 1
    assert legacy.behavior.calls["driver_constructor"] == 1
    assert legacy.behavior.calls["get_current_target"] == 1


def test_f6_rejects_incomplete_legacy_pair_before_probe_or_publication(
    f5_runtime: _Harness,
) -> None:
    omitted = min(f5_runtime.backends.BaseBackend.__abstractmethods__)
    legacy = f5_runtime.legacy("incomplete", omit_compiler=omitted)
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "bad", 32)
    with pytest.raises(BackendPluginInterfaceError):
        f5_runtime.backends.make_backend(target)
    record = _record(f5_runtime, legacy)
    assert record.state is PluginLifecycleState.REJECTED
    assert legacy.behavior.calls["supports_target"] == 0
    assert legacy.behavior.calls["compiler_constructor"] == 0
    assert legacy.behavior.calls["is_active"] == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert legacy.name not in f5_runtime.backends.backends


@pytest.mark.parametrize("supports", [False, True], ids=["zero", "multiple"])
def test_compiler_first_reports_zero_or_multiple_matches_without_publication(
    f5_runtime: _Harness,
    supports: bool,
) -> None:
    first = f5_runtime.legacy("first", supports=supports)
    fixtures: tuple[_LegacyFixture, ...] = (first,)
    if supports:
        fixtures += (f5_runtime.legacy("second", supports=True),)
    f5_runtime.install(*fixtures)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "ambiguous", 32)
    error_type = BackendPluginConflictError if supports else BackendPluginSelectionError
    with pytest.raises(error_type):
        f5_runtime.backends.make_backend(target)
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}
    assert all(item.behavior.calls["compiler_constructor"] == 0 for item in fixtures)


def test_kernel_capabilities_forbid_legacy_before_import(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("capability")
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "cap", 32)
    with pytest.raises(BackendPluginSelectionError) as caught:
        f5_runtime.backends.make_backend(
            target,
            kernel_required_capabilities=("kernel.tensor-map",),
        )
    assert caught.value.field == "kernel_required_capabilities"
    assert legacy.entry_point.load_calls == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_manifest_winner_prevents_legacy_import(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("manifest")
    legacy = f5_runtime.legacy("legacy-loser")
    f5_runtime.install(legacy, manifest)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "manifest", 32)
    compiler = f5_runtime.backends.make_backend(target)
    assert type(compiler).__name__ == "ManifestCompiler"
    assert manifest.entry_point.load_calls == 1
    assert legacy.entry_point.load_calls == 0
    assert f5_runtime.registry.get_selection("legacyv36").plugin_id == (
        "acceptance.f5.manifest"
    )


def test_rejected_manifest_never_falls_back_to_legacy(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("rejected", compatible=False)
    legacy = f5_runtime.legacy("unsafe-fallback")
    f5_runtime.install(legacy, manifest)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "reject", 32)
    with pytest.raises(BackendPluginCompatibilityError):
        f5_runtime.backends.make_backend(target)
    assert legacy.entry_point.load_calls == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_registry_reset_clears_legacy_publication_and_reloads_lazily(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("reset")
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "reset", 32)
    first = f5_runtime.backends.make_backend(target)
    first_record_id = _record(f5_runtime, legacy).record_id
    assert legacy.entry_point.load_calls == 1
    assert f5_runtime.registry.reset() == ()
    assert f5_runtime.backends.backends == {}
    assert f5_runtime.runtime_driver.driver._default is None
    assert f5_runtime.runtime_driver.driver._active is None
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert legacy.entry_point.load_calls == 1
    second = f5_runtime.backends.make_backend(target)
    assert type(first) is type(second) is legacy.compiler_cls
    assert legacy.entry_point.load_calls == 2
    assert _record(f5_runtime, legacy).record_id == first_record_id


def test_supports_target_failure_is_structured_and_does_not_publish(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("probe-error")
    legacy.behavior.supports_error = RuntimeError("probe sentinel")
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "probe", 32)
    with pytest.raises(BackendPluginLifecycleError) as caught:
        f5_runtime.backends.make_backend(target)
    assert caught.value.field == "compiler_cls.supports_target"
    assert legacy.behavior.calls["compiler_constructor"] == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_driver_constructor_failure_does_not_publish_runtime_state(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("driver-error")
    legacy.behavior.driver_constructor_error = RuntimeError("constructor sentinel")
    f5_runtime.install(legacy)
    with pytest.raises(BackendPluginLifecycleError) as caught:
        _ = f5_runtime.runtime_driver.driver.default
    assert caught.value.field == "driver_cls.__init__"
    assert f5_runtime.runtime_driver.driver._default is None
    assert f5_runtime.runtime_driver.driver._active is None
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}
