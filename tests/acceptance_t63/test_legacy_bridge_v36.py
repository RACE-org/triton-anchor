"""Triton 3.6 governed lazy-Legacy compatibility acceptance tests.

The Legacy layout in this file is the real Triton 3.6 convention: an entry
point value names a package root and the compiler/driver classes live in its
``.compiler`` and ``.driver`` modules.  The root deliberately exposes no
``compiler_cls`` or ``driver_cls`` aliases.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import inspect
import json
import subprocess
import sys
import threading
import types
from collections import Counter
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import triton_anchor.backends as backend_api
from packaging.tags import Tag
from triton_anchor.backends import (
    BACKEND_SELECTOR_ENV,
    BackendPluginCapabilityError,
    BackendPluginCompatibilityError,
    BackendPluginConflictError,
    BackendPluginError,
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
LEGACY_WHEEL = Path(
    "/home/dingbl/race_workspace/reports/t6.3-v3.6-evidence/f5/wheels/"
    "t63_v36_legacy_oracle-1.0.0-py3-none-any.whl"
)
LEGACY_WHEEL_SHA256 = "97402abf825cf56d800edf5f4cfa25a8513be585b2b067391d0766664d18a455"


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
    compiler_constructor_entered: threading.Event | None = None
    compiler_constructor_release: threading.Event | None = None
    driver_constructor_entered: threading.Event | None = None
    driver_constructor_release: threading.Event | None = None
    current_target_entered: threading.Event | None = None
    current_target_release: threading.Event | None = None
    supports_callback: Any = None
    active_callback: Any = None


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
    structural: bool = False,
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
                if behavior.supports_callback is not None:
                    behavior.supports_callback()
                return behavior.supports

            implementation = supports_target
        else:
            implementation = _generic_implementation(member)
        namespace[member] = _implementation_for(descriptor, member, implementation)

    def constructor(self: Any, target: Any) -> None:
        behavior.calls["compiler_constructor"] += 1
        behavior.seen_targets.append(target)
        if behavior.compiler_constructor_entered is not None:
            behavior.compiler_constructor_entered.set()
        if behavior.compiler_constructor_release is not None:
            assert behavior.compiler_constructor_release.wait(5)
        if behavior.compiler_constructor_error is not None:
            raise behavior.compiler_constructor_error
        self.target = target

    namespace["__init__"] = constructor
    bases = () if structural else (contract,)
    return type(f"{label}Compiler", bases, namespace)


def _driver_class(
    contract: type,
    module_name: str,
    behavior: _Behavior,
    *,
    label: str,
    omitted: frozenset[str] = frozenset(),
    structural: bool = False,
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
                if behavior.active_callback is not None:
                    behavior.active_callback()
                return behavior.active

            implementation = is_active
        elif member == "get_current_target":

            def get_current_target(_self: Any) -> Any:
                behavior.calls["get_current_target"] += 1
                if behavior.current_target_entered is not None:
                    behavior.current_target_entered.set()
                if behavior.current_target_release is not None:
                    assert behavior.current_target_release.wait(5)
                if behavior.current_target_error is not None:
                    raise behavior.current_target_error
                return behavior.current_target

            implementation = get_current_target
        else:
            implementation = _generic_implementation(member)
        namespace[member] = _implementation_for(descriptor, member, implementation)

    def constructor(self: Any) -> None:
        behavior.calls["driver_constructor"] += 1
        if behavior.driver_constructor_entered is not None:
            behavior.driver_constructor_entered.set()
        if behavior.driver_constructor_release is not None:
            assert behavior.driver_constructor_release.wait(5)
        if behavior.driver_constructor_error is not None:
            raise behavior.driver_constructor_error

    namespace["__init__"] = constructor
    bases = () if structural else (contract,)
    return type(f"{label}Driver", bases, namespace)


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
        structural: bool = False,
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
            structural=structural,
        )
        driver_cls = _driver_class(
            self.backends.DriverBase,
            driver_name,
            behavior,
            label=label.title(),
            omitted=frozenset({omit_driver}) if omit_driver else frozenset(),
            structural=structural,
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
        priority: int = 100,
        declared_targets: tuple[str, ...] | None = None,
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
                            "targets": list(declared_targets or (target_name,)),
                            "capabilities": ["acceptance.f5"],
                            "isolation_mode": "python_only",
                            "priority": priority,
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


def _manifest_record(harness: _Harness, fixture: _ManifestFixture):
    return next(
        record
        for record in harness.registry.list()
        if record.entry_point_name == fixture.name
    )


def _submit_daemon(callback: Any) -> Future[Any]:
    future: Future[Any] = Future()

    def invoke() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = callback()
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    threading.Thread(target=invoke, daemon=True).start()
    return future


def _bounded_result(future: Future[Any], timeout: float = 10) -> Any:
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        pytest.fail("F5 daemon operation did not complete before timeout")


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
        "RuntimeSelectionLease",
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
    assert callable(getattr(BackendPluginRegistry, "prepare_runtime_selection", None))
    assert callable(getattr(BackendPluginRegistry, "commit_runtime_selection", None))


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


def test_manifest_abi_rejection_never_falls_back_to_legacy(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("abi-rejected")
    document = json.loads(
        manifest.distribution._manifest_path.read_text(encoding="utf-8")
    )
    document["plugins"][0]["abi_fingerprint"] = "sha256:" + "b" * 64
    manifest.distribution._manifest_path.write_text(
        json.dumps(document, sort_keys=True),
        encoding="utf-8",
    )
    legacy = f5_runtime.legacy("forbidden-abi-fallback")
    f5_runtime.install(legacy, manifest)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "abi-rejected", 32)

    with pytest.raises(backend_api.BackendPluginManifestError) as caught:
        f5_runtime.backends.make_backend(target)

    assert caught.value.field in {"abi_fingerprint", "manifest"}
    assert manifest.entry_point.load_calls == 0
    assert legacy.entry_point.load_calls == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_manifest_capability_rejection_never_imports_legacy(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("capability-rejected")
    legacy = f5_runtime.legacy("forbidden-capability-fallback")
    f5_runtime.install(legacy, manifest)
    target = f5_runtime.backends.compiler.GPUTarget(
        "legacyv36", "capability-rejected", 32
    )

    with pytest.raises(BackendPluginCapabilityError):
        f5_runtime.backends.make_backend(
            target,
            kernel_required_capabilities=("missing.kernel.capability",),
        )

    assert manifest.entry_point.load_calls == 0
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


def test_compiler_constructor_failure_does_not_publish_selection(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("compiler-constructor-error")
    legacy.behavior.compiler_constructor_error = RuntimeError("constructor")
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "ctor", 32)
    with pytest.raises(BackendPluginLifecycleError) as caught:
        f5_runtime.backends.make_backend(target)
    assert caught.value.field == "compiler_cls.__init__"
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_structural_legacy_pair_passes_the_same_f6_gate(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("structural", structural=True)
    assert not issubclass(legacy.compiler_cls, f5_runtime.backends.BaseBackend)
    assert not issubclass(legacy.driver_cls, f5_runtime.backends.DriverBase)
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "struct", 32)
    compiler = f5_runtime.backends.make_backend(target)
    assert type(compiler) is legacy.compiler_cls
    assert _record(f5_runtime, legacy).state is PluginLifecycleState.SELECTED


def test_historical_abc_class_wins_over_module_helper_class(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("helper-class")
    compiler_module = sys.modules[legacy.package_root + ".compiler"]
    compiler_module.CompilerOptions = type(
        "CompilerOptions",
        (),
        {"__module__": compiler_module.__name__},
    )
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "helper", 32)
    assert type(f5_runtime.backends.make_backend(target)) is legacy.compiler_cls


def test_runtime_reports_zero_active_legacy_without_publication(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("inactive", active=False)
    f5_runtime.install(legacy)
    with pytest.raises(BackendPluginSelectionError) as caught:
        _ = f5_runtime.runtime_driver.driver.default
    assert caught.value.field == "driver_cls.is_active"
    assert legacy.behavior.calls["driver_constructor"] == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_runtime_reports_multiple_active_legacy_without_publication(
    f5_runtime: _Harness,
) -> None:
    first = f5_runtime.legacy("active-first")
    second = f5_runtime.legacy("active-second")
    f5_runtime.install(first, second)
    with pytest.raises(BackendPluginConflictError) as caught:
        _ = f5_runtime.runtime_driver.driver.default
    assert caught.value.field == "driver_cls.is_active"
    assert first.behavior.calls["driver_constructor"] == 0
    assert second.behavior.calls["driver_constructor"] == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None


def test_runtime_active_probe_failure_is_structured_and_clean(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("active-error")
    legacy.behavior.active_error = RuntimeError("active sentinel")
    f5_runtime.install(legacy)
    with pytest.raises(BackendPluginLifecycleError) as caught:
        _ = f5_runtime.runtime_driver.driver.default
    assert caught.value.field == "driver_cls.is_active"
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_runtime_current_target_failure_is_structured_and_clean(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("target-error")
    legacy.behavior.current_target_error = RuntimeError("target sentinel")
    f5_runtime.install(legacy)
    with pytest.raises(BackendPluginLifecycleError) as caught:
        _ = f5_runtime.runtime_driver.driver.default
    assert caught.value.field == "driver_cls.get_current_target"
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_runtime_rejects_target_not_supported_by_same_record_compiler(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("target-mismatch", supports=False)
    f5_runtime.install(legacy)
    with pytest.raises(BackendPluginSelectionError) as caught:
        _ = f5_runtime.runtime_driver.driver.default
    assert caught.value.field == "compiler_cls.supports_target"
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_explicit_legacy_selector_loads_only_the_exact_record(
    f5_runtime: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = f5_runtime.legacy("selector-first")
    second = f5_runtime.legacy("selector-second")
    f5_runtime.install(first, second)
    records = f5_runtime.registry.discover()
    selected = next(
        record for record in records if record.entry_point_name == second.name
    )
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, selected.record_id)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "selector", 32)
    compiler = f5_runtime.backends.make_backend(target)
    assert type(compiler) is second.compiler_cls
    assert first.entry_point.load_calls == 0
    assert second.entry_point.load_calls == 1


def test_unknown_explicit_selector_never_imports_legacy(
    f5_runtime: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = f5_runtime.legacy("unknown-selector")
    f5_runtime.install(legacy)
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, "does.not.exist")
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "selector", 32)
    with pytest.raises(BackendPluginSelectionError) as caught:
        f5_runtime.backends.make_backend(target)
    assert caught.value.field == "backend_selector"
    assert legacy.entry_point.load_calls == 0


def test_changing_explicit_legacy_selector_requires_reset(
    f5_runtime: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = f5_runtime.legacy("switch-first")
    second = f5_runtime.legacy("switch-second")
    f5_runtime.install(first, second)
    records = f5_runtime.registry.discover()
    by_name = {record.entry_point_name: record for record in records}
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, by_name[first.name].record_id)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "switch", 32)
    assert type(f5_runtime.backends.make_backend(target)) is first.compiler_cls
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, by_name[second.name].record_id)
    with pytest.raises(BackendPluginConflictError):
        f5_runtime.backends.make_backend(target)
    assert second.entry_point.load_calls == 0


def test_existing_manifest_selection_revalidates_kernel_capabilities(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("manifest-capability")
    legacy = f5_runtime.legacy("capability-fallback")
    f5_runtime.install(manifest, legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "cap", 32)
    f5_runtime.backends.make_backend(target)
    with pytest.raises(BackendPluginCapabilityError):
        f5_runtime.backends.make_backend(
            target,
            kernel_required_capabilities=("missing.kernel.capability",),
        )
    assert legacy.entry_point.load_calls == 0


def test_capability_recheck_preserves_existing_exact_manifest_selection(
    f5_runtime: _Harness,
) -> None:
    selected = f5_runtime.manifest("selected-low", priority=1)
    higher = f5_runtime.manifest("unselected-high", priority=200)
    f5_runtime.install(selected, higher)
    records = f5_runtime.registry.validate()
    selected_record = next(
        record for record in records if record.entry_point_name == selected.name
    )
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "caps", 32)
    f5_runtime.registry.select(
        target,
        explicit_selector=selected_record.record_id,
    )
    compiler = f5_runtime.backends.make_backend(
        target,
        kernel_required_capabilities=("acceptance.f5",),
    )
    decision = f5_runtime.registry.get_selection("legacyv36")
    assert type(compiler).__name__ == "Selected-LowCompiler"
    assert decision is not None and decision.record_id == selected_record.record_id
    assert higher.entry_point.load_calls == 0


def test_inactive_unrelated_manifest_driver_allows_governed_legacy(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("inactive-manifest", target_name="manifest-only")
    manifest.behavior.active = False
    legacy = f5_runtime.legacy("active-legacy")
    f5_runtime.install(manifest, legacy)
    driver = f5_runtime.runtime_driver.driver.default
    assert type(driver) is legacy.driver_cls
    assert manifest.behavior.calls["is_active"] == 1
    assert legacy.entry_point.load_calls == 1


def test_inactive_same_target_manifest_is_atomically_replaced_by_legacy(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("inactive-same-target")
    manifest.behavior.active = False
    legacy = f5_runtime.legacy("active-same-target")
    f5_runtime.install(manifest, legacy)
    driver = f5_runtime.runtime_driver.driver.default
    decision = f5_runtime.registry.get_selection("legacyv36")
    legacy_record = _record(f5_runtime, legacy)
    assert type(driver) is legacy.driver_cls
    assert decision is not None and decision.record_id == legacy_record.record_id
    assert legacy_record.state is PluginLifecycleState.ACTIVE
    assert manifest.name not in f5_runtime.backends.backends
    assert (
        f5_runtime.backends.backends[legacy.name].record_id == legacy_record.record_id
    )


def test_rejected_manifest_prevents_runtime_legacy_fallback(
    f5_runtime: _Harness,
) -> None:
    inactive = f5_runtime.manifest("inactive-good", target_name="manifest-only")
    inactive.behavior.active = False
    rejected = f5_runtime.manifest("rejected-runtime", compatible=False)
    legacy = f5_runtime.legacy("runtime-fallback")
    f5_runtime.install(inactive, rejected, legacy)
    with pytest.raises(BackendPluginCompatibilityError):
        _ = f5_runtime.runtime_driver.driver.default
    assert legacy.entry_point.load_calls == 0


def test_public_legacy_helper_cannot_bypass_manifest_first(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("helper-manifest")
    legacy = f5_runtime.legacy("helper-legacy")
    f5_runtime.install(manifest, legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "helper", 32)
    backend = f5_runtime.backends.resolve_legacy_compiler_for_target(target)
    assert backend.plugin_id == "acceptance.f5.helper-manifest"
    assert legacy.entry_point.load_calls == 0


def test_concurrent_compiler_first_publishes_one_selection(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("concurrent-compiler")
    f5_runtime.install(legacy)
    f5_runtime.registry.discover()
    generation = f5_runtime.registry.generation
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "threads", 32)
    barrier = threading.Barrier(8)
    futures = tuple(
        _submit_daemon(
            lambda: (
                barrier.wait(timeout=5),
                f5_runtime.backends.make_backend(target),
            )[1]
        )
        for _ in range(8)
    )
    compilers = tuple(_bounded_result(future) for future in futures)
    assert all(type(compiler) is legacy.compiler_cls for compiler in compilers)
    assert legacy.entry_point.load_calls == 1
    assert legacy.behavior.calls["supports_target"] == 8
    assert legacy.behavior.calls["compiler_constructor"] == 8
    assert f5_runtime.registry.generation == generation + 1
    assert len(f5_runtime.backends.backends) == 1


def test_concurrent_runtime_first_constructs_one_driver(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("concurrent-runtime")
    f5_runtime.install(legacy)
    barrier = threading.Barrier(8)
    futures = tuple(
        _submit_daemon(
            lambda: (
                barrier.wait(timeout=5),
                f5_runtime.runtime_driver.driver.default,
            )[1]
        )
        for _ in range(8)
    )
    drivers = tuple(_bounded_result(future) for future in futures)
    assert all(driver is drivers[0] for driver in drivers)
    assert type(drivers[0]) is legacy.driver_cls
    assert legacy.entry_point.load_calls == 1
    assert legacy.behavior.calls["is_active"] == 1
    assert legacy.behavior.calls["driver_constructor"] == 1
    assert legacy.behavior.calls["get_current_target"] == 1


def test_concurrent_compiler_and_runtime_finish_on_the_same_record(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("compiler-runtime-race")
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "race", 32)
    barrier = threading.Barrier(2)
    compiler_future = _submit_daemon(
        lambda: (
            barrier.wait(timeout=5),
            f5_runtime.backends.make_backend(target),
        )[1]
    )
    driver_future = _submit_daemon(
        lambda: (
            barrier.wait(timeout=5),
            f5_runtime.runtime_driver.driver.default,
        )[1]
    )
    compiler = _bounded_result(compiler_future)
    driver = _bounded_result(driver_future)
    record = _record(f5_runtime, legacy)
    decision = f5_runtime.registry.get_selection("legacyv36")
    assert type(compiler) is legacy.compiler_cls
    assert type(driver) is legacy.driver_cls
    assert decision is not None and decision.record_id == record.record_id
    assert record.state is PluginLifecycleState.ACTIVE
    assert f5_runtime.backends.backends[legacy.name].record_id == record.record_id


def test_reset_during_compiler_probe_invalidates_the_entire_candidate_scan(
    f5_runtime: _Harness,
) -> None:
    blocked = f5_runtime.legacy("aa-reset-compiler", supports=False)
    winner = f5_runtime.legacy("zz-reset-compiler", supports=True)
    blocked.behavior.supports_entered = threading.Event()
    blocked.behavior.supports_release = threading.Event()
    f5_runtime.install(blocked, winner)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "reset", 32)
    future = _submit_daemon(lambda: f5_runtime.backends.make_backend(target))
    assert blocked.behavior.supports_entered.wait(5)
    assert f5_runtime.registry.reset() == ()
    blocked.behavior.supports_release.set()
    with pytest.raises(BackendPluginLifecycleError):
        _bounded_result(future)
    assert winner.entry_point.load_calls == 0
    assert winner.behavior.calls["compiler_constructor"] == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_reset_during_runtime_probe_invalidates_the_entire_candidate_scan(
    f5_runtime: _Harness,
) -> None:
    blocked = f5_runtime.legacy("aa-reset-runtime", active=False)
    winner = f5_runtime.legacy("zz-reset-runtime", active=True)
    blocked.behavior.active_entered = threading.Event()
    blocked.behavior.active_release = threading.Event()
    f5_runtime.install(blocked, winner)
    future = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    assert blocked.behavior.active_entered.wait(5)
    assert f5_runtime.registry.reset() == ()
    blocked.behavior.active_release.set()
    with pytest.raises(BackendPluginLifecycleError):
        _bounded_result(future)
    assert winner.entry_point.load_calls == 0
    assert winner.behavior.calls["driver_constructor"] == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.runtime_driver.driver._default is None
    assert f5_runtime.runtime_driver.driver._active is None


def test_reset_during_inactive_manifest_probe_cannot_publish_legacy(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("reset-inactive", target_name="manifest-only")
    manifest.behavior.active = False
    manifest.behavior.active_entered = threading.Event()
    manifest.behavior.active_release = threading.Event()
    legacy = f5_runtime.legacy("reset-fallback")
    f5_runtime.install(manifest, legacy)
    future = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    assert manifest.behavior.active_entered.wait(5)
    assert f5_runtime.registry.reset() == ()
    manifest.behavior.active_release.set()
    with pytest.raises(BackendPluginLifecycleError):
        _bounded_result(future)
    assert legacy.entry_point.load_calls == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}
    assert f5_runtime.runtime_driver.driver._default is None
    assert f5_runtime.runtime_driver.driver._active is None


def test_reset_during_compiler_constructor_discards_the_result(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("reset-compiler-constructor")
    legacy.behavior.compiler_constructor_entered = threading.Event()
    legacy.behavior.compiler_constructor_release = threading.Event()
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "reset", 32)
    future = _submit_daemon(lambda: f5_runtime.backends.make_backend(target))
    assert legacy.behavior.compiler_constructor_entered.wait(5)
    assert f5_runtime.registry.reset() == ()
    legacy.behavior.compiler_constructor_release.set()
    with pytest.raises(BackendPluginLifecycleError):
        _bounded_result(future)
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_reset_during_driver_constructor_discards_the_result(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("reset-driver-constructor")
    legacy.behavior.driver_constructor_entered = threading.Event()
    legacy.behavior.driver_constructor_release = threading.Event()
    f5_runtime.install(legacy)
    future = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    assert legacy.behavior.driver_constructor_entered.wait(5)
    assert f5_runtime.registry.reset() == ()
    legacy.behavior.driver_constructor_release.set()
    with pytest.raises(BackendPluginLifecycleError):
        _bounded_result(future)
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}
    assert f5_runtime.runtime_driver.driver._default is None
    assert f5_runtime.runtime_driver.driver._active is None


def test_probe_registry_reentry_does_not_deadlock(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("reentry")
    legacy.behavior.supports_callback = f5_runtime.registry.diagnostics
    legacy.behavior.active_callback = f5_runtime.registry.list
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "reentry", 32)
    compiler = _bounded_result(
        _submit_daemon(lambda: f5_runtime.backends.make_backend(target))
    )
    driver = _bounded_result(
        _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    )
    assert type(compiler) is legacy.compiler_cls
    assert type(driver) is legacy.driver_cls


def test_explicit_legacy_driver_mismatch_does_not_select_another_record(
    f5_runtime: _Harness,
) -> None:
    other = f5_runtime.legacy("explicit-other", supports=True)
    supplied = f5_runtime.legacy("explicit-supplied", supports=False)
    f5_runtime.install(other, supplied)
    instance = supplied.driver_cls()
    with pytest.raises(BackendPluginSelectionError):
        f5_runtime.runtime_driver.driver.set_active(instance)
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}
    assert f5_runtime.runtime_driver.driver._active is None


def test_explicit_legacy_driver_publishes_its_exact_record(
    f5_runtime: _Harness,
) -> None:
    other = f5_runtime.legacy("explicit-other-valid")
    supplied = f5_runtime.legacy("explicit-supplied-valid")
    f5_runtime.install(other, supplied)
    instance = supplied.driver_cls()
    f5_runtime.runtime_driver.driver.set_active(instance)
    record = _record(f5_runtime, supplied)
    decision = f5_runtime.registry.get_selection("legacyv36")
    assert decision is not None and decision.record_id == record.record_id
    assert record.state is PluginLifecycleState.ACTIVE
    assert f5_runtime.runtime_driver.driver.active is instance
    assert f5_runtime.backends.backends[supplied.name].record_id == record.record_id


def test_repeated_reset_is_idempotent_and_keeps_lazy_contracts(
    f5_runtime: _Harness,
) -> None:
    legacy = f5_runtime.legacy("multi-reset")
    f5_runtime.install(legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "reset", 32)
    f5_runtime.backends.make_backend(target)
    for _ in range(3):
        assert f5_runtime.registry.reset() == ()
        assert f5_runtime.backends.backends == {}
        assert f5_runtime.runtime_driver.driver._default is None
        assert f5_runtime.runtime_driver.driver._active is None
    assert type(f5_runtime.backends.make_backend(target)) is legacy.compiler_cls
    assert legacy.entry_point.load_calls == 2


def test_manifest_priority_conflict_never_imports_legacy(
    f5_runtime: _Harness,
) -> None:
    first = f5_runtime.manifest("priority-first")
    second = f5_runtime.manifest("priority-second")
    legacy = f5_runtime.legacy("priority-legacy")
    f5_runtime.install(first, second, legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "priority", 32)
    with pytest.raises(BackendPluginSelectionError) as caught:
        f5_runtime.backends.make_backend(target)
    assert caught.value.field in {"priority", "targets"}
    assert legacy.entry_point.load_calls == 0


def test_runtime_selection_lease_is_opaque_one_shot_and_reset_bound(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("opaque-runtime-lease")
    f5_runtime.install(manifest)

    lease = f5_runtime.registry.prepare_runtime_selection("legacyv36")
    copied = replace(lease, target="undeclared")
    with pytest.raises(BackendPluginLifecycleError) as copied_error:
        f5_runtime.registry.commit_runtime_selection(copied)
    assert copied_error.value.field == "runtime selection proposal"
    assert f5_runtime.registry.get_selection("undeclared") is None

    object.__setattr__(lease, "target", "undeclared")
    with pytest.raises(BackendPluginLifecycleError) as changed_error:
        f5_runtime.registry.commit_runtime_selection(lease)
    assert changed_error.value.field == "runtime selection proposal"
    assert f5_runtime.registry.get_selection("undeclared") is None

    fresh = f5_runtime.registry.prepare_runtime_selection("legacyv36")
    decision = f5_runtime.registry.commit_runtime_selection(fresh)
    assert decision.record_id == fresh.record_id
    with pytest.raises(BackendPluginLifecycleError):
        f5_runtime.registry.commit_runtime_selection(fresh)

    assert f5_runtime.registry.reset() == ()
    stale = f5_runtime.registry.prepare_runtime_selection("legacyv36")
    assert f5_runtime.registry.reset() == ()
    with pytest.raises(BackendPluginLifecycleError) as stale_error:
        f5_runtime.registry.commit_runtime_selection(stale)
    assert stale_error.value.field in {
        "runtime selection proposal",
        "registry generation",
    }


def test_runtime_selection_commit_failure_is_atomic(
    f5_runtime: _Harness,
) -> None:
    first = f5_runtime.manifest("atomic-first", priority=10)
    replacement = f5_runtime.manifest("atomic-replacement", priority=20)
    active = f5_runtime.manifest(
        "atomic-active",
        target_name="other-target",
    )
    f5_runtime.install(first, replacement, active)
    records = {
        record.entry_point_name: record for record in f5_runtime.registry.validate()
    }
    first_decision = f5_runtime.registry.select(
        "legacyv36",
        explicit_selector=records[first.name].record_id,
    )
    active_decision = f5_runtime.registry.select(
        "other-target",
        explicit_selector=records[active.name].record_id,
    )
    f5_runtime.registry.activate(active_decision.record_id)
    lease = f5_runtime.registry.prepare_runtime_selection(
        "legacyv36",
        explicit_selector=records[replacement.name].record_id,
    )

    with pytest.raises(BackendPluginConflictError):
        f5_runtime.registry.commit_runtime_selection(lease, activate=True)

    after = f5_runtime.registry.get_selection("legacyv36")
    first_record = f5_runtime.registry.inspect(first_decision.record_id)
    assert after is not None and after.record_id == first_decision.record_id
    assert first_record.state is PluginLifecycleState.SELECTED
    assert first_record.selected_targets == ("legacyv36",)


@pytest.mark.parametrize(
    ("error_attribute", "expected_field"),
    [
        ("active_error", "driver_cls.is_active"),
        ("driver_constructor_error", "driver_cls.__init__"),
        ("current_target_error", "driver_cls.get_current_target"),
    ],
)
def test_manifest_runtime_failures_do_not_publish_any_runtime_state(
    f5_runtime: _Harness,
    error_attribute: str,
    expected_field: str,
) -> None:
    manifest = f5_runtime.manifest("manifest-runtime-failure")
    setattr(manifest.behavior, error_attribute, RuntimeError("runtime sentinel"))
    f5_runtime.install(manifest)
    with pytest.raises(BackendPluginLifecycleError) as caught:
        _ = f5_runtime.runtime_driver.driver.default
    assert caught.value.field == expected_field
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}
    assert f5_runtime.runtime_driver.driver._default is None
    assert f5_runtime.runtime_driver.driver._active is None


def test_manifest_runtime_requires_exact_declared_gputarget_without_legacy_fallback(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest(
        "manifest-target-owner",
        target_name="manifest-only",
    )
    legacy = f5_runtime.legacy("target-mismatch-legacy")
    manifest.behavior.current_target = f5_runtime.backends.compiler.GPUTarget(
        "legacyv36", "wrong-owner", 32
    )
    f5_runtime.install(manifest, legacy)
    with pytest.raises(BackendPluginSelectionError) as caught:
        _ = f5_runtime.runtime_driver.driver.default
    assert caught.value.field == "driver_cls.get_current_target"
    assert legacy.entry_point.load_calls == 0
    assert f5_runtime.registry.get_selection("manifest-only") is None
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}


def test_multi_target_manifest_commits_only_current_target_and_keeps_driver_cache(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest(
        "multi-target-manifest",
        target_name="target-a",
        declared_targets=("target-a", "target-b"),
    )
    f5_runtime.install(manifest)
    first_driver = f5_runtime.runtime_driver.driver.default
    assert f5_runtime.runtime_driver.driver.active is first_driver
    assert f5_runtime.registry.get_selection("target-a") is not None
    assert f5_runtime.registry.get_selection("target-b") is None
    counts = manifest.behavior.calls.copy()

    target_b = f5_runtime.backends.compiler.GPUTarget("target-b", "b", 32)
    compiler = f5_runtime.backends.make_backend(target_b)
    assert type(compiler) is manifest.entry_point.loaded_root.compiler_cls
    assert f5_runtime.registry.get_selection("target-b") is not None
    assert f5_runtime.runtime_driver.driver.default is first_driver
    assert f5_runtime.runtime_driver.driver.active is first_driver
    for name in ("is_active", "driver_constructor", "get_current_target"):
        assert manifest.behavior.calls[name] == counts[name]


def test_concurrent_manifest_compiler_calls_share_one_published_record(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("concurrent-manifest-compiler")
    f5_runtime.install(manifest)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "threads", 32)
    barrier = threading.Barrier(8)
    futures = tuple(
        _submit_daemon(
            lambda: (
                barrier.wait(timeout=5),
                f5_runtime.backends.make_backend(target),
            )[1]
        )
        for _ in range(8)
    )
    compilers = tuple(_bounded_result(future) for future in futures)
    compiler_cls = manifest.entry_point.loaded_root.compiler_cls
    assert all(type(compiler) is compiler_cls for compiler in compilers)
    assert manifest.entry_point.load_calls == 1
    assert manifest.behavior.calls["supports_target"] == 8
    assert manifest.behavior.calls["compiler_constructor"] == 8
    assert len(f5_runtime.backends.backends) == 1


def test_concurrent_manifest_compiler_and_runtime_finish_on_same_record(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("manifest-compiler-runtime-race")
    f5_runtime.install(manifest)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "race", 32)
    barrier = threading.Barrier(2)
    compiler_future = _submit_daemon(
        lambda: (
            barrier.wait(timeout=5),
            f5_runtime.backends.make_backend(target),
        )[1]
    )
    driver_future = _submit_daemon(
        lambda: (
            barrier.wait(timeout=5),
            f5_runtime.runtime_driver.driver.default,
        )[1]
    )
    compiler = _bounded_result(compiler_future)
    driver = _bounded_result(driver_future)
    record = _manifest_record(f5_runtime, manifest)
    decision = f5_runtime.registry.get_selection("legacyv36")
    assert type(compiler) is record.compiler_cls
    assert type(driver) is record.driver_cls
    assert decision is not None and decision.record_id == record.record_id
    assert f5_runtime.registry.inspect(record.record_id).state is (
        PluginLifecycleState.ACTIVE
    )


@pytest.mark.parametrize("source", ["manifest", "legacy"])
def test_runtime_can_activate_exact_record_after_concurrent_compiler_commit(
    f5_runtime: _Harness,
    source: str,
) -> None:
    fixture = (
        f5_runtime.manifest("ordered-runtime-compiler")
        if source == "manifest"
        else f5_runtime.legacy("ordered-runtime-compiler")
    )
    fixture.behavior.active_entered = threading.Event()
    fixture.behavior.active_release = threading.Event()
    f5_runtime.install(fixture)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "ordered", 32)

    runtime = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    assert fixture.behavior.active_entered.wait(5)
    compiler = f5_runtime.backends.make_backend(target)
    compiler_decision = f5_runtime.registry.get_selection("legacyv36")
    assert compiler_decision is not None
    fixture.behavior.active_release.set()
    driver = _bounded_result(runtime)

    final = f5_runtime.registry.get_selection("legacyv36")
    assert final is not None
    assert final.record_id == compiler_decision.record_id
    assert final.record.state is PluginLifecycleState.ACTIVE
    assert type(compiler) is final.record.compiler_cls
    assert type(driver) is final.record.driver_cls
    assert fixture.behavior.calls["driver_constructor"] == 1
    assert fixture.entry_point.load_calls == 1


def test_inactive_manifest_probe_cannot_replace_concurrent_compiler_selection(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("ordered-inactive-manifest")
    manifest.behavior.active = False
    manifest.behavior.active_entered = threading.Event()
    manifest.behavior.active_release = threading.Event()
    legacy = f5_runtime.legacy("forbidden-concurrent-fallback")
    f5_runtime.install(manifest, legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "ordered", 32)

    runtime = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    assert manifest.behavior.active_entered.wait(5)
    compiler = f5_runtime.backends.make_backend(target)
    manifest_record = _manifest_record(f5_runtime, manifest)
    manifest.behavior.active_release.set()
    with pytest.raises(BackendPluginSelectionError):
        _bounded_result(runtime)

    decision = f5_runtime.registry.get_selection("legacyv36")
    assert type(compiler) is manifest_record.compiler_cls
    assert decision is not None and decision.record_id == manifest_record.record_id
    assert legacy.entry_point.load_calls == 0
    assert f5_runtime.runtime_driver.driver._default is None
    assert f5_runtime.runtime_driver.driver._active is None


def test_legacy_runtime_scan_cannot_cross_concurrent_compiler_record(
    f5_runtime: _Harness,
) -> None:
    first = f5_runtime.legacy("aa-ordered-owner", active=False, supports=True)
    second = f5_runtime.legacy("zz-forbidden-owner", active=True, supports=False)
    first.behavior.active_entered = threading.Event()
    first.behavior.active_release = threading.Event()
    first.behavior.current_target = f5_runtime.backends.compiler.GPUTarget(
        "target-a", "a", 32
    )
    second.behavior.current_target = f5_runtime.backends.compiler.GPUTarget(
        "target-b", "b", 32
    )
    f5_runtime.install(first, second)
    target_a = f5_runtime.backends.compiler.GPUTarget("target-a", "a", 32)

    runtime = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    assert first.behavior.active_entered.wait(5)
    compiler = f5_runtime.backends.make_backend(target_a)
    first_decision = f5_runtime.registry.get_selection("target-a")
    assert first_decision is not None
    second.behavior.supports = True
    first.behavior.active_release.set()
    with pytest.raises(BackendPluginError):
        _bounded_result(runtime)

    assert type(compiler) is first.compiler_cls
    assert f5_runtime.registry.get_selection("target-b") is None
    assert second.behavior.calls["is_active"] == 0
    assert second.behavior.calls["driver_constructor"] == 0
    assert f5_runtime.runtime_driver.driver._default is None
    assert f5_runtime.runtime_driver.driver._active is None


@pytest.mark.parametrize("source", ["manifest", "legacy"])
def test_same_record_concurrent_targets_extend_selection_monotonically(
    f5_runtime: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    fixture = (
        f5_runtime.manifest(
            "concurrent-targets",
            target_name="target-a",
            declared_targets=("target-a", "target-b"),
        )
        if source == "manifest"
        else f5_runtime.legacy("concurrent-targets")
    )
    f5_runtime.install(fixture)
    target_a = f5_runtime.backends.compiler.GPUTarget("target-a", "a", 32)
    target_b = f5_runtime.backends.compiler.GPUTarget("target-b", "b", 32)
    entered = threading.Event()
    release = threading.Event()
    original_prune = f5_runtime.backends._prune_registry_backends
    call_lock = threading.Lock()
    calls = 0

    def block_first_prune(registry: Any) -> None:
        nonlocal calls
        with call_lock:
            calls += 1
            first_call = calls == 1
        if first_call:
            entered.set()
            assert release.wait(5)
        original_prune(registry)

    monkeypatch.setattr(
        f5_runtime.backends,
        "_prune_registry_backends",
        block_first_prune,
    )
    first = _submit_daemon(lambda: f5_runtime.backends.make_backend(target_a))
    assert entered.wait(5)
    second = f5_runtime.backends.make_backend(target_b)
    release.set()
    first_result = _bounded_result(first)

    decision_a = f5_runtime.registry.get_selection("target-a")
    decision_b = f5_runtime.registry.get_selection("target-b")
    assert decision_a is not None and decision_b is not None
    assert decision_a.record_id == decision_b.record_id
    assert type(first_result) is decision_a.record.compiler_cls
    assert type(second) is decision_b.record.compiler_cls


@pytest.mark.parametrize("manifest_first", [False, True])
def test_manifest_and_legacy_different_targets_are_order_independent(
    f5_runtime: _Harness,
    manifest_first: bool,
) -> None:
    manifest = f5_runtime.manifest(
        "disjoint-manifest",
        target_name="manifest-target",
    )
    legacy = f5_runtime.legacy("disjoint-legacy")
    legacy.behavior.current_target = f5_runtime.backends.compiler.GPUTarget(
        "legacy-target", "legacy", 32
    )
    f5_runtime.install(manifest, legacy)
    manifest_target = f5_runtime.backends.compiler.GPUTarget(
        "manifest-target", "manifest", 32
    )
    legacy_target = f5_runtime.backends.compiler.GPUTarget(
        "legacy-target", "legacy", 32
    )

    if manifest_first:
        manifest_compiler = f5_runtime.backends.make_backend(manifest_target)
        legacy_compiler = f5_runtime.backends.make_backend(legacy_target)
    else:
        legacy_compiler = f5_runtime.backends.make_backend(legacy_target)
        manifest_backend = f5_runtime.backends.get_backend(manifest_target)
        manifest_compiler = f5_runtime.backends.make_backend(manifest_target)
        assert (
            manifest_backend.record_id
            == _manifest_record(f5_runtime, manifest).record_id
        )

    manifest_decision = f5_runtime.registry.get_selection("manifest-target")
    legacy_decision = f5_runtime.registry.get_selection("legacy-target")
    assert manifest_decision is not None and legacy_decision is not None
    assert manifest_decision.record_id != legacy_decision.record_id
    assert type(manifest_compiler) is manifest_decision.record.compiler_cls
    assert type(legacy_compiler) is legacy_decision.record.compiler_cls
    assert manifest.entry_point.load_calls == 1
    assert legacy.entry_point.load_calls == 1


def test_driver_default_reentry_is_structured_and_never_deadlocks(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("default-reentry")
    manifest.behavior.active_callback = lambda: f5_runtime.runtime_driver.driver.default
    f5_runtime.install(manifest)
    future = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    with pytest.raises(BackendPluginLifecycleError) as caught:
        _bounded_result(future)
    assert caught.value.field == "driver.default"
    assert f5_runtime.runtime_driver.driver._default_initializing is False
    assert f5_runtime.registry.get_selection("legacyv36") is None


@pytest.mark.parametrize("entry_path", ["compiler", "runtime"])
def test_plugin_hook_cross_thread_backend_reentry_never_deadlocks(
    f5_runtime: _Harness,
    entry_path: str,
) -> None:
    manifest = f5_runtime.manifest("cross-thread-reentry")
    f5_runtime.install(manifest)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "nested", 32)
    callback_lock = threading.Lock()
    spawned = False
    nested_compilers = []

    def resolve_from_worker() -> None:
        nonlocal spawned
        with callback_lock:
            if spawned:
                return
            spawned = True
        nested = _submit_daemon(lambda: f5_runtime.backends.make_backend(target))
        nested_compilers.append(_bounded_result(nested))

    if entry_path == "compiler":
        manifest.behavior.supports_callback = resolve_from_worker
        outer = _submit_daemon(lambda: f5_runtime.backends.make_backend(target))
    else:
        manifest.behavior.active_callback = resolve_from_worker
        outer = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)

    result = _bounded_result(outer)
    record = _manifest_record(f5_runtime, manifest)
    assert len(nested_compilers) == 1
    assert type(nested_compilers[0]) is record.compiler_cls
    if entry_path == "compiler":
        assert type(result) is record.compiler_cls
    else:
        assert type(result) is record.driver_cls
    decision = f5_runtime.registry.get_selection("legacyv36")
    assert decision is not None and decision.record_id == record.record_id
    assert manifest.entry_point.load_calls == 1


def test_driver_default_waiters_observe_the_same_failed_attempt(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("default-wave")
    manifest.behavior.active_entered = threading.Event()
    manifest.behavior.active_release = threading.Event()
    calls = 0

    def fail_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("first wave")

    manifest.behavior.active_callback = fail_once
    f5_runtime.install(manifest)
    owner = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    assert manifest.behavior.active_entered.wait(5)
    waiter = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    for _ in range(100):
        if f5_runtime.runtime_driver.driver._default_waiters:
            break
        threading.Event().wait(0.01)
    assert f5_runtime.runtime_driver.driver._default_waiters
    manifest.behavior.active_release.set()
    errors = []
    for future in (owner, waiter):
        with pytest.raises(BackendPluginLifecycleError) as caught:
            _bounded_result(future)
        errors.append(caught.value)
    assert errors[0] is errors[1]
    assert errors[0].field == "driver_cls.is_active"

    manifest.behavior.active_entered = None
    manifest.behavior.active_release = None
    assert type(f5_runtime.runtime_driver.driver.default) is (
        manifest.entry_point.loaded_root.driver_cls
    )


def test_set_active_runs_f6_before_driver_target_callback(
    f5_runtime: _Harness,
) -> None:
    omitted = next(
        member
        for member in sorted(f5_runtime.backends.DriverBase.__abstractmethods__)
        if member != "get_current_target"
    )
    legacy = f5_runtime.legacy(
        "explicit-f6-order",
        omit_driver=omitted,
        structural=True,
    )
    f5_runtime.install(legacy)
    instance = legacy.driver_cls()
    with pytest.raises(BackendPluginInterfaceError):
        f5_runtime.runtime_driver.driver.set_active(instance)
    assert legacy.behavior.calls["get_current_target"] == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None


def test_set_active_manifest_does_not_import_unrelated_legacy(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("explicit-manifest")
    legacy = f5_runtime.legacy("explicit-unrelated-legacy")
    f5_runtime.install(manifest, legacy)
    instance = manifest.entry_point.loaded_root.driver_cls()
    f5_runtime.runtime_driver.driver.set_active(instance)
    record = _manifest_record(f5_runtime, manifest)
    decision = f5_runtime.registry.get_selection("legacyv36")
    assert f5_runtime.runtime_driver.driver._active is instance
    assert decision is not None and decision.record_id == record.record_id
    assert legacy.entry_point.load_calls == 0


def test_set_active_multitarget_manifest_commits_only_driver_target(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest(
        "explicit-multitarget-manifest",
        target_name="target-a",
        declared_targets=("target-a", "target-b"),
    )
    manifest.behavior.current_target = f5_runtime.backends.compiler.GPUTarget(
        "target-b", "explicit", 32
    )
    legacy = f5_runtime.legacy("explicit-multitarget-legacy")
    f5_runtime.install(manifest, legacy)

    instance = manifest.entry_point.loaded_root.driver_cls()
    f5_runtime.runtime_driver.driver.set_active(instance)

    record = _manifest_record(f5_runtime, manifest)
    assert f5_runtime.registry.get_selection("target-a") is None
    decision = f5_runtime.registry.get_selection("target-b")
    assert decision is not None and decision.record_id == record.record_id
    assert decision.record.state is PluginLifecycleState.ACTIVE
    assert legacy.entry_point.load_calls == 0


def test_compiler_published_manifest_cannot_be_replaced_by_legacy_runtime(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("compiler-published-manifest")
    manifest.behavior.active = False
    legacy = f5_runtime.legacy("runtime-fallback-legacy")
    f5_runtime.install(manifest, legacy)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "bound", 32)

    compiler = f5_runtime.backends.make_backend(target)
    manifest_record = _manifest_record(f5_runtime, manifest)
    first = f5_runtime.registry.get_selection("legacyv36")
    assert type(compiler) is manifest_record.compiler_cls
    assert first is not None and first.record_id == manifest_record.record_id

    with pytest.raises(BackendPluginSelectionError) as caught:
        _ = f5_runtime.runtime_driver.driver.default

    assert caught.value.field == "driver_cls.is_active"
    final = f5_runtime.registry.get_selection("legacyv36")
    assert final is not None and final.record_id == manifest_record.record_id
    assert legacy.entry_point.load_calls == 0
    assert f5_runtime.runtime_driver.driver._default is None
    assert f5_runtime.runtime_driver.driver._active is None


def test_manual_runtime_cannot_split_from_published_governed_legacy(
    f5_runtime: _Harness,
) -> None:
    governed = f5_runtime.legacy("governed-pair")
    manual = f5_runtime.legacy("manual-pair")
    f5_runtime.install(governed)
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "pair", 32)
    f5_runtime.backends.make_backend(target)
    f5_runtime.backends.backends["manual"] = f5_runtime.backends.Backend(
        compiler=manual.compiler_cls,
        driver=manual.driver_cls,
    )

    driver = f5_runtime.runtime_driver.driver.default
    assert type(driver) is governed.driver_cls
    assert manual.behavior.calls["is_active"] == 0
    manual_instance = manual.driver_cls()
    with pytest.raises(BackendPluginSelectionError):
        f5_runtime.runtime_driver.driver.set_active(manual_instance)
    assert f5_runtime.runtime_driver.driver._active is None
    assert f5_runtime.registry.get_selection("legacyv36").entry_point_name == (
        governed.name
    )


def test_manual_runtime_cannot_split_across_targets_from_governed_legacy(
    f5_runtime: _Harness,
) -> None:
    governed = f5_runtime.legacy("governed-cross-target")
    manual = f5_runtime.legacy("manual-cross-target")
    f5_runtime.install(governed)
    target_a = f5_runtime.backends.compiler.GPUTarget("target-a", "a", 32)
    target_b = f5_runtime.backends.compiler.GPUTarget("target-b", "b", 32)
    governed.behavior.current_target = target_a
    manual.behavior.current_target = target_b
    compiler_a = f5_runtime.backends.make_backend(target_a)
    f5_runtime.backends.backends["manual"] = f5_runtime.backends.Backend(
        compiler=manual.compiler_cls,
        driver=manual.driver_cls,
    )

    with pytest.raises(BackendPluginSelectionError):
        f5_runtime.runtime_driver.driver.set_active(manual.driver_cls())

    assert f5_runtime.runtime_driver.driver._active is None
    decision_a = f5_runtime.registry.get_selection("target-a")
    assert decision_a is not None
    assert decision_a.entry_point_name == governed.name
    assert type(compiler_a) is governed.compiler_cls
    compiler_b = f5_runtime.backends.make_backend(target_b)
    decision_b = f5_runtime.registry.get_selection("target-b")
    assert decision_b is not None
    assert decision_b.record_id == decision_a.record_id
    assert type(compiler_b) is governed.compiler_cls


def test_unselected_governed_legacy_blocks_manual_runtime_activation(
    f5_runtime: _Harness,
) -> None:
    governed = f5_runtime.legacy("governed-unselected")
    manual = f5_runtime.legacy("manual-before-governed")
    f5_runtime.install(governed)
    f5_runtime.backends.backends["manual"] = f5_runtime.backends.Backend(
        compiler=manual.compiler_cls,
        driver=manual.driver_cls,
    )

    with pytest.raises(BackendPluginSelectionError):
        f5_runtime.runtime_driver.driver.set_active(manual.driver_cls())

    assert manual.behavior.calls["get_current_target"] == 0
    assert f5_runtime.runtime_driver.driver._active is None
    assert f5_runtime.registry.get_selection("legacyv36") is None


def test_explicit_selector_never_falls_back_to_supplied_manual_driver(
    f5_runtime: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governed = f5_runtime.legacy("selector-governed")
    manual = f5_runtime.legacy("selector-manual")
    f5_runtime.install(governed)
    f5_runtime.backends.backends["manual"] = f5_runtime.backends.Backend(
        compiler=manual.compiler_cls,
        driver=manual.driver_cls,
    )
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, governed.name)

    with pytest.raises(BackendPluginSelectionError):
        f5_runtime.runtime_driver.driver.set_active(manual.driver_cls())

    assert manual.behavior.calls["get_current_target"] == 0
    assert manual.behavior.calls["supports_target"] == 0
    assert f5_runtime.runtime_driver.driver._active is None
    assert f5_runtime.registry.get_selection("legacyv36") is None


def test_governed_legacy_miss_never_falls_back_to_manual_compiler(
    f5_runtime: _Harness,
) -> None:
    governed = f5_runtime.legacy("governed-no-support")
    governed.behavior.supports = False
    manual = f5_runtime.legacy("manual-supports")
    f5_runtime.install(governed)
    f5_runtime.backends.backends["manual"] = f5_runtime.backends.Backend(
        compiler=manual.compiler_cls,
        driver=manual.driver_cls,
    )
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "miss", 32)

    with pytest.raises(backend_api.BackendPluginNoCandidateError):
        f5_runtime.backends.make_backend(target)

    assert governed.behavior.calls["supports_target"] == 1
    assert manual.behavior.calls["supports_target"] == 0
    assert manual.behavior.calls["compiler_constructor"] == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None


def test_manual_compiler_probe_failure_is_structured_and_foreign_values_ignored(
    f5_runtime: _Harness,
) -> None:
    manual = f5_runtime.legacy("manual-probe-error")
    manual.behavior.supports_error = ValueError("manual probe sentinel")
    f5_runtime.backends.backends["foreign"] = object()
    f5_runtime.backends.backends["manual"] = f5_runtime.backends.Backend(
        compiler=manual.compiler_cls,
        driver=manual.driver_cls,
    )
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "manual", 32)

    with pytest.raises(BackendPluginLifecycleError) as caught:
        f5_runtime.backends.make_backend(target)

    assert caught.value.field == "compiler_cls.supports_target"
    assert caught.value.actual == "builtins.ValueError"
    assert manual.behavior.calls["compiler_constructor"] == 0


def test_manual_callable_fields_fail_structurally_without_renderer_traceback(
    f5_runtime: _Harness,
) -> None:
    def compiler_factory(_target: Any) -> object:
        return object()

    compiler_factory.supports_target = lambda _target: True

    def driver_factory() -> object:
        return object()

    driver_factory.is_active = lambda: True
    f5_runtime.backends.backends["manual-callables"] = f5_runtime.backends.Backend(
        compiler=compiler_factory,
        driver=driver_factory,
    )
    target = f5_runtime.backends.compiler.GPUTarget("legacyv36", "manual", 32)

    with pytest.raises(BackendPluginSelectionError) as compiler_error:
        f5_runtime.backends.make_backend(target)
    assert compiler_error.value.field == "compiler_cls.__new__"
    assert compiler_error.value.expected == "<non-class builtins.function>"

    with pytest.raises(BackendPluginSelectionError) as driver_error:
        _ = f5_runtime.runtime_driver.driver.default
    assert driver_error.value.field == "driver_cls.__new__"
    assert driver_error.value.expected == "<non-class builtins.function>"


def test_unrelated_manifest_record_blocks_manual_compiler_and_runtime(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest(
        "unrelated-manifest",
        target_name="target-a",
    )
    manifest.behavior.active = False
    manual = f5_runtime.legacy("manual-target-b")
    manual.behavior.current_target = f5_runtime.backends.compiler.GPUTarget(
        "target-b", "manual", 32
    )
    f5_runtime.install(manifest)
    f5_runtime.backends.backends["manual"] = f5_runtime.backends.Backend(
        compiler=manual.compiler_cls,
        driver=manual.driver_cls,
    )
    target_b = f5_runtime.backends.compiler.GPUTarget("target-b", "b", 32)

    with pytest.raises(backend_api.BackendPluginNoCandidateError):
        f5_runtime.backends.make_backend(target_b)
    with pytest.raises(BackendPluginSelectionError):
        f5_runtime.runtime_driver.driver.set_active(manual.driver_cls())

    assert manual.behavior.calls["supports_target"] == 0
    assert manual.behavior.calls["get_current_target"] == 0
    assert f5_runtime.registry.get_selection("target-b") is None
    assert f5_runtime.runtime_driver.driver._active is None


def test_set_active_manual_result_cannot_cross_reset_lifecycle_epoch(
    f5_runtime: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manual = f5_runtime.legacy("manual-reset-window")
    f5_runtime.backends.backends["manual"] = f5_runtime.backends.Backend(
        compiler=manual.compiler_cls,
        driver=manual.driver_cls,
    )
    entered = threading.Event()
    release = threading.Event()
    original_activate = f5_runtime.backends.activate_backend

    def blocked_activate(backend: Any, *, target: Any = None) -> Any:
        entered.set()
        assert release.wait(5)
        return original_activate(backend, target=target)

    monkeypatch.setattr(
        f5_runtime.backends,
        "activate_backend",
        blocked_activate,
    )
    old_lifecycle_epoch = f5_runtime.registry.lifecycle_epoch
    activation = _submit_daemon(
        lambda: f5_runtime.runtime_driver.driver.set_active(manual.driver_cls())
    )
    assert entered.wait(5)

    cache_lock = f5_runtime.backends._backend_cache_lock
    cache_lock.acquire()
    try:
        reset = _submit_daemon(f5_runtime.registry.reset)
        for _ in range(200):
            if f5_runtime.registry.lifecycle_epoch != old_lifecycle_epoch:
                break
            threading.Event().wait(0.01)
        assert f5_runtime.registry.lifecycle_epoch != old_lifecycle_epoch
        release.set()
        threading.Event().wait(0.05)
        assert f5_runtime.runtime_driver.driver._active is None
    finally:
        cache_lock.release()

    assert _bounded_result(reset) == ()
    with pytest.raises(BackendPluginLifecycleError) as caught:
        _bounded_result(activation)
    assert caught.value.field in {"registry lifecycle epoch", "validate"}
    assert f5_runtime.runtime_driver.driver._active is None


def test_real_installed_v36_legacy_wheel_is_lazy_governed_and_resettable(
    f5_runtime: _Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert hashlib.sha256(LEGACY_WHEEL.read_bytes()).hexdigest() == (
        LEGACY_WHEEL_SHA256
    )
    site = tmp_path / "site-packages"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--target",
            str(site),
            str(LEGACY_WHEEL),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    distributions = tuple(importlib.metadata.distributions(path=[str(site)]))
    distribution = next(
        item
        for item in distributions
        if item.metadata["Name"] == "t63-v36-legacy-oracle"
    )
    event_log = tmp_path / "events.log"
    monkeypatch.setenv("T63_F5_EVENT_LOG", str(event_log))
    monkeypatch.setenv("T63_F5_DRIVER_ACTIVE", "1")
    sys.path.insert(0, str(site))
    try:
        assert f5_runtime.registry.reset() == ()
        f5_runtime.distributions[:] = [distribution]
        assert f5_runtime.backends._discover_backends() == {}
        record = f5_runtime.registry.list()[0]
        assert record.source is PluginSource.LEGACY
        assert record.compatibility_status is (
            PluginCompatibilityStatus.LEGACY_UNVERIFIED
        )
        f5_runtime.registry.inspect(record.record_id)
        f5_runtime.registry.diagnostics()
        assert not event_log.exists()

        target = f5_runtime.backends.compiler.GPUTarget(
            "legacyv36", "installed-wheel", 32
        )
        compiler = f5_runtime.backends.make_backend(target)
        driver = f5_runtime.runtime_driver.driver.default
        decision = f5_runtime.registry.get_selection("legacyv36")
        assert type(compiler).__module__ == "t63_v36_legacy_oracle.compiler"
        assert type(driver).__module__ == "t63_v36_legacy_oracle.driver"
        assert decision is not None and decision.record_id == record.record_id
        assert f5_runtime.backends.backends["baseline-legacy"].record_id == (
            record.record_id
        )
        events = event_log.read_text(encoding="utf-8").splitlines()
        assert events.count("package_import") == 1
        assert events.count("compiler_module_import") == 1
        assert events.count("driver_module_import") == 1

        assert f5_runtime.registry.reset() == ()
        assert f5_runtime.backends.backends == {}
        assert f5_runtime.runtime_driver.driver._default is None
        assert f5_runtime.runtime_driver.driver._active is None
        before = len(event_log.read_text(encoding="utf-8").splitlines())
        second = f5_runtime.backends.make_backend(target)
        assert type(second) is type(compiler)
        assert len(event_log.read_text(encoding="utf-8").splitlines()) > before
        assert f5_runtime.registry.inspect(record.record_id).compatibility_status is (
            PluginCompatibilityStatus.LEGACY_UNVERIFIED
        )
    finally:
        sys.path.remove(str(site))
        for name in tuple(sys.modules):
            if name == "t63_v36_legacy_oracle" or name.startswith(
                "t63_v36_legacy_oracle."
            ):
                sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("entered_attribute", "release_attribute", "downstream_counter"),
    [
        ("active_entered", "active_release", "driver_constructor"),
        (
            "driver_constructor_entered",
            "driver_constructor_release",
            "get_current_target",
        ),
        (
            "current_target_entered",
            "current_target_release",
            "supports_target",
        ),
    ],
)
def test_reset_stops_manifest_runtime_after_each_plugin_stage(
    f5_runtime: _Harness,
    entered_attribute: str,
    release_attribute: str,
    downstream_counter: str,
) -> None:
    manifest = f5_runtime.manifest("manifest-reset-stage")
    entered = threading.Event()
    release = threading.Event()
    setattr(manifest.behavior, entered_attribute, entered)
    setattr(manifest.behavior, release_attribute, release)
    f5_runtime.install(manifest)

    future = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.default)
    assert entered.wait(5)
    assert f5_runtime.registry.reset() == ()
    release.set()
    with pytest.raises(BackendPluginLifecycleError):
        _bounded_result(future)
    assert manifest.behavior.calls[downstream_counter] == 0
    assert f5_runtime.registry.get_selection("legacyv36") is None
    assert f5_runtime.backends.backends == {}
    assert f5_runtime.runtime_driver.driver._default is None
    assert f5_runtime.runtime_driver.driver._active is None


def test_reset_started_before_driver_cache_hook_never_returns_stale_fast_path(
    f5_runtime: _Harness,
) -> None:
    manifest = f5_runtime.manifest("delayed-reset-hook")
    f5_runtime.install(manifest)
    old_default = f5_runtime.runtime_driver.driver.default
    assert f5_runtime.runtime_driver.driver.active is old_default
    old_epoch = f5_runtime.registry.lifecycle_epoch

    cache_lock = f5_runtime.backends._backend_cache_lock
    cache_lock.acquire()
    try:
        reset_future = _submit_daemon(f5_runtime.registry.reset)
        for _ in range(200):
            if f5_runtime.registry.lifecycle_epoch != old_epoch:
                break
            threading.Event().wait(0.01)
        assert f5_runtime.registry.lifecycle_epoch != old_epoch
        default_future = _submit_daemon(
            lambda: f5_runtime.runtime_driver.driver.default
        )
        active_future = _submit_daemon(lambda: f5_runtime.runtime_driver.driver.active)
        threading.Event().wait(0.05)
        for future in (default_future, active_future):
            if future.done() and future.exception() is None:
                assert future.result() is not old_default
    finally:
        cache_lock.release()

    assert _bounded_result(reset_future) == ()
    results = []
    for future in (default_future, active_future):
        try:
            results.append(_bounded_result(future))
        except BackendPluginLifecycleError:
            pass
    assert all(result is not old_default for result in results)
    assert f5_runtime.runtime_driver.driver._default is not old_default
    assert f5_runtime.runtime_driver.driver._active is not old_default


def test_reset_active_cannot_publish_after_registry_reset_starts(
    f5_runtime: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = f5_runtime.manifest("reset-active-window")
    f5_runtime.install(manifest)
    _ = f5_runtime.runtime_driver.driver.default
    f5_runtime.runtime_driver.driver._active = None
    f5_runtime.runtime_driver.driver._active_registry_generation = None
    f5_runtime.runtime_driver.driver._active_registry_lifecycle_epoch = None
    old_lifecycle_epoch = f5_runtime.registry.lifecycle_epoch
    default_returned = threading.Event()
    release_default = threading.Event()
    default_property = f5_runtime.runtime_driver.DriverConfig.default
    original_default = default_property.fget
    assert original_default is not None

    def blocked_default(config: Any) -> Any:
        result = original_default(config)
        default_returned.set()
        assert release_default.wait(5)
        return result

    monkeypatch.setattr(
        f5_runtime.runtime_driver.DriverConfig,
        "default",
        property(blocked_default),
    )
    reset_active = _submit_daemon(f5_runtime.runtime_driver.driver.reset_active)
    assert default_returned.wait(5)

    cache_lock = f5_runtime.backends._backend_cache_lock
    cache_lock.acquire()
    try:
        reset = _submit_daemon(f5_runtime.registry.reset)
        for _ in range(200):
            if f5_runtime.registry.lifecycle_epoch != old_lifecycle_epoch:
                break
            threading.Event().wait(0.01)
        assert f5_runtime.registry.lifecycle_epoch != old_lifecycle_epoch
        release_default.set()
        with pytest.raises(BackendPluginLifecycleError) as caught:
            _bounded_result(reset_active)
        assert caught.value.field == "registry lifecycle epoch"
        assert f5_runtime.runtime_driver.driver._active is None
        assert f5_runtime.runtime_driver.driver._default is None
    finally:
        cache_lock.release()

    assert _bounded_result(reset) == ()
    assert f5_runtime.runtime_driver.driver._active is None
    assert f5_runtime.runtime_driver.driver._default is None
