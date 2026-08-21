"""Acceptance oracles for the governed Triton 3.3 Legacy fallback bridge.

The fast fixtures model real entry-point records without importing plugin code
during discovery.  The final three tests install a standards-compliant wheel
without a Manifest and exercise real ``importlib.metadata`` objects in an
isolated subprocess.  T10.2 and physical hardware are intentionally absent.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import importlib
import json
import os
import subprocess
import sys
import threading
import types
import zipfile
from abc import abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

import triton_anchor.backends as backend_api
from triton_anchor.backends import (
    BackendPluginCapabilityError,
    BackendPluginCompatibilityError,
    BackendPluginConflictError,
    BackendPluginError,
    BackendPluginInterfaceError,
    BackendPluginLifecycleError,
    BackendPluginManifestError,
    BackendPluginRegistry,
    BackendPluginSelectionError,
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
    collect_core_environment,
)


REPOSITORY = Path(__file__).resolve().parents[2]
TRITON_PACKAGE = REPOSITORY / "triton/python/triton"
TRITON_COMMIT = "523a1b235b213bc192f2d5a8999add5bf2d0fea5"
E2E_PROBE = Path(__file__).with_name("legacy_bridge_e2e_probe.py")


def _outcome(value: Any, *args: Any) -> Any:
    if isinstance(value, BaseException):
        raise value
    if callable(value):
        return value(*args)
    return value


@dataclass
class LegacySpec:
    name: str
    target: str
    supports: Any = True
    active: Any = True
    current_target: Any = None
    compiler_error: Any = None
    driver_error: Any = None
    load_error: BaseException | None = None
    invalid_compiler: bool = False
    invalid_driver: bool = False
    manifest: dict[str, Any] | None = None
    distribution_name: str | None = None
    calls: Counter[str] = field(default_factory=Counter)
    entry_point: Any = None
    distribution: Any = None
    compiler_cls: type | None = None
    driver_cls: type | None = None


class _EntryPoint:
    group = "triton.backends"

    def __init__(
        self,
        name: str,
        plugin: Any,
        load_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.value = f"t63_f5_{name}:plugin"
        self.plugin = plugin
        self.load_error = load_error
        self.load_calls = 0
        self.dist = None
        self._lock = threading.Lock()

    def load(self) -> Any:
        with self._lock:
            self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        return self.plugin


class _Distribution:
    def __init__(
        self,
        *,
        name: str,
        entry_point: _EntryPoint,
        root: Path,
        manifest: dict[str, Any] | None,
    ) -> None:
        self.metadata = {"Name": name}
        self.name = name
        self.version = "1.0.0"
        self.entry_points = (entry_point,)
        entry_point.dist = self
        self._manifest_path: Path | None = None
        if manifest is None:
            self.files = ()
        else:
            relative = Path("fixture") / "triton_anchor_backend.json"
            path = root / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            self.files = (str(relative),)
            self._manifest_path = path

    def locate_file(self, _path: Any) -> Path:
        assert self._manifest_path is not None
        return self._manifest_path

    def read_text(self, _filename: str) -> None:
        return None


def _manifest(
    entry_point: str,
    target: str,
    *,
    plugin_id: str | None = None,
    priority: int = 0,
    triton_version: str = ">=3.3,<3.4",
    capabilities: tuple[str, ...] = ("acceptance.f5",),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plugin = {
        "plugin_id": plugin_id or f"acceptance.f5.{entry_point}",
        "entry_point": entry_point,
        "backend_protocol": ">=1.0,<2.0",
        "requires_core": ">=0.2,<0.3",
        "requires_triton": {
            "version": triton_version,
            "commit": TRITON_COMMIT,
        },
        "targets": [target],
        "capabilities": list(capabilities),
        "isolation_mode": "python_only",
        "priority": priority,
    }
    plugin.update(dict(extra or {}))
    return {"schema_version": "1.0", "plugins": [plugin]}


@dataclass
class LegacyHarness:
    registry: BackendPluginRegistry
    distributions: list[Any]
    backends_module: types.ModuleType
    runtime_driver_module: types.ModuleType
    compiler_contract: type
    driver_contract: type
    target_cls: type
    root: Path

    def fixture(self, spec: LegacySpec) -> LegacySpec:
        calls = spec.calls
        supports_result = spec.supports
        active_result = spec.active
        current_target_result = (
            spec.current_target
            if spec.current_target is not None
            else spec.target
        )
        compiler_error = spec.compiler_error
        driver_error = spec.driver_error
        target_cls = self.target_cls

        class Compiler(self.compiler_contract):
            binary_ext = "f5bin"

            @classmethod
            def supports_target(cls, target):
                calls["supports_target"] += 1
                return _outcome(supports_result, target)

            def __init__(self, target):
                calls["compiler_constructor"] += 1
                if compiler_error is not None:
                    _outcome(compiler_error)
                super().__init__(target)

            def hash(self):
                return f"f5-{spec.name}"

            def parse_options(self, options):
                return options

            def add_stages(self, stages, options):
                return None

            def load_dialects(self, context):
                return None

            def get_module_map(self):
                return {}

        class Driver(self.driver_contract):
            @classmethod
            def is_active(cls):
                calls["is_active"] += 1
                return _outcome(active_result)

            def __init__(self):
                calls["driver_constructor"] += 1
                if driver_error is not None:
                    _outcome(driver_error)
                super().__init__()

            def get_current_target(self):
                calls["get_current_target"] += 1
                value = _outcome(current_target_result)
                if hasattr(value, "backend"):
                    return value
                return target_cls(str(value), "fixture-arch", 32)

            def get_active_torch_device(self):
                return "cpu"

            def get_benchmarker(self):
                return lambda _call, *, quantiles, **_kwargs: [
                    0.0 for _ in quantiles
                ]

        compiler_cls: type = Compiler
        driver_cls: type = Driver
        if spec.invalid_compiler:

            class InvalidCompiler(self.compiler_contract):
                @classmethod
                def supports_target(cls, target):
                    calls["supports_target"] += 1
                    return True

                def hash(self):
                    return "invalid"

                def parse_options(self, options):
                    return options

                def add_stages(self, stages, options):
                    return None

                def load_dialects(self, context):
                    return None

            compiler_cls = InvalidCompiler
        if spec.invalid_driver:

            class InvalidDriver(self.driver_contract):
                @classmethod
                def is_active(cls):
                    calls["is_active"] += 1
                    return True

                def get_current_target(self):
                    return target_cls(spec.target, "fixture-arch", 32)

                def get_active_torch_device(self):
                    return "cpu"

            driver_cls = InvalidDriver

        class Plugin:
            def initialize(self, _context: Mapping[str, Any]) -> None:
                calls["initialize"] += 1

            def shutdown(self) -> None:
                calls["shutdown"] += 1

        plugin = Plugin()
        plugin.compiler_cls = compiler_cls
        plugin.driver_cls = driver_cls
        entry_point = _EntryPoint(spec.name, plugin, spec.load_error)
        distribution = _Distribution(
            name=spec.distribution_name or f"t63-f5-{spec.name}",
            entry_point=entry_point,
            root=self.root,
            manifest=spec.manifest,
        )
        spec.entry_point = entry_point
        spec.distribution = distribution
        spec.compiler_cls = compiler_cls
        spec.driver_cls = driver_cls
        return spec

    def install(self, *specs: LegacySpec) -> None:
        assert self.registry.reset() == ()
        self.distributions[:] = [spec.distribution for spec in specs]

    def target(self, name: str):
        return self.target_cls(name, "fixture-arch", 32)

    def assert_unpublished(self, target: str) -> None:
        assert self.registry.get_selection(target) is None
        assert all(
            backend.record_id is None
            for backend in self.backends_module.backends.values()
        )
        assert self.runtime_driver_module.driver.default._obj is None


@pytest.fixture(scope="module")
def legacy_runtime(tmp_path_factory: pytest.TempPathFactory):
    registry_module = importlib.import_module(
        "triton_anchor.backends.registry"
    )
    previous_registry = registry_module.backend_plugin_registry
    previous_api_registry = backend_api.backend_plugin_registry
    previous_triton_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "triton" or name.startswith("triton.")
    }
    for name in tuple(previous_triton_modules):
        sys.modules.pop(name, None)

    distributions: list[Any] = []
    registry = BackendPluginRegistry(
        distribution_provider=lambda: tuple(distributions),
        environment_provider=collect_core_environment,
        preflight_profile="triton_version",
    )
    registry_module.backend_plugin_registry = registry
    backend_api.backend_plugin_registry = registry

    triton = types.ModuleType("triton")
    triton.__package__ = "triton"
    triton.__path__ = [str(TRITON_PACKAGE)]
    triton.__version__ = "3.3.0"
    sys.modules["triton"] = triton
    runtime = types.ModuleType("triton.runtime")
    runtime.__package__ = "triton.runtime"
    runtime.__path__ = [str(TRITON_PACKAGE / "runtime")]
    sys.modules["triton.runtime"] = runtime

    backends_module = importlib.import_module("triton.backends")
    runtime_driver_module = importlib.import_module("triton.runtime.driver")
    harness = LegacyHarness(
        registry=registry,
        distributions=distributions,
        backends_module=backends_module,
        runtime_driver_module=runtime_driver_module,
        compiler_contract=backends_module.BaseBackend,
        driver_contract=backends_module.DriverBase,
        target_cls=importlib.import_module(
            "triton.backends.compiler"
        ).GPUTarget,
        root=tmp_path_factory.mktemp("f5-legacy-fast"),
    )
    try:
        yield harness
    finally:
        registry.reset()
        registry_module.backend_plugin_registry = previous_registry
        backend_api.backend_plugin_registry = previous_api_registry
        for name in tuple(sys.modules):
            if name == "triton" or name.startswith("triton."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_triton_modules)


@pytest.fixture(autouse=True)
def _isolate_legacy_runtime(legacy_runtime: LegacyHarness):
    selector = os.environ.pop("TRITON_ANCHOR_BACKEND", None)
    assert legacy_runtime.registry.reset() == ()
    legacy_runtime.distributions.clear()
    legacy_runtime.backends_module.backends.clear()
    try:
        yield
    finally:
        assert legacy_runtime.registry.reset() == ()
        legacy_runtime.distributions.clear()
        legacy_runtime.backends_module.backends.clear()
        if selector is not None:
            os.environ["TRITON_ANCHOR_BACKEND"] = selector


def _legacy_record(harness: LegacyHarness, spec: LegacySpec):
    return next(
        record
        for record in harness.registry.list()
        if record.entry_point_name == spec.name
    )


def _materialize(harness: LegacyHarness, identifier: str):
    method = getattr(harness.registry, "materialize_legacy", None)
    assert callable(method), "Registry.materialize_legacy() is missing"
    return method(identifier)


def _lease_record(lease: Any) -> Any:
    return getattr(lease, "record", lease)


def _resolve_compiler(
    harness: LegacyHarness,
    target: Any,
    *,
    kernel_required_capabilities: Any = (),
):
    method = getattr(
        harness.backends_module,
        "resolve_legacy_compiler_for_target",
        None,
    )
    assert callable(method), (
        "triton.backends.resolve_legacy_compiler_for_target() is missing"
    )
    return method(
        target,
        kernel_required_capabilities=kernel_required_capabilities,
    )


def _install_manual_backend(
    harness: LegacyHarness,
    spec: LegacySpec,
) -> Any:
    """Publish one upstream-style manual mapping without a distribution."""
    backend = harness.backends_module.Backend(
        compiler=spec.compiler_cls,
        driver=spec.driver_cls,
    )
    harness.backends_module.backends[spec.name] = backend
    return backend


# ---------------------------------------------------------------------------
# Metadata-only discovery and exact Core materialization


def test_legacy_discovery_list_inspect_and_diagnostics_never_import(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("discovery_legacy", "discovery_target")
    )
    legacy_runtime.install(spec)

    assert legacy_runtime.backends_module._discover_backends() == {}
    records = legacy_runtime.registry.list()
    assert len(records) == 1
    inspected = legacy_runtime.registry.inspect(records[0].record_id)
    diagnostics = legacy_runtime.registry.diagnostics()

    assert spec.entry_point.load_calls == 0
    assert legacy_runtime.backends_module.backends == {}
    assert inspected.source is PluginSource.LEGACY
    assert (
        inspected.compatibility_status
        is PluginCompatibilityStatus.LEGACY_UNVERIFIED
    )
    payload = json.dumps(diagnostics, sort_keys=True).lower()
    assert "legacy_unverified" in payload
    assert '"source": "legacy"' in payload
    assert "manifest" in payload
    assert "migrat" in payload, diagnostics
    legacy_policy = diagnostics["plugins"][0]["legacy_compatibility"]
    assert legacy_policy["static_compatibility_proven"] is False
    assert set(legacy_policy["missing_static_proofs"]) >= {
        "protocol",
        "core",
        "triton",
        "llvm",
        "capabilities",
        "native_abi",
        "priority",
        "targets",
    }
    assert "manifest" in legacy_policy["remediation"].lower()


def test_registry_lists_only_legacy_records_without_loading_either_source(
    legacy_runtime: LegacyHarness,
) -> None:
    legacy = legacy_runtime.fixture(
        LegacySpec("listed_legacy", "listed_target")
    )
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "listed_manifest",
            "listed_target",
            manifest=_manifest("listed_manifest", "listed_target"),
        )
    )
    legacy_runtime.install(legacy, manifest)

    method = getattr(legacy_runtime.registry, "list_legacy_records", None)
    assert callable(method), "Registry.list_legacy_records() is missing"
    records = method()

    assert type(records) is tuple
    assert [record.record_id for record in records] == [
        _legacy_record(legacy_runtime, legacy).record_id
    ]
    assert all(record.source is PluginSource.LEGACY for record in records)
    assert legacy.entry_point.load_calls == 0
    assert manifest.entry_point.load_calls == 0


def test_registry_materializes_only_an_exact_legacy_record_key(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("exact_legacy", "exact_target")
    )
    legacy_runtime.install(spec)
    record = _legacy_record(legacy_runtime, spec)

    with pytest.raises(BackendPluginSelectionError):
        _materialize(legacy_runtime, spec.name)
    assert spec.entry_point.load_calls == 0

    lease = _materialize(legacy_runtime, record.registry_key)
    materialized = _lease_record(lease)
    assert materialized.record_id == record.record_id
    assert materialized.compiler_cls is spec.compiler_cls
    assert materialized.driver_cls is spec.driver_cls
    assert materialized.source is PluginSource.LEGACY
    assert (
        materialized.compatibility_status
        is PluginCompatibilityStatus.LEGACY_UNVERIFIED
    )
    assert spec.entry_point.load_calls == 1


def test_materialized_legacy_lease_cannot_publish_after_reset(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("stale_lease", "stale_target")
    )
    legacy_runtime.install(spec)
    record = _legacy_record(legacy_runtime, spec)
    lease = _materialize(legacy_runtime, record.record_id)

    assert legacy_runtime.registry.reset() == ()
    recreated = _legacy_record(legacy_runtime, spec)
    assert recreated.record_id == record.record_id
    assert recreated is not record
    method = getattr(
        legacy_runtime.registry, "select_materialized_legacy", None
    )
    assert callable(method), (
        "Registry.select_materialized_legacy() is missing"
    )
    with pytest.raises(BackendPluginLifecycleError) as caught:
        method(legacy_runtime.target(spec.target), lease=lease)
    assert caught.value.field == "registry lifecycle_epoch"
    assert spec.entry_point.load_calls == 1
    legacy_runtime.assert_unpublished(spec.target)


def test_manifest_multi_target_conditional_release_is_atomic(
    legacy_runtime: LegacyHarness,
) -> None:
    first_target = "conditional_release_a"
    second_target = "conditional_release_b"
    spec = legacy_runtime.fixture(
        LegacySpec(
            "conditional_release_manifest",
            first_target,
            manifest=_manifest(
                "conditional_release_manifest",
                first_target,
                extra={"targets": [first_target, second_target]},
            ),
        )
    )
    legacy_runtime.install(spec)

    legacy_runtime.registry.select(first_target, environment={})
    legacy_runtime.registry.select(second_target, environment={})
    decisions = (
        legacy_runtime.registry.get_selection(first_target),
        legacy_runtime.registry.get_selection(second_target),
    )
    generation = legacy_runtime.registry.generation

    released = legacy_runtime.registry.release_selections_if_current(
        decisions
    )

    assert len(released) == 1
    assert released[0].state is PluginLifecycleState.REGISTERED
    assert released[0].selected_targets == ()
    assert legacy_runtime.registry.get_selection(first_target) is None
    assert legacy_runtime.registry.get_selection(second_target) is None
    assert legacy_runtime.registry.generation == generation + 1
    with pytest.raises(BackendPluginLifecycleError):
        legacy_runtime.registry.release_selections_if_current(decisions)


def test_materialized_legacy_cannot_claim_kernel_capabilities(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("capability_legacy", "capability_target")
    )
    legacy_runtime.install(spec)

    with pytest.raises(BackendPluginCapabilityError) as caught:
        _resolve_compiler(
            legacy_runtime,
            legacy_runtime.target(spec.target),
            kernel_required_capabilities=("kernel.tensor_core",),
        )
    assert caught.value.field == "kernel_required_capabilities"
    assert spec.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(spec.target)


def test_materialized_legacy_target_property_runs_outside_registry_lock(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("target_property_lock_legacy", "target_property_lock_target")
    )
    legacy_runtime.install(spec)
    record = _legacy_record(legacy_runtime, spec)
    lease = _materialize(legacy_runtime, record.record_id)
    entered = threading.Event()
    release = threading.Event()
    listed = threading.Event()

    class BlockingTarget:
        @property
        def backend(self) -> str:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("target property barrier timed out")
            return spec.target

    def read_registry() -> tuple[Any, ...]:
        records = legacy_runtime.registry.list()
        listed.set()
        return records

    method = legacy_runtime.registry.select_materialized_legacy
    with ThreadPoolExecutor(max_workers=2) as executor:
        selecting = executor.submit(method, BlockingTarget(), lease=lease)
        assert entered.wait(timeout=1)
        reading = executor.submit(read_registry)
        try:
            assert listed.wait(timeout=1), (
                "Registry.list() was blocked by target.backend plugin code"
            )
        finally:
            release.set()
        assert reading.result(timeout=2)
        decision = selecting.result(timeout=2)

    assert decision.target == spec.target


def test_atomic_legacy_select_activate_conflict_leaves_no_selection(
    legacy_runtime: LegacyHarness,
) -> None:
    manifest_target = "atomic_active_manifest_target"
    legacy_target = "atomic_conflicting_legacy_target"
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "atomic_active_manifest",
            manifest_target,
            manifest=_manifest("atomic_active_manifest", manifest_target),
        )
    )
    legacy = legacy_runtime.fixture(
        LegacySpec("atomic_conflicting_legacy", legacy_target)
    )
    legacy_runtime.install(manifest, legacy)
    manifest_decision = legacy_runtime.registry.select(
        manifest_target, environment={}
    )
    active = legacy_runtime.registry.activate(manifest_decision.record_id)
    active_decision = legacy_runtime.registry.get_selection(manifest_target)
    lease = _materialize(
        legacy_runtime,
        _legacy_record(legacy_runtime, legacy).record_id,
    )
    method = getattr(
        legacy_runtime.registry,
        "select_and_activate_materialized_legacy",
        None,
    )
    assert callable(method), (
        "Registry.select_and_activate_materialized_legacy() is missing"
    )

    with pytest.raises(BackendPluginConflictError):
        method(legacy_runtime.target(legacy_target), lease=lease)

    assert legacy_runtime.registry.get_selection(legacy_target) is None
    assert legacy_runtime.registry.get_selection(manifest_target) is (
        active_decision
    )
    assert legacy_runtime.registry.inspect(active.record_id).state is (
        PluginLifecycleState.ACTIVE
    )


def test_exact_registry_legacy_selection_remains_supported(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("explicit_legacy", "explicit_target")
    )
    legacy_runtime.install(spec)
    record = _legacy_record(legacy_runtime, spec)

    decision = legacy_runtime.registry.select(
        legacy_runtime.target(spec.target),
        explicit_selector=record.registry_key,
    )
    compiler = legacy_runtime.backends_module.make_backend(
        legacy_runtime.target(spec.target)
    )

    assert decision.record_id == record.record_id
    assert decision.is_legacy is True
    assert compiler.__class__ is spec.compiler_cls
    assert legacy_runtime.registry.get_selection(spec.target).record_id == (
        record.record_id
    )


@pytest.mark.parametrize("consumer", ("compiler", "runtime"))
def test_python_explicit_legacy_is_not_mapped_before_probe_and_failure(
    legacy_runtime: LegacyHarness,
    consumer: str,
) -> None:
    target_name = f"python_explicit_{consumer}_probe_target"
    spec_name = f"python_explicit_{consumer}_probe"
    mapping_observations: list[bool] = []
    spec = legacy_runtime.fixture(
        LegacySpec(
            spec_name,
            target_name,
            supports=(
                (lambda _target: mapping_observations.append(
                    spec_name in legacy_runtime.backends_module.backends
                ) or False)
                if consumer == "compiler"
                else True
            ),
            active=(
                (lambda: mapping_observations.append(
                    spec_name in legacy_runtime.backends_module.backends
                ) or False)
                if consumer == "runtime"
                else True
            ),
        )
    )
    legacy_runtime.install(spec)
    record = _legacy_record(legacy_runtime, spec)
    decision = legacy_runtime.registry.select(
        legacy_runtime.target(target_name),
        explicit_selector=record.registry_key,
    )
    assert spec.name not in legacy_runtime.backends_module.backends

    with pytest.raises(BackendPluginSelectionError):
        if consumer == "compiler":
            legacy_runtime.backends_module.make_backend(
                legacy_runtime.target(target_name)
            )
        else:
            legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    assert mapping_observations == [False]
    current = legacy_runtime.registry.get_selection(target_name)
    assert current is not None
    assert current.record_id == decision.record_id
    assert spec.name not in legacy_runtime.backends_module.backends
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


# ---------------------------------------------------------------------------
# Compiler-first public path


def test_compiler_first_unique_legacy_match_publishes_one_record_pair(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("compiler_unique", "compiler_target")
    )
    legacy_runtime.install(spec)
    target = legacy_runtime.target(spec.target)

    compiler = legacy_runtime.backends_module.make_backend(target)
    decision = legacy_runtime.registry.get_selection(spec.target)
    mapping = legacy_runtime.backends_module.backends[spec.name]
    record = legacy_runtime.registry.inspect(decision.record_id)

    assert compiler.__class__ is spec.compiler_cls
    assert compiler.target == target
    assert mapping.compiler is spec.compiler_cls
    assert mapping.driver is spec.driver_cls
    assert mapping.record_id == decision.record_id == record.record_id
    assert record.source is PluginSource.LEGACY
    assert (
        record.compatibility_status
        is PluginCompatibilityStatus.LEGACY_UNVERIFIED
    )
    assert spec.entry_point.load_calls == 1
    assert spec.calls["compiler_constructor"] == 1


def test_compiler_first_zero_legacy_matches_is_deterministic_and_atomic(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("compiler_zero", "compiler_zero_target", supports=False)
    )
    legacy_runtime.install(spec)

    with pytest.raises(BackendPluginSelectionError) as caught:
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(spec.target)
        )
    assert caught.value.field == "compiler_cls.supports_target"
    assert spec.calls["compiler_constructor"] == 0
    legacy_runtime.assert_unpublished(spec.target)


def test_compiler_first_multiple_legacy_matches_raise_structured_conflict(
    legacy_runtime: LegacyHarness,
) -> None:
    first = legacy_runtime.fixture(
        LegacySpec("compiler_multi_a", "compiler_multi_target")
    )
    second = legacy_runtime.fixture(
        LegacySpec("compiler_multi_b", "compiler_multi_target")
    )
    legacy_runtime.install(first, second)

    with pytest.raises(BackendPluginConflictError) as caught:
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(first.target)
        )
    assert caught.value.field == "compiler_cls.supports_target"
    assert first.calls["compiler_constructor"] == 0
    assert second.calls["compiler_constructor"] == 0
    legacy_runtime.assert_unpublished(first.target)


def test_compiler_probe_exception_never_falls_through_to_another_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    failing = legacy_runtime.fixture(
        LegacySpec(
            "a_compiler_probe_error",
            "compiler_probe_target",
            supports=RuntimeError("supports sentinel"),
        )
    )
    fallback = legacy_runtime.fixture(
        LegacySpec("b_compiler_probe_true", "compiler_probe_target")
    )
    legacy_runtime.install(failing, fallback)

    with pytest.raises(BackendPluginLifecycleError) as caught:
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(failing.target)
        )
    assert caught.value.field == "compiler_cls.supports_target"
    assert "supports sentinel" in str(caught.value)
    assert fallback.calls["compiler_constructor"] == 0
    legacy_runtime.assert_unpublished(failing.target)


def test_compiler_constructor_exception_rolls_back_selection_and_mapping(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec(
            "compiler_constructor_error",
            "compiler_constructor_target",
            compiler_error=RuntimeError("compiler constructor sentinel"),
        )
    )
    legacy_runtime.install(spec)

    with pytest.raises(BackendPluginLifecycleError) as caught:
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(spec.target)
        )
    assert caught.value.field == "compiler_cls.__init__"
    assert "compiler constructor sentinel" in str(caught.value)
    assert spec.calls["compiler_constructor"] == 1
    legacy_runtime.assert_unpublished(spec.target)


def test_f6_rejects_incomplete_legacy_pair_before_target_probe(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec(
            "compiler_f6_invalid",
            "compiler_f6_target",
            invalid_compiler=True,
        )
    )
    legacy_runtime.install(spec)

    with pytest.raises(BackendPluginInterfaceError):
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(spec.target)
        )
    record = _legacy_record(legacy_runtime, spec)
    assert record.state is PluginLifecycleState.REJECTED
    assert spec.calls["supports_target"] == 0
    assert spec.calls["initialize"] == 0
    legacy_runtime.assert_unpublished(spec.target)


def test_legacy_public_key_collision_does_not_overwrite_manual_mapping(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("mapping_collision", "mapping_collision_target")
    )
    legacy_runtime.install(spec)
    original = legacy_runtime.backends_module.Backend(
        compiler=spec.compiler_cls,
        driver=spec.driver_cls,
    )
    legacy_runtime.backends_module.backends[spec.name] = original

    with pytest.raises(BackendPluginConflictError) as caught:
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(spec.target)
        )
    assert caught.value.field in {"backends", "entry_point"}
    assert legacy_runtime.backends_module.backends[spec.name] is original
    assert legacy_runtime.registry.get_selection(spec.target) is None


@pytest.mark.parametrize(
    ("failure", "supports", "compiler_error", "error_type"),
    (
        ("supports_false", False, None, BackendPluginSelectionError),
        (
            "supports_error",
            RuntimeError("manifest supports sentinel"),
            None,
            BackendPluginLifecycleError,
        ),
        (
            "constructor_error",
            True,
            RuntimeError("manifest compiler constructor sentinel"),
            BackendPluginLifecycleError,
        ),
    ),
)
def test_manifest_compiler_failure_releases_only_fresh_selection_and_mapping(
    legacy_runtime: LegacyHarness,
    failure: str,
    supports: Any,
    compiler_error: Any,
    error_type: type[BackendPluginError],
) -> None:
    preserved_target = f"manifest_compiler_preserved_{failure}"
    failing_target = f"manifest_compiler_failing_{failure}"
    preserved = legacy_runtime.fixture(
        LegacySpec(
            f"manifest_compiler_preserved_{failure}",
            preserved_target,
            manifest=_manifest(
                f"manifest_compiler_preserved_{failure}",
                preserved_target,
            ),
        )
    )
    failing = legacy_runtime.fixture(
        LegacySpec(
            f"manifest_compiler_failing_{failure}",
            failing_target,
            supports=supports,
            compiler_error=compiler_error,
            manifest=_manifest(
                f"manifest_compiler_failing_{failure}",
                failing_target,
            ),
        )
    )
    legacy_runtime.install(preserved, failing)

    preserved_backend = legacy_runtime.backends_module.get_backend(
        legacy_runtime.target(preserved_target)
    )
    preserved_decision = legacy_runtime.registry.get_selection(
        preserved_target
    )

    with pytest.raises(error_type):
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(failing_target)
        )

    assert legacy_runtime.registry.get_selection(preserved_target) is (
        preserved_decision
    )
    preserved_mapping = legacy_runtime.backends_module.backends[
        preserved.name
    ]
    assert preserved_mapping.record_id == preserved_decision.record_id
    assert preserved_mapping.compiler is preserved_backend.compiler
    assert preserved_mapping.driver is preserved_backend.driver
    assert legacy_runtime.registry.get_selection(failing_target) is None
    assert failing.name not in legacy_runtime.backends_module.backends


@pytest.mark.parametrize(
    "malformed",
    (
        pytest.param("kernel.tensor_core", id="string"),
        pytest.param(b"kernel.tensor_core", id="bytes"),
        pytest.param(None, id="none"),
        pytest.param(("duplicate", "duplicate"), id="duplicate"),
        pytest.param((" surrounding-space ",), id="whitespace"),
        pytest.param((object(),), id="non-string"),
    ),
)
def test_legacy_capability_input_is_validated_before_import(
    legacy_runtime: LegacyHarness,
    malformed: Any,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("malformed_capability_legacy", "malformed_capability_target")
    )
    legacy_runtime.install(spec)

    with pytest.raises(BackendPluginSelectionError) as caught:
        _resolve_compiler(
            legacy_runtime,
            legacy_runtime.target(spec.target),
            kernel_required_capabilities=malformed,
        )

    assert caught.value.field == "kernel_required_capabilities"
    assert spec.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(spec.target)


def test_manual_compiler_and_registry_legacy_share_zero_one_many_policy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manual_registry_compiler_unique_target"
    manual = legacy_runtime.fixture(
        LegacySpec("manual_registry_compiler_unique_manual", target_name)
    )
    registry_legacy = legacy_runtime.fixture(
        LegacySpec(
            "manual_registry_compiler_unique_registry",
            target_name,
            supports=False,
        )
    )
    legacy_runtime.install(registry_legacy)
    original = _install_manual_backend(legacy_runtime, manual)

    compiler = legacy_runtime.backends_module.make_backend(
        legacy_runtime.target(target_name)
    )

    assert compiler.__class__ is manual.compiler_cls
    assert manual.calls["compiler_constructor"] == 1
    assert registry_legacy.calls["supports_target"] == 1
    assert registry_legacy.calls["compiler_constructor"] == 0
    assert legacy_runtime.backends_module.backends[manual.name] is original
    assert registry_legacy.name not in legacy_runtime.backends_module.backends
    assert legacy_runtime.registry.get_selection(target_name) is None


def test_manual_compiler_and_registry_legacy_multiple_matches_conflict(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manual_registry_compiler_conflict_target"
    manual = legacy_runtime.fixture(
        LegacySpec("manual_registry_compiler_conflict_manual", target_name)
    )
    registry_legacy = legacy_runtime.fixture(
        LegacySpec("manual_registry_compiler_conflict_registry", target_name)
    )
    legacy_runtime.install(registry_legacy)
    original = _install_manual_backend(legacy_runtime, manual)

    with pytest.raises(BackendPluginConflictError) as caught:
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(target_name)
        )

    assert caught.value.field == "compiler_cls.supports_target"
    assert manual.calls["compiler_constructor"] == 0
    assert registry_legacy.calls["compiler_constructor"] == 0
    assert legacy_runtime.backends_module.backends == {manual.name: original}
    assert legacy_runtime.registry.get_selection(target_name) is None


@pytest.mark.parametrize("consumer", ("compiler", "runtime"))
def test_target_str_subclass_hash_and_equality_run_outside_adapter_lock(
    legacy_runtime: LegacyHarness,
    consumer: str,
) -> None:
    target_name = f"strict_target_normalization_{consumer}"
    observations: list[tuple[str, bool]] = []
    cache_lock = legacy_runtime.backends_module._backend_cache_lock

    class AdversarialTargetName(str):
        def __hash__(self) -> int:
            observations.append(("hash", cache_lock._is_owned()))
            return str.__hash__(self)

        def __eq__(self, other: Any) -> bool:
            observations.append(("eq", cache_lock._is_owned()))
            return str.__eq__(self, other)

    target = types.SimpleNamespace(
        backend=AdversarialTargetName(target_name),
        arch="fixture-arch",
        warp_size=32,
    )
    spec = legacy_runtime.fixture(
        LegacySpec(
            f"strict_target_normalization_{consumer}_manual",
            target_name,
            current_target=target,
        )
    )
    legacy_runtime.install()
    owner = legacy_runtime.backends_module.Backend(
        compiler=spec.compiler_cls,
        driver=spec.driver_cls,
        entry_point_name=spec.name,
    )
    legacy_runtime.backends_module.backends[spec.name] = owner
    with cache_lock:
        legacy_runtime.backends_module._legacy_target_owners[target_name] = owner

    with pytest.raises(BackendPluginSelectionError) as caught:
        if consumer == "compiler":
            legacy_runtime.backends_module.make_backend(target)
        else:
            legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    assert caught.value.field == "target.backend"
    assert not [method for method, lock_owned in observations if lock_owned]
    assert legacy_runtime.backends_module.backends == {spec.name: owner}
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


# ---------------------------------------------------------------------------
# Runtime-first public path


def test_runtime_first_unique_active_legacy_publishes_same_record_pair(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("runtime_unique", "runtime_target")
    )
    legacy_runtime.install(spec)
    resolver = getattr(
        legacy_runtime.backends_module, "resolve_active_legacy_driver", None
    )
    assert callable(resolver), (
        "triton.backends.resolve_active_legacy_driver() is missing"
    )

    target = legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    decision = legacy_runtime.registry.get_selection(target.backend)
    mapping = legacy_runtime.backends_module.backends[spec.name]
    concrete = legacy_runtime.runtime_driver_module.driver.default._obj

    assert target.backend == spec.target
    assert concrete.__class__ is spec.driver_cls
    assert mapping.compiler is spec.compiler_cls
    assert mapping.driver is spec.driver_cls
    assert mapping.record_id == decision.record_id
    assert decision.record_id == _legacy_record(
        legacy_runtime, spec
    ).record_id
    assert spec.entry_point.load_calls == 1
    assert spec.calls["driver_constructor"] == 1


def test_runtime_first_zero_active_legacy_is_deterministic_and_atomic(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("runtime_zero", "runtime_zero_target", active=False)
    )
    legacy_runtime.install(spec)

    with pytest.raises(BackendPluginSelectionError) as caught:
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    assert caught.value.field == "driver_cls.is_active"
    assert spec.entry_point.load_calls == 1
    assert spec.calls["is_active"] == 1
    assert spec.calls["driver_constructor"] == 0
    legacy_runtime.assert_unpublished(spec.target)


def test_runtime_first_multiple_active_legacy_raise_structured_conflict(
    legacy_runtime: LegacyHarness,
) -> None:
    first = legacy_runtime.fixture(
        LegacySpec("runtime_multi_a", "runtime_multi_target")
    )
    second = legacy_runtime.fixture(
        LegacySpec("runtime_multi_b", "runtime_multi_target")
    )
    legacy_runtime.install(first, second)

    with pytest.raises(BackendPluginConflictError) as caught:
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    assert caught.value.field == "driver_cls.is_active"
    assert first.calls["driver_constructor"] == 0
    assert second.calls["driver_constructor"] == 0
    legacy_runtime.assert_unpublished(first.target)


def test_runtime_active_probe_exception_never_falls_through_to_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    failing = legacy_runtime.fixture(
        LegacySpec(
            "a_runtime_probe_error",
            "runtime_probe_target",
            active=RuntimeError("active probe sentinel"),
        )
    )
    fallback = legacy_runtime.fixture(
        LegacySpec("b_runtime_probe_true", "runtime_probe_target")
    )
    legacy_runtime.install(failing, fallback)

    with pytest.raises(BackendPluginLifecycleError) as caught:
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    assert caught.value.field == "driver_cls.is_active"
    assert "active probe sentinel" in str(caught.value)
    assert fallback.calls["driver_constructor"] == 0
    legacy_runtime.assert_unpublished(failing.target)


def test_runtime_driver_constructor_exception_is_not_published(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec(
            "runtime_constructor_error",
            "runtime_constructor_target",
            driver_error=RuntimeError("driver constructor sentinel"),
        )
    )
    legacy_runtime.install(spec)

    with pytest.raises(BackendPluginLifecycleError) as caught:
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    assert caught.value.field == "driver_cls.__init__"
    assert "driver constructor sentinel" in str(caught.value)
    assert spec.calls["driver_constructor"] == 1
    legacy_runtime.assert_unpublished(spec.target)


def test_runtime_get_current_target_exception_is_not_published(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec(
            "runtime_target_error",
            "runtime_target_error_target",
            current_target=RuntimeError("current target sentinel"),
        )
    )
    legacy_runtime.install(spec)

    with pytest.raises(BackendPluginLifecycleError) as caught:
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    assert caught.value.field == "driver_cls.get_current_target"
    assert "current target sentinel" in str(caught.value)
    assert spec.calls["driver_constructor"] == 1
    legacy_runtime.assert_unpublished(spec.target)


@pytest.mark.parametrize(
    ("failure", "active", "driver_error", "current_target"),
    (
        (
            "active_probe",
            RuntimeError("Manifest active probe sentinel"),
            None,
            None,
        ),
        (
            "constructor",
            True,
            RuntimeError("Manifest driver constructor sentinel"),
            None,
        ),
        (
            "current_target",
            True,
            None,
            RuntimeError("Manifest current target sentinel"),
        ),
    ),
)
def test_manifest_runtime_callback_failure_releases_fresh_state_without_fallback(
    legacy_runtime: LegacyHarness,
    failure: str,
    active: Any,
    driver_error: Any,
    current_target: Any,
) -> None:
    target_name = f"manifest_runtime_failure_{failure}_target"
    manifest = legacy_runtime.fixture(
        LegacySpec(
            f"manifest_runtime_failure_{failure}",
            target_name,
            active=active,
            driver_error=driver_error,
            current_target=current_target,
            manifest=_manifest(
                f"manifest_runtime_failure_{failure}", target_name
            ),
        )
    )
    legacy = legacy_runtime.fixture(
        LegacySpec(f"manifest_runtime_failure_{failure}_legacy", target_name)
    )
    legacy_runtime.install(manifest, legacy)

    with pytest.raises(BackendPluginLifecycleError):
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    assert legacy.entry_point.load_calls == 0
    assert legacy_runtime.registry.get_selection(target_name) is None
    assert manifest.name not in legacy_runtime.backends_module.backends
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


def test_manifest_runtime_resolution_second_load_failure_rolls_back_first(
    legacy_runtime: LegacyHarness,
) -> None:
    first_target = "runtime_build_journal_a_target"
    second_target = "runtime_build_journal_b_target"
    first = legacy_runtime.fixture(
        LegacySpec(
            "runtime_build_journal_a",
            first_target,
            manifest=_manifest(
                "runtime_build_journal_a",
                first_target,
            ),
        )
    )
    second = legacy_runtime.fixture(
        LegacySpec(
            "runtime_build_journal_b",
            second_target,
            load_error=RuntimeError("second Manifest load sentinel"),
            manifest=_manifest(
                "runtime_build_journal_b",
                second_target,
            ),
        )
    )
    legacy_runtime.install(first, second)

    with pytest.raises(BackendPluginError) as caught:
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    assert "second Manifest load sentinel" in str(caught.value)
    assert first.entry_point.load_calls == 1
    assert second.entry_point.load_calls == 1
    assert legacy_runtime.registry.get_selection(first_target) is None
    assert legacy_runtime.registry.get_selection(second_target) is None
    assert first.name not in legacy_runtime.backends_module.backends
    assert second.name not in legacy_runtime.backends_module.backends
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


def test_runtime_releases_only_fresh_inactive_manifest_beside_preexisting_one(
    legacy_runtime: LegacyHarness,
) -> None:
    preserved_target = "runtime_inactive_preexisting_target"
    speculative_target = "runtime_inactive_speculative_target"
    preserved = legacy_runtime.fixture(
        LegacySpec(
            "runtime_inactive_preexisting",
            preserved_target,
            active=False,
            manifest=_manifest(
                "runtime_inactive_preexisting", preserved_target
            ),
        )
    )
    speculative = legacy_runtime.fixture(
        LegacySpec(
            "runtime_inactive_speculative",
            speculative_target,
            active=False,
            manifest=_manifest(
                "runtime_inactive_speculative", speculative_target
            ),
        )
    )
    legacy = legacy_runtime.fixture(
        LegacySpec("runtime_inactive_forbidden_legacy", speculative_target)
    )
    legacy_runtime.install(preserved, speculative, legacy)
    preserved_backend = legacy_runtime.backends_module.get_backend(
        legacy_runtime.target(preserved_target)
    )
    preserved_decision = legacy_runtime.registry.get_selection(
        preserved_target
    )

    with pytest.raises(BackendPluginSelectionError) as caught:
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    assert caught.value.field == "driver_cls.is_active"
    assert legacy.entry_point.load_calls == 0
    assert legacy_runtime.registry.get_selection(preserved_target) is (
        preserved_decision
    )
    preserved_mapping = legacy_runtime.backends_module.backends[
        preserved.name
    ]
    assert preserved_mapping.record_id == preserved_decision.record_id
    assert preserved_mapping.compiler is preserved_backend.compiler
    assert preserved_mapping.driver is preserved_backend.driver
    assert legacy_runtime.registry.get_selection(speculative_target) is None
    assert speculative.name not in legacy_runtime.backends_module.backends
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


def test_manual_and_registry_legacy_active_candidates_conflict_together(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manual_registry_runtime_conflict_target"
    manual = legacy_runtime.fixture(
        LegacySpec("manual_registry_runtime_conflict_manual", target_name)
    )
    registry_legacy = legacy_runtime.fixture(
        LegacySpec("manual_registry_runtime_conflict_registry", target_name)
    )
    legacy_runtime.install(registry_legacy)
    original = _install_manual_backend(legacy_runtime, manual)

    with pytest.raises(BackendPluginConflictError) as caught:
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    assert caught.value.field == "driver_cls.is_active"
    assert manual.calls["driver_constructor"] == 0
    assert registry_legacy.calls["driver_constructor"] == 0
    assert legacy_runtime.backends_module.backends == {manual.name: original}
    assert legacy_runtime.registry.get_selection(target_name) is None
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


def test_runtime_driver_target_must_be_supported_by_its_own_compiler(
    legacy_runtime: LegacyHarness,
) -> None:
    declared = "runtime_declared_target"
    actual = "runtime_actual_target"
    spec = legacy_runtime.fixture(
        LegacySpec(
            "runtime_pair_mismatch",
            declared,
            supports=lambda target: target.backend == declared,
            current_target=actual,
        )
    )
    legacy_runtime.install(spec)

    with pytest.raises(BackendPluginSelectionError) as caught:
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    assert caught.value.field == "compiler_cls.supports_target"
    assert actual in (caught.value.actual or str(caught.value))
    legacy_runtime.assert_unpublished(actual)
    assert declared not in legacy_runtime.backends_module.backends


def test_f6_rejects_incomplete_legacy_driver_before_active_probe(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec(
            "runtime_f6_invalid",
            "runtime_f6_target",
            invalid_driver=True,
        )
    )
    legacy_runtime.install(spec)

    with pytest.raises(BackendPluginInterfaceError):
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    record = _legacy_record(legacy_runtime, spec)
    assert record.state is PluginLifecycleState.REJECTED
    assert spec.calls["is_active"] == 0
    assert spec.calls["initialize"] == 0
    legacy_runtime.assert_unpublished(spec.target)


# ---------------------------------------------------------------------------
# Manifest-first fail-closed policy


def test_manifest_compiler_winner_never_imports_overlapping_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_first_target"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_first_legacy", target_name)
    )
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "manifest_first_winner",
            target_name,
            manifest=_manifest("manifest_first_winner", target_name),
        )
    )
    legacy_runtime.install(legacy, manifest)

    compiler = legacy_runtime.backends_module.make_backend(
        legacy_runtime.target(target_name)
    )

    assert compiler.__class__ is manifest.compiler_cls
    assert manifest.entry_point.load_calls == 1
    assert legacy.entry_point.load_calls == 0
    mapping = legacy_runtime.backends_module.backends[manifest.name]
    assert mapping.compiler is manifest.compiler_cls
    assert mapping.driver is manifest.driver_cls


def test_manifest_runtime_winner_never_imports_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_runtime_target"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_runtime_legacy", target_name)
    )
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "manifest_runtime_winner",
            target_name,
            manifest=_manifest("manifest_runtime_winner", target_name),
        )
    )
    legacy_runtime.install(legacy, manifest)

    target = legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    assert target.backend == target_name
    assert (
        legacy_runtime.runtime_driver_module.driver.default._obj.__class__
        is manifest.driver_cls
    )
    assert manifest.entry_point.load_calls == 1
    assert legacy.entry_point.load_calls == 0


@pytest.mark.parametrize("consumer", ("compiler", "runtime"))
def test_direct_legacy_resolver_cannot_bypass_manifest_first(
    legacy_runtime: LegacyHarness,
    consumer: str,
) -> None:
    target_name = f"direct_manifest_gate_{consumer}_target"
    manifest = legacy_runtime.fixture(
        LegacySpec(
            f"direct_manifest_gate_{consumer}_manifest",
            target_name,
            manifest=_manifest(
                f"direct_manifest_gate_{consumer}_manifest", target_name
            ),
        )
    )
    legacy = legacy_runtime.fixture(
        LegacySpec(f"direct_manifest_gate_{consumer}_legacy", target_name)
    )
    legacy_runtime.install(manifest, legacy)

    with pytest.raises(BackendPluginSelectionError) as caught:
        if consumer == "compiler":
            _resolve_compiler(
                legacy_runtime,
                legacy_runtime.target(target_name),
            )
        else:
            resolver = getattr(
                legacy_runtime.backends_module,
                "resolve_active_legacy_driver",
                None,
            )
            assert callable(resolver)
            resolver()

    assert caught.value.field == "legacy_fallback"
    assert legacy.entry_point.load_calls == 0
    assert legacy_runtime.registry.get_selection(target_name) is None
    assert legacy.name not in legacy_runtime.backends_module.backends


def test_active_manifest_wins_before_manual_runtime_probe(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_before_manual_active_target"
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "manifest_before_manual_active_manifest",
            target_name,
            manifest=_manifest(
                "manifest_before_manual_active_manifest", target_name
            ),
        )
    )
    manual = legacy_runtime.fixture(
        LegacySpec("manifest_before_manual_active_manual", target_name)
    )
    legacy_runtime.install(manifest)
    original = _install_manual_backend(legacy_runtime, manual)

    target = legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    assert target.backend == target_name
    assert (
        legacy_runtime.runtime_driver_module.driver.default._obj.__class__
        is manifest.driver_cls
    )
    assert manual.calls["is_active"] == 0
    assert manual.calls["driver_constructor"] == 0
    assert legacy_runtime.backends_module.backends[manual.name] is original
    decision = legacy_runtime.registry.get_selection(target_name)
    assert decision.record_id == _legacy_record(
        legacy_runtime, manifest
    ).record_id


def test_inactive_manifest_is_released_before_manual_runtime_fallback(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "inactive_manifest_manual_fallback_target"
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "inactive_manifest_manual_fallback_manifest",
            target_name,
            active=False,
            manifest=_manifest(
                "inactive_manifest_manual_fallback_manifest", target_name
            ),
        )
    )
    manual = legacy_runtime.fixture(
        LegacySpec("inactive_manifest_manual_fallback_manual", target_name)
    )
    legacy_runtime.install(manifest)
    original = _install_manual_backend(legacy_runtime, manual)

    target = legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    assert target.backend == target_name
    assert (
        legacy_runtime.runtime_driver_module.driver.default._obj.__class__
        is manual.driver_cls
    )
    assert manual.calls["is_active"] == 1
    assert manifest.calls["is_active"] == 1
    assert legacy_runtime.registry.get_selection(target_name) is None
    assert legacy_runtime.backends_module.backends == {manual.name: original}


def test_rejected_manifest_error_precedes_unrelated_manual_runtime(
    legacy_runtime: LegacyHarness,
) -> None:
    manual_target = "rejected_manifest_manual_runtime_target"
    rejected = legacy_runtime.fixture(
        LegacySpec(
            "rejected_manifest_before_manual_runtime",
            "rejected_manifest_unrelated_target",
            manifest=_manifest(
                "rejected_manifest_before_manual_runtime",
                "rejected_manifest_unrelated_target",
                triton_version=">=9,<10",
            ),
        )
    )
    manual = legacy_runtime.fixture(
        LegacySpec("rejected_manifest_manual_runtime", manual_target)
    )
    legacy_runtime.install(rejected)
    original = _install_manual_backend(legacy_runtime, manual)

    with pytest.raises(BackendPluginCompatibilityError):
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    assert manual.calls["is_active"] == 0
    assert manual.calls["driver_constructor"] == 0
    assert legacy_runtime.backends_module.backends == {manual.name: original}
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


def test_get_driver_backends_cannot_bypass_rejected_manifest_with_manual(
    legacy_runtime: LegacyHarness,
) -> None:
    rejected_target = "get_driver_backends_rejected_manifest_target"
    manual_target = "get_driver_backends_manual_target"
    rejected = legacy_runtime.fixture(
        LegacySpec(
            "get_driver_backends_rejected_manifest",
            rejected_target,
            manifest=_manifest(
                "get_driver_backends_rejected_manifest",
                rejected_target,
                triton_version=">=9,<10",
            ),
        )
    )
    manual = legacy_runtime.fixture(
        LegacySpec("get_driver_backends_manual", manual_target)
    )
    legacy_runtime.install(rejected)
    original = _install_manual_backend(legacy_runtime, manual)

    with pytest.raises(BackendPluginCompatibilityError):
        legacy_runtime.backends_module.get_driver_backends()

    assert rejected.entry_point.load_calls == 0
    assert manual.calls["is_active"] == 0
    assert manual.calls["driver_constructor"] == 0
    assert legacy_runtime.backends_module.backends == {manual.name: original}
    assert legacy_runtime.registry.get_selection(rejected_target) is None
    assert legacy_runtime.registry.get_selection(manual_target) is None


def test_manifest_priority_is_resolved_before_any_legacy_fallback(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_priority_target"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_priority_legacy", target_name)
    )
    low = legacy_runtime.fixture(
        LegacySpec(
            "manifest_priority_low",
            target_name,
            manifest=_manifest(
                "manifest_priority_low", target_name, priority=1
            ),
        )
    )
    high = legacy_runtime.fixture(
        LegacySpec(
            "manifest_priority_high",
            target_name,
            manifest=_manifest(
                "manifest_priority_high", target_name, priority=20
            ),
        )
    )
    legacy_runtime.install(legacy, low, high)

    compiler = legacy_runtime.backends_module.make_backend(
        legacy_runtime.target(target_name)
    )

    assert compiler.__class__ is high.compiler_cls
    assert high.entry_point.load_calls == 1
    assert low.entry_point.load_calls == 0
    assert legacy.entry_point.load_calls == 0


def test_explicit_manifest_selection_precedes_legacy_and_priority(
    legacy_runtime: LegacyHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_name = "manifest_explicit_target"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_explicit_legacy", target_name)
    )
    selected = legacy_runtime.fixture(
        LegacySpec(
            "manifest_explicit_selected",
            target_name,
            manifest=_manifest(
                "manifest_explicit_selected", target_name, priority=0
            ),
        )
    )
    higher = legacy_runtime.fixture(
        LegacySpec(
            "manifest_explicit_higher",
            target_name,
            manifest=_manifest(
                "manifest_explicit_higher", target_name, priority=50
            ),
        )
    )
    legacy_runtime.install(legacy, selected, higher)
    monkeypatch.setenv(
        "TRITON_ANCHOR_BACKEND",
        f"acceptance.f5.{selected.name}",
    )

    compiler = legacy_runtime.backends_module.make_backend(
        legacy_runtime.target(target_name)
    )

    assert compiler.__class__ is selected.compiler_cls
    assert selected.entry_point.load_calls == 1
    assert higher.entry_point.load_calls == 0
    assert legacy.entry_point.load_calls == 0


def test_incompatible_manifest_never_falls_back_to_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_version_target"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_version_legacy", target_name)
    )
    incompatible = legacy_runtime.fixture(
        LegacySpec(
            "manifest_version_incompatible",
            target_name,
            manifest=_manifest(
                "manifest_version_incompatible",
                target_name,
                triton_version=">=9,<10",
            ),
        )
    )
    legacy_runtime.install(legacy, incompatible)

    with pytest.raises(BackendPluginCompatibilityError):
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(target_name)
        )
    assert incompatible.entry_point.load_calls == 0
    assert legacy.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(target_name)


def test_manifest_abi_metadata_error_never_falls_back_to_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_abi_target"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_abi_legacy", target_name)
    )
    malformed = legacy_runtime.fixture(
        LegacySpec(
            "manifest_abi_malformed",
            target_name,
            manifest=_manifest(
                "manifest_abi_malformed",
                target_name,
                extra={"abi_fingerprint": "sha256:" + "a" * 64},
            ),
        )
    )
    legacy_runtime.install(legacy, malformed)

    with pytest.raises(BackendPluginManifestError) as caught:
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(target_name)
        )
    assert caught.value.field == "abi_fingerprint"
    assert malformed.entry_point.load_calls == 0
    assert legacy.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(target_name)


def test_manifest_identity_conflict_never_falls_back_to_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_conflict_target"
    plugin_id = "acceptance.f5.duplicate"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_conflict_legacy", target_name)
    )
    first = legacy_runtime.fixture(
        LegacySpec(
            "manifest_conflict_a",
            target_name,
            manifest=_manifest(
                "manifest_conflict_a", target_name, plugin_id=plugin_id
            ),
        )
    )
    second = legacy_runtime.fixture(
        LegacySpec(
            "manifest_conflict_b",
            target_name,
            manifest=_manifest(
                "manifest_conflict_b", target_name, plugin_id=plugin_id
            ),
        )
    )
    legacy_runtime.install(legacy, first, second)

    with pytest.raises(BackendPluginConflictError):
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(target_name)
        )
    assert first.entry_point.load_calls == 0
    assert second.entry_point.load_calls == 0
    assert legacy.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(target_name)


def test_manifest_capability_rejection_never_imports_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_capability_target"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_capability_legacy", target_name)
    )
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "manifest_capability_candidate",
            target_name,
            manifest=_manifest(
                "manifest_capability_candidate",
                target_name,
                capabilities=("acceptance.f5",),
            ),
        )
    )
    legacy_runtime.install(legacy, manifest)

    with pytest.raises(BackendPluginCapabilityError):
        legacy_runtime.registry.select(
            legacy_runtime.target(target_name),
            kernel_required_capabilities=("kernel.tensor_core",),
        )
    assert manifest.entry_point.load_calls == 0
    assert legacy.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(target_name)


def test_manifest_f6_interface_rejection_never_falls_back_to_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_interface_target"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_interface_legacy", target_name)
    )
    invalid = legacy_runtime.fixture(
        LegacySpec(
            "manifest_interface_invalid",
            target_name,
            invalid_compiler=True,
            manifest=_manifest("manifest_interface_invalid", target_name),
        )
    )
    legacy_runtime.install(legacy, invalid)

    with pytest.raises(BackendPluginInterfaceError):
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(target_name)
        )
    assert invalid.entry_point.load_calls == 1
    assert legacy.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(target_name)


def test_manifest_load_failure_never_falls_back_to_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_load_target"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_load_legacy", target_name)
    )
    broken = legacy_runtime.fixture(
        LegacySpec(
            "manifest_load_broken",
            target_name,
            load_error=RuntimeError("manifest load sentinel"),
            manifest=_manifest("manifest_load_broken", target_name),
        )
    )
    legacy_runtime.install(legacy, broken)

    with pytest.raises(BackendPluginError) as caught:
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(target_name)
        )
    assert caught.value.field == "entry_point.load"
    assert "manifest load sentinel" in str(caught.value)
    assert broken.entry_point.load_calls == 1
    assert legacy.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(target_name)


def test_manifest_equal_priority_conflict_never_falls_back_to_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "manifest_priority_conflict_target"
    legacy = legacy_runtime.fixture(
        LegacySpec("manifest_priority_conflict_legacy", target_name)
    )
    first = legacy_runtime.fixture(
        LegacySpec(
            "manifest_priority_conflict_a",
            target_name,
            manifest=_manifest(
                "manifest_priority_conflict_a", target_name, priority=7
            ),
        )
    )
    second = legacy_runtime.fixture(
        LegacySpec(
            "manifest_priority_conflict_b",
            target_name,
            manifest=_manifest(
                "manifest_priority_conflict_b", target_name, priority=7
            ),
        )
    )
    legacy_runtime.install(legacy, first, second)

    with pytest.raises(BackendPluginSelectionError) as caught:
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(target_name)
        )
    assert caught.value.field == "priority"
    assert first.entry_point.load_calls == 0
    assert second.entry_point.load_calls == 0
    assert legacy.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(target_name)


def test_runtime_compiler_probe_exception_does_not_try_another_legacy(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "runtime_compiler_probe_target"
    failing = legacy_runtime.fixture(
        LegacySpec(
            "a_runtime_compiler_probe_error",
            target_name,
            supports=RuntimeError("runtime compiler probe sentinel"),
            current_target=target_name,
        )
    )
    fallback = legacy_runtime.fixture(
        LegacySpec(
            "b_runtime_compiler_probe_fallback",
            target_name,
            active=False,
        )
    )
    legacy_runtime.install(failing, fallback)

    with pytest.raises(BackendPluginLifecycleError) as caught:
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    assert caught.value.field == "compiler_cls.supports_target"
    assert "runtime compiler probe sentinel" in str(caught.value)
    assert fallback.calls["driver_constructor"] == 0
    legacy_runtime.assert_unpublished(target_name)


def test_cached_legacy_selection_cannot_bypass_later_capability_requirement(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("cached_capability_legacy", "cached_capability_target")
    )
    legacy_runtime.install(spec)
    target = legacy_runtime.target(spec.target)
    compiler = legacy_runtime.backends_module.make_backend(target)
    assert compiler.__class__ is spec.compiler_cls
    decision = legacy_runtime.registry.get_selection(spec.target)

    with pytest.raises(BackendPluginCapabilityError):
        _resolve_compiler(
            legacy_runtime,
            target,
            kernel_required_capabilities=("kernel.tensor_core",),
        )
    assert legacy_runtime.registry.get_selection(spec.target) is decision


def test_exact_legacy_selection_with_capabilities_fails_before_import(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("exact_capability_legacy", "exact_capability_target")
    )
    legacy_runtime.install(spec)
    record = _legacy_record(legacy_runtime, spec)

    with pytest.raises(BackendPluginCapabilityError):
        legacy_runtime.registry.select(
            legacy_runtime.target(spec.target),
            explicit_selector=record.registry_key,
            kernel_required_capabilities=("kernel.tensor_core",),
        )
    assert spec.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(spec.target)


def test_runtime_does_not_use_active_legacy_beside_rejected_manifest(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "runtime_manifest_rejection_target"
    inactive_manifest = legacy_runtime.fixture(
        LegacySpec(
            "runtime_manifest_inactive",
            target_name,
            active=False,
            manifest=_manifest("runtime_manifest_inactive", target_name),
        )
    )
    rejected_manifest = legacy_runtime.fixture(
        LegacySpec(
            "runtime_manifest_rejected",
            "runtime_rejected_other_target",
            manifest=_manifest(
                "runtime_manifest_rejected",
                "runtime_rejected_other_target",
                triton_version=">=9,<10",
            ),
        )
    )
    legacy = legacy_runtime.fixture(
        LegacySpec("runtime_rejected_legacy", target_name, active=True)
    )
    legacy_runtime.install(inactive_manifest, rejected_manifest, legacy)

    with pytest.raises(BackendPluginCompatibilityError):
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    assert inactive_manifest.entry_point.load_calls <= 1
    assert rejected_manifest.entry_point.load_calls == 0
    assert legacy.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(target_name)


def test_runtime_falls_back_when_valid_manifest_driver_is_inactive(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "runtime_inactive_manifest_target"
    inactive_manifest = legacy_runtime.fixture(
        LegacySpec(
            "runtime_only_inactive_manifest",
            target_name,
            active=False,
            manifest=_manifest(
                "runtime_only_inactive_manifest", target_name
            ),
        )
    )
    legacy = legacy_runtime.fixture(
        LegacySpec("runtime_active_legacy_fallback", target_name, active=True)
    )
    legacy_runtime.install(inactive_manifest, legacy)

    target = legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    decision = legacy_runtime.registry.get_selection(target_name)
    mapping = legacy_runtime.backends_module.backends[legacy.name]
    driver = legacy_runtime.runtime_driver_module.driver.default._obj

    assert target.backend == target_name
    assert driver.__class__ is legacy.driver_cls
    assert mapping.compiler is legacy.compiler_cls
    assert mapping.driver is legacy.driver_cls
    assert mapping.record_id == decision.record_id
    assert decision.record_id == _legacy_record(
        legacy_runtime, legacy
    ).record_id
    assert inactive_manifest.entry_point.load_calls == 1
    assert legacy.entry_point.load_calls == 1
    assert legacy_runtime.registry.get_selection(target_name) is decision
    assert set(legacy_runtime.backends_module.backends) == {legacy.name}


# ---------------------------------------------------------------------------
# Reset, concurrency and lifecycle generation


def test_system_exit_in_reset_invalidator_cannot_leave_core_zombies(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "reset_invalidator_system_exit_target"
    spec = legacy_runtime.fixture(
        LegacySpec("reset_invalidator_system_exit", target_name)
    )
    registry = BackendPluginRegistry(
        distribution_provider=lambda: (spec.distribution,),
        environment_provider=collect_core_environment,
        preflight_profile="triton_version",
    )
    record = next(
        item
        for item in registry.list()
        if item.entry_point_name == spec.name
    )
    lease = registry.materialize_legacy(record.record_id)
    registry.select(
        target_name,
        explicit_selector=record.registry_key,
        environment={},
    )
    assert registry.get_selection(target_name) is not None
    assert registry._records
    assert registry._legacy_leases
    assert registry._legacy_lease_tokens
    hook_calls = 0

    def interrupt_invalidation() -> None:
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 1:
            raise SystemExit("reset invalidation SystemExit sentinel")

    registry.register_reset_invalidation_hook(interrupt_invalidation)

    with pytest.raises(
        SystemExit, match="reset invalidation SystemExit sentinel"
    ):
        registry.reset()

    assert registry._records == {}
    assert registry._selections == {}
    assert registry._legacy_leases == {}
    assert registry._legacy_lease_tokens == {}
    assert registry._cleanup_stack == []
    with pytest.raises(BackendPluginLifecycleError):
        registry.select_materialized_legacy(target_name, lease=lease)
    assert registry.reset() == ()


def test_reset_clears_legacy_mapping_selection_and_active_driver_then_relazies(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("reset_legacy", "reset_target")
    )
    legacy_runtime.install(spec)
    target = legacy_runtime.target(spec.target)

    first = legacy_runtime.backends_module.make_backend(target)
    active_target = (
        legacy_runtime.runtime_driver_module.driver.active.get_current_target()
    )
    assert first.__class__ is spec.compiler_cls
    assert active_target.backend == spec.target
    assert spec.name in legacy_runtime.backends_module.backends
    assert legacy_runtime.registry.get_selection(spec.target) is not None
    assert legacy_runtime.runtime_driver_module.driver.default._obj is not None
    first_load_count = spec.entry_point.load_calls

    assert legacy_runtime.registry.reset() == ()

    assert legacy_runtime.backends_module.backends == {}
    assert legacy_runtime.registry.get_selection(spec.target) is None
    assert legacy_runtime.runtime_driver_module.driver.active is (
        legacy_runtime.runtime_driver_module.driver.default
    )
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None
    assert spec.entry_point.load_calls == first_load_count

    second = legacy_runtime.backends_module.make_backend(target)
    assert second.__class__ is spec.compiler_cls
    assert spec.entry_point.load_calls == first_load_count + 1
    assert spec.name in legacy_runtime.backends_module.backends
    assert legacy_runtime.registry.get_selection(spec.target) is not None


def test_repeated_reset_is_idempotent_for_legacy_state(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec("idempotent_reset", "idempotent_reset_target")
    )
    legacy_runtime.install(spec)
    legacy_runtime.backends_module.make_backend(
        legacy_runtime.target(spec.target)
    )

    generations = []
    for _ in range(3):
        assert legacy_runtime.registry.reset() == ()
        generations.append(legacy_runtime.registry.generation)
        assert legacy_runtime.backends_module.backends == {}
        assert legacy_runtime.registry.get_selection(spec.target) is None
        assert legacy_runtime.runtime_driver_module.driver.default._obj is None
    assert generations == sorted(set(generations))


def test_speculative_cleanup_does_not_remove_new_same_record_mapping_after_reset(
    legacy_runtime: LegacyHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_name = "speculative_cleanup_same_record_target"
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "speculative_cleanup_same_record",
            target_name,
            active=False,
            manifest=_manifest(
                "speculative_cleanup_same_record", target_name
            ),
        )
    )
    legacy_runtime.install(manifest)
    resolution = (
        legacy_runtime.backends_module._get_driver_backend_resolution()
    )
    assert len(resolution.speculative_decisions) == 1
    old_decision = resolution.speculative_decisions[0]
    original_release = (
        legacy_runtime.registry.release_selections_if_current
    )
    interleaved: dict[str, Any] = {}

    def release_then_reselect(decisions: Any) -> Any:
        released = original_release(decisions)
        assert legacy_runtime.registry.reset() == ()
        new_decision = legacy_runtime.registry.select(
            target_name, environment={}
        )
        new_backend = legacy_runtime.backends_module._cache_decision(
            new_decision
        )
        interleaved.update(
            decision=new_decision,
            backend=new_backend,
        )
        return released

    monkeypatch.setattr(
        legacy_runtime.registry,
        "release_selections_if_current",
        release_then_reselect,
    )

    with pytest.raises(BackendPluginLifecycleError) as caught:
        legacy_runtime.backends_module._release_speculative_driver_selections(
            resolution
        )

    assert caught.value.field == "registry reset"
    assert interleaved["decision"] is not old_decision
    assert legacy_runtime.registry.get_selection(target_name) is (
        interleaved["decision"]
    )
    assert legacy_runtime.backends_module.backends[manifest.name] is (
        interleaved["backend"]
    )


def test_rejected_legacy_racing_successful_publication_leaves_no_mapping(
    legacy_runtime: LegacyHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_name = "rejected_publication_race_target"
    spec = legacy_runtime.fixture(
        LegacySpec("rejected_publication_race", target_name)
    )
    legacy_runtime.install(spec)
    record = _legacy_record(legacy_runtime, spec)
    lease = _materialize(legacy_runtime, record.record_id)
    expected_epoch = legacy_runtime.backends_module._cache_epoch()
    publisher_entered = threading.Event()
    rejection_waiting = threading.Event()
    original_reject = legacy_runtime.registry.reject_materialized_legacy

    def observed_reject(*args: Any, **kwargs: Any) -> Any:
        rejection_waiting.set()
        return original_reject(*args, **kwargs)

    monkeypatch.setattr(
        legacy_runtime.registry,
        "reject_materialized_legacy",
        observed_reject,
    )

    def publisher(decision: Any, selected: Any) -> Any:
        publisher_entered.set()
        if not rejection_waiting.wait(timeout=5):
            raise RuntimeError("Legacy rejection did not reach Core")
        return legacy_runtime.backends_module._publish_legacy_mapping(
            decision,
            selected,
            expected_epoch,
        )

    def publish_successfully() -> Any:
        return legacy_runtime.registry.commit_materialized_legacy(
            legacy_runtime.target(target_name),
            lease=lease,
            publisher=publisher,
        )

    rejection_error = BackendPluginLifecycleError(
        "concurrent Legacy rejection sentinel",
        field="compiler_cls.supports_target",
        expected="successful callback",
        actual="<error: rejection sentinel>",
        remediation="Reject this exact Legacy lease.",
    )

    def reject_concurrently() -> BackendPluginLifecycleError:
        try:
            legacy_runtime.backends_module._reject_legacy_failure(
                legacy_runtime.registry,
                lease,
                rejection_error,
            )
        except BackendPluginLifecycleError as caught:
            return caught
        raise AssertionError("Legacy rejection unexpectedly returned")

    with ThreadPoolExecutor(max_workers=2) as executor:
        publication = executor.submit(publish_successfully)
        assert publisher_entered.wait(timeout=2)
        rejection = executor.submit(reject_concurrently)
        _decision, selected, published_backend = publication.result(timeout=5)
        caught = rejection.result(timeout=5)

    assert caught is rejection_error
    assert selected.record_id == record.record_id
    assert published_backend.record_id == record.record_id
    rejected = legacy_runtime.registry.inspect(record.record_id)
    assert rejected.state is PluginLifecycleState.REJECTED
    assert legacy_runtime.registry.get_selection(target_name) is None
    assert spec.name not in legacy_runtime.backends_module.backends


def test_concurrent_compiler_requests_publish_one_legacy_selection(
    legacy_runtime: LegacyHarness,
) -> None:
    probes = threading.Barrier(8)

    def supports(target: Any) -> bool:
        probes.wait(timeout=5)
        return target.backend == "concurrent_compiler_target"

    spec = legacy_runtime.fixture(
        LegacySpec(
            "concurrent_compiler",
            "concurrent_compiler_target",
            supports=supports,
        )
    )
    legacy_runtime.install(spec)
    target = legacy_runtime.target(spec.target)
    legacy_runtime.registry.list()
    generation = legacy_runtime.registry.generation

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(legacy_runtime.backends_module.make_backend, target)
            for _ in range(8)
        ]
        compilers = [future.result(timeout=10) for future in futures]

    assert all(compiler.__class__ is spec.compiler_cls for compiler in compilers)
    assert spec.entry_point.load_calls == 1
    assert len(legacy_runtime.registry.list()) == 1
    assert legacy_runtime.registry.generation == generation + 1
    decision = legacy_runtime.registry.get_selection(spec.target)
    mapping = legacy_runtime.backends_module.backends
    assert set(mapping) == {spec.name}
    assert mapping[spec.name].record_id == decision.record_id


def test_compiler_runtime_race_with_different_legacy_winners_never_splits_pair(
    legacy_runtime: LegacyHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_name = "concurrent_different_winner_target"
    role = threading.local()

    def compiler_a_supports(_target: Any) -> bool:
        return getattr(role, "value", None) == "compiler"

    def compiler_b_supports(_target: Any) -> bool:
        return getattr(role, "value", None) == "runtime"

    def driver_a_active() -> bool:
        return False

    def driver_b_active() -> bool:
        return getattr(role, "value", None) == "runtime"

    compiler_winner = legacy_runtime.fixture(
        LegacySpec(
            "concurrent_compiler_only_winner",
            target_name,
            supports=compiler_a_supports,
            active=driver_a_active,
        )
    )
    runtime_winner = legacy_runtime.fixture(
        LegacySpec(
            "concurrent_runtime_only_winner",
            target_name,
            supports=compiler_b_supports,
            active=driver_b_active,
        )
    )
    legacy_runtime.install(compiler_winner, runtime_winner)
    publish_barrier = threading.Barrier(2)
    original_select = legacy_runtime.registry.select_materialized_legacy
    original_commit = getattr(
        legacy_runtime.registry,
        "commit_materialized_legacy",
        None,
    )
    original_atomic = getattr(
        legacy_runtime.registry,
        "select_and_activate_materialized_legacy",
        None,
    )

    def synchronized_select(*args: Any, **kwargs: Any) -> Any:
        if getattr(role, "publication_synchronized", False):
            return original_select(*args, **kwargs)
        role.publication_synchronized = True
        publish_barrier.wait(timeout=5)
        return original_select(*args, **kwargs)

    if callable(original_commit):

        def synchronized_commit(*args: Any, **kwargs: Any) -> Any:
            if not getattr(role, "publication_synchronized", False):
                role.publication_synchronized = True
                publish_barrier.wait(timeout=5)
            return original_commit(*args, **kwargs)

        monkeypatch.setattr(
            legacy_runtime.registry,
            "commit_materialized_legacy",
            synchronized_commit,
        )
    else:
        monkeypatch.setattr(
            legacy_runtime.registry,
            "select_materialized_legacy",
            synchronized_select,
        )

    if callable(original_atomic) and not callable(original_commit):

        def synchronized_atomic(*args: Any, **kwargs: Any) -> Any:
            role.publication_synchronized = True
            publish_barrier.wait(timeout=5)
            return original_atomic(*args, **kwargs)

        monkeypatch.setattr(
            legacy_runtime.registry,
            "select_and_activate_materialized_legacy",
            synchronized_atomic,
        )

    def compile_candidate() -> Any:
        role.value = "compiler"
        return legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(target_name)
        )

    def runtime_candidate() -> Any:
        role.value = "runtime"
        return legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    outcomes: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            "compiler": executor.submit(compile_candidate),
            "runtime": executor.submit(runtime_candidate),
        }
        for name, future in futures.items():
            try:
                outcomes[name] = future.result(timeout=10)
            except BaseException as error:
                errors[name] = error

    assert set(outcomes) in ({"compiler"}, {"runtime"}), {
        name: repr(error) for name, error in errors.items()
    }
    assert set(errors) == (
        {"runtime"} if "compiler" in outcomes else {"compiler"}
    ), {name: repr(error) for name, error in errors.items()}
    assert isinstance(next(iter(errors.values())), BackendPluginConflictError)
    decision = legacy_runtime.registry.get_selection(target_name)
    assert decision is not None
    mapping = legacy_runtime.backends_module.backends
    if "compiler" in outcomes:
        assert outcomes["compiler"].__class__ is compiler_winner.compiler_cls
        assert set(mapping) == {compiler_winner.name}
        assert mapping[compiler_winner.name].record_id == decision.record_id
        assert legacy_runtime.runtime_driver_module.driver.default._obj is None
    else:
        assert outcomes["runtime"].backend == target_name
        assert set(mapping) == {runtime_winner.name}
        assert mapping[runtime_winner.name].record_id == decision.record_id
        assert (
            legacy_runtime.runtime_driver_module.driver.default._obj.__class__
            is runtime_winner.driver_cls
        )


def test_concurrent_compiler_first_and_runtime_first_converge_on_one_record(
    legacy_runtime: LegacyHarness,
) -> None:
    winner = legacy_runtime.fixture(
        LegacySpec("concurrent_pair_winner", "concurrent_pair_target")
    )
    loser = legacy_runtime.fixture(
        LegacySpec(
            "concurrent_pair_loser",
            "concurrent_pair_target",
            supports=False,
            active=False,
        )
    )
    legacy_runtime.install(winner, loser)
    target = legacy_runtime.target(winner.target)
    start = threading.Barrier(2)

    def compile_first():
        start.wait(timeout=5)
        return legacy_runtime.backends_module.make_backend(target)

    def runtime_first():
        start.wait(timeout=5)
        return (
            legacy_runtime.runtime_driver_module.driver.active.get_current_target()
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        compiler_future = executor.submit(compile_first)
        runtime_future = executor.submit(runtime_first)
        compiler = compiler_future.result(timeout=10)
        runtime_target = runtime_future.result(timeout=10)

    decision = legacy_runtime.registry.get_selection(winner.target)
    mapping = legacy_runtime.backends_module.backends[winner.name]
    concrete_driver = legacy_runtime.runtime_driver_module.driver.default._obj
    assert compiler.__class__ is winner.compiler_cls
    assert runtime_target.backend == winner.target
    assert concrete_driver.__class__ is winner.driver_cls
    assert mapping.compiler is winner.compiler_cls
    assert mapping.driver is winner.driver_cls
    assert mapping.record_id == decision.record_id
    assert decision.record_id == _legacy_record(
        legacy_runtime, winner
    ).record_id
    assert loser.name not in legacy_runtime.backends_module.backends
    assert winner.entry_point.load_calls == 1


@pytest.mark.parametrize(
    "blocked_stage",
    (
        "supports_target",
        "compiler_constructor",
        "is_active",
        "driver_constructor",
        "get_current_target",
    ),
)
def test_reset_during_legacy_plugin_code_rejects_same_record_id_aba_result(
    legacy_runtime: LegacyHarness,
    blocked_stage: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    target_name = f"aba_{blocked_stage}_target"

    def block(*_args: Any) -> Any:
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("test barrier timed out")
        if blocked_stage == "is_active":
            return True
        if blocked_stage == "get_current_target":
            return target_name
        if blocked_stage == "supports_target":
            return True
        return None

    kwargs: dict[str, Any] = {}
    if blocked_stage == "supports_target":
        kwargs["supports"] = block
    elif blocked_stage == "compiler_constructor":
        kwargs["compiler_error"] = block
    elif blocked_stage == "is_active":
        kwargs["active"] = block
    elif blocked_stage == "driver_constructor":
        kwargs["driver_error"] = block
    else:
        kwargs["current_target"] = block

    spec = legacy_runtime.fixture(
        LegacySpec(f"aba_{blocked_stage}", target_name, **kwargs)
    )
    legacy_runtime.install(spec)
    original = _legacy_record(legacy_runtime, spec)
    compiler_path = blocked_stage in {
        "supports_target",
        "compiler_constructor",
    }

    def consume():
        if compiler_path:
            return legacy_runtime.backends_module.make_backend(
                legacy_runtime.target(target_name)
            )
        return legacy_runtime.runtime_driver_module.driver.active.get_current_target()

    with ThreadPoolExecutor(max_workers=2) as executor:
        operation = executor.submit(consume)
        try:
            assert entered.wait(timeout=1), (
                f"Legacy fallback did not reach {blocked_stage}"
            )
            reset = executor.submit(legacy_runtime.registry.reset)
            assert reset.result(timeout=3) == ()
            recreated = _legacy_record(legacy_runtime, spec)
            assert recreated.record_id == original.record_id
        finally:
            release.set()

        with pytest.raises(BackendPluginLifecycleError) as caught:
            operation.result(timeout=5)
    assert caught.value.field in {
        "registry generation",
        "registry lifecycle_epoch",
        "registry reset",
    }
    legacy_runtime.assert_unpublished(target_name)


def test_duplicate_legacy_entry_point_identity_fails_before_import(
    legacy_runtime: LegacyHarness,
) -> None:
    first = legacy_runtime.fixture(
        LegacySpec(
            "duplicate_legacy",
            "duplicate_target",
            distribution_name="t63-f5-duplicate-a",
        )
    )
    second = legacy_runtime.fixture(
        LegacySpec(
            "duplicate_legacy",
            "duplicate_target",
            distribution_name="t63-f5-duplicate-b",
        )
    )
    legacy_runtime.install(first, second)

    with pytest.raises(BackendPluginConflictError):
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(first.target)
        )
    assert first.entry_point.load_calls == 0
    assert second.entry_point.load_calls == 0
    legacy_runtime.assert_unpublished(first.target)


# ---------------------------------------------------------------------------
# Adversarial F6, ownership, rollback, and BaseException boundaries


@pytest.mark.parametrize("consumer", ("compiler", "driver"))
def test_manual_incomplete_pair_fails_f6_before_any_plugin_callback(
    legacy_runtime: LegacyHarness,
    consumer: str,
) -> None:
    target_name = f"manual_incomplete_{consumer}_target"
    spec = legacy_runtime.fixture(
        LegacySpec(f"manual_incomplete_{consumer}", target_name)
    )

    def reabstract_one_surface(
        name: str,
        valid_class: type,
        contract: type,
    ) -> type:
        required = tuple(sorted(contract.__abstractmethods__))
        assert required, "the live Triton contract has no abstract surface"
        missing_member = required[0]

        @abstractmethod
        def missing(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("an abstract runtime member was invoked")

        missing.__name__ = missing_member
        return type(name, (valid_class,), {missing_member: missing})

    incomplete_compiler = reabstract_one_surface(
        "IncompleteCompiler",
        spec.compiler_cls,
        legacy_runtime.compiler_contract,
    )
    incomplete_driver = reabstract_one_surface(
        "IncompleteDriver",
        spec.driver_cls,
        legacy_runtime.driver_contract,
    )

    compiler_cls = (
        incomplete_compiler
        if consumer == "compiler"
        else spec.compiler_cls
    )
    driver_cls = (
        incomplete_driver if consumer == "driver" else spec.driver_cls
    )
    original = legacy_runtime.backends_module.Backend(
        compiler=compiler_cls,
        driver=driver_cls,
    )
    legacy_runtime.backends_module.backends[spec.name] = original

    with pytest.raises(BackendPluginInterfaceError):
        if consumer == "compiler":
            legacy_runtime.backends_module.make_backend(
                legacy_runtime.target(target_name)
            )
        else:
            (
                legacy_runtime.runtime_driver_module.driver.active
                .get_current_target()
            )

    assert all(
        spec.calls[name] == 0
        for name in (
            "supports_target",
            "compiler_constructor",
            "is_active",
            "driver_constructor",
            "get_current_target",
        )
    )
    assert legacy_runtime.backends_module.backends[spec.name] is original
    assert legacy_runtime.backends_module._legacy_target_owners == {}
    assert legacy_runtime.registry.get_selection(target_name) is None
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


@pytest.mark.parametrize("consumer", ("compiler", "driver"))
@pytest.mark.parametrize("mapping_value", ("foreign", "backend_subclass"))
def test_manual_mapping_rejects_non_exact_backend_without_attribute_hooks(
    legacy_runtime: LegacyHarness,
    consumer: str,
    mapping_value: str,
) -> None:
    target_name = f"non_exact_mapping_{mapping_value}_{consumer}_target"
    spec = legacy_runtime.fixture(
        LegacySpec(
            f"non_exact_mapping_{mapping_value}_{consumer}", target_name
        )
    )
    attribute_reads: list[str] = []

    class ForeignMappingValue:
        def __getattribute__(self, name: str) -> Any:
            attribute_reads.append(name)
            raise AssertionError(f"foreign mapping attribute read: {name}")

    class BackendSubclass(legacy_runtime.backends_module.Backend):
        def __getattribute__(self, name: str) -> Any:
            attribute_reads.append(name)
            raise AssertionError(f"Backend subclass attribute read: {name}")

    if mapping_value == "foreign":
        value: Any = ForeignMappingValue()
    else:
        value = BackendSubclass(
            compiler=spec.compiler_cls,
            driver=spec.driver_cls,
        )
    legacy_runtime.backends_module.backends[spec.name] = value

    try:
        with pytest.raises(BackendPluginSelectionError) as caught:
            if consumer == "compiler":
                legacy_runtime.backends_module.make_backend(
                    legacy_runtime.target(target_name)
                )
            else:
                (
                    legacy_runtime.runtime_driver_module.driver.active
                    .get_current_target()
                )
        assert caught.value.field == "backends"
        assert attribute_reads == []
        assert legacy_runtime.backends_module._legacy_target_owners == {}
        assert legacy_runtime.registry.get_selection(target_name) is None
        assert legacy_runtime.runtime_driver_module.driver.default._obj is None
    finally:
        assert (
            legacy_runtime.backends_module.backends.pop(spec.name) is value
        )


@pytest.mark.parametrize("mapping_value", ("foreign", "backend_subclass"))
def test_reset_ignores_non_exact_manual_mapping_without_attribute_hooks(
    legacy_runtime: LegacyHarness,
    mapping_value: str,
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec(f"reset_non_exact_{mapping_value}", "unused_target")
    )
    attribute_reads: list[str] = []

    class ForeignMappingValue:
        def __getattribute__(self, name: str) -> Any:
            attribute_reads.append(name)
            raise AssertionError(f"foreign mapping attribute read: {name}")

    class BackendSubclass(legacy_runtime.backends_module.Backend):
        def __getattribute__(self, name: str) -> Any:
            attribute_reads.append(name)
            raise AssertionError(f"Backend subclass attribute read: {name}")

    value: Any
    if mapping_value == "foreign":
        value = ForeignMappingValue()
    else:
        value = BackendSubclass(
            compiler=spec.compiler_cls,
            driver=spec.driver_cls,
        )
    legacy_runtime.backends_module.backends[spec.name] = value

    try:
        assert legacy_runtime.registry.reset() == ()
        assert attribute_reads == []
        assert legacy_runtime.backends_module.backends[spec.name] is value
        assert legacy_runtime.backends_module._legacy_target_owners == {}
        assert legacy_runtime.runtime_driver_module.driver.default._obj is None
    finally:
        assert (
            legacy_runtime.backends_module.backends.pop(spec.name) is value
        )


def test_lazy_driver_same_thread_callback_reentry_fails_without_hanging(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "lazy_driver_same_thread_reentry_target"

    def reenter_driver() -> Any:
        return (
            legacy_runtime.runtime_driver_module.driver.active
            .get_current_target()
        )

    spec = legacy_runtime.fixture(
        LegacySpec(
            "lazy_driver_same_thread_reentry",
            target_name,
            active=reenter_driver,
        )
    )
    legacy_runtime.install(spec)
    outcome: dict[str, Any] = {}

    def consume() -> None:
        try:
            outcome["value"] = (
                legacy_runtime.runtime_driver_module.driver.active
                .get_current_target()
            )
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    worker.join(timeout=3)

    assert not worker.is_alive(), "same-thread LazyProxy re-entry deadlocked"
    assert set(outcome) == {"error"}
    error = outcome["error"]
    assert isinstance(error, BackendPluginLifecycleError)
    assert error.field == "driver initialization"
    assert spec.calls["is_active"] == 1
    assert spec.calls["driver_constructor"] == 0
    assert spec.calls["get_current_target"] == 0
    legacy_runtime.assert_unpublished(target_name)


def test_manual_target_claim_rejects_reset_during_second_registry_read(
    legacy_runtime: LegacyHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_name = "manual_claim_second_read_reset_target"
    spec = legacy_runtime.fixture(
        LegacySpec("manual_claim_second_read_reset", target_name)
    )
    backend = legacy_runtime.backends_module.Backend(
        compiler=spec.compiler_cls,
        driver=spec.driver_cls,
        entry_point_name=spec.name,
    )
    legacy_runtime.backends_module.backends[spec.name] = backend
    expected_epoch = legacy_runtime.backends_module._cache_epoch()
    original_get_selection = legacy_runtime.registry.get_selection
    reads = 0

    def reset_on_second_read(target: str) -> Any:
        nonlocal reads
        reads += 1
        if reads == 2:
            assert legacy_runtime.registry.reset() == ()
        return original_get_selection(target)

    monkeypatch.setattr(
        legacy_runtime.registry,
        "get_selection",
        reset_on_second_read,
    )

    with pytest.raises(BackendPluginLifecycleError) as caught:
        legacy_runtime.backends_module._claim_manual_legacy_target(
            legacy_runtime.registry,
            backend,
            legacy_runtime.target(target_name),
            expected_epoch,
        )

    assert reads == 2
    assert caught.value.field == "registry reset"
    assert legacy_runtime.backends_module._legacy_target_owners == {}
    assert original_get_selection(target_name) is None
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


def test_runtime_build_rollback_keeps_concurrent_same_record_target(
    legacy_runtime: LegacyHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_a = "runtime_refresh_a_owned_target"
    owned_b = "runtime_refresh_b_owned_target"
    concurrent_target = "runtime_refresh_c_concurrent_target"
    failing_target = "runtime_refresh_z_failure_target"
    manifest = _manifest("runtime_refresh_shared", owned_a)
    manifest["plugins"][0]["targets"] = [
        owned_a,
        owned_b,
        concurrent_target,
    ]
    shared = legacy_runtime.fixture(
        LegacySpec(
            "runtime_refresh_shared",
            owned_a,
            manifest=manifest,
        )
    )
    failing = legacy_runtime.fixture(
        LegacySpec(
            "runtime_refresh_failure",
            failing_target,
            load_error=RuntimeError("same-record refresh failure sentinel"),
            manifest=_manifest(
                "runtime_refresh_failure", failing_target
            ),
        )
    )
    legacy_runtime.install(shared, failing)
    original_select = legacy_runtime.registry.select
    interleaved: dict[str, Any] = {}

    def select_with_concurrent_target(target: Any, **kwargs: Any) -> Any:
        decision = original_select(target, **kwargs)
        if target == owned_a and not interleaved:
            interleaved["decision"] = original_select(
                concurrent_target,
                environment={},
            )
        return decision

    monkeypatch.setattr(
        legacy_runtime.registry,
        "select",
        select_with_concurrent_target,
    )

    with pytest.raises(BackendPluginError) as caught:
        legacy_runtime.backends_module._get_driver_backend_resolution()

    assert "same-record refresh failure sentinel" in str(caught.value)
    assert set(interleaved) == {"decision"}
    assert legacy_runtime.registry.get_selection(owned_a) is None
    assert legacy_runtime.registry.get_selection(owned_b) is None
    concurrent = legacy_runtime.registry.get_selection(concurrent_target)
    assert concurrent is not None
    assert (
        concurrent.ownership_token
        is interleaved["decision"].ownership_token
    )
    record = legacy_runtime.registry.inspect(concurrent.record_id)
    assert concurrent.record is record
    assert record.selected_targets == (concurrent_target,)
    assert record.state is PluginLifecycleState.SELECTED
    assert legacy_runtime.registry.get_selection(failing_target) is None
    assert legacy_runtime.backends_module.backends == {}
    assert legacy_runtime.backends_module._legacy_target_owners == {}
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


def test_runtime_build_rollback_keeps_new_same_target_ownership(
    legacy_runtime: LegacyHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replaced_target = "runtime_same_target_a_replaced"
    owned_target = "runtime_same_target_b_owned"
    failing_target = "runtime_same_target_z_failure"
    shared_manifest = _manifest("runtime_same_target_shared", replaced_target)
    shared_manifest["plugins"][0]["targets"] = [
        replaced_target,
        owned_target,
    ]
    shared = legacy_runtime.fixture(
        LegacySpec(
            "runtime_same_target_shared",
            replaced_target,
            manifest=shared_manifest,
        )
    )
    failing = legacy_runtime.fixture(
        LegacySpec(
            "runtime_same_target_failure",
            failing_target,
            load_error=RuntimeError("same-target ownership failure sentinel"),
            manifest=_manifest(
                "runtime_same_target_failure", failing_target
            ),
        )
    )
    legacy_runtime.install(shared, failing)
    original_select = legacy_runtime.registry.select
    interleaved: dict[str, Any] = {}

    def select_then_replace_same_target(target: Any, **kwargs: Any) -> Any:
        selected = original_select(target, **kwargs)
        if target == replaced_target and not interleaved:
            legacy_runtime.registry.release_selection_if_current(selected)
            replacement = original_select(target, environment={})
            assert replacement.ownership_token is not selected.ownership_token
            interleaved.update(old=selected, replacement=replacement)
        return selected

    monkeypatch.setattr(
        legacy_runtime.registry,
        "select",
        select_then_replace_same_target,
    )

    with pytest.raises(BackendPluginError) as caught:
        legacy_runtime.backends_module._get_driver_backend_resolution()

    assert "same-target ownership failure sentinel" in str(caught.value)
    assert set(interleaved) == {"old", "replacement"}
    current = legacy_runtime.registry.get_selection(replaced_target)
    assert current is not None
    assert current.ownership_token is interleaved["replacement"].ownership_token
    assert current.ownership_token is not interleaved["old"].ownership_token
    assert legacy_runtime.registry.get_selection(owned_target) is None
    record = legacy_runtime.registry.inspect(current.record_id)
    assert current.record is record
    assert record.selected_targets == (replaced_target,)
    assert record.state is PluginLifecycleState.SELECTED
    assert legacy_runtime.registry.get_selection(failing_target) is None
    assert legacy_runtime.backends_module.backends == {}
    assert legacy_runtime.backends_module._legacy_target_owners == {}
    assert legacy_runtime.runtime_driver_module.driver.default._obj is None


def test_forged_selection_string_subclasses_are_rejected_before_hooks(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "forged_selection_string_target"
    spec = legacy_runtime.fixture(
        LegacySpec(
            "forged_selection_string",
            target_name,
            manifest=_manifest("forged_selection_string", target_name),
        )
    )
    legacy_runtime.install(spec)
    decision = legacy_runtime.registry.select(target_name, environment={})
    hooks: Counter[str] = Counter()
    publications: list[Any] = []

    class HostileString(str):
        def __hash__(self) -> int:
            hooks["hash"] += 1
            raise AssertionError("hostile string hash hook ran")

        def __eq__(self, _other: object) -> bool:
            hooks["eq"] += 1
            raise AssertionError("hostile string equality hook ran")

    for field_name in ("target", "record_id"):
        forged = replace(
            decision,
            **{field_name: HostileString(getattr(decision, field_name))},
        )

        with pytest.raises(BackendPluginLifecycleError) as committed:
            legacy_runtime.registry.commit_selection_if_current(
                forged,
                publisher=lambda *_args: publications.append(_args),
                activate=True,
            )
        assert committed.value.field == f"selection.{field_name}"

        with pytest.raises(BackendPluginLifecycleError) as released:
            legacy_runtime.registry.release_selections_if_current((forged,))
        assert released.value.field == f"selection.{field_name}"

        assert hooks == Counter()
        assert publications == []
        assert legacy_runtime.registry.get_selection(target_name) is decision
        assert legacy_runtime.registry.inspect(decision.record_id).state is (
            PluginLifecycleState.SELECTED
        )


def test_failed_activating_commit_restores_canonical_legacy_lease(
    legacy_runtime: LegacyHarness,
) -> None:
    target_name = "failed_activate_legacy_lease_target"
    spec = legacy_runtime.fixture(
        LegacySpec("failed_activate_legacy_lease", target_name)
    )
    legacy_runtime.install(spec)
    metadata_record = _legacy_record(legacy_runtime, spec)
    lease = _materialize(legacy_runtime, metadata_record.record_id)
    decision = legacy_runtime.registry.select_materialized_legacy(
        target_name,
        lease=lease,
    )
    selected_record = decision.record
    failed_publications: list[Any] = []

    def fail_publication(current_decision: Any, current_record: Any) -> None:
        failed_publications.append((current_decision, current_record))
        assert current_record.state is PluginLifecycleState.ACTIVE
        raise RuntimeError("activating publisher failure sentinel")

    with pytest.raises(
        RuntimeError, match="activating publisher failure sentinel"
    ):
        legacy_runtime.registry.commit_selection_if_current(
            decision,
            publisher=fail_publication,
            activate=True,
        )

    assert len(failed_publications) == 1
    assert legacy_runtime.registry.inspect(decision.record_id) is selected_record
    assert lease.record is selected_record
    assert legacy_runtime.registry.get_selection(target_name) is decision
    assert selected_record.state is PluginLifecycleState.SELECTED

    publication_sentinel = object()
    reused_decision, active_record, publication = (
        legacy_runtime.registry.commit_materialized_legacy(
            target_name,
            lease=lease,
            publisher=lambda *_args: publication_sentinel,
            activate=True,
        )
    )

    assert publication is publication_sentinel
    assert reused_decision.ownership_token is decision.ownership_token
    assert active_record.state is PluginLifecycleState.ACTIVE
    assert lease.record is active_record
    assert legacy_runtime.registry.inspect(decision.record_id) is active_record
    assert legacy_runtime.registry.get_selection(target_name) is reused_decision


@pytest.mark.parametrize(
    ("stage", "supports", "compiler_error"),
    (
        pytest.param(
            "supports_target",
            SystemExit("Manifest compiler supports SystemExit sentinel"),
            None,
            id="supports-target",
        ),
        pytest.param(
            "constructor",
            True,
            SystemExit("Manifest compiler constructor SystemExit sentinel"),
            id="constructor",
        ),
    ),
)
def test_manifest_compiler_system_exit_releases_fresh_state(
    legacy_runtime: LegacyHarness,
    stage: str,
    supports: Any,
    compiler_error: Any,
) -> None:
    target_name = f"manifest_compiler_system_exit_{stage}_target"
    spec = legacy_runtime.fixture(
        LegacySpec(
            f"manifest_compiler_system_exit_{stage}",
            target_name,
            supports=supports,
            compiler_error=compiler_error,
            manifest=_manifest(
                f"manifest_compiler_system_exit_{stage}", target_name
            ),
        )
    )
    legacy_runtime.install(spec)

    with pytest.raises(SystemExit, match="SystemExit sentinel"):
        legacy_runtime.backends_module.make_backend(
            legacy_runtime.target(target_name)
        )

    assert spec.calls["supports_target"] == 1
    assert spec.calls["compiler_constructor"] == (
        1 if stage == "constructor" else 0
    )
    legacy_runtime.assert_unpublished(target_name)


@pytest.mark.parametrize(
    ("stage", "active", "driver_error", "current_target"),
    (
        pytest.param(
            "is_active",
            SystemExit("Manifest driver active SystemExit sentinel"),
            None,
            None,
            id="is-active",
        ),
        pytest.param(
            "constructor",
            True,
            SystemExit("Manifest driver constructor SystemExit sentinel"),
            None,
            id="constructor",
        ),
        pytest.param(
            "get_current_target",
            True,
            None,
            SystemExit("Manifest driver target SystemExit sentinel"),
            id="current-target",
        ),
    ),
)
def test_manifest_driver_system_exit_releases_fresh_state(
    legacy_runtime: LegacyHarness,
    stage: str,
    active: Any,
    driver_error: Any,
    current_target: Any,
) -> None:
    target_name = f"manifest_driver_system_exit_{stage}_target"
    spec = legacy_runtime.fixture(
        LegacySpec(
            f"manifest_driver_system_exit_{stage}",
            target_name,
            active=active,
            driver_error=driver_error,
            current_target=current_target,
            manifest=_manifest(
                f"manifest_driver_system_exit_{stage}", target_name
            ),
        )
    )
    legacy_runtime.install(spec)

    with pytest.raises(SystemExit, match="SystemExit sentinel"):
        (
            legacy_runtime.runtime_driver_module.driver.active
            .get_current_target()
        )

    assert spec.calls["is_active"] == 1
    assert spec.calls["driver_constructor"] == (
        0 if stage == "is_active" else 1
    )
    assert spec.calls["get_current_target"] == (
        1 if stage == "get_current_target" else 0
    )
    legacy_runtime.assert_unpublished(target_name)


# ---------------------------------------------------------------------------
# Real no-Manifest wheel / importlib.metadata E2E


_REAL_LEGACY_SOURCE = r'''
import json
import os
from pathlib import Path

from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.driver import DriverBase


def _emit(event):
    path = os.environ.get("T63_LEGACY_EVENTS")
    if path:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event}, sort_keys=True) + "\n")


_emit("import")


class Compiler(BaseBackend):
    binary_ext = "f5bin"

    @classmethod
    def supports_target(cls, target):
        _emit("supports_target")
        return target.backend == "wheel_legacy"

    def __init__(self, target):
        _emit("compiler_constructor")
        super().__init__(target)

    def hash(self):
        return "real-f5-legacy"

    def parse_options(self, options):
        return options

    def add_stages(self, stages, options):
        return None

    def load_dialects(self, context):
        return None

    def get_module_map(self):
        return {}


class Driver(DriverBase):
    @classmethod
    def is_active(cls):
        _emit("is_active")
        return True

    def __init__(self):
        _emit("driver_constructor")
        super().__init__()

    def get_current_target(self):
        _emit("get_current_target")
        return GPUTarget("wheel_legacy", "fixture-arch", 32)

    def get_active_torch_device(self):
        return "cpu"

    def get_benchmarker(self):
        return lambda _call, *, quantiles, **_kwargs: [0.0 for _ in quantiles]


compiler_cls = Compiler
driver_cls = Driver
'''


def _wheel_digest(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii")


def _build_real_legacy_wheel(directory: Path) -> Path:
    distribution = "t63_real_legacy"
    version = "1.0.0"
    module = "t63_real_legacy"
    dist_info = f"{distribution}-{version}.dist-info"
    files: dict[str, bytes] = {
        f"{module}/__init__.py": _REAL_LEGACY_SOURCE.encode("utf-8"),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            "Name: t63-real-legacy\n"
            f"Version: {version}\n"
        ).encode("utf-8"),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: T6.3 F5 acceptance\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode("utf-8"),
        f"{dist_info}/entry_points.txt": (
            "[triton.backends]\n"
            f"wheel_legacy = {module}\n"
        ).encode("utf-8"),
        f"{dist_info}/top_level.txt": f"{module}\n".encode("utf-8"),
    }
    record_path = f"{dist_info}/RECORD"
    record_stream = io.StringIO(newline="")
    writer = csv.writer(record_stream, lineterminator="\n")
    for name, payload in sorted(files.items()):
        writer.writerow((name, _wheel_digest(payload), str(len(payload))))
    writer.writerow((record_path, "", ""))
    files[record_path] = record_stream.getvalue().encode("utf-8")

    wheel = directory / f"{distribution}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
    return wheel


def _install_real_legacy_wheel(wheel: Path, site: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-compile",
            "--target",
            str(site),
            str(wheel),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture(scope="module")
def real_legacy_result(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("f5-real-legacy-wheel")
    site = root / "site"
    site.mkdir()
    wheel = _build_real_legacy_wheel(root)
    _install_real_legacy_wheel(wheel, site)
    event_path = root / "events.jsonl"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("TRITON_ANCHOR_BACKEND", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(E2E_PROBE),
            "--repository",
            str(REPOSITORY),
            "--site",
            str(site),
            "--events",
            str(event_path),
        ],
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_real_legacy_wheel_discovery_uses_metadata_without_import(
    real_legacy_result: dict[str, Any],
) -> None:
    metadata = real_legacy_result["metadata"]
    discovery = real_legacy_result["discovery"]
    assert metadata["distribution_types"] == ["PathDistribution"]
    assert metadata["entry_point_types"] == ["EntryPoint"]
    assert metadata["entry_points"] == [
        ["wheel_legacy", "t63_real_legacy"]
    ]
    assert metadata["manifest_files"] == []
    assert discovery["events"] == []
    assert discovery["mapping"] == {}
    assert discovery["record"]["source"] == "legacy"
    assert discovery["record"]["compatibility_status"] == (
        "legacy_unverified"
    )
    assert discovery["record"]["loaded"] is False


def test_real_legacy_wheel_compiler_first_is_lazy_and_same_record(
    real_legacy_result: dict[str, Any],
) -> None:
    assert real_legacy_result["compiler"]["ok"], real_legacy_result
    after = real_legacy_result["after_compiler"]
    mapping = after["mapping"]["wheel_legacy"]
    selection = after["selection"]
    assert selection["is_legacy"] is True
    assert selection["record_id"] == mapping["record_id"]
    assert mapping["entry_point"] == "wheel_legacy"
    events = [event["event"] for event in after["events"]]
    assert events.count("import") == 1
    assert "supports_target" in events
    assert "compiler_constructor" in events


def test_real_legacy_wheel_reset_then_runtime_first_relazies_same_pair(
    real_legacy_result: dict[str, Any],
) -> None:
    assert real_legacy_result["reset"] == {"ok": True, "value": []}
    assert real_legacy_result["after_reset"]["mapping"] == {}
    assert real_legacy_result["after_reset"]["selection"] is False
    assert (
        real_legacy_result["after_reset"]["lazy_driver_materialized"]
        is False
    )
    assert real_legacy_result["runtime"] == {
        "ok": True,
        "value": "wheel_legacy",
    }
    after = real_legacy_result["after_runtime"]
    mapping = after["mapping"]["wheel_legacy"]
    assert after["selection"]["record_id"] == mapping["record_id"]
    assert after["active_driver_class"] == "Driver"
    events = [event["event"] for event in after["events"]]
    assert events.count("import") == 1
    assert "is_active" in events
    assert "driver_constructor" in events
    assert "get_current_target" in events
