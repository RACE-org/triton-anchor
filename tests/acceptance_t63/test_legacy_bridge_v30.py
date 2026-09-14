"""Acceptance tests for the Triton 3.0 Legacy backend bridge."""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import types
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import triton_anchor.backends as backend_api
from triton_anchor.backends import (
    BackendPluginInterfaceError,
    BackendPluginLoadError,
    BackendPluginRegistry,
    BackendPluginSelectionError,
    collect_core_environment,
)

REPOSITORY = Path(__file__).resolve().parents[2]
TRITON_PACKAGE = REPOSITORY / "triton/python/triton"
TRITON_COMMIT = "757b6a61e7df814ba806f498f8bb3160f84b120c"


def _outcome(value: Any, *args: Any) -> Any:
    if isinstance(value, BaseException):
        raise value
    if callable(value):
        return value(*args)
    return value


class _EntryPoint:
    group = "triton.backends"

    def __init__(
        self,
        name: str,
        plugin: Any,
        load_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.value = f"t63_v30_{name}:plugin"
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
            relative = Path(name) / "triton_anchor_backend.json"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            self.files = (str(relative),)
            self._manifest_path = path

    def locate_file(self, _path: Any) -> Path:
        assert self._manifest_path is not None
        return self._manifest_path


def _manifest(entry_point: str, target: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "plugins": [
            {
                "plugin_id": f"acceptance.t63.v30.{entry_point}",
                "entry_point": entry_point,
                "backend_protocol": ">=1.0,<2.0",
                "requires_core": ">=0.2,<0.3",
                "requires_triton": {
                    "version": ">=3.0,<3.1",
                    "commit": TRITON_COMMIT,
                },
                "targets": [target],
                "capabilities": ["acceptance.t63"],
                "isolation_mode": "python_only",
                "priority": 0,
            }
        ],
    }


@dataclass
class LegacySpec:
    name: str
    target: str
    supports: Any = True
    active: Any = True
    current_target: Any = None
    load_error: BaseException | None = None
    incomplete_driver: bool = False
    manifest: dict[str, Any] | None = None
    calls: Counter[str] = field(default_factory=Counter)
    entry_point: _EntryPoint | None = None
    distribution: _Distribution | None = None
    compiler_cls: type | None = None
    driver_cls: type | None = None


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
        target_cls = self.target_cls
        current_target_result = (
            spec.current_target if spec.current_target is not None else spec.target
        )

        class Compiler(self.compiler_contract):
            binary_ext = "fixturebin"

            @classmethod
            def supports_target(cls, target):
                calls["supports_target"] += 1
                return _outcome(supports_result, target)

            def __init__(self, target):
                calls["compiler_constructor"] += 1
                super().__init__(target)

            def hash(self):
                return f"t63-v30-{spec.name}"

            def parse_options(self, options):
                return options

            def add_stages(self, stages, options):
                return None

            def load_dialects(self, context):
                return None

        class Driver(self.driver_contract):
            @classmethod
            def is_active(cls):
                calls["is_active"] += 1
                return _outcome(active_result)

            def __init__(self):
                calls["driver_constructor"] += 1
                super().__init__()

            def get_current_target(self):
                calls["get_current_target"] += 1
                value = _outcome(current_target_result)
                if hasattr(value, "backend"):
                    return value
                return target_cls(str(value), "fixture-arch", 32)

        compiler_cls: type = Compiler
        driver_cls: type = Driver
        if spec.incomplete_driver:

            class IncompleteDriver(self.driver_contract):
                @classmethod
                def is_active(cls):
                    calls["is_active"] += 1
                    return True

            driver_cls = IncompleteDriver

        class Plugin:
            def initialize(self, _context: Mapping[str, Any]) -> None:
                calls["initialize"] += 1

        plugin = Plugin()
        plugin.compiler_cls = compiler_cls
        plugin.driver_cls = driver_cls
        entry_point = _EntryPoint(spec.name, plugin, spec.load_error)
        distribution = _Distribution(
            name=f"t63-v30-{spec.name}",
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

    def target(self, backend: str):
        return self.target_cls(backend, "fixture-arch", 32)


@pytest.fixture(scope="module")
def legacy_runtime(tmp_path_factory: pytest.TempPathFactory):
    registry_module = importlib.import_module("triton_anchor.backends.registry")
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
    triton.__version__ = "3.0.0"
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
        target_cls=importlib.import_module("triton.backends.compiler").GPUTarget,
        root=tmp_path_factory.mktemp("t63-v30-legacy"),
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


def test_import_triton_backends_publishes_manifestless_legacy_entry_point(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(LegacySpec("sophgo", "sophgo"))
    legacy_runtime.install(spec)

    original_module = legacy_runtime.backends_module
    sys.modules.pop("triton.backends", None)
    try:
        imported = importlib.import_module("triton.backends")
        mapping = imported.backends
    finally:
        sys.modules["triton.backends"] = original_module
        sys.modules["triton"].backends = original_module

    assert mapping["sophgo"].compiler is spec.compiler_cls
    assert mapping["sophgo"].driver is spec.driver_cls
    assert mapping["sophgo"].record_id is None
    assert spec.entry_point.load_calls == 1


def test_get_backend_falls_back_to_unique_legacy_compiler(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(LegacySpec("legacy_compile", "sophgo"))
    legacy_runtime.install(spec)

    backend = legacy_runtime.backends_module.get_backend(
        legacy_runtime.target("sophgo")
    )

    assert backend.compiler is spec.compiler_cls
    assert backend.driver is spec.driver_cls
    assert legacy_runtime.backends_module.backends[spec.name] is backend
    assert spec.calls["supports_target"] == 1


def test_runtime_driver_first_discovers_active_legacy_driver(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(LegacySpec("legacy_runtime", "sophgo"))
    legacy_runtime.install(spec)

    driver = legacy_runtime.runtime_driver_module._create_driver()

    assert driver.get_current_target().backend == "sophgo"
    assert legacy_runtime.backends_module.backends[spec.name].driver is (
        spec.driver_cls
    )
    assert spec.calls["is_active"] == 1
    assert spec.calls["driver_constructor"] == 1


def test_registry_reset_clears_and_relazily_materializes_legacy_mapping(
    legacy_runtime: LegacyHarness,
) -> None:
    spec = legacy_runtime.fixture(LegacySpec("legacy_reset", "sophgo"))
    legacy_runtime.install(spec)
    legacy_runtime.backends_module._discover_backends()
    assert spec.name in legacy_runtime.backends_module.backends

    assert legacy_runtime.registry.reset() == ()
    assert legacy_runtime.backends_module.backends == {}

    backend = legacy_runtime.backends_module.get_backend(
        legacy_runtime.target("sophgo")
    )

    assert backend.compiler is spec.compiler_cls
    assert legacy_runtime.backends_module.backends[spec.name] is backend


def test_manifest_winner_prevents_legacy_compiler_fallback(
    legacy_runtime: LegacyHarness,
) -> None:
    target = "sophgo"
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "manifest_backend",
            target,
            manifest=_manifest("manifest_backend", target),
        )
    )
    legacy = legacy_runtime.fixture(LegacySpec("legacy_shadow", target))
    legacy_runtime.install(manifest, legacy)

    backend = legacy_runtime.backends_module.get_backend(legacy_runtime.target(target))

    assert backend.compiler is manifest.compiler_cls
    assert legacy.entry_point.load_calls == 0
    assert legacy.name not in legacy_runtime.backends_module.backends


def test_multiple_legacy_compiler_matches_raise_structured_ambiguity(
    legacy_runtime: LegacyHarness,
) -> None:
    first = legacy_runtime.fixture(LegacySpec("legacy_first", "sophgo"))
    second = legacy_runtime.fixture(LegacySpec("legacy_second", "sophgo"))
    legacy_runtime.install(first, second)

    with pytest.raises(BackendPluginSelectionError) as caught:
        legacy_runtime.backends_module.get_backend(legacy_runtime.target("sophgo"))

    assert caught.value.field == "compiler_cls.supports_target"
    assert first.name in legacy_runtime.backends_module.backends
    assert second.name in legacy_runtime.backends_module.backends


def test_failed_legacy_load_does_not_publish_partial_backend(
    legacy_runtime: LegacyHarness,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec(
            "legacy_broken_load",
            "sophgo",
            load_error=RuntimeError("boom"),
        )
    )
    legacy_runtime.install(spec)

    # Legacy discovery historically warned and continued. A failed plugin
    # must remain unpublished and diagnosable, without aborting triton import.
    mapping = legacy_runtime.backends_module._discover_backends()

    assert spec.name not in mapping
    record = next(
        record
        for record in legacy_runtime.registry.list()
        if record.entry_point_name == spec.name
    )
    assert isinstance(record.error, BackendPluginLoadError)
    assert record.error.entry_point == spec.name
    assert "boom" in str(record.error)
    diagnostics = capsys.readouterr().err
    assert spec.name in diagnostics
    assert "boom" in diagnostics


def test_incomplete_legacy_runtime_pair_is_structured_and_unpublished(
    legacy_runtime: LegacyHarness,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = legacy_runtime.fixture(
        LegacySpec(
            "legacy_incomplete_driver",
            "sophgo",
            incomplete_driver=True,
        )
    )
    legacy_runtime.install(spec)

    mapping = legacy_runtime.backends_module._discover_backends()

    assert spec.name not in mapping
    diagnostics = capsys.readouterr().err
    assert spec.name in diagnostics
    assert "driver_cls" in diagnostics
    # Suppressing a broken Legacy during enumeration does not relax its
    # interface contract or turn it into a valid runtime candidate.
    record = next(
        record
        for record in legacy_runtime.registry.list()
        if record.entry_point_name == spec.name
    )
    with pytest.raises(BackendPluginInterfaceError) as caught:
        legacy_runtime.backends_module._validate_legacy_runtime_pair(record)
    assert caught.value.entry_point == spec.name
