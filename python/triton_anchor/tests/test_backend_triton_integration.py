"""W9 tests for Triton's Registry-backed compiler/driver adapter."""

import ast
import importlib.util
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import pytest

import triton_anchor.backends as backend_api
from triton_anchor.backends import (
    BACKEND_SELECTOR_ENV,
    BackendPluginCompatibilityError,
    BackendPluginConflictError,
    BackendPluginError,
    BackendPluginLifecycleError,
    BackendPluginProtocolError,
    BackendPluginSelectionError,
    PluginLifecycleState,
)
from triton_anchor.tests.test_backend_registry import (
    FakeDistribution,
    manifest,
)
from triton_anchor.tests.test_backend_registry_selection import (
    registry_for,
    triton_plugin,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_TRITON_ROOT = REPOSITORY_ROOT / "triton" / "python" / "triton"
if SOURCE_TRITON_ROOT.is_dir():
    TRITON_ROOT = SOURCE_TRITON_ROOT
else:
    triton_spec = importlib.util.find_spec("triton")
    assert triton_spec is not None
    assert triton_spec.submodule_search_locations is not None
    TRITON_ROOT = Path(next(iter(triton_spec.submodule_search_locations)))


@dataclass(frozen=True)
class Target:
    backend: str
    arch: int = 1
    warp_size: int = 1


def runtime_plugin(
    label,
    *,
    target_name="mock",
    active=True,
    supports=True,
):
    class Compiler:
        @classmethod
        def supports_target(cls, target):
            return supports and target.backend == target_name

        def __init__(self, target):
            self.target = target

    class Driver:
        active_checks = 0

        @classmethod
        def is_active(cls):
            cls.active_checks += 1
            return active

        def get_current_target(self):
            return Target(target_name)

    Compiler.__name__ = label + "Compiler"
    Driver.__name__ = label + "Driver"
    plugin = type(
        label + "Plugin",
        (),
        {
            "compiler_cls": Compiler,
            "driver_cls": Driver,
        },
    )()
    return plugin, Compiler, Driver


def distribution_for(
    tmp_path,
    entry_point,
    *,
    plugin,
    name,
    declaration=None,
    wheel_text=None,
):
    kwargs = {}
    if wheel_text is not None:
        kwargs["wheel_text"] = wheel_text
    return FakeDistribution(
        tmp_path / name,
        name=name,
        manifest=manifest(
            declaration or triton_plugin(entry_point)
        ),
        entry_points=((entry_point, plugin),),
        **kwargs,
    )


def load_bridge(monkeypatch, registry):
    """Load the real source adapter under an isolated fake Triton package."""
    monkeypatch.setattr(
        backend_api,
        "get_backend_plugin_registry",
        lambda: registry,
    )
    package_name = "_w9_test_triton_" + str(id(registry))
    package = types.ModuleType(package_name)
    package.__path__ = [str(TRITON_ROOT)]
    monkeypatch.setitem(sys.modules, package_name, package)

    compiler_module = types.ModuleType(package_name + ".backends.compiler")
    compiler_module.BaseBackend = type("BaseBackend", (), {})
    driver_module = types.ModuleType(package_name + ".backends.driver")
    driver_module.DriverBase = type("DriverBase", (), {})
    monkeypatch.setitem(
        sys.modules,
        compiler_module.__name__,
        compiler_module,
    )
    monkeypatch.setitem(
        sys.modules,
        driver_module.__name__,
        driver_module,
    )

    module_name = package_name + ".backends"
    path = TRITON_ROOT / "backends" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
        submodule_search_locations=[str(path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return package_name, module


def load_runtime_driver(monkeypatch, package_name):
    runtime_name = package_name + ".runtime"
    runtime_package = types.ModuleType(runtime_name)
    runtime_package.__path__ = [str(TRITON_ROOT / "runtime")]
    monkeypatch.setitem(sys.modules, runtime_name, runtime_package)

    module_name = runtime_name + ".driver"
    path = TRITON_ROOT / "runtime" / "driver.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_w9_import_is_static_and_compiler_driver_share_priority_winner(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    low_plugin, _, _ = runtime_plugin("Low")
    high_plugin, high_compiler, high_driver = runtime_plugin("High")
    low = distribution_for(
        tmp_path,
        "low",
        plugin=low_plugin,
        name="low-backend",
        declaration=triton_plugin("low", priority=1),
    )
    high = distribution_for(
        tmp_path,
        "high",
        plugin=high_plugin,
        name="high-backend",
        declaration=triton_plugin("high", priority=2),
    )
    registry = registry_for((low, high))

    package_name, bridge = load_bridge(monkeypatch, registry)

    assert bridge.backends == {}
    assert [low.entry_points[0].load_calls,
            high.entry_points[0].load_calls] == [0, 0]

    compiler = bridge.make_backend(Target("mock"))
    runtime_driver = load_runtime_driver(
        monkeypatch,
        package_name,
    )._create_driver()

    assert isinstance(compiler, high_compiler)
    assert isinstance(runtime_driver, high_driver)
    assert [low.entry_points[0].load_calls,
            high.entry_points[0].load_calls] == [0, 1]

    high_record = next(
        record
        for record in registry.list()
        if record.plugin_id == "vendor.high"
    )
    assert high_record.state is PluginLifecycleState.ACTIVE
    assert [
        check.dimension
        for check in high_record.compatibility_report.checks
    ] == [
        "wheel platform tag",
        "Backend Plugin Protocol",
        "Triton version",
        "core Python SOABI",
        "core build platform",
    ]
    assert bridge.backends == {}


def test_w9_manifest_binding_preserves_same_key_manual_legacy_catalog(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    manifest_plugin, manifest_compiler, manifest_driver = runtime_plugin(
        "Manifest"
    )
    distribution = distribution_for(
        tmp_path,
        "same",
        plugin=manifest_plugin,
        name="manifest-backend",
    )
    manual_plugin, manual_compiler, manual_driver = runtime_plugin("Manual")
    registry = registry_for((distribution,))
    package_name, bridge = load_bridge(monkeypatch, registry)
    bridge.backends["same"] = bridge.Backend(
        compiler=manual_plugin.compiler_cls,
        driver=manual_plugin.driver_cls,
    )

    compiler = bridge.make_backend(Target("mock"))
    active_driver = load_runtime_driver(
        monkeypatch,
        package_name,
    )._create_driver()

    assert isinstance(compiler, manifest_compiler)
    assert isinstance(active_driver, manifest_driver)
    assert bridge.backends["same"].compiler is manual_compiler
    assert bridge.backends["same"].record_id is None
    assert manual_driver.active_checks == 0

    assert registry.reset() == ()
    assert bridge.backends["same"].compiler is manual_compiler
    assert bridge.backends["same"].record_id is None


def test_w9_registry_legacy_catalog_entry_is_resettable(
    tmp_path,
    monkeypatch,
):
    plugin, compiler_cls, _ = runtime_plugin("RegistryLegacy")
    distribution = FakeDistribution(
        tmp_path / "registry-legacy",
        name="registry-legacy-backend",
        entry_points=(("registry-legacy", plugin),),
    )
    registry = registry_for((distribution,))
    selector = registry.discover()[0].registry_key
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, selector)
    _, bridge = load_bridge(monkeypatch, registry)

    resolved = bridge.get_backend(Target("mock"))

    assert resolved.compiler is compiler_cls
    assert bridge.backends["registry-legacy"].compiler is compiler_cls
    assert bridge.backends["registry-legacy"].record_id == resolved.record_id
    assert distribution.entry_points[0].load_calls == 1

    assert registry.reset() == ()
    assert "registry-legacy" not in bridge.backends


def test_w9_registry_legacy_does_not_replace_same_key_manual_catalog(
    tmp_path,
    monkeypatch,
):
    registry_plugin, registry_compiler, _ = runtime_plugin("RegistryLegacy")
    distribution = FakeDistribution(
        tmp_path / "registry-legacy",
        name="registry-legacy-backend",
        entry_points=(("same", registry_plugin),),
    )
    registry = registry_for((distribution,))
    selector = registry.discover()[0].registry_key
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, selector)
    _, bridge = load_bridge(monkeypatch, registry)
    manual_plugin, manual_compiler, _ = runtime_plugin("ManualLegacy")
    bridge.backends["same"] = bridge.Backend(
        compiler=manual_plugin.compiler_cls,
        driver=manual_plugin.driver_cls,
    )

    resolved = bridge.get_backend(Target("mock"))

    assert resolved.compiler is registry_compiler
    assert bridge.backends["same"].compiler is manual_compiler
    assert bridge.backends["same"].record_id is None

    assert registry.reset() == ()
    assert bridge.backends["same"].compiler is manual_compiler
    assert bridge.backends["same"].record_id is None


def test_w9_triton_mismatch_is_rejected_before_import(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    plugin, _, _ = runtime_plugin("Future")
    distribution = distribution_for(
        tmp_path,
        "future",
        plugin=plugin,
        name="future-backend",
        declaration=triton_plugin(
            "future",
            triton_version=">=9.0",
        ),
    )
    registry = registry_for((distribution,))
    package_name, bridge = load_bridge(monkeypatch, registry)

    with pytest.raises(BackendPluginCompatibilityError) as compiler_error:
        bridge.make_backend(Target("mock"))
    with pytest.raises(BackendPluginCompatibilityError) as driver_error:
        load_runtime_driver(
            monkeypatch,
            package_name,
        )._create_driver()

    assert compiler_error.value.field == "Triton version"
    assert driver_error.value.field == "Triton version"
    assert distribution.entry_points[0].load_calls == 0


def test_w9_default_gate_rejects_protocol_mismatch_before_import(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    plugin, _, _ = runtime_plugin("Staged")
    declaration = triton_plugin("staged")
    declaration.update(
        {
            "backend_protocol": ">=9.0",
            "requires_core": ">=9.0",
            "requires_llvm_version": ">=99.0",
            "requires_mlir_version": ">=99.0",
        }
    )
    distribution = distribution_for(
        tmp_path,
        "staged",
        plugin=plugin,
        name="staged-backend",
        declaration=declaration,
        wheel_text="Wheel-Version: 1.0\nTag: py3-none-any\n",
    )
    registry = registry_for((distribution,))
    _, bridge = load_bridge(monkeypatch, registry)

    with pytest.raises(BackendPluginProtocolError) as caught:
        bridge.make_backend(Target("mock"))

    assert caught.value.field == "backend_protocol"
    assert distribution.entry_points[0].load_calls == 0


def test_w9_environment_selector_bootstraps_only_that_runtime_pair(
    tmp_path,
    monkeypatch,
):
    alpha_plugin, _, _ = runtime_plugin(
        "Alpha",
        target_name="alpha",
    )
    beta_plugin, _, beta_driver = runtime_plugin(
        "Beta",
        target_name="beta",
    )
    alpha = distribution_for(
        tmp_path,
        "alpha",
        plugin=alpha_plugin,
        name="alpha-backend",
        declaration=triton_plugin(
            "alpha",
            targets=("alpha",),
        ),
    )
    beta = distribution_for(
        tmp_path,
        "beta",
        plugin=beta_plugin,
        name="beta-backend",
        declaration=triton_plugin(
            "beta",
            targets=("beta",),
        ),
    )
    registry = registry_for((alpha, beta))
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, "vendor.beta")
    package_name, _ = load_bridge(monkeypatch, registry)

    active_driver = load_runtime_driver(
        monkeypatch,
        package_name,
    )._create_driver()

    assert isinstance(active_driver, beta_driver)
    assert [alpha.entry_points[0].load_calls,
            beta.entry_points[0].load_calls] == [0, 1]
    assert registry.list()[1].state is PluginLifecycleState.ACTIVE


def test_w9_cached_python_selection_overrides_environment(
    tmp_path,
    monkeypatch,
):
    low_plugin, low_compiler, low_driver = runtime_plugin("Low")
    high_plugin, _, _ = runtime_plugin("High")
    low = distribution_for(
        tmp_path,
        "low",
        plugin=low_plugin,
        name="low-backend",
        declaration=triton_plugin("low", priority=1),
    )
    high = distribution_for(
        tmp_path,
        "high",
        plugin=high_plugin,
        name="high-backend",
        declaration=triton_plugin("high", priority=2),
    )
    registry = registry_for((low, high))
    registry.select(
        "mock",
        explicit_selector="vendor.low",
        environment={},
    )
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, "vendor.high")
    package_name, bridge = load_bridge(monkeypatch, registry)

    compiler = bridge.make_backend(Target("mock"))
    active_driver = load_runtime_driver(
        monkeypatch,
        package_name,
    )._create_driver()

    assert isinstance(compiler, low_compiler)
    assert isinstance(active_driver, low_driver)
    assert [low.entry_points[0].load_calls,
            high.entry_points[0].load_calls] == [1, 0]


def test_w9_active_pair_is_stable_if_environment_changes(
    tmp_path,
    monkeypatch,
):
    low_plugin, _, _ = runtime_plugin("Low")
    high_plugin, high_compiler, high_driver = runtime_plugin("High")
    low = distribution_for(
        tmp_path,
        "low",
        plugin=low_plugin,
        name="low-backend",
        declaration=triton_plugin("low", priority=1),
    )
    high = distribution_for(
        tmp_path,
        "high",
        plugin=high_plugin,
        name="high-backend",
        declaration=triton_plugin("high", priority=2),
    )
    registry = registry_for((low, high))
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    package_name, bridge = load_bridge(monkeypatch, registry)
    active_driver = load_runtime_driver(
        monkeypatch,
        package_name,
    )._create_driver()

    monkeypatch.setenv(BACKEND_SELECTOR_ENV, "vendor.low")
    compiler = bridge.make_backend(Target("mock"))

    assert isinstance(active_driver, high_driver)
    assert isinstance(compiler, high_compiler)
    assert [low.entry_points[0].load_calls,
            high.entry_points[0].load_calls] == [0, 1]
    assert next(
        record
        for record in registry.list()
        if record.plugin_id == "vendor.high"
    ).state is PluginLifecycleState.ACTIVE


def test_w9_preserves_manually_registered_legacy_pair(
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    plugin, compiler_cls, driver_cls = runtime_plugin("Manual")
    registry = registry_for(())
    package_name, bridge = load_bridge(monkeypatch, registry)
    bridge.backends["manual"] = bridge.Backend(
        compiler=plugin.compiler_cls,
        driver=plugin.driver_cls,
    )

    compiler = bridge.make_backend(Target("mock"))
    active_driver = load_runtime_driver(
        monkeypatch,
        package_name,
    )._create_driver()

    assert isinstance(compiler, compiler_cls)
    assert isinstance(active_driver, driver_cls)
    assert registry.list() == ()


def test_w9_public_set_active_preserves_registered_legacy_pair(
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    plugin, compiler_cls, driver_cls = runtime_plugin("Manual")
    registry = registry_for(())
    package_name, bridge = load_bridge(monkeypatch, registry)
    bridge.backends["manual"] = bridge.Backend(
        compiler=plugin.compiler_cls,
        driver=plugin.driver_cls,
    )
    runtime = load_runtime_driver(monkeypatch, package_name)
    active_driver = driver_cls()

    runtime.driver.set_active(active_driver)
    compiler = bridge.make_backend(Target("mock"))

    assert runtime.driver.active is active_driver
    assert isinstance(compiler, compiler_cls)
    assert registry.list() == ()


def test_w9_public_set_active_cannot_split_manifest_and_legacy_pair(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    manifest_plugin, manifest_compiler, _ = runtime_plugin("Manifest")
    distribution = distribution_for(
        tmp_path,
        "manifest",
        plugin=manifest_plugin,
        name="manifest-backend",
    )
    manual_plugin, _, manual_driver = runtime_plugin("Manual")
    registry = registry_for((distribution,))
    package_name, bridge = load_bridge(monkeypatch, registry)
    bridge.backends["manual"] = bridge.Backend(
        compiler=manual_plugin.compiler_cls,
        driver=manual_plugin.driver_cls,
    )
    runtime = load_runtime_driver(monkeypatch, package_name)

    with pytest.raises(BackendPluginSelectionError) as caught:
        runtime.driver.set_active(manual_driver())

    assert caught.value.field == "compiler_cls,driver_cls"
    assert runtime.driver.active is runtime.driver.default
    assert isinstance(
        bridge.make_backend(Target("mock")),
        manifest_compiler,
    )
    assert distribution.entry_points[0].load_calls == 1


def test_w9_public_set_active_avoids_plugin_reset_lock_inversion(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    plugin, _, driver_cls = runtime_plugin("Locked")
    distribution = distribution_for(
        tmp_path,
        "locked",
        plugin=plugin,
        name="locked-backend",
    )
    registry = registry_for((distribution,))
    package_name, _ = load_bridge(monkeypatch, registry)
    runtime = load_runtime_driver(monkeypatch, package_name)
    plugin_lock = threading.Lock()
    resetter_has_plugin_lock = threading.Event()
    target_lookup_started = threading.Event()
    setter_errors = []
    reset_results = []

    def locked_target(self):
        target_lookup_started.set()
        with plugin_lock:
            return Target("mock")

    driver_cls.get_current_target = locked_target

    def reset_while_holding_plugin_lock():
        with plugin_lock:
            resetter_has_plugin_lock.set()
            assert target_lookup_started.wait(timeout=2)
            reset_results.append(registry.reset())

    def set_active():
        try:
            runtime.driver.set_active(driver_cls())
        except Exception as exc:
            setter_errors.append(exc)

    resetter = threading.Thread(
        target=reset_while_holding_plugin_lock,
        daemon=True,
    )
    setter = threading.Thread(
        target=set_active,
        daemon=True,
    )
    resetter.start()
    assert resetter_has_plugin_lock.wait(timeout=2)
    setter.start()
    setter.join(timeout=2)
    resetter.join(timeout=2)

    assert not setter.is_alive()
    assert not resetter.is_alive()
    assert reset_results == [()]
    assert len(setter_errors) == 1
    assert isinstance(setter_errors[0], BackendPluginLifecycleError)
    assert setter_errors[0].field == "registry reset"
    assert runtime.driver.active is runtime.driver.default


def test_w9_unrelated_manifest_does_not_hide_manual_legacy_pair(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    manifest_plugin, _, _ = runtime_plugin(
        "Other",
        target_name="other",
        active=False,
    )
    distribution = distribution_for(
        tmp_path,
        "other",
        plugin=manifest_plugin,
        name="other-backend",
        declaration=triton_plugin(
            "other",
            targets=("other",),
        ),
    )
    manual_plugin, manual_compiler, manual_driver = runtime_plugin("Manual")
    registry = registry_for((distribution,))
    package_name, bridge = load_bridge(monkeypatch, registry)
    bridge.backends["manual"] = bridge.Backend(
        compiler=manual_plugin.compiler_cls,
        driver=manual_plugin.driver_cls,
    )

    compiler = bridge.make_backend(Target("mock"))
    active_driver = load_runtime_driver(
        monkeypatch,
        package_name,
    )._create_driver()

    assert isinstance(compiler, manual_compiler)
    assert isinstance(active_driver, manual_driver)
    assert distribution.entry_points[0].load_calls == 1


def test_w9_rejected_unrelated_manifest_does_not_hide_manual_legacy_pair(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    rejected_plugin, _, _ = runtime_plugin(
        "Future",
        target_name="future",
    )
    distribution = distribution_for(
        tmp_path,
        "future",
        plugin=rejected_plugin,
        name="future-backend",
        declaration=triton_plugin(
            "future",
            targets=("future",),
            triton_version=">=9.0",
        ),
    )
    manual_plugin, manual_compiler, manual_driver = runtime_plugin("Manual")
    registry = registry_for((distribution,))
    package_name, bridge = load_bridge(monkeypatch, registry)
    bridge.backends["manual"] = bridge.Backend(
        compiler=manual_plugin.compiler_cls,
        driver=manual_plugin.driver_cls,
    )

    compiler = bridge.make_backend(Target("mock"))
    active_driver = load_runtime_driver(
        monkeypatch,
        package_name,
    )._create_driver()

    assert isinstance(compiler, manual_compiler)
    assert isinstance(active_driver, manual_driver)
    assert distribution.entry_points[0].load_calls == 0


def test_w9_rejected_same_target_cannot_be_bypassed_by_manual_legacy(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    rejected_plugin, _, _ = runtime_plugin("Future")
    distribution = distribution_for(
        tmp_path,
        "future",
        plugin=rejected_plugin,
        name="future-backend",
        declaration=triton_plugin(
            "future",
            triton_version=">=9.0",
        ),
    )
    manual_plugin, _, _ = runtime_plugin("Manual")
    registry = registry_for((distribution,))
    package_name, bridge = load_bridge(monkeypatch, registry)
    bridge.backends["manual"] = bridge.Backend(
        compiler=manual_plugin.compiler_cls,
        driver=manual_plugin.driver_cls,
    )

    with pytest.raises(BackendPluginCompatibilityError):
        bridge.make_backend(Target("mock"))
    with pytest.raises(BackendPluginCompatibilityError):
        load_runtime_driver(
            monkeypatch,
            package_name,
        )._create_driver()

    assert distribution.entry_points[0].load_calls == 0


def test_w9_unrelated_malformed_manifest_does_not_hide_manual_legacy(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    bad_plugin, _, _ = runtime_plugin("Bad", target_name="bad")
    malformed = triton_plugin("bad", targets=("bad",))
    malformed.pop("backend_protocol")
    distribution = FakeDistribution(
        tmp_path / "bad-backend",
        name="bad-backend",
        manifest=manifest(malformed),
        entry_points=(("bad", bad_plugin),),
    )
    manual_plugin, manual_compiler, manual_driver = runtime_plugin("Manual")
    registry = registry_for((distribution,))
    package_name, bridge = load_bridge(monkeypatch, registry)
    bridge.backends["manual"] = bridge.Backend(
        compiler=manual_plugin.compiler_cls,
        driver=manual_plugin.driver_cls,
    )

    compiler = bridge.make_backend(Target("mock"))
    active_driver = load_runtime_driver(
        monkeypatch,
        package_name,
    )._create_driver()

    assert isinstance(compiler, manual_compiler)
    assert isinstance(active_driver, manual_driver)
    assert distribution.entry_points[0].load_calls == 0


def test_w9_same_entry_point_malformed_manifest_blocks_manual_bypass(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    bad_plugin, _, _ = runtime_plugin("Bad")
    malformed = triton_plugin("bad")
    malformed.pop("backend_protocol")
    distribution = FakeDistribution(
        tmp_path / "bad-backend",
        name="bad-backend",
        manifest=manifest(malformed),
        entry_points=(("bad", bad_plugin),),
    )
    manual_plugin, _, _ = runtime_plugin("Manual")
    registry = registry_for((distribution,))
    package_name, bridge = load_bridge(monkeypatch, registry)
    bridge.backends["bad"] = bridge.Backend(
        compiler=manual_plugin.compiler_cls,
        driver=manual_plugin.driver_cls,
    )

    with pytest.raises(BackendPluginError):
        bridge.make_backend(Target("mock"))
    with pytest.raises(BackendPluginError):
        load_runtime_driver(
            monkeypatch,
            package_name,
        )._create_driver()

    assert distribution.entry_points[0].load_calls == 0


def test_w9_manifest_target_must_match_compiler_support(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    plugin, _, _ = runtime_plugin("Mismatch", supports=False)
    distribution = distribution_for(
        tmp_path,
        "mismatch",
        plugin=plugin,
        name="mismatch-backend",
    )
    registry = registry_for((distribution,))
    _, bridge = load_bridge(monkeypatch, registry)

    with pytest.raises(BackendPluginSelectionError) as caught:
        bridge.make_backend(Target("mock"))

    assert caught.value.field == "compiler_cls.supports_target"
    assert distribution.entry_points[0].load_calls == 1


def test_w9_registry_activation_is_idempotent_and_singleton(tmp_path):
    alpha_plugin, _, _ = runtime_plugin(
        "Alpha",
        target_name="alpha",
    )
    beta_plugin, _, _ = runtime_plugin(
        "Beta",
        target_name="beta",
    )
    alpha = distribution_for(
        tmp_path,
        "alpha",
        plugin=alpha_plugin,
        name="alpha-backend",
        declaration=triton_plugin(
            "alpha",
            targets=("alpha",),
        ),
    )
    beta = distribution_for(
        tmp_path,
        "beta",
        plugin=beta_plugin,
        name="beta-backend",
        declaration=triton_plugin(
            "beta",
            targets=("beta",),
        ),
    )
    registry = registry_for((alpha, beta))
    alpha_decision = registry.select("alpha", environment={})
    beta_decision = registry.select("beta", environment={})

    active = registry.activate(alpha_decision.record_id)

    assert registry.activate(alpha_decision.record_id) is active
    with pytest.raises(BackendPluginConflictError):
        registry.activate(beta_decision.record_id)
    assert registry.inspect(alpha_decision.record_id).state is (
        PluginLifecycleState.ACTIVE
    )
    assert registry.inspect(beta_decision.record_id).state is (
        PluginLifecycleState.SELECTED
    )


def test_w9_active_registry_selection_cannot_switch_before_runtime_reset(
    tmp_path,
):
    low_plugin, _, _ = runtime_plugin("Low")
    high_plugin, _, _ = runtime_plugin("High")
    low = distribution_for(
        tmp_path,
        "low",
        plugin=low_plugin,
        name="low-backend",
        declaration=triton_plugin("low", priority=1),
    )
    high = distribution_for(
        tmp_path,
        "high",
        plugin=high_plugin,
        name="high-backend",
        declaration=triton_plugin("high", priority=2),
    )
    registry = registry_for((low, high))
    high_decision = registry.select("mock", environment={})
    registry.activate(high_decision.record_id)

    with pytest.raises(BackendPluginLifecycleError) as caught:
        registry.select(
            "mock",
            explicit_selector="vendor.low",
            environment={},
        )

    assert caught.value.field == "active selection"
    assert [low.entry_points[0].load_calls,
            high.entry_points[0].load_calls] == [0, 1]
    assert registry.inspect(high_decision.record_id).state is (
        PluginLifecycleState.ACTIVE
    )


def test_w9_activate_detects_preexisting_multiple_active_corruption(
    tmp_path,
):
    alpha_plugin, _, _ = runtime_plugin(
        "Alpha",
        target_name="alpha",
    )
    beta_plugin, _, _ = runtime_plugin(
        "Beta",
        target_name="beta",
    )
    alpha = distribution_for(
        tmp_path,
        "alpha",
        plugin=alpha_plugin,
        name="alpha-backend",
        declaration=triton_plugin("alpha", targets=("alpha",)),
    )
    beta = distribution_for(
        tmp_path,
        "beta",
        plugin=beta_plugin,
        name="beta-backend",
        declaration=triton_plugin("beta", targets=("beta",)),
    )
    registry = registry_for((alpha, beta))
    alpha_decision = registry.select("alpha", environment={})
    beta_decision = registry.select("beta", environment={})
    registry.activate(alpha_decision.record_id)
    beta_record = registry.inspect(beta_decision.record_id)
    registry._replace(
        replace(beta_record, state=PluginLifecycleState.ACTIVE)
    )

    with pytest.raises(BackendPluginConflictError):
        registry.activate(alpha_decision.record_id)


def test_w9_registry_reset_clears_adapter_and_lazy_driver(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    plugin, _, _ = runtime_plugin("Reset")
    shutdown_calls = []
    plugin.shutdown = lambda: shutdown_calls.append("shutdown")
    distribution = distribution_for(
        tmp_path,
        "reset",
        plugin=plugin,
        name="reset-backend",
    )
    registry = registry_for((distribution,))
    package_name, bridge = load_bridge(monkeypatch, registry)
    runtime = load_runtime_driver(monkeypatch, package_name)

    assert runtime.driver.active.get_current_target() == Target("mock")
    assert runtime.driver.default._obj is not None
    assert bridge.backends == {}

    errors = registry.reset()

    assert errors == ()
    assert shutdown_calls == ["shutdown"]
    assert bridge.backends == {}
    assert runtime.driver.active is runtime.driver.default
    assert runtime.driver.default._obj is None
    assert registry.get_selection("mock") is None


def test_w9_reset_blocks_shutdown_reentry(tmp_path):
    plugin, _, _ = runtime_plugin("Reentrant")
    distribution = distribution_for(
        tmp_path,
        "reentrant",
        plugin=plugin,
        name="reentrant-backend",
    )
    registry = registry_for((distribution,))
    plugin.shutdown = lambda: registry.select("mock", environment={})
    registry.select("mock", environment={})

    errors = registry.reset()

    assert len(errors) == 1
    assert isinstance(errors[0], BackendPluginLifecycleError)
    assert errors[0].field == "select"
    assert distribution.entry_points[0].load_calls == 1


def test_w9_runtime_reports_structured_zero_and_multiple_active(
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    registry = registry_for(())
    package_name, bridge = load_bridge(monkeypatch, registry)
    inactive, _, _ = runtime_plugin("Inactive", active=False)
    bridge.backends["inactive"] = bridge.Backend(
        compiler=inactive.compiler_cls,
        driver=inactive.driver_cls,
    )
    runtime = load_runtime_driver(monkeypatch, package_name)

    with pytest.raises(BackendPluginSelectionError) as zero:
        runtime._create_driver()
    assert zero.value.field == "driver_cls.is_active"

    first, _, _ = runtime_plugin("First")
    second, _, _ = runtime_plugin("Second")
    bridge.backends.clear()
    bridge.backends["first"] = bridge.Backend(
        compiler=first.compiler_cls,
        driver=first.driver_cls,
    )
    bridge.backends["second"] = bridge.Backend(
        compiler=second.compiler_cls,
        driver=second.driver_cls,
    )

    with pytest.raises(BackendPluginConflictError) as multiple:
        runtime._create_driver()
    assert multiple.value.field == "driver_cls.is_active"


@pytest.mark.parametrize(
    ("failure", "field"),
    (
        ("probe", "driver_cls.is_active"),
        ("construct", "driver_cls.__init__"),
        ("target", "driver_cls.get_current_target"),
    ),
)
def test_w9_runtime_wraps_driver_lifecycle_failures(
    monkeypatch,
    failure,
    field,
):
    registry = registry_for(())
    package_name, bridge = load_bridge(monkeypatch, registry)
    plugin, _, driver_cls = runtime_plugin("Broken")

    if failure == "probe":
        def fail_probe(cls):
            raise ValueError("probe failure")

        driver_cls.is_active = classmethod(fail_probe)
    elif failure == "construct":
        def fail_construct(self):
            raise ValueError("construct failure")

        driver_cls.__init__ = fail_construct
    else:
        def fail_target(self):
            raise ValueError("target failure")

        driver_cls.get_current_target = fail_target

    bridge.backends["broken"] = bridge.Backend(
        compiler=plugin.compiler_cls,
        driver=driver_cls,
    )
    runtime = load_runtime_driver(monkeypatch, package_name)

    with pytest.raises(BackendPluginLifecycleError) as caught:
        runtime._create_driver()

    assert caught.value.field == field


def test_w9_lazy_driver_initializes_once_across_threads(
    monkeypatch,
):
    registry = registry_for(())
    package_name, _ = load_bridge(monkeypatch, registry)
    runtime = load_runtime_driver(monkeypatch, package_name)
    calls = []
    calls_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def initialize():
        with calls_lock:
            calls.append("initialize")
        time.sleep(0.01)
        return type("RuntimeObject", (), {"value": 42})()

    proxy = runtime.LazyProxy(initialize)

    def read_value():
        barrier.wait()
        return proxy.value

    with ThreadPoolExecutor(max_workers=8) as executor:
        values = tuple(executor.map(lambda _: read_value(), range(8)))

    assert values == (42,) * 8
    assert calls == ["initialize"]


def test_w9_lazy_driver_reset_avoids_plugin_lock_inversion(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    plugin, _, driver_cls = runtime_plugin("LockedLazy")
    distribution = distribution_for(
        tmp_path,
        "locked-lazy",
        plugin=plugin,
        name="locked-lazy-backend",
    )
    registry = registry_for((distribution,))
    package_name, _ = load_bridge(monkeypatch, registry)
    runtime = load_runtime_driver(monkeypatch, package_name)
    plugin_lock = threading.Lock()
    resetter_has_plugin_lock = threading.Event()
    active_probe_started = threading.Event()
    initialized_targets = []
    initializer_errors = []
    reset_results = []

    def locked_active_probe(cls):
        active_probe_started.set()
        with plugin_lock:
            return True

    driver_cls.is_active = classmethod(locked_active_probe)

    def reset_while_holding_plugin_lock():
        with plugin_lock:
            resetter_has_plugin_lock.set()
            assert active_probe_started.wait(timeout=2)
            reset_results.append(registry.reset())

    def initialize_default_driver():
        try:
            initialized_targets.append(
                runtime.driver.default.get_current_target()
            )
        except Exception as exc:
            initializer_errors.append(exc)

    resetter = threading.Thread(
        target=reset_while_holding_plugin_lock,
        daemon=True,
    )
    initializer = threading.Thread(
        target=initialize_default_driver,
        daemon=True,
    )
    resetter.start()
    assert resetter_has_plugin_lock.wait(timeout=2)
    initializer.start()
    initializer.join(timeout=2)
    resetter.join(timeout=2)

    assert not initializer.is_alive()
    assert not resetter.is_alive()
    assert initializer_errors == []
    assert reset_results == [()]
    assert initialized_targets == [Target("mock")]
    assert runtime.driver.default._obj is not None


def test_w9_lazy_proxy_repr_avoids_plugin_reset_lock_inversion(
    monkeypatch,
):
    registry = registry_for(())
    package_name, _ = load_bridge(monkeypatch, registry)
    runtime = load_runtime_driver(monkeypatch, package_name)
    plugin_lock = threading.Lock()
    resetter_has_plugin_lock = threading.Event()
    repr_started = threading.Event()
    rendered = []

    class LockedRepr:

        def __repr__(self):
            repr_started.set()
            with plugin_lock:
                return "LockedRepr()"

    proxy = runtime.LazyProxy(LockedRepr)
    assert isinstance(proxy._initialize_obj(), LockedRepr)

    def reset_while_holding_plugin_lock():
        with plugin_lock:
            resetter_has_plugin_lock.set()
            assert repr_started.wait(timeout=2)
            proxy.reset()

    resetter = threading.Thread(
        target=reset_while_holding_plugin_lock,
        daemon=True,
    )
    renderer = threading.Thread(
        target=lambda: rendered.append(repr(proxy)),
        daemon=True,
    )
    resetter.start()
    assert resetter_has_plugin_lock.wait(timeout=2)
    renderer.start()
    renderer.join(timeout=2)
    resetter.join(timeout=2)

    assert not renderer.is_alive()
    assert not resetter.is_alive()
    assert rendered == ["LockedRepr()"]
    assert proxy._obj is None


def test_w9_concurrent_reset_invalidates_cached_selection_snapshot(
    tmp_path,
    monkeypatch,
):
    plugin, _, _ = runtime_plugin("Snapshot")
    distribution = distribution_for(
        tmp_path,
        "snapshot",
        plugin=plugin,
        name="snapshot-backend",
    )
    registry = registry_for((distribution,))
    registry.select("mock", environment={})
    _, bridge = load_bridge(monkeypatch, registry)
    original_get_selection = registry.get_selection
    snapshot_read = threading.Event()
    reset_complete = threading.Event()

    def paused_get_selection(target):
        decision = original_get_selection(target)
        snapshot_read.set()
        assert reset_complete.wait(timeout=2)
        return decision

    monkeypatch.setattr(
        registry,
        "get_selection",
        paused_get_selection,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            bridge.get_backend,
            Target("mock"),
        )
        assert snapshot_read.wait(timeout=2)
        registry.reset()
        reset_complete.set()
        with pytest.raises(BackendPluginLifecycleError) as caught:
            future.result(timeout=2)

    assert caught.value.field in {
        "registry generation",
        "selected_targets",
    }
    assert bridge.backends == {}


def test_w9_legacy_generation_cleanup_does_not_remove_fresh_same_id_entry(
    tmp_path,
    monkeypatch,
):
    old_plugin, old_compiler, _ = runtime_plugin("OldLegacy")
    distribution = FakeDistribution(
        tmp_path / "legacy-aba",
        name="legacy-aba-backend",
        entry_points=(("legacy-aba", old_plugin),),
    )
    registry = registry_for((distribution,))
    selector = registry.discover()[0].registry_key
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, selector)
    _, bridge = load_bridge(monkeypatch, registry)
    original_prune = bridge._prune_registry_backends
    old_published = threading.Event()
    release_old = threading.Event()
    first_call = True

    def pause_first_prune(current_registry):
        nonlocal first_call
        if first_call:
            first_call = False
            old_published.set()
            assert release_old.wait(timeout=2)
        return original_prune(current_registry)

    monkeypatch.setattr(
        bridge,
        "_prune_registry_backends",
        pause_first_prune,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        stale = executor.submit(bridge.get_backend, Target("mock"))
        assert old_published.wait(timeout=2)
        assert bridge.backends["legacy-aba"].compiler is old_compiler

        assert registry.reset() == ()
        fresh_plugin, fresh_compiler, _ = runtime_plugin("FreshLegacy")
        distribution.entry_points[0].loaded_object = fresh_plugin
        fresh = bridge.get_backend(Target("mock"))
        release_old.set()

        with pytest.raises(BackendPluginLifecycleError) as caught:
            stale.result(timeout=2)

    assert caught.value.field == "registry generation"
    assert fresh.compiler is fresh_compiler
    assert bridge.backends["legacy-aba"].compiler is fresh_compiler
    assert distribution.entry_points[0].load_calls == 2


def test_w9_driver_candidates_fail_if_generation_changes_between_bindings(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    alpha_plugin, _, _ = runtime_plugin("Alpha", target_name="alpha")
    beta_plugin, _, _ = runtime_plugin("Beta", target_name="beta")
    alpha = distribution_for(
        tmp_path,
        "alpha",
        plugin=alpha_plugin,
        name="alpha-backend",
        declaration=triton_plugin("alpha", targets=("alpha",)),
    )
    beta = distribution_for(
        tmp_path,
        "beta",
        plugin=beta_plugin,
        name="beta-backend",
        declaration=triton_plugin("beta", targets=("beta",)),
    )
    registry = registry_for((alpha, beta))
    _, bridge = load_bridge(monkeypatch, registry)
    original_cache_decision = bridge._cache_decision
    first_materialized = threading.Event()
    continue_materialization = threading.Event()
    materialized_count = 0

    def pause_after_first_materialization(decision, **kwargs):
        nonlocal materialized_count
        backend = original_cache_decision(decision, **kwargs)
        materialized_count += 1
        if materialized_count == 1:
            first_materialized.set()
            assert continue_materialization.wait(timeout=2)
        return backend

    monkeypatch.setattr(
        bridge,
        "_cache_decision",
        pause_after_first_materialization,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        candidates = executor.submit(bridge.get_driver_backends)
        assert first_materialized.wait(timeout=2)
        registry.select("alpha", environment={})
        continue_materialization.set()

        with pytest.raises(BackendPluginLifecycleError) as caught:
            candidates.result(timeout=2)

    assert caught.value.field == "registry generation"
    assert [
        alpha.entry_points[0].load_calls,
        beta.entry_points[0].load_calls,
    ] == [1, 1]
    assert bridge.backends == {}


def test_w9_failed_driver_batch_prunes_published_legacy_candidate(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    alpha_plugin, _, _ = runtime_plugin("AlphaLegacy", target_name="alpha")
    beta_plugin, _, _ = runtime_plugin("BetaLegacy", target_name="beta")
    manifest_plugin, _, _ = runtime_plugin(
        "Manifest",
        target_name="alpha",
    )
    alpha = FakeDistribution(
        tmp_path / "alpha-legacy",
        name="alpha-legacy-backend",
        entry_points=(("alpha-legacy", alpha_plugin),),
    )
    beta = FakeDistribution(
        tmp_path / "beta-legacy",
        name="beta-legacy-backend",
        entry_points=(("beta-legacy", beta_plugin),),
    )
    manifest_distribution = distribution_for(
        tmp_path,
        "manifest",
        plugin=manifest_plugin,
        name="manifest-backend",
        declaration=triton_plugin("manifest", targets=("alpha",)),
    )
    registry = registry_for((alpha, beta, manifest_distribution))
    records = {record.entry_point_name: record for record in registry.discover()}
    registry.select(
        "alpha",
        explicit_selector=records["alpha-legacy"].record_id,
        environment={},
    )
    registry.select(
        "beta",
        explicit_selector=records["beta-legacy"].record_id,
        environment={},
    )
    _, bridge = load_bridge(monkeypatch, registry)
    original_cache_decision = bridge._cache_decision
    first_materialized = threading.Event()
    continue_materialization = threading.Event()
    materialized_count = 0

    def pause_after_first_materialization(decision, **kwargs):
        nonlocal materialized_count
        backend = original_cache_decision(decision, **kwargs)
        materialized_count += 1
        if materialized_count == 1:
            first_materialized.set()
            assert continue_materialization.wait(timeout=2)
        return backend

    monkeypatch.setattr(
        bridge,
        "_cache_decision",
        pause_after_first_materialization,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        candidates = executor.submit(bridge.get_driver_backends)
        assert first_materialized.wait(timeout=2)
        assert "alpha-legacy" in bridge.backends
        registry.select(
            "alpha",
            explicit_selector="vendor.manifest",
            environment={},
        )
        continue_materialization.set()

        with pytest.raises(BackendPluginLifecycleError) as caught:
            candidates.result(timeout=2)

    assert caught.value.field == "registry generation"
    assert "alpha-legacy" not in bridge.backends
    assert "beta-legacy" not in bridge.backends
    assert [
        alpha.entry_points[0].load_calls,
        beta.entry_points[0].load_calls,
        manifest_distribution.entry_points[0].load_calls,
    ] == [1, 1, 1]


def test_w9_stale_legacy_decision_failure_prunes_previous_projection(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(BACKEND_SELECTOR_ENV, raising=False)
    legacy_plugin, _, _ = runtime_plugin("Legacy")
    legacy = FakeDistribution(
        tmp_path / "legacy",
        name="legacy-backend",
        entry_points=(("legacy", legacy_plugin),),
    )
    manifest_plugin, _, _ = runtime_plugin("Manifest")
    manifest_distribution = distribution_for(
        tmp_path,
        "manifest",
        plugin=manifest_plugin,
        name="manifest-backend",
    )
    registry = registry_for((legacy, manifest_distribution))
    records = {record.entry_point_name: record for record in registry.discover()}
    registry.select(
        "mock",
        explicit_selector=records["legacy"].record_id,
        environment={},
    )
    _, bridge = load_bridge(monkeypatch, registry)
    bridge.get_backend(Target("mock"))
    assert "legacy" in bridge.backends

    original_get_selection = registry.get_selection
    stale_read = threading.Event()
    winner_switched = threading.Event()

    def paused_get_selection(target):
        decision = original_get_selection(target)
        stale_read.set()
        assert winner_switched.wait(timeout=2)
        return decision

    monkeypatch.setattr(registry, "get_selection", paused_get_selection)
    with ThreadPoolExecutor(max_workers=1) as executor:
        stale = executor.submit(bridge.get_backend, Target("mock"))
        assert stale_read.wait(timeout=2)
        registry.select(
            "mock",
            explicit_selector="vendor.manifest",
            environment={},
        )
        winner_switched.set()

        with pytest.raises(BackendPluginLifecycleError) as caught:
            stale.result(timeout=2)

    assert caught.value.field == "selected_targets"
    assert "legacy" not in bridge.backends
    assert [
        legacy.entry_points[0].load_calls,
        manifest_distribution.entry_points[0].load_calls,
    ] == [1, 1]


def test_w9_reset_shutdown_avoids_plugin_registry_lock_inversion(
    tmp_path,
):
    plugin, _, _ = runtime_plugin("Locked")
    distribution = distribution_for(
        tmp_path,
        "locked",
        plugin=plugin,
        name="locked-backend",
    )
    registry = registry_for((distribution,))
    plugin_lock = threading.Lock()
    worker_has_lock = threading.Event()
    shutdown_started = threading.Event()
    reentry_errors = []

    def shutdown():
        shutdown_started.set()
        with plugin_lock:
            pass

    plugin.shutdown = shutdown
    registry.select("mock", environment={})

    def hold_plugin_lock_and_reenter():
        with plugin_lock:
            worker_has_lock.set()
            assert shutdown_started.wait(timeout=2)
            try:
                registry.select("mock", environment={})
            except BackendPluginLifecycleError as exc:
                reentry_errors.append(exc)

    reset_results = []
    worker = threading.Thread(
        target=hold_plugin_lock_and_reenter,
        daemon=True,
    )
    resetter = threading.Thread(
        target=lambda: reset_results.append(registry.reset()),
        daemon=True,
    )
    worker.start()
    assert worker_has_lock.wait(timeout=2)
    resetter.start()
    worker.join(timeout=2)
    resetter.join(timeout=2)

    assert not worker.is_alive()
    assert not resetter.is_alive()
    assert len(reentry_errors) == 1
    assert reentry_errors[0].field == "select"
    assert reset_results == [()]


def test_compiler_make_backend_delegates_to_registry_adapter():
    path = TRITON_ROOT / "compiler" / "compiler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_backend"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    calls = []

    def selected_backend(target):
        calls.append(target)
        return "selected-compiler"

    namespace = {"_make_selected_backend": selected_backend}
    exec(compile(isolated, str(path), "exec"), namespace)
    target = Target("mock")

    assert namespace["make_backend"](target) == "selected-compiler"
    assert calls == [target]
