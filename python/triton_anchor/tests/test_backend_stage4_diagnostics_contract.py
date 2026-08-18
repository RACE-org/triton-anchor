"""Characterization contracts for the Stage 4 Registry diagnostics seam."""

import ast
import json
import threading
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from triton_anchor.backends import (
    BackendPluginRegistry,
    PluginLifecycleState,
    PluginSource,
)
from triton_anchor.tests.test_backend_registry import (
    DummyCompiler,
    DummyDriver,
    FakeDistribution,
    RuntimePlugin,
    SUPPORTED_TAG,
    core_environment,
    make_registry,
    manifest,
    plugin_record,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKENDS_ROOT = REPO_ROOT / "python" / "triton_anchor" / "backends"


class CountingDiagnosticsPlugin(RuntimePlugin):
    def __init__(self, value):
        self.value = value
        self.diagnostic_calls = 0

    def diagnostics(self):
        self.diagnostic_calls += 1
        return self.value


class FailingDiagnosticsPlugin(RuntimePlugin):
    def __init__(self):
        self.diagnostic_calls = 0

    def diagnostics(self):
        self.diagnostic_calls += 1
        raise RuntimeError("diagnostics failed")


class RaisingDiagnosticsAttributePlugin(RuntimePlugin):
    def __init__(self):
        self.diagnostics_reads = 0

    @property
    def diagnostics(self):
        self.diagnostics_reads += 1
        raise RuntimeError("diagnostics attribute failed")


class NonCallableDiagnosticsPlugin(RuntimePlugin):
    diagnostics = {"ignored": True}


class OrderedDiagnosticsPlugin(RuntimePlugin):
    def __init__(self, name, events, *, fail=False):
        self.name = name
        self.events = events
        self.fail = fail
        self.diagnostic_calls = 0

    def diagnostics(self):
        self.diagnostic_calls += 1
        self.events.append(self.name)
        if self.fail:
            raise RuntimeError(self.name + " diagnostics failed")
        return {"name": self.name}


class LegacyDiagnosticsPlugin(CountingDiagnosticsPlugin):
    def __init__(self):
        super().__init__({"legacy": True})
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


class BlockingDiagnosticsPlugin(RuntimePlugin):
    def __init__(self, entered, release):
        self.entered = entered
        self.release = release
        self.diagnostic_calls = 0

    def diagnostics(self):
        self.diagnostic_calls += 1
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("diagnostics test barrier timed out")
        return {"released": True}


class ReadOnlyReentrantDiagnosticsPlugin(RuntimePlugin):
    def __init__(self):
        self.registry = None
        self.record_id = None
        self.diagnostic_calls = 0

    def diagnostics(self):
        self.diagnostic_calls += 1
        record = self.registry.inspect(self.record_id)
        return {
            "generation": self.registry.generation,
            "record_id": record.record_id,
            "state": record.state.value,
            "selection_absent": self.registry.get_selection("missing") is None,
        }


class BrokenEntryPointDistribution:
    def __init__(self, name, marker):
        self.metadata = {"Name": name}
        self.version = "1.0.0"
        self.marker = marker

    @property
    def entry_points(self):
        raise RuntimeError(self.marker)


def _manifest_distribution(
    tmp_path,
    plugin,
    *,
    entry_point="mock",
    name="vendor-backend",
    plugin_id=None,
    targets=None,
):
    declaration = plugin_record(
        entry_point,
        plugin_id=plugin_id,
        **({"targets": list(targets)} if targets is not None else {}),
    )
    return FakeDistribution(
        tmp_path / name,
        name=name,
        manifest=manifest(declaration),
        entry_points=((entry_point, plugin),),
    )


def _registered(registry):
    discovered = registry.discover()[0]
    return registry.register(discovered.record_id)


def _start_daemon_call(function, *, name):
    results = []
    errors = []
    complete = threading.Event()

    def invoke():
        try:
            results.append(function())
        except Exception as exc:
            errors.append(exc)
        finally:
            complete.set()

    thread = threading.Thread(target=invoke, name=name, daemon=True)
    thread.start()
    return thread, complete, results, errors


def test_repeated_success_diagnostics_preserve_json_state_generation_and_counts(
    tmp_path,
):
    plugin = CountingDiagnosticsPlugin({"healthy": True})
    distribution = _manifest_distribution(tmp_path, plugin)
    registry = make_registry((distribution,))
    registered = _registered(registry)
    generation = registry.generation

    first = registry.diagnostics(registered.record_id)
    second = registry.diagnostics(registered.registry_key)

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert plugin.diagnostic_calls == 2
    assert distribution.entry_points[0].load_calls == 1
    current = registry.inspect(registered.record_id)
    assert current.state is PluginLifecycleState.REGISTERED
    assert current.errors == ()
    assert registry._state.registry_errors_snapshot() == ()
    assert registry.generation == generation


def test_repeated_failed_diagnostics_preserve_json_state_generation_and_counts(
    tmp_path,
):
    plugin = FailingDiagnosticsPlugin()
    distribution = _manifest_distribution(tmp_path, plugin)
    registry = make_registry((distribution,))
    registered = _registered(registry)
    generation = registry.generation

    first = registry.diagnostics(registered.record_id)
    second = registry.diagnostics(registered.registry_key)

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    error = first["plugin_diagnostics"]["error"]
    assert error["code"] == "backend_plugin_lifecycle_error"
    assert error["field"] == "diagnostics"
    assert error["actual"] == "<error: diagnostics failed>"
    assert plugin.diagnostic_calls == 2
    assert distribution.entry_points[0].load_calls == 1
    current = registry.inspect(registered.record_id)
    assert current.state is PluginLifecycleState.REGISTERED
    assert current.errors == ()
    assert registry._state.registry_errors_snapshot() == ()
    assert registry.generation == generation


def test_diagnostics_attribute_failure_is_stable_and_does_not_reject(tmp_path):
    plugin = RaisingDiagnosticsAttributePlugin()
    distribution = _manifest_distribution(tmp_path, plugin)
    registry = make_registry((distribution,))
    registered = _registered(registry)
    generation = registry.generation

    first = registry.diagnostics(registered.record_id)
    second = registry.diagnostics(registered.record_id)

    assert first == second
    error = first["plugin_diagnostics"]["error"]
    assert error["field"] == "diagnostics"
    assert error["expected"] == "a callable hook or no hook"
    assert error["actual"] == "<error: diagnostics attribute failed>"
    assert plugin.diagnostics_reads == 2
    current = registry.inspect(registered.record_id)
    assert current.state is PluginLifecycleState.REGISTERED
    assert current.errors == ()
    assert registry._state.registry_errors_snapshot() == ()
    assert registry.generation == generation


def test_noncallable_diagnostics_attribute_is_treated_as_no_hook(tmp_path):
    plugin = NonCallableDiagnosticsPlugin()
    distribution = _manifest_distribution(tmp_path, plugin)
    registry = make_registry((distribution,))
    registered = _registered(registry)
    generation = registry.generation

    result = registry.diagnostics(registered.record_id)

    assert result["plugin_diagnostics"] is None
    assert registry.inspect(registered.record_id).errors == ()
    assert registry.inspect(registered.record_id).state is (
        PluginLifecycleState.REGISTERED
    )
    assert distribution.entry_points[0].load_calls == 1
    assert registry.generation == generation


def test_aggregate_preserves_plugin_callback_and_selection_target_order_when_one_hook_fails(
    tmp_path,
):
    events = []
    alpha_plugin = OrderedDiagnosticsPlugin("alpha", events, fail=True)
    zulu_plugin = OrderedDiagnosticsPlugin("zulu", events)
    alpha = _manifest_distribution(
        tmp_path,
        alpha_plugin,
        entry_point="alpha",
        name="alpha-backend",
        plugin_id="vendor.alpha",
        targets=("z-target",),
    )
    zulu = _manifest_distribution(
        tmp_path,
        zulu_plugin,
        entry_point="zulu",
        name="zulu-backend",
        plugin_id="vendor.zulu",
        targets=("a-target",),
    )
    registry = make_registry((zulu, alpha))

    registry.select(
        "z-target",
        explicit_selector="vendor.alpha",
        environment={},
    )
    registry.select(
        "a-target",
        explicit_selector="vendor.zulu",
        environment={},
    )
    result = registry.diagnostics()

    assert events == ["alpha", "zulu"]
    assert [item["plugin_id"] for item in result["plugins"]] == [
        "vendor.alpha",
        "vendor.zulu",
    ]
    assert result["plugins"][0]["plugin_diagnostics"]["error"][
        "field"
    ] == "diagnostics"
    assert result["plugins"][1]["plugin_diagnostics"] == {"name": "zulu"}
    assert list(result["selections"]) == ["a-target", "z-target"]
    assert alpha_plugin.diagnostic_calls == 1
    assert zulu_plugin.diagnostic_calls == 1
    assert [alpha.entry_points[0].load_calls, zulu.entry_points[0].load_calls] == [
        1,
        1,
    ]
    assert all(record.errors == () for record in registry.list())


def test_structural_fake_mapping_result_is_shallow_copied(tmp_path):
    nested = []
    source = {"healthy": True, "nested": nested}
    calls = []

    def diagnostics():
        calls.append(True)
        return MappingProxyType(source)

    plugin = SimpleNamespace(
        compiler_cls=DummyCompiler,
        driver_cls=DummyDriver,
        diagnostics=diagnostics,
    )
    distribution = _manifest_distribution(tmp_path, plugin)
    registry = make_registry((distribution,))
    registered = _registered(registry)

    result = registry.diagnostics(registered.record_id)
    payload = result["plugin_diagnostics"]
    source["healthy"] = False
    source["later"] = True
    nested.append("shared")

    assert calls == [True]
    assert isinstance(payload, dict)
    assert payload is not source
    assert payload["healthy"] is True
    assert "later" not in payload
    assert payload["nested"] is nested
    assert payload["nested"] == ["shared"]
    assert distribution.entry_points[0].load_calls == 1


def test_non_mapping_diagnostics_result_is_passed_through_by_identity(tmp_path):
    sentinel = object()
    plugin = CountingDiagnosticsPlugin(sentinel)
    distribution = _manifest_distribution(tmp_path, plugin)
    registry = make_registry((distribution,))
    registered = _registered(registry)

    result = registry.diagnostics(registered.record_id)

    assert result["plugin_diagnostics"] is sentinel
    assert plugin.diagnostic_calls == 1
    assert distribution.entry_points[0].load_calls == 1


def test_loaded_legacy_plugin_diagnostics_preserve_legacy_lifecycle(tmp_path):
    plugin = LegacyDiagnosticsPlugin()
    distribution = FakeDistribution(
        tmp_path / "legacy-backend",
        name="legacy-backend",
        manifest=None,
        entry_points=(("legacy", plugin),),
    )
    registry = make_registry((distribution,))
    registered = _registered(registry)
    generation = registry.generation

    first = registry.diagnostics(registered.record_id)
    second = registry.diagnostics(registered.registry_key)

    assert registered.source is PluginSource.LEGACY
    assert registered.state is PluginLifecycleState.REGISTERED
    assert first == second
    assert first["plugin_diagnostics"] == {"legacy": True}
    assert plugin.diagnostic_calls == 2
    assert distribution.entry_points[0].load_calls == 1
    assert registry.generation == generation
    assert registry.reset() == ()
    assert plugin.shutdown_calls == 0


def test_aggregate_reports_fatal_conflict_without_loading_or_rejecting(tmp_path):
    alpha = _manifest_distribution(
        tmp_path,
        RuntimePlugin(),
        entry_point="alpha",
        name="alpha-backend",
        plugin_id="vendor.duplicate",
        targets=("alpha-target",),
    )
    zulu = _manifest_distribution(
        tmp_path,
        RuntimePlugin(),
        entry_point="zulu",
        name="zulu-backend",
        plugin_id="vendor.duplicate",
        targets=("zulu-target",),
    )
    registry = make_registry((zulu, alpha))

    result = registry.diagnostics()

    assert [item["entry_point"] for item in result["plugins"]] == [
        "alpha",
        "zulu",
    ]
    assert result["conflicts"]["has_fatal"] is True
    assert result["conflicts"]["conflicts"][0]["kind"] == (
        "duplicate_plugin_id"
    )
    assert all(
        record.state is PluginLifecycleState.DISCOVERED
        for record in registry.list()
    )
    assert [alpha.entry_points[0].load_calls, zulu.entry_points[0].load_calls] == [
        0,
        0,
    ]


def test_aggregate_registry_errors_preserve_append_order_and_repeat_stably():
    zulu = BrokenEntryPointDistribution("zulu-backend", "zulu failure")
    alpha = BrokenEntryPointDistribution("alpha-backend", "alpha failure")
    provider_calls = []
    registry = BackendPluginRegistry(
        distribution_provider=lambda: provider_calls.append(True) or (zulu, alpha),
        environment_provider=core_environment,
        supported_tags=(SUPPORTED_TAG,),
    )

    first = registry.diagnostics()
    second = registry.diagnostics()

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert [error["actual"] for error in first["registry_errors"]] == [
        "<error: zulu failure>",
        "<error: alpha failure>",
    ]
    assert first["plugins"] == []
    assert provider_calls == [True]


def test_cross_thread_reset_waits_for_diagnostics_callback_rlock(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    reset_started = threading.Event()
    plugin = BlockingDiagnosticsPlugin(entered, release)
    distribution = _manifest_distribution(tmp_path, plugin)
    registry = make_registry((distribution,))
    registered = _registered(registry)

    diagnostic_thread, diagnostic_done, diagnostic_results, diagnostic_errors = (
        _start_daemon_call(
            lambda: registry.diagnostics(registered.record_id),
            name="stage4-blocking-diagnostics",
        )
    )
    assert entered.wait(timeout=5)

    def reset_registry():
        reset_started.set()
        return registry.reset()

    reset_thread, reset_done, reset_results, reset_errors = _start_daemon_call(
        reset_registry,
        name="stage4-reset-after-diagnostics",
    )
    try:
        assert reset_started.wait(timeout=5)
        assert not reset_done.wait(timeout=0.25)
    finally:
        release.set()

    assert diagnostic_done.wait(timeout=5)
    assert reset_done.wait(timeout=5)
    diagnostic_thread.join(timeout=1)
    reset_thread.join(timeout=1)

    assert not diagnostic_thread.is_alive()
    assert not reset_thread.is_alive()
    assert diagnostic_errors == []
    assert reset_errors == []
    assert diagnostic_results[0]["plugin_diagnostics"] == {"released": True}
    assert reset_results == [()]
    assert plugin.diagnostic_calls == 1
    assert distribution.entry_points[0].load_calls == 1


def test_diagnostics_callback_allows_same_thread_read_only_registry_reentry(
    tmp_path,
):
    plugin = ReadOnlyReentrantDiagnosticsPlugin()
    distribution = _manifest_distribution(tmp_path, plugin)
    registry = make_registry((distribution,))
    registered = _registered(registry)
    plugin.registry = registry
    plugin.record_id = registered.record_id
    generation = registry.generation

    result = registry.diagnostics(registered.record_id)

    assert result["plugin_diagnostics"] == {
        "generation": generation,
        "record_id": registered.record_id,
        "state": PluginLifecycleState.REGISTERED.value,
        "selection_absent": True,
    }
    assert plugin.diagnostic_calls == 1
    assert registry.generation == generation
    assert registry.inspect(registered.record_id).errors == ()


def test_registry_internal_modules_do_not_reverse_import_registry_facade():
    offenders = []
    paths = sorted(BACKENDS_ROOT.glob("_registry_*.py"))
    assert paths

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name in {"registry", "triton_anchor.backends.registry"}
                    for alias in node.names
                ):
                    offenders.append((path.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports_registry_name = any(
                    alias.name == "registry" for alias in node.names
                )
                if (
                    (node.level and module == "registry")
                    or module == "triton_anchor.backends.registry"
                    or (
                        imports_registry_name
                        and module in {"", "triton_anchor.backends"}
                    )
                ):
                    offenders.append((path.name, node.lineno))

    assert offenders == []
