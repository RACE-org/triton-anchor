"""Failure and reset contracts for behavior-preserving Registry refactors."""

import json
import threading
from dataclasses import replace

import pytest

from triton_anchor.backends import (
    BackendPluginCompatibilityError,
    BackendPluginLifecycleError,
    BackendPluginLoadError,
    BackendPluginProtocolError,
    BackendPluginRegistry,
    PluginCompatibilityStatus,
    PluginLifecycleState,
    SelectionMethod,
)
from triton_anchor.tests.test_backend_registry import (
    DummyCompiler,
    DummyDriver,
    FakeDistribution,
    HookPlugin,
    RuntimePlugin,
    SUPPORTED_TAG,
    VALID_WHEEL,
    core_environment,
    make_registry,
    manifest,
    plugin_record,
)


ERROR_JSON_FIELDS = {
    "code",
    "message",
    "plugin_id",
    "entry_point",
    "detail",
    "field",
    "expected",
    "actual",
    "remediation",
}

RECORD_JSON_FIELDS = {
    "record_id",
    "registry_key",
    "plugin_id",
    "entry_point",
    "entry_point_value",
    "distribution_name",
    "distribution_version",
    "source",
    "state",
    "compatibility_status",
    "manifest",
    "compatibility_report",
    "capability_report",
    "protocol_field_diagnostics",
    "errors",
    "loaded",
    "registered",
    "compiler_cls",
    "driver_cls",
    "initialized",
    "shutdown_called",
    "selected_targets",
}


class RaisingEntryPoint:
    group = "triton.backends"

    def __init__(self, name, error):
        self.name = name
        self.value = "vendor_backend." + name
        self.error = error
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        raise self.error


class BlockingEntryPoint:
    group = "triton.backends"

    def __init__(self, name, plugin, entered, release):
        self.name = name
        self.value = "vendor_backend." + name
        self.plugin = plugin
        self.entered = entered
        self.release = release
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("load test barrier timed out")
        return self.plugin


class FailingInitializePlugin(RuntimePlugin):
    def __init__(self):
        self.initialize_calls = 0
        self.shutdown_calls = 0

    def initialize(self, context):
        self.initialize_calls += 1
        raise RuntimeError("initialize failed")

    def shutdown(self):
        self.shutdown_calls += 1


class FailingDiagnosticsPlugin(RuntimePlugin):
    def __init__(self):
        self.diagnostic_calls = 0

    def diagnostics(self):
        self.diagnostic_calls += 1
        raise RuntimeError("diagnostics failed")


class FailingShutdownPlugin(RuntimePlugin):
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1
        raise RuntimeError("shutdown failed")


class BlockingInitializePlugin(RuntimePlugin):
    def __init__(self, entered, release):
        self.entered = entered
        self.release = release
        self.initialize_calls = 0
        self.shutdown_calls = 0

    def initialize(self, context):
        self.initialize_calls += 1
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("initialize test barrier timed out")

    def shutdown(self):
        self.shutdown_calls += 1


def distribution_for_plugin(tmp_path, plugin):
    return FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record()),
        entry_points=(("mock", plugin),),
    )


def assert_structured_error(payload, *, code, field):
    assert set(payload) == ERROR_JSON_FIELDS
    assert payload["code"] == code
    assert payload["field"] == field
    assert payload["plugin_id"] == "vendor.mock"
    assert payload["entry_point"] == "mock"
    assert isinstance(payload["message"], str) and payload["message"]
    assert payload["expected"]
    assert payload["actual"]
    assert payload["remediation"]


def start_daemon_call(function, *, name):
    """Run one potentially blocking call without letting a deadlock hang pytest."""
    results = []
    errors = []
    completed = threading.Event()

    def invoke():
        try:
            results.append(function())
        except Exception as exc:
            errors.append(exc)
        finally:
            completed.set()

    thread = threading.Thread(target=invoke, name=name, daemon=True)
    thread.start()
    return thread, completed, results, errors


def test_entry_point_load_failure_is_terminal_and_structured(tmp_path):
    distribution = distribution_for_plugin(tmp_path, RuntimePlugin())
    load_error = RuntimeError("entry point failed")
    entry_point = RaisingEntryPoint("mock", load_error)
    distribution.entry_points = [entry_point]
    registry = make_registry((distribution,))
    discovered = registry.discover()[0]

    with pytest.raises(BackendPluginLoadError) as caught:
        registry.load(discovered.registry_key)

    rejected = registry.inspect(discovered.record_id)
    payload = caught.value.to_dict()
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert rejected.state is PluginLifecycleState.REJECTED
    assert (
        rejected.compatibility_status
        is PluginCompatibilityStatus.COMPATIBLE
    )
    assert rejected.to_dict()["errors"] == [payload]
    assert_structured_error(
        payload,
        code="backend_plugin_load_error",
        field="entry_point.load",
    )

    with pytest.raises(BackendPluginLoadError):
        registry.load(discovered.record_id)
    assert entry_point.load_calls == 1


def test_initialize_failure_rejects_and_runs_best_effort_shutdown(tmp_path):
    plugin = FailingInitializePlugin()
    distribution = distribution_for_plugin(tmp_path, plugin)
    registry = make_registry((distribution,))
    discovered = registry.discover()[0]

    with pytest.raises(BackendPluginLifecycleError) as caught:
        registry.register(discovered.registry_key)

    rejected = registry.inspect(discovered.record_id)
    payload = caught.value.to_dict()
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert rejected.state is PluginLifecycleState.REJECTED
    assert rejected.initialized is False
    assert rejected.shutdown_called is True
    assert plugin.initialize_calls == 1
    assert plugin.shutdown_calls == 1
    assert distribution.entry_points[0].load_calls == 1
    assert_structured_error(
        payload,
        code="backend_plugin_lifecycle_error",
        field="initialize",
    )

    with pytest.raises(BackendPluginLifecycleError):
        registry.register(discovered.record_id)
    assert plugin.initialize_calls == 1
    assert plugin.shutdown_calls == 1


def test_noncallable_initialize_is_structured_and_rejected(tmp_path):
    plugin = HookPlugin()
    plugin.initialize = object()
    distribution = distribution_for_plugin(tmp_path, plugin)
    registry = make_registry((distribution,))
    discovered = registry.discover()[0]

    with pytest.raises(BackendPluginLifecycleError) as caught:
        registry.register(discovered.registry_key)

    rejected = registry.inspect(discovered.record_id)
    payload = caught.value.to_dict()
    assert rejected.state is PluginLifecycleState.REJECTED
    assert (
        rejected.compatibility_status
        is PluginCompatibilityStatus.COMPATIBLE
    )
    assert rejected.initialized is False
    assert rejected.shutdown_called is False
    assert rejected.to_dict()["errors"] == [payload]
    assert plugin.initialize_calls == []
    assert plugin.shutdown_calls == 0
    assert distribution.entry_points[0].load_calls == 1
    assert_structured_error(
        payload,
        code="backend_plugin_lifecycle_error",
        field="initialize",
    )
    assert payload["actual"] == "object"


def test_diagnostics_failure_is_reported_without_rejecting_plugin(tmp_path):
    plugin = FailingDiagnosticsPlugin()
    distribution = distribution_for_plugin(tmp_path, plugin)
    registry = make_registry((distribution,))
    registered = registry.register(registry.discover()[0].registry_key)

    result = registry.diagnostics(registered.record_id)

    payload = result["plugin_diagnostics"]["error"]
    assert_structured_error(
        payload,
        code="backend_plugin_lifecycle_error",
        field="diagnostics",
    )
    assert registry.inspect(registered.record_id).state is (
        PluginLifecycleState.REGISTERED
    )
    assert plugin.diagnostic_calls == 1
    assert distribution.entry_points[0].load_calls == 1


def test_shutdown_failure_is_returned_after_registry_state_is_cleared(
    tmp_path,
):
    plugin = FailingShutdownPlugin()
    distribution = distribution_for_plugin(tmp_path, plugin)
    registry = make_registry((distribution,))
    registry.register(registry.discover()[0].registry_key)

    errors = registry.reset()

    assert len(errors) == 1
    assert isinstance(errors[0], BackendPluginLifecycleError)
    assert_structured_error(
        errors[0].to_dict(),
        code="backend_plugin_lifecycle_error",
        field="shutdown",
    )
    assert plugin.shutdown_calls == 1
    assert registry.get_selection("mock") is None
    assert registry.discover()[0].state is PluginLifecycleState.DISCOVERED
    assert registry.reset() == ()
    assert plugin.shutdown_calls == 1


def test_reset_allows_fresh_rediscovery_and_reselection(tmp_path):
    plugin = HookPlugin()
    distribution = distribution_for_plugin(tmp_path, plugin)
    provider_calls = 0

    def distribution_provider():
        nonlocal provider_calls
        provider_calls += 1
        return (distribution,)

    registry = BackendPluginRegistry(
        distribution_provider=distribution_provider,
        environment_provider=core_environment,
        supported_tags=(SUPPORTED_TAG,),
        preflight_profile="full",
    )

    first = registry.select("mock", environment={})
    assert first.method is SelectionMethod.SOLE_CANDIDATE
    assert registry.reset() == ()
    assert registry.get_selection("mock") is None

    rediscovered = registry.discover()[0]
    assert rediscovered.state is PluginLifecycleState.DISCOVERED
    second = registry.select("mock", environment={})

    assert second.method is SelectionMethod.SOLE_CANDIDATE
    assert second.record.state is PluginLifecycleState.SELECTED
    assert provider_calls == 2
    assert distribution.entry_points[0].load_calls == 2
    assert len(plugin.initialize_calls) == 2
    assert plugin.shutdown_calls == 1
    assert registry.reset() == ()
    assert plugin.shutdown_calls == 2


def test_registry_instances_isolate_state_load_and_reset(tmp_path):
    first_plugin = HookPlugin()
    second_plugin = HookPlugin()
    first_distribution = distribution_for_plugin(
        tmp_path / "first",
        first_plugin,
    )
    second_distribution = distribution_for_plugin(
        tmp_path / "second",
        second_plugin,
    )
    first_registry = make_registry((first_distribution,))
    second_registry = make_registry((second_distribution,))

    first_discovered = first_registry.discover()[0]
    second_discovered = second_registry.discover()[0]
    assert first_discovered.record_id == second_discovered.record_id
    assert first_discovered.registry_key == second_discovered.registry_key

    first_registry.select("mock", environment={})

    assert first_registry.inspect(first_discovered.record_id).state is (
        PluginLifecycleState.SELECTED
    )
    assert second_registry.inspect(second_discovered.record_id).state is (
        PluginLifecycleState.DISCOVERED
    )
    assert first_distribution.entry_points[0].load_calls == 1
    assert second_distribution.entry_points[0].load_calls == 0
    assert len(first_plugin.initialize_calls) == 1
    assert second_plugin.initialize_calls == []

    second_registry.select("mock", environment={})
    second_generation = second_registry.generation
    second_selection = second_registry.get_selection("mock").to_dict()
    second_record = second_registry.inspect(
        second_discovered.record_id
    ).to_dict()

    assert first_registry.reset() == ()

    assert first_registry.get_selection("mock") is None
    assert first_plugin.shutdown_calls == 1
    assert second_registry.generation == second_generation
    assert second_registry.get_selection("mock").to_dict() == second_selection
    assert (
        second_registry.inspect(second_discovered.record_id).to_dict()
        == second_record
    )
    assert second_distribution.entry_points[0].load_calls == 1
    assert len(second_plugin.initialize_calls) == 1
    assert second_plugin.shutdown_calls == 0

    rediscovered = first_registry.discover()[0]
    assert rediscovered.state is PluginLifecycleState.DISCOVERED
    assert first_distribution.entry_points[0].load_calls == 1
    assert len(first_plugin.initialize_calls) == 1
    assert second_registry.inspect(second_discovered.record_id).state is (
        PluginLifecycleState.SELECTED
    )

    assert first_registry.reset() == ()
    assert second_registry.reset() == ()
    assert second_plugin.shutdown_calls == 1


def test_two_plugins_keep_selection_capabilities_and_lifecycle_isolated(
    tmp_path,
):
    alpha_plugin = HookPlugin()
    beta_plugin = HookPlugin()
    alpha_distribution = FakeDistribution(
        tmp_path / "alpha",
        name="alpha-backend",
        manifest=manifest(
            plugin_record(
                "alpha",
                plugin_id="vendor.alpha",
                targets=["alpha"],
                capabilities=["feature.alpha"],
            )
        ),
        entry_points=(("alpha", alpha_plugin),),
    )
    beta_distribution = FakeDistribution(
        tmp_path / "beta",
        name="beta-backend",
        manifest=manifest(
            plugin_record(
                "beta",
                plugin_id="vendor.beta",
                targets=["beta"],
                capabilities=["feature.beta"],
            )
        ),
        entry_points=(("beta", beta_plugin),),
    )
    registry = make_registry((beta_distribution, alpha_distribution))

    records = registry.discover()
    assert [record.plugin_id for record in records] == [
        "vendor.alpha",
        "vendor.beta",
    ]
    assert registry.conflicts().ok
    assert [
        alpha_distribution.entry_points[0].load_calls,
        beta_distribution.entry_points[0].load_calls,
    ] == [0, 0]

    alpha = registry.select(
        "alpha",
        explicit_selector="vendor.alpha",
        kernel_required_capabilities=("feature.alpha",),
        environment={},
    )

    assert alpha.plugin_id == "vendor.alpha"
    assert registry.inspect("vendor.alpha").state is PluginLifecycleState.SELECTED
    assert registry.inspect("vendor.beta").state is PluginLifecycleState.VALIDATED
    assert [
        alpha_distribution.entry_points[0].load_calls,
        beta_distribution.entry_points[0].load_calls,
    ] == [1, 0]
    assert len(alpha_plugin.initialize_calls) == 1
    assert alpha_plugin.initialize_calls[0]["record_id"] == alpha.record_id
    assert beta_plugin.initialize_calls == []

    beta = registry.select(
        "beta",
        explicit_selector="vendor.beta",
        kernel_required_capabilities=("feature.beta",),
        environment={},
    )

    assert beta.plugin_id == "vendor.beta"
    assert registry.inspect("vendor.alpha").selected_targets == ("alpha",)
    assert registry.inspect("vendor.beta").selected_targets == ("beta",)
    assert [
        alpha_distribution.entry_points[0].load_calls,
        beta_distribution.entry_points[0].load_calls,
    ] == [1, 1]
    assert len(alpha_plugin.initialize_calls) == 1
    assert len(beta_plugin.initialize_calls) == 1
    assert alpha_plugin.initialize_calls[0]["record_id"] == alpha.record_id
    assert beta_plugin.initialize_calls[0]["record_id"] == beta.record_id

    assert registry.diagnostics("vendor.alpha")["plugin_diagnostics"] == {
        "healthy": True
    }
    assert alpha_plugin.diagnostic_calls == 1
    assert beta_plugin.diagnostic_calls == 0
    assert registry.diagnostics("vendor.beta")["plugin_diagnostics"] == {
        "healthy": True
    }
    assert alpha_plugin.diagnostic_calls == 1
    assert beta_plugin.diagnostic_calls == 1

    assert registry.reset() == ()
    assert alpha_plugin.shutdown_calls == 1
    assert beta_plugin.shutdown_calls == 1


def test_reset_waits_for_load_then_leaves_no_inflight_state(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    distribution = distribution_for_plugin(tmp_path, RuntimePlugin())
    entry_point = BlockingEntryPoint(
        "mock",
        RuntimePlugin(),
        entered,
        release,
    )
    distribution.entry_points = [entry_point]
    registry = make_registry((distribution,))
    discovered = registry.discover()[0]
    reset_started = threading.Event()

    def reset_registry():
        reset_started.set()
        return registry.reset()

    load_thread, load_done, load_results, load_errors = start_daemon_call(
        lambda: registry.load(discovered.registry_key),
        name="registry-load",
    )
    assert entered.wait(timeout=5)
    reset_thread, reset_done, reset_results, reset_failures = (
        start_daemon_call(reset_registry, name="registry-reset-after-load")
    )
    try:
        assert reset_started.wait(timeout=5)
        # Waiting on the Event yields execution to the reset thread.  It must
        # remain blocked until the in-lock entry-point callback is released.
        assert not reset_done.wait(timeout=0.25)
    finally:
        release.set()

    assert load_done.wait(timeout=5)
    assert reset_done.wait(timeout=5)
    load_thread.join(timeout=1)
    reset_thread.join(timeout=1)
    assert not load_thread.is_alive()
    assert not reset_thread.is_alive()
    assert load_errors == []
    assert reset_failures == []
    assert len(load_results) == 1
    assert len(reset_results) == 1
    loaded = load_results[0]
    reset_errors = reset_results[0]

    assert loaded.state is PluginLifecycleState.LOADED
    assert reset_errors == ()
    rediscovered = registry.discover()[0]
    assert rediscovered.state is PluginLifecycleState.DISCOVERED
    assert registry.load(rediscovered.registry_key).state is (
        PluginLifecycleState.LOADED
    )
    assert entry_point.load_calls == 2
    assert registry.reset() == ()


def test_reset_waits_for_register_then_leaves_no_inflight_state(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    plugin = BlockingInitializePlugin(entered, release)
    distribution = distribution_for_plugin(tmp_path, plugin)
    registry = make_registry((distribution,))
    discovered = registry.discover()[0]
    reset_started = threading.Event()

    def reset_registry():
        reset_started.set()
        return registry.reset()

    register_thread, register_done, register_results, register_errors = (
        start_daemon_call(
            lambda: registry.register(discovered.registry_key),
            name="registry-register",
        )
    )
    assert entered.wait(timeout=5)
    reset_thread, reset_done, reset_results, reset_failures = (
        start_daemon_call(reset_registry, name="registry-reset-after-register")
    )
    try:
        assert reset_started.wait(timeout=5)
        # initialize() still owns the Registry lock, so reset cannot complete.
        assert not reset_done.wait(timeout=0.25)
    finally:
        release.set()

    assert register_done.wait(timeout=5)
    assert reset_done.wait(timeout=5)
    register_thread.join(timeout=1)
    reset_thread.join(timeout=1)
    assert not register_thread.is_alive()
    assert not reset_thread.is_alive()
    assert register_errors == []
    assert reset_failures == []
    assert len(register_results) == 1
    assert len(reset_results) == 1
    registered = register_results[0]
    reset_errors = reset_results[0]

    assert registered.state is PluginLifecycleState.REGISTERED
    assert reset_errors == ()
    assert plugin.shutdown_calls == 1
    rediscovered = registry.discover()[0]
    assert rediscovered.state is PluginLifecycleState.DISCOVERED
    assert registry.register(rediscovered.registry_key).state is (
        PluginLifecycleState.REGISTERED
    )
    assert distribution.entry_points[0].load_calls == 2
    assert plugin.initialize_calls == 2
    assert registry.reset() == ()
    assert plugin.shutdown_calls == 2


@pytest.mark.parametrize(
    "first_failure,error_type,expected_field",
    [
        (
            "wheel",
            BackendPluginCompatibilityError,
            "wheel platform tag",
        ),
        (
            "provenance",
            BackendPluginCompatibilityError,
            "Core build provenance",
        ),
        ("protocol", BackendPluginProtocolError, "backend_protocol"),
        (
            "core",
            BackendPluginCompatibilityError,
            "triton-anchor Core version",
        ),
        (
            "triton",
            BackendPluginCompatibilityError,
            "Triton version",
        ),
    ],
)
def test_full_preflight_reports_first_failure_from_multiple_faults(
    tmp_path,
    first_failure,
    error_type,
    expected_field,
):
    declaration = plugin_record(
        backend_protocol=">=2.0,<3.0",
        requires_core=">=9.0",
        requires_triton={"version": ">=9.0"},
        requires_llvm_version=">=20.0",
    )
    environment = core_environment()
    wheel_text = VALID_WHEEL
    if first_failure == "wheel":
        wheel_text = (
            "Wheel-Version: 1.0\n"
            "Root-Is-Purelib: true\n"
            "Tag: cp39-cp39-win_amd64\n"
        )
    elif first_failure == "provenance":
        environment = replace(environment, build_info_generated=False)
    elif first_failure == "core":
        declaration["backend_protocol"] = ">=1.0,<2.0"
    elif first_failure == "triton":
        declaration["backend_protocol"] = ">=1.0,<2.0"
        declaration["requires_core"] = ">=0.2,<0.3"

    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(declaration),
        wheel_text=wheel_text,
    )
    registry = make_registry((distribution,), environment=environment)
    discovered = registry.discover()[0]

    with pytest.raises(error_type) as caught:
        registry.load(discovered.registry_key)

    rejected = registry.inspect(discovered.record_id)
    assert caught.value.to_dict()["field"] == expected_field
    assert rejected.state is PluginLifecycleState.REJECTED
    assert (
        rejected.compatibility_status
        is PluginCompatibilityStatus.INCOMPATIBLE
    )
    assert distribution.entry_points[0].load_calls == 0


def test_backend_plugin_record_to_dict_is_stable_normalized_public_view(
    tmp_path,
):
    plugin = HookPlugin()
    distribution = distribution_for_plugin(tmp_path, plugin)
    registry = make_registry((distribution,))

    discovered = registry.discover()[0].to_dict()

    assert set(discovered) == RECORD_JSON_FIELDS
    assert discovered["source"] == "manifest"
    assert discovered["state"] == "discovered"
    assert discovered["compatibility_status"] == "not_checked"
    assert discovered["manifest"] == {
        "plugin_id": "vendor.mock",
        "entry_point": "mock",
        "backend_protocol": ">=1.0,<2.0",
        "producer_protocol_version": None,
        "targets": ["mock"],
        "capabilities": [],
        "requires_capabilities": [],
        "isolation_mode": "python_only",
        "priority": 0,
    }
    assert discovered["errors"] == []
    assert discovered["loaded"] is False
    assert discovered["registered"] is False
    assert discovered["selected_targets"] == []
    assert json.loads(json.dumps(discovered)) == discovered

    registered = registry.register(discovered["registry_key"]).to_dict()

    assert set(registered) == RECORD_JSON_FIELDS
    assert registered["state"] == "registered"
    assert registered["compatibility_status"] == "compatible"
    assert registered["loaded"] is True
    assert registered["registered"] is True
    assert registered["initialized"] is True
    assert registered["compiler_cls"].endswith(DummyCompiler.__name__)
    assert registered["driver_cls"].endswith(DummyDriver.__name__)
    assert registered["selected_targets"] == []
    assert json.loads(json.dumps(registered)) == registered
