"""Observable ACTIVE and reset ordering contracts for stage 3."""

import threading
from concurrent.futures import ThreadPoolExecutor

from triton_anchor.backends import (
    BackendPluginConflictError,
    BackendPluginLifecycleError,
    PluginLifecycleState,
)
from triton_anchor.tests.test_backend_registry import HookPlugin, make_registry
from triton_anchor.tests.test_backend_registry_resilience import (
    BlockingInitializePlugin,
    distribution_for_plugin,
    start_daemon_call,
)
from triton_anchor.tests.test_backend_registry_selection import (
    distribution_for,
    registry_for,
    triton_plugin,
)


def test_concurrent_activation_has_exactly_one_active_record(tmp_path):
    alpha = distribution_for(
        tmp_path,
        "alpha",
        name="alpha-backend",
        declaration=triton_plugin("alpha", targets=("alpha",)),
    )
    beta = distribution_for(
        tmp_path,
        "beta",
        name="beta-backend",
        declaration=triton_plugin("beta", targets=("beta",)),
    )
    registry = registry_for((alpha, beta))
    alpha_decision = registry.select("alpha", environment={})
    beta_decision = registry.select("beta", environment={})
    start = threading.Barrier(3, timeout=5)

    def activate(record_id):
        start.wait()
        try:
            return registry.activate(record_id)
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(activate, alpha_decision.record_id),
            executor.submit(activate, beta_decision.record_id),
        )
        start.wait()
        outcomes = tuple(future.result(timeout=5) for future in futures)

    activated = tuple(
        outcome
        for outcome in outcomes
        if not isinstance(outcome, Exception)
    )
    rejected = tuple(
        outcome for outcome in outcomes if isinstance(outcome, Exception)
    )
    assert len(activated) == 1
    assert len(rejected) == 1
    assert isinstance(rejected[0], BackendPluginConflictError)
    assert rejected[0].field == "active"
    assert sorted(record.state.value for record in registry.list()) == [
        PluginLifecycleState.ACTIVE.value,
        PluginLifecycleState.SELECTED.value,
    ]


def test_select_holding_registry_lock_completes_before_waiting_reset(tmp_path):
    initialize_entered = threading.Event()
    release_initialize = threading.Event()
    plugin = BlockingInitializePlugin(
        initialize_entered,
        release_initialize,
    )
    distribution = distribution_for_plugin(tmp_path, plugin)
    registry = make_registry((distribution,))
    reset_started = threading.Event()

    select_thread, select_done, selections, select_errors = start_daemon_call(
        lambda: registry.select("mock", environment={}),
        name="stage3-select-before-reset",
    )
    assert initialize_entered.wait(timeout=5)

    def reset_registry():
        reset_started.set()
        return registry.reset()

    reset_thread, reset_done, reset_results, reset_errors = start_daemon_call(
        reset_registry,
        name="stage3-reset-after-select",
    )
    try:
        assert reset_started.wait(timeout=5)
        assert not reset_done.wait(timeout=0.25)
    finally:
        release_initialize.set()

    assert select_done.wait(timeout=5)
    assert reset_done.wait(timeout=5)
    select_thread.join(timeout=1)
    reset_thread.join(timeout=1)
    assert not select_thread.is_alive()
    assert not reset_thread.is_alive()
    assert select_errors == []
    assert reset_errors == []
    assert len(selections) == 1
    assert reset_results == [()]
    assert registry.get_selection("mock") is None

    retry = registry.select("mock", environment={})
    assert retry.record.state is PluginLifecycleState.SELECTED
    assert distribution.entry_points[0].load_calls == 2
    assert plugin.initialize_calls == 2
    assert plugin.shutdown_calls == 1
    assert registry.reset() == ()


def test_select_during_reset_hook_fails_then_succeeds_after_reset(tmp_path):
    plugin = HookPlugin()
    distribution = distribution_for_plugin(tmp_path, plugin)
    registry = make_registry((distribution,))
    registry.select("mock", environment={})
    hook_entered = threading.Event()
    release_hook = threading.Event()

    def blocking_hook():
        hook_entered.set()
        if not release_hook.wait(timeout=5):
            raise RuntimeError("reset hook barrier timed out")

    registry.register_reset_hook(blocking_hook)
    reset_thread, reset_done, reset_results, reset_errors = start_daemon_call(
        registry.reset,
        name="stage3-reset-before-select",
    )
    assert hook_entered.wait(timeout=5)
    select_thread, select_done, _, select_errors = start_daemon_call(
        lambda: registry.select("mock", environment={}),
        name="stage3-select-during-reset",
    )
    try:
        assert select_done.wait(timeout=2)
    finally:
        release_hook.set()

    assert reset_done.wait(timeout=5)
    select_thread.join(timeout=1)
    reset_thread.join(timeout=1)
    assert not select_thread.is_alive()
    assert not reset_thread.is_alive()
    assert len(select_errors) == 1
    assert isinstance(select_errors[0], BackendPluginLifecycleError)
    assert select_errors[0].field == "select"
    assert reset_errors == []
    assert reset_results == [()]

    retry = registry.select("mock", environment={})
    assert retry.record.state is PluginLifecycleState.SELECTED
    assert distribution.entry_points[0].load_calls == 2
    assert plugin.shutdown_calls == 1
    assert registry.reset() == ()


def test_reset_hook_failure_keeps_order_and_does_not_skip_shutdown(tmp_path):
    plugin = HookPlugin()
    distribution = distribution_for_plugin(tmp_path, plugin)
    registry = make_registry((distribution,))
    registry.select("mock", environment={})
    events = []

    def failing_hook():
        events.append("first-hook")
        raise RuntimeError("hook failure")

    def succeeding_hook():
        events.append("second-hook")

    plugin.shutdown = lambda: events.append("shutdown")
    registry.register_reset_hook(failing_hook)
    registry.register_reset_hook(succeeding_hook)

    errors = registry.reset()

    assert events == ["first-hook", "second-hook", "shutdown"]
    assert len(errors) == 1
    assert isinstance(errors[0], BackendPluginLifecycleError)
    assert errors[0].field == "reset_hook"
    assert registry.get_selection("mock") is None
