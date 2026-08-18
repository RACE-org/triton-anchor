"""Public state-source contracts required before the stage-3 refactor."""

import pytest

from triton_anchor.backends import (
    BackendPluginLifecycleError,
    BackendPluginSelectionError,
    PluginLifecycleState,
)
from triton_anchor.tests.test_backend_registry_selection import (
    distribution_for,
    registry_for,
    triton_plugin,
)


def test_current_selection_materializes_active_record_without_generation_change(
    tmp_path,
):
    distribution = distribution_for(
        tmp_path,
        "mock",
        name="mock-backend",
    )
    registry = registry_for((distribution,))

    historical = registry.select("mock", environment={})
    selected_json = historical.to_dict()
    selected_generation = registry.generation
    activated = registry.activate(historical.record_id)
    current = registry.get_selection("mock")

    assert current is not None
    assert current.to_dict() == selected_json
    assert current.record == registry.inspect(historical.record_id)
    assert current.record.state is PluginLifecycleState.ACTIVE
    assert current.record.selected_targets == ("mock",)
    assert activated == current.record
    assert registry.generation == selected_generation
    assert historical.record.state is PluginLifecycleState.SELECTED


def test_multi_target_current_decisions_share_latest_record_snapshot(tmp_path):
    distribution = distribution_for(
        tmp_path,
        "multi",
        name="multi-backend",
        declaration=triton_plugin(
            "multi",
            targets=("target-a", "target-b"),
        ),
    )
    registry = registry_for((distribution,))

    historical_b = registry.select("target-b", environment={})
    generation_after_b = registry.generation
    current_a = registry.select("target-a", environment={})
    current_b = registry.get_selection("target-b")
    canonical = registry.inspect(current_a.record_id)

    assert current_b is not None
    assert current_a.record == canonical
    assert current_b.record == canonical
    assert current_a.record_id == current_b.record_id
    assert canonical.selected_targets == ("target-a", "target-b")
    assert historical_b.record.selected_targets == ("target-b",)
    assert registry.generation != generation_after_b


def test_generation_and_reset_hook_order_remain_facade_owned(tmp_path):
    distribution = distribution_for(
        tmp_path,
        "mock",
        name="mock-backend",
    )
    plugin = distribution.entry_points[0].loaded_object
    registry = registry_for((distribution,))
    initial_generation = registry.generation

    discovered = registry.discover()[0]
    registry.validate()
    registry.load(discovered.record_id)
    registry.register(discovered.record_id)
    assert registry.generation == initial_generation

    decision = registry.select("mock", environment={})
    selection_generation = registry.generation
    assert selection_generation != initial_generation

    registry.activate(decision.record_id)
    assert registry.generation == selection_generation

    with pytest.raises(BackendPluginSelectionError):
        registry.select(
            "mock",
            explicit_selector="missing-plugin",
            environment={},
        )
    assert registry.generation == selection_generation

    events = []

    def reset_hook():
        try:
            registry.discover()
        except BackendPluginLifecycleError as exc:
            reentry_field = exc.field
        else:  # pragma: no cover - the frozen contract requires rejection.
            reentry_field = None
        events.append(
            (
                "hook",
                registry.get_selection("mock"),
                registry.generation,
                reentry_field,
            )
        )

    registry.register_reset_hook(reset_hook)
    plugin.shutdown = lambda: events.append(("shutdown",))

    assert registry.reset() == ()
    assert events[0][0] == "hook"
    assert events[0][1] is None
    assert events[0][2] != selection_generation
    assert events[0][3] == "discover"
    assert events[1] == ("shutdown",)
