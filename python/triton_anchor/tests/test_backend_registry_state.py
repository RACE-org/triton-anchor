"""Focused invariants for the internal Registry state container."""

from dataclasses import dataclass

from triton_anchor.backends._registry_selection_state import SelectionState
from triton_anchor.backends._registry_state import RegistryState
from triton_anchor.backends.errors import BackendPluginError
from triton_anchor.backends.registry import BackendPluginRegistry
from triton_anchor.backends.selection import SelectionMethod


@dataclass(frozen=True)
class Record:
    record_id: str
    revision: int = 0


def selection_state(record_id="alpha"):
    return SelectionState(
        target="mock",
        record_id=record_id,
        registry_key=record_id,
        plugin_id=f"vendor.{record_id}",
        entry_point_name=record_id,
        method=SelectionMethod.SOLE_CANDIDATE,
        selector=None,
        priority=0,
        is_legacy=False,
        candidate_record_ids=(record_id,),
        capability_report=None,
    )


def test_record_snapshots_are_stable_and_preserve_insertion_order():
    state = RegistryState()
    alpha = Record("alpha")
    beta = Record("beta")

    state.insert_record(alpha)
    snapshot = state.records_snapshot()
    state.insert_record(beta)

    assert snapshot == (alpha,)
    assert state.record_ids_snapshot() == ("alpha", "beta")
    assert state.records_snapshot() == (alpha, beta)
    assert "alpha" in state
    assert "missing" not in state
    assert state.contains_record("alpha")
    assert state.get_record("missing") is None


def test_state_is_a_constant_time_container_for_record_id_allocation():
    registry = BackendPluginRegistry(distribution_provider=lambda: ())
    state = registry._state
    state.insert_record(Record("alpha-backend:mock"))
    state.insert_record(Record("alpha-backend:mock#2"))

    def fail_snapshot():
        raise AssertionError("record ID allocation copied a snapshot")

    state.record_ids_snapshot = fail_snapshot

    assert registry._allocate_record_id("alpha-backend", "mock") == (
        "alpha-backend:mock#3"
    )


def test_replace_record_does_not_mutate_selection_or_generation():
    state = RegistryState()
    original = Record("alpha")
    updated = Record("alpha", revision=1)
    selection = selection_state()
    state.insert_record(original)
    state.set_selection_state("mock", selection)

    state.replace_record(updated)

    assert state.get_record("alpha") is updated
    assert state.get_selection_state("mock") is selection
    assert state.generation == 0


def test_selection_and_generation_mutations_are_explicit():
    state = RegistryState()
    selection = selection_state()

    state.set_selection_state("mock", selection)
    assert state.selection_states_snapshot() == (("mock", selection),)
    assert state.generation == 0

    state.increment_generation()
    state.clear_selection_states()
    assert state.get_selection_state("mock") is None
    assert state.generation == 1


def test_registry_errors_use_identity_deduplication_and_explicit_replace():
    state = RegistryState()
    first = BackendPluginError("first")
    second = BackendPluginError("second")

    state.append_registry_error(first)
    state.append_registry_error(first)
    assert state.registry_errors_snapshot() == (first,)

    state.replace_registry_errors((second,))
    assert state.registry_errors_snapshot() == (second,)


def test_inflight_snapshots_are_immutable_and_independent():
    state = RegistryState()
    state.begin_loading("alpha")
    state.begin_registering("beta")
    loading = state.loading_snapshot()
    registering = state.registering_snapshot()

    state.end_loading("alpha")
    state.end_registering("beta")

    assert loading == frozenset({"alpha"})
    assert registering == frozenset({"beta"})
    assert state.loading_snapshot() == frozenset()
    assert state.registering_snapshot() == frozenset()


def test_clear_for_reset_preserves_epoch_hooks_and_resetting_marker():
    state = RegistryState()
    record = Record("alpha")
    selection = selection_state()
    environment = object()
    environment_error = BackendPluginError("environment")
    hook_calls = []

    def hook():
        hook_calls.append("hook")

    state.insert_record(record)
    state.append_registry_error(environment_error)
    state.cache_environment(environment)
    state.cache_environment_error(environment_error)
    state.mark_discovered()
    state.set_selection_state("mock", selection)
    state.begin_loading("alpha")
    state.begin_registering("alpha")
    state.add_reset_hook(hook)
    state.add_reset_hook(hook)
    state.begin_reset()
    state.increment_generation()

    state.clear_for_reset()

    assert state.records_snapshot() == ()
    assert state.registry_errors_snapshot() == ()
    assert state.environment is None
    assert state.environment_error is None
    assert state.discovered is False
    assert state.selection_states_snapshot() == ()
    assert state.loading_snapshot() == frozenset()
    assert state.registering_snapshot() == frozenset()
    assert state.reset_hooks_snapshot() == (hook,)
    assert state.resetting is True
    assert state.generation == 1
    assert hook_calls == []

    state.end_reset()
    assert state.resetting is False
