"""Pure-plan coverage for Registry lifecycle and selection bookkeeping."""

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest

from triton_anchor.backends._registry_lifecycle_selection import (
    plan_activation,
    plan_load_precheck,
    plan_previous_selection_release,
    plan_register_precheck,
    plan_selection_switch,
    plan_winner_selection,
)
from triton_anchor.backends.errors import (
    BackendPluginConflictError,
    BackendPluginError,
    BackendPluginLifecycleError,
    BackendPluginLoadError,
)
from triton_anchor.backends.protocol import (
    PluginLifecycleState,
    PluginSource,
)


@dataclass(frozen=True)
class Record:
    record_id: str
    state: PluginLifecycleState
    selected_targets: Tuple[str, ...] = ()
    source: Optional[PluginSource] = PluginSource.MANIFEST
    plugin_id: Optional[str] = None
    entry_point_name: str = "mock"
    stored_error: Optional[BackendPluginError] = None

    @property
    def error(self):
        return self.stored_error


def record(
    state=PluginLifecycleState.VALIDATED,
    *,
    record_id="vendor.mock:mock",
    selected_targets=(),
    source=PluginSource.MANIFEST,
    error=None,
):
    return Record(
        record_id=record_id,
        state=state,
        selected_targets=selected_targets,
        source=source,
        plugin_id=record_id.split(":", 1)[0],
        stored_error=error,
    )


def test_load_precheck_plans_transition_without_mutating_record():
    candidate = record()

    plan = plan_load_precheck(candidate, is_loading=False)

    assert plan.record_id == candidate.record_id
    assert plan.operation == "load"
    assert plan.transition_to is PluginLifecycleState.LOADED
    assert plan.return_existing is False
    assert plan.increment_generation is False
    assert candidate.state is PluginLifecycleState.VALIDATED


@pytest.mark.parametrize(
    "state",
    (
        PluginLifecycleState.LOADED,
        PluginLifecycleState.REGISTERED,
        PluginLifecycleState.SELECTED,
        PluginLifecycleState.ACTIVE,
    ),
)
def test_load_precheck_returns_later_states_before_recursion_check(state):
    plan = plan_load_precheck(record(state), is_loading=True)

    assert plan.return_existing is True
    assert plan.transition_to is None
    assert plan.increment_generation is False


def test_load_precheck_preserves_rejection_and_recursion_errors():
    original = BackendPluginLifecycleError("original rejection")
    rejected = record(PluginLifecycleState.REJECTED, error=original)

    with pytest.raises(BackendPluginLifecycleError) as rejected_error:
        plan_load_precheck(rejected, is_loading=False)
    assert rejected_error.value is original

    with pytest.raises(BackendPluginLoadError) as missing_error:
        plan_load_precheck(
            record(PluginLifecycleState.REJECTED),
            is_loading=False,
        )
    assert missing_error.value.code == "backend_plugin_load_error"

    with pytest.raises(BackendPluginLifecycleError) as recursive_error:
        plan_load_precheck(record(), is_loading=True)
    assert recursive_error.value.field == "load"
    assert recursive_error.value.actual == "recursive load"


def test_load_precheck_keeps_manifest_and_legacy_transition_rules():
    with pytest.raises(BackendPluginLifecycleError) as manifest_error:
        plan_load_precheck(
            record(PluginLifecycleState.DISCOVERED),
            is_loading=False,
        )
    assert manifest_error.value.expected == PluginLifecycleState.LOADED.value
    assert manifest_error.value.actual == PluginLifecycleState.DISCOVERED.value

    legacy = record(
        PluginLifecycleState.DISCOVERED,
        source=PluginSource.LEGACY,
    )
    assert (
        plan_load_precheck(legacy, is_loading=False).transition_to
        is PluginLifecycleState.LOADED
    )


def test_register_precheck_preserves_idempotence_recursion_and_transition():
    loaded = record(PluginLifecycleState.LOADED)
    plan = plan_register_precheck(loaded, is_registering=False)
    assert plan.transition_to is PluginLifecycleState.REGISTERED
    assert plan.return_existing is False
    assert plan.increment_generation is False

    registered = record(PluginLifecycleState.REGISTERED)
    assert plan_register_precheck(
        registered, is_registering=True
    ).return_existing is True

    with pytest.raises(BackendPluginLifecycleError) as recursive_error:
        plan_register_precheck(loaded, is_registering=True)
    assert recursive_error.value.field == "register"
    assert recursive_error.value.actual == "recursive registration"

    with pytest.raises(BackendPluginLifecycleError):
        plan_register_precheck(record(), is_registering=False)


@pytest.mark.parametrize(
    ("operation", "state", "target_state"),
    (
        (
            "load",
            PluginLifecycleState.VALIDATED,
            PluginLifecycleState.LOADED,
        ),
        (
            "register",
            PluginLifecycleState.LOADED,
            PluginLifecycleState.REGISTERED,
        ),
        (
            "previous_release",
            PluginLifecycleState.SELECTED,
            PluginLifecycleState.REGISTERED,
        ),
        (
            "winner",
            PluginLifecycleState.REGISTERED,
            PluginLifecycleState.SELECTED,
        ),
        (
            "activate",
            PluginLifecycleState.SELECTED,
            PluginLifecycleState.ACTIVE,
        ),
    ),
)
def test_transition_plans_use_injected_validator_once_and_propagate_error(
    operation,
    state,
    target_state,
):
    candidate = record(state, selected_targets=("mock",))
    marker = BackendPluginLifecycleError("injected transition rejection")
    calls = []

    def reject_transition(actual_record, actual_target):
        calls.append((actual_record, actual_target))
        raise marker

    with pytest.raises(BackendPluginLifecycleError) as caught:
        if operation == "load":
            plan_load_precheck(
                candidate,
                is_loading=False,
                transition_validator=reject_transition,
            )
        elif operation == "register":
            plan_register_precheck(
                candidate,
                is_registering=False,
                transition_validator=reject_transition,
            )
        elif operation == "previous_release":
            plan_previous_selection_release(
                target="mock",
                winner_record_id="vendor.next:mock",
                previous_record=candidate,
                transition_validator=reject_transition,
            )
        elif operation == "winner":
            plan_winner_selection(
                target="mock",
                winner=candidate,
                transition_validator=reject_transition,
            )
        else:
            plan_activation(
                candidate,
                (candidate,),
                transition_validator=reject_transition,
            )

    assert caught.value is marker
    assert calls == [(candidate, target_state)]


@pytest.mark.parametrize(
    "operation",
    ("load", "register", "previous_release", "winner", "activate"),
)
def test_transition_plans_do_not_validate_idempotent_or_stable_states(
    operation,
):
    def unexpected_transition(record, target):
        pytest.fail(f"unexpected transition: {record.record_id} -> {target}")

    if operation == "load":
        plan_load_precheck(
            record(PluginLifecycleState.LOADED),
            is_loading=False,
            transition_validator=unexpected_transition,
        )
    elif operation == "register":
        plan_register_precheck(
            record(PluginLifecycleState.REGISTERED),
            is_registering=False,
            transition_validator=unexpected_transition,
        )
    elif operation == "previous_release":
        plan_previous_selection_release(
            target="mock",
            winner_record_id="vendor.next:mock",
            previous_record=record(
                PluginLifecycleState.SELECTED,
                selected_targets=("mock", "other"),
            ),
            transition_validator=unexpected_transition,
        )
    elif operation == "winner":
        plan_winner_selection(
            target="mock",
            winner=record(PluginLifecycleState.SELECTED),
            transition_validator=unexpected_transition,
        )
    else:
        active = record(PluginLifecycleState.ACTIVE)
        plan_activation(
            active,
            (active,),
            transition_validator=unexpected_transition,
        )


def test_selection_switch_rejects_active_previous_winner_before_side_effects():
    previous = record(
        PluginLifecycleState.ACTIVE,
        record_id="vendor.previous:mock",
        selected_targets=("mock",),
    )

    with pytest.raises(BackendPluginLifecycleError) as caught:
        plan_selection_switch(
            target="mock",
            winner_record_id="vendor.next:mock",
            previous_record_id=previous.record_id,
            previous_record=previous,
        )

    assert caught.value.field == "active selection"
    assert caught.value.expected == previous.record_id
    assert caught.value.actual == "vendor.next:mock"


def test_selection_switch_identifies_same_and_changed_winners():
    previous = record(
        PluginLifecycleState.ACTIVE,
        selected_targets=("mock",),
    )
    same = plan_selection_switch(
        target="mock",
        winner_record_id=previous.record_id,
        previous_record_id=previous.record_id,
        previous_record=previous,
    )
    changed_without_record = plan_selection_switch(
        target="mock",
        winner_record_id="vendor.next:mock",
        previous_record_id="missing:mock",
        previous_record=None,
    )

    assert same.changes_winner is False
    assert same.increment_generation is False
    assert changed_without_record.changes_winner is True


def test_previous_selection_release_preserves_target_order_and_demotion():
    previous = record(
        PluginLifecycleState.SELECTED,
        record_id="vendor.previous:mock",
        selected_targets=("z", "mock", "a", "mock"),
    )
    retained = plan_previous_selection_release(
        target="mock",
        winner_record_id="vendor.next:mock",
        previous_record=previous,
    )

    assert retained.record_id == previous.record_id
    assert retained.state is PluginLifecycleState.SELECTED
    assert retained.selected_targets == ("z", "a")
    assert retained.increment_generation is False

    released = plan_previous_selection_release(
        target="mock",
        winner_record_id="vendor.next:mock",
        previous_record=record(
            PluginLifecycleState.ACTIVE,
            record_id=previous.record_id,
            selected_targets=("mock",),
        ),
    )
    assert released.state is PluginLifecycleState.REGISTERED
    assert released.selected_targets == ()


def test_previous_selection_release_skips_missing_or_same_winner():
    winner = record(
        PluginLifecycleState.SELECTED,
        selected_targets=("mock",),
    )

    assert plan_previous_selection_release(
        target="mock",
        winner_record_id=winner.record_id,
        previous_record=None,
    ) is None
    assert plan_previous_selection_release(
        target="mock",
        winner_record_id=winner.record_id,
        previous_record=winner,
    ) is None


@pytest.mark.parametrize(
    "state",
    (PluginLifecycleState.SELECTED, PluginLifecycleState.ACTIVE),
)
def test_winner_selection_preserves_selected_or_active_state(state):
    winner = record(state, selected_targets=("z", "a", "z"))

    plan = plan_winner_selection(target="mock", winner=winner)

    assert plan.state is state
    assert plan.selected_targets == ("a", "mock", "z")
    assert plan.increment_generation is True
    assert winner.selected_targets == ("z", "a", "z")


def test_winner_selection_transitions_registered_and_rejects_other_states():
    plan = plan_winner_selection(
        target="mock",
        winner=record(PluginLifecycleState.REGISTERED),
    )
    assert plan.state is PluginLifecycleState.SELECTED
    assert plan.selected_targets == ("mock",)

    with pytest.raises(BackendPluginLifecycleError) as caught:
        plan_winner_selection(
            target="mock",
            winner=record(PluginLifecycleState.LOADED),
        )
    assert caught.value.field == "state"
    assert caught.value.expected == "registered, selected, or active"
    assert caught.value.actual == PluginLifecycleState.LOADED.value


def test_activation_plans_transition_or_idempotent_return_without_generation():
    selected = record(PluginLifecycleState.SELECTED)
    plan = plan_activation(selected, (selected,))
    assert plan.transition_to is PluginLifecycleState.ACTIVE
    assert plan.return_existing is False
    assert plan.increment_generation is False

    active = record(PluginLifecycleState.ACTIVE)
    plan = plan_activation(active, (active,))
    assert plan.transition_to is None
    assert plan.return_existing is True


def test_activation_preserves_conflict_order_and_selected_precondition():
    candidate = record(
        PluginLifecycleState.SELECTED,
        record_id="vendor.candidate:mock",
    )
    active_z = record(
        PluginLifecycleState.ACTIVE,
        record_id="vendor.z:mock",
    )
    active_a = record(
        PluginLifecycleState.ACTIVE,
        record_id="vendor.a:mock",
    )

    with pytest.raises(BackendPluginConflictError) as conflict:
        plan_activation(candidate, (active_z, candidate, active_a))
    assert conflict.value.field == "active"
    assert conflict.value.actual == (
        "vendor.a:mock, vendor.z:mock, vendor.candidate:mock"
    )

    with pytest.raises(BackendPluginLifecycleError) as lifecycle:
        plan_activation(
            record(PluginLifecycleState.REGISTERED),
            (),
        )
    assert lifecycle.value.expected == PluginLifecycleState.SELECTED.value
    assert lifecycle.value.actual == PluginLifecycleState.REGISTERED.value
