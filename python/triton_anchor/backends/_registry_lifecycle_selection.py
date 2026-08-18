"""Pure lifecycle and selection plans for the Registry facade.

The helpers in this module only inspect immutable record snapshots and return
immutable plans.  They do not load plugins, invoke lifecycle callbacks, mutate
``RegistryState``, acquire locks, or advance the Registry generation.  The
facade remains responsible for applying every plan in its established order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol, Tuple

from ._registry_lifecycle import ensure_transition
from .errors import (
    BackendPluginConflictError,
    BackendPluginError,
    BackendPluginLifecycleError,
    BackendPluginLoadError,
)
from .protocol import PluginLifecycleState, PluginSource


class LifecycleSelectionRecord(Protocol):
    """Read-only record fields needed to build lifecycle/selection plans."""

    record_id: str
    entry_point_name: str
    source: Optional[PluginSource]
    state: PluginLifecycleState
    selected_targets: Tuple[str, ...]

    @property
    def plugin_id(self) -> Optional[str]: ...

    @property
    def error(self) -> Optional[BackendPluginError]: ...


TransitionValidator = Callable[
    [LifecycleSelectionRecord, PluginLifecycleState], None
]


@dataclass(frozen=True)
class LifecyclePrecheckPlan:
    """Result of a load/register/activate precheck.

    ``transition_to`` is applied by the facade only when ``return_existing``
    is false.  Lifecycle operations represented here never advance the
    Registry generation.
    """

    record_id: str
    operation: str
    transition_to: Optional[PluginLifecycleState]
    return_existing: bool
    increment_generation: bool = False


@dataclass(frozen=True)
class SelectionSwitchPlan:
    """Winner identity resolved before the facade registers the winner."""

    target: str
    winner_record_id: str
    previous_record_id: Optional[str]
    changes_winner: bool
    increment_generation: bool = False


@dataclass(frozen=True)
class SelectionRecordUpdatePlan:
    """One record replacement for selection bookkeeping."""

    record_id: str
    state: PluginLifecycleState
    selected_targets: Tuple[str, ...]
    increment_generation: bool


_LOADED_OR_LATER = frozenset({
    PluginLifecycleState.LOADED,
    PluginLifecycleState.REGISTERED,
    PluginLifecycleState.SELECTED,
    PluginLifecycleState.ACTIVE,
})

_REGISTERED_OR_LATER = frozenset({
    PluginLifecycleState.REGISTERED,
    PluginLifecycleState.SELECTED,
    PluginLifecycleState.ACTIVE,
})


def plan_load_precheck(
    record: LifecycleSelectionRecord,
    *,
    is_loading: bool,
    transition_validator: TransitionValidator = ensure_transition,
) -> LifecyclePrecheckPlan:
    """Validate the import precheck without importing or mutating anything."""
    if record.state is PluginLifecycleState.REJECTED:
        if record.error is not None:
            raise record.error
        raise BackendPluginLoadError(
            "Rejected backend plugin cannot be loaded",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
        )
    if record.state in _LOADED_OR_LATER:
        return LifecyclePrecheckPlan(
            record_id=record.record_id,
            operation="load",
            transition_to=None,
            return_existing=True,
        )
    if is_loading:
        raise BackendPluginLifecycleError(
            "Recursive backend plugin load is not allowed",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="load",
            expected="one non-reentrant load operation",
            actual="recursive load",
            remediation=(
                "Do not call Registry load/register for the same plugin "
                "from its entry-point loader or constructor."
            ),
        )
    transition_validator(record, PluginLifecycleState.LOADED)
    return LifecyclePrecheckPlan(
        record_id=record.record_id,
        operation="load",
        transition_to=PluginLifecycleState.LOADED,
        return_existing=False,
    )


def plan_register_precheck(
    record: LifecycleSelectionRecord,
    *,
    is_registering: bool,
    transition_validator: TransitionValidator = ensure_transition,
) -> LifecyclePrecheckPlan:
    """Validate registration before the facade inspects runtime interfaces."""
    if record.state in _REGISTERED_OR_LATER:
        return LifecyclePrecheckPlan(
            record_id=record.record_id,
            operation="register",
            transition_to=None,
            return_existing=True,
        )
    if is_registering:
        raise BackendPluginLifecycleError(
            "Recursive backend plugin registration is not allowed",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="register",
            expected="one non-reentrant registration operation",
            actual="recursive registration",
            remediation=(
                "Do not call Registry register for the same plugin from "
                "runtime attributes or initialize()."
            ),
        )
    transition_validator(record, PluginLifecycleState.REGISTERED)
    return LifecyclePrecheckPlan(
        record_id=record.record_id,
        operation="register",
        transition_to=PluginLifecycleState.REGISTERED,
        return_existing=False,
    )


def plan_selection_switch(
    *,
    target: str,
    winner_record_id: str,
    previous_record_id: Optional[str],
    previous_record: Optional[LifecycleSelectionRecord],
) -> SelectionSwitchPlan:
    """Reject an ACTIVE winner switch before registration side effects."""
    changes_winner = (
        previous_record_id is not None
        and previous_record_id != winner_record_id
    )
    if (
        changes_winner
        and previous_record is not None
        and previous_record.state is PluginLifecycleState.ACTIVE
    ):
        raise BackendPluginLifecycleError(
            "Cannot switch an ACTIVE backend selection",
            plugin_id=previous_record.plugin_id,
            entry_point=previous_record.entry_point_name,
            field="active selection",
            expected=previous_record.record_id,
            actual=winner_record_id,
            remediation=(
                "Reset the Registry-backed runtime driver before selecting "
                "a different plugin for this target."
            ),
        )
    return SelectionSwitchPlan(
        target=target,
        winner_record_id=winner_record_id,
        previous_record_id=previous_record_id,
        changes_winner=changes_winner,
    )


def plan_previous_selection_release(
    *,
    target: str,
    winner_record_id: str,
    previous_record: Optional[LifecycleSelectionRecord],
    transition_validator: TransitionValidator = ensure_transition,
) -> Optional[SelectionRecordUpdatePlan]:
    """Plan removal of a target from a previous winner after registration.

    This is intentionally separate from :func:`plan_winner_selection`.  The
    facade must apply this plan before planning/applying the winner so the
    existing state-write and error ordering remains unchanged.
    """
    if (
        previous_record is None
        or previous_record.record_id == winner_record_id
    ):
        return None
    remaining_targets = tuple(
        item for item in previous_record.selected_targets if item != target
    )
    previous_state = previous_record.state
    if (
        previous_state
        in {PluginLifecycleState.SELECTED, PluginLifecycleState.ACTIVE}
        and not remaining_targets
    ):
        transition_validator(
            previous_record,
            PluginLifecycleState.REGISTERED,
        )
        previous_state = PluginLifecycleState.REGISTERED
    return SelectionRecordUpdatePlan(
        record_id=previous_record.record_id,
        state=previous_state,
        selected_targets=remaining_targets,
        increment_generation=False,
    )


def plan_winner_selection(
    *,
    target: str,
    winner: LifecycleSelectionRecord,
    transition_validator: TransitionValidator = ensure_transition,
) -> SelectionRecordUpdatePlan:
    """Plan winner state/target bookkeeping after successful registration."""
    selected_targets = tuple(
        sorted(set(winner.selected_targets).union({target}))
    )
    selected_state = winner.state
    if selected_state is PluginLifecycleState.REGISTERED:
        transition_validator(winner, PluginLifecycleState.SELECTED)
        selected_state = PluginLifecycleState.SELECTED
    elif selected_state not in {
        PluginLifecycleState.SELECTED,
        PluginLifecycleState.ACTIVE,
    }:
        raise BackendPluginLifecycleError(
            "Selected backend is not registered",
            plugin_id=winner.plugin_id,
            entry_point=winner.entry_point_name,
            field="state",
            expected="registered, selected, or active",
            actual=selected_state.value,
            remediation=(
                "Use Registry.select() so validation, loading, and "
                "registration complete before state selection."
            ),
        )
    return SelectionRecordUpdatePlan(
        record_id=winner.record_id,
        state=selected_state,
        selected_targets=selected_targets,
        increment_generation=True,
    )


def plan_activation(
    record: LifecycleSelectionRecord,
    records: Iterable[LifecycleSelectionRecord],
    *,
    transition_validator: TransitionValidator = ensure_transition,
) -> LifecyclePrecheckPlan:
    """Plan activation while preserving the process-wide ACTIVE invariant."""
    other_active = tuple(
        candidate
        for candidate in records
        if (
            candidate.record_id != record.record_id
            and candidate.state is PluginLifecycleState.ACTIVE
        )
    )
    if other_active:
        active_ids = tuple(
            sorted(candidate.record_id for candidate in other_active)
        )
        raise BackendPluginConflictError(
            "Cannot activate more than one backend plugin: "
            + ", ".join(active_ids + (record.record_id,)),
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="active",
            expected="one ACTIVE backend plugin",
            actual=", ".join(active_ids + (record.record_id,)),
            remediation=(
                "Keep the current active driver or explicitly reset "
                "runtime state before activating another backend."
            ),
        )
    if record.state is PluginLifecycleState.ACTIVE:
        return LifecyclePrecheckPlan(
            record_id=record.record_id,
            operation="activate",
            transition_to=None,
            return_existing=True,
        )
    if record.state is not PluginLifecycleState.SELECTED:
        raise BackendPluginLifecycleError(
            "Backend plugin must be selected before activation",
            plugin_id=record.plugin_id,
            entry_point=record.entry_point_name,
            field="state",
            expected=PluginLifecycleState.SELECTED.value,
            actual=record.state.value,
            remediation=(
                "Resolve the backend through Registry.select() before "
                "constructing and activating its runtime driver."
            ),
        )
    transition_validator(record, PluginLifecycleState.ACTIVE)
    return LifecyclePrecheckPlan(
        record_id=record.record_id,
        operation="activate",
        transition_to=PluginLifecycleState.ACTIVE,
        return_existing=False,
    )


__all__ = [
    "LifecyclePrecheckPlan",
    "SelectionRecordUpdatePlan",
    "SelectionSwitchPlan",
    "plan_activation",
    "plan_load_precheck",
    "plan_previous_selection_release",
    "plan_register_precheck",
    "plan_selection_switch",
    "plan_winner_selection",
]
