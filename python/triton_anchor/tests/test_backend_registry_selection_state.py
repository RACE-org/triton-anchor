"""Focused tests for record-free Registry selection state."""

from dataclasses import FrozenInstanceError, dataclass

import pytest

from triton_anchor.backends._registry_selection_state import (
    SelectionState,
    materialize_selection_decision,
    selection_state_from_decision,
)
from triton_anchor.backends.capabilities import evaluate_capabilities
from triton_anchor.backends.selection import SelectionDecision, SelectionMethod


@dataclass(frozen=True)
class Record:
    record_id: str
    revision: int = 0


def decision(record):
    report = evaluate_capabilities(
        core_provided=("core",),
        plugin_provided=("plugin",),
        plugin_required=("core",),
        kernel_required=("plugin",),
        plugin_id="vendor.alpha",
        entry_point="alpha",
    )
    return SelectionDecision(
        record_id="alpha-record",
        registry_key="vendor.alpha",
        plugin_id="vendor.alpha",
        entry_point_name="alpha",
        target="mock",
        method=SelectionMethod.ENVIRONMENT,
        selector="vendor.alpha",
        priority=7,
        is_legacy=False,
        candidate_record_ids=("alpha-record", "beta-record"),
        capability_report=report,
        record=record,
    )


def test_selection_state_drops_only_the_record_snapshot():
    original = decision(Record("alpha-record"))

    state = selection_state_from_decision(original)

    assert state == SelectionState(
        target=original.target,
        record_id=original.record_id,
        registry_key=original.registry_key,
        plugin_id=original.plugin_id,
        entry_point_name=original.entry_point_name,
        method=original.method,
        selector=original.selector,
        priority=original.priority,
        is_legacy=original.is_legacy,
        candidate_record_ids=original.candidate_record_ids,
        capability_report=original.capability_report,
    )
    assert not hasattr(state, "record")

    with pytest.raises(FrozenInstanceError):
        state.priority = 8


def test_materialization_uses_current_record_without_changing_public_view():
    original_record = Record("alpha-record")
    current_record = Record("alpha-record", revision=1)
    original = decision(original_record)
    state = selection_state_from_decision(original)

    materialized = materialize_selection_decision(state, current_record)

    assert materialized == original
    assert materialized.to_dict() == original.to_dict()
    assert materialized.record is current_record
    assert original.record is original_record
