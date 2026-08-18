"""Immutable Registry selection state without a record snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .capabilities import CapabilityReport
from .selection import SelectionDecision, SelectionMethod


@dataclass(frozen=True)
class SelectionState:
    """Selection identity plus the provenance needed for public materialization.

    Lifecycle state, selected targets, and runtime classes deliberately remain
    owned by the canonical BackendPluginRecord in RegistryState.
    """

    target: str
    record_id: str
    registry_key: str
    plugin_id: Optional[str]
    entry_point_name: str
    method: SelectionMethod
    selector: Optional[str]
    priority: int
    is_legacy: bool
    candidate_record_ids: Tuple[str, ...]
    capability_report: Optional[CapabilityReport]


def selection_state_from_decision(
    decision: SelectionDecision,
) -> SelectionState:
    """Drop only the replaceable record snapshot from a public decision."""
    return SelectionState(
        target=decision.target,
        record_id=decision.record_id,
        registry_key=decision.registry_key,
        plugin_id=decision.plugin_id,
        entry_point_name=decision.entry_point_name,
        method=decision.method,
        selector=decision.selector,
        priority=decision.priority,
        is_legacy=decision.is_legacy,
        candidate_record_ids=decision.candidate_record_ids,
        capability_report=decision.capability_report,
    )


def materialize_selection_decision(
    state: SelectionState,
    record: Any,
) -> SelectionDecision:
    """Attach the current canonical record to stored selection provenance."""
    return SelectionDecision(
        record_id=state.record_id,
        registry_key=state.registry_key,
        plugin_id=state.plugin_id,
        entry_point_name=state.entry_point_name,
        target=state.target,
        method=state.method,
        selector=state.selector,
        priority=state.priority,
        is_legacy=state.is_legacy,
        candidate_record_ids=state.candidate_record_ids,
        capability_report=state.capability_report,
        record=record,
    )
