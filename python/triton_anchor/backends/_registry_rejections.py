"""Pure rejection plans applied by the Registry facade.

This module does not acquire locks, read Registry state, invoke callbacks, or
mutate records.  The facade preserves the established apply and propagation
order inside its existing ``RLock`` boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Protocol, Tuple

from .errors import BackendPluginCompatibilityError, BackendPluginError
from .protocol import (
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
)


class RejectableRecord(Protocol):
    """The immutable record fields retained by a rejection plan."""

    record_id: str
    errors: Tuple[BackendPluginError, ...]
    source: Optional[PluginSource]
    state: PluginLifecycleState
    distribution: object
    entry_point_name: str

    @property
    def plugin_id(self) -> Optional[str]: ...


@dataclass(frozen=True)
class RecordRejectionPlan:
    """One exact record snapshot and the error/status to append to it."""

    record_id: str
    record: RejectableRecord = field(repr=False, compare=False)
    error: BackendPluginError = field(repr=False, compare=False)
    compatibility_status: PluginCompatibilityStatus


def plan_record_rejection(
    record: RejectableRecord,
    error: BackendPluginError,
    compatibility_status: PluginCompatibilityStatus,
) -> RecordRejectionPlan:
    """Capture a side-effect-free rejection of one exact record snapshot."""
    return RecordRejectionPlan(
        record_id=record.record_id,
        record=record,
        error=error,
        compatibility_status=compatibility_status,
    )


def plan_distribution_rejections(
    records: Iterable[RejectableRecord],
    error: BackendPluginError,
    compatibility_status: PluginCompatibilityStatus,
    *,
    distribution: object,
    specialize_error: bool,
) -> Iterator[RecordRejectionPlan]:
    """Yield ordered updates for one exact distribution identity."""
    for candidate in records:
        if (
            candidate.source is not PluginSource.MANIFEST
            or candidate.state is not PluginLifecycleState.DISCOVERED
            or candidate.distribution is not distribution
        ):
            continue
        candidate_error = error
        if (
            specialize_error
            and isinstance(error, BackendPluginCompatibilityError)
        ):
            candidate_error = BackendPluginCompatibilityError(
                error.dimension,
                error.expected,
                error.actual,
                plugin_id=candidate.plugin_id,
                entry_point=candidate.entry_point_name,
                remediation=error.remediation,
            )
        yield plan_record_rejection(
            candidate,
            candidate_error,
            compatibility_status,
        )


def plan_process_rejections(
    records: Iterable[RejectableRecord],
    error: BackendPluginError,
    compatibility_status: PluginCompatibilityStatus,
) -> Iterator[RecordRejectionPlan]:
    """Yield ordered updates for the process-wide Manifest validation scope."""
    for candidate in records:
        if (
            candidate.source is not PluginSource.MANIFEST
            or candidate.state is not PluginLifecycleState.DISCOVERED
        ):
            continue
        yield plan_record_rejection(
            candidate,
            error,
            compatibility_status,
        )


__all__ = [
    "RecordRejectionPlan",
    "RejectableRecord",
    "plan_distribution_rejections",
    "plan_process_rejections",
    "plan_record_rejection",
]
