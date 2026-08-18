"""Pure rejection plans applied by the Registry facade.

This module does not acquire locks, read Registry state, invoke callbacks, or
mutate records.  The facade preserves the established apply and propagation
order inside its existing ``RLock`` boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Tuple

from .errors import BackendPluginError
from .protocol import PluginCompatibilityStatus


class RejectableRecord(Protocol):
    """The immutable record fields retained by a rejection plan."""

    record_id: str
    errors: Tuple[BackendPluginError, ...]


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


__all__ = [
    "RecordRejectionPlan",
    "RejectableRecord",
    "plan_record_rejection",
]
