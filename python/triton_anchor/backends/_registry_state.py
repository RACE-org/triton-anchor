"""Lock-free mutable state owned by the backend Registry facade.

``BackendPluginRegistry`` remains responsible for acquiring its ``RLock`` and
for every provider call, plugin callback, and lifecycle side effect.  This
module only stores Registry-local state and exposes immutable snapshots to its
caller.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple

from .errors import BackendPluginError


class RegistryState:
    """The single mutable state source for one backend Registry instance.

    Methods intentionally do not acquire locks.  The owning Registry facade
    must hold its existing ``RLock`` whenever concurrent access is possible.
    """

    def __init__(self) -> None:
        self.__records: Dict[str, Any] = {}
        self.__registry_errors: Tuple[BackendPluginError, ...] = ()
        self.__environment: Any = None
        self.__environment_error: Optional[BackendPluginError] = None
        self.__discovered = False
        self.__selections: Dict[str, Any] = {}
        self.__generation = 0
        self.__loading = set()
        self.__registering = set()
        self.__resetting = False
        self.__reset_hooks = []

    def __contains__(self, record_id: object) -> bool:
        return self.contains_record(record_id)

    def contains_record(self, record_id: object) -> bool:
        return record_id in self.__records

    def get_record(self, record_id: str) -> Any:
        return self.__records.get(record_id)

    def record_ids_snapshot(self) -> Tuple[str, ...]:
        return tuple(self.__records)

    def records_snapshot(self) -> Tuple[Any, ...]:
        return tuple(self.__records.values())

    def insert_record(self, record: Any) -> Any:
        self.__records[record.record_id] = record
        return record

    def replace_record(self, record: Any) -> Any:
        """Replace a record and synchronize cached decision snapshots."""
        self.__records[record.record_id] = record
        for target, decision in tuple(self.__selections.items()):
            if decision.record_id == record.record_id:
                self.__selections[target] = replace(decision, record=record)
        return record

    def registry_errors_snapshot(self) -> Tuple[BackendPluginError, ...]:
        return self.__registry_errors

    def replace_registry_errors(
        self,
        errors: Tuple[BackendPluginError, ...],
    ) -> None:
        self.__registry_errors = tuple(errors)

    def append_registry_error(self, error: BackendPluginError) -> None:
        if all(existing is not error for existing in self.__registry_errors):
            self.__registry_errors += (error,)

    @property
    def environment(self) -> Any:
        return self.__environment

    def cache_environment(self, environment: Any) -> None:
        self.__environment = environment

    @property
    def environment_error(self) -> Optional[BackendPluginError]:
        return self.__environment_error

    def cache_environment_error(self, error: BackendPluginError) -> None:
        self.__environment_error = error

    @property
    def discovered(self) -> bool:
        return self.__discovered

    def mark_discovered(self) -> None:
        self.__discovered = True

    def get_selection(self, target: str) -> Any:
        return self.__selections.get(target)

    def selections_snapshot(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(self.__selections.items())

    def set_selection(self, target: str, decision: Any) -> None:
        self.__selections[target] = decision

    def clear_selections(self) -> None:
        self.__selections.clear()

    @property
    def generation(self) -> int:
        return self.__generation

    def increment_generation(self) -> None:
        self.__generation += 1

    def is_loading(self, record_id: str) -> bool:
        return record_id in self.__loading

    def begin_loading(self, record_id: str) -> None:
        self.__loading.add(record_id)

    def end_loading(self, record_id: str) -> None:
        self.__loading.discard(record_id)

    def loading_snapshot(self) -> FrozenSet[str]:
        return frozenset(self.__loading)

    def is_registering(self, record_id: str) -> bool:
        return record_id in self.__registering

    def begin_registering(self, record_id: str) -> None:
        self.__registering.add(record_id)

    def end_registering(self, record_id: str) -> None:
        self.__registering.discard(record_id)

    def registering_snapshot(self) -> FrozenSet[str]:
        return frozenset(self.__registering)

    @property
    def resetting(self) -> bool:
        return self.__resetting

    def begin_reset(self) -> None:
        self.__resetting = True

    def end_reset(self) -> None:
        self.__resetting = False

    def add_reset_hook(self, callback: Callable[[], None]) -> None:
        if callback not in self.__reset_hooks:
            self.__reset_hooks.append(callback)

    def reset_hooks_snapshot(self) -> Tuple[Callable[[], None], ...]:
        return tuple(self.__reset_hooks)

    def clear_for_reset(self) -> None:
        """Clear one Registry epoch without callbacks or generation changes."""
        self.__records.clear()
        self.__registry_errors = ()
        self.__environment = None
        self.__environment_error = None
        self.__discovered = False
        self.__loading.clear()
        self.__registering.clear()
        self.clear_selections()
