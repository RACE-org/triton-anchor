"""Focused contracts for the lock-free Registry catalog seam."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from triton_anchor.backends._registry_catalog import RegistryCatalog
from triton_anchor.backends._registry_discovery import discover_distributions
from triton_anchor.backends._registry_state import RegistryState
from triton_anchor.backends.errors import (
    BackendPluginConflictError,
    BackendPluginError,
    BackendPluginSelectionError,
)
from triton_anchor.backends.protocol import (
    PluginCompatibilityStatus,
    PluginIsolationMode,
    PluginLifecycleState,
    PluginSource,
)
from triton_anchor.backends.registry import BackendPluginRecord


@dataclass
class EntryPoint:
    name: str
    value: str
    group: str = "triton.backends"
    load_calls: int = 0

    def load(self):
        self.load_calls += 1
        raise AssertionError("catalog discovery must not import plugin code")


@dataclass
class Distribution:
    name: str
    version: str
    entry_points: tuple

    @property
    def metadata(self):
        return {"Name": self.name}


def _manifest(entry_point, plugin_id, targets=()):
    return SimpleNamespace(
        entry_point=entry_point,
        plugin_id=plugin_id,
        targets=tuple(targets),
        isolation_mode=PluginIsolationMode.PYTHON_ONLY,
    )


def _accept(
    catalog,
    distribution,
    entry_point,
    *,
    plugin_id,
    targets=(),
    error=None,
):
    return catalog.accept_discovery_result(
        entry_point=entry_point,
        distribution=distribution,
        source=PluginSource.MANIFEST,
        manifest=_manifest(entry_point.name, plugin_id, targets),
        state=(
            PluginLifecycleState.REJECTED
            if error is not None
            else PluginLifecycleState.DISCOVERED
        ),
        compatibility_status=PluginCompatibilityStatus.NOT_CHECKED,
        error=error,
    )


def _catalog():
    state = RegistryState()
    return state, RegistryCatalog(
        state,
        record_factory=BackendPluginRecord,
    )


def test_discovery_acceptance_allocates_identity_without_loading():
    state, catalog = _catalog()
    first_entry_point = EntryPoint("mock", "vendor.first:plugin")
    second_entry_point = EntryPoint("mock", "vendor.second:plugin")
    distribution = Distribution(
        "Vendor_Backend",
        "1.2.3",
        (first_entry_point, second_entry_point),
    )
    error = BackendPluginError("rejected metadata")

    first = _accept(
        catalog,
        distribution,
        first_entry_point,
        plugin_id="vendor.first",
    )
    snapshot = catalog.list_snapshot()
    second = _accept(
        catalog,
        distribution,
        second_entry_point,
        plugin_id="vendor.second",
        error=error,
    )

    assert first.record_id == "vendor-backend:mock"
    assert second.record_id == "vendor-backend:mock#2"
    assert first.distribution_name == "Vendor_Backend"
    assert first.distribution_version == "1.2.3"
    assert first.entry_point_value == "vendor.first:plugin"
    assert second.errors == (error,)
    assert snapshot == (first,)
    assert catalog.list_snapshot() == (first, second)
    assert state.records_snapshot() == (first, second)
    assert [first_entry_point.load_calls, second_entry_point.load_calls] == [0, 0]


def test_discovery_helper_and_catalog_preserve_stable_metadata_order():
    state, catalog = _catalog()
    zulu_entry_point = EntryPoint("mock", "zulu:plugin")
    alpha_zeta = EntryPoint("zeta", "alpha:zeta")
    alpha_first = EntryPoint("alpha", "alpha:first")
    zulu = Distribution("zulu-backend", "1.0", (zulu_entry_point,))
    alpha = Distribution(
        "Alpha_Backend",
        "2.0",
        (alpha_zeta, alpha_first),
    )
    documents = {
        id(alpha): SimpleNamespace(
            plugins=(
                _manifest("zeta", "vendor.zeta", ("zeta",)),
                _manifest("alpha", "vendor.alpha", ("alpha",)),
            )
        ),
        id(zulu): SimpleNamespace(
            plugins=(_manifest("mock", "vendor.mock", ("mock",)),)
        ),
    }

    discover_distributions(
        (zulu, alpha),
        store_record=catalog.accept_discovery_result,
        record_registry_error=lambda error: pytest.fail(str(error)),
        load_manifest=lambda distribution: documents[id(distribution)],
    )

    assert tuple(record.record_id for record in catalog.list_snapshot()) == (
        "alpha-backend:alpha",
        "alpha-backend:zeta",
        "zulu-backend:mock",
    )
    assert state.registry_errors_snapshot() == ()
    assert [
        entry_point.load_calls
        for entry_point in (zulu_entry_point, alpha_zeta, alpha_first)
    ] == [0, 0, 0]


def test_resolve_prefers_exact_record_id_and_preserves_error_fields():
    _, catalog = _catalog()
    distribution = Distribution("vendor", "1.0", ())
    alias_owner = _accept(
        catalog,
        distribution,
        EntryPoint("first", "vendor:first"),
        plugin_id="vendor:second",
    )
    exact_owner = _accept(
        catalog,
        distribution,
        EntryPoint("second", "vendor:second"),
        plugin_id="vendor.exact",
    )

    assert alias_owner.registry_key == exact_owner.record_id
    assert catalog.resolve(exact_owner.record_id) is exact_owner
    assert catalog.inspect_snapshot("vendor.exact") is exact_owner

    with pytest.raises(BackendPluginSelectionError) as caught:
        catalog.resolve("missing")
    assert caught.value.to_dict() == {
        "code": "backend_plugin_selection_error",
        "message": "Unknown backend plugin record 'missing'",
        "plugin_id": None,
        "entry_point": None,
        "detail": None,
        "field": "registry_key",
        "expected": "an existing record_id or unique registry_key",
        "actual": "missing",
        "remediation": (
            "Call registry.list() and use one of the reported record_id or "
            "registry_key values."
        ),
    }


def test_ambiguous_key_and_conflict_view_keep_deterministic_record_order():
    _, catalog = _catalog()
    distribution = Distribution("vendor", "1.0", ())
    zeta = _accept(
        catalog,
        distribution,
        EntryPoint("zeta", "vendor:zeta"),
        plugin_id="vendor.shared",
        targets=("mock",),
    )
    alpha = _accept(
        catalog,
        distribution,
        EntryPoint("alpha", "vendor:alpha"),
        plugin_id="vendor.shared",
        targets=("mock",),
    )

    with pytest.raises(BackendPluginConflictError) as caught:
        catalog.resolve("vendor.shared")
    assert caught.value.to_dict() == {
        "code": "backend_plugin_conflict_error",
        "message": (
            "Backend plugin key 'vendor.shared' is ambiguous across records: "
            "vendor:zeta, vendor:alpha"
        ),
        "plugin_id": "vendor.shared",
        "entry_point": None,
        "detail": None,
        "field": "registry_key",
        "expected": "a unique plugin identity",
        "actual": "vendor:zeta, vendor:alpha",
        "remediation": (
            "Use a record_id for inspection now; W7 will report and govern "
            "the underlying identity conflict."
        ),
    }

    report = catalog.conflict_view()
    assert tuple(conflict.kind.value for conflict in report.conflicts) == (
        "duplicate_plugin_id",
        "target_overlap",
    )
    assert report.conflicts[0].record_ids == tuple(
        sorted((zeta.record_id, alpha.record_id))
    )


def test_conflict_view_propagates_malformed_record_contract():
    state, catalog = _catalog()
    malformed = SimpleNamespace(record_id="")
    state.insert_record(malformed)

    with pytest.raises(TypeError, match="non-empty string record_id"):
        catalog.conflict_view()
