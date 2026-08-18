"""Characterization contracts for the pre-Stage-4 Registry identity seams."""

from types import SimpleNamespace

import pytest

import triton_anchor.backends._registry_catalog as catalog_module
import triton_anchor.backends._registry_discovery as discovery_module
import triton_anchor.backends.registry as registry_module
from triton_anchor.backends._registry_catalog import RegistryCatalog
from triton_anchor.backends._registry_state import RegistryState
from triton_anchor.backends.errors import BackendPluginSelectionError
from triton_anchor.backends.protocol import (
    PluginCompatibilityStatus,
    PluginIsolationMode,
    PluginLifecycleState,
    PluginSource,
)
from triton_anchor.backends.registry import BackendPluginRecord
from triton_anchor.backends.selection import select_backend


class _Unstringifiable:
    def __str__(self):
        raise RuntimeError("string conversion failed")


class _MetadataGetFailure:
    def get(self, key):
        raise RuntimeError(f"cannot read {key}")


class _MetadataGetterFailure:
    name = "fallback-backend"
    version = "1.0"

    @property
    def metadata(self):
        raise RuntimeError("metadata getter failed")


class _NameGetterFailure:
    metadata = None
    version = "1.0"

    @property
    def name(self):
        raise RuntimeError("name getter failed")


class _VersionGetterFailure:
    metadata = {"Name": "versionless-backend"}

    @property
    def version(self):
        raise RuntimeError("version getter failed")


class _EntryPointGetterFailure:
    @property
    def name(self):
        raise RuntimeError("name getter failed")

    @property
    def value(self):
        raise RuntimeError("value getter failed")


def _entry_point(name="mock", value="vendor.mock:plugin"):
    return SimpleNamespace(
        name=name,
        value=value,
        group="triton.backends",
    )


def _distribution(name="vendor-backend", version="1.0"):
    return SimpleNamespace(
        metadata={"Name": name},
        name=f"fallback-{name}",
        version=version,
    )


def _manifest(entry_point, plugin_id, targets=()):
    return SimpleNamespace(
        entry_point=entry_point,
        plugin_id=plugin_id,
        targets=tuple(targets),
        capabilities=(),
        requires_capabilities=(),
        isolation_mode=PluginIsolationMode.PYTHON_ONLY,
        priority=0,
    )


def _catalog():
    state = RegistryState()
    return state, RegistryCatalog(
        state,
        record_factory=BackendPluginRecord,
    )


def _accept(catalog, distribution, entry_point, *, plugin_id):
    return catalog.accept_discovery_result(
        entry_point=entry_point,
        distribution=distribution,
        source=PluginSource.MANIFEST,
        manifest=_manifest(entry_point.name, plugin_id),
        state=PluginLifecycleState.DISCOVERED,
        compatibility_status=PluginCompatibilityStatus.NOT_CHECKED,
    )


def _selection_record(
    record_id,
    registry_key,
    plugin_id,
    entry_point_name,
):
    return SimpleNamespace(
        record_id=record_id,
        registry_key=registry_key,
        plugin_id=plugin_id,
        entry_point_name=entry_point_name,
        manifest=_manifest(entry_point_name, plugin_id, targets=("mock",)),
        source=PluginSource.MANIFEST,
        state=PluginLifecycleState.VALIDATED,
        compatibility_status=PluginCompatibilityStatus.COMPATIBLE,
        compatibility_report=None,
        error=None,
    )


def test_distribution_identity_prefers_metadata_name():
    distribution = SimpleNamespace(
        metadata={"Name": "Metadata_Backend"},
        name="attribute-backend",
        version=7,
    )

    assert catalog_module.distribution_identity(distribution) == (
        "Metadata_Backend",
        "7",
    )


def test_distribution_identity_metadata_get_failure_falls_back_to_name():
    distribution = SimpleNamespace(
        metadata=_MetadataGetFailure(),
        name="fallback-backend",
        version="2.0",
    )

    assert catalog_module.distribution_identity(distribution) == (
        "fallback-backend",
        "2.0",
    )


def test_distribution_identity_without_metadata_uses_name_and_version():
    distribution = SimpleNamespace(
        metadata=None,
        name="attribute-backend",
        version=3,
    )

    assert catalog_module.distribution_identity(distribution) == (
        "attribute-backend",
        "3",
    )


def test_distribution_identity_preserves_missing_version():
    distribution = SimpleNamespace(
        metadata={"Name": "versionless-backend"},
        name="ignored-backend",
        version=None,
    )

    assert catalog_module.distribution_identity(distribution) == (
        "versionless-backend",
        None,
    )


def test_distribution_identity_getter_failures_are_isolated():
    assert catalog_module.distribution_identity(_MetadataGetterFailure()) == (
        "fallback-backend",
        "1.0",
    )
    assert catalog_module.distribution_identity(_NameGetterFailure()) == (
        None,
        "1.0",
    )
    assert catalog_module.distribution_identity(_VersionGetterFailure()) == (
        "versionless-backend",
        None,
    )


def test_distribution_identity_stringification_failures_are_isolated():
    bad_name = SimpleNamespace(
        metadata={"Name": _Unstringifiable()},
        name="unused-fallback",
        version="1.0",
    )
    bad_version = SimpleNamespace(
        metadata={"Name": "vendor-backend"},
        version=_Unstringifiable(),
    )

    assert catalog_module.distribution_identity(bad_name) == (None, "1.0")
    assert catalog_module.distribution_identity(bad_version) == (
        "vendor-backend",
        None,
    )


def test_entry_point_name_none_normalizes_to_empty_string():
    assert catalog_module.entry_point_name(_entry_point(name=None)) == ""


def test_entry_point_value_none_normalizes_to_empty_string():
    assert catalog_module.entry_point_value(_entry_point(value=None)) == ""


def test_entry_point_getter_failures_normalize_to_empty_strings():
    entry_point = _EntryPointGetterFailure()

    assert catalog_module.entry_point_name(entry_point) == ""
    assert catalog_module.entry_point_value(entry_point) == ""


def test_entry_point_stringification_failures_normalize_to_empty_strings():
    entry_point = _entry_point(
        name=_Unstringifiable(),
        value=_Unstringifiable(),
    )

    assert catalog_module.entry_point_name(entry_point) == ""
    assert catalog_module.entry_point_value(entry_point) == ""


def test_record_ids_canonicalize_and_ignore_version_value_and_plugin_id():
    _, catalog = _catalog()
    records = (
        _accept(
            catalog,
            _distribution("Vendor_Backend", "1.0"),
            _entry_point(value="vendor.first:plugin"),
            plugin_id="vendor.first",
        ),
        _accept(
            catalog,
            _distribution("vendor-backend", "2.0"),
            _entry_point(value="vendor.second:plugin"),
            plugin_id="vendor.second",
        ),
        _accept(
            catalog,
            _distribution("VENDOR.BACKEND", "3.0"),
            _entry_point(value="vendor.third:plugin"),
            plugin_id="vendor.third",
        ),
    )

    assert tuple(record.record_id for record in records) == (
        "vendor-backend:mock",
        "vendor-backend:mock#2",
        "vendor-backend:mock#3",
    )
    assert tuple(record.distribution_version for record in records) == (
        "1.0",
        "2.0",
        "3.0",
    )
    assert tuple(record.entry_point_value for record in records) == (
        "vendor.first:plugin",
        "vendor.second:plugin",
        "vendor.third:plugin",
    )
    assert tuple(record.plugin_id for record in records) == (
        "vendor.first",
        "vendor.second",
        "vendor.third",
    )


def test_catalog_acceptance_preserves_original_object_identity():
    _, catalog = _catalog()
    distribution = _distribution()
    entry_point = _entry_point()
    manifest = _manifest(entry_point.name, "vendor.mock")

    record = catalog.accept_discovery_result(
        entry_point=entry_point,
        distribution=distribution,
        source=PluginSource.MANIFEST,
        manifest=manifest,
        state=PluginLifecycleState.DISCOVERED,
        compatibility_status=PluginCompatibilityStatus.NOT_CHECKED,
    )

    assert record.entry_point is entry_point
    assert record.distribution is distribution
    assert record.manifest is manifest
    assert catalog.list_snapshot()[0] is record


def test_catalog_exact_record_id_precedes_cross_alias_but_selection_rejects_it():
    state = RegistryState()
    catalog = RegistryCatalog(
        state,
        record_factory=lambda **fields: SimpleNamespace(**fields),
    )
    exact = _selection_record(
        "shared-selector",
        "vendor.exact",
        "vendor.exact",
        "exact",
    )
    alias = _selection_record(
        "alias-record",
        "shared-selector",
        "vendor.alias",
        "alias",
    )
    state.insert_record(exact)
    state.insert_record(alias)

    assert catalog.resolve("shared-selector") is exact

    with pytest.raises(BackendPluginSelectionError) as caught:
        select_backend(
            (alias, exact),
            target="mock",
            explicit_selector="shared-selector",
        )

    assert caught.value.field == "backend_selector"
    assert caught.value.expected == "one backend record"
    assert caught.value.actual == "alias-record, shared-selector"


def test_legacy_identity_import_aliases_and_registry_namespace_are_frozen():
    identity_names = (
        "DiscoveryRecordMetadata",
        "allocate_record_id",
        "distribution_identity",
        "entry_point_name",
        "entry_point_value",
        "record_metadata",
    )
    for name in identity_names:
        assert getattr(discovery_module, name) is getattr(catalog_module, name)

    assert registry_module._distribution_identity is (
        catalog_module.distribution_identity
    )
    assert registry_module._entry_point_name is catalog_module.entry_point_name
    assert registry_module._entry_point_value is catalog_module.entry_point_value

    assert tuple(
        sorted(name for name in vars(registry_module) if not name.startswith("_"))
    ) == (
        "Any",
        "BackendPluginCompatibilityError",
        "BackendPluginConflictError",
        "BackendPluginDiscoveryError",
        "BackendPluginError",
        "BackendPluginInterfaceError",
        "BackendPluginLifecycleError",
        "BackendPluginLoadError",
        "BackendPluginManifest",
        "BackendPluginManifestError",
        "BackendPluginRecord",
        "BackendPluginRegistry",
        "BackendPluginSelectionError",
        "Callable",
        "CapabilityReport",
        "CompatibilityReport",
        "ConflictReport",
        "CoreEnvironment",
        "Dict",
        "Iterable",
        "Mapping",
        "Optional",
        "PluginCompatibilityStatus",
        "PluginIsolationMode",
        "PluginLifecycleState",
        "PluginSource",
        "SelectionDecision",
        "Set",
        "Tag",
        "Tuple",
        "annotations",
        "backend_plugin_registry",
        "can_transition",
        "canonicalize_name",
        "collect_core_environment",
        "dataclass",
        "detect_conflicts",
        "evaluate_capabilities",
        "evaluate_record_preflight",
        "field",
        "get_backend_plugin_registry",
        "importlib",
        "load_distribution_manifest",
        "os",
        "replace",
        "select_backend",
        "threading",
        "validate_backend_plugin",
        "validate_plugin_capabilities",
        "validate_triton_version_requirement",
    )
