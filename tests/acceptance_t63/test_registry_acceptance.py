"""Independent acceptance tests for the T6.3 backend Registry contracts.

The test oracles in this module come from the packaged Manifest Schema 1.0,
the public ``triton_anchor.backends`` API/docstrings, and the lifecycle and
selection rules exposed by those public types.  The fixtures deliberately do
not reuse Registry internals or the excluded T10.2 conformance kit.
"""

from __future__ import annotations

import copy
import importlib.metadata
import itertools
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional

import pytest
from packaging.tags import Tag

from triton_anchor.backends import (
    BACKEND_SELECTOR_ENV,
    BackendPluginBase,
    BackendPluginCapabilityError,
    BackendPluginCompatibilityError,
    BackendPluginConflictError,
    BackendPluginInterfaceError,
    BackendPluginLifecycleError,
    BackendPluginLoadError,
    BackendPluginManifestError,
    BackendPluginProtocolError,
    BackendPluginRegistry,
    BackendPluginSelectionError,
    CoreEnvironment,
    PluginCompatibilityStatus,
    PluginIsolationMode,
    PluginLifecycleState,
    PluginSource,
    ProtocolFieldStatus,
    SelectionMethod,
    consume_protocol_field,
    evaluate_protocol_field_removal,
    load_manifest,
    parse_manifest,
    select_backend,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "python/triton_anchor/backends/schemas/backend_manifest.schema.json"
)
MANIFEST_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
SCHEMA_ROOT_REQUIRED = tuple(MANIFEST_SCHEMA["required"])
SCHEMA_PLUGIN_REQUIRED = tuple(
    MANIFEST_SCHEMA["$defs"]["plugin"]["required"]
)

TRITON_COMMIT = "6cc4505027d7b39fe18a44a7f89085b8babb7400"
LLVM_COMMIT = "a992f29451b9e140424f35ac5e20177db4afbdc0"
OTHER_COMMIT = "b" * 40
ABI_FINGERPRINT = "sha256:" + "a" * 64
UNIVERSAL_TAG = Tag("py3", "none", "any")
WHEEL_METADATA = "Wheel-Version: 1.0\nTag: py3-none-any\n"


class Compiler:
    pass


class Driver:
    pass


class FakeEntryPoint:
    """Entry point whose load call is also an import-side-effect sentinel."""

    group = "triton.backends"

    def __init__(
        self,
        name: str,
        loaded_value: Any,
        *,
        value: Optional[str] = None,
        marker: Optional[Path] = None,
        load_error: Optional[BaseException] = None,
    ) -> None:
        self.name = name
        self.value = value or f"fixture_{name}:plugin"
        self.loaded_value = loaded_value
        self.marker = marker
        self.load_error = load_error
        self.load_calls = 0
        self.dist = None

    def load(self) -> Any:
        self.load_calls += 1
        if self.marker is not None:
            self.marker.write_text("plugin module imported", encoding="utf-8")
        if self.load_error is not None:
            raise self.load_error
        return self.loaded_value


class FakeDistribution:
    """Installed-wheel metadata double; plugin Python is never inspected."""

    def __init__(
        self,
        *,
        name: str,
        entry_points: Iterable[FakeEntryPoint],
        manifest_path: Optional[Path],
        files: Optional[Iterable[str]] = None,
        version: str = "1.0.0",
        wheel_metadata: Optional[str] = WHEEL_METADATA,
    ) -> None:
        self.metadata = {"Name": name}
        self.name = name
        self.version = version
        self.entry_points = tuple(entry_points)
        for entry_point in self.entry_points:
            entry_point.dist = self
        self._manifest_path = manifest_path
        if files is None:
            self.files = (
                ("fixture/triton_anchor_backend.json",)
                if manifest_path is not None
                else ("fixture/__init__.py",)
            )
        else:
            self.files = None if files is False else tuple(files)
        self._wheel_metadata = wheel_metadata

    def locate_file(self, _file: Any) -> Path:
        if self._manifest_path is None:
            raise FileNotFoundError("distribution has no Manifest")
        return self._manifest_path

    def read_text(self, filename: str) -> Optional[str]:
        if filename == "WHEEL":
            return self._wheel_metadata
        return None


def good_plugin(**overrides: Any) -> dict[str, Any]:
    plugin = {
        "plugin_id": "vendor.alpha",
        "entry_point": "alpha",
        "backend_protocol": ">=1.0,<2.0",
        "requires_core": ">=0.2,<0.3",
        "requires_triton": {
            "version": ">=3.6,<3.7",
            "commit": TRITON_COMMIT,
        },
        "targets": ["alpha"],
        "capabilities": ["fixture.compile"],
        "isolation_mode": "python_only",
        "priority": 0,
    }
    plugin.update(copy.deepcopy(overrides))
    return plugin


def good_manifest(
    plugins: Optional[Iterable[Mapping[str, Any]]] = None,
    **root_overrides: Any,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "plugins": list(plugins) if plugins is not None else [good_plugin()],
    }
    document.update(copy.deepcopy(root_overrides))
    return document


def good_environment(**overrides: Any) -> CoreEnvironment:
    values = {
        "core_version": "0.2.0",
        "build_info_generated": True,
        "backend_protocol_version": "1.0",
        "manifest_schema_version": "1.0",
        "triton_version": "3.6.0",
        "vendored_triton_commit": TRITON_COMMIT,
        "expected_llvm_project_commit": LLVM_COMMIT,
        "actual_llvm_version_raw": "22.0.0git",
        "actual_llvm_version": "22.0.0",
        "actual_llvm_version_suffix": "git",
        "actual_llvm_commit": LLVM_COMMIT,
        "actual_mlir_version_raw": "22.0.0git",
        "actual_mlir_version": "22.0.0",
        "actual_mlir_version_suffix": "git",
        "actual_mlir_commit": LLVM_COMMIT,
        "cxx_standard": "17",
        "cxx_compiler_id": "GNU",
        "cxx_compiler_version": "13.3.0",
        "cxx11_abi": "1",
        "build_type": "Release",
        "ttgpu": True,
        "built_python_version": "3.12.3",
        "built_python_soabi": "cpython-fixture",
        "built_platform": "fixture-platform",
        "core_abi_fingerprint_schema": "triton-anchor-core-abi-v1",
        "core_library_sha256": "sha256:" + "c" * 64,
        "core_abi_fingerprint": ABI_FINGERPRINT,
        "runtime_python_version": "3.12.3",
        "runtime_python_implementation": "CPython",
        "runtime_python_soabi": "cpython-fixture",
        "runtime_platform": "fixture-platform",
        "runtime_system": "Linux",
        "runtime_machine": "x86_64",
    }
    values.update(overrides)
    return CoreEnvironment(**values)


def structural_plugin(label: str = "fixture") -> Any:
    compiler_cls = type(f"{label.title()}Compiler", (), {"plugin_label": label})
    driver_cls = type(f"{label.title()}Driver", (), {"plugin_label": label})
    return SimpleNamespace(compiler_cls=compiler_cls, driver_cls=driver_cls)


def make_distribution(
    directory: Path,
    *,
    name: str = "fixture-alpha",
    plugins: Optional[Iterable[Mapping[str, Any]]] = None,
    entry_points: Optional[Iterable[FakeEntryPoint]] = None,
    manifest_data: Optional[Mapping[str, Any]] = None,
    manifest_text: Optional[str] = None,
    files: Optional[Iterable[str]] = None,
    write_manifest_file: bool = True,
) -> FakeDistribution:
    directory.mkdir(parents=True, exist_ok=True)
    plugin_list = list(plugins) if plugins is not None else [good_plugin()]
    if entry_points is None:
        entry_points = [
            FakeEntryPoint(
                plugin["entry_point"], structural_plugin(plugin["plugin_id"])
            )
            for plugin in plugin_list
        ]
    manifest_path = None
    manifest_path = directory / "triton_anchor_backend.json"
    if write_manifest_file:
        if manifest_text is None:
            data = (
                manifest_data
                if manifest_data is not None
                else good_manifest(plugin_list)
            )
            manifest_text = json.dumps(data)
        manifest_path.write_text(manifest_text, encoding="utf-8")
    return FakeDistribution(
        name=name,
        entry_points=entry_points,
        manifest_path=manifest_path,
        files=files,
    )


def registry_for(
    distributions: Iterable[Any],
    *,
    environment: Optional[CoreEnvironment] = None,
    core_capabilities: Iterable[str] = (),
    environment_provider: Optional[Any] = None,
) -> BackendPluginRegistry:
    dist_tuple = tuple(distributions)
    return BackendPluginRegistry(
        distribution_provider=lambda: dist_tuple,
        environment_provider=(
            environment_provider
            if environment_provider is not None
            else lambda: environment or good_environment()
        ),
        supported_tags=(UNIVERSAL_TAG,),
        core_capabilities=core_capabilities,
    )


def one_record(registry: BackendPluginRegistry):
    records = registry.list()
    assert len(records) == 1
    return records[0]


def assert_structured_error(
    error: Any,
    *,
    plugin_id: Optional[str] = None,
    field_contains: Optional[str] = None,
) -> dict[str, Any]:
    diagnostic = error.to_dict()
    assert diagnostic["code"].startswith("backend_plugin_")
    assert diagnostic["message"]
    if plugin_id is not None:
        assert diagnostic["plugin_id"] == plugin_id
    if field_contains is not None:
        assert field_contains in diagnostic["field"]
    assert diagnostic.get("expected")
    assert diagnostic.get("actual")
    return diagnostic


# ---------------------------------------------------------------------------
# Manifest Schema 1.0 and metadata-only discovery
# ---------------------------------------------------------------------------


def test_registry_discovers_no_plugins_without_collecting_environment() -> None:
    environment_calls = 0

    def unexpected_environment() -> CoreEnvironment:
        nonlocal environment_calls
        environment_calls += 1
        raise AssertionError("empty discovery must not collect the environment")

    registry = registry_for([], environment_provider=unexpected_environment)
    assert registry.discover() == ()
    assert registry.list() == ()
    assert registry.diagnostics()["plugins"] == []
    assert environment_calls == 0


def test_registry_discovers_one_valid_manifest_without_loading(tmp_path: Path) -> None:
    marker = tmp_path / "imported"
    entry_point = FakeEntryPoint(
        "alpha", structural_plugin("alpha"), marker=marker
    )
    distribution = make_distribution(
        tmp_path / "dist", entry_points=[entry_point]
    )
    registry = registry_for([distribution])

    record = one_record(registry)
    assert record.plugin_id == "vendor.alpha"
    assert record.source is PluginSource.MANIFEST
    assert record.state is PluginLifecycleState.DISCOVERED
    assert entry_point.load_calls == 0
    assert not marker.exists()


def test_registry_discovers_two_independent_plugins_deterministically(
    tmp_path: Path,
) -> None:
    alpha = good_plugin()
    beta = good_plugin(
        plugin_id="vendor.beta", entry_point="beta", targets=["beta"]
    )
    alpha_ep = FakeEntryPoint("alpha", structural_plugin("alpha"))
    beta_ep = FakeEntryPoint("beta", structural_plugin("beta"))
    alpha_dist = make_distribution(
        tmp_path / "z", name="z-alpha", plugins=[alpha], entry_points=[alpha_ep]
    )
    beta_dist = make_distribution(
        tmp_path / "a", name="a-beta", plugins=[beta], entry_points=[beta_ep]
    )

    registry = registry_for([alpha_dist, beta_dist])
    records = registry.discover()
    assert [record.distribution_name for record in records] == ["a-beta", "z-alpha"]
    assert registry.conflicts().conflicts == ()
    assert alpha_ep.load_calls == beta_ep.load_calls == 0


@pytest.mark.parametrize("field", SCHEMA_ROOT_REQUIRED)
def test_registry_schema_rejects_each_missing_root_required_field(field: str) -> None:
    document = good_manifest()
    del document[field]
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(document)
    assert caught.value.field == field


@pytest.mark.parametrize("field", SCHEMA_PLUGIN_REQUIRED)
def test_registry_schema_rejects_each_missing_plugin_required_field(field: str) -> None:
    document = good_manifest()
    del document["plugins"][0][field]
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(document)
    assert caught.value.field == field


@pytest.mark.parametrize("root", [None, [], "manifest", 1, True])
def test_registry_schema_rejects_non_object_roots(root: Any) -> None:
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(root)
    assert caught.value.field == "<root>"


@pytest.mark.parametrize("plugins", [None, {}, "plugin", [], [None], ["bad"]])
def test_registry_schema_rejects_wrong_plugins_types(plugins: Any) -> None:
    with pytest.raises(BackendPluginManifestError):
        parse_manifest({"schema_version": "1.0", "plugins": plugins})


@pytest.mark.parametrize("schema_version", ["0.9", "2.0", "1", "v1.0", 1, None])
def test_registry_rejects_wrong_manifest_schema_versions(schema_version: Any) -> None:
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(good_manifest(schema_version=schema_version))
    assert caught.value.field == "schema_version"


@pytest.mark.parametrize("schema_version", ["1.0", "1.1", "1.999"])
def test_registry_accepts_schema_compatible_minor_versions(schema_version: str) -> None:
    document = parse_manifest(good_manifest(schema_version=schema_version))
    assert document.schema_version == schema_version


def test_registry_preserves_schema_allowed_unknown_fields() -> None:
    plugin = good_plugin(
        future_optional={"enabled": True},
        requires_triton={
            "version": ">=3.6,<3.7",
            "commit": TRITON_COMMIT,
            "future_triton_field": 17,
        },
    )
    document = parse_manifest(
        good_manifest([plugin], future_root_field=["preserve-me"])
    )
    assert document.extensions == {"future_root_field": ["preserve-me"]}
    assert document.plugins[0].extensions == {
        "future_optional": {"enabled": True}
    }
    assert document.plugins[0].requires_triton.extensions == {
        "future_triton_field": 17
    }


def test_registry_rejects_duplicate_json_member_names(tmp_path: Path) -> None:
    path = tmp_path / "triton_anchor_backend.json"
    path.write_text(
        '{"schema_version":"1.0","schema_version":"1.1","plugins":[]}',
        encoding="utf-8",
    )
    with pytest.raises(BackendPluginManifestError) as caught:
        load_manifest(path)
    assert caught.value.field == "schema_version"


@pytest.mark.parametrize(
    "manifest_text",
    ["{", "not JSON", "[1, 2", b"\xff"],
)
def test_registry_discovery_rejects_malformed_manifest_without_import(
    tmp_path: Path, manifest_text: Any
) -> None:
    marker = tmp_path / "imported"
    entry_point = FakeEntryPoint(
        "alpha", structural_plugin(), marker=marker
    )
    manifest_path = tmp_path / "dist/triton_anchor_backend.json"
    manifest_path.parent.mkdir(parents=True)
    if isinstance(manifest_text, bytes):
        manifest_path.write_bytes(manifest_text)
    else:
        manifest_path.write_text(manifest_text, encoding="utf-8")
    distribution = FakeDistribution(
        name="malformed", entry_points=[entry_point], manifest_path=manifest_path
    )
    registry = registry_for([distribution])

    record = one_record(registry)
    assert record.state is PluginLifecycleState.REJECTED
    assert isinstance(record.error, BackendPluginManifestError)
    assert entry_point.load_calls == 0
    assert not marker.exists()
    with pytest.raises(BackendPluginManifestError):
        registry.discover(strict=True)


def test_registry_rejects_recorded_but_missing_manifest_file_without_import(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "imported"
    entry_point = FakeEntryPoint("alpha", structural_plugin(), marker=marker)
    distribution = make_distribution(
        tmp_path / "missing",
        entry_points=[entry_point],
        write_manifest_file=False,
    )
    record = one_record(registry_for([distribution]))
    assert record.state is PluginLifecycleState.REJECTED
    assert isinstance(record.error, BackendPluginManifestError)
    assert entry_point.load_calls == 0
    assert not marker.exists()


def test_registry_manifest_must_exactly_cover_distribution_entry_points(
    tmp_path: Path,
) -> None:
    alpha_ep = FakeEntryPoint("alpha", structural_plugin())
    beta_ep = FakeEntryPoint("beta", structural_plugin())
    distribution = make_distribution(
        tmp_path / "partial",
        plugins=[good_plugin()],
        entry_points=[alpha_ep, beta_ep],
    )
    records = registry_for([distribution]).discover()
    assert len(records) == 2
    assert all(record.state is PluginLifecycleState.REJECTED for record in records)
    assert all(record.error.field == "plugins[].entry_point" for record in records)
    assert alpha_ep.load_calls == beta_ep.load_calls == 0


def test_registry_rejects_manifest_entry_point_absent_from_distribution(
    tmp_path: Path,
) -> None:
    """Exercise the opposite coverage direction: declared but not installed."""
    marker = tmp_path / "must-not-import"
    installed_ep = FakeEntryPoint(
        "beta", structural_plugin("beta"), marker=marker
    )
    distribution = make_distribution(
        tmp_path / "unknown-manifest-entry-point",
        plugins=[good_plugin(entry_point="alpha")],
        entry_points=[installed_ep],
    )

    record = one_record(registry_for([distribution]))
    assert record.state is PluginLifecycleState.REJECTED
    assert isinstance(record.error, BackendPluginManifestError)
    diagnostic = assert_structured_error(
        record.error, field_contains="plugins[].entry_point"
    )
    assert diagnostic["expected"] == "beta"
    assert diagnostic["actual"] == "alpha"
    assert "unknown records for alpha" in diagnostic["message"]
    assert installed_ep.load_calls == 0
    assert not marker.exists()


def test_registry_rejects_duplicate_plugin_ids_inside_one_manifest() -> None:
    first = good_plugin()
    second = good_plugin(entry_point="beta", targets=["beta"])
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(good_manifest([first, second]))
    assert caught.value.field == "plugins[].plugin_id"


def test_registry_rejects_duplicate_entry_points_inside_one_manifest() -> None:
    first = good_plugin()
    second = good_plugin(plugin_id="vendor.beta", targets=["beta"])
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(good_manifest([first, second]))
    assert caught.value.field == "plugins[].entry_point"


@pytest.mark.parametrize(
    "field,value",
    [
        ("capabilities", "compile"),
        ("capabilities", [""]),
        ("capabilities", [" compile"]),
        ("capabilities", ["compile", "compile"]),
        ("requires_capabilities", [1]),
        ("targets", []),
        ("targets", ["alpha", "alpha"]),
    ],
)
def test_registry_rejects_malformed_capability_and_target_arrays(
    field: str, value: Any
) -> None:
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(good_manifest([good_plugin(**{field: value})]))
    assert caught.value.field == field


@pytest.mark.parametrize("priority", [-(2**63), 0, 2**63 - 1])
def test_registry_accepts_schema_unbounded_integer_priority_examples(priority: int) -> None:
    plugin = parse_manifest(good_manifest([good_plugin(priority=priority)])).plugins[0]
    assert plugin.priority == priority


@pytest.mark.parametrize("priority", [True, False, 1.5, "1", None, [], {}])
def test_registry_rejects_non_integer_priority(priority: Any) -> None:
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(good_manifest([good_plugin(priority=priority)]))
    assert caught.value.field == "priority"


def test_registry_accepts_all_schema_isolation_modes() -> None:
    python_only = parse_manifest(good_manifest()).plugins[0]
    subprocess = parse_manifest(
        good_manifest(
            [
                good_plugin(
                    isolation_mode="subprocess",
                    native_libraries=["vendor/worker.so"],
                )
            ]
        )
    ).plugins[0]
    native = parse_manifest(
        good_manifest(
            [
                good_plugin(
                    isolation_mode="native_in_process",
                    native_libraries=["vendor/libbackend.so"],
                    abi_fingerprint=ABI_FINGERPRINT,
                )
            ]
        )
    ).plugins[0]
    assert {python_only.isolation_mode, subprocess.isolation_mode, native.isolation_mode} == {
        PluginIsolationMode.PYTHON_ONLY,
        PluginIsolationMode.SUBPROCESS,
        PluginIsolationMode.NATIVE_IN_PROCESS,
    }


@pytest.mark.parametrize(
    "overrides,field",
    [
        ({"isolation_mode": "thread"}, "isolation_mode"),
        (
            {"isolation_mode": "python_only", "native_libraries": ["x.so"]},
            "native_libraries",
        ),
        (
            {"isolation_mode": "python_only", "abi_fingerprint": ABI_FINGERPRINT},
            "abi_fingerprint",
        ),
        (
            {"isolation_mode": "native_in_process", "native_libraries": ["x.so"]},
            "abi_fingerprint",
        ),
        (
            {"isolation_mode": "native_in_process", "abi_fingerprint": ABI_FINGERPRINT},
            "native_libraries",
        ),
        (
            {"isolation_mode": "subprocess", "abi_fingerprint": ABI_FINGERPRINT},
            "abi_fingerprint",
        ),
    ],
)
def test_registry_rejects_illegal_isolation_combinations(
    overrides: dict[str, Any], field: str
) -> None:
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(good_manifest([good_plugin(**overrides)]))
    assert caught.value.field == field


def test_registry_discovery_query_and_validation_do_not_import_plugin(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "import-side-effect"
    entry_point = FakeEntryPoint(
        "alpha", structural_plugin(), marker=marker
    )
    distribution = make_distribution(
        tmp_path / "dist", entry_points=[entry_point]
    )
    registry = registry_for([distribution])

    discovered = registry.discover()[0]
    registry.list()
    registry.inspect(discovered.registry_key)
    registry.conflicts()
    registry.diagnostics()
    validated = registry.validate(discovered.registry_key)
    registry.diagnostics(discovered.registry_key)

    assert validated.state is PluginLifecycleState.VALIDATED
    assert entry_point.load_calls == 0
    assert not marker.exists()


def test_registry_incompatible_plugin_is_rejected_before_import(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-import"
    entry_point = FakeEntryPoint("alpha", structural_plugin(), marker=marker)
    distribution = make_distribution(
        tmp_path / "dist",
        plugins=[good_plugin(requires_triton={"version": ">=3.7,<3.8"})],
        entry_points=[entry_point],
    )
    registry = registry_for([distribution])
    record_id = one_record(registry).record_id

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.load(record_id)
    assert_structured_error(
        caught.value, plugin_id="vendor.alpha", field_contains="Triton version"
    )
    assert entry_point.load_calls == 0
    assert not marker.exists()


def _write_real_metadata_distribution(
    site: Path,
    *,
    distribution_name: str,
    module_name: str,
    plugin_id: str,
    entry_point_name: str,
    target: str,
    marker: Path,
) -> None:
    package = site / module_name
    dist_info = site / f"{distribution_name.replace('-', '_')}-1.0.0.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        "class Compiler: pass\n"
        "class Driver: pass\n"
        "class Plugin:\n"
        "    compiler_cls = Compiler\n"
        "    driver_cls = Driver\n"
        "plugin = Plugin()\n",
        encoding="utf-8",
    )
    manifest = good_manifest(
        [
            good_plugin(
                plugin_id=plugin_id,
                entry_point=entry_point_name,
                targets=[target],
            )
        ]
    )
    (package / "triton_anchor_backend.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution_name}\nVersion: 1.0.0\n",
        encoding="utf-8",
    )
    (dist_info / "WHEEL").write_text(WHEEL_METADATA, encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(
        "[triton.backends]\n"
        f"{entry_point_name} = {module_name}:plugin\n",
        encoding="utf-8",
    )
    record_paths = [
        f"{module_name}/__init__.py",
        f"{module_name}/triton_anchor_backend.json",
        f"{dist_info.name}/METADATA",
        f"{dist_info.name}/WHEEL",
        f"{dist_info.name}/entry_points.txt",
        f"{dist_info.name}/RECORD",
    ]
    (dist_info / "RECORD").write_text(
        "".join(f"{path},,\n" for path in record_paths), encoding="utf-8"
    )


def test_registry_real_importlib_metadata_discovery_of_two_distributions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real PathDistribution/EntryPoint objects, not metadata mocks."""

    site = tmp_path / "site-packages"
    site.mkdir()
    alpha_marker = tmp_path / "alpha-imported"
    beta_marker = tmp_path / "beta-imported"
    suffix = tmp_path.name.replace("-", "_")
    alpha_module = f"registry_real_alpha_{suffix}"
    beta_module = f"registry_real_beta_{suffix}"
    _write_real_metadata_distribution(
        site,
        distribution_name=f"registry-real-alpha-{suffix}",
        module_name=alpha_module,
        plugin_id=f"vendor.real.alpha.{suffix}",
        entry_point_name="real_alpha",
        target="real_alpha",
        marker=alpha_marker,
    )
    _write_real_metadata_distribution(
        site,
        distribution_name=f"registry-real-beta-{suffix}",
        module_name=beta_module,
        plugin_id=f"vendor.real.beta.{suffix}",
        entry_point_name="real_beta",
        target="real_beta",
        marker=beta_marker,
    )
    monkeypatch.syspath_prepend(str(site))
    distributions = tuple(importlib.metadata.distributions(path=[str(site)]))
    assert len(distributions) == 2
    registry = registry_for(distributions)

    records = registry.discover()
    assert len(records) == 2
    assert all(type(record.distribution).__name__ == "PathDistribution" for record in records)
    assert not alpha_marker.exists() and not beta_marker.exists()
    registry.validate(strict=True)
    assert not alpha_marker.exists() and not beta_marker.exists()

    alpha_decision = registry.select("real_alpha")
    assert alpha_decision.plugin_id and ".alpha." in alpha_decision.plugin_id
    assert alpha_marker.exists() and not beta_marker.exists()
    beta_decision = registry.select("real_beta")
    assert beta_decision.plugin_id and ".beta." in beta_decision.plugin_id
    assert beta_marker.exists()
    sys.modules.pop(alpha_module, None)
    sys.modules.pop(beta_module, None)


# ---------------------------------------------------------------------------
# Protocol, environment-version, and capability negotiation
# ---------------------------------------------------------------------------


def _validation_fixture(
    tmp_path: Path,
    *,
    plugin_overrides: Optional[Mapping[str, Any]] = None,
    environment_overrides: Optional[Mapping[str, Any]] = None,
    core_capabilities: Iterable[str] = (),
) -> tuple[BackendPluginRegistry, FakeEntryPoint]:
    plugin = good_plugin(**dict(plugin_overrides or {}))
    entry_point = FakeEntryPoint("alpha", structural_plugin())
    distribution = make_distribution(
        tmp_path,
        plugins=[plugin],
        entry_points=[entry_point],
    )
    registry = registry_for(
        [distribution],
        environment=good_environment(**dict(environment_overrides or {})),
        core_capabilities=core_capabilities,
    )
    return registry, entry_point


def test_registry_accepts_compatible_protocol_core_triton_and_llvm(
    tmp_path: Path,
) -> None:
    registry, entry_point = _validation_fixture(
        tmp_path,
        plugin_overrides={
            "backend_protocol": ">=1.0,<2.0",
            "requires_core": ">=0.2,<0.3",
            "requires_triton": {
                "version": ">=3.6,<3.7",
                "commit": TRITON_COMMIT.upper(),
            },
            "requires_llvm_version": ">=22,<23",
            "requires_llvm_commit": LLVM_COMMIT.upper(),
        },
    )
    record = registry.validate(one_record(registry).record_id)
    dimensions = {check.dimension for check in record.compatibility_report.checks}
    assert {
        "Backend Plugin Protocol",
        "triton-anchor Core version",
        "Triton version",
        "vendored Triton commit",
        "LLVM version",
        "LLVM commit",
    } <= dimensions
    assert entry_point.load_calls == 0


@pytest.mark.parametrize("protocol", ["<1.0", ">=2.0"])
def test_registry_rejects_protocol_too_old_or_too_new_before_import(
    tmp_path: Path, protocol: str
) -> None:
    registry, entry_point = _validation_fixture(
        tmp_path, plugin_overrides={"backend_protocol": protocol}
    )
    with pytest.raises(BackendPluginProtocolError) as caught:
        registry.validate(one_record(registry).record_id)
    diagnostic = assert_structured_error(
        caught.value, plugin_id="vendor.alpha", field_contains="backend_protocol"
    )
    assert diagnostic["actual"] == "1.0"
    assert diagnostic["expected"] == protocol
    assert entry_point.load_calls == 0


@pytest.mark.parametrize("protocol", ["not-a-specifier", ">=>1", "", 1])
def test_registry_rejects_malformed_protocol_constraint(protocol: Any) -> None:
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(good_manifest([good_plugin(backend_protocol=protocol)]))
    assert caught.value.field == "backend_protocol"


@pytest.mark.parametrize(
    "core_requirement,compatible",
    [(">=0.2,<0.3", True), (">=0.1,<0.2", False), (">=0.3,<0.4", False)],
)
def test_registry_negotiates_core_minor_compatibility_range(
    tmp_path: Path, core_requirement: str, compatible: bool
) -> None:
    registry, entry_point = _validation_fixture(
        tmp_path, plugin_overrides={"requires_core": core_requirement}
    )
    record_id = one_record(registry).record_id
    if compatible:
        assert registry.validate(record_id).compatibility_status is PluginCompatibilityStatus.COMPATIBLE
    else:
        with pytest.raises(BackendPluginCompatibilityError) as caught:
            registry.validate(record_id)
        assert_structured_error(
            caught.value,
            plugin_id="vendor.alpha",
            field_contains="triton-anchor Core version",
        )
    assert entry_point.load_calls == 0


@pytest.mark.parametrize(
    "core_version,compatible",
    [("0.2.0", True), ("0.2.99", True), ("0.1.9", False), ("0.3.0", False)],
)
def test_registry_core_02_range_accepts_only_actual_02_versions(
    tmp_path: Path, core_version: str, compatible: bool
) -> None:
    registry, entry_point = _validation_fixture(
        tmp_path,
        plugin_overrides={"requires_core": ">=0.2,<0.3"},
        environment_overrides={"core_version": core_version},
    )
    record_id = one_record(registry).record_id
    if compatible:
        registry.validate(record_id)
    else:
        with pytest.raises(BackendPluginCompatibilityError) as caught:
            registry.validate(record_id)
        diagnostic = assert_structured_error(
            caught.value,
            plugin_id="vendor.alpha",
            field_contains="triton-anchor Core version",
        )
        assert diagnostic["actual"] == core_version
    assert entry_point.load_calls == 0


@pytest.mark.parametrize(
    "triton_requirement,compatible",
    [(">=3.6,<3.7", True), (">=3.5,<3.6", False), (">=3.7,<3.8", False)],
)
def test_registry_negotiates_triton_36_range(
    tmp_path: Path, triton_requirement: str, compatible: bool
) -> None:
    registry, entry_point = _validation_fixture(
        tmp_path,
        plugin_overrides={"requires_triton": {"version": triton_requirement}},
    )
    record_id = one_record(registry).record_id
    if compatible:
        registry.validate(record_id)
    else:
        with pytest.raises(BackendPluginCompatibilityError) as caught:
            registry.validate(record_id)
        diagnostic = assert_structured_error(
            caught.value, plugin_id="vendor.alpha", field_contains="Triton version"
        )
        assert diagnostic["actual"] == "3.6.0"
    assert entry_point.load_calls == 0


@pytest.mark.parametrize(
    "triton_version,compatible",
    [("3.6.0", True), ("3.6.99", True), ("3.5.9", False), ("3.7.0", False)],
)
def test_registry_triton_36_range_rejects_adjacent_minor_versions(
    tmp_path: Path, triton_version: str, compatible: bool
) -> None:
    registry, entry_point = _validation_fixture(
        tmp_path,
        plugin_overrides={"requires_triton": {"version": ">=3.6,<3.7"}},
        environment_overrides={"triton_version": triton_version},
    )
    record_id = one_record(registry).record_id
    if compatible:
        registry.validate(record_id)
    else:
        with pytest.raises(BackendPluginCompatibilityError) as caught:
            registry.validate(record_id)
        diagnostic = assert_structured_error(
            caught.value, plugin_id="vendor.alpha", field_contains="Triton version"
        )
        assert diagnostic["actual"] == triton_version
    assert entry_point.load_calls == 0


def test_registry_llvm_commit_match_mismatch_missing_and_unavailable(
    tmp_path: Path,
) -> None:
    matching, _ = _validation_fixture(
        tmp_path / "matching",
        plugin_overrides={"requires_llvm_commit": LLVM_COMMIT},
    )
    matching.validate(one_record(matching).record_id)

    mismatch, mismatch_ep = _validation_fixture(
        tmp_path / "mismatch",
        plugin_overrides={"requires_llvm_commit": LLVM_COMMIT},
        environment_overrides={"actual_llvm_commit": OTHER_COMMIT},
    )
    with pytest.raises(BackendPluginCompatibilityError) as caught:
        mismatch.validate(one_record(mismatch).record_id)
    mismatch_diag = assert_structured_error(
        caught.value, plugin_id="vendor.alpha", field_contains="LLVM commit"
    )
    assert mismatch_diag["actual"] == OTHER_COMMIT
    assert mismatch_ep.load_calls == 0

    omitted, _ = _validation_fixture(
        tmp_path / "omitted",
        environment_overrides={"actual_llvm_commit": None, "actual_llvm_version": None},
    )
    omitted.validate(one_record(omitted).record_id)

    unavailable, unavailable_ep = _validation_fixture(
        tmp_path / "unavailable",
        plugin_overrides={"requires_llvm_commit": LLVM_COMMIT},
        environment_overrides={"actual_llvm_commit": None},
    )
    with pytest.raises(BackendPluginCompatibilityError) as caught:
        unavailable.validate(one_record(unavailable).record_id)
    unavailable_diag = assert_structured_error(
        caught.value, plugin_id="vendor.alpha", field_contains="LLVM commit"
    )
    assert unavailable_diag["actual"] == "<unknown>"
    assert unavailable_ep.load_calls == 0


def test_registry_capability_required_by_plugin_satisfied_and_missing(
    tmp_path: Path,
) -> None:
    satisfied, _ = _validation_fixture(
        tmp_path / "satisfied",
        plugin_overrides={"requires_capabilities": ["core.fixture"]},
        core_capabilities=["core.fixture"],
    )
    record = satisfied.validate(one_record(satisfied).record_id)
    assert record.capability_report.compatible

    missing, missing_ep = _validation_fixture(
        tmp_path / "missing",
        plugin_overrides={"requires_capabilities": ["core.unknown"]},
    )
    with pytest.raises(BackendPluginCapabilityError) as caught:
        missing.validate(one_record(missing).record_id)
    diagnostic = assert_structured_error(
        caught.value,
        plugin_id="vendor.alpha",
        field_contains="requires_capabilities",
    )
    assert diagnostic["missing_capabilities"] == ["core.unknown"]
    assert missing_ep.load_calls == 0


def test_registry_capability_names_are_opaque_exact_identifiers(tmp_path: Path) -> None:
    registry, entry_point = _validation_fixture(
        tmp_path,
        plugin_overrides={"capabilities": ["future.vendor.capability"]},
    )
    decision = registry.select(
        "alpha", kernel_required_capabilities=["future.vendor.capability"]
    )
    assert decision.capability_report.compatible
    assert entry_point.load_calls == 1


def test_registry_unknown_kernel_capability_rejected_before_loading(tmp_path: Path) -> None:
    registry, entry_point = _validation_fixture(
        tmp_path,
        plugin_overrides={"capabilities": ["fixture.compile"]},
    )
    with pytest.raises(BackendPluginCapabilityError) as caught:
        registry.select("alpha", kernel_required_capabilities=["kernel.unknown"])
    diagnostic = assert_structured_error(
        caught.value,
        plugin_id="vendor.alpha",
        field_contains="kernel_required_capabilities",
    )
    assert diagnostic["missing_capabilities"] == ["kernel.unknown"]
    assert entry_point.load_calls == 0


def test_registry_reports_every_simultaneous_compatibility_failure(
    tmp_path: Path,
) -> None:
    """The acceptance requirement demands complete multi-constraint reasons."""

    registry, entry_point = _validation_fixture(
        tmp_path,
        plugin_overrides={
            "backend_protocol": ">=2.0,<3.0",
            "requires_core": ">=0.3,<0.4",
            "requires_triton": {"version": ">=3.7,<3.8"},
            "requires_llvm_commit": OTHER_COMMIT,
        },
    )
    record = registry.validate()[0]
    diagnostics = [error.to_dict() for error in record.errors]
    dimensions = {
        item.get("dimension", item.get("field")) for item in diagnostics
    }
    assert {
        "backend_protocol",
        "triton-anchor Core version",
        "Triton version",
        "LLVM commit",
    } <= dimensions
    assert all(item["plugin_id"] == "vendor.alpha" for item in diagnostics)
    assert all(item.get("expected") and item.get("actual") for item in diagnostics)
    assert entry_point.load_calls == 0


def test_registry_protocol_optional_field_old_producer_gets_default() -> None:
    result = consume_protocol_field(
        object(), producer_protocol_version="1.0", consumer_protocol_version="1.1"
    )
    assert result.status is ProtocolFieldStatus.COMPATIBLE_DEFAULT
    assert result.value == {}
    assert result.diagnostics == ()


def test_registry_protocol_unknown_field_is_ignored_by_old_consumer() -> None:
    producer = SimpleNamespace(diagnostics={"new": True})
    result = consume_protocol_field(
        producer,
        producer_protocol_version="1.1",
        consumer_protocol_version="1.0",
    )
    assert result.status is ProtocolFieldStatus.COMPATIBLE_IGNORED
    assert result.value is None


def test_registry_protocol_field_preserved_then_warned_during_deprecation() -> None:
    producer = SimpleNamespace(diagnostics=lambda: {"healthy": True})
    preserved = consume_protocol_field(
        producer,
        producer_protocol_version="1.1",
        consumer_protocol_version="1.1",
    )
    deprecated = consume_protocol_field(
        producer,
        producer_protocol_version="1.2",
        consumer_protocol_version="1.2",
    )
    assert preserved.status is ProtocolFieldStatus.COMPATIBLE_PRESERVED
    assert preserved.value == {"healthy": True}
    assert deprecated.status is ProtocolFieldStatus.ACCEPTED_WITH_DEPRECATION_DIAGNOSTIC
    assert deprecated.value == {"healthy": True}
    assert deprecated.diagnostics[0].severity == "warning"
    assert deprecated.diagnostics[0].field == "diagnostics"


def test_registry_protocol_cross_major_field_access_fails_explicitly() -> None:
    result = consume_protocol_field(
        SimpleNamespace(diagnostics={"must_not_be_read": True}),
        producer_protocol_version="2.0",
        consumer_protocol_version="1.2",
    )
    assert result.status is ProtocolFieldStatus.EXPLICIT_PROTOCOL_INCOMPATIBILITY
    assert isinstance(result.error, BackendPluginProtocolError)
    assert result.value is None


def test_registry_protocol_field_removal_requires_major_boundary() -> None:
    forbidden = evaluate_protocol_field_removal("1.9")
    removed = evaluate_protocol_field_removal("2.0")
    assert forbidden.status is ProtocolFieldStatus.FORBIDDEN
    assert forbidden.diagnostics[0].severity == "error"
    assert removed.status is ProtocolFieldStatus.COMPATIBLE


# ---------------------------------------------------------------------------
# Runtime interface, lifecycle, reset, concurrency, conflicts, and selection
# ---------------------------------------------------------------------------


def test_registry_minimal_structural_plugin_full_lifecycle(tmp_path: Path) -> None:
    events: list[Any] = []

    class MinimalPlugin:
        compiler_cls = Compiler
        driver_cls = Driver

        def initialize(self, context: Mapping[str, Any]) -> None:
            events.append(("initialize", context))

        def shutdown(self) -> None:
            events.append(("shutdown", None))

    plugin = MinimalPlugin()
    entry_point = FakeEntryPoint("alpha", plugin)
    distribution = make_distribution(
        tmp_path, entry_points=[entry_point]
    )
    registry = registry_for([distribution])
    discovered = one_record(registry)
    assert discovered.state is PluginLifecycleState.DISCOVERED
    validated = registry.validate(discovered.record_id)
    assert validated.state is PluginLifecycleState.VALIDATED
    assert entry_point.load_calls == 0
    loaded = registry.load(discovered.record_id)
    assert loaded.state is PluginLifecycleState.LOADED
    registered = registry.register(discovered.record_id)
    assert registered.state is PluginLifecycleState.REGISTERED
    assert registered.compiler_cls is Compiler
    assert registered.driver_cls is Driver
    assert registered.initialized
    assert events[0][0] == "initialize"
    context = events[0][1]
    assert isinstance(context["environment"], CoreEnvironment)
    assert context["manifest"].plugin_id == "vendor.alpha"
    assert context["record_id"] == discovered.record_id

    selected = registry.select("alpha")
    assert selected.record.state is PluginLifecycleState.SELECTED
    active = registry.activate(selected.record_id)
    assert active.state is PluginLifecycleState.ACTIVE
    assert registry.activate(selected.record_id) == active
    assert registry.reset() == ()
    assert [event[0] for event in events] == ["initialize", "shutdown"]
    assert registry.get_selection("alpha") is None


def test_registry_base_class_optional_hooks_have_documented_defaults(
    tmp_path: Path,
) -> None:
    class BasePlugin(BackendPluginBase):
        compiler_cls = Compiler
        driver_cls = Driver

    plugin = BasePlugin()
    entry_point = FakeEntryPoint("alpha", plugin)
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    registered = registry.register(one_record(registry).record_id)
    assert registered.initialized
    assert plugin.diagnostics() == {}
    assert registry.diagnostics(registered.record_id)["plugin_diagnostics"] == {}
    assert registry.reset() == ()


def test_registry_optional_lifecycle_hooks_may_be_absent(tmp_path: Path) -> None:
    plugin = structural_plugin()
    entry_point = FakeEntryPoint("alpha", plugin)
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    registered = registry.register(one_record(registry).record_id)
    assert registered.state is PluginLifecycleState.REGISTERED
    assert not registered.initialized
    assert registry.diagnostics(registered.record_id)["plugin_diagnostics"] is None
    assert registry.reset() == ()


@pytest.mark.parametrize("missing", ["compiler_cls", "driver_cls"])
def test_registry_rejects_each_missing_required_runtime_member(
    tmp_path: Path, missing: str
) -> None:
    values = {"compiler_cls": Compiler, "driver_cls": Driver}
    del values[missing]
    entry_point = FakeEntryPoint("alpha", SimpleNamespace(**values))
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    with pytest.raises(BackendPluginInterfaceError) as caught:
        registry.register(one_record(registry).record_id)
    diagnostic = assert_structured_error(
        caught.value, plugin_id="vendor.alpha", field_contains=missing
    )
    assert missing in diagnostic["missing_fields"]


@pytest.mark.parametrize("invalid", ["compiler_cls", "driver_cls"])
def test_registry_rejects_wrong_required_runtime_member_types(
    tmp_path: Path, invalid: str
) -> None:
    values = {"compiler_cls": Compiler, "driver_cls": Driver}
    values[invalid] = object()
    entry_point = FakeEntryPoint("alpha", SimpleNamespace(**values))
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    with pytest.raises(BackendPluginInterfaceError) as caught:
        registry.register(one_record(registry).record_id)
    diagnostic = assert_structured_error(
        caught.value, plugin_id="vendor.alpha", field_contains=invalid
    )
    assert invalid in diagnostic["invalid_fields"]


def test_registry_reports_unreadable_runtime_member(tmp_path: Path) -> None:
    class UnreadablePlugin:
        driver_cls = Driver

        @property
        def compiler_cls(self):
            raise RuntimeError("compiler metadata exploded")

    entry_point = FakeEntryPoint("alpha", UnreadablePlugin())
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    with pytest.raises(BackendPluginInterfaceError) as caught:
        registry.register(one_record(registry).record_id)
    diagnostic = caught.value.to_dict()
    assert diagnostic["field_errors"] == {
        "compiler_cls": "compiler metadata exploded"
    }


def test_registry_rejects_entry_point_import_failure_with_identity(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "load-attempted"
    entry_point = FakeEntryPoint(
        "alpha",
        None,
        marker=marker,
        load_error=ModuleNotFoundError("fixture dependency missing"),
    )
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    with pytest.raises(BackendPluginLoadError) as caught:
        registry.load(one_record(registry).record_id)
    diagnostic = assert_structured_error(
        caught.value, plugin_id="vendor.alpha", field_contains="entry_point.load"
    )
    assert diagnostic["entry_point"] == "alpha"
    assert marker.exists()
    assert entry_point.load_calls == 1


def test_registry_rejects_entry_point_returning_wrong_object(tmp_path: Path) -> None:
    entry_point = FakeEntryPoint("alpha", object())
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    with pytest.raises(BackendPluginInterfaceError):
        registry.register(one_record(registry).record_id)
    rejected = one_record(registry)
    assert rejected.state is PluginLifecycleState.REJECTED
    assert entry_point.load_calls == 1


def test_registry_initialize_failure_attempts_cleanup_and_rejects(
    tmp_path: Path,
) -> None:
    class FailingPlugin:
        compiler_cls = Compiler
        driver_cls = Driver

        def __init__(self) -> None:
            self.initialize_calls = 0
            self.shutdown_calls = 0

        def initialize(self, _context: Mapping[str, Any]) -> None:
            self.initialize_calls += 1
            raise RuntimeError("partial initialization")

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    plugin = FailingPlugin()
    entry_point = FakeEntryPoint("alpha", plugin)
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    with pytest.raises(BackendPluginLifecycleError) as caught:
        registry.register(one_record(registry).record_id)
    diagnostic = assert_structured_error(
        caught.value, plugin_id="vendor.alpha", field_contains="initialize"
    )
    assert "partial initialization" in diagnostic["actual"]
    assert plugin.initialize_calls == 1
    assert plugin.shutdown_calls == 1
    record = one_record(registry)
    assert record.state is PluginLifecycleState.REJECTED
    assert record.shutdown_called
    assert registry.reset() == ()
    assert plugin.shutdown_calls == 1


def test_registry_rejects_noncallable_initialize_hook(tmp_path: Path) -> None:
    plugin = structural_plugin()
    plugin.initialize = "not-callable"
    entry_point = FakeEntryPoint("alpha", plugin)
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    with pytest.raises(BackendPluginLifecycleError) as caught:
        registry.register(one_record(registry).record_id)
    assert_structured_error(
        caught.value, plugin_id="vendor.alpha", field_contains="initialize"
    )


def test_registry_rejects_non_none_initialize_return_value(tmp_path: Path) -> None:
    """Protocol 1.0 publicly declares ``initialize(context) -> None``."""

    class WrongReturnPlugin:
        compiler_cls = Compiler
        driver_cls = Driver

        def __init__(self) -> None:
            self.shutdown_calls = 0

        def initialize(self, _context: Mapping[str, Any]) -> str:
            return "unexpected runtime payload"

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    plugin = WrongReturnPlugin()
    entry_point = FakeEntryPoint("alpha", plugin)
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    with pytest.raises(BackendPluginLifecycleError) as caught:
        registry.register(one_record(registry).record_id)
    diagnostic = assert_structured_error(
        caught.value, plugin_id="vendor.alpha", field_contains="initialize"
    )
    assert "None" in diagnostic["expected"]
    assert "str" in diagnostic["actual"]
    assert plugin.shutdown_calls == 1
    assert one_record(registry).state is PluginLifecycleState.REJECTED


@pytest.mark.parametrize("case", ["noncallable", "wrong_return"])
def test_registry_reports_invalid_shutdown_hook_contract(
    tmp_path: Path, case: str
) -> None:
    """A present optional hook must honor ``shutdown() -> None``."""

    class Plugin:
        compiler_cls = Compiler
        driver_cls = Driver

    plugin = Plugin()
    plugin.shutdown = (
        object()
        if case == "noncallable"
        else lambda: "unexpected shutdown payload"
    )
    entry_point = FakeEntryPoint("alpha", plugin)
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    registry.select("alpha")

    errors = registry.reset()
    assert len(errors) == 1
    diagnostic = assert_structured_error(errors[0], field_contains="shutdown")
    assert diagnostic["plugin_id"] == "vendor.alpha"


@pytest.mark.parametrize("case", ["noncallable", "wrong_return"])
def test_registry_reports_invalid_diagnostics_hook_contract(
    tmp_path: Path, case: str
) -> None:
    """A present optional hook must honor ``diagnostics() -> Dict``."""

    class Plugin:
        compiler_cls = Compiler
        driver_cls = Driver

    plugin = Plugin()
    plugin.diagnostics = (
        object()
        if case == "noncallable"
        else lambda: "unexpected diagnostics payload"
    )
    entry_point = FakeEntryPoint("alpha", plugin)
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    selected = registry.select("alpha")

    value = registry.diagnostics(selected.record_id)["plugin_diagnostics"]
    assert isinstance(value, dict) and "error" in value
    diagnostic = value["error"]
    assert diagnostic["code"] == "backend_plugin_lifecycle_error"
    assert diagnostic["plugin_id"] == "vendor.alpha"
    assert diagnostic["field"] == "diagnostics"
    assert diagnostic["expected"]
    assert diagnostic["actual"]


def test_registry_concurrent_register_loads_and_initializes_exactly_once(
    tmp_path: Path,
) -> None:
    class CountingPlugin:
        compiler_cls = Compiler
        driver_cls = Driver

        def __init__(self) -> None:
            self.initialize_calls = 0
            self.contexts: list[Mapping[str, Any]] = []

        def initialize(self, context: Mapping[str, Any]) -> None:
            self.initialize_calls += 1
            self.contexts.append(context)

    plugin = CountingPlugin()
    entry_point = FakeEntryPoint("alpha", plugin)
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    record_id = one_record(registry).record_id
    start = threading.Barrier(12)

    def register() -> Any:
        start.wait(timeout=10)
        return registry.register(record_id)

    with ThreadPoolExecutor(max_workers=12) as executor:
        records = list(executor.map(lambda _index: register(), range(12)))
    assert {record.record_id for record in records} == {record_id}
    assert all(record.state is PluginLifecycleState.REGISTERED for record in records)
    assert entry_point.load_calls == 1
    assert plugin.initialize_calls == 1
    assert len(plugin.contexts) == 1


def test_registry_reset_racing_initialize_cannot_publish_stale_instance(
    tmp_path: Path,
) -> None:
    initialize_started = threading.Event()
    allow_initialize = threading.Event()
    reset_attempted = threading.Event()

    class BlockingPlugin:
        compiler_cls = Compiler
        driver_cls = Driver

        def __init__(self) -> None:
            self.shutdown_calls = 0

        def initialize(self, _context: Mapping[str, Any]) -> None:
            initialize_started.set()
            if not allow_initialize.wait(timeout=10):
                raise RuntimeError("test did not release initialize")

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    plugin = BlockingPlugin()
    entry_point = FakeEntryPoint("alpha", plugin)
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    record_id = one_record(registry).record_id

    def reset() -> Any:
        reset_attempted.set()
        return registry.reset()

    with ThreadPoolExecutor(max_workers=2) as executor:
        register_future = executor.submit(registry.register, record_id)
        assert initialize_started.wait(timeout=10)
        reset_future = executor.submit(reset)
        assert reset_attempted.wait(timeout=10)
        assert not reset_future.done()
        allow_initialize.set()
        assert register_future.result(timeout=10).state is PluginLifecycleState.REGISTERED
        assert reset_future.result(timeout=10) == ()

    assert plugin.shutdown_calls == 1
    assert registry.get_selection("alpha") is None
    rediscovered = one_record(registry)
    assert rediscovered.state is PluginLifecycleState.DISCOVERED
    assert rediscovered.plugin_object is None


def test_registry_reset_is_idempotent_and_allows_fresh_rediscovery(
    tmp_path: Path,
) -> None:
    class FreshPlugin:
        compiler_cls = Compiler
        driver_cls = Driver
        instances = 0
        initializes = 0
        shutdowns = 0

        def __init__(self) -> None:
            type(self).instances += 1

        def initialize(self, _context: Mapping[str, Any]) -> None:
            type(self).initializes += 1

        def shutdown(self) -> None:
            type(self).shutdowns += 1

    entry_point = FakeEntryPoint("alpha", FreshPlugin)
    distribution = make_distribution(tmp_path, entry_points=[entry_point])
    registry = registry_for([distribution])
    registry.register(one_record(registry).record_id)
    assert (FreshPlugin.instances, FreshPlugin.initializes, FreshPlugin.shutdowns) == (1, 1, 0)
    assert registry.reset() == ()
    assert registry.reset() == ()
    assert (FreshPlugin.instances, FreshPlugin.initializes, FreshPlugin.shutdowns) == (1, 1, 1)
    registry.register(one_record(registry).record_id)
    assert (FreshPlugin.instances, FreshPlugin.initializes, FreshPlugin.shutdowns) == (2, 2, 1)
    assert registry.reset() == ()
    assert (FreshPlugin.instances, FreshPlugin.initializes, FreshPlugin.shutdowns) == (2, 2, 2)


def _selection_registry(
    directory: Path,
    *,
    priorities: tuple[int, int] = (1, 10),
    order: tuple[int, int] = (0, 1),
    same_target: bool = True,
    second_compatible: bool = True,
) -> tuple[BackendPluginRegistry, tuple[FakeEntryPoint, FakeEntryPoint]]:
    alpha = good_plugin(priority=priorities[0])
    beta = good_plugin(
        plugin_id="vendor.beta",
        entry_point="beta",
        targets=["alpha" if same_target else "beta"],
        priority=priorities[1],
        requires_triton=(
            {"version": ">=3.6,<3.7", "commit": TRITON_COMMIT}
            if second_compatible
            else {"version": ">=3.7,<3.8"}
        ),
    )
    alpha_ep = FakeEntryPoint("alpha", structural_plugin("alpha"))
    beta_ep = FakeEntryPoint("beta", structural_plugin("beta"))
    distributions = (
        make_distribution(
            directory / "alpha",
            name="fixture-alpha",
            plugins=[alpha],
            entry_points=[alpha_ep],
        ),
        make_distribution(
            directory / "beta",
            name="fixture-beta",
            plugins=[beta],
            entry_points=[beta_ep],
        ),
    )
    return registry_for(distributions[index] for index in order), (alpha_ep, beta_ep)


def test_registry_duplicate_plugin_id_is_fatal_before_either_import(
    tmp_path: Path,
) -> None:
    first = good_plugin()
    second = good_plugin(entry_point="beta", targets=["beta"])
    first_ep = FakeEntryPoint("alpha", structural_plugin("first"))
    second_ep = FakeEntryPoint("beta", structural_plugin("second"))
    first_dist = make_distribution(
        tmp_path / "first", name="first", plugins=[first], entry_points=[first_ep]
    )
    second_dist = make_distribution(
        tmp_path / "second", name="second", plugins=[second], entry_points=[second_ep]
    )
    registry = registry_for([second_dist, first_dist])
    records = registry.validate()
    report = registry.conflicts()
    duplicate = [item for item in report.fatal_conflicts if item.kind.value == "duplicate_plugin_id"]
    assert len(duplicate) == 1
    with pytest.raises(BackendPluginConflictError):
        registry.load(records[0].record_id)
    assert first_ep.load_calls == second_ep.load_calls == 0


def test_registry_duplicate_entry_point_name_is_fatal_before_import(
    tmp_path: Path,
) -> None:
    first = good_plugin()
    second = good_plugin(plugin_id="vendor.beta", targets=["beta"])
    first_ep = FakeEntryPoint("alpha", structural_plugin("first"))
    second_ep = FakeEntryPoint("alpha", structural_plugin("second"))
    first_dist = make_distribution(
        tmp_path / "first", name="first", plugins=[first], entry_points=[first_ep]
    )
    second_dist = make_distribution(
        tmp_path / "second", name="second", plugins=[second], entry_points=[second_ep]
    )
    registry = registry_for([first_dist, second_dist])
    records = registry.validate()
    assert any(
        item.kind.value == "duplicate_entry_point"
        for item in registry.conflicts().fatal_conflicts
    )
    with pytest.raises(BackendPluginConflictError):
        registry.load(records[0].record_id)
    assert first_ep.load_calls == second_ep.load_calls == 0


def test_registry_shared_target_unique_priority_selects_only_winner(
    tmp_path: Path,
) -> None:
    registry, (alpha_ep, beta_ep) = _selection_registry(tmp_path)
    report = registry.conflicts()
    assert report.requires_selection and not report.has_fatal
    decision = registry.select("alpha")
    assert decision.plugin_id == "vendor.beta"
    assert decision.method is SelectionMethod.MANIFEST_PRIORITY
    assert decision.priority == 10
    assert alpha_ep.load_calls == 0
    assert beta_ep.load_calls == 1
    assert decision.record.compiler_cls.plugin_label == "beta"
    assert decision.record.driver_cls.plugin_label == "beta"


def test_registry_shared_target_equal_priority_is_explicit_ambiguity_without_import(
    tmp_path: Path,
) -> None:
    registry, (alpha_ep, beta_ep) = _selection_registry(
        tmp_path, priorities=(7, 7)
    )
    with pytest.raises(BackendPluginSelectionError) as caught:
        registry.select("alpha")
    diagnostic = assert_structured_error(caught.value, field_contains="priority")
    assert "ambiguous" in diagnostic["message"]
    assert alpha_ep.load_calls == beta_ep.load_calls == 0


def test_registry_selection_precedence_python_then_environment_then_default(
    tmp_path: Path,
) -> None:
    default_registry, _ = _selection_registry(tmp_path / "default")
    default = default_registry.select("alpha", environment={})
    assert default.plugin_id == "vendor.beta"
    assert default.method is SelectionMethod.MANIFEST_PRIORITY

    env_registry, _ = _selection_registry(tmp_path / "environment")
    from_environment = env_registry.select(
        "alpha", environment={BACKEND_SELECTOR_ENV: "vendor.alpha"}
    )
    assert from_environment.plugin_id == "vendor.alpha"
    assert from_environment.method is SelectionMethod.ENVIRONMENT

    explicit_registry, _ = _selection_registry(tmp_path / "explicit")
    explicit = explicit_registry.select(
        "alpha",
        explicit_selector="vendor.beta",
        environment={BACKEND_SELECTOR_ENV: "vendor.alpha"},
    )
    assert explicit.plugin_id == "vendor.beta"
    assert explicit.method is SelectionMethod.PYTHON_EXPLICIT


def test_registry_explicit_selection_rejects_unknown_and_incompatible_without_import(
    tmp_path: Path,
) -> None:
    unknown_registry, unknown_eps = _selection_registry(tmp_path / "unknown")
    with pytest.raises(BackendPluginSelectionError):
        unknown_registry.select("alpha", explicit_selector="vendor.missing")
    assert all(entry_point.load_calls == 0 for entry_point in unknown_eps)

    incompatible_registry, incompatible_eps = _selection_registry(
        tmp_path / "incompatible", second_compatible=False
    )
    with pytest.raises(BackendPluginCompatibilityError) as caught:
        incompatible_registry.select("alpha", explicit_selector="vendor.beta")
    assert_structured_error(
        caught.value, plugin_id="vendor.beta", field_contains="Triton version"
    )
    assert all(entry_point.load_calls == 0 for entry_point in incompatible_eps)


def test_registry_selection_and_conflicts_are_enumeration_order_independent(
    tmp_path: Path,
) -> None:
    decisions = []
    conflict_views = []
    for index, order in enumerate(itertools.permutations((0, 1))):
        registry, _ = _selection_registry(
            tmp_path / str(index), order=order
        )
        conflict_views.append(registry.conflicts().to_dict())
        records = registry.validate()
        decision = select_backend(records[::-1], target="alpha")
        decisions.append(decision.to_dict())
    assert conflict_views[0] == conflict_views[1]
    assert decisions[0] == decisions[1]
    assert decisions[0]["plugin_id"] == "vendor.beta"


def test_registry_equal_priority_error_is_enumeration_order_independent(
    tmp_path: Path,
) -> None:
    diagnostics = []
    for index, order in enumerate(itertools.permutations((0, 1))):
        registry, entry_points = _selection_registry(
            tmp_path / str(index), priorities=(5, 5), order=order
        )
        with pytest.raises(BackendPluginSelectionError) as caught:
            registry.select("alpha")
        diagnostics.append(caught.value.to_dict())
        assert all(entry_point.load_calls == 0 for entry_point in entry_points)
    assert diagnostics[0] == diagnostics[1]


def test_registry_two_targets_can_be_selected_without_compiler_driver_split(
    tmp_path: Path,
) -> None:
    registry, _ = _selection_registry(tmp_path, same_target=False)
    alpha = registry.select("alpha")
    beta = registry.select("beta")
    assert alpha.plugin_id == "vendor.alpha"
    assert beta.plugin_id == "vendor.beta"
    assert alpha.record.compiler_cls.plugin_label == alpha.record.driver_cls.plugin_label == "alpha"
    assert beta.record.compiler_cls.plugin_label == beta.record.driver_cls.plugin_label == "beta"


def test_registry_cannot_activate_two_plugins_without_reset(tmp_path: Path) -> None:
    registry, _ = _selection_registry(tmp_path, same_target=False)
    alpha = registry.select("alpha")
    beta = registry.select("beta")
    registry.activate(alpha.record_id)
    with pytest.raises(BackendPluginConflictError) as caught:
        registry.activate(beta.record_id)
    assert_structured_error(caught.value, field_contains="active")
