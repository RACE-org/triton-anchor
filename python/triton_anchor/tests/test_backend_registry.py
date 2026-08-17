"""High-value contract tests for the W4 Registry and W5 pre-load gate."""

import base64
import hashlib
import json
import subprocess
import sysconfig
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import pytest
from packaging.tags import Tag

from triton_anchor.backends import (
    BackendPluginCompatibilityError,
    BackendPluginConflictError,
    BackendPluginDiscoveryError,
    BackendPluginInterfaceError,
    BackendPluginManifestError,
    BackendPluginProtocolError,
    BackendPluginRegistry,
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
    collect_core_environment,
)


SUPPORTED_TAG = Tag("py3", "none", "any")
NATIVE_PLATFORM = sysconfig.get_platform().replace("-", "_").replace(".", "_")
NATIVE_TAG = Tag("py3", "none", NATIVE_PLATFORM)
NATIVE_WHEEL = """\
Wheel-Version: 1.0
Generator: triton-anchor-tests
Root-Is-Purelib: false
Tag: py3-none-{}
""".format(NATIVE_PLATFORM)
TEST_ABI_FINGERPRINT = "sha256:" + ("a" * 64)
VALID_WHEEL = """\
Wheel-Version: 1.0
Generator: triton-anchor-tests
Root-Is-Purelib: true
Tag: py3-none-any
"""


class DummyCompiler:
    pass


class DummyDriver:
    pass


class RuntimePlugin:
    compiler_cls = DummyCompiler
    driver_cls = DummyDriver


class HookPlugin(RuntimePlugin):
    def __init__(self):
        self.initialize_calls = []
        self.shutdown_calls = 0
        self.diagnostic_calls = 0

    def initialize(self, context):
        self.initialize_calls.append(context)

    def shutdown(self):
        self.shutdown_calls += 1

    def diagnostics(self):
        self.diagnostic_calls += 1
        return {"healthy": True}


class BrokenRuntimePlugin:
    compiler_cls = DummyCompiler
    driver_cls = "not-a-class"


@dataclass
class SpyEntryPoint:
    """Python 3.8-style entry point: deliberately has no ``dist`` field."""

    name: str
    loaded_object: object
    value: str = ""
    group: str = "triton.backends"
    load_calls: int = 0

    def __post_init__(self):
        if not self.value:
            self.value = "vendor_backend." + self.name

    def load(self):
        self.load_calls += 1
        return self.loaded_object


class FakeDistribution:
    def __init__(
        self,
        root,
        *,
        name="vendor-backend",
        version="1.0.0",
        manifest=None,
        entry_points=(("mock", RuntimePlugin()),),
        wheel_text=VALID_WHEEL,
        record_files=(),
        materialized_files=(),
        files_available=True,
    ):
        self.root = Path(root)
        self.metadata = {"Name": name}
        self.version = version
        self.entry_points = [
            SpyEntryPoint(entry_name, loaded)
            for entry_name, loaded in entry_points
        ]
        self.wheel_text = wheel_text
        self.record_text = None
        self.files = [
            PurePosixPath(
                "{}-{}.dist-info/WHEEL".format(
                    name.replace("-", "_"), version
                )
            )
        ]

        if manifest is not None:
            manifest_file = PurePosixPath(
                name.replace("-", "_") + "/triton_anchor_backend.json"
            )
            manifest_path = self.root / str(manifest_file)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(manifest, str):
                manifest_path.write_text(manifest, encoding="utf-8")
            else:
                manifest_path.write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
            self.files.append(manifest_file)

        self.files.extend(PurePosixPath(item) for item in record_files)
        for item in materialized_files:
            path = self.root / str(PurePosixPath(item))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test-native-library")

        if not files_available:
            self.files = None

    def locate_file(self, file):
        return self.root / str(file)

    def read_text(self, filename):
        if filename == "WHEEL":
            return self.wheel_text
        if filename == "RECORD":
            if self.record_text is not None:
                return self.record_text
            rows = []
            for item in self.files or ():
                path = self.locate_file(item)
                if path.is_file():
                    digest = base64.urlsafe_b64encode(
                        hashlib.sha256(path.read_bytes()).digest()
                    ).decode("ascii").rstrip("=")
                    rows.append("{},sha256={},{}".format(
                        item, digest, path.stat().st_size
                    ))
                else:
                    rows.append("{},,".format(item))
            return "\n".join(rows) + "\n"
        return None


def build_test_shared_library(
    root,
    relative_path,
    *,
    soname="libvendor_test.so",
    symbol="vendor_test_entry",
):
    output = Path(root) / relative_path
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output.with_suffix(".c")
    source.write_text(
        "int {}(void) {{ return 7; }}\n".format(symbol),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-Wl,-soname," + soname,
            "-o",
            str(output),
            str(source),
        ],
        check=True,
    )
    return output


def plugin_record(
    entry_point="mock",
    *,
    plugin_id=None,
    isolation_mode="python_only",
    **overrides
):
    record = {
        "plugin_id": plugin_id or "vendor." + entry_point,
        "entry_point": entry_point,
        "backend_protocol": ">=1.0,<2.0",
        "requires_core": ">=0.2,<0.3",
        "requires_triton": {
            "version": ">=3.0,<3.1",
            "commit": "757b6a61e7df814ba806f498f8bb3160f84b120c",
        },
        "targets": [entry_point],
        "isolation_mode": isolation_mode,
    }
    record.update(overrides)
    return record


def manifest(*records):
    return {"schema_version": "1.0", "plugins": list(records)}


def core_environment(**overrides):
    source_environment = collect_core_environment()
    environment = replace(
        source_environment,
        build_info_generated=True,
        actual_llvm_version_raw="19.0.0git",
        actual_llvm_version="19.0.0",
        actual_llvm_version_suffix="git",
        actual_llvm_commit="a" * 40,
        actual_mlir_version_raw="19.0.0git",
        actual_mlir_version="19.0.0",
        actual_mlir_version_suffix="git",
        actual_mlir_commit="b" * 40,
        built_python_version=source_environment.runtime_python_version,
        built_python_soabi=source_environment.runtime_python_soabi,
        built_platform=source_environment.runtime_platform,
        core_abi_fingerprint_schema="triton-anchor-core-abi-v1",
        core_library_sha256=TEST_ABI_FINGERPRINT,
        core_abi_fingerprint=TEST_ABI_FINGERPRINT,
    )
    return replace(environment, **overrides)


def make_registry(distributions, *, environment=None, core_abi_fingerprint=None):
    return BackendPluginRegistry(
        distribution_provider=lambda: tuple(distributions),
        environment_provider=lambda: environment or core_environment(),
        core_abi_fingerprint=core_abi_fingerprint,
        supported_tags=(SUPPORTED_TAG, NATIVE_TAG),
        preflight_profile="full",
    )


def test_discovery_is_python38_safe_deterministic_and_import_free(tmp_path):
    alpha = FakeDistribution(
        tmp_path / "alpha",
        name="alpha-backend",
        manifest=manifest(
            plugin_record("zeta"),
            plugin_record("alpha"),
        ),
        entry_points=(
            ("zeta", RuntimePlugin()),
            ("alpha", RuntimePlugin()),
        ),
    )
    zulu = FakeDistribution(
        tmp_path / "zulu",
        name="zulu-backend",
        manifest=manifest(plugin_record("mock")),
    )
    environment_calls = []
    registry = BackendPluginRegistry(
        distribution_provider=lambda: (zulu, alpha),
        environment_provider=lambda: environment_calls.append(True),
        supported_tags=(SUPPORTED_TAG,),
    )

    records = registry.discover()
    assert [record.record_id for record in records] == [
        "alpha-backend:alpha",
        "alpha-backend:zeta",
        "zulu-backend:mock",
    ]
    assert all(
        record.state is PluginLifecycleState.DISCOVERED
        for record in records
    )
    assert all(
        record.compatibility_status is PluginCompatibilityStatus.NOT_CHECKED
        for record in records
    )
    assert environment_calls == []
    assert sum(
        ep.load_calls
        for distribution in (alpha, zulu)
        for ep in distribution.entry_points
    ) == 0

    assert registry.discover() == records
    assert registry.list() == records
    registry.inspect("vendor.alpha")
    registry.diagnostics()
    assert environment_calls == []
    assert sum(
        ep.load_calls
        for distribution in (alpha, zulu)
        for ep in distribution.entry_points
    ) == 0


def test_invalid_distribution_is_isolated_from_valid_plugin(tmp_path):
    bad = FakeDistribution(
        tmp_path / "bad",
        name="bad-backend",
        manifest="{not-json",
    )
    good = FakeDistribution(
        tmp_path / "good",
        name="good-backend",
        manifest=manifest(
            plugin_record("mock", plugin_id="good.mock")
        ),
    )
    registry = make_registry((bad, good))

    discovered = registry.discover()
    assert [record.state for record in discovered] == [
        PluginLifecycleState.REJECTED,
        PluginLifecycleState.DISCOVERED,
    ]

    validated = registry.validate()
    assert [record.state for record in validated] == [
        PluginLifecycleState.REJECTED,
        PluginLifecycleState.VALIDATED,
    ]
    assert isinstance(validated[0].error, BackendPluginManifestError)
    assert validated[0].error.remediation
    assert all(
        ep.load_calls == 0
        for distribution in (bad, good)
        for ep in distribution.entry_points
    )


def test_unavailable_file_list_is_rejected_not_assumed_legacy(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=None,
        files_available=False,
    )
    registry = make_registry((distribution,))

    record = registry.discover()[0]
    assert record.state is PluginLifecycleState.REJECTED
    assert record.source is None
    assert record.compatibility_status is PluginCompatibilityStatus.NOT_CHECKED
    assert isinstance(record.error, BackendPluginManifestError)
    assert distribution.entry_points[0].load_calls == 0


def test_manifest_coverage_failure_rejects_every_entry_point_without_import(
    tmp_path,
):
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record("first")),
        entry_points=(
            ("first", RuntimePlugin()),
            ("second", RuntimePlugin()),
        ),
    )
    registry = make_registry((distribution,))

    records = registry.discover()
    assert len(records) == 2
    assert all(
        record.state is PluginLifecycleState.REJECTED
        for record in records
    )
    assert all(record.source is PluginSource.MANIFEST for record in records)
    assert all(
        isinstance(record.error, BackendPluginManifestError)
        for record in records
    )
    assert [ep.load_calls for ep in distribution.entry_points] == [0, 0]


def test_legacy_validation_does_not_claim_compatibility_or_import(tmp_path):
    plugin = RuntimePlugin()
    distribution = FakeDistribution(
        tmp_path,
        manifest=None,
        entry_points=(("legacy", plugin),),
        wheel_text=None,
    )
    registry = make_registry((distribution,))

    record = registry.validate()[0]
    assert record.source is PluginSource.LEGACY
    assert record.state is PluginLifecycleState.DISCOVERED
    assert (
        record.compatibility_status
        is PluginCompatibilityStatus.LEGACY_UNVERIFIED
    )
    assert distribution.entry_points[0].load_calls == 0

    registered = registry.register(record.registry_key)
    assert registered.state is PluginLifecycleState.REGISTERED
    assert registered.compiler_cls is DummyCompiler
    assert registered.driver_cls is DummyDriver
    assert (
        registered.compatibility_status
        is PluginCompatibilityStatus.LEGACY_UNVERIFIED
    )
    assert distribution.entry_points[0].load_calls == 1


def test_legacy_runtime_fields_must_still_be_classes(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=None,
        entry_points=(("legacy", BrokenRuntimePlugin()),),
        wheel_text=None,
    )
    registry = make_registry((distribution,))
    record = registry.discover()[0]

    with pytest.raises(BackendPluginInterfaceError) as caught:
        registry.register(record.registry_key)

    rejected = registry.inspect(record.record_id)
    assert rejected.state is PluginLifecycleState.REJECTED
    assert caught.value.invalid_fields == ("driver_cls",)
    assert (
        rejected.compatibility_status
        is PluginCompatibilityStatus.LEGACY_UNVERIFIED
    )
    assert distribution.entry_points[0].load_calls == 1


@pytest.mark.parametrize(
    "wheel_text,error_fragment",
    [
        (None, "missing WHEEL"),
        (
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\n",
            "at least one wheel Tag",
        ),
        (
            "Wheel-Version: 1.0\nTag: cp39-cp39-win_amd64\n",
            "wheel platform tag",
        ),
        (
            "Wheel-Version: 1.0\nTag: broken\n",
            "wheel platform metadata",
        ),
    ],
)
def test_wheel_metadata_failure_blocks_direct_load(
    tmp_path, wheel_text, error_fragment
):
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record()),
        wheel_text=wheel_text,
    )
    registry = make_registry((distribution,))
    record = registry.discover()[0]

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.load(record.registry_key)

    rejected = registry.inspect(record.record_id)
    assert error_fragment in str(caught.value)
    assert rejected.state is PluginLifecycleState.REJECTED
    assert (
        rejected.compatibility_status
        is PluginCompatibilityStatus.INCOMPATIBLE
    )
    diagnostic = caught.value.to_dict()
    assert diagnostic["field"]
    assert diagnostic["expected"]
    assert diagnostic["actual"]
    assert diagnostic["remediation"]
    assert distribution.entry_points[0].load_calls == 0


def test_any_compatible_wheel_tag_is_sufficient(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record()),
        wheel_text=(
            "Wheel-Version: 1.0\n"
            "Tag: cp39-cp39-win_amd64\n"
            "Tag: py3-none-any\n"
        ),
    )
    registry = make_registry((distribution,))

    record = registry.validate()[0]
    assert record.state is PluginLifecycleState.VALIDATED
    assert distribution.entry_points[0].load_calls == 0
    dimensions = {
        check.dimension for check in record.compatibility_report.checks
    }
    assert "wheel platform tag" in dimensions


def test_distribution_wheel_failure_rejects_all_records_without_import(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(
            plugin_record("first"),
            plugin_record("second"),
        ),
        entry_points=(
            ("second", RuntimePlugin()),
            ("first", RuntimePlugin()),
        ),
        wheel_text="Wheel-Version: 1.0\nTag: cp39-cp39-win_amd64\n",
    )
    registry = make_registry((distribution,))

    records = registry.validate()
    assert len(records) == 2
    assert all(
        record.state is PluginLifecycleState.REJECTED
        for record in records
    )
    assert all(
        record.compatibility_status
        is PluginCompatibilityStatus.INCOMPATIBLE
        for record in records
    )
    assert all(
        record.error.field == "wheel platform tag" for record in records
    )
    assert [ep.load_calls for ep in distribution.entry_points] == [0, 0]


def test_native_library_requires_exact_record_path_and_real_file(tmp_path):
    declared = "vendor_backend/lib/libvendor.so"
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(
            plugin_record(
                isolation_mode="native_in_process",
                native_libraries=[declared],
                abi_fingerprint=TEST_ABI_FINGERPRINT,
            )
        ),
        record_files=("other/libvendor.so",),
        materialized_files=("other/libvendor.so",),
    )
    registry = make_registry(
        (distribution,), core_abi_fingerprint=TEST_ABI_FINGERPRINT
    )
    record = registry.discover()[0]

    with pytest.raises(BackendPluginManifestError) as caught:
        registry.load(record.registry_key)

    assert "missing from the installed distribution" in str(caught.value)
    diagnostic = caught.value.to_dict()
    assert diagnostic["field"] == "native_libraries"
    assert diagnostic["expected"]
    assert diagnostic["actual"] == declared
    assert diagnostic["remediation"]
    assert distribution.entry_points[0].load_calls == 0


def test_missing_native_file_rejects_only_its_plugin_record(tmp_path):
    missing_library = "vendor_backend/lib/missing.so"
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(
            plugin_record("good", plugin_id="vendor.good"),
            plugin_record(
                "bad",
                plugin_id="vendor.bad",
                isolation_mode="native_in_process",
                native_libraries=[missing_library],
                abi_fingerprint=TEST_ABI_FINGERPRINT,
            ),
        ),
        entry_points=(
            ("bad", RuntimePlugin()),
            ("good", RuntimePlugin()),
        ),
    )
    registry = make_registry(
        (distribution,), core_abi_fingerprint=TEST_ABI_FINGERPRINT
    )

    records = {record.plugin_id: record for record in registry.validate()}
    assert records["vendor.good"].state is PluginLifecycleState.VALIDATED
    assert records["vendor.bad"].state is PluginLifecycleState.REJECTED
    assert (
        records["vendor.bad"].compatibility_status
        is PluginCompatibilityStatus.NOT_CHECKED
    )
    assert [ep.load_calls for ep in distribution.entry_points] == [0, 0]


@pytest.mark.parametrize(
    "case,error_type,expected_field",
    [
        ("protocol", BackendPluginProtocolError, "backend_protocol"),
        (
            "core",
            BackendPluginCompatibilityError,
            "triton-anchor Core version",
        ),
        ("triton_version", BackendPluginCompatibilityError, "Triton version"),
        (
            "triton_commit",
            BackendPluginCompatibilityError,
            "vendored Triton commit",
        ),
        ("llvm_version", BackendPluginCompatibilityError, "LLVM version"),
        ("llvm_commit", BackendPluginCompatibilityError, "LLVM commit"),
        ("mlir_version", BackendPluginCompatibilityError, "MLIR version"),
        ("mlir_commit", BackendPluginCompatibilityError, "MLIR commit"),
        ("unknown_llvm", BackendPluginCompatibilityError, "LLVM version"),
    ],
)
def test_every_declared_version_dimension_rejects_before_import(
    tmp_path, case, error_type, expected_field
):
    declaration = plugin_record()
    environment = core_environment()

    if case == "protocol":
        declaration["backend_protocol"] = ">=2.0,<3.0"
    elif case == "core":
        declaration["requires_core"] = ">=9.0"
    elif case == "triton_version":
        declaration["requires_triton"] = {"version": ">=9.0"}
    elif case == "triton_commit":
        declaration["requires_triton"] = {
            "version": ">=3.0,<3.1",
            "commit": "f" * 40,
        }
    elif case == "llvm_version":
        declaration["requires_llvm_version"] = ">=20.0"
    elif case == "llvm_commit":
        declaration["requires_llvm_commit"] = "f" * 40
    elif case == "mlir_version":
        declaration["requires_mlir_version"] = ">=20.0"
    elif case == "mlir_commit":
        declaration["requires_mlir_commit"] = "f" * 40
    elif case == "unknown_llvm":
        declaration["requires_llvm_version"] = ">=19.0,<20.0"
        environment = replace(
            environment,
            actual_llvm_version_raw=None,
            actual_llvm_version=None,
            actual_llvm_version_suffix=None,
        )

    distribution = FakeDistribution(
        tmp_path, manifest=manifest(declaration)
    )
    registry = make_registry((distribution,), environment=environment)
    record = registry.discover()[0]

    with pytest.raises(error_type) as caught:
        registry.load(record.registry_key)

    rejected = registry.inspect(record.record_id)
    assert rejected.state is PluginLifecycleState.REJECTED
    assert (
        rejected.compatibility_status
        is PluginCompatibilityStatus.INCOMPATIBLE
    )
    diagnostic = caught.value.to_dict()
    assert diagnostic["field"] == expected_field
    assert diagnostic["expected"]
    assert diagnostic["actual"]
    assert diagnostic["remediation"]
    assert distribution.entry_points[0].load_calls == 0


def test_expected_llvm_pin_is_not_used_as_actual_commit(tmp_path):
    environment = replace(
        core_environment(),
        actual_llvm_commit=None,
        expected_llvm_project_commit="f" * 40,
    )
    declaration = plugin_record(requires_llvm_commit="f" * 40)
    distribution = FakeDistribution(
        tmp_path, manifest=manifest(declaration)
    )
    registry = make_registry((distribution,), environment=environment)

    record = registry.discover()[0]
    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.load(record.registry_key)

    assert caught.value.dimension == "LLVM commit"
    assert caught.value.actual == "<unknown>"
    assert distribution.entry_points[0].load_calls == 0


@pytest.mark.parametrize(
    "environment_override,expected_dimension",
    [
        (
            {
                "build_info_generated": True,
                "built_python_soabi": "incompatible-soabi",
            },
            "core Python SOABI",
        ),
        (
            {
                "build_info_generated": True,
                "built_platform": "incompatible-platform",
            },
            "core build platform",
        ),
    ],
)
def test_generated_core_runtime_mismatch_blocks_plugin_import(
    tmp_path, environment_override, expected_dimension
):
    base = core_environment()
    complete_override = {
        "built_python_soabi": base.runtime_python_soabi,
        "built_platform": base.runtime_platform,
    }
    complete_override.update(environment_override)
    environment = replace(base, **complete_override)
    distribution = FakeDistribution(
        tmp_path, manifest=manifest(plugin_record())
    )
    registry = make_registry((distribution,), environment=environment)

    record = registry.discover()[0]
    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.load(record.registry_key)

    assert caught.value.dimension == expected_dimension
    assert distribution.entry_points[0].load_calls == 0


def test_full_profile_rejects_unproven_source_build_before_import(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record()),
    )
    registry = make_registry(
        (distribution,),
        environment=collect_core_environment(),
    )
    record = registry.discover()[0]

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.load(record.record_id)

    assert caught.value.dimension == "Core build provenance"
    assert distribution.entry_points[0].load_calls == 0


def test_native_in_process_requires_exact_core_abi_before_import(tmp_path):
    declared = "vendor_backend/lib/libvendor.so"
    declaration = plugin_record(
        isolation_mode="native_in_process",
        native_libraries=[declared],
        abi_fingerprint=TEST_ABI_FINGERPRINT,
    )
    build_test_shared_library(tmp_path, declared)
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(declaration),
        record_files=(declared,),
        wheel_text=NATIVE_WHEEL,
    )
    registry = make_registry(
        (distribution,),
        environment=replace(
            core_environment(),
            core_abi_fingerprint=None,
        ),
    )
    record = registry.discover()[0]

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.load(record.registry_key)

    assert caught.value.dimension == "Core ABI fingerprint"
    assert caught.value.actual == "<unknown>"
    assert distribution.entry_points[0].load_calls == 0

    compatible_distribution = FakeDistribution(
        tmp_path / "compatible",
        name="compatible-backend",
        manifest=manifest(declaration),
        record_files=(declared,),
        wheel_text=NATIVE_WHEEL,
    )
    build_test_shared_library(tmp_path / "compatible", declared)
    compatible_registry = make_registry((compatible_distribution,))
    compatible = compatible_registry.validate()[0]
    assert compatible.state is PluginLifecycleState.VALIDATED
    assert compatible_distribution.entry_points[0].load_calls == 0


def test_python_only_cannot_hide_native_wheel_file(tmp_path):
    hidden = "vendor_backend/lib/libhidden.so"
    build_test_shared_library(tmp_path, hidden)
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record()),
        record_files=(hidden,),
        wheel_text=NATIVE_WHEEL,
    )
    registry = make_registry((distribution,))
    record = registry.discover()[0]

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.load(record.record_id)

    assert caught.value.dimension == "python_only wheel contents"
    assert hidden in caught.value.actual
    assert distribution.entry_points[0].load_calls == 0


def test_native_record_hash_tamper_is_rejected_before_import(tmp_path):
    declared = "vendor_backend/lib/libtampered.so"
    build_test_shared_library(tmp_path, declared)
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(
            plugin_record(
                isolation_mode="native_in_process",
                native_libraries=[declared],
                abi_fingerprint=TEST_ABI_FINGERPRINT,
            )
        ),
        record_files=(declared,),
        wheel_text=NATIVE_WHEEL,
    )
    distribution.record_text = distribution.read_text("RECORD")
    path = distribution.locate_file(declared)
    path.write_bytes(path.read_bytes() + b"modified-after-install")
    registry = make_registry((distribution,))
    record = registry.discover()[0]

    with pytest.raises(BackendPluginManifestError) as caught:
        registry.load(record.record_id)

    assert caught.value.field == "RECORD"
    assert "hash mismatch" in str(caught.value)
    assert distribution.entry_points[0].load_calls == 0


def test_native_library_requires_platform_wheel_layout(tmp_path):
    declared = "vendor_backend/lib/liblayout.so"
    build_test_shared_library(tmp_path, declared)
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(
            plugin_record(
                isolation_mode="native_in_process",
                native_libraries=[declared],
                abi_fingerprint=TEST_ABI_FINGERPRINT,
            )
        ),
        record_files=(declared,),
        wheel_text=VALID_WHEEL,
    )
    registry = make_registry((distribution,))
    record = registry.discover()[0]

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.load(record.record_id)

    assert caught.value.dimension == "native wheel layout"
    assert distribution.entry_points[0].load_calls == 0


def test_native_architecture_mismatch_is_rejected_before_import(
    tmp_path,
    monkeypatch,
):
    declared = "vendor_backend/lib/libarch.so"
    build_test_shared_library(tmp_path, declared)
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(
            plugin_record(
                isolation_mode="native_in_process",
                native_libraries=[declared],
                abi_fingerprint=TEST_ABI_FINGERPRINT,
            )
        ),
        record_files=(declared,),
        wheel_text=NATIVE_WHEEL,
    )
    monkeypatch.setattr(
        "triton_anchor.backends.native.platform.machine",
        lambda: "aarch64",
    )
    registry = make_registry((distribution,))
    record = registry.discover()[0]

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.load(record.record_id)

    assert caught.value.dimension == "native architecture"
    assert distribution.entry_points[0].load_calls == 0


def test_native_soname_and_symbol_conflicts_block_every_import(tmp_path):
    distributions = []
    for entry_point, symbol_suffix in (("alpha", "one"), ("beta", "two")):
        root = tmp_path / entry_point
        declared = "{}/lib/libbackend.so".format(entry_point)
        build_test_shared_library(
            root,
            declared,
            soname="libvendor_collision.so",
            symbol="vendor_collision",
        )
        distributions.append(
            FakeDistribution(
                root,
                name=entry_point + "-backend",
                manifest=manifest(
                    plugin_record(
                        entry_point,
                        isolation_mode="native_in_process",
                        native_libraries=[declared],
                        abi_fingerprint=TEST_ABI_FINGERPRINT,
                        display_name=symbol_suffix,
                    )
                ),
                entry_points=((entry_point, RuntimePlugin()),),
                record_files=(declared,),
                wheel_text=NATIVE_WHEEL,
            )
        )

    registry = make_registry(tuple(reversed(distributions)))
    validated = registry.validate()
    assert all(
        record.state is PluginLifecycleState.VALIDATED
        for record in validated
    )
    report = registry.conflicts()
    assert report.has_fatal
    assert {
        conflict.kind.value for conflict in report.fatal_conflicts
    } == {
        "duplicate_native_identity",
        "duplicate_exported_symbol",
    }

    with pytest.raises(BackendPluginConflictError):
        registry.select("alpha", environment={})
    assert [
        distribution.entry_points[0].load_calls
        for distribution in distributions
    ] == [0, 0]


def test_subprocess_is_rejected_until_ir_contract_is_versioned(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record(isolation_mode="subprocess")),
    )
    registry = make_registry((distribution,))
    record = registry.discover()[0]

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.load(record.registry_key)

    assert caught.value.dimension == "subprocess IR contract"
    assert distribution.entry_points[0].load_calls == 0


def test_duplicate_plugin_ids_are_preserved_and_not_silently_selected(tmp_path):
    first = FakeDistribution(
        tmp_path / "first",
        name="first-backend",
        manifest=manifest(plugin_record(plugin_id="vendor.duplicate")),
    )
    second = FakeDistribution(
        tmp_path / "second",
        name="second-backend",
        manifest=manifest(plugin_record(plugin_id="vendor.duplicate")),
    )
    registry = make_registry((second, first))

    records = registry.discover()
    assert len(records) == 2
    assert {record.plugin_id for record in records} == {"vendor.duplicate"}
    with pytest.raises(BackendPluginConflictError):
        registry.inspect("vendor.duplicate")
    assert registry.inspect(records[0].record_id) == records[0]
    assert [first.entry_points[0].load_calls, second.entry_points[0].load_calls] == [
        0,
        0,
    ]


def test_distribution_provider_exception_is_structured():
    def broken_provider():
        raise RuntimeError("metadata database unavailable")

    registry = BackendPluginRegistry(
        distribution_provider=broken_provider,
        environment_provider=core_environment,
        supported_tags=(SUPPORTED_TAG,),
    )

    with pytest.raises(BackendPluginDiscoveryError) as caught:
        registry.discover()

    diagnostic = caught.value.to_dict()
    assert diagnostic["field"] == "installed_distributions"
    assert "metadata database unavailable" in diagnostic["actual"]
    assert diagnostic["expected"]
    assert diagnostic["remediation"]


def test_unexpected_distribution_discovery_error_is_isolated(
    tmp_path, monkeypatch
):
    import triton_anchor.backends.registry as registry_module

    bad = FakeDistribution(
        tmp_path / "bad",
        name="bad-backend",
        manifest=manifest(plugin_record()),
    )
    good = FakeDistribution(
        tmp_path / "good",
        name="good-backend",
        manifest=manifest(
            plugin_record(plugin_id="good.mock")
        ),
    )
    original_loader = registry_module.load_distribution_manifest

    def sometimes_broken(distribution):
        if distribution is bad:
            raise RuntimeError("unexpected metadata reader failure")
        return original_loader(distribution)

    monkeypatch.setattr(
        registry_module, "load_distribution_manifest", sometimes_broken
    )
    registry = make_registry((bad, good))

    records = registry.discover()
    assert [record.state for record in records] == [
        PluginLifecycleState.REJECTED,
        PluginLifecycleState.DISCOVERED,
    ]
    assert isinstance(records[0].error, BackendPluginManifestError)
    assert "unexpected metadata reader failure" in str(records[0].error)
    assert records[0].error.remediation
    assert [bad.entry_points[0].load_calls, good.entry_points[0].load_calls] == [
        0,
        0,
    ]


def test_load_register_diagnostics_and_reset_are_idempotent(tmp_path):
    plugin = HookPlugin()
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record()),
        entry_points=(("mock", plugin),),
    )
    registry = make_registry((distribution,))

    discovered = registry.discover()[0]
    assert registry.inspect(discovered.record_id).state is PluginLifecycleState.DISCOVERED
    assert registry.diagnostics(discovered.record_id)["plugin_diagnostics"] is None
    assert distribution.entry_points[0].load_calls == 0

    registered = registry.register(discovered.registry_key)
    assert registered.state is PluginLifecycleState.REGISTERED
    assert registered.compiler_cls is DummyCompiler
    assert registered.driver_cls is DummyDriver
    assert len(plugin.initialize_calls) == 1
    assert distribution.entry_points[0].load_calls == 1

    assert registry.register(discovered.registry_key) == registered
    assert len(plugin.initialize_calls) == 1
    assert distribution.entry_points[0].load_calls == 1
    assert registry.diagnostics(discovered.registry_key)["plugin_diagnostics"] == {
        "healthy": True
    }

    assert registry.reset() == ()
    assert plugin.shutdown_calls == 1
    assert registry.reset() == ()
    assert plugin.shutdown_calls == 1


def test_explicit_empty_initialization_context_is_preserved(tmp_path):
    plugin = HookPlugin()
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record()),
        entry_points=(("mock", plugin),),
    )
    registry = make_registry((distribution,))
    record = registry.discover()[0]

    registry.register(record.registry_key, context={})

    assert plugin.initialize_calls == [{}]
    assert distribution.entry_points[0].load_calls == 1


def test_runtime_pair_is_checked_only_after_successful_preflight(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=manifest(plugin_record()),
        entry_points=(("mock", BrokenRuntimePlugin()),),
    )
    registry = make_registry((distribution,))

    record = registry.validate()[0]
    assert record.state is PluginLifecycleState.VALIDATED
    assert distribution.entry_points[0].load_calls == 0

    with pytest.raises(BackendPluginInterfaceError):
        registry.register(record.registry_key)

    rejected = registry.inspect(record.record_id)
    assert rejected.state is PluginLifecycleState.REJECTED
    assert distribution.entry_points[0].load_calls == 1
