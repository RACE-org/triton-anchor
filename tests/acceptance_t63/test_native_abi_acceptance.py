"""Independent native ABI acceptance fixtures for T6.3.

These tests build tiny ELF shared objects at runtime.  They exercise the
public Manifest, compatibility, native inspection, conflict, and Registry
APIs without importing a real hardware backend or relying on T10.2.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from packaging.tags import Tag

from triton_anchor.backends import (
    BackendPluginCompatibilityError,
    BackendPluginManifestError,
    BackendPluginRecord,
    BackendPluginRegistry,
    CompatibilityReport,
    CoreEnvironment,
    NativeArtifact,
    PluginCompatibilityStatus,
    PluginIsolationMode,
    PluginLifecycleState,
    PluginSource,
    SelectionDecision,
    SelectionMethod,
    detect_conflicts,
    inspect_native_artifacts,
    load_build_info,
    parse_manifest,
    select_backend,
    validate_backend_plugin,
    validate_triton_requirement,
)


TRITON_COMMIT = "523a1b235b213bc192f2d5a8999add5bf2d0fea5"
LLVM_COMMIT = "a66376b0dc3b2ea8a84fda26faca287980986f78"
FINGERPRINT = "sha256:" + "a" * 64
WHEEL_TAG = Tag("py3", "none", "linux_x86_64")


class FakeEntryPoint:
    group = "triton.backends"

    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = f"fixture_{name}:plugin"
        self._value = value
        self.load_calls = 0

    def load(self) -> Any:
        self.load_calls += 1
        if isinstance(self._value, BaseException):
            raise self._value
        return self._value


class FakeDistribution:
    """Small wheel-like distribution understood by the public validators."""

    def __init__(
        self,
        root: Path,
        *,
        entry_point: FakeEntryPoint,
        manifest: dict[str, Any],
        library_paths: tuple[str, ...],
        purelib: bool = False,
        platform_tag: str = "py3-none-linux_x86_64",
        record_hash_overrides: dict[str, str] | None = None,
        extra_files: tuple[str, ...] = (),
    ) -> None:
        self.root = root
        self.metadata = {"Name": f"t63-native-{entry_point.name}"}
        self.name = self.metadata["Name"]
        self.version = "1.0.0"
        self.entry_points = (entry_point,)
        self.manifest_path = "fixture/triton_anchor_backend.json"
        manifest_file = root / self.manifest_path
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        self.files = [self.manifest_path, *library_paths, *extra_files]
        hashes = {}
        for relative in library_paths:
            digest = hashlib.sha256((root / relative).read_bytes()).digest()
            encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
            hashes[relative] = "sha256=" + encoded
        hashes.update(record_hash_overrides or {})
        self._record = "".join(
            f"{relative},{hashes.get(relative, '')},\n" for relative in self.files
        )
        self._wheel = (
            "Wheel-Version: 1.0\n"
            f"Root-Is-Purelib: {'true' if purelib else 'false'}\n"
            f"Tag: {platform_tag}\n"
        )

    def locate_file(self, item: Any) -> Path:
        return self.root / str(item)

    def read_text(self, filename: str) -> str | None:
        if filename == "WHEEL":
            return self._wheel
        if filename == "RECORD":
            return self._record
        return None


def _compile_shared(
    root: Path,
    *,
    relative: str,
    symbol: str = "vendor_t63_entry",
    soname: str | None = None,
) -> Path:
    output = root / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output.with_suffix(".c")
    source.write_text(
        "__attribute__((visibility(\"default\"))) "
        f"int {symbol}(void) {{ return 63; }}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-fvisibility=hidden",
            f"-Wl,-soname,{soname or output.name}",
            "-o",
            str(output),
            str(source),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    return output


def _plugin_manifest(
    *,
    entry_point: str = "native",
    plugin_id: str = "vendor.native",
    library: str = "fixture/libvendor_native.so",
    fingerprint: str = FINGERPRINT,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "plugins": [
            {
                "plugin_id": plugin_id,
                "entry_point": entry_point,
                "backend_protocol": ">=1.0,<2.0",
                "requires_core": ">=0.2,<0.3",
                "requires_triton": {
                    "version": ">=3.3,<3.4",
                    "commit": TRITON_COMMIT,
                },
                "requires_llvm_version": ">=21,<22",
                "requires_llvm_commit": LLVM_COMMIT,
                "requires_mlir_version": ">=21,<22",
                "requires_mlir_commit": LLVM_COMMIT,
                "targets": ["native"],
                "isolation_mode": "native_in_process",
                "native_libraries": [library],
                "abi_fingerprint": fingerprint,
                "priority": 0,
            }
        ],
    }


def _native_plugin(
    *,
    entry_point: str = "native",
    plugin_id: str = "vendor.native",
    library: str = "fixture/libvendor_native.so",
    fingerprint: str = FINGERPRINT,
):
    """Construct evidence-only native metadata from a legal 1.0 record.

    The public parser must never accept this result.  ``dataclasses.replace``
    deliberately models an in-process caller bypassing structural parsing so
    low-level inspection and operational fail-closed guards can be tested.
    """
    document = _plugin_manifest(
        entry_point=entry_point,
        plugin_id=plugin_id,
        library=library,
        fingerprint=fingerprint,
    )
    wire_plugin = document["plugins"][0]
    wire_plugin["isolation_mode"] = "python_only"
    wire_plugin.pop("native_libraries")
    wire_plugin.pop("abi_fingerprint")
    parsed = parse_manifest(document).plugins[0]
    return replace(
        parsed,
        isolation_mode=PluginIsolationMode.NATIVE_IN_PROCESS,
        native_libraries=(library,),
        abi_fingerprint=fingerprint,
    )


def _python_only_document(
    *, entry_point: str = "native", plugin_id: str = "vendor.native"
) -> dict[str, Any]:
    document = _plugin_manifest(entry_point=entry_point, plugin_id=plugin_id)
    wire_plugin = document["plugins"][0]
    wire_plugin["isolation_mode"] = "python_only"
    wire_plugin.pop("native_libraries")
    wire_plugin.pop("abi_fingerprint")
    return document


class UntouchedDistribution:
    """Sentinel proving unsupported modes do not inspect package metadata."""

    def __init__(self) -> None:
        self.accesses: list[str] = []

    def __getattribute__(self, name: str) -> Any:
        if name in {"accesses", "__dict__", "__class__"}:
            return object.__getattribute__(self, name)
        object.__getattribute__(self, "accesses").append(name)
        raise AssertionError(f"unsupported isolation touched distribution.{name}")


def _crafted_native_registry(
    state: PluginLifecycleState,
    mode: PluginIsolationMode = PluginIsolationMode.NATIVE_IN_PROCESS,
) -> tuple[
    BackendPluginRegistry,
    BackendPluginRecord,
    FakeEntryPoint,
    UntouchedDistribution,
]:
    plugin = replace(
        _native_plugin(),
        isolation_mode=mode,
        native_libraries=(
            ("fixture/libvendor_native.so",)
            if mode is PluginIsolationMode.NATIVE_IN_PROCESS
            else ()
        ),
        abi_fingerprint=(
            FINGERPRINT
            if mode is PluginIsolationMode.NATIVE_IN_PROCESS
            else None
        ),
    )
    class NativeCompiler:
        supports_calls = 0
        constructor_calls = 0

        @classmethod
        def supports_target(cls, _target: Any) -> bool:
            cls.supports_calls += 1
            return True

        def __init__(self, _target: Any) -> None:
            type(self).constructor_calls += 1

    class NativeDriver:
        active_probe_calls = 0
        constructor_calls = 0

        @classmethod
        def is_active(cls) -> bool:
            cls.active_probe_calls += 1
            return True

        def __init__(self) -> None:
            type(self).constructor_calls += 1

    class NativePlugin:
        compiler_cls = NativeCompiler
        driver_cls = NativeDriver

        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    plugin_object = NativePlugin()
    entry_point = FakeEntryPoint("native", plugin_object)
    distribution = UntouchedDistribution()
    selected_targets = (
        ("native",)
        if state in {PluginLifecycleState.SELECTED, PluginLifecycleState.ACTIVE}
        else ()
    )
    record = BackendPluginRecord(
        record_id="crafted-native:native",
        entry_point_name="native",
        entry_point_value="fixture_native:plugin",
        distribution_name="crafted-native",
        distribution_version="1.0.0",
        source=PluginSource.MANIFEST,
        state=state,
        compatibility_status=PluginCompatibilityStatus.COMPATIBLE,
        entry_point=entry_point,
        distribution=distribution,
        manifest=plugin,
        compatibility_report=CompatibilityReport(
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            checks=(),
            compatible=True,
        ),
        plugin_object=plugin_object,
        compiler_cls=plugin_object.compiler_cls,
        driver_cls=plugin_object.driver_cls,
        initialized=state
        in {
            PluginLifecycleState.REGISTERED,
            PluginLifecycleState.SELECTED,
            PluginLifecycleState.ACTIVE,
        },
        selected_targets=selected_targets,
    )
    registry = BackendPluginRegistry(
        distribution_provider=lambda: (),
        environment_provider=lambda: (_ for _ in ()).throw(
            AssertionError("unsupported isolation touched CoreEnvironment")
        ),
    )
    registry._records = {record.record_id: record}
    registry._discovered = True
    if record.initialized:
        registry._cleanup_stack.append(record)
    if selected_targets:
        registry._selections["native"] = SelectionDecision(
            record_id=record.record_id,
            registry_key=record.registry_key,
            plugin_id=record.plugin_id,
            entry_point_name=record.entry_point_name,
            target="native",
            method=SelectionMethod.SOLE_CANDIDATE,
            selector=None,
            priority=record.manifest.priority,
            is_legacy=False,
            candidate_record_ids=(record.record_id,),
            capability_report=None,
            record=record,
        )
    return registry, record, entry_point, distribution


def _load_triton_backend_adapter(
    registry: BackendPluginRegistry,
    monkeypatch: pytest.MonkeyPatch,
):
    """Load the checked-out adapter without importing Triton's native module."""
    import triton_anchor.backends as backend_api

    package_dir = (
        Path(__file__).resolve().parents[2]
        / "triton/python/triton/backends"
    )
    module_name = f"_t63_triton_backends_{id(registry)}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(
        backend_api, "get_backend_plugin_registry", lambda: registry
    )
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _environment(*, fingerprint: str = FINGERPRINT) -> CoreEnvironment:
    return CoreEnvironment(
        core_version="0.2.0",
        build_info_generated=True,
        backend_protocol_version="1.0",
        manifest_schema_version="1.0",
        triton_version="3.3.0",
        vendored_triton_commit=TRITON_COMMIT,
        expected_llvm_project_commit=LLVM_COMMIT,
        actual_llvm_version_raw="21.0.0git",
        actual_llvm_version="21.0.0",
        actual_llvm_version_suffix="git",
        actual_llvm_commit=LLVM_COMMIT,
        actual_mlir_version_raw="21.0.0git",
        actual_mlir_version="21.0.0",
        actual_mlir_version_suffix="git",
        actual_mlir_commit=LLVM_COMMIT,
        cxx_standard="17",
        cxx_compiler_id="GNU",
        cxx_compiler_version="13.3.0",
        cxx11_abi="1",
        build_type="Release",
        ttgpu=False,
        built_python_version=platform.python_version(),
        built_python_soabi=sysconfig.get_config_var("SOABI"),
        built_platform=sysconfig.get_platform(),
        core_abi_fingerprint_schema="triton-anchor-core-abi-v1",
        core_library_sha256="sha256:" + "b" * 64,
        core_abi_fingerprint=fingerprint,
        runtime_python_version=platform.python_version(),
        runtime_python_implementation=platform.python_implementation(),
        runtime_python_soabi=sysconfig.get_config_var("SOABI"),
        runtime_platform=sysconfig.get_platform(),
        runtime_system=platform.system(),
        runtime_machine=platform.machine(),
    )


def _distribution(
    tmp_path: Path,
    *,
    entry_point: FakeEntryPoint | None = None,
    manifest: dict[str, Any] | None = None,
    symbol: str = "vendor_t63_entry",
    soname: str = "libvendor_native.so",
    record_hash_overrides: dict[str, str] | None = None,
    purelib: bool = False,
) -> FakeDistribution:
    relative = "fixture/libvendor_native.so"
    _compile_shared(tmp_path, relative=relative, symbol=symbol, soname=soname)
    entry_point = entry_point or FakeEntryPoint(
        "native",
        SimpleNamespace(compiler_cls=type("Compiler", (), {}), driver_cls=type("Driver", (), {})),
    )
    return FakeDistribution(
        tmp_path,
        entry_point=entry_point,
        manifest=manifest or _plugin_manifest(),
        library_paths=(relative,),
        purelib=purelib,
        record_hash_overrides=record_hash_overrides,
    )


def _python_distribution_with_native_file(
    tmp_path: Path,
    *,
    relative: str,
    payload: bytes,
    plugin_object: Any = None,
) -> tuple[FakeDistribution, FakeEntryPoint]:
    output = tmp_path / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    if plugin_object is None:
        plugin_object = SimpleNamespace(
            compiler_cls=type("PythonCompiler", (), {}),
            driver_cls=type("PythonDriver", (), {}),
        )
    entry_point = FakeEntryPoint("native", plugin_object)
    distribution = FakeDistribution(
        tmp_path,
        entry_point=entry_point,
        manifest=_python_only_document(),
        library_paths=(relative,),
    )
    return distribution, entry_point


def test_native_valid_elf_record_hash_soname_and_public_symbol(tmp_path: Path) -> None:
    distribution = _distribution(tmp_path)
    plugin = _native_plugin()
    report = inspect_native_artifacts(plugin, distribution)
    assert report.paths == ("fixture/libvendor_native.so",)
    artifact = report.artifacts[0]
    assert artifact.binary_format == "ELF"
    assert artifact.architecture == platform.machine()
    assert artifact.identity == "libvendor_native.so"
    assert "vendor_t63_entry" in artifact.exported_symbols

    with pytest.raises(BackendPluginManifestError) as caught:
        validate_backend_plugin(
            plugin,
            _environment(),
            distribution=distribution,
            core_abi_fingerprint=FINGERPRINT,
            supported_tags=(WHEEL_TAG,),
        )
    assert caught.value.field == "isolation_mode"
    assert caught.value.expected == "python_only"
    assert caught.value.actual == "native_in_process"


def test_native_record_tampering_is_rejected(tmp_path: Path) -> None:
    distribution = _distribution(
        tmp_path,
        record_hash_overrides={
            "fixture/libvendor_native.so": "sha256=" + "A" * 43,
        },
    )
    plugin = _native_plugin()
    with pytest.raises(BackendPluginManifestError) as caught:
        inspect_native_artifacts(plugin, distribution)
    assert caught.value.field == "RECORD"
    assert "mismatch" in str(caught.value).lower()


def test_native_in_process_cannot_hide_undeclared_library(tmp_path: Path) -> None:
    distribution = _distribution(tmp_path)
    hidden = _compile_shared(
        tmp_path,
        relative="fixture/libhidden.so",
        symbol="vendor_hidden",
    )
    distribution.files.append("fixture/libhidden.so")
    digest = hashlib.sha256(hidden.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    distribution._record += f"fixture/libhidden.so,sha256={encoded},\n"
    plugin = _native_plugin()
    with pytest.raises(BackendPluginManifestError) as caught:
        inspect_native_artifacts(plugin, distribution)
    assert caught.value.field == "native_libraries"
    assert "undeclared" in str(caught.value).lower()


def test_native_in_process_cannot_use_pure_python_wheel_tag(tmp_path: Path) -> None:
    distribution = _distribution(tmp_path, purelib=True)
    plugin = _native_plugin()
    with pytest.raises(BackendPluginCompatibilityError) as caught:
        inspect_native_artifacts(plugin, distribution)
    assert caught.value.dimension == "native wheel layout"


@pytest.mark.parametrize(
    "relative,payload",
    [
        ("fixture/backend.so", b"not-even-elf"),
        ("fixture/backend.dylib", b"not-even-macho"),
        ("fixture/backend.dll", b"not-even-pe"),
        ("fixture/backend.pyd", b"not-even-pe"),
        ("fixture/libbackend.so.1", b"versioned-native-name"),
        ("fixture/libbackend.dylib.1", b"versioned-native-name"),
        ("fixture/disguised.data", b"\x7fELF" + b"\0" * 16),
        ("fixture/disguised-macho.data", b"\xcf\xfa\xed\xfe" + b"\0" * 16),
        ("fixture/disguised-pe.data", b"MZ" + b"\0" * 18),
    ],
    ids=[
        "so",
        "dylib",
        "dll",
        "pyd",
        "versioned-so",
        "versioned-dylib",
        "elf-magic",
        "macho-magic",
        "pe-magic",
    ],
)
def test_python_only_distribution_rejects_native_inventory(
    tmp_path: Path,
    relative: str,
    payload: bytes,
) -> None:
    distribution, entry_point = _python_distribution_with_native_file(
        tmp_path,
        relative=relative,
        payload=payload,
    )
    plugin = parse_manifest(_python_only_document()).plugins[0]

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        validate_backend_plugin(
            plugin,
            _environment(),
            distribution=distribution,
            supported_tags=(WHEEL_TAG,),
        )
    assert caught.value.dimension == "python_only wheel contents"
    assert relative in caught.value.actual
    assert "native_in_process" not in (caught.value.remediation or "")
    assert entry_point.load_calls == 0


def test_python_only_native_inventory_blocks_registry_profiles_and_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Plugin:
        compiler_cls = type("PythonCompiler", (), {})
        driver_cls = type("PythonDriver", (), {})

        def __init__(self) -> None:
            self.initialize_calls = 0

        def initialize(self, _context: Any) -> None:
            self.initialize_calls += 1

    plugin_object = Plugin()
    distribution, entry_point = _python_distribution_with_native_file(
        tmp_path,
        relative="fixture/hidden.so.1",
        payload=b"native-by-name",
        plugin_object=plugin_object,
    )

    for profile in ("full", "triton_version"):
        registry = BackendPluginRegistry(
            distribution_provider=lambda: (distribution,),
            environment_provider=_environment,
            supported_tags=(WHEEL_TAG,),
            preflight_profile=profile,
        )
        record = registry.validate()[0]
        assert record.state is PluginLifecycleState.REJECTED
        assert record.compatibility_status is PluginCompatibilityStatus.INCOMPATIBLE
        assert isinstance(record.error, BackendPluginCompatibilityError)
        assert record.error.dimension == "python_only wheel contents"
        for callback in (
            lambda: registry.load(record.record_id),
            lambda: registry.register(record.record_id),
            lambda: registry.select(
                "native", explicit_selector=record.record_id
            ),
        ):
            with pytest.raises(BackendPluginCompatibilityError) as caught:
                callback()
            assert caught.value.dimension == "python_only wheel contents"
        assert registry.get_selection("native") is None

    triton_backends = _load_triton_backend_adapter(registry, monkeypatch)
    with pytest.raises(BackendPluginCompatibilityError) as caught:
        triton_backends.get_backend("native")
    assert caught.value.dimension == "python_only wheel contents"
    assert "native" not in triton_backends.backends
    assert entry_point.load_calls == 0
    assert plugin_object.initialize_calls == 0


def test_native_fingerprint_mismatch_rejected_before_import_or_initialize(
    tmp_path: Path,
) -> None:
    class Plugin:
        compiler_cls = type("Compiler", (), {})
        driver_cls = type("Driver", (), {})

        def __init__(self) -> None:
            self.initialize_calls = 0

        def initialize(self, _context: Any) -> None:
            self.initialize_calls += 1

    plugin_object = Plugin()
    entry_point = FakeEntryPoint("native", plugin_object)
    distribution = _distribution(tmp_path, entry_point=entry_point)
    environment_calls = 0

    def environment_provider() -> CoreEnvironment:
        nonlocal environment_calls
        environment_calls += 1
        return _environment()

    registry = BackendPluginRegistry(
        distribution_provider=lambda: (distribution,),
        environment_provider=environment_provider,
        core_abi_fingerprint="sha256:" + "c" * 64,
        supported_tags=(WHEEL_TAG,),
    )
    records = registry.validate()
    assert len(records) == 1
    record = records[0]
    assert record.state is PluginLifecycleState.REJECTED
    assert record.compatibility_status is PluginCompatibilityStatus.NOT_CHECKED
    assert isinstance(record.error, BackendPluginManifestError)
    assert record.error.field == "isolation_mode"
    assert record.error.plugin_id == "vendor.native"
    assert record.error.expected == "python_only"
    assert record.error.actual == "native_in_process"
    assert environment_calls == 0
    assert entry_point.load_calls == 0
    assert plugin_object.initialize_calls == 0


@pytest.mark.parametrize(
    "mode",
    [
        PluginIsolationMode.NATIVE_IN_PROCESS,
        PluginIsolationMode.SUBPROCESS,
    ],
    ids=["native-in-process", "subprocess"],
)
def test_unsupported_isolation_public_validators_fail_before_metadata(
    mode: PluginIsolationMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = replace(
        _native_plugin(),
        isolation_mode=mode,
        native_libraries=("fixture/libvendor_native.so",) if mode
        is PluginIsolationMode.NATIVE_IN_PROCESS else (),
        abi_fingerprint=FINGERPRINT if mode
        is PluginIsolationMode.NATIVE_IN_PROCESS else None,
    )
    distribution = UntouchedDistribution()

    def forbidden_inspector(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unsupported isolation reached native inspector")

    monkeypatch.setattr(
        "triton_anchor.backends.compatibility.inspect_native_artifacts",
        forbidden_inspector,
    )
    for validator in (
        lambda: validate_backend_plugin(
            plugin,
            object(),  # type: ignore[arg-type]
            distribution=distribution,
        ),
        lambda: validate_triton_requirement(
            plugin,
            object(),  # type: ignore[arg-type]
        ),
    ):
        with pytest.raises(BackendPluginManifestError) as caught:
            validator()
        assert caught.value.to_dict() == {
            "code": "backend_plugin_manifest_error",
            "message": str(caught.value),
            "plugin_id": "vendor.native",
            "entry_point": "native",
            "field": "isolation_mode",
            "dimension": None,
            "expected": "python_only",
            "actual": mode.value,
            "remediation": caught.value.remediation,
            "detail": None,
        }
        assert caught.value.remediation
    assert distribution.accesses == []


@pytest.mark.parametrize(
    "state",
    [
        PluginLifecycleState.DISCOVERED,
        PluginLifecycleState.VALIDATED,
        PluginLifecycleState.LOADED,
        PluginLifecycleState.REGISTERED,
        PluginLifecycleState.SELECTED,
        PluginLifecycleState.ACTIVE,
    ],
    ids=lambda state: state.value,
)
@pytest.mark.parametrize(
    "mode",
    [
        PluginIsolationMode.NATIVE_IN_PROCESS,
        PluginIsolationMode.SUBPROCESS,
    ],
    ids=lambda mode: mode.value,
)
def test_unsupported_isolation_is_rejected_from_every_manifest_state(
    state: PluginLifecycleState,
    mode: PluginIsolationMode,
) -> None:
    registry, record, entry_point, distribution = _crafted_native_registry(
        state, mode
    )
    had_selection = registry.get_selection("native") is not None

    with pytest.raises(BackendPluginManifestError) as caught:
        registry.validate(record.record_id)
    assert caught.value.field == "isolation_mode"
    assert caught.value.expected == "python_only"
    assert caught.value.actual == mode.value

    rejected = registry.inspect(record.record_id)
    assert rejected.state is PluginLifecycleState.REJECTED
    assert rejected.compatibility_status is PluginCompatibilityStatus.NOT_CHECKED
    assert rejected.error is caught.value
    assert rejected.plugin_object is None
    assert rejected.compiler_cls is None
    assert rejected.driver_cls is None
    assert not rejected.initialized
    assert rejected.selected_targets == ()
    assert registry.get_selection("native") is None
    assert registry.generation == (1 if had_selection else 0)
    assert registry.reset() == ()
    assert record.plugin_object.shutdown_calls == 0
    assert entry_point.load_calls == 0
    assert distribution.accesses == []


@pytest.mark.parametrize(
    "operation",
    ["validate", "load", "register", "select", "activate"],
)
@pytest.mark.parametrize(
    "mode",
    [
        PluginIsolationMode.NATIVE_IN_PROCESS,
        PluginIsolationMode.SUBPROCESS,
    ],
    ids=lambda mode: mode.value,
)
def test_registry_operational_apis_reject_crafted_native_active_record(
    operation: str,
    mode: PluginIsolationMode,
) -> None:
    registry, record, entry_point, distribution = _crafted_native_registry(
        PluginLifecycleState.ACTIVE, mode
    )
    callbacks = {
        "validate": lambda: registry.validate(record.record_id),
        "load": lambda: registry.load(record.record_id),
        "register": lambda: registry.register(record.record_id),
        "select": lambda: registry.select(
            "native", explicit_selector=record.record_id
        ),
        "activate": lambda: registry.activate(record.record_id),
    }

    with pytest.raises(BackendPluginManifestError) as caught:
        callbacks[operation]()
    assert caught.value.field == "isolation_mode"
    assert registry.inspect(record.record_id).state is PluginLifecycleState.REJECTED
    assert registry.get_selection("native") is None
    assert entry_point.load_calls == 0
    assert distribution.accesses == []


@pytest.mark.parametrize(
    "mode",
    [
        PluginIsolationMode.NATIVE_IN_PROCESS,
        PluginIsolationMode.SUBPROCESS,
    ],
    ids=lambda mode: mode.value,
)
def test_exported_selection_rejects_crafted_native_compatible_record(
    mode: PluginIsolationMode,
) -> None:
    registry, record, entry_point, distribution = _crafted_native_registry(
        PluginLifecycleState.VALIDATED, mode
    )
    del registry
    with pytest.raises(BackendPluginManifestError) as caught:
        select_backend(
            (record,),
            target="native",
            explicit_selector=record.record_id,
        )
    assert caught.value.field == "isolation_mode"
    assert entry_point.load_calls == 0
    assert distribution.accesses == []


def test_registry_isolation_guard_does_not_trust_forged_source_metadata() -> None:
    registry, record, entry_point, distribution = _crafted_native_registry(
        PluginLifecycleState.VALIDATED
    )
    forged = replace(record, source=None)
    registry._records[record.record_id] = forged

    with pytest.raises(BackendPluginManifestError) as caught:
        registry.load(record.record_id)
    assert caught.value.field == "isolation_mode"
    assert registry.inspect(record.record_id).state is PluginLifecycleState.REJECTED
    assert entry_point.load_calls == 0
    assert distribution.accesses == []


@pytest.mark.parametrize(
    "operation",
    ["get_backend", "make_backend", "get_driver_backends", "activate_backend"],
)
@pytest.mark.parametrize(
    "mode",
    [
        PluginIsolationMode.NATIVE_IN_PROCESS,
        PluginIsolationMode.SUBPROCESS,
    ],
    ids=lambda mode: mode.value,
)
def test_triton_cached_decision_revalidates_and_prunes_crafted_native(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    mode: PluginIsolationMode,
) -> None:
    registry, record, entry_point, distribution = _crafted_native_registry(
        PluginLifecycleState.ACTIVE, mode
    )
    triton_backends = _load_triton_backend_adapter(registry, monkeypatch)
    decision = registry.get_selection("native")
    assert decision is not None
    stale = triton_backends.Backend(
        compiler=record.compiler_cls,
        driver=record.driver_cls,
        record_id=record.record_id,
        plugin_id=record.plugin_id,
        entry_point_name=record.entry_point_name,
    )
    monkeypatch.setattr(triton_backends, "_registry", lambda: registry)
    monkeypatch.delenv("TRITON_ANCHOR_BACKEND", raising=False)
    triton_backends.backends[record.entry_point_name] = stale

    callbacks = {
        "get_backend": lambda: triton_backends.get_backend("native"),
        "make_backend": lambda: triton_backends.make_backend("native"),
        "get_driver_backends": triton_backends.get_driver_backends,
        "activate_backend": lambda: triton_backends.activate_backend(stale),
    }
    with pytest.raises(BackendPluginManifestError) as caught:
        callbacks[operation]()
    assert caught.value.field == "isolation_mode"
    assert record.entry_point_name not in triton_backends.backends
    assert registry.get_selection("native") is None
    assert registry.inspect(record.record_id).state is PluginLifecycleState.REJECTED
    assert record.compiler_cls.supports_calls == 0
    assert record.compiler_cls.constructor_calls == 0
    assert record.driver_cls.active_probe_calls == 0
    assert record.driver_cls.constructor_calls == 0
    assert entry_point.load_calls == 0
    assert distribution.accesses == []


@dataclass(frozen=True)
class ConflictRecord:
    record_id: str
    plugin_id: str
    entry_point_name: str
    manifest: Any
    compatibility_report: CompatibilityReport
    state: PluginLifecycleState = PluginLifecycleState.VALIDATED
    compatibility_status: PluginCompatibilityStatus = (
        PluginCompatibilityStatus.COMPATIBLE
    )
    errors: tuple[Any, ...] = ()


def _native_conflict_record(
    record_id: str,
    *,
    plugin_id: str,
    entry_point: str,
    target: str,
    soname: str,
    symbols: tuple[str, ...],
) -> ConflictRecord:
    manifest = SimpleNamespace(
        targets=(target,), isolation_mode=PluginIsolationMode.NATIVE_IN_PROCESS
    )
    artifact = NativeArtifact(
        path=f"fixture/{soname}",
        sha256="0" * 64,
        binary_format="ELF",
        architecture=platform.machine(),
        identity=soname,
        needed_libraries=(),
        exported_symbols=symbols,
    )
    return ConflictRecord(
        record_id=record_id,
        plugin_id=plugin_id,
        entry_point_name=entry_point,
        manifest=manifest,
        compatibility_report=CompatibilityReport(
            plugin_id=plugin_id,
            entry_point=entry_point,
            checks=(),
            native_artifacts=(artifact,),
        ),
    )


def test_two_native_plugins_duplicate_soname_and_symbol_are_fatal() -> None:
    first = _native_conflict_record(
        "dist-a:a",
        plugin_id="vendor.a",
        entry_point="a",
        target="a",
        soname="libcollision.so",
        symbols=("vendor_shared_entry",),
    )
    second = _native_conflict_record(
        "dist-b:b",
        plugin_id="vendor.b",
        entry_point="b",
        target="b",
        soname="libcollision.so",
        symbols=("vendor_shared_entry",),
    )
    report = detect_conflicts((second, first))
    kinds = {conflict.kind.value for conflict in report.fatal_conflicts}
    assert kinds == {"duplicate_native_identity", "duplicate_exported_symbol"}
    for conflict in report.fatal_conflicts:
        assert conflict.record_ids == ("dist-a:a", "dist-b:b")
        diagnostic = conflict.to_error().to_dict()
        assert diagnostic["plugin_id"] is None
        assert diagnostic["expected"]
        assert diagnostic["actual"]

    registry = BackendPluginRegistry(distribution_provider=lambda: ())
    registry._records = {
        record.record_id: record for record in (second, first)
    }
    registry._discovered = True
    registry._reject_all_fatal_conflicts((second, first))
    rejected = {record.record_id: record for record in registry.list()}
    for record in rejected.values():
        diagnostics = [error.to_dict() for error in record.errors]
        assert [item["conflict_kind"] for item in diagnostics] == [
            "duplicate_native_identity",
            "duplicate_exported_symbol",
        ]
        assert all(item["plugin_id"] == record.plugin_id for item in diagnostics)
        assert all(
            item["entry_point"] == record.entry_point_name
            for item in diagnostics
        )
        assert all(
            item["related_record_ids"]
            == sorted(set(rejected) - {record.record_id})
            for item in diagnostics
        )
        assert all(
            item["related_plugin_ids"]
            == sorted(
                other.plugin_id
                for other in rejected.values()
                if other.record_id != record.record_id
            )
            for item in diagnostics
        )

    registry._reject_all_fatal_conflicts(registry.list())
    assert all(len(record.errors) == 2 for record in registry.list())


def test_vendor_qualified_unique_native_symbols_do_not_conflict() -> None:
    first = _native_conflict_record(
        "dist-a:a",
        plugin_id="vendor.a",
        entry_point="a",
        target="a",
        soname="libvendor_a.so",
        symbols=("vendor_a_entry",),
    )
    second = _native_conflict_record(
        "dist-b:b",
        plugin_id="vendor.b",
        entry_point="b",
        target="b",
        soname="libvendor_b.so",
        symbols=("vendor_b_entry",),
    )
    assert detect_conflicts((first, second)).fatal_conflicts == ()


def _abi_fingerprint(build_info: dict[str, Any]) -> str:
    keys = (
        "core_version",
        "vendored_triton_commit",
        "actual_llvm_version_raw",
        "actual_llvm_commit",
        "actual_mlir_version_raw",
        "actual_mlir_commit",
        "cxx_standard",
        "cxx_compiler_id",
        "cxx_compiler_version",
        "cxx11_abi",
        "built_python_soabi",
        "built_platform",
        "ttgpu",
        "core_library_sha256",
    )
    payload = {
        "schema": "triton-anchor-core-abi-v1",
        "material": {key: build_info[key] for key in keys},
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def test_generated_build_info_detects_compiler_abi_material_tampering(
    tmp_path: Path,
) -> None:
    info = {
        "schema_version": "1.1",
        "generated": True,
        "core_version": "0.2.0",
        "backend_protocol_version": "1.0",
        "manifest_schema_version": "1.0",
        "triton_version": "3.3.0",
        "vendored_triton_commit": TRITON_COMMIT,
        "expected_llvm_project_commit": LLVM_COMMIT,
        "actual_llvm_version_raw": "21.0.0git",
        "actual_llvm_commit": LLVM_COMMIT,
        "actual_mlir_version_raw": "21.0.0git",
        "actual_mlir_commit": LLVM_COMMIT,
        "cxx_standard": "17",
        "cxx_compiler_id": "GNU",
        "cxx_compiler_version": "13.3.0",
        "cxx11_abi": "1",
        "build_type": "Release",
        "ttgpu": False,
        "built_python_version": platform.python_version(),
        "built_python_soabi": sysconfig.get_config_var("SOABI"),
        "built_platform": sysconfig.get_platform(),
        "core_abi_fingerprint_schema": "triton-anchor-core-abi-v1",
        "core_library_sha256": "sha256:" + "d" * 64,
    }
    info["core_abi_fingerprint"] = _abi_fingerprint(info)
    path = tmp_path / "_build_info.json"
    path.write_text(json.dumps(info), encoding="utf-8")
    assert load_build_info(path)["cxx_compiler_id"] == "GNU"

    # The fingerprint is an integrity boundary: changing compiler/stdlib ABI
    # material without regenerating it must fail closed.
    info["cxx_compiler_id"] = "Clang"
    path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint does not match"):
        load_build_info(path)


def test_native_required_symbol_name_is_not_defined_by_manifest_1_0(
    tmp_path: Path,
) -> None:
    """No invented symbol can turn unsupported native metadata loadable."""

    distribution = _distribution(tmp_path, symbol="arbitrary_vendor_export")
    plugin = _native_plugin()
    artifact = inspect_native_artifacts(plugin, distribution).artifacts[0]
    assert artifact.exported_symbols == ("arbitrary_vendor_export",)
    with pytest.raises(BackendPluginManifestError) as caught:
        validate_backend_plugin(
            plugin,
            _environment(),
            distribution=distribution,
            core_abi_fingerprint=FINGERPRINT,
            supported_tags=(WHEEL_TAG,),
        )
    assert caught.value.field == "isolation_mode"
    assert caught.value.expected == "python_only"


def test_native_plugin_cxx_abi_is_not_attested_by_manifest_1_0(
    tmp_path: Path,
) -> None:
    """Self-asserted Core fingerprints never authorize native Protocol 1.0."""

    artifacts = []
    for abi in ("0", "1"):
        root = tmp_path / f"abi-{abi}"
        relative = "fixture/libvendor_native.so"
        output = root / relative
        output.parent.mkdir(parents=True)
        source = output.with_suffix(".cc")
        source.write_text(
            "#include <string>\n"
            "extern \"C\" __attribute__((visibility(\"default\")))\n"
            "unsigned long vendor_cxx_string_size(void) {\n"
            "  return std::string(\"t63\").size();\n"
            "}\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                "c++",
                "-shared",
                "-fPIC",
                "-fvisibility=hidden",
                f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
                "-Wl,-soname,libvendor_native.so",
                "-o",
                str(output),
                str(source),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout
        entry_point = FakeEntryPoint(f"native-{abi}", object())
        manifest = _plugin_manifest(
            entry_point=f"native-{abi}", plugin_id=f"vendor.native.{abi}"
        )
        distribution = FakeDistribution(
            root,
            entry_point=entry_point,
            manifest=manifest,
            library_paths=(relative,),
        )
        plugin = _native_plugin(
            entry_point=f"native-{abi}", plugin_id=f"vendor.native.{abi}"
        )
        artifacts.append(inspect_native_artifacts(plugin, distribution).artifacts[0])
        with pytest.raises(BackendPluginManifestError) as caught:
            validate_backend_plugin(
                plugin,
                _environment(),
                distribution=distribution,
                core_abi_fingerprint=FINGERPRINT,
                supported_tags=(WHEEL_TAG,),
            )
        assert caught.value.field == "isolation_mode"

    assert all("vendor_cxx_string_size" in item.exported_symbols for item in artifacts)
    assert artifacts[0].sha256 != artifacts[1].sha256


def test_native_static_inspection_does_not_prove_dlopen_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static evidence is non-operational and the host loader stays untouched."""

    relative = "fixture/libvendor_native.so"
    output = tmp_path / relative
    output.parent.mkdir(parents=True)
    source = output.with_suffix(".c")
    source.write_text(
        "extern int t63_symbol_that_does_not_exist(void);\n"
        "__attribute__((visibility(\"default\")))\n"
        "int vendor_t63_entry(void) {\n"
        "  return t63_symbol_that_does_not_exist();\n"
        "}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-fvisibility=hidden",
            "-Wl,-soname,libvendor_native.so",
            "-o",
            str(output),
            str(source),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    entry_point = FakeEntryPoint("native", object())
    manifest = _plugin_manifest()
    distribution = FakeDistribution(
        tmp_path,
        entry_point=entry_point,
        manifest=manifest,
        library_paths=(relative,),
    )
    plugin = _native_plugin()
    artifact = inspect_native_artifacts(plugin, distribution).artifacts[0]
    assert "vendor_t63_entry" in artifact.exported_symbols
    dlopen_calls = []

    def forbidden_dlopen(*args: Any, **kwargs: Any) -> None:
        dlopen_calls.append((args, kwargs))
        raise AssertionError("operational rejection must not call ctypes.CDLL")

    monkeypatch.setattr(ctypes, "CDLL", forbidden_dlopen)
    with pytest.raises(BackendPluginManifestError) as caught:
        validate_backend_plugin(
            plugin,
            _environment(),
            distribution=distribution,
            core_abi_fingerprint=FINGERPRINT,
            supported_tags=(WHEEL_TAG,),
        )
    assert caught.value.field == "isolation_mode"
    assert dlopen_calls == []
