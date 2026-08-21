"""Independent native ABI acceptance fixtures for T6.3.

These tests build tiny ELF shared objects at runtime.  They exercise the
public Manifest, compatibility, native inspection, conflict, and Registry
APIs without importing a real hardware backend or relying on T10.2.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import sysconfig
from collections import UserDict
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import triton_anchor.backends.native as native_module
from packaging.tags import Tag
from triton_anchor.backends import (
    BACKEND_SELECTOR_ENV,
    BackendPluginCompatibilityError,
    BackendPluginManifest,
    BackendPluginManifestError,
    BackendPluginRecord,
    BackendPluginRegistry,
    BackendPluginSelectionError,
    CompatibilityReport,
    CoreEnvironment,
    NativeArtifact,
    PluginCompatibilityStatus,
    PluginIsolationMode,
    PluginLifecycleState,
    PluginSource,
    TritonRequirement,
    detect_conflicts,
    inspect_native_artifacts,
    load_build_info,
    operational_record_manifest_error,
    parse_manifest,
    select_backend,
    unsupported_isolation_mode_error,
    validate_backend_plugin,
    validate_triton_requirement,
)

TRITON_COMMIT = "6cc4505027d7b39fe18a44a7f89085b8babb7400"
LLVM_COMMIT = "a992f29451b9e140424f35ac5e20177db4afbdc0"
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
        f'__attribute__((visibility("default"))) int {symbol}(void) {{ return 63; }}\n',
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
                    "version": ">=3.6,<3.7",
                    "commit": TRITON_COMMIT,
                },
                "requires_llvm_version": ">=22,<23",
                "requires_llvm_commit": LLVM_COMMIT,
                "requires_mlir_version": ">=22,<23",
                "requires_mlir_commit": LLVM_COMMIT,
                "targets": ["native"],
                "isolation_mode": "native_in_process",
                "native_libraries": [library],
                "abi_fingerprint": fingerprint,
                "priority": 0,
            }
        ],
    }


def _python_manifest(
    *,
    entry_point: str = "python",
    plugin_id: str = "vendor.python",
) -> dict[str, Any]:
    manifest = _plugin_manifest(
        entry_point=entry_point,
        plugin_id=plugin_id,
    )
    record = manifest["plugins"][0]
    record["targets"] = ["python"]
    record["isolation_mode"] = "python_only"
    record.pop("native_libraries")
    record.pop("abi_fingerprint")
    return manifest


def _native_plugin(
    *,
    entry_point: str = "native",
    plugin_id: str = "vendor.native",
    library: str = "fixture/libvendor_native.so",
    fingerprint: str = FINGERPRINT,
    isolation_mode: Any = PluginIsolationMode.NATIVE_IN_PROCESS,
) -> BackendPluginManifest:
    """Construct evidence-only native metadata without using the parser."""
    return BackendPluginManifest(
        plugin_id=plugin_id,
        entry_point=entry_point,
        backend_protocol=">=1.0,<2.0",
        requires_core=">=0.2,<0.3",
        requires_triton=TritonRequirement(version=">=3.6,<3.7", commit=TRITON_COMMIT),
        requires_llvm_version=">=22,<23",
        requires_llvm_commit=LLVM_COMMIT,
        requires_mlir_version=">=22,<23",
        requires_mlir_commit=LLVM_COMMIT,
        targets=("native",),
        isolation_mode=isolation_mode,
        native_libraries=(library,),
        abi_fingerprint=fingerprint,
    )


def _python_plugin(
    *,
    entry_point: str = "python",
    plugin_id: str = "vendor.python",
) -> BackendPluginManifest:
    return replace(
        _native_plugin(entry_point=entry_point, plugin_id=plugin_id),
        isolation_mode=PluginIsolationMode.PYTHON_ONLY,
        native_libraries=(),
        abi_fingerprint=None,
        targets=("python",),
    )


def _environment(*, fingerprint: str = FINGERPRINT) -> CoreEnvironment:
    return CoreEnvironment(
        core_version="0.2.0",
        build_info_generated=True,
        backend_protocol_version="1.0",
        manifest_schema_version="1.0",
        triton_version="3.6.0",
        vendored_triton_commit=TRITON_COMMIT,
        expected_llvm_project_commit=LLVM_COMMIT,
        actual_llvm_version_raw="22.0.0git",
        actual_llvm_version="22.0.0",
        actual_llvm_version_suffix="git",
        actual_llvm_commit=LLVM_COMMIT,
        actual_mlir_version_raw="22.0.0git",
        actual_mlir_version="22.0.0",
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
        SimpleNamespace(
            compiler_cls=type("Compiler", (), {}), driver_cls=type("Driver", (), {})
        ),
    )
    return FakeDistribution(
        tmp_path,
        entry_point=entry_point,
        manifest=manifest or _plugin_manifest(),
        library_paths=(relative,),
        purelib=purelib,
        record_hash_overrides=record_hash_overrides,
    )


def _assert_explicit_rejection(
    error: BackendPluginManifestError,
    *,
    actual: str = "native_in_process",
) -> None:
    assert error.code == "backend_plugin_manifest_error"
    assert error.field == "isolation_mode"
    assert error.expected == "python_only"
    assert error.actual == actual
    assert error.plugin_id is not None
    assert error.remediation


def _assert_missing_manifest_rejection(
    error: BackendPluginManifestError,
) -> None:
    assert error.code == "backend_plugin_manifest_error"
    assert error.field == "manifest"
    assert error.expected == "a valid parsed BackendPluginManifest"
    assert error.actual == "<missing>"
    assert error.entry_point == "native"
    assert error.remediation


def _runtime_sentinels() -> tuple[dict[str, int], FakeEntryPoint, Any, type, type]:
    calls = {
        "construct": 0,
        "initialize": 0,
        "supports_target": 0,
        "compiler_construct": 0,
        "compile": 0,
        "diagnostics": 0,
        "shutdown": 0,
    }

    class Compiler:
        @staticmethod
        def supports_target(_target: Any) -> bool:
            calls["supports_target"] += 1
            return True

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            calls["compiler_construct"] += 1

        def compile(self, *_args: Any, **_kwargs: Any) -> None:
            calls["compile"] += 1

    class Driver:
        pass

    class Plugin:
        compiler_cls = Compiler
        driver_cls = Driver

        def __init__(self) -> None:
            calls["construct"] += 1

        def initialize(self, _context: Any) -> None:
            calls["initialize"] += 1

        def diagnostics(self) -> dict[str, bool]:
            calls["diagnostics"] += 1
            return {"unexpected": True}

        def shutdown(self) -> None:
            calls["shutdown"] += 1

    # Model a caller-forged already-loaded object without invoking __init__.
    preloaded = object.__new__(Plugin)
    return calls, FakeEntryPoint("native", Plugin), preloaded, Compiler, Driver


class _BombDistribution:
    @property
    def files(self) -> Any:
        raise AssertionError("unsupported manifest reached distribution inventory")

    def locate_file(self, _item: Any) -> Path:
        raise AssertionError("unsupported manifest reached distribution lookup")

    def read_text(self, _filename: str) -> str:
        raise AssertionError("unsupported manifest reached wheel metadata")


class _ReprBomb:
    def __init__(self) -> None:
        self.repr_calls = 0

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise AssertionError("manifest rejection invoked arbitrary __repr__")


class _IterBomb(_ReprBomb):
    def __init__(self) -> None:
        super().__init__()
        self.iter_calls = 0

    def __iter__(self) -> Any:
        self.iter_calls += 1
        raise AssertionError("manifest diagnostics invoked arbitrary __iter__")


def _forged_native_record(
    plugin: BackendPluginManifest,
    *,
    state: PluginLifecycleState,
    entry_point: FakeEntryPoint,
    plugin_object: Any,
    compiler_cls: type,
    driver_cls: type,
    distribution: Any = None,
) -> BackendPluginRecord:
    old_errors: tuple[Any, ...] = ()
    if state is PluginLifecycleState.REJECTED:
        old_errors = (
            BackendPluginCompatibilityError(
                "forged old error",
                "old expected",
                "old actual",
                plugin_id=plugin.plugin_id,
                entry_point=plugin.entry_point,
            ),
        )
    return BackendPluginRecord(
        record_id="forged-native:native",
        entry_point_name=plugin.entry_point,
        entry_point_value=entry_point.value,
        distribution_name="forged-native",
        distribution_version="1.0.0",
        source=PluginSource.MANIFEST,
        state=state,
        compatibility_status=PluginCompatibilityStatus.COMPATIBLE,
        entry_point=entry_point,
        distribution=(_BombDistribution() if distribution is None else distribution),
        manifest=plugin,
        compatibility_report=CompatibilityReport(
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            checks=(),
            compatible=True,
        ),
        errors=old_errors,
        plugin_object=plugin_object,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        initialized=True,
        shutdown_called=True,
        selected_targets=("native",),
    )


def _registry_with_forged_record(
    record: BackendPluginRecord,
) -> BackendPluginRegistry:
    registry = BackendPluginRegistry(
        distribution_provider=lambda: (),
        environment_provider=_environment,
        supported_tags=(WHEEL_TAG,),
    )
    registry._records = {record.record_id: record}
    registry._discovered = True
    return registry


def _forged_parser_rejected_record(
    *,
    state: PluginLifecycleState,
    entry_point: FakeEntryPoint,
    plugin_object: Any,
    compiler_cls: type,
    driver_cls: type,
) -> BackendPluginRecord:
    error = unsupported_isolation_mode_error(
        "native_in_process",
        plugin_id="vendor.native",
        entry_point="native",
    )
    return replace(
        _forged_native_record(
            _native_plugin(),
            state=state,
            entry_point=entry_point,
            plugin_object=plugin_object,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        ),
        manifest=None,
        errors=(error,),
    )


def _forged_missing_manifest_record(
    *,
    state: PluginLifecycleState,
    entry_point: FakeEntryPoint,
    plugin_object: Any,
    compiler_cls: type,
    driver_cls: type,
) -> BackendPluginRecord:
    return replace(
        _forged_native_record(
            _native_plugin(),
            state=state,
            entry_point=entry_point,
            plugin_object=plugin_object,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        ),
        manifest=None,
        errors=(),
    )


def _assert_rejected_record(record: BackendPluginRecord) -> None:
    assert record.state is PluginLifecycleState.REJECTED
    assert record.compatibility_status is PluginCompatibilityStatus.NOT_CHECKED
    assert isinstance(record.error, BackendPluginManifestError)
    _assert_explicit_rejection(record.error)
    assert record.compatibility_report is None
    assert record.capability_report is None
    assert record.plugin_object is None
    assert record.compiler_cls is None
    assert record.driver_cls is None
    assert not record.initialized
    assert not record.shutdown_called
    assert record.selected_targets == ()


def _assert_missing_manifest_record(record: BackendPluginRecord) -> None:
    assert record.state is PluginLifecycleState.REJECTED
    assert record.compatibility_status is PluginCompatibilityStatus.NOT_CHECKED
    assert isinstance(record.error, BackendPluginManifestError)
    _assert_missing_manifest_rejection(record.error)
    assert record.compatibility_report is None
    assert record.capability_report is None
    assert record.plugin_object is None
    assert record.compiler_cls is None
    assert record.driver_cls is None
    assert not record.initialized
    assert not record.shutdown_called
    assert record.selected_targets == ()


def _assert_all_operational_paths_reject(
    plugin: BackendPluginManifest,
    *,
    distribution: Any = None,
) -> None:
    calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
    operational_distribution = (
        _BombDistribution() if distribution is None else distribution
    )

    with pytest.raises(BackendPluginManifestError) as caught:
        validate_backend_plugin(
            plugin,
            _environment(),
            distribution=operational_distribution,
            core_abi_fingerprint=FINGERPRINT,
            supported_tags=(WHEEL_TAG,),
        )
    _assert_explicit_rejection(caught.value)

    with pytest.raises(BackendPluginManifestError) as caught:
        validate_triton_requirement(plugin, _environment())
    _assert_explicit_rejection(caught.value)

    selectable = _forged_native_record(
        plugin,
        state=PluginLifecycleState.VALIDATED,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        distribution=operational_distribution,
    )
    with pytest.raises(BackendPluginManifestError) as caught:
        select_backend((selectable,), target="native")
    _assert_explicit_rejection(caught.value)

    operations = {
        "validate": lambda registry, record: registry.validate(record.record_id),
        "load": lambda registry, record: registry.load(record.record_id),
        "register": lambda registry, record: registry.register(record.record_id),
        "select": lambda registry, _record: registry.select("native"),
        "activate": lambda registry, record: registry.activate(record.record_id),
    }
    for operation in operations.values():
        for state in PluginLifecycleState:
            record = _forged_native_record(
                plugin,
                state=state,
                entry_point=entry_point,
                plugin_object=preloaded,
                compiler_cls=compiler_cls,
                driver_cls=driver_cls,
                distribution=operational_distribution,
            )
            registry = _registry_with_forged_record(record)
            with pytest.raises(BackendPluginManifestError) as caught:
                operation(registry, record)
            _assert_explicit_rejection(caught.value)
            _assert_rejected_record(registry.inspect(record.record_id))
            assert registry.get_selection("native") is None

    diagnostics_record = _forged_native_record(
        plugin,
        state=PluginLifecycleState.ACTIVE,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        distribution=operational_distribution,
    )
    diagnostics_registry = _registry_with_forged_record(diagnostics_record)
    diagnostics = diagnostics_registry.diagnostics(diagnostics_record.record_id)
    assert diagnostics["state"] == "rejected"
    assert diagnostics["plugin_diagnostics"] is None
    assert diagnostics["errors"][0]["field"] == "isolation_mode"

    reset_record = _forged_native_record(
        plugin,
        state=PluginLifecycleState.ACTIVE,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        distribution=operational_distribution,
    )
    reset_registry = _registry_with_forged_record(reset_record)
    reset_registry._cleanup_stack = [reset_record]
    assert reset_registry.reset() == ()

    assert entry_point.load_calls == 0
    assert calls == {
        "construct": 0,
        "initialize": 0,
        "supports_target": 0,
        "compiler_construct": 0,
        "compile": 0,
        "diagnostics": 0,
        "shutdown": 0,
    }


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
    registry = BackendPluginRegistry(
        distribution_provider=lambda: (distribution,),
        environment_provider=_environment,
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
    assert entry_point.load_calls == 0
    assert plugin_object.initialize_calls == 0


def test_manual_manifest_shape_cannot_bypass_operational_public_apis() -> None:
    parsed_python = parse_manifest(_python_manifest()).plugins[0]
    variants = (
        (
            replace(parsed_python, isolation_mode="python_only"),
            "isolation_mode",
            "python_only",
        ),
        (
            replace(parsed_python, isolation_mode="native_in_process"),
            "isolation_mode",
            "native_in_process",
        ),
        (
            replace(
                parsed_python,
                isolation_mode=PluginIsolationMode.SUBPROCESS,
            ),
            "isolation_mode",
            "subprocess",
        ),
        (
            replace(
                parsed_python,
                native_libraries=("fixture/forged.so",),
            ),
            "native_libraries",
            None,
        ),
        (
            replace(parsed_python, abi_fingerprint=FINGERPRINT),
            "abi_fingerprint",
            None,
        ),
    )
    for plugin, expected_field, expected_actual in variants:
        with pytest.raises(BackendPluginManifestError) as caught:
            validate_backend_plugin(
                plugin,
                _environment(),
                distribution=_BombDistribution(),
                supported_tags=(WHEEL_TAG,),
            )
        assert caught.value.field == expected_field
        if expected_actual is not None:
            assert caught.value.actual == expected_actual

        with pytest.raises(BackendPluginManifestError) as caught:
            validate_triton_requirement(plugin, _environment())
        assert caught.value.field == expected_field

        calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
        record = _forged_native_record(
            plugin,
            state=PluginLifecycleState.ACTIVE,
            entry_point=entry_point,
            plugin_object=preloaded,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        )
        with pytest.raises(BackendPluginManifestError) as caught:
            select_backend((record,), target=plugin.targets[0])
        assert caught.value.field == expected_field

        registry = _registry_with_forged_record(record)
        with pytest.raises(BackendPluginManifestError) as caught:
            registry.validate(record.record_id)
        assert caught.value.field == expected_field
        rejected = registry.inspect(record.record_id)
        assert rejected.state is PluginLifecycleState.REJECTED
        assert rejected.compatibility_status is PluginCompatibilityStatus.NOT_CHECKED
        assert rejected.error is not None
        assert rejected.error.field == expected_field
        assert entry_point.load_calls == 0
        assert all(value == 0 for value in calls.values())


def test_standalone_selector_reports_isolation_before_forged_priority() -> None:
    plugin = replace(_native_plugin(), priority=True)
    calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
    record = _forged_native_record(
        plugin,
        state=PluginLifecycleState.VALIDATED,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    with pytest.raises(BackendPluginManifestError) as caught:
        select_backend((record,), target="native")
    _assert_explicit_rejection(caught.value)
    assert entry_point.load_calls == 0
    assert all(value == 0 for value in calls.values())


def test_parser_rejected_shape_cannot_be_forged_into_any_lifecycle_state() -> None:
    calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
    operations = {
        "validate": lambda registry, record: registry.validate(record.record_id),
        "load": lambda registry, record: registry.load(record.record_id),
        "register": lambda registry, record: registry.register(record.record_id),
        "select": lambda registry, _record: registry.select("native"),
        "activate": lambda registry, record: registry.activate(record.record_id),
    }
    for state in PluginLifecycleState:
        selectable = _forged_parser_rejected_record(
            state=state,
            entry_point=entry_point,
            plugin_object=preloaded,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        )
        with pytest.raises(BackendPluginManifestError) as caught:
            select_backend((selectable,), target="native")
        _assert_explicit_rejection(caught.value)

        for operation in operations.values():
            record = _forged_parser_rejected_record(
                state=state,
                entry_point=entry_point,
                plugin_object=preloaded,
                compiler_cls=compiler_cls,
                driver_cls=driver_cls,
            )
            registry = _registry_with_forged_record(record)
            with pytest.raises(BackendPluginManifestError) as caught:
                operation(registry, record)
            _assert_explicit_rejection(caught.value)
            _assert_rejected_record(registry.inspect(record.record_id))

        diagnostics_record = _forged_parser_rejected_record(
            state=state,
            entry_point=entry_point,
            plugin_object=preloaded,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        )
        diagnostics_registry = _registry_with_forged_record(diagnostics_record)
        diagnostic = diagnostics_registry.diagnostics(diagnostics_record.record_id)
        assert diagnostic["state"] == "rejected"
        assert diagnostic["plugin_diagnostics"] is None
        assert diagnostic["errors"][0]["field"] == "isolation_mode"
        _assert_rejected_record(
            diagnostics_registry.inspect(diagnostics_record.record_id)
        )

        reset_record = _forged_parser_rejected_record(
            state=state,
            entry_point=entry_point,
            plugin_object=preloaded,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        )
        reset_registry = _registry_with_forged_record(reset_record)
        reset_registry._cleanup_stack = [reset_record]
        assert reset_registry.reset() == ()

    assert entry_point.load_calls == 0
    assert all(value == 0 for value in calls.values())


def test_missing_manifest_shape_cannot_be_forged_into_any_lifecycle_state() -> None:
    calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
    operations = {
        "validate": lambda registry, record: registry.validate(record.record_id),
        "load": lambda registry, record: registry.load(record.record_id),
        "register": lambda registry, record: registry.register(record.record_id),
        "select": lambda registry, _record: registry.select("native"),
        "activate": lambda registry, record: registry.activate(record.record_id),
    }
    for state in PluginLifecycleState:
        selectable = _forged_missing_manifest_record(
            state=state,
            entry_point=entry_point,
            plugin_object=preloaded,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        )
        with pytest.raises(BackendPluginManifestError) as caught:
            select_backend((selectable,), target="native")
        _assert_missing_manifest_rejection(caught.value)

        for operation in operations.values():
            record = _forged_missing_manifest_record(
                state=state,
                entry_point=entry_point,
                plugin_object=preloaded,
                compiler_cls=compiler_cls,
                driver_cls=driver_cls,
            )
            registry = _registry_with_forged_record(record)
            with pytest.raises(BackendPluginManifestError) as caught:
                operation(registry, record)
            _assert_missing_manifest_rejection(caught.value)
            _assert_missing_manifest_record(registry.inspect(record.record_id))

        diagnostics_record = _forged_missing_manifest_record(
            state=state,
            entry_point=entry_point,
            plugin_object=preloaded,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        )
        diagnostics_registry = _registry_with_forged_record(diagnostics_record)
        diagnostic = diagnostics_registry.diagnostics(diagnostics_record.record_id)
        assert diagnostic["state"] == "rejected"
        assert diagnostic["plugin_diagnostics"] is None
        assert diagnostic["errors"][0]["field"] == "manifest"
        assert diagnostic["errors"][0]["actual"] == "<missing>"
        _assert_missing_manifest_record(
            diagnostics_registry.inspect(diagnostics_record.record_id)
        )

        reset_record = _forged_missing_manifest_record(
            state=state,
            entry_point=entry_point,
            plugin_object=preloaded,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        )
        reset_registry = _registry_with_forged_record(reset_record)
        reset_registry._cleanup_stack = [reset_record]
        assert reset_registry.reset() == ()

    assert entry_point.load_calls == 0
    assert all(value == 0 for value in calls.values())


def test_missing_manifest_preserves_later_structured_manifest_error() -> None:
    calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
    record = _forged_missing_manifest_record(
        state=PluginLifecycleState.ACTIVE,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    original = unsupported_isolation_mode_error(
        "native_in_process",
        plugin_id="vendor.native",
        entry_point="native",
    )
    record = replace(
        record,
        errors=(
            BackendPluginCompatibilityError(
                "earlier compatibility diagnostic",
                "compatible",
                "incompatible",
            ),
            original,
        ),
    )
    with pytest.raises(BackendPluginManifestError) as caught:
        select_backend((record,), target="native")
    assert caught.value is original

    registry = _registry_with_forged_record(record)
    with pytest.raises(BackendPluginManifestError) as caught:
        registry.load(record.record_id)
    assert caught.value is original
    _assert_rejected_record(registry.inspect(record.record_id))
    assert entry_point.load_calls == 0
    assert all(value == 0 for value in calls.values())


def test_only_exact_legacy_source_may_omit_manifest() -> None:
    calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
    missing = _forged_missing_manifest_record(
        state=PluginLifecycleState.VALIDATED,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    for source in ("legacy", "manifest", None):
        record = replace(missing, source=source)
        with pytest.raises(BackendPluginManifestError) as caught:
            select_backend((record,), target="native")
        _assert_missing_manifest_rejection(caught.value)

        registry = _registry_with_forged_record(record)
        diagnostic = registry.diagnostics(record.record_id)
        assert diagnostic["source"] == source
        assert diagnostic["errors"][0]["field"] == "manifest"
        _assert_missing_manifest_record(registry.inspect(record.record_id))

    exact_legacy = replace(
        missing,
        source=PluginSource.LEGACY,
        state=PluginLifecycleState.DISCOVERED,
        compatibility_status=PluginCompatibilityStatus.LEGACY_UNVERIFIED,
        plugin_object=None,
        compiler_cls=None,
        driver_cls=None,
        initialized=False,
        selected_targets=(),
    )
    assert operational_record_manifest_error(exact_legacy) is None
    assert entry_point.load_calls == 0
    assert all(value == 0 for value in calls.values())


def test_manifest_rejection_actual_is_stable_and_never_executes_repr() -> None:
    python_plugin = _python_plugin()
    first_object = replace(python_plugin, isolation_mode=object())
    second_object = replace(python_plugin, isolation_mode=object())
    actuals = []
    for plugin in (first_object, second_object):
        with pytest.raises(BackendPluginManifestError) as caught:
            validate_triton_requirement(plugin, _environment())
        assert caught.value.field == "isolation_mode"
        assert "0x" not in caught.value.actual
        actuals.append(caught.value.actual)
    assert actuals == ["<invalid type: builtins.object>"] * 2

    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    cyclic_dict: dict[str, Any] = {}
    cyclic_dict["self"] = cyclic_dict
    repr_bomb = _ReprBomb()
    variants = (
        (replace(python_plugin, isolation_mode=repr_bomb), "isolation_mode"),
        (replace(python_plugin, isolation_mode=cyclic_list), "isolation_mode"),
        (replace(python_plugin, isolation_mode=cyclic_dict), "isolation_mode"),
        (replace(python_plugin, native_libraries=repr_bomb), "native_libraries"),
        (replace(python_plugin, abi_fingerprint=cyclic_dict), "abi_fingerprint"),
    )
    for plugin, expected_field in variants:
        with pytest.raises(BackendPluginManifestError) as caught:
            validate_triton_requirement(plugin, _environment())
        assert caught.value.field == expected_field
        assert caught.value.actual.startswith("<invalid type: ")
        assert "0x" not in caught.value.actual

        calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
        record = _forged_native_record(
            plugin,
            state=PluginLifecycleState.ACTIVE,
            entry_point=entry_point,
            plugin_object=preloaded,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        )
        registry = _registry_with_forged_record(record)
        diagnostics = registry.diagnostics(record.record_id)
        error = diagnostics["errors"][0]
        assert error["field"] == expected_field
        assert error["actual"] == caught.value.actual
        assert "0x" not in diagnostics["manifest"]["isolation_mode"]
        assert registry.inspect(record.record_id).state is PluginLifecycleState.REJECTED
        assert all(value == 0 for value in calls.values())
    assert repr_bomb.repr_calls == 0

    parser_payload = _python_manifest()
    parser_payload["plugins"][0]["native_libraries"] = repr_bomb
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(parser_payload)
    assert caught.value.field == "native_libraries"
    assert "0x" not in caught.value.actual
    assert repr_bomb.repr_calls == 0


def test_diagnostics_does_not_touch_unsupported_secondary_iterables() -> None:
    targets = _IterBomb()
    plugin = replace(
        _native_plugin(),
        targets=targets,
        capabilities=targets,
        requires_capabilities=targets,
        priority=targets,
    )
    calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
    record = _forged_native_record(
        plugin,
        state=PluginLifecycleState.ACTIVE,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    registry = _registry_with_forged_record(record)
    diagnostic = registry.diagnostics(record.record_id)
    assert diagnostic["errors"][0]["field"] == "isolation_mode"
    assert diagnostic["manifest"]["targets"].endswith("._IterBomb>")
    assert "0x" not in diagnostic["manifest"]["targets"]
    assert targets.repr_calls == 0
    assert targets.iter_calls == 0
    assert all(value == 0 for value in calls.values())


def test_diagnostics_safely_renders_forged_record_source() -> None:
    source_bomb = _ReprBomb()
    variants = (
        ("manifest", "manifest"),
        ("legacy", "legacy"),
        (None, None),
        (
            source_bomb,
            "<invalid type: test_native_abi_acceptance._ReprBomb>",
        ),
    )
    for source, expected_source in variants:
        calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
        record = replace(
            _forged_native_record(
                _native_plugin(),
                state=PluginLifecycleState.ACTIVE,
                entry_point=entry_point,
                plugin_object=preloaded,
                compiler_cls=compiler_cls,
                driver_cls=driver_cls,
            ),
            source=source,
        )
        registry = _registry_with_forged_record(record)
        diagnostic = registry.diagnostics(record.record_id)
        assert diagnostic["source"] == expected_source
        assert diagnostic["errors"][0]["field"] == "isolation_mode"
        _assert_rejected_record(registry.inspect(record.record_id))
        assert entry_point.load_calls == 0
        assert all(value == 0 for value in calls.values())
    assert source_bomb.repr_calls == 0


def test_duplicate_unsupported_identifier_rejects_before_conflict_projection() -> None:
    targets = _IterBomb()
    plugin = replace(_native_plugin(), targets=targets)
    calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
    first = _forged_native_record(
        plugin,
        state=PluginLifecycleState.VALIDATED,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    second = replace(first, record_id="forged-native-duplicate:native")
    registry = _registry_with_forged_record(first)
    registry._records[second.record_id] = second
    with pytest.raises(BackendPluginManifestError) as caught:
        registry.validate("vendor.native")
    _assert_explicit_rejection(caught.value)
    _assert_rejected_record(registry.inspect(first.record_id))
    _assert_rejected_record(registry.inspect(second.record_id))
    assert targets.repr_calls == 0
    assert targets.iter_calls == 0
    assert entry_point.load_calls == 0
    assert all(value == 0 for value in calls.values())


def test_standalone_selector_rejects_duck_typed_manifest_before_projection() -> None:
    secondary_bomb = _ReprBomb()
    raw_manifest = SimpleNamespace(
        plugin_id="vendor.raw",
        entry_point="raw",
        backend_protocol=">=1.0,<2.0",
        requires_triton=TritonRequirement(version=">=3.6,<3.7"),
        targets=("raw",),
        capabilities=(),
        requires_capabilities=(),
        isolation_mode="native_in_process",
        native_libraries=("fixture/raw.so",),
        abi_fingerprint=FINGERPRINT,
        priority=secondary_bomb,
    )
    calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
    record = _forged_native_record(
        raw_manifest,
        state=PluginLifecycleState.VALIDATED,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    with pytest.raises(BackendPluginManifestError) as caught:
        select_backend((record,), target="raw")
    assert caught.value.field == "manifest"
    assert caught.value.actual == "<invalid type: types.SimpleNamespace>"

    registry = _registry_with_forged_record(record)
    with pytest.raises(BackendPluginManifestError) as caught:
        registry.select("raw")
    assert caught.value.field == "manifest"
    diagnostic = registry.diagnostics(record.record_id)
    assert diagnostic["errors"][0]["field"] == "manifest"
    assert diagnostic["manifest"] == {"type": "<invalid type: types.SimpleNamespace>"}
    assert secondary_bomb.repr_calls == 0
    assert entry_point.load_calls == 0
    assert all(value == 0 for value in calls.values())


def test_registry_mixed_candidates_report_unsupported_target_first() -> None:
    native_calls, native_ep, native_object, native_compiler, native_driver = (
        _runtime_sentinels()
    )
    native_record = _forged_native_record(
        _native_plugin(),
        state=PluginLifecycleState.VALIDATED,
        entry_point=native_ep,
        plugin_object=native_object,
        compiler_cls=native_compiler,
        driver_cls=native_driver,
    )
    python_calls, python_ep, python_object, python_compiler, python_driver = (
        _runtime_sentinels()
    )
    python_record = replace(
        _forged_native_record(
            _python_plugin(),
            state=PluginLifecycleState.VALIDATED,
            entry_point=python_ep,
            plugin_object=python_object,
            compiler_cls=python_compiler,
            driver_cls=python_driver,
        ),
        record_id="forged-python:python",
        selected_targets=(),
        initialized=False,
    )
    registry = _registry_with_forged_record(native_record)
    registry._records[python_record.record_id] = python_record

    class PropertyTarget:
        @property
        def backend(self) -> str:
            return "native"

    targets = (
        "native",
        {"backend": "native"},
        UserDict({"backend": "native"}),
        SimpleNamespace(backend="native"),
        PropertyTarget(),
    )
    for target in targets:
        with pytest.raises(BackendPluginManifestError) as caught:
            registry.select(target)
        _assert_explicit_rejection(caught.value)

    class BrokenPropertyTarget:
        @property
        def backend(self) -> str:
            raise AssertionError("target.backend implementation sentinel")

    with pytest.raises(
        AssertionError,
        match="target.backend implementation sentinel",
    ):
        registry.select(BrokenPropertyTarget())

    malformed_targets = _IterBomb()
    malformed_native = replace(
        native_record,
        manifest=replace(
            _native_plugin(),
            targets=malformed_targets,
        ),
    )
    malformed_registry = _registry_with_forged_record(malformed_native)
    malformed_registry._records[python_record.record_id] = python_record
    with pytest.raises(BackendPluginManifestError) as caught:
        malformed_registry.select("native")
    _assert_explicit_rejection(caught.value)
    assert malformed_targets.repr_calls == 0
    assert malformed_targets.iter_calls == 0
    assert native_ep.load_calls == 0
    assert python_ep.load_calls == 0
    assert all(value == 0 for value in native_calls.values())
    assert all(value == 0 for value in python_calls.values())


def test_unsupported_sibling_does_not_poison_valid_same_target_selection() -> None:
    def candidates() -> tuple[
        BackendPluginRecord,
        BackendPluginRecord,
        FakeEntryPoint,
        FakeEntryPoint,
        dict[str, int],
    ]:
        bad_calls, bad_ep, bad_object, bad_compiler, bad_driver = _runtime_sentinels()
        bad_record = _forged_native_record(
            _native_plugin(),
            state=PluginLifecycleState.VALIDATED,
            entry_point=bad_ep,
            plugin_object=bad_object,
            compiler_cls=bad_compiler,
            driver_cls=bad_driver,
        )
        _valid_calls, valid_ep, valid_object, valid_compiler, valid_driver = (
            _runtime_sentinels()
        )
        valid_plugin = replace(_python_plugin(), targets=("native",))
        valid_record = replace(
            _forged_native_record(
                valid_plugin,
                state=PluginLifecycleState.VALIDATED,
                entry_point=valid_ep,
                plugin_object=valid_object,
                compiler_cls=valid_compiler,
                driver_cls=valid_driver,
            ),
            record_id="forged-python:native",
            selected_targets=(),
            initialized=False,
        )
        return bad_record, valid_record, bad_ep, valid_ep, bad_calls

    bad_record, valid_record, bad_ep, valid_ep, bad_calls = candidates()
    standalone_cases = (
        {},
        {"explicit_selector": valid_record.plugin_id},
        {"environment": {BACKEND_SELECTOR_ENV: valid_record.plugin_id}},
    )
    for kwargs in standalone_cases:
        decision = select_backend(
            (bad_record, valid_record),
            target="native",
            **kwargs,
        )
        assert decision.record_id == valid_record.record_id
    with pytest.raises(BackendPluginManifestError) as caught:
        select_backend(
            (bad_record, valid_record),
            target="native",
            explicit_selector=bad_record.plugin_id,
        )
    _assert_explicit_rejection(caught.value)
    assert bad_ep.load_calls == 0
    assert valid_ep.load_calls == 0
    assert all(value == 0 for value in bad_calls.values())

    registry_cases = (
        {},
        {"explicit_selector": "vendor.python"},
        {"environment": {BACKEND_SELECTOR_ENV: "vendor.python"}},
    )
    for kwargs in registry_cases:
        bad_record, valid_record, bad_ep, _valid_ep, bad_calls = candidates()
        registry = _registry_with_forged_record(bad_record)
        registry._records[valid_record.record_id] = valid_record
        decision = registry.select("native", **kwargs)
        assert decision.record_id == valid_record.record_id
        _assert_rejected_record(registry.inspect(bad_record.record_id))
        assert bad_ep.load_calls == 0
        assert all(value == 0 for value in bad_calls.values())

    bad_record, valid_record, bad_ep, valid_ep, bad_calls = candidates()
    registry = _registry_with_forged_record(bad_record)
    registry._records[valid_record.record_id] = valid_record
    with pytest.raises(BackendPluginManifestError) as caught:
        registry.select(
            "native",
            explicit_selector=bad_record.plugin_id,
        )
    _assert_explicit_rejection(caught.value)
    _assert_rejected_record(registry.inspect(bad_record.record_id))
    assert bad_ep.load_calls == 0
    assert valid_ep.load_calls == 0
    assert all(value == 0 for value in bad_calls.values())


def test_rejected_discovery_identity_remains_an_explicit_selector(
    tmp_path: Path,
) -> None:
    native_ep = FakeEntryPoint("native", object())
    python_ep = FakeEntryPoint("python", object())
    native_distribution = FakeDistribution(
        tmp_path / "native",
        entry_point=native_ep,
        manifest=_plugin_manifest(),
        library_paths=(),
    )
    python_distribution = FakeDistribution(
        tmp_path / "python",
        entry_point=python_ep,
        manifest=_python_manifest(),
        library_paths=(),
    )
    registry = BackendPluginRegistry(
        distribution_provider=lambda: (
            native_distribution,
            python_distribution,
        ),
        environment_provider=_environment,
        supported_tags=(WHEEL_TAG,),
    )
    records = registry.validate()
    native_record = next(
        record for record in records if record.entry_point_name == "native"
    )
    assert native_record.manifest is None
    assert native_record.plugin_id is None
    assert native_record.error is not None
    assert native_record.error.plugin_id == "vendor.native"

    operations = (
        registry.validate,
        registry.load,
        registry.register,
        registry.activate,
    )
    for operation in operations:
        with pytest.raises(BackendPluginManifestError) as caught:
            operation("vendor.native")
        _assert_explicit_rejection(caught.value)

    selectors = ("vendor.native", native_record.record_id)
    for selector in selectors:
        with pytest.raises(BackendPluginManifestError) as caught:
            registry.select(
                "native",
                explicit_selector=selector,
                environment={},
            )
        _assert_explicit_rejection(caught.value)
    assert native_ep.load_calls == 0
    assert python_ep.load_calls == 0


def test_python_only_inventory_detects_native_suffixes_and_magic(
    tmp_path: Path,
) -> None:
    suffixes = (".so", ".so.7", ".dylib", ".dll", ".pyd", ".SO", ".bin")
    for index, suffix in enumerate(suffixes):
        root = tmp_path / str(index)
        relative = f"fixture/libinventory{suffix}"
        _compile_shared(root, relative=relative)
        entry_point = FakeEntryPoint(f"python-{index}", object())
        manifest = _python_manifest(
            entry_point=entry_point.name,
            plugin_id=f"vendor.python.{index}",
        )
        distribution = FakeDistribution(
            root,
            entry_point=entry_point,
            manifest=manifest,
            library_paths=(relative,),
        )
        plugin = parse_manifest(manifest).plugins[0]
        with pytest.raises(BackendPluginCompatibilityError) as caught:
            inspect_native_artifacts(plugin, distribution)
        assert caught.value.dimension == "python_only wheel contents"
        assert relative in caught.value.actual
        assert entry_point.load_calls == 0


def test_python_only_inventory_accepts_only_known_install_scheme_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "prefix"
    real_site = prefix / "lib/python3.12/site-packages"
    scripts = prefix / "bin"
    data = prefix
    real_site.mkdir(parents=True)
    scripts.mkdir(parents=True)
    site_link = tmp_path / "site-link"
    site_link.symlink_to(real_site, target_is_directory=True)
    script_relative = "../../../bin/vendor-tool"
    script = scripts / "vendor-tool"
    script.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    paths = {
        "purelib": os.fspath(real_site),
        "platlib": os.fspath(real_site),
        "scripts": os.fspath(scripts),
        "data": os.fspath(data),
    }
    monkeypatch.setattr(
        native_module.sysconfig,
        "get_scheme_names",
        lambda: ("t63-fixture",),
    )
    monkeypatch.setattr(
        native_module.sysconfig,
        "get_paths",
        lambda scheme=None: paths,
    )

    entry_point = FakeEntryPoint("scheme-python", object())
    manifest = _python_manifest(
        entry_point=entry_point.name,
        plugin_id="vendor.scheme.python",
    )
    distribution = FakeDistribution(
        site_link,
        entry_point=entry_point,
        manifest=manifest,
        library_paths=(),
        extra_files=(script_relative,),
    )
    plugin = parse_manifest(manifest).plugins[0]
    assert inspect_native_artifacts(plugin, distribution).artifacts == ()

    outside = tmp_path / "outside.txt"
    outside.write_text("outside install scheme\n", encoding="utf-8")
    escaped_relative = "../../../../outside.txt"
    escaped_distribution = FakeDistribution(
        site_link,
        entry_point=entry_point,
        manifest=manifest,
        library_paths=(),
        extra_files=(escaped_relative,),
    )
    with pytest.raises(BackendPluginManifestError) as caught:
        inspect_native_artifacts(plugin, escaped_distribution)
    assert caught.value.field == "distribution.files"
    assert caught.value.actual == escaped_relative


def test_python_only_inventory_propagates_programming_error_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint("sentinel-python", object())
    manifest = _python_manifest(
        entry_point=entry_point.name,
        plugin_id="vendor.sentinel.python",
    )
    distribution = FakeDistribution(
        tmp_path,
        entry_point=entry_point,
        manifest=manifest,
        library_paths=(),
    )
    plugin = parse_manifest(manifest).plugins[0]

    def broken_scheme_names() -> Any:
        raise AssertionError("sysconfig implementation sentinel")

    monkeypatch.setattr(
        native_module.sysconfig,
        "get_scheme_names",
        broken_scheme_names,
    )
    with pytest.raises(AssertionError, match="sysconfig implementation sentinel"):
        inspect_native_artifacts(plugin, distribution)

    class BrokenFilesDistribution:
        @property
        def files(self) -> Any:
            raise AssertionError("distribution.files implementation sentinel")

    with pytest.raises(
        AssertionError,
        match="distribution.files implementation sentinel",
    ):
        inspect_native_artifacts(plugin, BrokenFilesDistribution())


def test_python_only_inventory_is_mandatory_in_both_registry_profiles(
    tmp_path: Path,
) -> None:
    for profile in ("full", "triton_version"):
        root = tmp_path / profile
        dirty_relative = "fixture/renamed-native.bin"
        _compile_shared(root / "dirty", relative=dirty_relative)
        dirty_entry_point = FakeEntryPoint(f"dirty-{profile}", object())
        dirty_manifest = _python_manifest(
            entry_point=dirty_entry_point.name,
            plugin_id=f"vendor.dirty.{profile.replace('_', '-')}",
        )
        dirty_distribution = FakeDistribution(
            root / "dirty",
            entry_point=dirty_entry_point,
            manifest=dirty_manifest,
            library_paths=(dirty_relative,),
        )

        clean_entry_point = FakeEntryPoint(f"clean-{profile}", object())
        clean_manifest = _python_manifest(
            entry_point=clean_entry_point.name,
            plugin_id=f"vendor.clean.{profile.replace('_', '-')}",
        )
        clean_distribution = FakeDistribution(
            root / "clean",
            entry_point=clean_entry_point,
            manifest=clean_manifest,
            library_paths=(),
        )
        registry = BackendPluginRegistry(
            distribution_provider=lambda dirty=dirty_distribution, clean=clean_distribution: (
                dirty,
                clean,
            ),
            environment_provider=_environment,
            supported_tags=(WHEEL_TAG,),
            preflight_profile=profile,
        )
        records = {record.plugin_id: record for record in registry.validate()}
        dirty = records[dirty_manifest["plugins"][0]["plugin_id"]]
        clean = records[clean_manifest["plugins"][0]["plugin_id"]]
        assert dirty.state is PluginLifecycleState.REJECTED
        assert isinstance(dirty.error, BackendPluginCompatibilityError)
        assert dirty.error.dimension == "python_only wheel contents"
        assert clean.state is PluginLifecycleState.VALIDATED
        assert clean.compatibility_status is PluginCompatibilityStatus.COMPATIBLE
        with pytest.raises(BackendPluginCompatibilityError) as caught:
            registry.load(dirty.record_id)
        assert caught.value.dimension == "python_only wheel contents"
        assert dirty_entry_point.load_calls == 0
        assert clean_entry_point.load_calls == 0


def test_triton_adapter_rechecks_manifest_before_public_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_name = "_t63_adapter_fixture"
    module_name = package_name + ".backends"
    package = ModuleType(package_name)
    package.__path__ = []
    driver_module = ModuleType(module_name + ".driver")
    compiler_module = ModuleType(module_name + ".compiler")
    driver_module.DriverBase = type("DriverBase", (), {})
    compiler_module.BaseBackend = type("BaseBackend", (), {})
    compiler_module.GPUTarget = type("GPUTarget", (), {})
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, module_name + ".driver", driver_module)
    monkeypatch.setitem(sys.modules, module_name + ".compiler", compiler_module)

    adapter_path = (
        Path(__file__).resolve().parents[2]
        / "triton/python/triton/backends/__init__.py"
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        adapter_path,
        submodule_search_locations=[str(adapter_path.parent)],
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, adapter)
    spec.loader.exec_module(adapter)

    calls, entry_point, preloaded, compiler_cls, driver_cls = _runtime_sentinels()
    record = _forged_native_record(
        _native_plugin(),
        state=PluginLifecycleState.SELECTED,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    registry = _registry_with_forged_record(record)
    monkeypatch.setattr(adapter, "_registry", lambda: registry)
    decision = SimpleNamespace(
        record_id=record.record_id,
        plugin_id=record.plugin_id,
        entry_point_name=record.entry_point_name,
        target="native",
    )
    with pytest.raises(BackendPluginManifestError) as caught:
        adapter._cache_decision(decision)
    _assert_explicit_rejection(caught.value)
    assert adapter.backends == {}
    rejected = registry.inspect(record.record_id)
    _assert_rejected_record(rejected)

    adapter.backends[record.entry_point_name] = adapter.Backend(
        compiler=compiler_cls,
        driver=driver_cls,
        record_id=record.record_id,
        plugin_id=record.plugin_id,
        entry_point_name=record.entry_point_name,
    )
    adapter._prune_registry_backends(registry)
    assert adapter.backends == {}

    class DuckManifestWithBombTargets:
        plugin_id = "vendor.duck"
        entry_point = "native"

        def __init__(self, targets: Any, *, explode: bool = False) -> None:
            self._targets = targets
            self._explode = explode
            self.target_reads = 0

        @property
        def targets(self) -> Any:
            self.target_reads += 1
            if self._explode:
                raise AssertionError("duck manifest targets property sentinel")
            return self._targets

    property_bomb_manifest = DuckManifestWithBombTargets((), explode=True)
    property_bomb_record = _forged_native_record(
        property_bomb_manifest,
        state=PluginLifecycleState.VALIDATED,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    clean_calls, clean_ep, clean_object, clean_compiler, clean_driver = (
        _runtime_sentinels()
    )
    clean_record = replace(
        _forged_native_record(
            _python_plugin(),
            state=PluginLifecycleState.VALIDATED,
            entry_point=clean_ep,
            plugin_object=clean_object,
            compiler_cls=clean_compiler,
            driver_cls=clean_driver,
        ),
        record_id="forged-python:python",
        initialized=False,
        shutdown_called=False,
        selected_targets=(),
    )
    duck_registry = _registry_with_forged_record(property_bomb_record)
    duck_registry._records[clean_record.record_id] = clean_record
    monkeypatch.setattr(adapter, "_registry", lambda: duck_registry)
    monkeypatch.setenv("TRITON_ANCHOR_BACKEND", "vendor.unknown")
    with pytest.raises(BackendPluginSelectionError) as caught:
        adapter.get_backend("native")
    assert caught.value.field == "backend_selector"
    assert caught.value.actual == "vendor.unknown"
    assert property_bomb_manifest.target_reads == 0
    assert entry_point.load_calls == 0
    assert clean_ep.load_calls == 0
    assert all(value == 0 for value in clean_calls.values())
    monkeypatch.delenv("TRITON_ANCHOR_BACKEND")

    target_variants = (
        property_bomb_manifest,
        DuckManifestWithBombTargets(_IterBomb()),
    )
    for duck_manifest in target_variants:
        duck_record = _forged_native_record(
            duck_manifest,
            state=PluginLifecycleState.ACTIVE,
            entry_point=entry_point,
            plugin_object=preloaded,
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
        )
        duck_registry = _registry_with_forged_record(duck_record)
        monkeypatch.setattr(adapter, "_registry", lambda: duck_registry)
        manual_duck_legacy = adapter.Backend(
            compiler=compiler_cls,
            driver=driver_cls,
            record_id=None,
            entry_point_name="manual-legacy",
        )
        with pytest.raises(BackendPluginManifestError) as caught:
            adapter.activate_backend(manual_duck_legacy, target="native")
        assert caught.value.field == "manifest"
        assert duck_manifest.target_reads == 0
        payload = duck_manifest._targets
        if isinstance(payload, _IterBomb):
            assert payload.iter_calls == 0
            assert payload.repr_calls == 0
        rejected_duck = duck_registry.inspect(duck_record.record_id)
        assert rejected_duck.state is PluginLifecycleState.REJECTED
        assert rejected_duck.compatibility_status is (
            PluginCompatibilityStatus.NOT_CHECKED
        )
        assert rejected_duck.error.field == "manifest"
        assert rejected_duck.plugin_object is None
        assert rejected_duck.compiler_cls is None
        assert rejected_duck.driver_cls is None
        assert not rejected_duck.initialized
        assert not rejected_duck.shutdown_called
        assert rejected_duck.selected_targets == ()

    activate_record = _forged_native_record(
        _native_plugin(),
        state=PluginLifecycleState.SELECTED,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    activate_registry = _registry_with_forged_record(activate_record)
    monkeypatch.setattr(adapter, "_registry", lambda: activate_registry)
    compiler_resolution_calls = []

    def forbidden_compiler_resolution(_target: Any) -> Any:
        compiler_resolution_calls.append(_target)
        raise AssertionError("unsupported adapter activation resolved target")

    monkeypatch.setattr(adapter, "get_backend", forbidden_compiler_resolution)
    forged_backend = adapter.Backend(
        compiler=compiler_cls,
        driver=driver_cls,
        record_id=activate_record.record_id,
        plugin_id=activate_record.plugin_id,
        entry_point_name=activate_record.entry_point_name,
    )
    with pytest.raises(BackendPluginManifestError) as caught:
        adapter.activate_backend(forged_backend, target="native")
    _assert_explicit_rejection(caught.value)
    _assert_rejected_record(activate_registry.inspect(activate_record.record_id))
    assert compiler_resolution_calls == []

    unreadable_targets = _IterBomb()
    legacy_overlap_record = _forged_native_record(
        replace(_native_plugin(), targets=unreadable_targets),
        state=PluginLifecycleState.ACTIVE,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    legacy_overlap_registry = _registry_with_forged_record(legacy_overlap_record)
    monkeypatch.setattr(
        adapter,
        "_registry",
        lambda: legacy_overlap_registry,
    )
    manual_legacy = adapter.Backend(
        compiler=compiler_cls,
        driver=driver_cls,
        record_id=None,
        entry_point_name="manual-legacy",
    )
    with pytest.raises(BackendPluginManifestError) as caught:
        adapter.activate_backend(manual_legacy, target="native")
    _assert_explicit_rejection(caught.value)
    assert unreadable_targets.repr_calls == 0
    assert unreadable_targets.iter_calls == 0
    _assert_rejected_record(
        legacy_overlap_registry.inspect(legacy_overlap_record.record_id)
    )

    parser_rejected_record = _forged_parser_rejected_record(
        state=PluginLifecycleState.SELECTED,
        entry_point=entry_point,
        plugin_object=preloaded,
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
    )
    parser_rejected_registry = _registry_with_forged_record(parser_rejected_record)
    monkeypatch.setattr(
        adapter,
        "_registry",
        lambda: parser_rejected_registry,
    )
    parser_decision = SimpleNamespace(
        record_id=parser_rejected_record.record_id,
        plugin_id="vendor.native",
        entry_point_name="native",
        target="native",
    )
    with pytest.raises(BackendPluginManifestError) as caught:
        adapter._cache_decision(parser_decision)
    _assert_explicit_rejection(caught.value)
    _assert_rejected_record(
        parser_rejected_registry.inspect(parser_rejected_record.record_id)
    )
    assert adapter._is_selected_record(parser_rejected_record) is False

    for state in PluginLifecycleState:
        for operation in ("cache", "activate", "prune"):
            missing_record = _forged_missing_manifest_record(
                state=state,
                entry_point=entry_point,
                plugin_object=preloaded,
                compiler_cls=compiler_cls,
                driver_cls=driver_cls,
            )
            missing_registry = _registry_with_forged_record(missing_record)
            monkeypatch.setattr(
                adapter,
                "_registry",
                lambda registry=missing_registry: registry,
            )
            assert adapter._is_selected_record(missing_record) is False
            if operation == "cache":
                missing_decision = SimpleNamespace(
                    record_id=missing_record.record_id,
                    plugin_id=None,
                    entry_point_name="native",
                    target="native",
                )
                with pytest.raises(BackendPluginManifestError) as caught:
                    adapter._cache_decision(missing_decision)
            elif operation == "activate":
                missing_backend = adapter.Backend(
                    compiler=compiler_cls,
                    driver=driver_cls,
                    record_id=missing_record.record_id,
                    plugin_id=None,
                    entry_point_name="native",
                )
                with pytest.raises(BackendPluginManifestError) as caught:
                    adapter.activate_backend(
                        missing_backend,
                        target="native",
                    )
            else:
                adapter.backends["native"] = adapter.Backend(
                    compiler=compiler_cls,
                    driver=driver_cls,
                    record_id=missing_record.record_id,
                    plugin_id=None,
                    entry_point_name="native",
                )
                adapter._prune_registry_backends(missing_registry)
                assert "native" not in adapter.backends
                _assert_missing_manifest_record(
                    missing_registry.inspect(missing_record.record_id)
                )
                continue
            _assert_missing_manifest_rejection(caught.value)
            _assert_missing_manifest_record(
                missing_registry.inspect(missing_record.record_id)
            )

    native_ep = FakeEntryPoint("native", object())
    python_ep = FakeEntryPoint("python", object())
    native_distribution = FakeDistribution(
        tmp_path / "adapter-native",
        entry_point=native_ep,
        manifest=_plugin_manifest(),
        library_paths=(),
    )
    python_distribution = FakeDistribution(
        tmp_path / "adapter-python",
        entry_point=python_ep,
        manifest=_python_manifest(),
        library_paths=(),
    )
    selector_registry = BackendPluginRegistry(
        distribution_provider=lambda: (
            native_distribution,
            python_distribution,
        ),
        environment_provider=_environment,
        supported_tags=(WHEEL_TAG,),
    )
    monkeypatch.setattr(adapter, "_registry", lambda: selector_registry)
    monkeypatch.setenv("TRITON_ANCHOR_BACKEND", "vendor.native")
    with pytest.raises(BackendPluginManifestError) as caught:
        adapter.get_driver_backends()
    _assert_explicit_rejection(caught.value)
    assert adapter.backends == {}
    assert native_ep.load_calls == 0
    assert python_ep.load_calls == 0
    assert entry_point.load_calls == 0
    assert all(value == 0 for value in calls.values())

    class BrokenRegistry:
        generation = 63
        lifecycle_epoch = 7

        def validate(self, _record_id: str) -> Any:
            raise AssertionError("adapter registry implementation sentinel")

    broken_registry = BrokenRegistry()
    sentinel_backend = adapter.Backend(
        record_id="sentinel:record",
        entry_point_name="sentinel",
    )
    adapter.backends["sentinel"] = sentinel_backend
    with pytest.raises(
        AssertionError,
        match="adapter registry implementation sentinel",
    ):
        adapter._prune_registry_backends(broken_registry)
    monkeypatch.setattr(adapter, "_registry", lambda: broken_registry)
    sentinel_decision = SimpleNamespace(
        record_id="sentinel:record",
        plugin_id="vendor.sentinel",
        entry_point_name="sentinel",
        target="sentinel",
    )
    with pytest.raises(
        AssertionError,
        match="adapter registry implementation sentinel",
    ):
        adapter._cache_decision(sentinel_decision)


@dataclass(frozen=True)
class ConflictRecord:
    record_id: str
    plugin_id: str
    entry_point_name: str
    manifest: Any
    compatibility_report: CompatibilityReport
    state: PluginLifecycleState = PluginLifecycleState.DISCOVERED
    compatibility_status: PluginCompatibilityStatus = (
        PluginCompatibilityStatus.NOT_CHECKED
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
            compatible=False,
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
    registry._records = {record.record_id: record for record in (second, first)}
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
            item["entry_point"] == record.entry_point_name for item in diagnostics
        )
        assert all(
            item["related_record_ids"] == sorted(set(rejected) - {record.record_id})
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
        "triton_version": "3.6.0",
        "vendored_triton_commit": TRITON_COMMIT,
        "expected_llvm_project_commit": LLVM_COMMIT,
        "actual_llvm_version_raw": "22.0.0git",
        "actual_llvm_commit": LLVM_COMMIT,
        "actual_mlir_version_raw": "22.0.0git",
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
    """No symbol is invented; static facts never authorize native loading."""

    distribution = _distribution(tmp_path, symbol="arbitrary_vendor_export")
    plugin = _native_plugin()
    artifact = inspect_native_artifacts(plugin, distribution).artifacts[0]
    assert artifact.exported_symbols == ("arbitrary_vendor_export",)
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(_plugin_manifest())
    _assert_explicit_rejection(caught.value)
    _assert_all_operational_paths_reject(
        plugin,
        distribution=distribution,
    )


def test_native_plugin_cxx_abi_is_not_attested_by_manifest_1_0(
    tmp_path: Path,
) -> None:
    """Opposite C++ ABIs remain evidence only and are both rejected."""

    artifacts = []
    for abi in ("0", "1"):
        root = tmp_path / f"abi-{abi}"
        relative = "fixture/libvendor_native.so"
        output = root / relative
        output.parent.mkdir(parents=True)
        source = output.with_suffix(".cc")
        source.write_text(
            "#include <string>\n"
            'extern "C" __attribute__((visibility("default")))\n'
            "unsigned long vendor_cxx_string_size(void) {\n"
            '  return std::string("t63").size();\n'
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
            entry_point=f"native-{abi}",
            plugin_id=f"vendor.native.{abi}",
        )
        distribution = FakeDistribution(
            root,
            entry_point=entry_point,
            manifest=manifest,
            library_paths=(relative,),
        )
        plugin = _native_plugin(
            entry_point=f"native-{abi}",
            plugin_id=f"vendor.native.{abi}",
        )
        artifacts.append(inspect_native_artifacts(plugin, distribution).artifacts[0])
        with pytest.raises(BackendPluginManifestError) as caught:
            parse_manifest(manifest)
        _assert_explicit_rejection(caught.value)
        _assert_all_operational_paths_reject(plugin)

    assert all("vendor_cxx_string_size" in item.exported_symbols for item in artifacts)
    assert artifacts[0].sha256 != artifacts[1].sha256


def test_native_static_inspection_does_not_prove_dlopen_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loadable constructor witness proves rejection performs no dlopen."""

    relative = "fixture/libvendor_native.so"
    output = tmp_path / relative
    output.parent.mkdir(parents=True)
    marker = tmp_path / "native-constructor-ran"
    source = output.with_suffix(".c")
    source.write_text(
        "#include <stdio.h>\n"
        "__attribute__((constructor)) static void t63_constructor(void) {\n"
        f'  FILE *stream = fopen({json.dumps(os.fspath(marker))}, "w");\n'
        '  if (stream != NULL) { fputs("loaded", stream); fclose(stream); }\n'
        "}\n"
        '__attribute__((visibility("default")))\n'
        "int vendor_t63_entry(void) { return 63; }\n",
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
    assert not marker.exists()

    dlopen_calls = []

    def forbidden_dlopen(*args: Any, **kwargs: Any) -> Any:
        dlopen_calls.append((args, kwargs))
        raise AssertionError("explicit rejection attempted ctypes dlopen")

    monkeypatch.setattr(ctypes, "CDLL", forbidden_dlopen)
    monkeypatch.setattr(ctypes, "PyDLL", forbidden_dlopen)
    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(manifest)
    _assert_explicit_rejection(caught.value)
    _assert_all_operational_paths_reject(
        plugin,
        distribution=distribution,
    )
    assert dlopen_calls == []
    assert not marker.exists()
