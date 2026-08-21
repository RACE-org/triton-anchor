"""Independent native ABI acceptance fixtures for T6.3.

These tests build tiny ELF shared objects at runtime.  They exercise the
public Manifest, compatibility, native inspection, conflict, and Registry
APIs without importing a real hardware backend or relying on T10.2.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from packaging.tags import Tag

from triton_anchor.backends import (
    BackendPluginCompatibilityError,
    BackendPluginManifestError,
    BackendPluginRegistry,
    CompatibilityReport,
    CoreEnvironment,
    NativeArtifact,
    PluginCompatibilityStatus,
    PluginIsolationMode,
    PluginLifecycleState,
    detect_conflicts,
    inspect_native_artifacts,
    load_build_info,
    parse_manifest,
    validate_backend_plugin,
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


def test_native_valid_elf_record_hash_soname_and_public_symbol(tmp_path: Path) -> None:
    distribution = _distribution(tmp_path)
    plugin = parse_manifest(_plugin_manifest()).plugins[0]
    report = inspect_native_artifacts(plugin, distribution)
    assert report.paths == ("fixture/libvendor_native.so",)
    artifact = report.artifacts[0]
    assert artifact.binary_format == "ELF"
    assert artifact.architecture == platform.machine()
    assert artifact.identity == "libvendor_native.so"
    assert "vendor_t63_entry" in artifact.exported_symbols

    compatibility = validate_backend_plugin(
        plugin,
        _environment(),
        distribution=distribution,
        core_abi_fingerprint=FINGERPRINT,
        supported_tags=(WHEEL_TAG,),
    )
    assert compatibility.compatible
    assert compatibility.native_artifacts == report.artifacts


def test_native_record_tampering_is_rejected(tmp_path: Path) -> None:
    distribution = _distribution(
        tmp_path,
        record_hash_overrides={
            "fixture/libvendor_native.so": "sha256=" + "A" * 43,
        },
    )
    plugin = parse_manifest(_plugin_manifest()).plugins[0]
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
    plugin = parse_manifest(_plugin_manifest()).plugins[0]
    with pytest.raises(BackendPluginManifestError) as caught:
        inspect_native_artifacts(plugin, distribution)
    assert caught.value.field == "native_libraries"
    assert "undeclared" in str(caught.value).lower()


def test_native_in_process_cannot_use_pure_python_wheel_tag(tmp_path: Path) -> None:
    distribution = _distribution(tmp_path, purelib=True)
    plugin = parse_manifest(_plugin_manifest()).plugins[0]
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
    assert record.compatibility_status is PluginCompatibilityStatus.INCOMPATIBLE
    assert isinstance(record.error, BackendPluginCompatibilityError)
    assert record.error.dimension == "Core ABI fingerprint"
    assert record.error.plugin_id == "vendor.native"
    assert record.error.expected == FINGERPRINT
    assert record.error.actual == "sha256:" + "c" * 64
    assert entry_point.load_calls == 0
    assert plugin_object.initialize_calls == 0


@dataclass(frozen=True)
class ConflictRecord:
    record_id: str
    plugin_id: str
    entry_point_name: str
    manifest: Any
    compatibility_report: CompatibilityReport
    state: PluginLifecycleState = PluginLifecycleState.VALIDATED


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
        assert diagnostic["expected"]
        assert diagnostic["actual"]


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
    """Evidence for a SPEC GAP, not a PASS for a required-symbol contract.

    Manifest 1.0 has no field or normative name for an initialization symbol.
    The public inspector therefore accepts an arbitrary vendor-qualified export.
    The acceptance report classifies missing/wrong required-symbol cases as
    SPEC GAP rather than inventing a symbol name here.
    """

    distribution = _distribution(tmp_path, symbol="arbitrary_vendor_export")
    plugin = parse_manifest(_plugin_manifest()).plugins[0]
    artifact = inspect_native_artifacts(plugin, distribution).artifacts[0]
    assert artifact.exported_symbols == ("arbitrary_vendor_export",)


def test_native_plugin_cxx_abi_is_not_attested_by_manifest_1_0(
    tmp_path: Path,
) -> None:
    """Evidence for the compiler/C++ ABI attestation SPEC GAP.

    Two binaries built with opposite libstdc++ ABI switches can repeat the same
    self-asserted Core fingerprint.  Manifest 1.0 carries no independently
    verifiable plugin compiler or ``_GLIBCXX_USE_CXX11_ABI`` fact, so both pass
    the current public static validator.
    """

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
            entry_point=f"native-{abi}",
            plugin_id=f"vendor.native.{abi}",
        )
        distribution = FakeDistribution(
            root,
            entry_point=entry_point,
            manifest=manifest,
            library_paths=(relative,),
        )
        plugin = parse_manifest(manifest).plugins[0]
        report = validate_backend_plugin(
            plugin,
            _environment(),
            distribution=distribution,
            core_abi_fingerprint=FINGERPRINT,
            supported_tags=(WHEEL_TAG,),
        )
        artifacts.append(report.native_artifacts[0])

    assert all("vendor_cxx_string_size" in item.exported_symbols for item in artifacts)
    assert artifacts[0].sha256 != artifacts[1].sha256


def test_native_static_inspection_does_not_prove_dlopen_success(
    tmp_path: Path,
) -> None:
    """Evidence for the undefined native load/initialization policy gap."""

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
    plugin = parse_manifest(manifest).plugins[0]
    artifact = inspect_native_artifacts(plugin, distribution).artifacts[0]
    assert "vendor_t63_entry" in artifact.exported_symbols
    with pytest.raises(OSError, match="t63_symbol_that_does_not_exist"):
        ctypes.CDLL(str(output), mode=os.RTLD_NOW)
