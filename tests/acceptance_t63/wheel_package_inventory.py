#!/usr/bin/env python3
"""First-principles package and wheel inventory helpers for T6.3 F4.

The source oracle is derived from tracked payload paths and does not call
setuptools.  The declared inventory is captured independently by executing the
real ``setup.py`` with only its final ``setup()`` call replaced.  Wheel and
RECORD checks operate on the built archive rather than a build directory.
"""

from __future__ import annotations

import base64
import csv
import email.parser
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
import types
import venv
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import setuptools
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PACKAGE_ROOTS = (
    (PurePosixPath("python/triton_anchor"), "triton_anchor"),
    (PurePosixPath("triton/python/triton"), "triton"),
)
EXPECTED_BUILD_INFO = {
    "schema_version": "1.1",
    "core_version": "0.2.0",
    "backend_protocol_version": "1.0",
    "manifest_schema_version": "1.0",
    "triton_version": "3.6.0",
    "vendored_triton_commit": "6cc4505027d7b39fe18a44a7f89085b8babb7400",
    "expected_llvm_project_commit": "a992f29451b9e140424f35ac5e20177db4afbdc0",
    "actual_llvm_commit": "a992f29451b9e140424f35ac5e20177db4afbdc0",
    "actual_mlir_commit": "a992f29451b9e140424f35ac5e20177db4afbdc0",
    "cxx_standard": "17",
    "core_abi_fingerprint_schema": "triton-anchor-core-abi-v1",
}
ABI_MATERIAL_KEYS = (
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
REQUIRED_FILES = {
    "triton_anchor/_build_info.json",
    "triton_anchor/backends/registry.py",
    "triton_anchor/backends/schemas/backend_manifest.schema.json",
    "triton_anchor/backends/examples/triton_anchor_backend.example.json",
    "triton_anchor/language/ext/__init__.py",
    "triton/_C/libtriton.so",
}
FORBIDDEN_PATH_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
)
SOURCE_PAYLOAD_SUFFIXES = frozenset({".py", ".pyi", ".json"})
WHEEL_PACKAGE_SUFFIXES = frozenset({".py", ".pyi", ".json", ".so"})
GENERATED_PAYLOAD_ALLOWLIST = frozenset({"triton_anchor/_build_info.json"})


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {command!r}\n{completed.stdout}"
        )
    return completed


def tracked_source_payload(repo_root: Path) -> dict[str, Path]:
    """Map tracked publishable source payload to its canonical wheel path."""
    repo_root = repo_root.resolve(strict=True)
    completed = _run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            *(str(root) for root, _ in PACKAGE_ROOTS),
        ],
        cwd=repo_root,
    )
    result: dict[str, Path] = {}
    for raw_path in completed.stdout.split("\0"):
        if not raw_path:
            continue
        relative = PurePosixPath(raw_path)
        if relative.suffix not in SOURCE_PAYLOAD_SUFFIXES:
            continue
        for source_root, package_name in PACKAGE_ROOTS:
            try:
                inside = relative.relative_to(source_root)
            except ValueError:
                continue
            wheel_path = PurePosixPath(package_name) / inside
            result[str(wheel_path)] = repo_root / Path(*relative.parts)
            break
    return result


def source_package_inventory(repo_root: Path) -> set[str]:
    """Derive source packages from tracked payload parents and ancestors."""
    packages: set[str] = set()
    for wheel_path in tracked_source_payload(repo_root):
        parts = PurePosixPath(wheel_path).parent.parts
        for length in range(1, len(parts) + 1):
            candidate = parts[:length]
            if all(part.isidentifier() for part in candidate):
                packages.add(".".join(candidate))
    return packages


def capture_setup_configuration(repo_root: Path) -> dict[str, Any]:
    """Execute the real setup discovery while replacing only ``setup()``."""
    repo_root = repo_root.resolve(strict=True)
    captured: dict[str, Any] = {}
    original_setup = setuptools.setup
    original_pybind11 = sys.modules.get("pybind11")
    pybind11_stub = types.ModuleType("pybind11")
    pybind11_stub.get_cmake_dir = lambda: "/t63/f4/not-used"

    def record_setup(**kwargs: Any) -> None:
        if captured:
            raise AssertionError("setup.py called setup() more than once")
        captured.update(kwargs)

    setuptools.setup = record_setup
    sys.modules["pybind11"] = pybind11_stub
    try:
        runpy.run_path(str(repo_root / "setup.py"), run_name="__t63_f4_setup__")
    finally:
        setuptools.setup = original_setup
        if original_pybind11 is None:
            sys.modules.pop("pybind11", None)
        else:
            sys.modules["pybind11"] = original_pybind11
    if not captured:
        raise AssertionError("setup.py did not call setuptools.setup()")
    return captured


def _wheel_package_inventory(names: set[str]) -> set[str]:
    packages: set[str] = set()
    for name in names:
        path = PurePosixPath(name)
        if not path.parts or path.parts[0] not in {"triton", "triton_anchor"}:
            continue
        if path.suffix not in WHEEL_PACKAGE_SUFFIXES:
            continue
        parts = path.parent.parts
        for length in range(1, len(parts) + 1):
            candidate = parts[:length]
            if all(part.isidentifier() for part in candidate):
                packages.add(".".join(candidate))
    return packages


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _validate_record(archive: zipfile.ZipFile, names: set[str]) -> str:
    record_paths = sorted(name for name in names if name.endswith(".dist-info/RECORD"))
    if len(record_paths) != 1:
        raise AssertionError(f"wheel must contain one RECORD: {record_paths}")
    record_path = record_paths[0]
    rows = list(
        csv.reader(io.StringIO(archive.read(record_path).decode("utf-8"), newline=""))
    )
    if any(len(row) != 3 for row in rows):
        raise AssertionError("RECORD rows must have path, hash, and size")
    indexed = {row[0]: (row[1], row[2]) for row in rows}
    if len(indexed) != len(rows):
        raise AssertionError("RECORD contains duplicate paths")
    if set(indexed) != names:
        raise AssertionError(
            "RECORD/archive mismatch: "
            f"record-only={sorted(set(indexed) - names)}, "
            f"archive-only={sorted(names - set(indexed))}"
        )
    for name in sorted(names):
        digest, size = indexed[name]
        if name == record_path:
            if digest or size:
                raise AssertionError("RECORD must omit its own hash and size")
            continue
        payload = archive.read(name)
        expected_digest = (
            base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        if digest != "sha256=" + expected_digest:
            raise AssertionError(f"RECORD hash mismatch: {name}")
        if size != str(len(payload)):
            raise AssertionError(f"RECORD size mismatch: {name}")
    return record_path


def validate_wheel_archive(wheel: Path, repo_root: Path) -> dict[str, Any]:
    """Audit source/discovery/wheel inventories and every RECORD member."""
    wheel = wheel.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    source_packages = source_package_inventory(repo_root)
    setup_configuration = capture_setup_configuration(repo_root)
    declared_packages = set(setup_configuration["packages"])
    source_payload = tracked_source_payload(repo_root)

    with zipfile.ZipFile(wheel) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise AssertionError(f"wheel contains corrupt member: {corrupt}")
        member_names = archive.namelist()
        if len(member_names) != len(set(member_names)):
            duplicates = sorted(
                name for name in set(member_names) if member_names.count(name) > 1
            )
            raise AssertionError(f"wheel contains duplicate members: {duplicates}")
        unsafe = sorted(
            name
            for name in member_names
            if "\\" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
        )
        if unsafe:
            raise AssertionError(f"wheel contains unsafe paths: {unsafe}")
        forbidden = sorted(
            name
            for name in member_names
            if PurePosixPath(name).suffix == ".pyc"
            or FORBIDDEN_PATH_PARTS.intersection(PurePosixPath(name).parts)
        )
        if forbidden:
            raise AssertionError(f"wheel contains cache/build payload: {forbidden}")
        file_names = {name for name in member_names if not name.endswith("/")}

        wheel_packages = _wheel_package_inventory(file_names)
        if source_packages != declared_packages or declared_packages != wheel_packages:
            raise AssertionError(
                "package inventory mismatch: "
                f"A-B={sorted(source_packages - declared_packages)}, "
                f"B-A={sorted(declared_packages - source_packages)}, "
                f"B-C={sorted(declared_packages - wheel_packages)}, "
                f"C-B={sorted(wheel_packages - declared_packages)}, "
                f"A-C={sorted(source_packages - wheel_packages)}"
            )
        missing_required = sorted(REQUIRED_FILES - file_names)
        if missing_required:
            raise AssertionError(
                "wheel is missing required payload: " + ", ".join(missing_required)
            )
        for wheel_path, source_path in sorted(source_payload.items()):
            if wheel_path in GENERATED_PAYLOAD_ALLOWLIST:
                continue
            if wheel_path not in file_names:
                raise AssertionError(f"current source payload omitted: {wheel_path}")
            if archive.read(wheel_path) != source_path.read_bytes():
                raise AssertionError(f"wheel payload is stale: {wheel_path}")

        record_path = _validate_record(archive, file_names)
        metadata_paths = sorted(
            name for name in file_names if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_paths) != 1:
            raise AssertionError(f"wheel must contain one METADATA: {metadata_paths}")
        metadata = email.parser.BytesParser().parsebytes(
            archive.read(metadata_paths[0])
        )
        if metadata.get("Name") != "triton-anchor":
            raise AssertionError("wheel project Name is not triton-anchor")
        if metadata.get("Version") != "0.2.0":
            raise AssertionError("wheel project Version is not 0.2.0")
        requirements = [
            Requirement(value) for value in (metadata.get_all("Requires-Dist") or [])
        ]
        packaging_requirements = [
            requirement
            for requirement in requirements
            if canonicalize_name(requirement.name) == "packaging"
        ]
        if (
            len(packaging_requirements) != 1
            or str(packaging_requirements[0].specifier) != ">=21"
            or packaging_requirements[0].extras
            or packaging_requirements[0].marker is not None
            or packaging_requirements[0].url is not None
        ):
            raise AssertionError(
                f"wheel must declare exactly packaging>=21: {requirements}"
            )

        build_info = json.loads(archive.read("triton_anchor/_build_info.json"))
        if build_info.get("generated") is not True:
            raise AssertionError("wheel build-info is not generated")
        mismatches = {
            key: {"expected": expected, "actual": build_info.get(key)}
            for key, expected in EXPECTED_BUILD_INFO.items()
            if build_info.get(key) != expected
        }
        if mismatches:
            raise AssertionError(f"wheel build-info mismatch: {mismatches}")
        null_fields = sorted(
            key for key, value in build_info.items() if value is None or value == ""
        )
        if null_fields:
            raise AssertionError(f"generated build-info is incomplete: {null_fields}")
        library_hash = _sha256(archive.read("triton/_C/libtriton.so"))
        if build_info.get("core_library_sha256") != library_hash:
            raise AssertionError("core_library_sha256 does not match wheel payload")
        material = {key: build_info.get(key) for key in ABI_MATERIAL_KEYS}
        canonical = json.dumps(
            {
                "schema": "triton-anchor-core-abi-v1",
                "material": material,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        if build_info.get("core_abi_fingerprint") != _sha256(canonical):
            raise AssertionError("Core ABI fingerprint is not bound to wheel material")

    return {
        "wheel": str(wheel),
        "wheel_size": wheel.stat().st_size,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "source_package_count": len(source_packages),
        "declared_package_count": len(declared_packages),
        "wheel_package_count": len(wheel_packages),
        "a_minus_b": sorted(source_packages - declared_packages),
        "b_minus_c": sorted(declared_packages - wheel_packages),
        "a_minus_c": sorted(source_packages - wheel_packages),
        "record": record_path,
        "member_count": len(file_names),
        "requires_dist": [str(requirement) for requirement in requirements],
        "build_info": build_info,
        "archive_status": "PASS",
    }


INSTALL_PROBE = r"""
import importlib.metadata
import json
import pathlib
import site
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
prefix = pathlib.Path(sys.prefix).resolve()
site_packages = tuple(pathlib.Path(path).resolve() for path in site.getsitepackages())

if sys.flags.isolated != 1:
    raise AssertionError("fresh import probe is not running in isolated mode")

import packaging
import triton
import triton_anchor
import triton_anchor.language
import triton_anchor.language.ext
from triton._C import libtriton
from triton._C.libtriton import ir
from triton_anchor.backends import get_backend_plugin_registry, load_build_info
from triton_anchor.pipeline import build_ttir_pipeline

modules = {
    "packaging": pathlib.Path(packaging.__file__).resolve(),
    "triton": pathlib.Path(triton.__file__).resolve(),
    "triton_anchor": pathlib.Path(triton_anchor.__file__).resolve(),
    "triton_anchor.language": pathlib.Path(triton_anchor.language.__file__).resolve(),
    "triton_anchor.language.ext": pathlib.Path(triton_anchor.language.ext.__file__).resolve(),
    "libtriton": pathlib.Path(libtriton.__file__).resolve(),
}
for name, path in modules.items():
    if not any(root == path or root in path.parents for root in site_packages):
        raise AssertionError(f"{name} loaded outside fresh site-packages: {path}")
    if path == repo or repo in path.parents:
        raise AssertionError(f"{name} leaked from source checkout: {path}")

for item in map(pathlib.Path, sys.path):
    try:
        resolved = item.resolve()
    except OSError:
        continue
    if resolved == repo or repo in resolved.parents:
        raise AssertionError(f"source checkout leaked onto sys.path: {resolved}")

context = ir.context()
ir.load_dialects(context)
libtriton.triton_shared.load_dialects(context)
pass_manager = ir.pass_manager(context)
build_ttir_pipeline(pass_manager, hw=None)

build_info = load_build_info()
if build_info["generated"] is not True:
    raise AssertionError("installed package loaded source build-info template")
registry = get_backend_plugin_registry()
distribution = importlib.metadata.distribution("triton-anchor")
if distribution.version != "0.2.0":
    raise AssertionError(f"installed version mismatch: {distribution.version}")

print(json.dumps({
    "python": sys.executable,
    "prefix": str(prefix),
    "site_packages": [str(path) for path in site_packages],
    "sys_path": sys.path,
    "module_paths": {name: str(path) for name, path in modules.items()},
    "distribution_version": distribution.version,
    "packaging_version": importlib.metadata.version("packaging"),
    "build_info": build_info,
    "registry_type": type(registry).__name__,
    "dialect_context": type(context).__name__,
    "pass_manager": type(pass_manager).__name__,
    "installed_status": "PASS",
}, sort_keys=True))
"""


def _clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "TRITON_PASS_PLUGIN_PATH",
    ):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def install_and_probe(
    wheel: Path,
    repo_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Install into a new venv and import only under Python isolated mode."""
    wheel = wheel.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = work_dir / "fresh-venv"
    venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
    python = venv_dir / "bin" / "python"
    env = _clean_environment()

    install = _run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(wheel),
        ],
        cwd=work_dir,
        env=env,
    )
    pip_check = _run(
        [str(python), "-I", "-m", "pip", "check"],
        cwd=work_dir,
        env=env,
    )
    probe = _run(
        [str(python), "-I", "-c", INSTALL_PROBE, str(repo_root)],
        cwd=work_dir,
        env=env,
    )
    installed_smoke = _run(
        [str(python), "-I", str(repo_root / "tests/test_smoke.py")],
        cwd=work_dir,
        env=env,
    )
    lines = [line for line in probe.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError("fresh isolated probe produced no JSON evidence")
    result = json.loads(lines[-1])
    result.update(
        {
            "pip_install_exit": install.returncode,
            "pip_check_exit": pip_check.returncode,
            "pip_check_output": pip_check.stdout.strip(),
            "import_probe_exit": probe.returncode,
            "installed_smoke_exit": installed_smoke.returncode,
            "installed_smoke_output": installed_smoke.stdout,
        }
    )
    return result
