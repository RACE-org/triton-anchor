#!/usr/bin/env python3
"""First-principles package and wheel inventory helpers for T6.3 F4.

The source oracle is derived from tracked payload paths and does not call
setuptools.  The declared inventory is captured independently by executing the
real ``setup.py`` with only its final ``setup()`` call replaced.  Wheel and
RECORD checks operate on the built archive rather than a build directory.
"""

from __future__ import annotations

import base64
import configparser
import csv
import email.parser
import hashlib
import io
import json
import os
import runpy
import stat
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
SOURCE_TESTS_PREFIX = PurePosixPath("python/triton_anchor/tests")
WHEEL_TESTS_PREFIX = PurePosixPath("triton_anchor/tests")
WHEEL_PACKAGE_ROOTS = frozenset({"triton", "triton_anchor"})
EXPECTED_WHEEL_FILENAME = "triton_anchor-0.2.0-cp312-cp312-linux_x86_64.whl"
DIST_INFO_DIRECTORY = "triton_anchor-0.2.0.dist-info"
DIST_INFO_MEMBERS = frozenset(
    {
        f"{DIST_INFO_DIRECTORY}/licenses/LICENSE",
        f"{DIST_INFO_DIRECTORY}/METADATA",
        f"{DIST_INFO_DIRECTORY}/WHEEL",
        f"{DIST_INFO_DIRECTORY}/entry_points.txt",
        f"{DIST_INFO_DIRECTORY}/top_level.txt",
        f"{DIST_INFO_DIRECTORY}/RECORD",
    }
)
BINARY_MEMBERS = frozenset(
    {
        "triton/_C/libtriton.so",
        "triton/bin/triton-shared-opt",
    }
)
EXPECTED_ENTRY_POINT = (
    "triton.adapters",
    "triton-shared",
    "triton_anchor.adapters.triton_shared_adapter:TritonSharedAdapter",
)
SMOKE_SCHEMA = "triton-anchor-t63-f4-smoke-v1"
SMOKE_IDENTITIES = (
    "test_import_triton",
    "test_import_triton_anchor",
    "test_libtriton_binding",
    "test_triton_shared_plugin_binding",
    "test_mlir_context_and_dialects",
    "test_hw_capability",
    "test_anchor_ir_validator",
    "test_ttir_pipeline",
    "test_adapter_discovery",
    "test_ttir_generation",
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
    "triton_anchor/adapters/triton_shared_adapter.py",
    "triton_anchor/language/ext/__init__.py",
    "triton/_C/libtriton.so",
    "triton/bin/triton-shared-opt",
    *DIST_INFO_MEMBERS,
}
FORBIDDEN_PATH_PARTS = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".git",
        "build",
        "dist",
    }
)
SOURCE_PAYLOAD_SUFFIXES = frozenset({".py", ".pyi", ".json"})
WHEEL_PACKAGE_SUFFIXES = frozenset({".py", ".pyi", ".json", ".so"})
GENERATED_PAYLOAD_ALLOWLIST = frozenset({"triton_anchor/_build_info.json"})


def _is_under(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    return path == prefix or prefix in path.parents


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
            "--stage",
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
        metadata, separator, path_text = raw_path.partition("\t")
        if not separator:
            raise AssertionError(f"malformed git ls-files entry: {raw_path!r}")
        fields = metadata.split()
        if len(fields) != 3:
            raise AssertionError(f"malformed git index metadata: {metadata!r}")
        mode, _object_id, stage = fields
        if stage != "0" or mode not in {"100644", "100755"}:
            raise AssertionError(
                f"package source must be a stage-0 regular blob: "
                f"mode={mode}, stage={stage}, path={path_text}"
            )

        relative = PurePosixPath(path_text)
        source_path = repo_root / Path(*relative.parts)
        if source_path.is_symlink() or not source_path.is_file():
            raise AssertionError(
                f"tracked package source is not a regular file: {relative}"
            )
        if _is_under(relative, SOURCE_TESTS_PREFIX):
            continue
        if relative.suffix not in SOURCE_PAYLOAD_SUFFIXES:
            continue
        for source_root, package_name in PACKAGE_ROOTS:
            try:
                inside = relative.relative_to(source_root)
            except ValueError:
                continue
            wheel_path = PurePosixPath(package_name) / inside
            wheel_name = str(wheel_path)
            if wheel_name in result:
                raise AssertionError(f"duplicate source-to-wheel mapping: {wheel_name}")
            result[wheel_name] = source_path
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
    """Derive C_wheel without treating binary-only ``triton/bin`` as a package."""
    packages: set[str] = set()
    for name in names:
        path = PurePosixPath(name)
        if not path.parts or path.parts[0] not in WHEEL_PACKAGE_ROOTS:
            continue
        if _is_under(path, WHEEL_TESTS_PREFIX):
            continue
        if path.suffix not in WHEEL_PACKAGE_SUFFIXES:
            continue
        parts = path.parent.parts
        for length in range(1, len(parts) + 1):
            candidate = parts[:length]
            if all(part.isidentifier() for part in candidate):
                packages.add(".".join(candidate))
    return packages


def expected_release_file_members(repo_root: Path) -> set[str]:
    """Return the exact regular-file equation frozen by ADR-0002 F4."""
    return (
        set(tracked_source_payload(repo_root))
        | set(BINARY_MEMBERS)
        | set(DIST_INFO_MEMBERS)
    )


def release_member_delta(
    actual: set[str],
    expected: set[str],
) -> dict[str, list[str]]:
    """Compare actual wheel files with the exact release payload equation."""
    return {
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
        "tests": sorted(
            name
            for name in actual
            if _is_under(PurePosixPath(name), WHEEL_TESTS_PREFIX)
        ),
    }


def wheel_release_member_evidence(wheel: Path, repo_root: Path) -> dict[str, Any]:
    """Collect exact-member evidence without short-circuiting a RED assertion."""
    wheel = wheel.resolve(strict=True)
    expected = expected_release_file_members(repo_root)
    with zipfile.ZipFile(wheel) as archive:
        actual = {
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and not info.filename.endswith("/")
        }
    delta = release_member_delta(actual, expected)
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        **delta,
    }


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


def _allowed_directory_entries(file_names: set[str]) -> set[str]:
    allowed: set[str] = set()
    for name in file_names:
        parent = PurePosixPath(name).parent
        while parent.parts:
            allowed.add(str(parent) + "/")
            parent = parent.parent
    return allowed


def validate_zip_members(
    archive: zipfile.ZipFile,
) -> tuple[set[str], list[str]]:
    infos = archive.infolist()
    member_names = [info.filename for info in infos]
    if len(member_names) != len(set(member_names)):
        duplicates = sorted(
            name for name in set(member_names) if member_names.count(name) > 1
        )
        raise AssertionError(f"wheel contains duplicate members: {duplicates}")

    unsafe: list[str] = []
    encrypted: list[str] = []
    links_or_special: list[str] = []
    invalid_directories: list[dict[str, object]] = []
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        canonical = str(path) + ("/" if info.is_dir() else "")
        if (
            not name
            or "\x00" in name
            or "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or name != canonical
        ):
            unsafe.append(name)
        if info.flag_bits & 0x1:
            encrypted.append(name)
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if info.is_dir() and (
            info.file_size != 0 or file_type not in {0, stat.S_IFDIR}
        ):
            invalid_directories.append(
                {
                    "name": name,
                    "size": info.file_size,
                    "mode": oct(unix_mode),
                }
            )
        if stat.S_ISLNK(unix_mode) or (
            not info.is_dir() and file_type not in {0, stat.S_IFREG}
        ):
            links_or_special.append(name)
    if unsafe:
        raise AssertionError(f"wheel contains unsafe paths: {sorted(unsafe)}")
    if encrypted:
        raise AssertionError(f"wheel contains encrypted members: {sorted(encrypted)}")
    if links_or_special:
        raise AssertionError(
            f"wheel contains symlink/special members: {sorted(links_or_special)}"
        )
    if invalid_directories:
        raise AssertionError(
            "wheel contains non-empty or non-directory directory entries: "
            + json.dumps(invalid_directories, sort_keys=True)
        )

    forbidden = sorted(
        name
        for name in member_names
        if PurePosixPath(name).suffix in {".pyc", ".pyo"}
        or FORBIDDEN_PATH_PARTS.intersection(PurePosixPath(name).parts)
    )
    if forbidden:
        raise AssertionError(f"wheel contains cache/build payload: {forbidden}")

    file_names = {
        info.filename
        for info in infos
        if not info.is_dir() and not info.filename.endswith("/")
    }
    directory_names = sorted(
        info.filename for info in infos if info.is_dir() or info.filename.endswith("/")
    )
    unexpected_directories = sorted(
        set(directory_names) - _allowed_directory_entries(file_names)
    )
    if unexpected_directories:
        raise AssertionError(
            f"wheel contains unexpected directory entries: {unexpected_directories}"
        )
    return file_names, directory_names


def _entry_point_tuples(payload: bytes) -> list[tuple[str, str, str]]:
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read_string(payload.decode("utf-8"))
    return sorted(
        (section, name, value.strip())
        for section in parser.sections()
        for name, value in parser.items(section)
    )


def validate_requires_dist(values: list[str]) -> list[Requirement]:
    """Require the single runtime dependency frozen by ADR-0002 F4-06."""
    requirements = [Requirement(value) for value in values]
    if len(requirements) != 1:
        raise AssertionError(
            f"wheel must declare exactly one Requires-Dist: {requirements}"
        )
    requirement = requirements[0]
    if (
        canonicalize_name(requirement.name) != "packaging"
        or str(requirement.specifier) != ">=21"
        or requirement.extras
        or requirement.marker is not None
        or requirement.url is not None
    ):
        raise AssertionError(
            f"wheel must declare exactly packaging>=21: {requirements}"
        )
    return requirements


def validate_wheel_archive(wheel: Path, repo_root: Path) -> dict[str, Any]:
    """Audit source/discovery/wheel inventories and every RECORD member."""
    wheel = wheel.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    if wheel.name != EXPECTED_WHEEL_FILENAME:
        raise AssertionError(
            f"wheel filename mismatch: expected {EXPECTED_WHEEL_FILENAME}, got {wheel.name}"
        )
    source_packages = source_package_inventory(repo_root)
    setup_configuration = capture_setup_configuration(repo_root)
    declared_packages = set(setup_configuration["packages"])
    source_payload = tracked_source_payload(repo_root)
    expected_members = expected_release_file_members(repo_root)

    with zipfile.ZipFile(wheel) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise AssertionError(f"wheel contains corrupt member: {corrupt}")
        file_names, directory_names = validate_zip_members(archive)
        member_delta = release_member_delta(file_names, expected_members)
        if any(member_delta.values()):
            raise AssertionError(
                "wheel violates exact release member equation: "
                + json.dumps(member_delta, sort_keys=True)
            )

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
        metadata_path = f"{DIST_INFO_DIRECTORY}/METADATA"
        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_path))
        if metadata.get("Name") != "triton-anchor":
            raise AssertionError("wheel project Name is not triton-anchor")
        if metadata.get("Version") != "0.2.0":
            raise AssertionError("wheel project Version is not 0.2.0")
        if metadata.get("Requires-Python") != ">=3.10":
            raise AssertionError("wheel project Requires-Python is not >=3.10")
        requirements = validate_requires_dist(metadata.get_all("Requires-Dist") or [])

        wheel_metadata = email.parser.BytesParser().parsebytes(
            archive.read(f"{DIST_INFO_DIRECTORY}/WHEEL")
        )
        if wheel_metadata.get("Root-Is-Purelib") != "false":
            raise AssertionError("wheel Root-Is-Purelib is not false")
        if wheel_metadata.get_all("Tag") != ["cp312-cp312-linux_x86_64"]:
            raise AssertionError(f"wheel Tag mismatch: {wheel_metadata.get_all('Tag')}")
        entry_points = _entry_point_tuples(
            archive.read(f"{DIST_INFO_DIRECTORY}/entry_points.txt")
        )
        if entry_points != [EXPECTED_ENTRY_POINT]:
            raise AssertionError(f"wheel entry points mismatch: {entry_points}")
        if archive.read(f"{DIST_INFO_DIRECTORY}/top_level.txt") != (
            b"triton\ntriton_anchor\n"
        ):
            raise AssertionError("wheel top_level.txt is not canonical")
        if (
            archive.read(f"{DIST_INFO_DIRECTORY}/licenses/LICENSE")
            != (repo_root / "LICENSE").read_bytes()
        ):
            raise AssertionError("wheel LICENSE does not match candidate LICENSE")

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
        "directory_entry_count": len(directory_names),
        "missing_members": member_delta["missing"],
        "unexpected_members": member_delta["unexpected"],
        "tests_members": member_delta["tests"],
        "entry_points": entry_points,
        "requires_dist": [str(requirement) for requirement in requirements],
        "build_info": build_info,
        "archive_status": "PASS",
    }


INSTALL_PROBE = r"""
import importlib.metadata
import importlib.util
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

if triton.__version__ != "3.6.0":
    raise AssertionError(f"Triton version mismatch: {triton.__version__}")
if triton_anchor.__version__ != "0.2.0":
    raise AssertionError(f"Triton Anchor version mismatch: {triton_anchor.__version__}")

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
tests_spec = importlib.util.find_spec("triton_anchor.tests")
shared_opt_path = (pathlib.Path(triton.__file__).resolve().parent / "bin" / "triton-shared-opt").resolve(strict=True)
if not any(root == shared_opt_path or root in shared_opt_path.parents for root in site_packages):
    raise AssertionError(f"triton-shared-opt is outside fresh site-packages: {shared_opt_path}")
entry_points = sorted(
    (entry.group, entry.name, entry.value)
    for entry in distribution.entry_points
)

print(json.dumps({
    "python": sys.executable,
    "prefix": str(prefix),
    "site_packages": [str(path) for path in site_packages],
    "sys_path": sys.path,
    "module_paths": {name: str(path) for name, path in modules.items()},
    "distribution_version": distribution.version,
    "entry_points": entry_points,
    "packaging_version": importlib.metadata.version("packaging"),
    "shared_opt_path": str(shared_opt_path),
    "tests_spec": None if tests_spec is None else {
        "origin": tests_spec.origin,
        "submodule_search_locations": list(tests_spec.submodule_search_locations or ()),
    },
    "build_info": build_info,
    "registry_type": type(registry).__name__,
    "dialect_context": type(context).__name__,
    "pass_manager": type(pass_manager).__name__,
    "installed_status": "PASS",
}, sort_keys=True))
"""


SMOKE_DRIVER = r"""
import hashlib
import json
import pathlib
import runpy
import sys

schema = "triton-anchor-t63-f4-smoke-v1"
identities = (
    "test_import_triton",
    "test_import_triton_anchor",
    "test_libtriton_binding",
    "test_triton_shared_plugin_binding",
    "test_mlir_context_and_dialects",
    "test_hw_capability",
    "test_anchor_ir_validator",
    "test_ttir_pipeline",
    "test_adapter_discovery",
    "test_ttir_generation",
)
driver = pathlib.Path(sys.argv[1]).resolve(strict=True)
driver_sha256 = hashlib.sha256(driver.read_bytes()).hexdigest()
namespace = runpy.run_path(str(driver), run_name="__t63_f4_installed_smoke__")
results = []
for identity in identities:
    item = {"schema": schema, "identity": identity, "status": "PASS"}
    try:
        function = namespace[identity]
        if not callable(function):
            raise TypeError(f"smoke identity is not callable: {identity}")
        function()
    except BaseException as error:
        item["status"] = "FAIL"
        item["error_type"] = type(error).__name__
        try:
            item["error"] = str(error)
        except BaseException:
            item["error"] = "<unprintable>"
    results.append(item)
    print("T63_SMOKE_RESULT=" + json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

passed = sum(item["status"] == "PASS" for item in results)
failed = sum(item["status"] == "FAIL" for item in results)
summary = {
    "schema": schema,
    "ordered_identities": [item["identity"] for item in results],
    "pass": passed,
    "fail": failed,
    "skip": 0,
    "total": len(results),
    "driver_sha256": driver_sha256,
}
print("T63_SMOKE_SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
if failed:
    raise SystemExit(1)
"""


def _parse_installed_smoke(
    output: str,
    *,
    driver_sha256: str,
) -> dict[str, Any]:
    result_prefix = "T63_SMOKE_RESULT="
    summary_prefix = "T63_SMOKE_SUMMARY="
    results = [
        json.loads(line.removeprefix(result_prefix))
        for line in output.splitlines()
        if line.startswith(result_prefix)
    ]
    summaries = [
        json.loads(line.removeprefix(summary_prefix))
        for line in output.splitlines()
        if line.startswith(summary_prefix)
    ]
    expected_results = [
        {"schema": SMOKE_SCHEMA, "identity": identity, "status": "PASS"}
        for identity in SMOKE_IDENTITIES
    ]
    expected_summary = {
        "schema": SMOKE_SCHEMA,
        "ordered_identities": list(SMOKE_IDENTITIES),
        "pass": 10,
        "fail": 0,
        "skip": 0,
        "total": 10,
        "driver_sha256": driver_sha256,
    }
    if results != expected_results or summaries != [expected_summary]:
        raise AssertionError(
            "installed smoke must produce exact 10/0/0 identities: "
            + json.dumps(
                {"results": results, "summaries": summaries},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return {"results": results, "summary": summaries[0]}


def _clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "LD_AUDIT",
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
    smoke_source = repo_root / "tests/test_smoke.py"
    smoke_driver_sha256 = hashlib.sha256(smoke_source.read_bytes()).hexdigest()
    installed_smoke = _run(
        [str(python), "-I", "-c", SMOKE_DRIVER, str(smoke_source)],
        cwd=work_dir,
        env=env,
    )
    lines = [line for line in probe.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError("fresh isolated probe produced no JSON evidence")
    result = json.loads(lines[-1])
    smoke_evidence = _parse_installed_smoke(
        installed_smoke.stdout,
        driver_sha256=smoke_driver_sha256,
    )
    shared_opt_path = Path(result["shared_opt_path"]).resolve(strict=True)
    cli_command = [str(shared_opt_path), "--version"]
    cli = subprocess.run(
        cli_command,
        cwd=work_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result.update(
        {
            "pip_install_exit": install.returncode,
            "pip_check_exit": pip_check.returncode,
            "pip_check_output": pip_check.stdout.strip(),
            "import_probe_exit": probe.returncode,
            "installed_smoke_exit": installed_smoke.returncode,
            "installed_smoke_output": installed_smoke.stdout,
            "installed_smoke_results": smoke_evidence["results"],
            "installed_smoke_summary": smoke_evidence["summary"],
            "shared_opt_command": cli_command,
            "shared_opt_exit": cli.returncode,
            "shared_opt_output": cli.stdout,
            "loader_environment_absent": {
                key: key not in env
                for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")
            },
        }
    )
    return result
