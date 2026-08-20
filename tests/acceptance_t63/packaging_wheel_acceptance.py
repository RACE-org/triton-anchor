#!/usr/bin/env python3
"""Isolated acceptance probe for a built triton-anchor wheel.

This script deliberately does not build the wheel.  It validates an existing
artifact, installs it (and its declared dependencies) into a fresh virtual
environment, and imports it with Python isolated mode so a source checkout
cannot make a broken wheel look healthy.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


EXPECTED_BUILD_INFO = {
    "schema_version": "1.1",
    "core_version": "0.2.0",
    "backend_protocol_version": "1.0",
    "manifest_schema_version": "1.0",
    "triton_version": "3.3.0",
    "vendored_triton_commit": "523a1b235b213bc192f2d5a8999add5bf2d0fea5",
    "expected_llvm_project_commit": "a66376b0dc3b2ea8a84fda26faca287980986f78",
    "actual_llvm_commit": "a66376b0dc3b2ea8a84fda26faca287980986f78",
    "actual_mlir_commit": "a66376b0dc3b2ea8a84fda26faca287980986f78",
    "cxx_standard": "17",
    "core_abi_fingerprint_schema": "triton-anchor-core-abi-v1",
}

REQUIRED_WHEEL_FILES = {
    "triton_anchor/__init__.py",
    "triton_anchor/_version.py",
    "triton_anchor/_build_info.json",
    "triton_anchor/backends/__init__.py",
    "triton_anchor/backends/capabilities.py",
    "triton_anchor/backends/compatibility.py",
    "triton_anchor/backends/conflicts.py",
    "triton_anchor/backends/environment.py",
    "triton_anchor/backends/errors.py",
    "triton_anchor/backends/legacy.py",
    "triton_anchor/backends/manifest.py",
    "triton_anchor/backends/native.py",
    "triton_anchor/backends/protocol.py",
    "triton_anchor/backends/registry.py",
    "triton_anchor/backends/selection.py",
    "triton_anchor/backends/schemas/backend_manifest.schema.json",
    "triton_anchor/backends/examples/triton_anchor_backend.example.json",
    "triton_anchor/language/ext/__init__.py",
    "triton/__init__.py",
    "triton/backends/__init__.py",
    "triton/backends/compiler.py",
    "triton/backends/driver.py",
    "triton/compiler/compiler.py",
    "triton/_C/libtriton.so",
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

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_wheel_archive(wheel: Path) -> dict:
    """Validate package payload and independently verify build provenance."""
    wheel = Path(wheel).resolve(strict=True)
    _assert(wheel.suffix == ".whl", f"not a wheel path: {wheel}")

    with zipfile.ZipFile(wheel) as archive:
        corrupt = archive.testzip()
        _assert(corrupt is None, f"wheel contains corrupt member: {corrupt}")
        names = tuple(archive.namelist())
        name_set = set(names)
        unsafe = [
            name
            for name in names
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        ]
        _assert(not unsafe, f"wheel contains unsafe member path(s): {unsafe}")

        missing = sorted(REQUIRED_WHEEL_FILES.difference(name_set))
        _assert(not missing, "wheel is missing required file(s): " + ", ".join(missing))

        build_info_paths = [
            name for name in names if PurePosixPath(name).name == "_build_info.json"
        ]
        _assert(
            build_info_paths == ["triton_anchor/_build_info.json"],
            f"wheel must contain exactly one canonical _build_info.json: {build_info_paths}",
        )
        build_info = json.loads(archive.read(build_info_paths[0]))
        _assert(isinstance(build_info, dict), "_build_info.json root must be an object")
        _assert(build_info.get("generated") is True, "wheel build info is not generated")

        mismatches = {
            field: {"expected": expected, "actual": build_info.get(field)}
            for field, expected in EXPECTED_BUILD_INFO.items()
            if build_info.get(field) != expected
        }
        _assert(not mismatches, f"wheel build-info mismatch(es): {mismatches}")

        # The acceptance contract says source-only unknowns may be null, but a
        # final wheel may not retain any of those null template values.
        null_fields = sorted(field for field, value in build_info.items() if value is None)
        _assert(not null_fields, "generated wheel has null build field(s): " + ", ".join(null_fields))

        for field in ("core_library_sha256", "core_abi_fingerprint"):
            _assert(
                isinstance(build_info.get(field), str)
                and SHA256_RE.fullmatch(build_info[field]) is not None,
                f"{field} is not a canonical sha256 fingerprint",
            )

        actual_library_hash = _sha256(archive.read("triton/_C/libtriton.so"))
        _assert(
            build_info["core_library_sha256"] == actual_library_hash,
            "core_library_sha256 does not match packaged triton/_C/libtriton.so",
        )
        material = {key: build_info.get(key) for key in ABI_MATERIAL_KEYS}
        _assert(
            all(value is not None and value != "" for value in material.values()),
            f"Core ABI material is incomplete: {material}",
        )
        fingerprint_payload = {
            "schema": "triton-anchor-core-abi-v1",
            "material": material,
        }
        canonical = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        expected_fingerprint = _sha256(canonical)
        _assert(
            build_info["core_abi_fingerprint"] == expected_fingerprint,
            "Core ABI fingerprint does not match independently canonicalized material",
        )

        schema = json.loads(
            archive.read("triton_anchor/backends/schemas/backend_manifest.schema.json")
        )
        example = json.loads(
            archive.read(
                "triton_anchor/backends/examples/triton_anchor_backend.example.json"
            )
        )
        _assert(schema.get("type") == "object", "packaged Manifest Schema is malformed")
        _assert(
            schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
            "packaged Manifest Schema uses the wrong JSON Schema dialect",
        )
        _assert(example.get("schema_version") == "1.0", "example has wrong schema_version")
        _assert(isinstance(example.get("plugins"), list) and example["plugins"], "example has no plugin")

        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_metadata_paths = [name for name in names if name.endswith(".dist-info/WHEEL")]
        record_paths = [name for name in names if name.endswith(".dist-info/RECORD")]
        _assert(len(metadata_paths) == 1, f"expected one METADATA file: {metadata_paths}")
        _assert(len(wheel_metadata_paths) == 1, f"expected one WHEEL file: {wheel_metadata_paths}")
        _assert(len(record_paths) == 1, f"expected one RECORD file: {record_paths}")
        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_paths[0]))
        _assert(metadata.get("Name") == "triton-anchor", "wheel project Name is not triton-anchor")
        _assert(metadata.get("Version") == "0.2.0", "wheel project Version is not 0.2.0")
        requirements = [
            Requirement(value) for value in (metadata.get_all("Requires-Dist") or [])
        ]
        packaging_requirements = [
            req for req in requirements if canonicalize_name(req.name) == "packaging"
        ]
        _assert(
            len(packaging_requirements) == 1,
            f"wheel must declare packaging exactly once: {requirements}",
        )
        _assert(
            str(packaging_requirements[0].specifier) == ">=21",
            f"unexpected packaging constraint: {packaging_requirements[0]}",
        )

    return {
        "wheel": str(wheel),
        "wheel_size": wheel.stat().st_size,
        "member_count": len(names),
        "build_info": build_info,
        "requires_dist": [str(req) for req in requirements],
        "archive_status": "PASS",
    }


INSTALL_PROBE = r'''
import importlib.metadata
import importlib.resources
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
prefix = pathlib.Path(sys.prefix).resolve()

import packaging
import triton
import triton_anchor
from triton._C import libtriton
from triton_anchor.backends import get_backend_plugin_registry, load_build_info

modules = {
    "packaging": pathlib.Path(packaging.__file__).resolve(),
    "triton": pathlib.Path(triton.__file__).resolve(),
    "triton_anchor": pathlib.Path(triton_anchor.__file__).resolve(),
    "libtriton": pathlib.Path(libtriton.__file__).resolve(),
}
for name, path in modules.items():
    if prefix not in path.parents:
        raise AssertionError(f"{name} loaded outside fresh venv: {path}")
    if path == repo or repo in path.parents:
        raise AssertionError(f"{name} leaked from repository: {path}")

for item in map(pathlib.Path, sys.path):
    try:
        resolved = item.resolve()
    except OSError:
        continue
    if resolved == repo or repo in resolved.parents:
        raise AssertionError(f"repository leaked onto sys.path: {resolved}")

dist = importlib.metadata.distribution("triton-anchor")
if dist.version != "0.2.0":
    raise AssertionError(f"installed distribution version mismatch: {dist.version}")
requires = dist.requires or []
if not any(value.lower().startswith("packaging") for value in requires):
    raise AssertionError(f"installed metadata omits packaging: {requires}")
if importlib.metadata.version("packaging") is None:
    raise AssertionError("packaging dependency is not installed")

build_info = load_build_info()
if build_info["generated"] is not True:
    raise AssertionError("installed package loaded source build metadata")
schema = importlib.resources.files("triton_anchor").joinpath(
    "backends/schemas/backend_manifest.schema.json"
)
example = importlib.resources.files("triton_anchor").joinpath(
    "backends/examples/triton_anchor_backend.example.json"
)
if not schema.is_file() or not example.is_file():
    raise AssertionError("installed package data is incomplete")

registry = get_backend_plugin_registry()
result = {
    "python": sys.executable,
    "prefix": str(prefix),
    "module_paths": {key: str(value) for key, value in modules.items()},
    "sys_path": sys.path,
    "distribution_version": dist.version,
    "packaging_version": importlib.metadata.version("packaging"),
    "build_info": build_info,
    "registry_type": type(registry).__name__,
    "installed_status": "PASS",
}
print(json.dumps(result, sort_keys=True))
'''


def _clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def install_and_probe(wheel: Path, repo_root: Path, work_dir: Path) -> dict:
    """Install normally into an empty venv and import with ``python -I``."""
    wheel = Path(wheel).resolve(strict=True)
    repo_root = Path(repo_root).resolve(strict=True)
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = work_dir / "fresh-venv"
    venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
    python = venv_dir / "bin" / "python"
    env = _clean_environment()

    install = subprocess.run(
        [str(python), "-I", "-m", "pip", "install", str(wheel)],
        cwd=work_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print("[fresh-venv pip install]\n" + install.stdout, file=sys.stderr)
    _assert(install.returncode == 0, f"fresh venv install failed: exit {install.returncode}")

    pip_check = subprocess.run(
        [str(python), "-I", "-m", "pip", "check"],
        cwd=work_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print("[fresh-venv pip check]\n" + pip_check.stdout, file=sys.stderr)
    _assert(pip_check.returncode == 0, f"pip check failed: exit {pip_check.returncode}")

    probe = subprocess.run(
        [str(python), "-I", "-c", INSTALL_PROBE, str(repo_root)],
        cwd=work_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    print("[fresh-venv import probe stderr]\n" + probe.stderr, file=sys.stderr)
    _assert(probe.returncode == 0, f"fresh venv import probe failed: {probe.stderr}")
    lines = [line for line in probe.stdout.splitlines() if line.strip()]
    _assert(lines, "fresh venv import probe produced no JSON evidence")
    result = json.loads(lines[-1])
    result.update(
        {
            "pip_install_exit": install.returncode,
            "pip_check_exit": pip_check.returncode,
            "import_probe_exit": probe.returncode,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Fresh empty directory to retain. If omitted, use a temporary directory.",
    )
    args = parser.parse_args()

    archive_result = validate_wheel_archive(args.wheel)
    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="t63-wheel-acceptance-") as temporary:
            installed_result = install_and_probe(
                args.wheel, args.repo_root, Path(temporary)
            )
    else:
        _assert(
            not args.work_dir.exists() or not any(args.work_dir.iterdir()),
            f"--work-dir must be absent or empty: {args.work_dir}",
        )
        installed_result = install_and_probe(
            args.wheel, args.repo_root, args.work_dir
        )
    print(
        json.dumps(
            {"archive": archive_result, "installed": installed_result},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
