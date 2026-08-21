"""Archive-level checks for a wheel supplied by the acceptance runner."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from packaging_wheel_acceptance import (  # noqa: E402
    ABI_MATERIAL_KEYS,
    EXPECTED_BUILD_INFO,
    REQUIRED_EXTENSION_PACKAGE,
    setup_distribution_package_inventory,
    source_python_package_inventory,
    validate_wheel_archive,
)


def _supplied_wheel() -> Path:
    wheel = os.environ.get("T63_CORE_WHEEL")
    if not wheel:
        pytest.skip("T63_CORE_WHEEL was not supplied; project wheel not available")
    return Path(wheel).resolve(strict=True)


def test_built_wheel_payload_and_provenance() -> None:
    result = validate_wheel_archive(_supplied_wheel(), HERE.parents[1])
    assert result["archive_status"] == "PASS"


def test_built_wheel_build_info_and_abi_binding_independent_of_payload() -> None:
    """Keep provenance observable even when another required member is absent."""
    with zipfile.ZipFile(_supplied_wheel()) as archive:
        names = archive.namelist()
        assert names.count("triton_anchor/_build_info.json") == 1
        info = json.loads(archive.read("triton_anchor/_build_info.json"))
        assert info["generated"] is True
        assert {
            field: info.get(field)
            for field in EXPECTED_BUILD_INFO
        } == EXPECTED_BUILD_INFO
        assert not [field for field, value in info.items() if value is None]

        library = archive.read("triton/_C/libtriton.so")
        library_hash = "sha256:" + hashlib.sha256(library).hexdigest()
        assert info["core_library_sha256"] == library_hash
        material = {field: info.get(field) for field in ABI_MATERIAL_KEYS}
        assert all(value is not None and value != "" for value in material.values())
        payload = {
            "schema": "triton-anchor-core-abi-v1",
            "material": material,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        fingerprint = "sha256:" + hashlib.sha256(canonical).hexdigest()
        assert info["core_abi_fingerprint"] == fingerprint


def test_packaging_is_a_runtime_dependency_in_setup() -> None:
    setup_text = (HERE.parents[1] / "setup.py").read_text(encoding="utf-8")
    assert 'install_requires=["packaging>=21"]' in setup_text


def test_source_template_is_explicitly_not_generated() -> None:
    import json

    template_path = HERE.parents[1] / "python/triton_anchor/_build_info.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["generated"] is False
    assert template["core_version"] is None
    assert template["backend_protocol_version"] is None
    assert template["manifest_schema_version"] is None


def test_every_source_python_package_is_declared_for_the_wheel() -> None:
    repo_root = HERE.parents[1]
    source_packages = source_python_package_inventory(repo_root)
    declared_packages = setup_distribution_package_inventory(repo_root)

    assert REQUIRED_EXTENSION_PACKAGE in source_packages
    assert REQUIRED_EXTENSION_PACKAGE in declared_packages
    omitted = sorted(source_packages - declared_packages)
    assert not omitted, (
        "source Python package(s) omitted from setup.py Distribution.packages: "
        + ", ".join(omitted)
    )


def test_source_language_extension_namespace_is_importable() -> None:
    repo_root = HERE.parents[1]
    probe = r'''\
import importlib
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo / "python"))
language = importlib.import_module("triton_anchor.language")
extension = importlib.import_module("triton_anchor.language.ext")
if not hasattr(language, "__path__"):
    raise AssertionError("triton_anchor.language lost its package identity")
expected = repo / "python/triton_anchor/language/ext/__init__.py"
if pathlib.Path(extension.__file__).resolve() != expected:
    raise AssertionError(f"extension loaded from unexpected path: {extension.__file__}")
'''
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(repo_root)],
        cwd=repo_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
