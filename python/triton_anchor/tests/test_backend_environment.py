"""Tests for W2 version sources and environment collection."""

import json
import re
from pathlib import Path

import pytest

import triton_anchor
from triton_anchor._version import (
    BACKEND_MANIFEST_SCHEMA_VERSION,
    BACKEND_PLUGIN_PROTOCOL_VERSION,
    CORE_VERSION,
)
from triton_anchor.backends import (
    collect_core_environment,
    load_build_info,
    normalize_toolchain_version,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_INFO_PATH = REPO_ROOT / "python" / "triton_anchor" / "_build_info.json"


def test_public_versions_have_one_runtime_source():
    assert CORE_VERSION == "0.2.0"
    assert triton_anchor.__version__ == CORE_VERSION
    assert BACKEND_PLUGIN_PROTOCOL_VERSION == "1.0"
    assert BACKEND_MANIFEST_SCHEMA_VERSION == "1.0"


def test_source_template_does_not_duplicate_public_versions():
    raw = json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8"))
    assert raw["generated"] is False
    assert raw["core_version"] is None
    assert raw["backend_protocol_version"] is None
    assert raw["manifest_schema_version"] is None

    resolved = load_build_info()
    assert resolved["core_version"] == CORE_VERSION
    assert resolved["backend_protocol_version"] == BACKEND_PLUGIN_PROTOCOL_VERSION
    assert resolved["manifest_schema_version"] == BACKEND_MANIFEST_SCHEMA_VERSION


def test_source_build_info_matches_vendored_version_markers():
    info = load_build_info()
    triton_init = (
        REPO_ROOT / "triton" / "python" / "triton" / "__init__.py"
    ).read_text(encoding="utf-8")
    triton_version = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
        triton_init,
        flags=re.MULTILINE,
    ).group(1)
    triton_marker = (REPO_ROOT / "triton" / "TRITON_VERSION").read_text(
        encoding="utf-8"
    )
    triton_commit = re.search(
        r"^# Commit:\s*([0-9a-fA-F]+)\s*$",
        triton_marker,
        flags=re.MULTILINE,
    ).group(1)
    llvm_commit = (
        REPO_ROOT / "triton" / "cmake" / "llvm-hash.txt"
    ).read_text(encoding="utf-8").strip()

    assert info["triton_version"] == triton_version
    assert info["vendored_triton_commit"] == triton_commit
    assert info["expected_llvm_project_commit"] == llvm_commit


def test_source_build_info_does_not_claim_unknown_build_facts():
    info = load_build_info()
    assert info["generated"] is False
    assert info["actual_llvm_commit"] is None
    assert info["actual_mlir_commit"] is None
    assert info["cxx11_abi"] is None
    assert info["core_library_sha256"] is None
    assert info["core_abi_fingerprint"] is None


def test_environment_separates_build_and_runtime_information():
    environment = collect_core_environment()
    assert environment.core_version == CORE_VERSION
    assert environment.triton_version == "3.0.0"
    assert environment.runtime_python_version
    assert environment.runtime_platform
    assert environment.build_info_generated is False

    serialized = environment.to_dict()
    assert serialized["runtime_python_version"] == environment.runtime_python_version
    assert "expected_llvm_project_commit" in serialized
    assert "core_abi_fingerprint" in serialized


def test_abi_material_does_not_treat_expected_llvm_pin_as_actual():
    material = collect_core_environment().abi_material()
    assert "expected_llvm_project_commit" not in material
    assert material["actual_llvm_commit"] is None
    assert material["vendored_triton_commit"]
    assert material["core_library_sha256"] is None


def test_build_info_schema_1_0_remains_readable(tmp_path):
    legacy = json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8"))
    legacy["schema_version"] = "1.0"
    legacy["generated"] = True
    legacy["core_version"] = CORE_VERSION
    legacy["backend_protocol_version"] = BACKEND_PLUGIN_PROTOCOL_VERSION
    legacy["manifest_schema_version"] = BACKEND_MANIFEST_SCHEMA_VERSION
    legacy.pop("core_abi_fingerprint_schema")
    legacy.pop("core_library_sha256")
    legacy.pop("core_abi_fingerprint")
    path = tmp_path / "legacy-build-info.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = load_build_info(path)
    assert loaded["schema_version"] == "1.0"
    assert loaded["core_abi_fingerprint_schema"] is None
    assert loaded["core_library_sha256"] is None
    assert loaded["core_abi_fingerprint"] is None


def test_build_info_schema_1_1_requires_abi_fields(tmp_path):
    current = json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8"))
    current.pop("core_library_sha256")
    path = tmp_path / "incomplete-build-info.json"
    path.write_text(json.dumps(current), encoding="utf-8")

    with pytest.raises(RuntimeError, match="core_library_sha256"):
        load_build_info(path)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("19.0.0git", ("19.0.0", "git")),
        ("19.1", ("19.1", None)),
        ("vendor-build", (None, "vendor-build")),
        (None, (None, None)),
    ],
)
def test_toolchain_version_normalization_is_explicit(raw, expected):
    assert normalize_toolchain_version(raw) == expected


def test_build_info_contains_no_workspace_absolute_paths():
    raw = BUILD_INFO_PATH.read_text(encoding="utf-8")
    assert str(REPO_ROOT) not in raw


@pytest.mark.parametrize(
    "data, match",
    [
        ([], "root must be a JSON object"),
        ({"schema_version": "2.0"}, "missing required field"),
        (
            {
                **json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8")),
                "schema_version": "2.0",
            },
            "Unsupported",
        ),
        (
            {
                **json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8")),
                "generated": "false",
            },
            "must be a boolean",
        ),
        (
            {
                **json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8")),
                "actual_llvm_commit": "not-a-commit",
            },
            "invalid git commit",
        ),
    ],
)
def test_invalid_build_info_is_rejected(tmp_path, data, match):
    path = tmp_path / "build_info.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match=match):
        load_build_info(path)
