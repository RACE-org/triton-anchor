"""Core, toolchain, and runtime environment description for plugin checks."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sysconfig
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .._version import (
    BACKEND_MANIFEST_SCHEMA_VERSION,
    BACKEND_PLUGIN_PROTOCOL_VERSION,
    CORE_VERSION,
)


_BUILD_INFO_PATH = Path(__file__).resolve().parents[1] / "_build_info.json"
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SUPPORTED_BUILD_INFO_SCHEMAS = {"1.0", "1.1"}
CORE_ABI_FINGERPRINT_SCHEMA = "triton-anchor-core-abi-v1"


@dataclass(frozen=True)
class CoreEnvironment:
    """Serializable environment used by pre-load compatibility checks.

    Unknown build-time values remain ``None``.  They must never be guessed:
    an exact check requiring an unknown value must fail closed later.
    """

    core_version: str
    build_info_generated: bool
    backend_protocol_version: str
    manifest_schema_version: str
    triton_version: str
    vendored_triton_commit: str
    expected_llvm_project_commit: str
    actual_llvm_version_raw: Optional[str]
    actual_llvm_version: Optional[str]
    actual_llvm_version_suffix: Optional[str]
    actual_llvm_commit: Optional[str]
    actual_mlir_version_raw: Optional[str]
    actual_mlir_version: Optional[str]
    actual_mlir_version_suffix: Optional[str]
    actual_mlir_commit: Optional[str]
    cxx_standard: str
    cxx_compiler_id: Optional[str]
    cxx_compiler_version: Optional[str]
    cxx11_abi: Optional[str]
    build_type: Optional[str]
    ttgpu: Optional[bool]
    built_python_version: Optional[str]
    built_python_soabi: Optional[str]
    built_platform: Optional[str]
    core_abi_fingerprint_schema: Optional[str]
    core_library_sha256: Optional[str]
    core_abi_fingerprint: Optional[str]
    runtime_python_version: str
    runtime_python_implementation: str
    runtime_python_soabi: Optional[str]
    runtime_platform: str
    runtime_system: str
    runtime_machine: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def abi_material(self) -> Dict[str, Any]:
        """Return the exact inputs covered by ``core_abi_fingerprint``."""
        return {
            "core_version": self.core_version,
            # The vendored Triton commit is an upstream base marker, not a
            # complete fingerprint of triton-anchor's local C++ changes.
            "vendored_triton_commit": self.vendored_triton_commit,
            "actual_llvm_version_raw": self.actual_llvm_version_raw,
            "actual_llvm_commit": self.actual_llvm_commit,
            "actual_mlir_version_raw": self.actual_mlir_version_raw,
            "actual_mlir_commit": self.actual_mlir_commit,
            "cxx_standard": self.cxx_standard,
            "cxx_compiler_id": self.cxx_compiler_id,
            "cxx_compiler_version": self.cxx_compiler_version,
            "cxx11_abi": self.cxx11_abi,
            "built_python_soabi": self.built_python_soabi,
            "built_platform": self.built_platform,
            "ttgpu": self.ttgpu,
            # Hashing the Core library makes local C++ changes part of the
            # identity without treating any plugin binary as Core input.
            "core_library_sha256": self.core_library_sha256,
        }


def _complete_abi_material(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
    material = {key: data.get(key) for key in keys}
    if any(value is None or value == "" for value in material.values()):
        return None
    return material


def _compute_core_abi_fingerprint(data: Dict[str, Any]) -> Optional[str]:
    material = _complete_abi_material(data)
    if material is None:
        return None
    payload = {
        "schema": CORE_ABI_FINGERPRINT_SCHEMA,
        "material": material,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def normalize_toolchain_version(
    raw: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Split values such as ``19.0.0git`` into PEP-440 input and suffix."""
    if raw is None:
        return None, None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+){1,2})(.*)", raw)
    if match is None:
        return None, raw
    version = match.group(1)
    suffix = match.group(2) or None
    return version, suffix


def load_build_info(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load build metadata without importing Triton or any backend plugin."""
    build_info_path = Path(path) if path is not None else _BUILD_INFO_PATH
    try:
        data = json.loads(build_info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to read triton-anchor build info: {build_info_path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("triton-anchor build info root must be a JSON object")

    required = {
        "schema_version",
        "generated",
        "core_version",
        "backend_protocol_version",
        "manifest_schema_version",
        "triton_version",
        "vendored_triton_commit",
        "expected_llvm_project_commit",
        "actual_llvm_version_raw",
        "actual_llvm_commit",
        "actual_mlir_version_raw",
        "actual_mlir_commit",
        "cxx_standard",
        "cxx_compiler_id",
        "cxx_compiler_version",
        "cxx11_abi",
        "build_type",
        "ttgpu",
        "built_python_version",
        "built_python_soabi",
        "built_platform",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise RuntimeError(
            "triton-anchor build info is missing required field(s): "
            + ", ".join(missing)
        )

    schema_version = data["schema_version"]
    if schema_version not in _SUPPORTED_BUILD_INFO_SCHEMAS:
        raise RuntimeError(
            "Unsupported triton-anchor build info schema: "
            f"{schema_version}"
        )
    abi_fields = {
        "core_abi_fingerprint_schema",
        "core_library_sha256",
        "core_abi_fingerprint",
    }
    if schema_version == "1.1":
        missing_abi_fields = sorted(abi_fields.difference(data))
        if missing_abi_fields:
            raise RuntimeError(
                "triton-anchor build info is missing required field(s): "
                + ", ".join(missing_abi_fields)
            )
    else:
        # Schema 1.0 predates the Core binary binding.  Preserve the unknown
        # state rather than fabricating a digest or a fingerprint.
        for field in abi_fields:
            data.setdefault(field, None)

    if not isinstance(data["generated"], bool):
        raise RuntimeError("triton-anchor build info 'generated' must be a boolean")

    expected_versions = {
        "core_version": CORE_VERSION,
        "backend_protocol_version": BACKEND_PLUGIN_PROTOCOL_VERSION,
        "manifest_schema_version": BACKEND_MANIFEST_SCHEMA_VERSION,
    }
    for field, expected in expected_versions.items():
        actual = data[field]
        if data["generated"]:
            if actual != expected:
                raise RuntimeError(
                    f"triton-anchor build info {field} mismatch: "
                    f"expected {expected}, got {actual}"
                )
        elif actual not in (None, expected):
            raise RuntimeError(
                f"triton-anchor source build info {field} mismatch: "
                f"expected null or {expected}, got {actual}"
            )
        # A source checkout uses a null template so public versions remain
        # authoritative in _version.py.  Built wheels contain a fixed snapshot.
        data[field] = expected

    optional_strings = {
        "actual_llvm_version_raw",
        "actual_llvm_commit",
        "actual_mlir_version_raw",
        "actual_mlir_commit",
        "cxx_compiler_id",
        "cxx_compiler_version",
        "cxx11_abi",
        "build_type",
        "built_python_version",
        "built_python_soabi",
        "built_platform",
    }
    required_strings = required.difference(
        optional_strings | {"generated", "ttgpu"}
    )
    invalid_strings = sorted(
        field
        for field in required_strings
        if not isinstance(data[field], str) or not data[field].strip()
    )
    if invalid_strings:
        raise RuntimeError(
            "triton-anchor build info has invalid string field(s): "
            + ", ".join(invalid_strings)
        )

    invalid_optional_strings = sorted(
        field
        for field in optional_strings
        if data[field] is not None
        and (not isinstance(data[field], str) or not data[field].strip())
    )
    if invalid_optional_strings:
        raise RuntimeError(
            "triton-anchor build info has invalid optional string field(s): "
            + ", ".join(invalid_optional_strings)
        )
    if data["ttgpu"] is not None and not isinstance(data["ttgpu"], bool):
        raise RuntimeError("triton-anchor build info 'ttgpu' must be null or boolean")
    if data["cxx11_abi"] not in (None, "0", "1"):
        raise RuntimeError(
            "triton-anchor build info 'cxx11_abi' must be null, '0', or '1'"
        )

    commit_fields = {
        "vendored_triton_commit",
        "expected_llvm_project_commit",
        "actual_llvm_commit",
        "actual_mlir_commit",
    }
    invalid_commits = sorted(
        field
        for field in commit_fields
        if data[field] is not None
        and not _GIT_COMMIT_PATTERN.fullmatch(data[field])
    )
    if invalid_commits:
        raise RuntimeError(
            "triton-anchor build info has invalid git commit field(s): "
            + ", ".join(invalid_commits)
        )

    for field in ("vendored_triton_commit", "expected_llvm_project_commit"):
        if not _GIT_COMMIT_PATTERN.fullmatch(data[field]):
            raise RuntimeError(
                f"triton-anchor build info '{field}' must be a 40-character "
                "git commit"
            )

    fingerprint_schema = data["core_abi_fingerprint_schema"]
    if fingerprint_schema is not None and (
        not isinstance(fingerprint_schema, str)
        or fingerprint_schema != CORE_ABI_FINGERPRINT_SCHEMA
    ):
        raise RuntimeError(
            "Unsupported triton-anchor Core ABI fingerprint schema: "
            f"{fingerprint_schema}"
        )
    if schema_version == "1.1" and fingerprint_schema is None:
        raise RuntimeError(
            "triton-anchor build info 'core_abi_fingerprint_schema' "
            "must be a non-empty string"
        )

    for field in ("core_library_sha256", "core_abi_fingerprint"):
        value = data[field]
        if value is not None and (
            not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value)
        ):
            raise RuntimeError(
                f"triton-anchor build info '{field}' must be null or "
                "'sha256:' followed by 64 lowercase hex characters"
            )

    material = _complete_abi_material(data)
    fingerprint = data["core_abi_fingerprint"]
    if fingerprint is not None:
        if material is None:
            raise RuntimeError(
                "triton-anchor Core ABI fingerprint has incomplete material"
            )
        expected_fingerprint = _compute_core_abi_fingerprint(data)
        if fingerprint != expected_fingerprint:
            raise RuntimeError(
                "triton-anchor Core ABI fingerprint does not match build info"
            )
    elif data["generated"] and material is not None:
        raise RuntimeError(
            "triton-anchor generated build info is missing its Core ABI "
            "fingerprint"
        )

    if not data["generated"] and (
        data["core_library_sha256"] is not None or fingerprint is not None
    ):
        raise RuntimeError(
            "triton-anchor source build info must not claim a Core library "
            "digest or ABI fingerprint"
        )
    return data


def collect_core_environment(path: Optional[Path] = None) -> CoreEnvironment:
    """Collect build-time and current Python runtime information."""
    info = load_build_info(path)
    llvm_version, llvm_suffix = normalize_toolchain_version(
        info.get("actual_llvm_version_raw")
    )
    mlir_version, mlir_suffix = normalize_toolchain_version(
        info.get("actual_mlir_version_raw")
    )
    return CoreEnvironment(
        core_version=info["core_version"],
        build_info_generated=info["generated"],
        backend_protocol_version=info["backend_protocol_version"],
        manifest_schema_version=info["manifest_schema_version"],
        triton_version=info["triton_version"],
        vendored_triton_commit=info["vendored_triton_commit"],
        expected_llvm_project_commit=info["expected_llvm_project_commit"],
        actual_llvm_version_raw=info.get("actual_llvm_version_raw"),
        actual_llvm_version=llvm_version,
        actual_llvm_version_suffix=llvm_suffix,
        actual_llvm_commit=info.get("actual_llvm_commit"),
        actual_mlir_version_raw=info.get("actual_mlir_version_raw"),
        actual_mlir_version=mlir_version,
        actual_mlir_version_suffix=mlir_suffix,
        actual_mlir_commit=info.get("actual_mlir_commit"),
        cxx_standard=info["cxx_standard"],
        cxx_compiler_id=info.get("cxx_compiler_id"),
        cxx_compiler_version=info.get("cxx_compiler_version"),
        cxx11_abi=info.get("cxx11_abi"),
        build_type=info.get("build_type"),
        ttgpu=info.get("ttgpu"),
        built_python_version=info.get("built_python_version"),
        built_python_soabi=info.get("built_python_soabi"),
        built_platform=info.get("built_platform"),
        core_abi_fingerprint_schema=info.get("core_abi_fingerprint_schema"),
        core_library_sha256=info.get("core_library_sha256"),
        core_abi_fingerprint=info.get("core_abi_fingerprint"),
        runtime_python_version=platform.python_version(),
        runtime_python_implementation=platform.python_implementation(),
        runtime_python_soabi=sysconfig.get_config_var("SOABI"),
        runtime_platform=sysconfig.get_platform(),
        runtime_system=platform.system(),
        runtime_machine=platform.machine(),
    )
