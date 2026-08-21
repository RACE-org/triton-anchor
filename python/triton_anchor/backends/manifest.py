"""Static, import-free backend plugin Manifest Schema 1.0."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence, Tuple

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from .._version import BACKEND_MANIFEST_SCHEMA_VERSION
from .errors import BackendPluginManifestError
from .protocol import PluginIsolationMode, PluginSource


MANIFEST_FILENAME = "triton_anchor_backend.json"

_ROOT_FIELDS = {"schema_version", "plugins"}
_PLUGIN_FIELDS = {
    "plugin_id",
    "display_name",
    "vendor",
    "entry_point",
    "backend_protocol",
    "requires_core",
    "requires_triton",
    "requires_llvm_version",
    "requires_llvm_commit",
    "requires_mlir_version",
    "requires_mlir_commit",
    "targets",
    "capabilities",
    "requires_capabilities",
    "isolation_mode",
    "native_libraries",
    "abi_fingerprint",
    "priority",
}
_SCHEMA_VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+")
_PLUGIN_ID_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?")
_ENTRY_POINT_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
_ABI_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_NON_EMPTY_STRING_PATTERN = re.compile(r"(?![\s\S]*[\r\n])\S(?:[\s\S]*\S)?")
_NATIVE_LIBRARY_PATH_PATTERN = re.compile(
    r"(?!/)(?![\s\S]*(?:^|/)\.\.?(?:/|$))(?![\s\S]*\\)"
    r"(?![\s\S]*[\r\n])\S(?:[\s\S]*\S)?"
)
_TRITON_REQUIREMENT_FIELDS = {"version", "commit"}
_STRUCTURAL_MANIFEST_FIELDS = frozenset(
    {
        "manifest",
        "isolation_mode",
        "native_libraries",
        "abi_fingerprint",
    }
)

_SEMANTIC_FIELD_ORDER = {
    "backend_protocol": 40,
    "requires_core": 50,
    "requires_triton.version": 60,
    "requires_llvm_version": 70,
    "requires_mlir_version": 80,
    "plugins[].plugin_id": 120,
    "plugins[].entry_point": 121,
}


def _manifest_actual(data: Mapping[str, Any], field_name: str) -> str:
    """Render one invalid value without confusing absence with JSON null."""
    if field_name not in data:
        return "<missing>"
    return stable_manifest_actual(data[field_name])


@dataclass(frozen=True)
class TritonRequirement:
    """Triton semantic-version range plus an optional exact vendored commit."""

    version: str
    commit: Optional[str] = None
    extensions: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )


@dataclass(frozen=True)
class BackendPluginManifest:
    """One backend entry-point declaration from a distribution manifest.

    Direct construction is intentionally available to evidence-only tooling.
    It does not authorize a manifest for loading; every operational API must
    call :func:`require_operational_manifest` first.
    """

    plugin_id: str
    entry_point: str
    backend_protocol: str
    requires_triton: TritonRequirement
    targets: Tuple[str, ...]
    isolation_mode: PluginIsolationMode
    display_name: Optional[str] = None
    vendor: Optional[str] = None
    requires_core: Optional[str] = None
    requires_llvm_version: Optional[str] = None
    requires_llvm_commit: Optional[str] = None
    requires_mlir_version: Optional[str] = None
    requires_mlir_commit: Optional[str] = None
    capabilities: Tuple[str, ...] = ()
    native_libraries: Tuple[str, ...] = ()
    abi_fingerprint: Optional[str] = None
    priority: int = 0
    distribution_name: Optional[str] = None
    distribution_version: Optional[str] = None
    extensions: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )
    requires_capabilities: Tuple[str, ...] = ()

    def with_distribution(
        self, name: Optional[str], version: Optional[str]
    ) -> "BackendPluginManifest":
        """Attach authoritative package identity from importlib.metadata."""
        return replace(
            self,
            distribution_name=name,
            distribution_version=version,
        )


def _stable_runtime_type_name(value: Any) -> str:
    """Return a qualified runtime type without invoking instance methods."""
    value_type = type(value)
    try:
        module = type.__getattribute__(value_type, "__module__")
        qualname = type.__getattribute__(value_type, "__qualname__")
    except BaseException:
        return "builtins.object"
    if type(module) is not str or type(qualname) is not str:
        return "builtins.object"
    return f"{module}.{qualname}" if module else qualname


def _json_manifest_value(
    value: Any,
    *,
    seen: Optional[set[int]] = None,
    depth: int = 0,
    budget: Optional[list[int]] = None,
) -> Tuple[bool, Any]:
    """Project bounded, acyclic builtin JSON values without user code."""
    if seen is None:
        seen = set()
    if budget is None:
        budget = [1024]
    if depth > 32 or budget[0] <= 0:
        return False, None
    budget[0] -= 1
    value_type = type(value)
    if value is None or value_type is bool:
        return True, value
    if value_type is int:
        return (True, value) if value.bit_length() <= 16384 else (False, None)
    if value_type is str:
        return (True, value) if len(value) <= 65536 else (False, None)
    if value_type is float:
        return (True, value) if math.isfinite(value) else (False, None)
    if value_type in {list, tuple}:
        identity = id(value)
        if identity in seen or len(value) > budget[0]:
            return False, None
        seen.add(identity)
        projected = []
        try:
            for item in value:
                valid, normalized = _json_manifest_value(
                    item,
                    seen=seen,
                    depth=depth + 1,
                    budget=budget,
                )
                if not valid:
                    return False, None
                projected.append(normalized)
            return True, projected
        finally:
            seen.remove(identity)
    if value_type is dict:
        identity = id(value)
        if identity in seen or len(value) > budget[0]:
            return False, None
        seen.add(identity)
        projected_mapping = {}
        try:
            for key, item in value.items():
                if type(key) is not str or len(key) > 65536:
                    return False, None
                valid, normalized = _json_manifest_value(
                    item,
                    seen=seen,
                    depth=depth + 1,
                    budget=budget,
                )
                if not valid:
                    return False, None
                projected_mapping[key] = normalized
            return True, projected_mapping
        finally:
            seen.remove(identity)
    return False, None


def stable_manifest_actual(value: Any) -> str:
    """Render manifest diagnostics deterministically and without ``repr``.

    Protocol enums and strings retain their user-facing value.  Other exact
    JSON-like builtins use canonical JSON.  Arbitrary runtime objects are
    represented solely by their qualified type, so an object's ``__repr__``
    cannot execute while the structural rejection gate is reporting it.
    """
    if type(value) is PluginIsolationMode:
        return value.value
    if type(value) is str:
        return value
    valid, normalized = _json_manifest_value(value)
    if valid:
        try:
            return json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            pass
    return f"<invalid type: {_stable_runtime_type_name(value)}>"


def _manifest_identity(value: Any) -> Optional[str]:
    """Keep malformed manually-constructed identity fields out of errors."""
    return value if type(value) is str and value else None


def unsupported_isolation_mode_error(
    actual: Any,
    *,
    plugin_id: Optional[str] = None,
    entry_point: Optional[str] = None,
) -> BackendPluginManifestError:
    """Build the Protocol 1.0 structural isolation rejection."""
    actual_text = stable_manifest_actual(actual)
    return BackendPluginManifestError(
        "Backend Plugin Protocol 1.0 only supports isolation_mode "
        "'python_only'",
        plugin_id=plugin_id,
        entry_point=entry_point,
        field="isolation_mode",
        expected=PluginIsolationMode.PYTHON_ONLY.value,
        actual=actual_text,
        remediation=(
            "Publish a genuinely python_only backend, or use a future "
            "protocol release that defines the required isolation contract."
        ),
    )


def operational_manifest_error(
    plugin: Any,
) -> Optional[BackendPluginManifestError]:
    """Return the first structural error that forbids operational use.

    ``native_in_process`` and ``subprocess`` remain representable for static
    evidence tests, but Protocol/Schema 1.0 never authorizes either mode.
    Requiring the exact enum member also prevents a manually constructed raw
    string from taking a different downstream identity-comparison branch.
    """
    if type(plugin) is not BackendPluginManifest:
        return BackendPluginManifestError(
            "Operational backend manifests must use BackendPluginManifest",
            field="manifest",
            expected="triton_anchor.backends.BackendPluginManifest",
            actual=stable_manifest_actual(plugin),
            remediation=(
                "Pass a manifest returned by parse_manifest(), or construct "
                "the exact BackendPluginManifest dataclass for static tests."
            ),
        )
    plugin_id = _manifest_identity(plugin.plugin_id)
    entry_point = _manifest_identity(plugin.entry_point)
    if plugin.isolation_mode is not PluginIsolationMode.PYTHON_ONLY:
        return unsupported_isolation_mode_error(
            plugin.isolation_mode,
            plugin_id=plugin_id,
            entry_point=entry_point,
        )
    native_libraries = plugin.native_libraries
    if type(native_libraries) is not tuple or len(native_libraries) != 0:
        return BackendPluginManifestError(
            "python_only plugins cannot declare native_libraries",
            plugin_id=plugin_id,
            entry_point=entry_point,
            field="native_libraries",
            expected="an omitted field for python_only",
            actual=stable_manifest_actual(native_libraries),
            remediation=(
                "Remove native_libraries and publish a genuinely pure-Python "
                "backend distribution."
            ),
        )
    if plugin.abi_fingerprint is not None:
        return BackendPluginManifestError(
            "python_only plugins cannot declare abi_fingerprint",
            plugin_id=plugin_id,
            entry_point=entry_point,
            field="abi_fingerprint",
            expected="an omitted field for python_only",
            actual=stable_manifest_actual(plugin.abi_fingerprint),
            remediation=(
                "Remove abi_fingerprint; Protocol 1.0 does not authorize "
                "in-process native backend code."
            ),
        )
    return None


def operational_record_manifest_error(
    record: Any,
) -> Optional[BackendPluginManifestError]:
    """Return the structural error that forbids a record's operational use.

    Parser-rejected records intentionally retain ``manifest=None`` while their
    structured error carries the parsed plugin identity.  Treat that shape as
    unsupported too; forged lifecycle state must not turn a rejected manifest
    into a Legacy record.
    """
    manifest = getattr(record, "manifest", None)
    if manifest is not None:
        error = operational_manifest_error(manifest)
        if error is not None:
            return error
    source = getattr(record, "source", None)
    if manifest is None and source is PluginSource.LEGACY:
        return None
    if manifest is None:
        errors = getattr(record, "errors", ())
        if type(errors) in {list, tuple}:
            for candidate in errors:
                if isinstance(candidate, BackendPluginManifestError):
                    return candidate
        error = getattr(record, "error", None)
        if isinstance(error, BackendPluginManifestError):
            return error
        plugin_id = getattr(record, "plugin_id", None)
        entry_point = getattr(record, "entry_point_name", None)
        return BackendPluginManifestError(
            "Manifest-governed backend record has no parsed Manifest",
            plugin_id=_manifest_identity(plugin_id),
            entry_point=_manifest_identity(entry_point),
            field="manifest",
            expected="a valid parsed BackendPluginManifest",
            actual="<missing>",
            remediation=(
                "Rediscover the backend from a valid Protocol/Schema 1.0 "
                "Manifest; only explicitly Legacy records may omit it."
            ),
        )
    error = getattr(record, "error", None)
    if (
        isinstance(error, BackendPluginManifestError)
        and error.field in _STRUCTURAL_MANIFEST_FIELDS
    ):
        return error
    return None


def require_operational_manifest(
    plugin: Any,
) -> BackendPluginManifest:
    """Return a Protocol 1.0 operational manifest or raise its first error."""
    error = operational_manifest_error(plugin)
    if error is not None:
        raise error
    return plugin


@dataclass(frozen=True)
class BackendManifestDocument:
    """All backend records declared by one installed distribution."""

    schema_version: str
    plugins: Tuple[BackendPluginManifest, ...]
    extensions: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    def get_by_entry_point(self, name: str) -> BackendPluginManifest:
        matches = tuple(
            plugin for plugin in self.plugins if plugin.entry_point == name
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise BackendPluginManifestError(
                f"Manifest declares backend entry point '{name}' more than once",
                entry_point=name,
                field="plugins[].entry_point",
                expected="one record matching the installed entry point",
                actual=str(len(matches)),
                remediation=(
                    "Give every Manifest plugin record a unique entry_point."
                ),
            )
        raise BackendPluginManifestError(
            f"Manifest does not declare backend entry point '{name}'",
            entry_point=name,
            field="entry_point",
            expected="one record matching the installed entry point",
            actual=name,
            remediation=(
                "Add a Manifest plugin record for this entry point or remove "
                "the stale entry-point declaration."
            ),
        )

    def with_distribution(
        self, name: Optional[str], version: Optional[str]
    ) -> "BackendManifestDocument":
        return replace(
            self,
            plugins=tuple(
                plugin.with_distribution(name, version) for plugin in self.plugins
            ),
        )


def _is_non_empty_string(value: Any) -> bool:
    """Mirror the Schema 1.x nonEmptyString acceptance predicate."""
    return isinstance(value, str) and _NON_EMPTY_STRING_PATTERN.fullmatch(value) is not None


def _validated_string(
    data: Mapping[str, Any],
    field_name: str,
    *,
    required: bool,
    plugin_id: Optional[str] = None,
) -> Optional[str]:
    if field_name not in data:
        if not required:
            return None
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' must be a non-empty string",
            plugin_id=plugin_id,
            field=field_name,
            expected="a non-empty string",
            actual=_manifest_actual(data, field_name),
            remediation=f"Set '{field_name}' to a non-empty string.",
        )
    value = data[field_name]
    if not _is_non_empty_string(value):
        presence = "" if required else " when present"
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' must be a non-empty string{presence} "
            "without surrounding whitespace or line breaks",
            plugin_id=plugin_id,
            field=field_name,
            expected=(
                "a non-empty string without surrounding whitespace, CR, or LF"
            ),
            actual=stable_manifest_actual(value),
            remediation=(
                f"Set '{field_name}' to a non-empty string without surrounding "
                "whitespace or line breaks."
            ),
        )
    return value


def _required_string(
    data: Mapping[str, Any],
    field_name: str,
    *,
    plugin_id: Optional[str] = None,
) -> str:
    value = _validated_string(
        data, field_name, required=True, plugin_id=plugin_id
    )
    assert value is not None
    return value


def _optional_string(
    data: Mapping[str, Any],
    field_name: str,
    *,
    plugin_id: Optional[str] = None,
) -> Optional[str]:
    return _validated_string(
        data, field_name, required=False, plugin_id=plugin_id
    )


def _optional_commit(
    data: Mapping[str, Any],
    field_name: str,
    *,
    plugin_id: Optional[str] = None,
) -> Optional[str]:
    value = _optional_string(data, field_name, plugin_id=plugin_id)
    if value is None:
        return None
    if not _COMMIT_PATTERN.fullmatch(value):
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' must be a 40-character git commit",
            plugin_id=plugin_id,
            field=field_name,
            expected="a 40-character hexadecimal git commit",
            actual=value,
            remediation=(
                f"Record the exact 40-character commit in '{field_name}' or "
                "omit this optional constraint."
            ),
        )
    return value.lower()


def _optional_abi_fingerprint(
    data: Mapping[str, Any],
    *,
    plugin_id: Optional[str] = None,
) -> Optional[str]:
    value = _optional_string(
        data, "abi_fingerprint", plugin_id=plugin_id
    )
    if value is None:
        return None
    if not _ABI_FINGERPRINT_PATTERN.fullmatch(value):
        raise BackendPluginManifestError(
            "Manifest field 'abi_fingerprint' must use "
            "'sha256:<64 lowercase hexadecimal characters>'",
            plugin_id=plugin_id,
            field="abi_fingerprint",
            expected="sha256 followed by exactly 64 lowercase hexadecimal characters",
            actual=value,
            remediation=(
                "Set 'abi_fingerprint' to the exact Core ABI fingerprint "
                "reported by the matching triton-anchor build, for example "
                "'sha256:<64 lowercase hexadecimal characters>'."
            ),
        )
    return value


def _required_version_string(
    data: Mapping[str, Any],
    field_name: str,
    *,
    plugin_id: Optional[str] = None,
) -> str:
    return _required_string(data, field_name, plugin_id=plugin_id)


def _optional_version_string(
    data: Mapping[str, Any],
    field_name: str,
    *,
    plugin_id: Optional[str] = None,
) -> Optional[str]:
    return _optional_string(data, field_name, plugin_id=plugin_id)


def _string_tuple(
    data: Mapping[str, Any],
    field_name: str,
    *,
    required: bool,
    plugin_id: Optional[str] = None,
) -> Tuple[str, ...]:
    if field_name not in data and not required:
        return ()
    value = data.get(field_name)
    if not isinstance(value, list) or (required and not value):
        requirement = "a non-empty array" if required else "an array"
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' must be {requirement} of strings",
            plugin_id=plugin_id,
            field=field_name,
            expected=f"{requirement} of strings",
            actual=_manifest_actual(data, field_name),
            remediation=(
                f"Set '{field_name}' to {requirement} containing unique, "
                "non-empty strings."
            ),
        )
    if any(not _is_non_empty_string(item) for item in value):
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' must contain non-empty strings "
            "without surrounding whitespace or line breaks",
            plugin_id=plugin_id,
            field=field_name,
            expected=(
                "an array containing only non-empty strings without surrounding "
                "whitespace, CR, or LF"
            ),
            actual=stable_manifest_actual(value),
            remediation=(
                f"Remove empty, non-string, surrounding-whitespace, CR, or LF "
                f"values from '{field_name}'."
            ),
        )
    normalized = tuple(value)
    if len(set(normalized)) != len(normalized):
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' contains duplicate values",
            plugin_id=plugin_id,
            field=field_name,
            expected="unique string values",
            actual=stable_manifest_actual(value),
            remediation=f"Remove duplicate values from '{field_name}'.",
        )
    return normalized


def _parse_triton_requirement(
    value: Any, plugin_id: str
) -> TritonRequirement:
    if not isinstance(value, dict):
        raise BackendPluginManifestError(
            "Manifest field 'requires_triton' must be an object",
            plugin_id=plugin_id,
            field="requires_triton",
            expected="an object containing a version specifier",
            actual=stable_manifest_actual(value),
            remediation=(
                "Set 'requires_triton' to an object such as "
                "{'version': '==<triton-version>'}."
            ),
        )
    version = _required_version_string(
        value, "version", plugin_id=plugin_id
    )
    commit = _optional_commit(value, "commit", plugin_id=plugin_id)
    return TritonRequirement(
        version=version,
        commit=commit,
        extensions={
            key: item
            for key, item in value.items()
            if key not in _TRITON_REQUIREMENT_FIELDS
        },
    )


def _parse_plugin(data: Any) -> BackendPluginManifest:
    if not isinstance(data, dict):
        raise BackendPluginManifestError(
            "Each manifest plugin must be an object",
            field="plugins[]",
            expected="a plugin object",
            actual=stable_manifest_actual(data),
            remediation="Replace each plugins array item with a plugin object.",
        )

    plugin_id = _required_string(data, "plugin_id")
    entry_point = _required_string(data, "entry_point", plugin_id=plugin_id)
    if not _PLUGIN_ID_PATTERN.fullmatch(plugin_id):
        raise BackendPluginManifestError(
            "Manifest field 'plugin_id' must use lowercase letters, digits, "
            "'.', '_' or '-'",
            plugin_id=plugin_id,
            field="plugin_id",
            expected="a stable lowercase identifier",
            actual=plugin_id,
            remediation=(
                "Use lowercase letters, digits, '.', '_' or '-' for plugin_id."
            ),
        )
    if not _ENTRY_POINT_PATTERN.fullmatch(entry_point):
        raise BackendPluginManifestError(
            "Manifest field 'entry_point' contains invalid characters",
            plugin_id=plugin_id,
            field="entry_point",
            expected="letters, digits, '.', '_' or '-'",
            actual=entry_point,
            remediation=(
                "Make the Manifest entry_point exactly match a valid "
                "triton.backends entry-point name."
            ),
        )
    backend_protocol = _required_version_string(
        data, "backend_protocol", plugin_id=plugin_id
    )
    isolation_value = _required_string(
        data, "isolation_mode", plugin_id=plugin_id
    )
    if isolation_value != PluginIsolationMode.PYTHON_ONLY.value:
        raise unsupported_isolation_mode_error(
            isolation_value,
            plugin_id=plugin_id,
            entry_point=entry_point,
        )
    isolation_mode = PluginIsolationMode.PYTHON_ONLY

    native_libraries = _string_tuple(
        data, "native_libraries", required=False, plugin_id=plugin_id
    )
    invalid_native_paths = [
        path
        for path in native_libraries
        if _NATIVE_LIBRARY_PATH_PATTERN.fullmatch(path) is None
    ]
    if invalid_native_paths:
        raise BackendPluginManifestError(
            "native_libraries must use relative package paths without '..': "
            + ", ".join(invalid_native_paths),
            plugin_id=plugin_id,
            field="native_libraries",
            expected="relative POSIX package paths without '..'",
            actual=", ".join(invalid_native_paths),
            remediation=(
                "Replace native_libraries entries with exact relative paths "
                "present in the wheel RECORD."
            ),
        )
    abi_fingerprint = _optional_abi_fingerprint(
        data, plugin_id=plugin_id
    )
    if (
        isolation_mode is PluginIsolationMode.PYTHON_ONLY
        and "native_libraries" in data
    ):
        raise BackendPluginManifestError(
            "python_only plugins cannot declare native_libraries",
            plugin_id=plugin_id,
            field="native_libraries",
            expected="an omitted field for python_only",
            actual=_manifest_actual(data, "native_libraries"),
            remediation=(
                "Remove native_libraries and publish a genuinely pure-Python "
                "backend distribution."
            ),
        )
    if (
        isolation_mode is PluginIsolationMode.PYTHON_ONLY
        and "abi_fingerprint" in data
    ):
        raise BackendPluginManifestError(
            "python_only plugins cannot declare abi_fingerprint",
            plugin_id=plugin_id,
            field="abi_fingerprint",
            expected="an omitted field for python_only",
            actual=_manifest_actual(data, "abi_fingerprint"),
            remediation=(
                "Remove abi_fingerprint; Python-only plugins do not share the "
                "Core C++ ABI."
            ),
        )

    priority_value = data.get("priority", 0)
    if isinstance(priority_value, bool) or not isinstance(priority_value, (int, float)):
        raise BackendPluginManifestError(
            "Manifest field 'priority' must be an integer",
            plugin_id=plugin_id,
            field="priority",
            expected="an integer",
            actual=stable_manifest_actual(priority_value),
            remediation="Set 'priority' to an integer.",
        )
    if isinstance(priority_value, float) and (
        not math.isfinite(priority_value) or not priority_value.is_integer()
    ):
        raise BackendPluginManifestError(
            "Manifest field 'priority' must be a finite integer",
            plugin_id=plugin_id,
            field="priority",
            expected="a finite JSON integer",
            actual=stable_manifest_actual(priority_value),
            remediation="Set 'priority' to a finite integer.",
        )
    priority = int(priority_value)

    return BackendPluginManifest(
        plugin_id=plugin_id,
        display_name=_optional_string(data, "display_name", plugin_id=plugin_id),
        vendor=_optional_string(data, "vendor", plugin_id=plugin_id),
        entry_point=entry_point,
        backend_protocol=backend_protocol,
        requires_core=_optional_version_string(
            data, "requires_core", plugin_id=plugin_id
        ),
        requires_triton=_parse_triton_requirement(
            data.get("requires_triton"), plugin_id
        ),
        requires_llvm_version=_optional_version_string(
            data, "requires_llvm_version", plugin_id=plugin_id
        ),
        requires_llvm_commit=_optional_commit(
            data, "requires_llvm_commit", plugin_id=plugin_id
        ),
        requires_mlir_version=_optional_version_string(
            data, "requires_mlir_version", plugin_id=plugin_id
        ),
        requires_mlir_commit=_optional_commit(
            data, "requires_mlir_commit", plugin_id=plugin_id
        ),
        targets=_string_tuple(
            data, "targets", required=True, plugin_id=plugin_id
        ),
        capabilities=_string_tuple(
            data, "capabilities", required=False, plugin_id=plugin_id
        ),
        requires_capabilities=_string_tuple(
            data, "requires_capabilities", required=False, plugin_id=plugin_id
        ),
        isolation_mode=isolation_mode,
        native_libraries=native_libraries,
        abi_fingerprint=abi_fingerprint,
        priority=priority,
        extensions={
            key: value for key, value in data.items() if key not in _PLUGIN_FIELDS
        },
    )


def parse_manifest(data: Any) -> BackendManifestDocument:
    """Parse and validate an in-memory Manifest Schema 1.x document."""
    if not isinstance(data, dict):
        raise BackendPluginManifestError(
            "Manifest root must be a JSON object",
            field="<root>",
            expected="a JSON object",
            actual=type(data).__name__,
            remediation="Replace the Manifest root with a JSON object.",
        )

    schema_version = _required_string(data, "schema_version")
    if not _SCHEMA_VERSION_PATTERN.fullmatch(schema_version):
        raise BackendPluginManifestError(
            "Manifest schema_version must use '<major>.<minor>' numeric form",
            field="schema_version",
            expected="'<major>.<minor>' numeric form",
            actual=schema_version,
            remediation=(
                "Set schema_version to a supported numeric value such as '1.0'."
            ),
        )
    supported_major = BACKEND_MANIFEST_SCHEMA_VERSION.split(".", 1)[0]
    actual_major = schema_version.split(".", 1)[0]
    if actual_major != supported_major:
        raise BackendPluginManifestError(
            f"Unsupported manifest schema '{schema_version}'; "
            f"supported major is {supported_major}",
            field="schema_version",
            expected=f"major version {supported_major}",
            actual=schema_version,
            remediation=(
                "Install a plugin using a supported Manifest major version, "
                "or upgrade triton-anchor."
            ),
        )

    raw_plugins = data.get("plugins")
    if not isinstance(raw_plugins, list) or not raw_plugins:
        raise BackendPluginManifestError(
            "Manifest field 'plugins' must be a non-empty array",
            field="plugins",
            expected="a non-empty array of plugin records",
            actual=_manifest_actual(data, "plugins"),
            remediation="Declare at least one backend plugin record.",
        )
    plugins = tuple(_parse_plugin(plugin) for plugin in raw_plugins)

    return BackendManifestDocument(
        schema_version=schema_version,
        plugins=plugins,
        extensions={
            key: value for key, value in data.items() if key not in _ROOT_FIELDS
        },
    )


def _invalid_version_specifier_error(
    plugin: BackendPluginManifest,
    field_name: str,
    value: str,
) -> Optional[BackendPluginManifestError]:
    try:
        SpecifierSet(value)
    except InvalidSpecifier:
        return BackendPluginManifestError(
            f"Manifest field '{field_name}' has invalid version specifier "
            f"'{value}'",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field=field_name,
            expected="a valid PEP 440 version specifier",
            actual=value,
            remediation=(
                f"Replace '{field_name}' with a valid PEP 440 specifier."
            ),
        )
    return None


def _manifest_semantic_sort_key(
    error: BackendPluginManifestError,
) -> Tuple[int, str, str, str]:
    field_name = error.field or ""
    return (
        _SEMANTIC_FIELD_ORDER.get(field_name, 999),
        error.plugin_id or "",
        error.entry_point or "",
        error.code,
    )


def evaluate_manifest_semantics(
    document: BackendManifestDocument,
    *,
    installed_entry_points: Optional[Sequence[str]] = None,
) -> Tuple[BackendPluginManifestError, ...]:
    """Return deterministic, import-free semantic Manifest errors.

    JSON Schema and :func:`parse_manifest` intentionally share only the
    structural domain.  PEP 440 and cross-record/distribution constraints live
    here so operational loaders can reject them before importing plugin code.
    """
    errors = []
    for plugin in document.plugins:
        version_fields = (
            ("backend_protocol", plugin.backend_protocol),
            ("requires_core", plugin.requires_core),
            ("requires_triton.version", plugin.requires_triton.version),
            ("requires_llvm_version", plugin.requires_llvm_version),
            ("requires_mlir_version", plugin.requires_mlir_version),
        )
        for field_name, value in version_fields:
            if value is None:
                continue
            error = _invalid_version_specifier_error(plugin, field_name, value)
            if error is not None:
                errors.append(error)

    for attribute, field_name, remediation in (
        (
            "plugin_id",
            "plugins[].plugin_id",
            "Give every plugin record a unique stable plugin_id.",
        ),
        (
            "entry_point",
            "plugins[].entry_point",
            "Declare every triton.backends entry point exactly once.",
        ),
    ):
        groups = {}
        for plugin in document.plugins:
            groups.setdefault(getattr(plugin, attribute), []).append(plugin)
        for value in sorted(groups):
            matches = groups[value]
            if len(matches) < 2:
                continue
            for plugin in matches:
                errors.append(
                    BackendPluginManifestError(
                        f"Manifest contains duplicate {attribute} value '{value}'",
                        plugin_id=plugin.plugin_id,
                        entry_point=plugin.entry_point,
                        field=field_name,
                        expected=f"unique {attribute} values within a distribution",
                        actual=value,
                        remediation=remediation,
                    )
                )

    if installed_entry_points is not None:
        installed = tuple(sorted(installed_entry_points))
        declared = tuple(sorted(plugin.entry_point for plugin in document.plugins))
        installed_counts = Counter(installed)
        declared_counts = Counter(declared)
        if installed_counts != declared_counts:
            missing = sorted((installed_counts - declared_counts).elements())
            unexpected = sorted((declared_counts - installed_counts).elements())
            details = []
            if missing:
                details.append("missing records for " + ", ".join(missing))
            if unexpected:
                details.append("unknown records for " + ", ".join(unexpected))
            errors.append(
                BackendPluginManifestError(
                    "Manifest entry-point coverage mismatch: " + "; ".join(details),
                    field="plugins[].entry_point",
                    expected=", ".join(installed),
                    actual=", ".join(declared),
                    remediation=(
                        "Make Manifest records exactly cover all triton.backends "
                        "entry points in this distribution."
                    ),
                )
            )

    return tuple(sorted(errors, key=_manifest_semantic_sort_key))


def validate_manifest_semantics(
    document: BackendManifestDocument,
    *,
    installed_entry_points: Optional[Sequence[str]] = None,
) -> BackendManifestDocument:
    """Raise the first semantic error while preserving the pure evaluator."""
    errors = evaluate_manifest_semantics(
        document, installed_entry_points=installed_entry_points
    )
    if errors:
        raise errors[0]
    return document


def load_manifest(path: Path) -> BackendManifestDocument:
    """Load a static manifest file without importing plugin code."""
    manifest_path = Path(path)
    try:
        data = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BackendPluginManifestError(
            f"Unable to read backend manifest '{manifest_path}': {exc}",
            field="manifest_file",
            expected="readable UTF-8 JSON without duplicate fields",
            actual=f"<error: {exc}>",
            remediation=(
                "Fix the JSON file, package it once in the wheel, and reinstall "
                "the backend."
            ),
        ) from exc
    return parse_manifest(data)


def _object_without_duplicate_keys(
    pairs: Sequence[Tuple[str, Any]],
) -> Mapping[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise BackendPluginManifestError(
                f"Manifest contains duplicate JSON field '{key}'",
                field=key,
                expected="one JSON member with this name",
                actual="multiple members",
                remediation=f"Remove the duplicate JSON field '{key}'.",
            )
        result[key] = value
    return result


def _distribution_identity(distribution: Any) -> Tuple[Optional[str], Optional[str]]:
    metadata = getattr(distribution, "metadata", None)
    name = None
    if metadata is not None:
        try:
            name = metadata.get("Name")
        except AttributeError:
            name = None
    if name is None:
        name = getattr(distribution, "name", None)
    version = getattr(distribution, "version", None)
    return name, version


def _backend_entry_point_names(distribution: Any) -> Tuple[str, ...]:
    entry_points = getattr(distribution, "entry_points", ()) or ()
    return tuple(
        entry_point.name
        for entry_point in entry_points
        if getattr(entry_point, "group", None) == "triton.backends"
    )


def load_distribution_manifest(
    distribution: Any,
) -> Optional[BackendManifestDocument]:
    """Load one distribution's unique static manifest, if present.

    A distribution with no manifest is Legacy.  Once a manifest exists it must
    cover every ``triton.backends`` entry point in that distribution exactly;
    partial declarations are rejected and cannot fall back to Legacy.
    """
    files_value = getattr(distribution, "files", None)
    if files_value is None:
        raise BackendPluginManifestError(
            "Cannot verify Legacy status because the distribution file list "
            "is unavailable",
            field="distribution.files",
            expected="an installed wheel RECORD file list",
            actual="<unavailable>",
            remediation=(
                "Install the backend from a standards-compliant wheel with a "
                "complete RECORD; Legacy status cannot be guessed."
            ),
        )
    files: Sequence[Any] = files_value
    candidates = [
        file
        for file in files
        if PurePosixPath(str(file)).name == MANIFEST_FILENAME
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise BackendPluginManifestError(
            f"Distribution contains multiple '{MANIFEST_FILENAME}' files",
            field=MANIFEST_FILENAME,
            expected="exactly one Manifest file per distribution",
            actual=str(len(candidates)),
            remediation=(
                "Package exactly one backend Manifest in the distribution."
            ),
        )

    try:
        manifest_path = distribution.locate_file(candidates[0])
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        error_type = _stable_runtime_type_name(exc)
        raise BackendPluginManifestError(
            "Unable to locate the distribution Manifest recorded by "
            "installed metadata",
            field=MANIFEST_FILENAME,
            expected="an installed readable Manifest path",
            actual=f"<error: {error_type}>",
            remediation="Reinstall the backend wheel with a valid RECORD.",
        ) from exc
    document = load_manifest(Path(manifest_path))
    validate_manifest_semantics(
        document,
        installed_entry_points=_backend_entry_point_names(distribution),
    )

    name, version = _distribution_identity(distribution)
    return document.with_distribution(name, version)


def load_manifest_for_entry_point(
    entry_point: Any,
    distribution: Any = None,
) -> Optional[BackendPluginManifest]:
    """Return the record for an entry point, or ``None`` for Legacy."""
    if distribution is None:
        distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        raise BackendPluginManifestError(
            "Cannot determine the entry point's owning distribution; "
            "Legacy status is unverified",
            entry_point=getattr(entry_point, "name", None),
            field="entry_point.distribution",
            expected="authoritative owning distribution metadata",
            actual="<unavailable>",
            remediation=(
                "Discover via importlib.metadata.distributions() and pass the "
                "owning distribution explicitly."
            ),
        )
    document = load_distribution_manifest(distribution)
    if document is None:
        return None
    return document.get_by_entry_point(entry_point.name)
