"""Static, import-free backend plugin Manifest Schema 1.0."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence, Tuple

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from .._version import BACKEND_MANIFEST_SCHEMA_VERSION
from .errors import BackendPluginManifestError
from .protocol import PluginIsolationMode


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
_SCHEMA_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+$")
_PLUGIN_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")
_ENTRY_POINT_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_ABI_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_TRITON_REQUIREMENT_FIELDS = {"version", "commit"}


def _manifest_actual(data: Mapping[str, Any], field_name: str) -> str:
    """Render one invalid value without confusing absence with JSON null."""
    if field_name not in data:
        return "<missing>"
    return repr(data[field_name])


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
    """One backend entry-point declaration from a distribution manifest."""

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


@dataclass(frozen=True)
class BackendManifestDocument:
    """All backend records declared by one installed distribution."""

    schema_version: str
    plugins: Tuple[BackendPluginManifest, ...]
    extensions: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    def get_by_entry_point(self, name: str) -> BackendPluginManifest:
        for plugin in self.plugins:
            if plugin.entry_point == name:
                return plugin
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


def _required_string(
    data: Mapping[str, Any],
    field_name: str,
    *,
    plugin_id: Optional[str] = None,
) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' must be a non-empty string",
            plugin_id=plugin_id,
            field=field_name,
            expected="a non-empty string",
            actual=_manifest_actual(data, field_name),
            remediation=f"Set '{field_name}' to a non-empty string.",
        )
    if value != value.strip():
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' cannot have surrounding whitespace",
            plugin_id=plugin_id,
            field=field_name,
            expected="a string without surrounding whitespace",
            actual=repr(value),
            remediation=f"Remove surrounding whitespace from '{field_name}'.",
        )
    return value


def _optional_string(
    data: Mapping[str, Any],
    field_name: str,
    *,
    plugin_id: Optional[str] = None,
) -> Optional[str]:
    if field_name not in data:
        return None
    value = data[field_name]
    if not isinstance(value, str) or not value.strip():
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' must be a non-empty string when present",
            plugin_id=plugin_id,
            field=field_name,
            expected="a non-empty string or an omitted field",
            actual=repr(value),
            remediation=(
                f"Set '{field_name}' to a non-empty string or omit the field."
            ),
        )
    if value != value.strip():
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' cannot have surrounding whitespace",
            plugin_id=plugin_id,
            field=field_name,
            expected="a string without surrounding whitespace",
            actual=repr(value),
            remediation=f"Remove surrounding whitespace from '{field_name}'.",
        )
    return value


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


def _required_version_specifier(
    data: Mapping[str, Any],
    field_name: str,
    *,
    plugin_id: Optional[str] = None,
) -> str:
    value = _required_string(data, field_name, plugin_id=plugin_id)
    try:
        SpecifierSet(value)
    except InvalidSpecifier as exc:
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' has invalid version specifier "
            f"'{value}'",
            plugin_id=plugin_id,
            field=field_name,
            expected="a valid PEP 440 version specifier",
            actual=value,
            remediation=(
                f"Replace '{field_name}' with a valid PEP 440 specifier."
            ),
        ) from exc
    return value


def _optional_version_specifier(
    data: Mapping[str, Any],
    field_name: str,
    *,
    plugin_id: Optional[str] = None,
) -> Optional[str]:
    value = _optional_string(data, field_name, plugin_id=plugin_id)
    if value is None:
        return None
    try:
        SpecifierSet(value)
    except InvalidSpecifier as exc:
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' has invalid version specifier "
            f"'{value}'",
            plugin_id=plugin_id,
            field=field_name,
            expected="a valid PEP 440 version specifier",
            actual=value,
            remediation=(
                f"Replace '{field_name}' with a valid PEP 440 specifier or "
                "omit this optional constraint."
            ),
        ) from exc
    return value


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
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' must contain non-empty strings",
            plugin_id=plugin_id,
            field=field_name,
            expected="an array containing only non-empty strings",
            actual=repr(value),
            remediation=(
                f"Remove empty or non-string values from '{field_name}'."
            ),
        )
    if any(item != item.strip() for item in value):
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' cannot contain surrounding whitespace",
            plugin_id=plugin_id,
            field=field_name,
            expected="strings without surrounding whitespace",
            actual=repr(value),
            remediation=(
                f"Remove surrounding whitespace from values in '{field_name}'."
            ),
        )
    normalized = tuple(value)
    if len(set(normalized)) != len(normalized):
        raise BackendPluginManifestError(
            f"Manifest field '{field_name}' contains duplicate values",
            plugin_id=plugin_id,
            field=field_name,
            expected="unique string values",
            actual=repr(value),
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
            actual=repr(value),
            remediation=(
                "Set 'requires_triton' to an object such as "
                "{'version': '==<triton-version>'}."
            ),
        )
    version = _required_version_specifier(
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
            actual=repr(data),
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
    backend_protocol = _required_version_specifier(
        data, "backend_protocol", plugin_id=plugin_id
    )
    isolation_value = _required_string(
        data, "isolation_mode", plugin_id=plugin_id
    )
    try:
        isolation_mode = PluginIsolationMode(isolation_value)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in PluginIsolationMode)
        raise BackendPluginManifestError(
            f"Unknown isolation_mode '{isolation_value}'; expected one of: {allowed}",
            plugin_id=plugin_id,
            field="isolation_mode",
            expected=allowed,
            actual=isolation_value,
            remediation="Choose one of the supported isolation_mode values.",
        ) from exc

    native_libraries = _string_tuple(
        data, "native_libraries", required=False, plugin_id=plugin_id
    )
    invalid_native_paths = [
        path
        for path in native_libraries
        if (
            "\\" in path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or not PurePosixPath(path).parts
            or path == "."
        )
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
        isolation_mode is PluginIsolationMode.NATIVE_IN_PROCESS
        and not native_libraries
    ):
        raise BackendPluginManifestError(
            "native_in_process plugins must declare at least one "
            "'native_libraries' path",
            plugin_id=plugin_id,
            field="native_libraries",
            expected="a non-empty array of installed native library paths",
            actual=_manifest_actual(data, "native_libraries"),
            remediation=(
                "List every in-process native library using its exact relative "
                "path from the wheel RECORD, or choose a non-native isolation "
                "mode."
            ),
        )
    if (
        isolation_mode is PluginIsolationMode.NATIVE_IN_PROCESS
        and abi_fingerprint is None
    ):
        raise BackendPluginManifestError(
            "native_in_process plugins must declare 'abi_fingerprint'",
            plugin_id=plugin_id,
            field="abi_fingerprint",
            expected="the exact Core ABI fingerprint",
            actual="<missing>",
            remediation=(
                "Declare the exact Core ABI fingerprint, or use a non-native "
                "isolation mode."
            ),
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
                "Remove native_libraries or choose the appropriate native "
                "isolation mode."
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
    if (
        isolation_mode is PluginIsolationMode.SUBPROCESS
        and "abi_fingerprint" in data
    ):
        raise BackendPluginManifestError(
            "subprocess plugins cannot declare abi_fingerprint",
            plugin_id=plugin_id,
            field="abi_fingerprint",
            expected="an omitted field for subprocess isolation",
            actual=_manifest_actual(data, "abi_fingerprint"),
            remediation=(
                "Remove abi_fingerprint; subprocess compatibility must be "
                "governed by a versioned process/IR contract instead of the "
                "Core in-process C++ ABI."
            ),
        )

    priority = data.get("priority", 0)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise BackendPluginManifestError(
            "Manifest field 'priority' must be an integer",
            plugin_id=plugin_id,
            field="priority",
            expected="an integer",
            actual=repr(priority),
            remediation="Set 'priority' to an integer.",
        )

    return BackendPluginManifest(
        plugin_id=plugin_id,
        display_name=_optional_string(data, "display_name", plugin_id=plugin_id),
        vendor=_optional_string(data, "vendor", plugin_id=plugin_id),
        entry_point=entry_point,
        backend_protocol=backend_protocol,
        requires_core=_optional_version_specifier(
            data, "requires_core", plugin_id=plugin_id
        ),
        requires_triton=_parse_triton_requirement(
            data.get("requires_triton"), plugin_id
        ),
        requires_llvm_version=_optional_version_specifier(
            data, "requires_llvm_version", plugin_id=plugin_id
        ),
        requires_llvm_commit=_optional_commit(
            data, "requires_llvm_commit", plugin_id=plugin_id
        ),
        requires_mlir_version=_optional_version_specifier(
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

    plugin_ids = [plugin.plugin_id for plugin in plugins]
    entry_points = [plugin.entry_point for plugin in plugins]
    if len(set(plugin_ids)) != len(plugin_ids):
        raise BackendPluginManifestError(
            "Manifest contains duplicate plugin_id values",
            field="plugins[].plugin_id",
            expected="unique plugin_id values within a distribution",
            actual=", ".join(plugin_ids),
            remediation="Give each plugin record a unique stable plugin_id.",
        )
    if len(set(entry_points)) != len(entry_points):
        raise BackendPluginManifestError(
            "Manifest contains duplicate entry_point values",
            field="plugins[].entry_point",
            expected="unique entry_point values within a distribution",
            actual=", ".join(entry_points),
            remediation=(
                "Declare each triton.backends entry point exactly once."
            ),
        )

    return BackendManifestDocument(
        schema_version=schema_version,
        plugins=plugins,
        extensions={
            key: value for key, value in data.items() if key not in _ROOT_FIELDS
        },
    )


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
    except Exception as exc:
        raise BackendPluginManifestError(
            f"Unable to locate distribution manifest '{candidates[0]}': {exc}",
            field=MANIFEST_FILENAME,
            expected="an installed readable Manifest path",
            actual=f"<error: {exc}>",
            remediation="Reinstall the backend wheel with a valid RECORD.",
        ) from exc
    document = load_manifest(Path(manifest_path))
    actual_entry_points = set(_backend_entry_point_names(distribution))
    declared_entry_points = {plugin.entry_point for plugin in document.plugins}
    if actual_entry_points != declared_entry_points:
        missing = sorted(actual_entry_points - declared_entry_points)
        unexpected = sorted(declared_entry_points - actual_entry_points)
        details = []
        if missing:
            details.append("missing records for " + ", ".join(missing))
        if unexpected:
            details.append("unknown records for " + ", ".join(unexpected))
        raise BackendPluginManifestError(
            "Manifest entry-point coverage mismatch: " + "; ".join(details),
            field="plugins[].entry_point",
            expected=", ".join(sorted(actual_entry_points)),
            actual=", ".join(sorted(declared_entry_points)),
            remediation=(
                "Make Manifest records exactly cover all triton.backends entry "
                "points in this distribution."
            ),
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
