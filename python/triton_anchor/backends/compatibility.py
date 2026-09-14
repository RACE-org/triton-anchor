"""Import-free compatibility checks for backend plugin manifests."""

from __future__ import annotations

from dataclasses import dataclass
from email.parser import Parser
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag, parse_tag, sys_tags
from packaging.version import InvalidVersion, Version

from .environment import CoreEnvironment
from .errors import (
    BackendPluginCompatibilityError,
    BackendPluginManifestError,
    BackendPluginProtocolError,
)
from .manifest import BackendPluginManifest
from .native import NativeArtifact, inspect_native_artifacts
from .protocol import PluginIsolationMode


@dataclass(frozen=True)
class CompatibilityCheck:
    """One successful pre-load compatibility comparison."""

    dimension: str
    expected: str
    actual: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "dimension": self.dimension,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class CompatibilityReport:
    """Successful result of the complete W5 pre-load validation."""

    plugin_id: str
    entry_point: str
    checks: Tuple[CompatibilityCheck, ...]
    native_artifacts: Tuple[NativeArtifact, ...] = ()
    compatible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "entry_point": self.entry_point,
            "compatible": self.compatible,
            "checks": [check.to_dict() for check in self.checks],
            "native_artifacts": [
                artifact.to_dict() for artifact in self.native_artifacts
            ],
        }


def _actual_or_unknown(value: Optional[str]) -> str:
    return value if value is not None else "<unknown>"


def _manifest_field_for_dimension(dimension: str) -> str:
    return {
        "Backend Plugin Protocol": "backend_protocol",
        "triton-anchor Core version": "requires_core",
        "Triton version": "requires_triton.version",
        "vendored Triton commit": "requires_triton.commit",
        "LLVM version": "requires_llvm_version",
        "LLVM commit": "requires_llvm_commit",
        "MLIR version": "requires_mlir_version",
        "MLIR commit": "requires_mlir_commit",
        "Core ABI fingerprint": "abi_fingerprint",
    }.get(dimension, dimension)


def _version_check(
    plugin: BackendPluginManifest,
    dimension: str,
    expected: str,
    actual: Optional[str],
    *,
    protocol: bool = False,
) -> CompatibilityCheck:
    try:
        specifier = SpecifierSet(expected)
    except InvalidSpecifier as exc:
        raise BackendPluginManifestError(
            f"Invalid {dimension} version specifier '{expected}': {exc}",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field=_manifest_field_for_dimension(dimension),
            expected="a valid PEP 440 version specifier",
            actual=expected,
            remediation=f"Replace {dimension} with a valid PEP 440 specifier.",
        ) from exc

    actual_text = _actual_or_unknown(actual)
    if actual is None:
        if protocol:
            raise BackendPluginProtocolError(
                expected,
                actual_text,
                plugin_id=plugin.plugin_id,
                entry_point=plugin.entry_point,
            )
        raise BackendPluginCompatibilityError(
            dimension,
            expected,
            actual_text,
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                f"The plugin requires an exact {dimension} check, but the "
                "current build cannot prove that value. Rebuild triton-anchor "
                "with generated build metadata or relax the plugin requirement."
            ),
        )

    try:
        actual_version = Version(actual)
    except InvalidVersion as exc:
        if protocol:
            raise BackendPluginProtocolError(
                expected,
                actual,
                plugin_id=plugin.plugin_id,
                entry_point=plugin.entry_point,
            ) from exc
        raise BackendPluginCompatibilityError(
            dimension,
            expected,
            actual,
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                f"The current {dimension} value is not a comparable version. "
                "Use a build that records a normalized version."
            ),
        ) from exc

    if actual_version not in specifier:
        if protocol:
            raise BackendPluginProtocolError(
                expected,
                actual,
                plugin_id=plugin.plugin_id,
                entry_point=plugin.entry_point,
            )
        raise BackendPluginCompatibilityError(
            dimension,
            expected,
            actual,
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
        )
    return CompatibilityCheck(dimension, expected, actual)


def _exact_check(
    plugin: BackendPluginManifest,
    dimension: str,
    expected: str,
    actual: Optional[str],
    *,
    remediation: Optional[str] = None,
    case_sensitive: bool = True,
) -> CompatibilityCheck:
    actual_text = _actual_or_unknown(actual)
    matches = actual == expected
    if not case_sensitive and actual is not None:
        matches = actual.lower() == expected.lower()
    if actual is None or not matches:
        raise BackendPluginCompatibilityError(
            dimension,
            expected,
            actual_text,
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=remediation,
        )
    return CompatibilityCheck(dimension, expected, actual)


def validate_triton_version_requirement(
    plugin: BackendPluginManifest, environment: CoreEnvironment
) -> CompatibilityReport:
    """Validate only the staged Triton version-number example."""
    check = _version_check(
        plugin,
        "Triton version",
        plugin.requires_triton.version,
        environment.triton_version,
    )
    return CompatibilityReport(
        plugin_id=plugin.plugin_id,
        entry_point=plugin.entry_point,
        checks=(check,),
    )


def validate_triton_requirement(
    plugin: BackendPluginManifest, environment: CoreEnvironment
) -> CompatibilityReport:
    """Validate the W3 Triton version and optional exact commit example."""
    checks = list(
        validate_triton_version_requirement(plugin, environment).checks
    )
    if plugin.requires_triton.commit is not None:
        checks.append(
            _exact_check(
                plugin,
                "vendored Triton commit",
                plugin.requires_triton.commit,
                environment.vendored_triton_commit,
                case_sensitive=False,
            )
        )
    return CompatibilityReport(
        plugin_id=plugin.plugin_id,
        entry_point=plugin.entry_point,
        checks=tuple(checks),
    )


def _distribution_wheel_tags(
    plugin: BackendPluginManifest,
    distribution: Any,
) -> Set[Tag]:
    read_text = getattr(distribution, "read_text", None)
    if not callable(read_text):
        raise BackendPluginCompatibilityError(
            "wheel platform metadata",
            "readable WHEEL metadata",
            "<unavailable>",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Install the backend as a standards-compliant wheel containing "
                "dist-info/WHEEL metadata."
            ),
        )
    try:
        wheel_text = read_text("WHEEL")
    except Exception as exc:
        raise BackendPluginCompatibilityError(
            "wheel platform metadata",
            "readable WHEEL metadata",
            f"<error: {exc}>",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Reinstall the backend from a valid wheel and verify its "
                "dist-info/WHEEL file."
            ),
        ) from exc
    if not wheel_text:
        raise BackendPluginCompatibilityError(
            "wheel platform metadata",
            "at least one wheel Tag",
            "<missing WHEEL metadata>",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Install the backend from a wheel containing one or more Tag "
                "headers."
            ),
        )

    try:
        headers = Parser().parsestr(wheel_text)
        raw_tags = headers.get_all("Tag", [])
        tags = set()
        for raw_tag in raw_tags:
            tags.update(parse_tag(raw_tag))
    except Exception as exc:
        raise BackendPluginCompatibilityError(
            "wheel platform metadata",
            "valid wheel Tag values",
            f"<invalid: {exc}>",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation="Rebuild the backend wheel with valid PEP 425 tags.",
        ) from exc
    if not tags:
        raise BackendPluginCompatibilityError(
            "wheel platform metadata",
            "at least one wheel Tag",
            "<none>",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation="Rebuild the backend wheel with a valid Tag header.",
        )
    return tags


def _validate_distribution_platform(
    plugin: BackendPluginManifest,
    distribution: Any,
    supported_tags: Optional[Iterable[Tag]],
) -> CompatibilityCheck:
    wheel_tags = _distribution_wheel_tags(plugin, distribution)
    current_tags = set(supported_tags) if supported_tags is not None else set(sys_tags())
    actual = ",".join(sorted(str(tag) for tag in wheel_tags))
    if wheel_tags.isdisjoint(current_tags):
        raise BackendPluginCompatibilityError(
            "wheel platform tag",
            "a tag supported by the current Python and platform",
            actual,
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Install a backend wheel built for the current Python ABI and "
                "platform."
            ),
        )
    return CompatibilityCheck(
        "wheel platform tag",
        "a tag supported by the current Python and platform",
        actual,
    )


def validate_backend_plugin(
    plugin: BackendPluginManifest,
    environment: CoreEnvironment,
    *,
    distribution: Any,
    core_abi_fingerprint: Optional[str] = None,
    supported_tags: Optional[Iterable[Tag]] = None,
) -> CompatibilityReport:
    """Run every W5 check that is applicable before ``entry_point.load()``."""
    checks = []

    if distribution is None:
        raise BackendPluginCompatibilityError(
            "distribution metadata",
            "the authoritative owning distribution",
            "<unavailable>",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Pass the distribution discovered through "
                "importlib.metadata.distributions(); complete pre-load "
                "validation cannot skip wheel and artifact checks."
            ),
        )

    checks.append(
        _validate_distribution_platform(plugin, distribution, supported_tags)
    )
    if not environment.build_info_generated:
        raise BackendPluginCompatibilityError(
            "Core build provenance",
            "generated wheel build metadata",
            "source/editable metadata",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Validate production Manifest plugins against a freshly built "
                "triton-anchor wheel. Use the explicit triton_version profile "
                "only for the frozen migration test, not as a production "
                "security downgrade."
            ),
        )
    native_report = inspect_native_artifacts(plugin, distribution)
    if native_report.artifacts:
        checks.append(
            CompatibilityCheck(
                "native library artifacts",
                ",".join(plugin.native_libraries),
                ",".join(native_report.paths),
            )
        )

    checks.append(
        _version_check(
            plugin,
            "Backend Plugin Protocol",
            plugin.backend_protocol,
            environment.backend_protocol_version,
            protocol=True,
        )
    )

    if plugin.requires_core is not None:
        checks.append(
            _version_check(
                plugin,
                "triton-anchor Core version",
                plugin.requires_core,
                environment.core_version,
            )
        )
    checks.append(
        _version_check(
            plugin,
            "Triton version",
            plugin.requires_triton.version,
            environment.triton_version,
        )
    )
    if plugin.requires_triton.commit is not None:
        checks.append(
            _exact_check(
                plugin,
                "vendored Triton commit",
                plugin.requires_triton.commit,
                environment.vendored_triton_commit,
                case_sensitive=False,
            )
        )
    if plugin.requires_llvm_version is not None:
        checks.append(
            _version_check(
                plugin,
                "LLVM version",
                plugin.requires_llvm_version,
                environment.actual_llvm_version,
            )
        )
    if plugin.requires_llvm_commit is not None:
        checks.append(
            _exact_check(
                plugin,
                "LLVM commit",
                plugin.requires_llvm_commit,
                environment.actual_llvm_commit,
                case_sensitive=False,
            )
        )
    if plugin.requires_mlir_version is not None:
        checks.append(
            _version_check(
                plugin,
                "MLIR version",
                plugin.requires_mlir_version,
                environment.actual_mlir_version,
            )
        )
    if plugin.requires_mlir_commit is not None:
        checks.append(
            _exact_check(
                plugin,
                "MLIR commit",
                plugin.requires_mlir_commit,
                environment.actual_mlir_commit,
                case_sensitive=False,
            )
        )

    if environment.build_info_generated:
        checks.append(
            _exact_check(
                plugin,
                "core Python SOABI",
                _actual_or_unknown(environment.built_python_soabi),
                environment.runtime_python_soabi,
                remediation=(
                    "Use the Python interpreter for which triton-anchor was "
                    "built, or rebuild triton-anchor in this interpreter."
                ),
            )
        )
        checks.append(
            _exact_check(
                plugin,
                "core build platform",
                _actual_or_unknown(environment.built_platform),
                environment.runtime_platform,
                remediation=(
                    "Install or rebuild triton-anchor for the current platform."
                ),
            )
        )

    if plugin.isolation_mode is PluginIsolationMode.NATIVE_IN_PROCESS:
        checks.append(
            _exact_check(
                plugin,
                "Core ABI fingerprint",
                plugin.abi_fingerprint or "<missing>",
                core_abi_fingerprint,
                remediation=(
                    "A native_in_process backend requires the exact Core ABI "
                    "fingerprint. Complete ABI governance or use subprocess "
                    "isolation; never guess this value."
                ),
            )
        )
    elif plugin.isolation_mode is PluginIsolationMode.SUBPROCESS:
        raise BackendPluginCompatibilityError(
            "subprocess IR contract",
            "a declared and supported input/output IR contract",
            "<unavailable in Manifest 1.0>",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Use python_only for the current protocol, or wait for the "
                "versioned subprocess IR contract introduced with capability "
                "negotiation. Do not treat an unchecked subprocess as compatible."
            ),
        )

    return CompatibilityReport(
        plugin_id=plugin.plugin_id,
        entry_point=plugin.entry_point,
        checks=tuple(checks),
        native_artifacts=native_report.artifacts,
    )
