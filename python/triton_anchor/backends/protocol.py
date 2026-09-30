"""Backend Plugin Protocol 1.0.

The protocol is deliberately small and structural: existing backend modules
that expose ``compiler_cls`` and ``driver_cls`` remain valid runtime objects.
Environment collection and static Manifest parsing are separate modules;
the Registry enforces them before load. Triton's compiler and runtime driver
consume the paired classes through the Registry-backed W9 adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

from packaging.version import InvalidVersion, Version

from .._version import BACKEND_PLUGIN_PROTOCOL_VERSION
from .errors import BackendPluginProtocolError


def _protocol_version(value: str) -> Version:
    try:
        return Version(value)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid Backend Plugin Protocol version {value!r}") from exc


class PluginSource(str, Enum):
    """How a plugin entered the registry."""

    MANIFEST = "manifest"
    LEGACY = "legacy"


class PluginIsolationMode(str, Enum):
    """Boundary between a plugin and the triton-anchor process."""

    PYTHON_ONLY = "python_only"
    SUBPROCESS = "subprocess"
    NATIVE_IN_PROCESS = "native_in_process"


class PluginCompatibilityStatus(str, Enum):
    """Result of compatibility validation."""

    NOT_CHECKED = "not_checked"
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    LEGACY_UNVERIFIED = "legacy_unverified"


class ProtocolFieldStatus(str, Enum):
    """Stable outcomes for one optional Protocol field negotiation."""

    COMPATIBLE_DEFAULT = "compatible_default"
    COMPATIBLE_PRESERVED = "compatible_preserved"
    COMPATIBLE_IGNORED = "compatible_ignored"
    ACCEPTED_WITH_DEPRECATION_DIAGNOSTIC = (
        "accepted_with_structured_deprecation_diagnostic"
    )
    EXPLICIT_PROTOCOL_INCOMPATIBILITY = (
        "explicit_protocol_incompatibility"
    )
    COMPATIBLE = "compatible"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True)
class ProtocolFieldPolicy:
    """Version policy for one real, optional Backend Plugin Protocol field.

    Exact producer and consumer versions govern introduction, deprecation,
    and major removal without changing the package-wide Protocol version.
    """

    name: str
    introduced_in: str
    deprecated_in: str
    removed_in: str
    default_factory: Callable[[], Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        introduced = _protocol_version(self.introduced_in)
        deprecated = _protocol_version(self.deprecated_in)
        removed = _protocol_version(self.removed_in)
        if not self.name:
            raise ValueError("Protocol field name must not be empty")
        if deprecated < introduced:
            raise ValueError("Protocol field cannot be deprecated before introduction")
        if removed <= deprecated:
            raise ValueError("Protocol field removal must follow deprecation")
        if removed.major <= introduced.major:
            raise ValueError("Protocol field removal requires a later major version")


@dataclass(frozen=True)
class ProtocolFieldDiagnostic:
    """Machine-readable diagnostic for Protocol field evolution."""

    code: str
    severity: str
    field: str
    message: str
    introduced_in: str
    deprecated_in: str
    removed_in: str
    producer_protocol_version: Optional[str] = None
    consumer_protocol_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "field": self.field,
            "message": self.message,
            "introduced_in": self.introduced_in,
            "deprecated_in": self.deprecated_in,
            "removed_in": self.removed_in,
        }
        if self.producer_protocol_version is not None:
            result["producer_protocol_version"] = self.producer_protocol_version
        if self.consumer_protocol_version is not None:
            result["consumer_protocol_version"] = self.consumer_protocol_version
        return result


@dataclass(frozen=True)
class ProtocolFieldResult:
    """Observed result of consuming or removing one Protocol field."""

    field: str
    status: ProtocolFieldStatus
    value: Any = None
    diagnostics: Tuple[ProtocolFieldDiagnostic, ...] = ()
    error: Optional[BackendPluginProtocolError] = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "status": self.status.value,
            "value": self.value,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "error": self.error.to_dict() if self.error is not None else None,
        }


DIAGNOSTICS_FIELD_POLICY = ProtocolFieldPolicy(
    name="diagnostics",
    introduced_in="1.1",
    deprecated_in="1.2",
    removed_in="2.0",
    default_factory=dict,
)


def consume_protocol_field(
    producer: Any,
    *,
    producer_protocol_version: str,
    consumer_protocol_version: str,
    policy: ProtocolFieldPolicy = DIAGNOSTICS_FIELD_POLICY,
) -> ProtocolFieldResult:
    """Consume one optional Protocol field across a producer/consumer pair.

    A consumer never reads a field it does not know.  A consumer that knows an
    optional field supplies the documented default for an older producer.
    Different protocol majors fail before reading plugin code.  At and after
    the declared removal major, same-major new producers and consumers remain
    compatible without the removed field.
    """

    producer_version = _protocol_version(producer_protocol_version)
    consumer_version = _protocol_version(consumer_protocol_version)
    introduced_version = _protocol_version(policy.introduced_in)
    removed_version = _protocol_version(policy.removed_in)
    deprecated_version = _protocol_version(policy.deprecated_in)

    if producer_version.major != consumer_version.major:
        expected = (
            f">={consumer_version.major}.0,<{consumer_version.major + 1}.0"
        )
        error = BackendPluginProtocolError(
            expected,
            producer_protocol_version,
        )
        return ProtocolFieldResult(
            field=policy.name,
            status=ProtocolFieldStatus.EXPLICIT_PROTOCOL_INCOMPATIBILITY,
            error=error,
        )

    if consumer_version >= removed_version:
        return ProtocolFieldResult(
            field=policy.name,
            status=ProtocolFieldStatus.COMPATIBLE,
        )

    if consumer_version < introduced_version:
        return ProtocolFieldResult(
            field=policy.name,
            status=ProtocolFieldStatus.COMPATIBLE_IGNORED,
        )

    if producer_version < introduced_version:
        return ProtocolFieldResult(
            field=policy.name,
            status=ProtocolFieldStatus.COMPATIBLE_DEFAULT,
            value=policy.default_factory(),
        )

    try:
        value = getattr(producer, policy.name)
    except AttributeError:
        return ProtocolFieldResult(
            field=policy.name,
            status=ProtocolFieldStatus.COMPATIBLE_DEFAULT,
            value=policy.default_factory(),
        )

    if callable(value):
        value = value()

    if consumer_version >= deprecated_version:
        diagnostic = ProtocolFieldDiagnostic(
            code="backend_plugin_protocol_field_deprecated",
            severity="warning",
            field=policy.name,
            message=(
                f"Backend Plugin Protocol field '{policy.name}' is deprecated "
                f"since {policy.deprecated_in} and will be removed in "
                f"{policy.removed_in}."
            ),
            introduced_in=policy.introduced_in,
            deprecated_in=policy.deprecated_in,
            removed_in=policy.removed_in,
            producer_protocol_version=producer_protocol_version,
            consumer_protocol_version=consumer_protocol_version,
        )
        return ProtocolFieldResult(
            field=policy.name,
            status=ProtocolFieldStatus.ACCEPTED_WITH_DEPRECATION_DIAGNOSTIC,
            value=value,
            diagnostics=(diagnostic,),
        )

    return ProtocolFieldResult(
        field=policy.name,
        status=ProtocolFieldStatus.COMPATIBLE_PRESERVED,
        value=value,
    )


def evaluate_protocol_field_removal(
    candidate_version: str,
    *,
    policy: ProtocolFieldPolicy = DIAGNOSTICS_FIELD_POLICY,
) -> ProtocolFieldResult:
    """Return whether removing ``policy`` in ``candidate_version`` is legal."""

    candidate = _protocol_version(candidate_version)
    introduced = _protocol_version(policy.introduced_in)
    if candidate.major == introduced.major:
        diagnostic = ProtocolFieldDiagnostic(
            code="backend_plugin_protocol_field_removal_forbidden",
            severity="error",
            field=policy.name,
            message=(
                f"Backend Plugin Protocol field '{policy.name}' cannot be "
                f"removed in major version {candidate.major}."
            ),
            introduced_in=policy.introduced_in,
            deprecated_in=policy.deprecated_in,
            removed_in=policy.removed_in,
            consumer_protocol_version=candidate_version,
        )
        return ProtocolFieldResult(
            field=policy.name,
            status=ProtocolFieldStatus.FORBIDDEN,
            diagnostics=(diagnostic,),
        )
    if candidate < _protocol_version(policy.removed_in):
        raise ValueError(
            f"Removal candidate {candidate_version!r} precedes the declared "
            f"removal {policy.removed_in!r}"
        )
    return ProtocolFieldResult(
        field=policy.name,
        status=ProtocolFieldStatus.COMPATIBLE,
    )


class PluginLifecycleState(str, Enum):
    """Lifecycle owned by the future BackendPluginRegistry."""

    DISCOVERED = "discovered"
    VALIDATED = "validated"
    LOADED = "loaded"
    REGISTERED = "registered"
    SELECTED = "selected"
    ACTIVE = "active"
    REJECTED = "rejected"


_ALLOWED_TRANSITIONS: Mapping[PluginLifecycleState, Set[PluginLifecycleState]] = {
    PluginLifecycleState.DISCOVERED: {
        PluginLifecycleState.VALIDATED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.VALIDATED: {
        PluginLifecycleState.LOADED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.LOADED: {
        PluginLifecycleState.REGISTERED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.REGISTERED: {
        PluginLifecycleState.SELECTED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.SELECTED: {
        PluginLifecycleState.ACTIVE,
        PluginLifecycleState.REGISTERED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.ACTIVE: {
        PluginLifecycleState.REGISTERED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.REJECTED: set(),
}


def can_transition(
    current: PluginLifecycleState,
    target: PluginLifecycleState,
    source: PluginSource = PluginSource.MANIFEST,
) -> bool:
    """Return whether Protocol 1.0 permits a lifecycle transition."""
    if (
        current is PluginLifecycleState.DISCOVERED
        and target is PluginLifecycleState.LOADED
    ):
        # Only manifest-less legacy plugins may bypass pre-load validation.
        return source is PluginSource.LEGACY
    return target in _ALLOWED_TRANSITIONS[current]


@runtime_checkable
class BackendPlugin(Protocol):
    """Minimal runtime interface for a Protocol 1.0 backend.

    Both attributes are required in Protocol 1.0.  A future protocol may add
    separately deployable components, but P0 only promises paired compiler and
    driver delivery.
    """

    compiler_cls: type
    driver_cls: type


class BackendPluginBase:
    """Convenience base with default behavior for optional lifecycle hooks.

    Subclassing this class is optional; structural plugins and modules that
    expose ``compiler_cls`` and ``driver_cls`` remain supported.
    """

    compiler_cls: type
    driver_cls: type

    def initialize(self, context: Mapping[str, Any]) -> None:
        """Initialize after validation and loading.  Default: no-op."""

    def shutdown(self) -> None:
        """Release plugin-owned resources.  Default: no-op."""

    def diagnostics(self) -> Dict[str, Any]:
        """Return optional plugin diagnostics.  Default: empty."""
        return {}
