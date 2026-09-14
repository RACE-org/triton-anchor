"""Structured errors shared by the backend plugin protocol and registry."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


class BackendPluginError(RuntimeError):
    """Base class for backend plugin failures.

    Attributes are intentionally machine-readable so future registry and
    conformance tooling do not need to parse the human-readable message.
    """

    code = "backend_plugin_error"

    def __init__(
        self,
        message: str,
        *,
        plugin_id: Optional[str] = None,
        entry_point: Optional[str] = None,
        detail: Optional[str] = None,
        field: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        remediation: Optional[str] = None,
    ) -> None:
        self.plugin_id = plugin_id
        self.entry_point = entry_point
        self.detail = detail
        self.field = field
        self.expected = expected
        self.actual = actual
        self.remediation = remediation
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Return a stable diagnostic representation."""
        result = {
            "code": self.code,
            "message": str(self),
            "plugin_id": self.plugin_id,
            "entry_point": self.entry_point,
            "detail": self.detail,
        }
        optional = {
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "remediation": self.remediation,
        }
        result.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return result


class BackendPluginInterfaceError(BackendPluginError):
    """The loaded Python object does not satisfy the runtime interface."""

    code = "backend_plugin_interface_error"

    def __init__(
        self,
        missing_fields: Iterable[str] = (),
        *,
        invalid_fields: Iterable[str] = (),
        field_errors: Optional[Mapping[str, str]] = None,
        plugin_id: Optional[str] = None,
        entry_point: Optional[str] = None,
    ) -> None:
        runtime_field_order = ("compiler_cls", "driver_cls")
        unique_missing_fields = tuple(dict.fromkeys(missing_fields))
        self.missing_fields: Tuple[str, ...] = tuple(
            name for name in runtime_field_order if name in unique_missing_fields
        ) + tuple(
            name
            for name in unique_missing_fields
            if name not in runtime_field_order
        )
        unique_invalid_fields = tuple(dict.fromkeys(invalid_fields))
        self.invalid_fields: Tuple[str, ...] = tuple(
            name for name in runtime_field_order if name in unique_invalid_fields
        ) + tuple(
            name for name in unique_invalid_fields if name not in runtime_field_order
        )
        raw_field_errors = dict(field_errors or {})
        self.field_errors: Dict[str, str] = {
            name: raw_field_errors[name]
            for name in runtime_field_order
            if name in raw_field_errors
        }
        self.field_errors.update(
            {
                name: raw_field_errors[name]
                for name in sorted(raw_field_errors)
                if name not in runtime_field_order
            }
        )
        details = []
        if self.missing_fields:
            details.append("missing=" + ", ".join(self.missing_fields))
        if self.invalid_fields:
            details.append("invalid_type=" + ", ".join(self.invalid_fields))
        if self.field_errors:
            details.append(
                "unreadable="
                + ", ".join(
                    f"{name}: {message}"
                    for name, message in sorted(self.field_errors.items())
                )
            )
        detail = "; ".join(details)
        affected_fields = set(self.missing_fields)
        affected_fields.update(self.invalid_fields)
        affected_fields.update(self.field_errors)
        ordered_affected_fields = tuple(
            name for name in runtime_field_order if name in affected_fields
        ) + tuple(
            name
            for name in sorted(affected_fields)
            if name not in runtime_field_order
        )
        # Keep the historical defensive fallback for callers that construct
        # the error without details, while attributing every real validation
        # failure to the exact field(s) observed by the Registry.
        field = ",".join(ordered_affected_fields or runtime_field_order)
        super().__init__(
            "Backend plugin runtime interface is invalid: " + detail,
            plugin_id=plugin_id,
            entry_point=entry_point,
            detail=detail,
            field=field,
            expected="class objects",
            actual=detail,
            remediation=(
                "Expose compiler_cls and driver_cls as class objects from the "
                "backend entry point."
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["missing_fields"] = list(self.missing_fields)
        result["invalid_fields"] = list(self.invalid_fields)
        result["field_errors"] = dict(self.field_errors)
        return result


class BackendPluginManifestError(BackendPluginError):
    """A required manifest is missing, or a declared manifest is invalid."""

    code = "backend_plugin_manifest_error"


class BackendPluginDiscoveryError(BackendPluginError):
    """Installed distribution metadata could not be enumerated safely."""

    code = "backend_plugin_discovery_error"


class BackendPluginProtocolError(BackendPluginError):
    """The plugin protocol version is unsupported or internally inconsistent."""

    code = "backend_plugin_protocol_error"

    def __init__(
        self,
        expected: str,
        actual: str,
        *,
        plugin_id: Optional[str] = None,
        entry_point: Optional[str] = None,
    ) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Incompatible Backend Plugin Protocol: expected {expected}, got {actual}",
            plugin_id=plugin_id,
            entry_point=entry_point,
            detail=f"expected={expected}; actual={actual}",
            field="backend_protocol",
            expected=expected,
            actual=actual,
            remediation=(
                "Install a backend supporting the current protocol, or use a "
                "matching triton-anchor release."
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({"expected": self.expected, "actual": self.actual})
        return result


class BackendPluginCompatibilityError(BackendPluginError):
    """A compatibility dimension does not match the current environment."""

    code = "backend_plugin_compatibility_error"

    def __init__(
        self,
        dimension: str,
        expected: str,
        actual: str,
        *,
        plugin_id: Optional[str] = None,
        entry_point: Optional[str] = None,
        remediation: Optional[str] = None,
    ) -> None:
        self.dimension = dimension
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Incompatible {dimension}: expected {expected}, got {actual}",
            plugin_id=plugin_id,
            entry_point=entry_point,
            detail=f"expected={expected}; actual={actual}",
            field=dimension,
            expected=expected,
            actual=actual,
            remediation=remediation
            or (
                f"Install a backend compatible with the current {dimension}, "
                "or use a matching triton-anchor environment."
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "dimension": self.dimension,
                "expected": self.expected,
                "actual": self.actual,
            }
        )
        return result


class BackendPluginCapabilityError(BackendPluginError):
    """Required Core, plugin, or kernel capabilities are unavailable."""

    code = "backend_plugin_capability_error"

    def __init__(
        self,
        missing_capabilities: Iterable[str],
        *,
        scope: str,
        available_capabilities: Iterable[str] = (),
        missing_plugin_capabilities: Iterable[str] = (),
        missing_kernel_capabilities: Iterable[str] = (),
        plugin_id: Optional[str] = None,
        entry_point: Optional[str] = None,
    ) -> None:
        self.scope = scope
        self.missing_capabilities = tuple(sorted(set(missing_capabilities)))
        self.available_capabilities = tuple(
            sorted(set(available_capabilities))
        )
        self.missing_plugin_capabilities = tuple(
            sorted(set(missing_plugin_capabilities))
        )
        self.missing_kernel_capabilities = tuple(
            sorted(set(missing_kernel_capabilities))
        )
        missing = ", ".join(self.missing_capabilities) or "<none>"
        available = ", ".join(self.available_capabilities) or "<none>"
        if scope == "plugin":
            message = (
                "Core is missing capabilities required by the plugin: "
                + missing
            )
            field = "requires_capabilities"
        elif scope == "kernel":
            message = (
                "Backend is missing capabilities required by the kernel: "
                + missing
            )
            field = "kernel_required_capabilities"
        else:
            message = (
                "Plugin and kernel capability requirements are not satisfied: "
                + missing
            )
            field = (
                "requires_capabilities,kernel_required_capabilities"
            )
        super().__init__(
            message,
            plugin_id=plugin_id,
            entry_point=entry_point,
            detail=f"missing={missing}; available={available}",
            field=field,
            expected="all declared required capabilities are available",
            actual=f"missing={missing}; available={available}",
            remediation=(
                "Choose a backend that provides the required capabilities, "
                "or adjust the kernel/plugin capability requirements."
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "scope": self.scope,
                "missing_capabilities": list(self.missing_capabilities),
                "available_capabilities": list(self.available_capabilities),
                "missing_plugin_capabilities": list(
                    self.missing_plugin_capabilities
                ),
                "missing_kernel_capabilities": list(
                    self.missing_kernel_capabilities
                ),
            }
        )
        return result


class BackendPluginConflictError(BackendPluginError):
    """Two or more plugins make conflicting identity or target claims."""

    code = "backend_plugin_conflict_error"


class BackendPluginLoadError(BackendPluginError):
    """A plugin failed while importing or loading native components."""

    code = "backend_plugin_load_error"


class BackendPluginSelectionError(BackendPluginError):
    """No unique compatible plugin can be selected."""

    code = "backend_plugin_selection_error"


class BackendPluginLifecycleError(BackendPluginError):
    """A plugin attempted an invalid lifecycle transition."""

    code = "backend_plugin_lifecycle_error"
