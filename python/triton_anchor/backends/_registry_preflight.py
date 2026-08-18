"""Compatibility entry point for Registry preflight evaluation.

Pure outcome and application-plan modeling lives in ``_registry_validator``.
This wrapper preserves the existing call site and dependency-injection seam
until the Registry facade is integrated in its own commit.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

from packaging.tags import Tag

from ._registry_validator import (
    ValidationPlan,
    ValidationRecord as PreflightRecord,
    plan_record_validation,
)
from .capabilities import CapabilityReport, validate_plugin_capabilities
from .compatibility import (
    CompatibilityReport,
    validate_backend_plugin,
    validate_triton_version_requirement,
)
from .environment import CoreEnvironment


PreflightOutcome = ValidationPlan


def evaluate_record_preflight(
    record: PreflightRecord,
    environment: CoreEnvironment,
    *,
    preflight_profile: str,
    core_abi_fingerprint: Optional[str],
    supported_tags: Optional[Tuple[Tag, ...]],
    core_capabilities: Iterable[str],
    compatibility_validator: Callable[..., CompatibilityReport] = (
        validate_backend_plugin
    ),
    triton_version_validator: Callable[..., CompatibilityReport] = (
        validate_triton_version_requirement
    ),
    capability_validator: Callable[..., CapabilityReport] = (
        validate_plugin_capabilities
    ),
) -> ValidationPlan:
    """Evaluate compatibility without mutating Registry state."""
    return plan_record_validation(
        record,
        environment,
        preflight_profile=preflight_profile,
        core_abi_fingerprint=core_abi_fingerprint,
        supported_tags=supported_tags,
        core_capabilities=core_capabilities,
        compatibility_validator=compatibility_validator,
        triton_version_validator=triton_version_validator,
        capability_validator=capability_validator,
    )
