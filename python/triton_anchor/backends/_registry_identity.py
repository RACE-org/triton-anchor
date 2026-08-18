"""Pure identity normalization for backend Registry discovery records.

This module owns no Registry state and performs no provider, plugin, or
entry-point callbacks.  Compatibility wrappers in ``_registry_catalog`` keep
the historical internal import paths and function identities stable.
"""

from __future__ import annotations

from typing import Any, Callable, Container, Optional, Tuple

from packaging.utils import canonicalize_name


def canonical_distribution_key(distribution_name: Optional[str]) -> str:
    """Return the canonical distribution component used in record IDs."""
    return canonicalize_name(distribution_name or "unknown-distribution")


def distribution_identity(
    distribution: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Read a distribution identity without importing plugin code."""
    try:
        metadata = getattr(distribution, "metadata", None)
    except Exception:
        metadata = None
    name = None
    if metadata is not None:
        try:
            name = metadata.get("Name")
        except Exception:
            name = None
    if name is None:
        try:
            name = getattr(distribution, "name", None)
        except Exception:
            name = None
    try:
        version = getattr(distribution, "version", None)
    except Exception:
        version = None

    try:
        name_text = str(name) if name is not None else None
    except Exception:
        name_text = None
    try:
        version_text = str(version) if version is not None else None
    except Exception:
        version_text = None
    return name_text, version_text


def entry_point_name(entry_point: Any) -> str:
    """Read and normalize an entry-point name defensively."""
    try:
        value = getattr(entry_point, "name", "")
        return str(value) if value is not None else ""
    except Exception:
        return ""


def entry_point_value(entry_point: Any) -> str:
    """Read and normalize an entry-point value defensively."""
    try:
        value = getattr(entry_point, "value", None)
        return str(value) if value is not None else ""
    except Exception:
        return ""


def allocate_record_id(
    existing_ids: Container[str],
    distribution_name: Optional[str],
    entry_point: str,
) -> str:
    """Allocate the historical deterministic Registry record identity."""
    return _allocate_record_id(
        existing_ids,
        distribution_name,
        entry_point,
        canonicalizer=canonicalize_name,
    )


def _allocate_record_id(
    existing_ids: Container[str],
    distribution_name: Optional[str],
    entry_point: str,
    *,
    canonicalizer: Callable[[str], str],
) -> str:
    """Allocate an ID with the caller's established canonicalizer binding."""
    distribution_key = canonicalizer(
        distribution_name or "unknown-distribution"
    )
    base = f"{distribution_key}:{entry_point}"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}#{suffix}"
        suffix += 1
    return candidate


__all__ = [
    "allocate_record_id",
    "canonical_distribution_key",
    "distribution_identity",
    "entry_point_name",
    "entry_point_value",
]
