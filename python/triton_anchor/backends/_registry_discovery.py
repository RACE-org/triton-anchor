"""Metadata-only discovery helpers for :mod:`triton_anchor.backends.registry`.

The public Registry owns locking and all state mutation.  Helpers in this
module only inspect installed distribution metadata and emit records through
the callback supplied by that Registry.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Optional

from packaging.utils import canonicalize_name

from ._registry_catalog import (
    DiscoveryRecordMetadata,
    allocate_record_id,
    distribution_identity,
    entry_point_name,
    entry_point_value,
    record_metadata,
)

from .errors import (
    BackendPluginDiscoveryError,
    BackendPluginError,
    BackendPluginManifestError,
)
from .manifest import (
    MANIFEST_FILENAME,
    BackendPluginManifest,
    load_distribution_manifest,
)
from .protocol import (
    PluginCompatibilityStatus,
    PluginLifecycleState,
    PluginSource,
)


BACKEND_ENTRY_POINT_GROUP = "triton.backends"


def source_hint(distribution: Any) -> Optional[PluginSource]:
    try:
        files = getattr(distribution, "files", None)
    except Exception:
        return None
    if files is None:
        return None
    try:
        if any(
            PurePosixPath(str(item)).name == MANIFEST_FILENAME
            for item in files
        ):
            return PluginSource.MANIFEST
    except Exception:
        return None
    return PluginSource.LEGACY


def copy_manifest_error(
    error: BackendPluginError,
    *,
    entry_point: str,
    plugin_id: Optional[str] = None,
) -> BackendPluginManifestError:
    return BackendPluginManifestError(
        str(error),
        plugin_id=plugin_id if plugin_id is not None else error.plugin_id,
        entry_point=entry_point,
        detail=error.detail,
        field=error.field,
        expected=error.expected,
        actual=error.actual,
        remediation=error.remediation
        or "Fix the installed backend Manifest and reinstall its wheel.",
    )


StoreRecord = Callable[..., Any]
RecordRegistryError = Callable[[BackendPluginError], None]
ManifestLoader = Callable[[Any], Any]


def discover_distributions(
    distributions: Iterable[Any],
    *,
    store_record: StoreRecord,
    record_registry_error: RecordRegistryError,
    load_manifest: ManifestLoader = load_distribution_manifest,
) -> None:
    """Inspect and store backend entry points in deterministic order."""
    candidates = []
    for distribution in distributions:
        try:
            entry_points = tuple(
                entry_point
                for entry_point in (
                    getattr(distribution, "entry_points", ()) or ()
                )
                if getattr(entry_point, "group", None)
                == BACKEND_ENTRY_POINT_GROUP
            )
        except Exception as exc:
            name, _ = distribution_identity(distribution)
            error = BackendPluginDiscoveryError(
                "Unable to enumerate distribution entry points: "
                f"{name or '<unknown>'}: {exc}",
                field="distribution.entry_points",
                expected="readable entry-point metadata",
                actual=f"<error: {exc}>",
                remediation=(
                    "Reinstall the affected distribution with valid "
                    "entry-point metadata."
                ),
            )
            record_registry_error(error)
            continue
        if not entry_points:
            continue
        name, version = distribution_identity(distribution)
        if not name:
            error = BackendPluginDiscoveryError(
                "Backend distribution has no readable package name",
                field="distribution.metadata.Name",
                expected="a non-empty installed distribution name",
                actual="<unavailable>",
                remediation=(
                    "Reinstall the affected backend with valid "
                    "distribution metadata."
                ),
            )
            record_registry_error(error)
            source = source_hint(distribution)
            for entry_point in entry_points:
                store_record(
                    entry_point=entry_point,
                    distribution=distribution,
                    source=source,
                    manifest=None,
                    state=PluginLifecycleState.REJECTED,
                    compatibility_status=(
                        PluginCompatibilityStatus.NOT_CHECKED
                    ),
                    error=error,
                )
            continue
        if any(not entry_point_name(ep) for ep in entry_points):
            error = BackendPluginDiscoveryError(
                f"Backend distribution '{name}' has an unreadable "
                "entry-point name",
                field="entry_point.name",
                expected="a non-empty entry-point name",
                actual="<unavailable>",
                remediation=(
                    "Reinstall the affected backend with valid "
                    "entry-point metadata."
                ),
            )
            record_registry_error(error)
            source = source_hint(distribution)
            for entry_point in entry_points:
                store_record(
                    entry_point=entry_point,
                    distribution=distribution,
                    source=source,
                    manifest=None,
                    state=PluginLifecycleState.REJECTED,
                    compatibility_status=(
                        PluginCompatibilityStatus.NOT_CHECKED
                    ),
                    error=error,
                )
            continue
        candidates.append(
            (
                (
                    canonicalize_name(name or "unknown-distribution"),
                    str(version or ""),
                    tuple(
                        sorted(
                            (
                                entry_point_name(ep),
                                entry_point_value(ep),
                            )
                            for ep in entry_points
                        )
                    ),
                ),
                distribution,
                entry_points,
            )
        )

    for _, distribution, entry_points in sorted(
        candidates, key=lambda item: item[0]
    ):
        ordered_entry_points = tuple(
            sorted(
                entry_points,
                key=lambda ep: (
                    entry_point_name(ep),
                    entry_point_value(ep),
                ),
            )
        )
        try:
            document = load_manifest(distribution)
        except BackendPluginError as exc:
            source = source_hint(distribution)
            for entry_point in ordered_entry_points:
                error = copy_manifest_error(
                    exc,
                    entry_point=entry_point_name(entry_point),
                )
                store_record(
                    entry_point=entry_point,
                    distribution=distribution,
                    source=source,
                    manifest=None,
                    state=PluginLifecycleState.REJECTED,
                    compatibility_status=(
                        PluginCompatibilityStatus.NOT_CHECKED
                    ),
                    error=error,
                )
            continue
        except Exception as exc:
            source = source_hint(distribution)
            unexpected = BackendPluginManifestError(
                "Unexpected error while reading distribution Manifest: "
                f"{exc}",
                field="distribution_manifest",
                expected="readable, valid static Manifest metadata",
                actual=f"<error: {exc}>",
                remediation=(
                    "Repair or reinstall the affected backend "
                    "distribution; discovery did not import plugin code."
                ),
            )
            for entry_point in ordered_entry_points:
                store_record(
                    entry_point=entry_point,
                    distribution=distribution,
                    source=source,
                    manifest=None,
                    state=PluginLifecycleState.REJECTED,
                    compatibility_status=(
                        PluginCompatibilityStatus.NOT_CHECKED
                    ),
                    error=copy_manifest_error(
                        unexpected,
                        entry_point=entry_point_name(entry_point),
                    ),
                )
            continue

        if document is None:
            for entry_point in ordered_entry_points:
                store_record(
                    entry_point=entry_point,
                    distribution=distribution,
                    source=PluginSource.LEGACY,
                    manifest=None,
                    state=PluginLifecycleState.DISCOVERED,
                    compatibility_status=(
                        PluginCompatibilityStatus.LEGACY_UNVERIFIED
                    ),
                )
            continue

        manifests = {
            plugin.entry_point: plugin for plugin in document.plugins
        }
        for entry_point in ordered_entry_points:
            manifest: BackendPluginManifest = manifests[
                entry_point_name(entry_point)
            ]
            store_record(
                entry_point=entry_point,
                distribution=distribution,
                source=PluginSource.MANIFEST,
                manifest=manifest,
                state=PluginLifecycleState.DISCOVERED,
                compatibility_status=PluginCompatibilityStatus.NOT_CHECKED,
            )
