"""Pure, deterministic backend selection before plugin loading.

W8 consumes record-like objects produced by discovery/validation.  It only
reads record metadata and never calls ``entry_point.load()`` or accesses the
runtime compiler/driver classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .capabilities import (
    CapabilityReport,
    evaluate_plugin_capabilities,
    validate_plugin_capabilities,
)
from .conflicts import detect_static_conflicts
from .errors import BackendPluginError, BackendPluginSelectionError


BACKEND_SELECTOR_ENV = "TRITON_ANCHOR_BACKEND"
_SELECTABLE_MANIFEST_STATES = {
    "validated",
    "loaded",
    "registered",
    "selected",
    "active",
}


class SelectionMethod(str, Enum):
    """The policy tier that produced a selection."""

    PYTHON_EXPLICIT = "python_explicit"
    ENVIRONMENT = "environment"
    SOLE_CANDIDATE = "sole_candidate"
    MANIFEST_PRIORITY = "manifest_priority"


@dataclass(frozen=True)
class SelectionDecision:
    """Stable selection result plus the original record for later loading."""

    record_id: str
    registry_key: str
    plugin_id: Optional[str]
    entry_point_name: str
    target: str
    method: SelectionMethod
    selector: Optional[str]
    priority: int
    is_legacy: bool
    candidate_record_ids: Tuple[str, ...]
    capability_report: Optional[CapabilityReport]
    record: Any = field(repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic, JSON-compatible diagnostic view."""
        return {
            "record_id": self.record_id,
            "registry_key": self.registry_key,
            "plugin_id": self.plugin_id,
            "entry_point": self.entry_point_name,
            "target": self.target,
            "method": self.method.value,
            "selector": self.selector,
            "priority": self.priority,
            "is_legacy": self.is_legacy,
            "candidate_record_ids": list(self.candidate_record_ids),
            "capabilities": (
                self.capability_report.to_dict()
                if self.capability_report is not None
                else None
            ),
        }


@dataclass(frozen=True)
class _RecordView:
    record: Any = field(repr=False, compare=False)
    record_id: str
    registry_key: str
    plugin_id: Optional[str]
    entry_point_name: str
    manifest: Any = field(repr=False, compare=False)
    is_legacy: bool
    state: Optional[str]
    compatibility_status: Optional[str]
    priority: int

    @property
    def sort_key(self) -> Tuple[str, str, str, str]:
        return (
            self.record_id,
            self.registry_key,
            self.plugin_id or "",
            self.entry_point_name,
        )

    @property
    def selector_keys(self) -> Tuple[str, ...]:
        values = {self.record_id, self.registry_key}
        if not self.is_legacy and self.plugin_id is not None:
            values.add(self.plugin_id)
        return tuple(sorted(values))


def _enum_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return raw if isinstance(raw, str) else str(raw)


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BackendPluginSelectionError(
            f"Selection input '{field_name}' must be a non-empty string",
            field=field_name,
            expected="a non-empty string",
            actual=repr(value),
            remediation=f"Provide a valid {field_name}.",
        )
    if value != value.strip():
        raise BackendPluginSelectionError(
            f"Selection input '{field_name}' cannot have surrounding whitespace",
            field=field_name,
            expected="a string without surrounding whitespace",
            actual=repr(value),
            remediation=f"Remove surrounding whitespace from {field_name}.",
        )
    return value


def _optional_record_string(record: Any, field_name: str) -> Optional[str]:
    try:
        value = getattr(record, field_name, None)
    except Exception as exc:
        raise BackendPluginSelectionError(
            f"Backend record field '{field_name}' is unreadable: {exc}",
            field=field_name,
            expected="readable record metadata",
            actual=f"<error: {exc}>",
            remediation="Repair the discovered backend record before selection.",
        ) from exc
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BackendPluginSelectionError(
            f"Backend record field '{field_name}' must be a non-empty string",
            field=field_name,
            expected="a non-empty string",
            actual=repr(value),
            remediation="Repair the discovered backend record before selection.",
        )
    return value


def _project_record(record: Any) -> _RecordView:
    record_id = _optional_record_string(record, "record_id")
    if record_id is None:
        raise BackendPluginSelectionError(
            "Every backend selection record must have a record_id",
            field="record_id",
            expected="a stable non-empty record identifier",
            actual="<missing>",
            remediation="Pass records created by BackendPluginRegistry.discover().",
        )

    try:
        manifest = getattr(record, "manifest", None)
        source = _enum_value(getattr(record, "source", None))
        state = _enum_value(getattr(record, "state", None))
        compatibility_status = _enum_value(
            getattr(record, "compatibility_status", None)
        )
    except Exception as exc:
        raise BackendPluginSelectionError(
            f"Backend record '{record_id}' metadata is unreadable: {exc}",
            field="record",
            expected="readable discovery and validation metadata",
            actual=f"<error: {exc}>",
            remediation="Repair or rediscover the backend record before selection.",
        ) from exc

    is_legacy = manifest is None and source == "legacy"
    plugin_id = (
        getattr(manifest, "plugin_id", None)
        if manifest is not None
        else _optional_record_string(record, "plugin_id")
    )
    if plugin_id is not None and (
        not isinstance(plugin_id, str) or not plugin_id
    ):
        raise BackendPluginSelectionError(
            f"Backend record '{record_id}' has an invalid plugin_id",
            field="plugin_id",
            expected="a non-empty string or null for Legacy",
            actual=repr(plugin_id),
            remediation="Repair the backend Manifest before selection.",
        )

    registry_key = _optional_record_string(record, "registry_key")
    if registry_key is None:
        registry_key = plugin_id or record_id

    entry_point_name = _optional_record_string(record, "entry_point_name")
    if entry_point_name is None and manifest is not None:
        entry_point_name = getattr(manifest, "entry_point", None)
    if not isinstance(entry_point_name, str) or not entry_point_name:
        raise BackendPluginSelectionError(
            f"Backend record '{record_id}' has no entry-point name",
            field="entry_point_name",
            expected="a non-empty triton.backends entry-point name",
            actual=repr(entry_point_name),
            remediation="Rediscover the backend from valid package metadata.",
        )

    priority = getattr(manifest, "priority", 0) if manifest is not None else 0
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise BackendPluginSelectionError(
            f"Backend record '{record_id}' has an invalid priority",
            plugin_id=plugin_id,
            entry_point=entry_point_name,
            field="priority",
            expected="an integer",
            actual=repr(priority),
            remediation="Set Manifest priority to an integer.",
        )

    return _RecordView(
        record=record,
        record_id=record_id,
        registry_key=registry_key,
        plugin_id=plugin_id,
        entry_point_name=entry_point_name,
        manifest=manifest,
        is_legacy=is_legacy,
        state=state,
        compatibility_status=compatibility_status,
        priority=priority,
    )


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return _non_empty_string(target, "target")
    if isinstance(target, Mapping):
        value = target.get("backend")
    else:
        value = getattr(target, "backend", None)
    return _non_empty_string(value, "target.backend")


def _capability_names(
    values: Iterable[str], field_name: str
) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise BackendPluginSelectionError(
            f"Selection input '{field_name}' must be an iterable of strings",
            field=field_name,
            expected="an iterable of unique capability strings",
            actual=repr(values),
            remediation=f"Pass {field_name} as a list, tuple, or set.",
        )
    try:
        result = tuple(values)
    except TypeError as exc:
        raise BackendPluginSelectionError(
            f"Selection input '{field_name}' is not iterable",
            field=field_name,
            expected="an iterable of unique capability strings",
            actual=repr(values),
            remediation=f"Pass {field_name} as a list, tuple, or set.",
        ) from exc
    if any(
        not isinstance(value, str)
        or not value
        or value != value.strip()
        for value in result
    ):
        raise BackendPluginSelectionError(
            f"Selection input '{field_name}' contains an invalid capability",
            field=field_name,
            expected="non-empty strings without surrounding whitespace",
            actual=repr(result),
            remediation="Use exact, non-empty capability identifiers.",
        )
    if len(result) != len(set(result)):
        raise BackendPluginSelectionError(
            f"Selection input '{field_name}' contains duplicates",
            field=field_name,
            expected="unique capability identifiers",
            actual=repr(result),
            remediation="Remove duplicate capability identifiers.",
        )
    return tuple(sorted(result))


def _manifest_ineligible_reason(view: _RecordView) -> Optional[str]:
    if view.manifest is None:
        return "record has neither a Manifest nor Legacy source metadata"
    if view.state not in _SELECTABLE_MANIFEST_STATES:
        return (
            f"Manifest record state is '{view.state}', not a validated-or-later "
            "selectable state"
        )
    if view.compatibility_status != "compatible":
        return (
            "Manifest compatibility status is "
            f"'{view.compatibility_status}', not 'compatible'"
        )
    return None


def _legacy_ineligible_reason(view: _RecordView) -> Optional[str]:
    if not view.is_legacy:
        return "record has neither a selectable Manifest nor Legacy source"
    if view.state not in {
        "discovered",
        "loaded",
        "registered",
        "selected",
        "active",
    }:
        return f"Legacy record state is '{view.state}', not selectable"
    if view.compatibility_status != "legacy_unverified":
        return (
            "Legacy compatibility status is "
            f"'{view.compatibility_status}', not 'legacy_unverified'"
        )
    return None


def _capability_report(
    view: _RecordView,
    *,
    core_provided: Tuple[str, ...],
    kernel_required: Tuple[str, ...],
) -> CapabilityReport:
    try:
        return evaluate_plugin_capabilities(
            view.manifest,
            core_provided=core_provided,
            kernel_required=kernel_required,
        )
    except ValueError as exc:
        raise BackendPluginSelectionError(
            f"Invalid capability input while selecting '{view.record_id}': {exc}",
            plugin_id=view.plugin_id,
            entry_point=view.entry_point_name,
            field="capabilities",
            expected="valid unique capability declarations",
            actual=str(exc),
            remediation="Repair the capability declarations before selection.",
        ) from exc


def _matching_selector(
    views: Tuple[_RecordView, ...],
    selector: str,
) -> _RecordView:
    matches = tuple(
        view for view in views if selector in view.selector_keys
    )
    if not matches:
        raise BackendPluginSelectionError(
            f"No backend plugin matches selector '{selector}'",
            field="backend_selector",
            expected="a Manifest plugin_id/record key or a Legacy record_id/registry_key",
            actual=selector,
            remediation=(
                "Use a selector shown by Registry list/inspect diagnostics. "
                "Legacy entry-point names are intentionally not accepted."
            ),
        )
    if len(matches) > 1:
        record_ids = tuple(sorted(view.record_id for view in matches))
        raise BackendPluginSelectionError(
            f"Backend selector '{selector}' is ambiguous: "
            + ", ".join(record_ids),
            field="backend_selector",
            expected="one backend record",
            actual=", ".join(record_ids),
            remediation="Select one record by its unique record_id.",
        )
    return matches[0]


def _selected_view(
    view: _RecordView,
    *,
    target_name: str,
    core_provided: Tuple[str, ...],
    kernel_required: Tuple[str, ...],
) -> Optional[CapabilityReport]:
    if view.state == "rejected":
        error = getattr(view.record, "error", None)
        if isinstance(error, BackendPluginError):
            raise error
    if view.is_legacy:
        reason = _legacy_ineligible_reason(view)
        if reason is not None:
            raise BackendPluginSelectionError(
                f"Legacy backend '{view.record_id}' cannot be selected: {reason}",
                entry_point=view.entry_point_name,
                field="state",
                expected="a discovered Legacy record",
                actual=view.state or "<unknown>",
                remediation="Rediscover the Legacy backend or select a valid record.",
            )
        if kernel_required:
            missing = ", ".join(kernel_required)
            raise BackendPluginSelectionError(
                "Legacy backend cannot satisfy declared kernel capabilities: "
                + missing,
                entry_point=view.entry_point_name,
                field="kernel_required_capabilities",
                expected="no capability requirements for an unverified Legacy backend",
                actual=missing,
                remediation=(
                    "Migrate the backend to a Manifest with static capabilities, "
                    "then select it again."
                ),
            )
        return None

    reason = _manifest_ineligible_reason(view)
    if reason is not None:
        raise BackendPluginSelectionError(
            f"Manifest backend '{view.record_id}' cannot be selected: {reason}",
            plugin_id=view.plugin_id,
            entry_point=view.entry_point_name,
            field="state",
            expected="validated-or-later and compatible",
            actual=(
                f"state={view.state or '<unknown>'}; "
                f"compatibility={view.compatibility_status or '<unknown>'}"
            ),
            remediation="Run Registry validation and resolve its diagnostics first.",
        )

    targets = tuple(getattr(view.manifest, "targets", ()) or ())
    if target_name not in targets:
        raise BackendPluginSelectionError(
            f"Backend '{view.record_id}' does not declare target '{target_name}'",
            plugin_id=view.plugin_id,
            entry_point=view.entry_point_name,
            field="targets",
            expected=target_name,
            actual=", ".join(sorted(targets)) or "<empty>",
            remediation="Select a backend whose Manifest declares this target.",
        )

    report = _capability_report(
        view,
        core_provided=core_provided,
        kernel_required=kernel_required,
    )
    if not report.compatible:
        validate_plugin_capabilities(
            view.manifest,
            core_provided=core_provided,
            kernel_required=kernel_required,
        )
    return report


def _decision(
    view: _RecordView,
    *,
    target_name: str,
    method: SelectionMethod,
    selector: Optional[str],
    candidate_record_ids: Tuple[str, ...],
    capability_report: Optional[CapabilityReport],
) -> SelectionDecision:
    return SelectionDecision(
        record_id=view.record_id,
        registry_key=view.registry_key,
        plugin_id=view.plugin_id,
        entry_point_name=view.entry_point_name,
        target=target_name,
        method=method,
        selector=selector,
        priority=view.priority,
        is_legacy=view.is_legacy,
        candidate_record_ids=candidate_record_ids,
        capability_report=capability_report,
        record=view.record,
    )


def select_backend(
    records: Iterable[Any],
    *,
    target: Any,
    kernel_required_capabilities: Iterable[str] = (),
    core_provided_capabilities: Iterable[str] = (),
    explicit_selector: Optional[str] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> SelectionDecision:
    """Select one backend using the frozen W8 precedence.

    Precedence is:

    1. explicit Python selector;
    2. ``TRITON_ANCHOR_BACKEND`` from the supplied environment mapping;
    3. the sole compatible Manifest candidate;
    4. the unique highest Manifest priority.

    Legacy records never participate in automatic selection.  They can only
    be selected by exact ``record_id``/``registry_key`` and cannot be used
    when the kernel declares capability requirements.
    """
    try:
        record_tuple = tuple(records)
    except TypeError as exc:
        raise BackendPluginSelectionError(
            "Backend selection records must be iterable",
            field="records",
            expected="an iterable of discovered/validated records",
            actual=repr(records),
            remediation="Pass Registry list/validate results to selection.",
        ) from exc

    # Identity conflicts are fatal regardless of selector or enumeration
    # order.  Target overlap deliberately remains for this function to resolve.
    try:
        detect_static_conflicts(record_tuple).raise_for_fatal()
    except (AttributeError, TypeError, ValueError) as exc:
        raise BackendPluginSelectionError(
            f"Backend records cannot be analyzed for conflicts: {exc}",
            field="records",
            expected="records with unique, readable record_id values",
            actual=str(exc),
            remediation="Repair or rediscover backend records before selection.",
        ) from exc

    views = tuple(
        sorted((_project_record(record) for record in record_tuple),
               key=lambda view: view.sort_key)
    )
    target_name = _target_name(target)
    core_provided = _capability_names(
        core_provided_capabilities, "core_provided_capabilities"
    )
    kernel_required = _capability_names(
        kernel_required_capabilities, "kernel_required_capabilities"
    )

    if environment is None:
        environment = {}
    if not isinstance(environment, Mapping):
        raise BackendPluginSelectionError(
            "Selection environment must be a mapping",
            field="environment",
            expected="a mapping of environment variable names to strings",
            actual=type(environment).__name__,
            remediation="Pass os.environ or another read-only mapping.",
        )

    selector = None
    method = None
    if explicit_selector is not None:
        selector = _non_empty_string(explicit_selector, "explicit_selector")
        method = SelectionMethod.PYTHON_EXPLICIT
    else:
        # A Python selector fully overrides the environment; an ignored
        # environment value cannot make the higher-precedence choice fail.
        environment_selector = environment.get(BACKEND_SELECTOR_ENV)
        if environment_selector == "":
            environment_selector = None
        if environment_selector is not None and not isinstance(
            environment_selector, str
        ):
            raise BackendPluginSelectionError(
                f"{BACKEND_SELECTOR_ENV} must be a string",
                field=BACKEND_SELECTOR_ENV,
                expected="a backend selector string",
                actual=repr(environment_selector),
                remediation=f"Set or remove {BACKEND_SELECTOR_ENV}.",
            )
        if environment_selector is not None:
            selector = _non_empty_string(
                environment_selector, BACKEND_SELECTOR_ENV
            )
            method = SelectionMethod.ENVIRONMENT

    eligible = []
    reports: Dict[str, CapabilityReport] = {}
    incompatible_reports: Dict[str, Tuple[_RecordView, CapabilityReport]] = {}
    for view in views:
        if view.is_legacy or _manifest_ineligible_reason(view) is not None:
            continue
        targets = tuple(getattr(view.manifest, "targets", ()) or ())
        if target_name not in targets:
            continue
        report = _capability_report(
            view,
            core_provided=core_provided,
            kernel_required=kernel_required,
        )
        if report.compatible:
            eligible.append(view)
            reports[view.record_id] = report
        else:
            incompatible_reports[view.record_id] = (view, report)

    eligible_tuple = tuple(sorted(eligible, key=lambda view: view.sort_key))
    eligible_ids = tuple(view.record_id for view in eligible_tuple)

    if selector is not None:
        selected = _matching_selector(views, selector)
        report = _selected_view(
            selected,
            target_name=target_name,
            core_provided=core_provided,
            kernel_required=kernel_required,
        )
        candidates = eligible_ids
        if selected.record_id not in candidates:
            candidates = tuple(sorted(candidates + (selected.record_id,)))
        return _decision(
            selected,
            target_name=target_name,
            method=method,
            selector=selector,
            candidate_record_ids=candidates,
            capability_report=report,
        )

    if not eligible_tuple:
        if incompatible_reports:
            first_record_id = sorted(incompatible_reports)[0]
            first_view, _ = incompatible_reports[first_record_id]
            # Use W6's public validator so its structured error contract stays
            # authoritative instead of duplicating error construction here.
            validate_plugin_capabilities(
                first_view.manifest,
                core_provided=core_provided,
                kernel_required=kernel_required,
            )
            raise AssertionError("incompatible capability report did not raise")
        rejected = []
        for view in views:
            if view.state != "rejected" or view.manifest is None:
                continue
            targets = tuple(getattr(view.manifest, "targets", ()) or ())
            error = getattr(view.record, "error", None)
            if (
                target_name in targets
                and isinstance(error, BackendPluginError)
            ):
                rejected.append((view.record_id, error))
        if rejected:
            raise sorted(rejected, key=lambda item: item[0])[0][1]
        legacy_ids = tuple(
            view.record_id
            for view in views
            if view.is_legacy and _legacy_ineligible_reason(view) is None
        )
        detail = (
            " Legacy records require explicit record_id/registry_key: "
            + ", ".join(legacy_ids)
            if legacy_ids
            else ""
        )
        raise BackendPluginSelectionError(
            f"No compatible Manifest backend declares target '{target_name}'."
            + detail,
            field="targets",
            expected=target_name,
            actual="<no compatible candidate>",
            remediation=(
                "Install a compatible Manifest backend, or explicitly select "
                "a Legacy record if no kernel capabilities are required."
            ),
        )

    if len(eligible_tuple) == 1:
        selected = eligible_tuple[0]
        return _decision(
            selected,
            target_name=target_name,
            method=SelectionMethod.SOLE_CANDIDATE,
            selector=None,
            candidate_record_ids=eligible_ids,
            capability_report=reports[selected.record_id],
        )

    highest_priority = max(view.priority for view in eligible_tuple)
    highest = tuple(
        view for view in eligible_tuple if view.priority == highest_priority
    )
    if len(highest) != 1:
        record_ids = tuple(view.record_id for view in highest)
        raise BackendPluginSelectionError(
            f"Backend selection for target '{target_name}' is ambiguous at "
            f"priority {highest_priority}: "
            + ", ".join(record_ids),
            field="priority",
            expected="one unique highest-priority candidate",
            actual=", ".join(record_ids),
            remediation=(
                "Select a plugin explicitly with the Python API or "
                f"{BACKEND_SELECTOR_ENV}."
            ),
        )

    selected = highest[0]
    return _decision(
        selected,
        target_name=target_name,
        method=SelectionMethod.MANIFEST_PRIORITY,
        selector=None,
        candidate_record_ids=eligible_ids,
        capability_report=reports[selected.record_id],
    )


select_backend_plugin = select_backend


__all__ = [
    "BACKEND_SELECTOR_ENV",
    "SelectionDecision",
    "SelectionMethod",
    "select_backend",
    "select_backend_plugin",
]
