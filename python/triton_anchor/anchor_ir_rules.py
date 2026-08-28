"""Versioned AnchorIR policy loader and request diagnostics."""

from __future__ import annotations

import json
import pkgutil
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple, Union

from .anchor_ir_schema import (
    AnchorIRDiagnostic,
    AnchorIRObjectKind,
    AnchorIRPhase,
    AnchorIRTrack,
    AnchorIRValidationReport,
)

ANCHOR_IR_SPEC_VERSION = "anchor-ir/1.1.0"
_RULE_RESOURCES = {
    "anchor-ir/1.0.0": "spec/anchor-ir-1.0.0.json",
    ANCHOR_IR_SPEC_VERSION: "spec/anchor-ir-1.1.0.json",
}
SUPPORTED_SPEC_VERSIONS: Tuple[str, ...] = tuple(sorted(_RULE_RESOURCES))
_DIALECT_NAMESPACE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INVARIANT_NAME = re.compile(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+")
_DIAGNOSTIC_CODE = re.compile(r"AIR-[A-Z]+-[0-9]{3}")
_IMPLEMENTED_INVARIANTS = MappingProxyType(
    {
        AnchorIRTrack.LINALG.value: frozenset(
            {
                "linalg.no_unrealized_conversion_cast",
                "linalg.ranked_shaped_values",
                "linalg.generic_region_contract",
            }
        ),
        AnchorIRTrack.TRITON_GPU.value: frozenset(
            {
                "gpu.tensor_encoding",
                "gpu.module_configuration",
                "gpu.encoding_rank",
                "gpu.encoding_components",
                "gpu.shaped_element_type",
                "gpu.operation_contract",
                "gpu.dot_encoding_contract",
            }
        ),
    }
)


@dataclass(frozen=True)
class AnchorIRDiagnosticTemplate:
    """Versioned message contract for a policy failure."""

    code: str
    message: str
    hint: str

    def render(
        self,
        *,
        dialect: str,
        operation: str,
        object_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        values = {
            "dialect": dialect,
            "operation": operation,
            "object": operation if object_name is None else object_name,
        }
        return self.message.format(**values), self.hint.format(**values)

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class AnchorIRPolicy:
    """Immutable policy resolved for one version, Track and Phase."""

    spec_version: str
    track: AnchorIRTrack
    phase: AnchorIRPhase
    core_allowed_dialects: FrozenSet[str]
    allowed_dialects: FrozenSet[str]
    extension_dialects: FrozenSet[str]
    forbidden_dialects: FrozenSet[str]
    enabled_invariants: Tuple[str, ...]
    semantic_diagnostics: Mapping[str, AnchorIRDiagnosticTemplate]
    unknown_dialect_diagnostic: AnchorIRDiagnosticTemplate
    forbidden_dialect_diagnostic: AnchorIRDiagnosticTemplate
    unknown_type_diagnostic: AnchorIRDiagnosticTemplate
    forbidden_type_diagnostic: AnchorIRDiagnosticTemplate
    unknown_attribute_diagnostic: AnchorIRDiagnosticTemplate
    forbidden_attribute_diagnostic: AnchorIRDiagnosticTemplate
    resource_limit_diagnostic: AnchorIRDiagnosticTemplate
    parse_failure_diagnostic: AnchorIRDiagnosticTemplate
    verify_failure_diagnostic: AnchorIRDiagnosticTemplate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "track": self.track.value,
            "phase": self.phase.value,
            "core_allowed_dialects": sorted(self.core_allowed_dialects),
            "allowed_dialects": sorted(self.allowed_dialects),
            "extension_dialects": sorted(self.extension_dialects),
            "forbidden_dialects": sorted(self.forbidden_dialects),
            "enabled_invariants": list(self.enabled_invariants),
            "semantic_diagnostics": {
                name: diagnostic.to_dict()
                for name, diagnostic in self.semantic_diagnostics.items()
            },
            "unknown_dialect_diagnostic": (self.unknown_dialect_diagnostic.to_dict()),
            "forbidden_dialect_diagnostic": (
                self.forbidden_dialect_diagnostic.to_dict()
            ),
            "unknown_type_diagnostic": (self.unknown_type_diagnostic.to_dict()),
            "forbidden_type_diagnostic": (self.forbidden_type_diagnostic.to_dict()),
            "unknown_attribute_diagnostic": (
                self.unknown_attribute_diagnostic.to_dict()
            ),
            "forbidden_attribute_diagnostic": (
                self.forbidden_attribute_diagnostic.to_dict()
            ),
            "resource_limit_diagnostic": (
                self.resource_limit_diagnostic.to_dict()
            ),
            "parse_failure_diagnostic": (self.parse_failure_diagnostic.to_dict()),
            "verify_failure_diagnostic": (self.verify_failure_diagnostic.to_dict()),
        }


class AnchorIRPolicyError(ValueError):
    """Raised when a policy request cannot be resolved."""

    def __init__(self, report: AnchorIRValidationReport):
        self.report = report
        first = report.diagnostics[0]
        super().__init__("%s: %s" % (first.code, first.message))


def _request_diagnostic(
    *,
    code: str,
    message: str,
    hint: str,
    spec_version: str,
    track: Optional[AnchorIRTrack],
    phase: Optional[AnchorIRPhase],
    object_name: str,
) -> AnchorIRDiagnostic:
    return AnchorIRDiagnostic(
        code=code,
        severity="error",
        message=message,
        hint=hint,
        spec_version=spec_version,
        track=track,
        phase=phase,
        object_kind=AnchorIRObjectKind.REQUEST,
        object_name=object_name,
    )


def _parse_track(
    value: Union[AnchorIRTrack, str, None],
) -> Tuple[Optional[AnchorIRTrack], Optional[str]]:
    if value is None or value == "":
        return None, "missing"
    try:
        return AnchorIRTrack(value), None
    except (TypeError, ValueError):
        return None, "unsupported"


def _parse_phase(
    value: Union[AnchorIRPhase, str, None],
) -> Tuple[Optional[AnchorIRPhase], Optional[str]]:
    if value is None or value == "":
        return None, "missing"
    try:
        return AnchorIRPhase(value), None
    except (TypeError, ValueError):
        return None, "unsupported"


def validate_policy_request(
    *,
    spec_version: Optional[str],
    track: Union[AnchorIRTrack, str, None],
    phase: Union[AnchorIRPhase, str, None],
) -> AnchorIRValidationReport:
    """Validate policy identity and return stable request diagnostics."""

    raw_version = "" if spec_version is None else str(spec_version)
    parsed_track, track_error = _parse_track(track)
    parsed_phase, phase_error = _parse_phase(phase)
    diagnostics = []

    if not raw_version:
        diagnostics.append(
            _request_diagnostic(
                code="AIR-REQUEST-001",
                message="AnchorIR spec_version is required",
                hint=(
                    "Pass an installed full version such as "
                    "'%s'." % ANCHOR_IR_SPEC_VERSION
                ),
                spec_version=raw_version,
                track=parsed_track,
                phase=parsed_phase,
                object_name="spec_version",
            )
        )
    elif raw_version not in _RULE_RESOURCES:
        diagnostics.append(
            _request_diagnostic(
                code="AIR-REQUEST-002",
                message="Unsupported AnchorIR spec_version '%s'" % raw_version,
                hint=(
                    "Use one of the installed versions: %s."
                    % ", ".join(SUPPORTED_SPEC_VERSIONS)
                ),
                spec_version=raw_version,
                track=parsed_track,
                phase=parsed_phase,
                object_name=raw_version,
            )
        )

    if track_error == "missing":
        diagnostics.append(
            _request_diagnostic(
                code="AIR-REQUEST-003",
                message="AnchorIR track is required",
                hint="Pass track='linalg' or track='triton_gpu'.",
                spec_version=raw_version,
                track=None,
                phase=parsed_phase,
                object_name="track",
            )
        )
    elif track_error == "unsupported":
        diagnostics.append(
            _request_diagnostic(
                code="AIR-REQUEST-004",
                message="Unsupported AnchorIR track '%s'" % track,
                hint="Pass track='linalg' or track='triton_gpu'.",
                spec_version=raw_version,
                track=None,
                phase=parsed_phase,
                object_name=str(track),
            )
        )

    if phase_error == "missing":
        diagnostics.append(
            _request_diagnostic(
                code="AIR-REQUEST-005",
                message="AnchorIR phase is required",
                hint="Pass phase='pre_hook' or phase='post_hook'.",
                spec_version=raw_version,
                track=parsed_track,
                phase=None,
                object_name="phase",
            )
        )
    elif phase_error == "unsupported":
        diagnostics.append(
            _request_diagnostic(
                code="AIR-REQUEST-006",
                message="Unsupported AnchorIR phase '%s'" % phase,
                hint="Pass phase='pre_hook' or phase='post_hook'.",
                spec_version=raw_version,
                track=parsed_track,
                phase=None,
                object_name=str(phase),
            )
        )

    return AnchorIRValidationReport.build(
        spec_version=raw_version,
        track=parsed_track,
        phase=parsed_phase,
        diagnostics=diagnostics,
    )


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("%s must be a JSON object" % field_name)
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    field_name: str,
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing: %s" % ", ".join(missing))
        if unexpected:
            details.append("unexpected: %s" % ", ".join(unexpected))
        raise RuntimeError(
            "%s must contain exactly the versioned schema fields (%s)"
            % (field_name, "; ".join(details))
        )


def _load_unique_string_list(
    value: Any,
    field_name: str,
    *,
    pattern: re.Pattern[str],
) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise RuntimeError("%s must be a JSON array" % field_name)
    if any(
        not isinstance(item, str) or pattern.fullmatch(item) is None for item in value
    ):
        raise RuntimeError("%s contains an invalid name" % field_name)
    if len(value) != len(set(value)):
        raise RuntimeError("%s must not contain duplicates" % field_name)
    return tuple(value)


def _load_rule_document(spec_version: str) -> Dict[str, Any]:
    resource_name = _RULE_RESOURCES[spec_version]
    raw = pkgutil.get_data("triton_anchor", resource_name)
    if raw is None:
        raise RuntimeError("AnchorIR rule resource is missing: %s" % resource_name)

    try:
        decoded = raw.decode("utf-8")
        def reject_duplicate_keys(
            pairs: list[tuple[str, Any]],
        ) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise RuntimeError(
                        "AnchorIR rule resource contains duplicate JSON key '%s'"
                        % key
                    )
                result[key] = value
            return result

        document = json.loads(decoded, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("AnchorIR rule resource is not valid UTF-8 JSON") from error
    document = dict(_require_mapping(document, "AnchorIR rule document"))
    common_template_names = (
        "unknown_dialect_diagnostic",
        "unknown_type_diagnostic",
        "unknown_attribute_diagnostic",
        "resource_limit_diagnostic",
        "parse_failure_diagnostic",
        "verify_failure_diagnostic",
    )
    _require_exact_keys(
        document,
        "AnchorIR rule document",
        {
            "schema_version",
            "spec_version",
            "tracks",
            *common_template_names,
        },
    )
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
    ):
        raise RuntimeError("Unsupported AnchorIR rule schema")
    if document.get("spec_version") != spec_version:
        raise RuntimeError("AnchorIR rule resource version mismatch")
    tracks = _require_mapping(document.get("tracks"), "tracks")
    if set(tracks) != {item.value for item in AnchorIRTrack}:
        raise RuntimeError("AnchorIR rule resource must define exactly both Tracks")

    all_templates = [
        _load_diagnostic_template(_require_mapping(document.get(name), name))
        for name in common_template_names
    ]
    for track_name, raw_track_rules in tracks.items():
        track_rules = _require_mapping(raw_track_rules, "tracks.%s" % track_name)
        _require_exact_keys(
            track_rules,
            "tracks.%s" % track_name,
            {
                "allowed_dialects",
                "forbidden_dialects",
                "enabled_invariants",
                "semantic_diagnostics",
                "forbidden_dialect_diagnostic",
                "forbidden_type_diagnostic",
                "forbidden_attribute_diagnostic",
            },
        )
        allowed = _load_unique_string_list(
            track_rules.get("allowed_dialects"),
            "tracks.%s.allowed_dialects" % track_name,
            pattern=_DIALECT_NAMESPACE,
        )
        forbidden = _load_unique_string_list(
            track_rules.get("forbidden_dialects"),
            "tracks.%s.forbidden_dialects" % track_name,
            pattern=_DIALECT_NAMESPACE,
        )
        if set(allowed) & set(forbidden):
            raise RuntimeError(
                "AnchorIR allowed and forbidden dialect sets must be disjoint"
            )
        invariants = _load_unique_string_list(
            track_rules.get("enabled_invariants"),
            "tracks.%s.enabled_invariants" % track_name,
            pattern=_INVARIANT_NAME,
        )
        semantic = _require_mapping(
            track_rules.get("semantic_diagnostics"),
            "tracks.%s.semantic_diagnostics" % track_name,
        )
        if set(invariants) != set(semantic):
            raise RuntimeError(
                "Every enabled AnchorIR invariant must have exactly one diagnostic"
            )
        expected_prefix = "linalg." if track_name == "linalg" else "gpu."
        if any(not invariant.startswith(expected_prefix) for invariant in invariants):
            raise RuntimeError("AnchorIR invariant is assigned to the wrong Track")
        unsupported_invariants = set(invariants) - _IMPLEMENTED_INVARIANTS[track_name]
        if unsupported_invariants:
            raise RuntimeError(
                "AnchorIR invariant is not implemented by the structured "
                "Validator: %s" % ", ".join(sorted(unsupported_invariants))
            )
        all_templates.extend(
            _load_diagnostic_template(
                _require_mapping(
                    semantic[invariant],
                    "tracks.%s.semantic_diagnostics.%s" % (track_name, invariant),
                )
            )
            for invariant in invariants
        )
        for name in (
            "forbidden_dialect_diagnostic",
            "forbidden_type_diagnostic",
            "forbidden_attribute_diagnostic",
        ):
            all_templates.append(
                _load_diagnostic_template(
                    _require_mapping(
                        track_rules.get(name),
                        "tracks.%s.%s" % (track_name, name),
                    )
                )
            )
    codes = [template.code for template in all_templates]
    if len(codes) != len(set(codes)):
        raise RuntimeError("AnchorIR diagnostic codes must be globally unique")
    return document


def _load_diagnostic_template(
    value: Mapping[str, Any],
) -> AnchorIRDiagnosticTemplate:
    _require_exact_keys(
        value,
        "AnchorIR diagnostic template",
        {"code", "message", "hint"},
    )
    try:
        template = AnchorIRDiagnosticTemplate(
            code=value["code"],
            message=value["message"],
            hint=value["hint"],
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError("Invalid AnchorIR diagnostic template") from error
    if (
        not isinstance(template.code, str)
        or _DIAGNOSTIC_CODE.fullmatch(template.code) is None
        or not isinstance(template.message, str)
        or not template.message
        or not isinstance(template.hint, str)
        or not template.hint
    ):
        raise RuntimeError(
            "AnchorIR diagnostic template fields must be well-formed non-empty strings"
        )
    for field_name in ("code", "message", "hint"):
        try:
            getattr(template, field_name).encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise RuntimeError(
                "AnchorIR diagnostic template fields must be valid UTF-8"
            ) from error
    for rendered_field in (template.message, template.hint):
        remainder = rendered_field
        for placeholder in ("{dialect}", "{operation}", "{object}"):
            remainder = remainder.replace(placeholder, "")
        if "{" in remainder or "}" in remainder:
            raise RuntimeError(
                "AnchorIR diagnostic template contains an unsupported placeholder"
            )
    try:
        template.message.format(dialect="", operation="", object="")
        template.hint.format(dialect="", operation="", object="")
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "AnchorIR diagnostic template contains an unsupported placeholder"
        ) from error
    return template


def resolve_policy(
    *,
    spec_version: Optional[str],
    track: Union[AnchorIRTrack, str, None],
    phase: Union[AnchorIRPhase, str, None],
) -> AnchorIRPolicy:
    """Resolve an immutable policy or raise a report-carrying error."""

    report = validate_policy_request(
        spec_version=spec_version,
        track=track,
        phase=phase,
    )
    if not report.valid:
        raise AnchorIRPolicyError(report)

    assert report.track is not None
    assert report.phase is not None
    document = _load_rule_document(report.spec_version)
    track_rules = document["tracks"][report.track.value]
    allowed = frozenset(track_rules["allowed_dialects"])
    forbidden = frozenset(track_rules["forbidden_dialects"])
    enabled_invariants = tuple(track_rules["enabled_invariants"])
    semantic_diagnostics = {
        name: _load_diagnostic_template(value)
        for name, value in track_rules["semantic_diagnostics"].items()
    }

    return AnchorIRPolicy(
        spec_version=report.spec_version,
        track=report.track,
        phase=report.phase,
        core_allowed_dialects=allowed,
        allowed_dialects=allowed,
        extension_dialects=frozenset(),
        forbidden_dialects=forbidden,
        enabled_invariants=enabled_invariants,
        semantic_diagnostics=MappingProxyType(semantic_diagnostics),
        unknown_dialect_diagnostic=_load_diagnostic_template(
            document["unknown_dialect_diagnostic"]
        ),
        forbidden_dialect_diagnostic=_load_diagnostic_template(
            track_rules["forbidden_dialect_diagnostic"]
        ),
        unknown_type_diagnostic=_load_diagnostic_template(
            document["unknown_type_diagnostic"]
        ),
        forbidden_type_diagnostic=_load_diagnostic_template(
            track_rules["forbidden_type_diagnostic"]
        ),
        unknown_attribute_diagnostic=_load_diagnostic_template(
            document["unknown_attribute_diagnostic"]
        ),
        forbidden_attribute_diagnostic=_load_diagnostic_template(
            track_rules["forbidden_attribute_diagnostic"]
        ),
        resource_limit_diagnostic=_load_diagnostic_template(
            document["resource_limit_diagnostic"]
        ),
        parse_failure_diagnostic=_load_diagnostic_template(
            document["parse_failure_diagnostic"]
        ),
        verify_failure_diagnostic=_load_diagnostic_template(
            document["verify_failure_diagnostic"]
        ),
    )
