"""Stable request and diagnostic schema for structured AnchorIR validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Tuple


def _require_utf8_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError("%s must be a string" % field_name)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("%s must be valid UTF-8 encodable Unicode" % field_name) from error
    return value


def _escape_text_field(value: str) -> str:
    """Render untrusted report text as one terminal-safe visible field."""

    rendered = []
    for character in value:
        codepoint = ord(character)
        if character == "\\":
            rendered.append("\\\\")
        elif character == "\n":
            rendered.append("\\n")
        elif character == "\r":
            rendered.append("\\r")
        elif character == "\t":
            rendered.append("\\t")
        elif not character.isprintable():
            if codepoint <= 0xFF:
                rendered.append("\\x%02X" % codepoint)
            elif codepoint <= 0xFFFF:
                rendered.append("\\u%04X" % codepoint)
            else:
                rendered.append("\\U%08X" % codepoint)
        else:
            rendered.append(character)
    return "".join(rendered)


class AnchorIRTrack(str, Enum):
    """The two supported AnchorIR output contracts."""

    LINALG = "linalg"
    TRITON_GPU = "triton_gpu"


class AnchorIRPhase(str, Enum):
    """Validation points around the backend AnchorIR hook."""

    PRE_HOOK = "pre_hook"
    POST_HOOK = "post_hook"


class AnchorIRObjectKind(str, Enum):
    """Kind of request or IR object identified by a diagnostic."""

    REQUEST = "request"
    MODULE = "module"
    OPERATION = "operation"
    TYPE = "type"
    ATTRIBUTE = "attribute"
    REGION = "region"
    BLOCK = "block"


@dataclass(frozen=True)
class AnchorIRLocation:
    """Source location when MLIR preserves a FileLineColLoc."""

    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None

    def __post_init__(self) -> None:
        if self.file is not None:
            _require_utf8_string(self.file, "AnchorIRLocation.file")
        for field_name in ("line", "column"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(
                    "AnchorIRLocation.%s must be a positive integer or None"
                    % field_name
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AnchorIRLocation":
        required = {"file", "line", "column"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(
                "location dictionary must contain exactly: %s"
                % ", ".join(sorted(required))
            )
        return cls(
            file=value.get("file"),
            line=value.get("line"),
            column=value.get("column"),
        )


def _enum_value(value: Optional[Enum]) -> Optional[str]:
    return None if value is None else str(value.value)


@dataclass(frozen=True)
class AnchorIRDiagnostic:
    """One stable, machine-readable AnchorIR validation failure."""

    code: str
    severity: str
    message: str
    hint: str
    spec_version: str
    track: Optional[AnchorIRTrack]
    phase: Optional[AnchorIRPhase]
    object_kind: AnchorIRObjectKind
    object_name: str
    operation_path: str = ""
    object_path: str = ""
    location: Optional[AnchorIRLocation] = None

    def __post_init__(self) -> None:
        for field_name in (
            "code",
            "severity",
            "message",
            "hint",
            "spec_version",
            "object_name",
            "operation_path",
            "object_path",
        ):
            value = getattr(self, field_name)
            _require_utf8_string(value, "AnchorIRDiagnostic.%s" % field_name)
        for field_name in ("code", "severity", "message", "hint", "object_name"):
            if not getattr(self, field_name):
                raise ValueError("AnchorIRDiagnostic.%s must be non-empty" % field_name)
        if self.track is not None and not isinstance(self.track, AnchorIRTrack):
            raise TypeError("AnchorIRDiagnostic.track must be AnchorIRTrack or None")
        if self.phase is not None and not isinstance(self.phase, AnchorIRPhase):
            raise TypeError("AnchorIRDiagnostic.phase must be AnchorIRPhase or None")
        if not isinstance(self.object_kind, AnchorIRObjectKind):
            raise TypeError("AnchorIRDiagnostic.object_kind must be AnchorIRObjectKind")
        if self.location is not None and not isinstance(
            self.location, AnchorIRLocation
        ):
            raise TypeError(
                "AnchorIRDiagnostic.location must be AnchorIRLocation or None"
            )

    def sort_key(self) -> Tuple[Any, ...]:
        """Return a total ordering independent of hash/set iteration order."""

        if self.object_kind == AnchorIRObjectKind.REQUEST:
            return (0, self.code, self.object_name, self.message, self.hint)

        location = self.location
        return (
            1,
            "" if location is None or location.file is None else location.file,
            -1 if location is None or location.line is None else location.line,
            -1 if location is None or location.column is None else location.column,
            self.operation_path,
            self.object_path,
            self.object_kind.value,
            self.object_name,
            self.code,
            self.message,
            self.hint,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "hint": self.hint,
            "spec_version": self.spec_version,
            "track": _enum_value(self.track),
            "phase": _enum_value(self.phase),
            "object_kind": self.object_kind.value,
            "object_name": self.object_name,
            "operation_path": self.operation_path,
            "object_path": self.object_path,
            "location": None if self.location is None else self.location.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AnchorIRDiagnostic":
        required = {
            "code",
            "severity",
            "message",
            "hint",
            "spec_version",
            "track",
            "phase",
            "object_kind",
            "object_name",
            "operation_path",
            "object_path",
            "location",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(
                "diagnostic dictionary must contain exactly: %s"
                % ", ".join(sorted(required))
            )
        raw_location = value.get("location")
        raw_track = value.get("track")
        raw_phase = value.get("phase")
        return cls(
            code=value["code"],
            severity=value["severity"],
            message=value["message"],
            hint=value["hint"],
            spec_version=value["spec_version"],
            track=None if raw_track is None else AnchorIRTrack(raw_track),
            phase=None if raw_phase is None else AnchorIRPhase(raw_phase),
            object_kind=AnchorIRObjectKind(value["object_kind"]),
            object_name=value["object_name"],
            operation_path=value["operation_path"],
            object_path=value["object_path"],
            location=(
                None
                if raw_location is None
                else AnchorIRLocation.from_dict(raw_location)
            ),
        )


@dataclass(frozen=True)
class AnchorIRValidationReport:
    """The sole result model shared by the C++ core, Python and CLI."""

    spec_version: str
    track: Optional[AnchorIRTrack]
    phase: Optional[AnchorIRPhase]
    diagnostics: Tuple[AnchorIRDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _require_utf8_string(
            self.spec_version,
            "AnchorIRValidationReport.spec_version",
        )
        if self.track is not None and not isinstance(self.track, AnchorIRTrack):
            raise TypeError(
                "AnchorIRValidationReport.track must be AnchorIRTrack or None"
            )
        if self.phase is not None and not isinstance(self.phase, AnchorIRPhase):
            raise TypeError(
                "AnchorIRValidationReport.phase must be AnchorIRPhase or None"
            )
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, AnchorIRDiagnostic) for item in diagnostics):
            raise TypeError(
                "AnchorIRValidationReport.diagnostics must contain diagnostics"
            )
        for diagnostic in diagnostics:
            if (
                diagnostic.spec_version != self.spec_version
                or diagnostic.track != self.track
                or diagnostic.phase != self.phase
            ):
                raise ValueError(
                    "diagnostic request identity does not match its report"
                )
        ordered = tuple(sorted(diagnostics, key=lambda item: item.sort_key()))
        object.__setattr__(self, "diagnostics", ordered)

    @property
    def valid(self) -> bool:
        return not self.diagnostics

    @classmethod
    def build(
        cls,
        *,
        spec_version: str,
        track: Optional[AnchorIRTrack],
        phase: Optional[AnchorIRPhase],
        diagnostics: Iterable[AnchorIRDiagnostic] = (),
    ) -> "AnchorIRValidationReport":
        return cls(
            spec_version=spec_version,
            track=track,
            phase=phase,
            diagnostics=tuple(diagnostics),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "spec_version": self.spec_version,
            "track": _enum_value(self.track),
            "phase": _enum_value(self.phase),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AnchorIRValidationReport":
        required = {
            "valid",
            "spec_version",
            "track",
            "phase",
            "diagnostics",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(
                "report dictionary must contain exactly: %s"
                % ", ".join(sorted(required))
            )
        if type(value["valid"]) is not bool:
            raise TypeError("report.valid must be a boolean")
        if not isinstance(value["diagnostics"], (list, tuple)):
            raise TypeError("report.diagnostics must be a list")
        raw_track = value.get("track")
        raw_phase = value.get("phase")
        report = cls.build(
            spec_version=value["spec_version"],
            track=None if raw_track is None else AnchorIRTrack(raw_track),
            phase=None if raw_phase is None else AnchorIRPhase(raw_phase),
            diagnostics=(
                AnchorIRDiagnostic.from_dict(item) for item in value["diagnostics"]
            ),
        )
        if value["valid"] != report.valid:
            raise ValueError("report.valid does not match report diagnostics")
        return report


def format_anchor_ir_validation_report(report: AnchorIRValidationReport) -> str:
    """Render one deterministic, actionable validation report."""

    if not isinstance(report, AnchorIRValidationReport):
        raise TypeError("report must be an AnchorIRValidationReport")
    track = "-" if report.track is None else report.track.value
    phase = "-" if report.phase is None else report.phase.value
    lines = [
        "AnchorIR validation: %s" % ("PASS" if report.valid else "FAIL"),
        "spec_version: %s" % _escape_text_field(report.spec_version),
        "track: %s" % track,
        "phase: %s" % phase,
        "diagnostics: %d" % len(report.diagnostics),
    ]
    for diagnostic in report.diagnostics:
        lines.extend(
            [
                "",
                "[%s] %s"
                % (
                    _escape_text_field(diagnostic.code),
                    _escape_text_field(diagnostic.message),
                ),
                "  object: %s %s"
                % (
                    diagnostic.object_kind.value,
                    _escape_text_field(diagnostic.object_name),
                ),
            ]
        )
        if diagnostic.operation_path:
            lines.append(
                "  operation_path: %s"
                % _escape_text_field(diagnostic.operation_path)
            )
        if diagnostic.object_path:
            lines.append(
                "  object_path: %s" % _escape_text_field(diagnostic.object_path)
            )
        if diagnostic.location is not None:
            location = diagnostic.location
            rendered = (
                "<unknown>"
                if location.file is None
                else _escape_text_field(location.file)
            )
            if location.line is not None:
                rendered += ":%d" % location.line
                if location.column is not None:
                    rendered += ":%d" % location.column
            lines.append("  location: %s" % rendered)
        lines.append("  hint: %s" % _escape_text_field(diagnostic.hint))
    return "\n".join(lines)
