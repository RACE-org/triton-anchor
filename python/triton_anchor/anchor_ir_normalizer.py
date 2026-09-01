"""Versioned, validation-gated AnchorIR normalization and hashing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional, Union

from .anchor_ir_rules import validate_policy_request
from .anchor_ir_schema import (
    AnchorIRPhase,
    AnchorIRTrack,
    AnchorIRValidationReport,
)
from .anchor_ir_validator import StructuredAnchorIRValidator

ANCHOR_IR_NORMALIZATION_VERSION = "anchor-ir-normalization/1.0.0"
SUPPORTED_NORMALIZATION_VERSIONS = (ANCHOR_IR_NORMALIZATION_VERSION,)


@dataclass(frozen=True)
class AnchorIRNormalizationResult:
    """Normalized UTF-8 text and SHA-256, gated by strict validation."""

    normalization_version: str
    validation_report: AnchorIRValidationReport
    normalized_text: Optional[str]
    sha256: Optional[str]

    def __post_init__(self) -> None:
        if self.normalization_version not in SUPPORTED_NORMALIZATION_VERSIONS:
            raise ValueError(
                "unsupported AnchorIR normalization version %r"
                % self.normalization_version
            )
        if not isinstance(self.validation_report, AnchorIRValidationReport):
            raise TypeError("validation_report must be AnchorIRValidationReport")
        if not self.validation_report.valid:
            if self.normalized_text is not None or self.sha256 is not None:
                raise ValueError("invalid AnchorIR must not have normalized artifacts")
            return
        if not isinstance(self.normalized_text, str):
            raise TypeError("valid AnchorIR must have normalized text")
        if "\r" in self.normalized_text:
            raise ValueError("normalized AnchorIR must use LF line endings")
        if not self.normalized_text.endswith("\n") or self.normalized_text.endswith(
            "\n\n"
        ):
            raise ValueError(
                "normalized AnchorIR must have exactly one trailing newline"
            )
        encoded = self.normalized_text.encode("utf-8", errors="strict")
        expected_hash = hashlib.sha256(encoded).hexdigest()
        if self.sha256 != expected_hash:
            raise ValueError(
                "normalized AnchorIR SHA-256 does not match its UTF-8 text"
            )

    @property
    def acceptable(self) -> bool:
        """Whether this result may be stored or compared as a Golden."""

        return self.validation_report.valid

    @property
    def normalized_bytes(self) -> Optional[bytes]:
        if self.normalized_text is None:
            return None
        return self.normalized_text.encode("utf-8", errors="strict")

    def to_dict(self):
        return {
            "acceptable": self.acceptable,
            "normalization_version": self.normalization_version,
            "sha256": self.sha256,
            "normalized_text": self.normalized_text,
            "validation_report": self.validation_report.to_dict(),
        }


class AnchorIRNormalizer:
    """Normalize only IR accepted by the versioned structured Validator."""

    @staticmethod
    def _request_report(
        *,
        spec_version: Optional[str],
        track: Union[AnchorIRTrack, str, None],
        phase: Union[AnchorIRPhase, str, None],
    ) -> AnchorIRValidationReport:
        return validate_policy_request(
            spec_version=spec_version,
            track=track,
            phase=phase,
        )

    @staticmethod
    def _result(
        *,
        normalization_version: str,
        validation_report: AnchorIRValidationReport,
        normalized_text: Optional[str],
    ) -> AnchorIRNormalizationResult:
        digest = (
            None
            if normalized_text is None
            else hashlib.sha256(
                normalized_text.encode("utf-8", errors="strict")
            ).hexdigest()
        )
        return AnchorIRNormalizationResult(
            normalization_version=normalization_version,
            validation_report=validation_report,
            normalized_text=normalized_text,
            sha256=digest,
        )

    @staticmethod
    def _check_normalization_version(normalization_version: str) -> None:
        if normalization_version not in SUPPORTED_NORMALIZATION_VERSIONS:
            raise ValueError(
                "unsupported AnchorIR normalization version %r; supported: %s"
                % (
                    normalization_version,
                    ", ".join(SUPPORTED_NORMALIZATION_VERSIONS),
                )
            )

    def normalize_module(
        self,
        module,
        *,
        normalization_version: str,
        spec_version: Optional[str],
        track: Union[AnchorIRTrack, str, None],
        phase: Union[AnchorIRPhase, str, None],
        extension_dialects: Optional[Iterable[str]] = None,
    ) -> AnchorIRNormalizationResult:
        """Validate and normalize a real ModuleOp."""

        self._check_normalization_version(normalization_version)
        request = self._request_report(
            spec_version=spec_version,
            track=track,
            phase=phase,
        )
        if not request.valid:
            return self._result(
                normalization_version=normalization_version,
                validation_report=request,
                normalized_text=None,
            )
        policy = StructuredAnchorIRValidator._resolve_policy(
            spec_version=spec_version,
            track=track,
            phase=phase,
            extension_dialects=extension_dialects,
        )
        from triton._C.libtriton import anchor

        raw = anchor.normalize_anchor_ir(module, policy.to_dict())
        report = AnchorIRValidationReport.from_dict(raw["validation_report"])
        return self._result(
            normalization_version=normalization_version,
            validation_report=report,
            normalized_text=raw["normalized_text"],
        )

    def normalize_text(
        self,
        ir_text: str,
        *,
        normalization_version: str,
        spec_version: Optional[str],
        track: Union[AnchorIRTrack, str, None],
        phase: Union[AnchorIRPhase, str, None],
        context=None,
        source_name: str = "<anchor-ir>",
        extension_dialects: Optional[Iterable[str]] = None,
        _worker_timeout_seconds: Optional[float] = None,
    ) -> AnchorIRNormalizationResult:
        """Validate and normalize MLIR text through the same C++ core."""

        self._check_normalization_version(normalization_version)
        request = self._request_report(
            spec_version=spec_version,
            track=track,
            phase=phase,
        )
        if not request.valid:
            return self._result(
                normalization_version=normalization_version,
                validation_report=request,
                normalized_text=None,
            )
        policy = StructuredAnchorIRValidator._resolve_policy(
            spec_version=spec_version,
            track=track,
            phase=phase,
            extension_dialects=extension_dialects,
        )
        policy_dict = policy.to_dict()
        from ._anchor_ir_text_isolation import (
            lock_explicit_anchor_ir_context,
            text_input_limit_report,
            utf8_byte_length,
            validate_source_name,
            run_isolated_native_text,
        )

        ir_text_length = utf8_byte_length(ir_text, "ir_text")
        validate_source_name(source_name)
        limit_report = text_input_limit_report(policy_dict, ir_text_length)
        if limit_report is not None:
            return self._result(
                normalization_version=normalization_version,
                validation_report=AnchorIRValidationReport.from_dict(limit_report),
                normalized_text=None,
            )

        if context is None:
            raw = run_isolated_native_text(
                "normalize",
                ir_text,
                policy_dict,
                source_name,
                timeout_seconds=_worker_timeout_seconds,
            )
        else:
            from triton._C.libtriton import anchor, ir

            with lock_explicit_anchor_ir_context(context):
                ir.load_dialects(context)
                anchor.load_dialects(context)
                raw = anchor.normalize_anchor_ir_text(
                    ir_text,
                    context,
                    policy_dict,
                    source_name,
                )
        report = AnchorIRValidationReport.from_dict(raw["validation_report"])
        return self._result(
            normalization_version=normalization_version,
            validation_report=report,
            normalized_text=raw["normalized_text"],
        )
