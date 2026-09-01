"""Strict structured AnchorIR validation facade."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable, Optional, Union

from .anchor_ir_rules import resolve_policy, validate_policy_request
from .anchor_ir_schema import (
    AnchorIRPhase,
    AnchorIRTrack,
    AnchorIRValidationReport,
    format_anchor_ir_validation_report,
)


class AnchorIRValidationError(RuntimeError):
    """Fail-closed exception carrying the complete structured Report."""

    def __init__(self, report: AnchorIRValidationReport):
        if not isinstance(report, AnchorIRValidationReport):
            raise TypeError("report must be an AnchorIRValidationReport")
        if report.valid:
            raise ValueError("cannot raise AnchorIRValidationError for a valid report")
        self.report = report
        super().__init__(format_anchor_ir_validation_report(report))


class StructuredAnchorIRValidator:
    """Validate ModuleOp structure through the shared C++ MLIR core."""

    @staticmethod
    def _resolve_policy(
        *,
        spec_version: Optional[str],
        track: Union[AnchorIRTrack, str, None],
        phase: Union[AnchorIRPhase, str, None],
        extension_dialects: Optional[Iterable[str]],
    ):
        policy = resolve_policy(
            spec_version=spec_version,
            track=track,
            phase=phase,
        )
        if isinstance(extension_dialects, str):
            raise ValueError(
                "extension_dialects must be an iterable of dialect strings, not a string"
            )
        extensions = frozenset(extension_dialects or ())
        if not extensions:
            return policy
        if policy.phase is not AnchorIRPhase.POST_HOOK:
            raise ValueError(
                "extension_dialects are only valid for post_hook validation"
            )
        if any(
            not isinstance(dialect, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", dialect) is None
            for dialect in extensions
        ):
            raise ValueError(
                "extension_dialects must contain valid MLIR dialect namespaces"
            )

        # The C++ core checks forbidden namespaces before allowed namespaces.
        # Reject the declaration itself rather than waiting for forbidden IR to
        # appear: a backend must never advertise that it can override the core
        # contract.
        forbidden_extensions = extensions & policy.forbidden_dialects
        if forbidden_extensions:
            raise ValueError(
                "extension_dialects cannot include core-forbidden dialect(s): %s"
                % ", ".join(sorted(forbidden_extensions))
            )
        core_extensions = extensions & policy.allowed_dialects
        if core_extensions:
            raise ValueError(
                "extension_dialects cannot redeclare core dialect(s): %s"
                % ", ".join(sorted(core_extensions))
            )
        return replace(
            policy,
            allowed_dialects=policy.allowed_dialects | extensions,
            extension_dialects=extensions,
        )

    def validate_module(
        self,
        module,
        *,
        spec_version: Optional[str],
        track: Union[AnchorIRTrack, str, None],
        phase: Union[AnchorIRPhase, str, None],
        extension_dialects: Optional[Iterable[str]] = None,
    ) -> AnchorIRValidationReport:
        request = validate_policy_request(
            spec_version=spec_version,
            track=track,
            phase=phase,
        )
        if not request.valid:
            return request

        policy = self._resolve_policy(
            spec_version=spec_version,
            track=track,
            phase=phase,
            extension_dialects=extension_dialects,
        )
        from triton._C.libtriton import anchor

        raw_report = anchor.validate_anchor_ir(module, policy.to_dict())
        return AnchorIRValidationReport.from_dict(raw_report)

    def validate_text(
        self,
        ir_text: str,
        *,
        spec_version: Optional[str],
        track: Union[AnchorIRTrack, str, None],
        phase: Union[AnchorIRPhase, str, None],
        context=None,
        source_name: str = "<anchor-ir>",
        extension_dialects: Optional[Iterable[str]] = None,
    ) -> AnchorIRValidationReport:
        request = validate_policy_request(
            spec_version=spec_version,
            track=track,
            phase=phase,
        )
        if not request.valid:
            return request

        policy = self._resolve_policy(
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
            return AnchorIRValidationReport.from_dict(limit_report)

        # Default-context text is untrusted input and always runs out of
        # process: custom parsers in the pinned Triton/MLIR revision contain
        # native assertions that cannot be converted to Python exceptions.
        # An explicitly supplied context is the opt-in compatibility path for
        # dynamically registered vendor parsers.  The native parser entry has
        # its own preflight for every pinned crash class, so replacing that
        # context with a fresh worker context would incorrectly lose those
        # registrations.
        if context is None:
            raw_report = run_isolated_native_text(
                "validate",
                ir_text,
                policy_dict,
                source_name,
            )
        else:
            from triton._C.libtriton import anchor, ir

            with lock_explicit_anchor_ir_context(context):
                ir.load_dialects(context)
                anchor.load_dialects(context)
                raw_report = anchor.validate_anchor_ir_text(
                    ir_text,
                    context,
                    policy_dict,
                    source_name,
                )
        return AnchorIRValidationReport.from_dict(raw_report)
