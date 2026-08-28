"""Versioned AnchorIR stage manifests and first-divergence comparison."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .anchor_ir_normalizer import (
    ANCHOR_IR_NORMALIZATION_VERSION,
    SUPPORTED_NORMALIZATION_VERSIONS,
    AnchorIRNormalizer,
)
from .anchor_ir_rules import (
    ANCHOR_IR_SPEC_VERSION,
    SUPPORTED_SPEC_VERSIONS,
    resolve_policy,
)
from .anchor_ir_schema import (
    AnchorIRPhase,
    AnchorIRTrack,
    AnchorIRValidationReport,
)

ANCHOR_IR_GOLDEN_MANIFEST_VERSION = "anchor-ir-golden-manifest/1.0.0"
SUPPORTED_GOLDEN_MANIFEST_VERSIONS = (ANCHOR_IR_GOLDEN_MANIFEST_VERSION,)
# Golden manifests can be loaded from an untrusted regression artifact.  Bound
# the UTF-8 payload before json.loads() expands it into Python objects.
MAX_ANCHOR_IR_GOLDEN_MANIFEST_BYTES = 16 * 1024 * 1024
# ``json.loads`` recursively descends through arrays and objects.  Limit the
# textual nesting before decoding so a tiny but deeply nested untrusted
# manifest cannot escape the public Golden error domain as ``RecursionError``.
MAX_ANCHOR_IR_GOLDEN_MANIFEST_NESTING = 256
# Apply the same aggregate byte ceiling to direct Python construction and
# ``from_dict`` as ``from_json``.  Per-Stage text limits alone otherwise allow
# a compact Python sequence to multiply the accepted payload by Stage count.
MAX_ANCHOR_IR_GOLDEN_PAYLOAD_BYTES = 16 * 1024 * 1024
# Each serialized Stage is re-parsed in an isolated native worker. Bound the
# collection itself so one compact artifact cannot amplify into an unbounded
# number of subprocesses.
MAX_ANCHOR_IR_GOLDEN_STAGES = 256
# Default-context payloads run in isolated native workers.  Bound the entire
# first verification pass, not just each worker, so a sequence of individually
# successful but slow Stages cannot multiply the per-worker timeout.
MAX_ANCHOR_IR_GOLDEN_VERIFICATION_SECONDS = 60.0

_STAGE_COMPONENT = r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*"
_PASS_STAGE = re.compile(r"pass\.(?P<name>%s)\.after" % _STAGE_COMPONENT)
_HOOK_STAGE = re.compile(r"hook\.(?P<name>%s)\.after" % _STAGE_COMPONENT)
_DIALECT_NAMESPACE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class AnchorIRGoldenError(ValueError):
    """Base error for malformed or incompatible Golden data."""


class AnchorIRGoldenValidationError(AnchorIRGoldenError):
    """A Stage cannot become a Golden because strict validation rejected it."""

    def __init__(
        self,
        stage_id: "AnchorIRStageId",
        report: AnchorIRValidationReport,
    ):
        self.stage_id = stage_id
        self.report = report
        codes = ", ".join(item.code for item in report.diagnostics)
        super().__init__(
            "AnchorIR stage '%s' failed validation%s"
            % (stage_id.value, "" if not codes else ": " + codes)
        )


def _strict_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """Build one JSON object while rejecting duplicate member names."""

    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnchorIRGoldenError(
                "invalid Golden manifest JSON: duplicate key %r" % key
            )
        result[key] = value
    return result


def _check_manifest_json_nesting(value: str) -> None:
    """Reject deeply nested JSON before Python expands or recurses through it."""

    stack: List[str] = []
    in_string = False
    escaped = False
    matching = {"}": "{", "]": "["}
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "[{":
            if len(stack) >= MAX_ANCHOR_IR_GOLDEN_MANIFEST_NESTING:
                raise AnchorIRGoldenError(
                    "Golden manifest JSON exceeds the %d-level nesting limit"
                    % MAX_ANCHOR_IR_GOLDEN_MANIFEST_NESTING
                )
            stack.append(character)
            continue
        if character in "}]" and stack and stack[-1] == matching[character]:
            stack.pop()


@dataclass(frozen=True)
class AnchorIRStageId:
    """Stable, parseable identity for one AnchorIR pipeline observation."""

    value: str

    ADAPTER_OUTPUT = "adapter.output"
    POST_HOOK_BOUNDARY = "boundary.post_hook"

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("AnchorIRStageId.value must be a string")
        if (
            self.value not in (self.ADAPTER_OUTPUT, self.POST_HOOK_BOUNDARY)
            and _PASS_STAGE.fullmatch(self.value) is None
            and _HOOK_STAGE.fullmatch(self.value) is None
        ):
            raise AnchorIRGoldenError(
                "invalid AnchorIR Stage ID %r; expected adapter.output, "
                "pass.<stable-name>.after, hook.<stable-name>.after, or "
                "boundary.post_hook" % self.value
            )

    @classmethod
    def adapter_output(cls) -> "AnchorIRStageId":
        return cls(cls.ADAPTER_OUTPUT)

    @classmethod
    def after_pass(cls, stable_name: str) -> "AnchorIRStageId":
        return cls("pass.%s.after" % stable_name)

    @classmethod
    def after_hook(cls, stable_name: str) -> "AnchorIRStageId":
        return cls("hook.%s.after" % stable_name)

    @classmethod
    def post_hook_boundary(cls) -> "AnchorIRStageId":
        return cls(cls.POST_HOOK_BOUNDARY)

    @property
    def kind(self) -> str:
        if self.value == self.ADAPTER_OUTPUT:
            return "adapter"
        if self.value == self.POST_HOOK_BOUNDARY:
            return "boundary"
        if _PASS_STAGE.fullmatch(self.value):
            return "pass"
        return "hook"

    @property
    def phase(self) -> AnchorIRPhase:
        if self.kind in ("adapter", "pass"):
            return AnchorIRPhase.PRE_HOOK
        return AnchorIRPhase.POST_HOOK

    @property
    def order_group(self) -> int:
        return {
            "adapter": 0,
            "pass": 1,
            "hook": 2,
            "boundary": 3,
        }[self.kind]

    def __str__(self) -> str:
        return self.value


def _coerce_stage_id(
    value: Union[AnchorIRStageId, str],
) -> AnchorIRStageId:
    if isinstance(value, AnchorIRStageId):
        return value
    try:
        return AnchorIRStageId(value)
    except (TypeError, ValueError) as error:
        raise AnchorIRGoldenError("invalid Golden Stage ID %r" % (value,)) from error


def _coerce_phase(value: Union[AnchorIRPhase, str]) -> AnchorIRPhase:
    if isinstance(value, AnchorIRPhase):
        return value
    try:
        return AnchorIRPhase(value)
    except (TypeError, ValueError) as error:
        raise AnchorIRGoldenError("invalid Golden Stage phase %r" % (value,)) from error


def _coerce_track(value: Union[AnchorIRTrack, str]) -> AnchorIRTrack:
    if isinstance(value, AnchorIRTrack):
        return value
    try:
        return AnchorIRTrack(value)
    except (TypeError, ValueError) as error:
        raise AnchorIRGoldenError("invalid Golden track %r" % (value,)) from error


def _validate_extensions(
    extensions: Iterable[str],
    *,
    phase: AnchorIRPhase,
) -> Tuple[str, ...]:
    if isinstance(extensions, str):
        raise TypeError("extension_dialects must be an iterable, not a string")
    try:
        normalized = tuple(sorted(set(extensions)))
    except TypeError as error:
        raise TypeError("extension_dialects must be an iterable of strings") from error
    if any(
        not isinstance(dialect, str) or _DIALECT_NAMESPACE.fullmatch(dialect) is None
        for dialect in normalized
    ):
        raise AnchorIRGoldenError(
            "extension_dialects contains an invalid MLIR dialect namespace"
        )
    if normalized and phase is not AnchorIRPhase.POST_HOOK:
        raise AnchorIRGoldenError(
            "extension_dialects are only valid for post-hook Stages"
        )
    return normalized


@dataclass(frozen=True)
class AnchorIRGoldenStage:
    """One normalized Stage entry stored in a Golden manifest."""

    stage_id: AnchorIRStageId
    phase: AnchorIRPhase
    extension_dialects: Tuple[str, ...]
    sha256: str
    normalized_ir: str

    def __post_init__(self) -> None:
        stage_id = _coerce_stage_id(self.stage_id)
        object.__setattr__(self, "stage_id", stage_id)
        if not isinstance(self.phase, AnchorIRPhase):
            raise TypeError("AnchorIRGoldenStage.phase must be AnchorIRPhase")
        if self.phase is not stage_id.phase:
            raise AnchorIRGoldenError(
                "Stage '%s' requires phase '%s', not '%s'"
                % (stage_id.value, stage_id.phase.value, self.phase.value)
            )
        extensions = _validate_extensions(
            self.extension_dialects,
            phase=self.phase,
        )
        object.__setattr__(self, "extension_dialects", extensions)
        if not isinstance(self.normalized_ir, str):
            raise TypeError("AnchorIRGoldenStage.normalized_ir must be a string")
        if (
            "\r" in self.normalized_ir
            or not self.normalized_ir.endswith("\n")
            or self.normalized_ir.endswith("\n\n")
        ):
            raise AnchorIRGoldenError(
                "Golden normalized_ir must use LF and one trailing newline"
            )
        try:
            normalized_bytes = self.normalized_ir.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise AnchorIRGoldenError(
                "Golden normalized_ir must be valid UTF-8"
            ) from error
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise AnchorIRGoldenError(
                "AnchorIRGoldenStage.sha256 must be 64 lowercase hex digits"
            )
        actual_hash = hashlib.sha256(normalized_bytes).hexdigest()
        if self.sha256 != actual_hash:
            raise AnchorIRGoldenError(
                "Stage '%s' hash does not match normalized_ir" % stage_id.value
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id.value,
            "phase": self.phase.value,
            "extension_dialects": list(self.extension_dialects),
            "sha256": self.sha256,
            "normalized_ir": self.normalized_ir,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AnchorIRGoldenStage":
        required = {
            "stage_id",
            "phase",
            "extension_dialects",
            "sha256",
            "normalized_ir",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise AnchorIRGoldenError(
                "Golden Stage must contain exactly: %s" % ", ".join(sorted(required))
            )
        if not isinstance(value["extension_dialects"], (list, tuple)):
            raise TypeError("extension_dialects must be a list")
        return cls(
            stage_id=_coerce_stage_id(value["stage_id"]),
            phase=_coerce_phase(value["phase"]),
            extension_dialects=tuple(value["extension_dialects"]),
            sha256=value["sha256"],
            normalized_ir=value["normalized_ir"],
        )


def _validate_case_id(case_id: str) -> None:
    if not isinstance(case_id, str):
        raise TypeError("Golden case_id must be a string")
    try:
        case_id.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise AnchorIRGoldenError("Golden case_id must be valid UTF-8") from error
    if (
        not case_id
        or case_id.strip() != case_id
        or any(
            ord(character) < 32 or 0x7F <= ord(character) <= 0x9F
            for character in case_id
        )
    ):
        raise AnchorIRGoldenError(
            "Golden case_id must be non-empty and contain no surrounding "
            "whitespace or control characters"
        )


def _validate_stage_sequence(stages: Sequence[AnchorIRGoldenStage]) -> None:
    if len(stages) < 2:
        raise AnchorIRGoldenError(
            "Golden manifest requires adapter.output and boundary.post_hook"
        )
    if stages[0].stage_id.value != AnchorIRStageId.ADAPTER_OUTPUT:
        raise AnchorIRGoldenError("first Golden Stage must be adapter.output")
    if stages[-1].stage_id.value != AnchorIRStageId.POST_HOOK_BOUNDARY:
        raise AnchorIRGoldenError("last Golden Stage must be boundary.post_hook")
    stage_values = [stage.stage_id.value for stage in stages]
    if len(stage_values) != len(set(stage_values)):
        raise AnchorIRGoldenError("Golden Stage IDs must be unique")
    groups = [stage.stage_id.order_group for stage in stages]
    if groups != sorted(groups):
        raise AnchorIRGoldenError(
            "Golden Stages must follow adapter, pass, hook, boundary order"
        )

    hook_stages = [stage for stage in stages if stage.stage_id.kind == "hook"]
    if not hook_stages:
        if stages[-1].extension_dialects:
            raise AnchorIRGoldenError(
                "boundary.post_hook cannot declare extension dialects without "
                "a Hook Stage"
            )
        previous = stages[-2]
        if (
            previous.normalized_ir != stages[-1].normalized_ir
            or previous.sha256 != stages[-1].sha256
        ):
            raise AnchorIRGoldenError(
                "Stage immediately before boundary.post_hook must exactly match it"
            )
        return

    # A Hook stage is the direct observation immediately before the post-hook
    # boundary validation.  Validation itself does not rewrite IR, so a
    # differing boundary would conceal an unrecorded transformation.
    boundary = stages[-1]
    last_hook = hook_stages[-1]
    if (
        last_hook.normalized_ir != boundary.normalized_ir
        or last_hook.sha256 != boundary.sha256
    ):
        raise AnchorIRGoldenError(
            "last Hook Stage must exactly match boundary.post_hook"
        )
    post_hook_extensions = {stage.extension_dialects for stage in hook_stages}
    post_hook_extensions.add(boundary.extension_dialects)
    if len(post_hook_extensions) != 1:
        raise AnchorIRGoldenError(
            "all post-hook Stages must declare the same extension dialects"
        )


@dataclass(frozen=True)
class AnchorIRGoldenManifest:
    """Self-contained, deterministic manifest for one AnchorIR corpus case."""

    manifest_version: str
    case_id: str
    spec_version: str
    normalization_version: str
    track: AnchorIRTrack
    stages: Tuple[AnchorIRGoldenStage, ...]
    _payloads_verified: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.manifest_version not in SUPPORTED_GOLDEN_MANIFEST_VERSIONS:
            raise AnchorIRGoldenError(
                "unsupported Golden manifest version %r" % self.manifest_version
            )
        _validate_case_id(self.case_id)
        if self.spec_version not in SUPPORTED_SPEC_VERSIONS:
            raise AnchorIRGoldenError(
                "unsupported AnchorIR spec version %r" % self.spec_version
            )
        if self.normalization_version not in SUPPORTED_NORMALIZATION_VERSIONS:
            raise AnchorIRGoldenError(
                "unsupported normalization version %r" % self.normalization_version
            )
        if not isinstance(self.track, AnchorIRTrack):
            raise TypeError("Golden track must be AnchorIRTrack")
        stages = tuple(islice(iter(self.stages), MAX_ANCHOR_IR_GOLDEN_STAGES + 1))
        if len(stages) > MAX_ANCHOR_IR_GOLDEN_STAGES:
            raise AnchorIRGoldenError(
                "Golden manifest exceeds the %d-Stage limit"
                % MAX_ANCHOR_IR_GOLDEN_STAGES
            )
        if any(not isinstance(stage, AnchorIRGoldenStage) for stage in stages):
            raise TypeError("Golden stages must contain AnchorIRGoldenStage")
        payload_bytes = sum(
            len(stage.normalized_ir.encode("utf-8", errors="strict"))
            for stage in stages
        )
        if payload_bytes > MAX_ANCHOR_IR_GOLDEN_PAYLOAD_BYTES:
            raise AnchorIRGoldenError(
                "Golden manifest normalized IR exceeds the %d-byte total limit"
                % MAX_ANCHOR_IR_GOLDEN_PAYLOAD_BYTES
            )
        post_policy = resolve_policy(
            spec_version=self.spec_version,
            track=self.track,
            phase=AnchorIRPhase.POST_HOOK,
        )
        for stage in stages:
            forbidden_extensions = (
                set(stage.extension_dialects) & post_policy.forbidden_dialects
            )
            if forbidden_extensions:
                raise AnchorIRGoldenError(
                    "Golden Stage '%s' declares core-forbidden extension "
                    "dialect(s): %s"
                    % (
                        stage.stage_id.value,
                        ", ".join(sorted(forbidden_extensions)),
                    )
                )
            core_extensions = (
                set(stage.extension_dialects) & post_policy.allowed_dialects
            )
            if core_extensions:
                raise AnchorIRGoldenError(
                    "Golden Stage '%s' redeclares core dialect(s): %s"
                    % (stage.stage_id.value, ", ".join(sorted(core_extensions)))
                )
        _validate_stage_sequence(stages)
        object.__setattr__(self, "stages", stages)

    def _verify_stage_payloads(self, *, context=None) -> None:
        """Re-validate serialized Stage text before using it as a Golden."""

        if self._payloads_verified:
            return
        normalizer = AnchorIRNormalizer()
        verified_payloads = {}
        deadline = (
            None
            if context is not None
            else time.monotonic() + MAX_ANCHOR_IR_GOLDEN_VERIFICATION_SECONDS
        )
        for stage in self.stages:
            cache_key = (
                stage.phase,
                stage.extension_dialects,
                stage.sha256,
                stage.normalized_ir,
            )
            result = verified_payloads.get(cache_key)
            if result is None:
                worker_timeout = None
                if deadline is not None:
                    worker_timeout = deadline - time.monotonic()
                    if worker_timeout <= 0:
                        raise AnchorIRGoldenError(
                            "Golden manifest payload verification exceeded the "
                            "%g-second total limit"
                            % MAX_ANCHOR_IR_GOLDEN_VERIFICATION_SECONDS
                        )
                result = normalizer.normalize_text(
                    stage.normalized_ir,
                    normalization_version=self.normalization_version,
                    spec_version=self.spec_version,
                    track=self.track,
                    phase=stage.phase,
                    context=context,
                    source_name="<anchor-ir-golden:%s>" % stage.stage_id.value,
                    extension_dialects=stage.extension_dialects,
                    _worker_timeout_seconds=worker_timeout,
                )
            if not result.acceptable:
                raise AnchorIRGoldenValidationError(
                    stage.stage_id,
                    result.validation_report,
                )
            if result.normalized_text != stage.normalized_ir:
                raise AnchorIRGoldenError(
                    "Stage '%s' is not canonical normalized AnchorIR"
                    % stage.stage_id.value
                )
            if result.sha256 != stage.sha256:
                raise AnchorIRGoldenError(
                    "Stage '%s' hash does not match re-normalized AnchorIR"
                    % stage.stage_id.value
                )
            verified_payloads[cache_key] = result
        object.__setattr__(self, "_payloads_verified", True)

    def to_dict(self) -> Dict[str, Any]:
        self._verify_stage_payloads()
        return {
            "manifest_version": self.manifest_version,
            "case_id": self.case_id,
            "spec_version": self.spec_version,
            "normalization_version": self.normalization_version,
            "track": self.track.value,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def to_json(self) -> str:
        """Return canonical, UTF-8-ready JSON with one trailing newline."""

        encoded = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        utf8_length = len(encoded.encode("utf-8", errors="strict"))
        if utf8_length > MAX_ANCHOR_IR_GOLDEN_MANIFEST_BYTES:
            raise AnchorIRGoldenError(
                "Golden manifest JSON exceeds the %d-byte UTF-8 output limit"
                % MAX_ANCHOR_IR_GOLDEN_MANIFEST_BYTES
            )
        return encoded

    @classmethod
    def from_dict(
        cls,
        value: Dict[str, Any],
        *,
        context=None,
    ) -> "AnchorIRGoldenManifest":
        required = {
            "manifest_version",
            "case_id",
            "spec_version",
            "normalization_version",
            "track",
            "stages",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise AnchorIRGoldenError(
                "Golden manifest must contain exactly: %s" % ", ".join(sorted(required))
            )
        if not isinstance(value["stages"], (list, tuple)):
            raise TypeError("Golden manifest stages must be a list")
        if len(value["stages"]) > MAX_ANCHOR_IR_GOLDEN_STAGES:
            raise AnchorIRGoldenError(
                "Golden manifest exceeds the %d-Stage limit"
                % MAX_ANCHOR_IR_GOLDEN_STAGES
            )
        manifest = cls(
            manifest_version=value["manifest_version"],
            case_id=value["case_id"],
            spec_version=value["spec_version"],
            normalization_version=value["normalization_version"],
            track=_coerce_track(value["track"]),
            stages=tuple(
                AnchorIRGoldenStage.from_dict(stage) for stage in value["stages"]
            ),
        )
        manifest._verify_stage_payloads(context=context)
        return manifest

    @classmethod
    def from_json(cls, value: str, *, context=None) -> "AnchorIRGoldenManifest":
        if not isinstance(value, str):
            raise TypeError("Golden manifest JSON must be a string")
        try:
            utf8_length = len(value.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise AnchorIRGoldenError(
                "Golden manifest JSON must be valid UTF-8"
            ) from error
        if utf8_length > MAX_ANCHOR_IR_GOLDEN_MANIFEST_BYTES:
            raise AnchorIRGoldenError(
                "Golden manifest JSON exceeds the %d-byte UTF-8 input limit"
                % MAX_ANCHOR_IR_GOLDEN_MANIFEST_BYTES
            )
        _check_manifest_json_nesting(value)
        try:
            parsed = json.loads(value, object_pairs_hook=_strict_json_object)
        except AnchorIRGoldenError:
            raise
        except (json.JSONDecodeError, RecursionError, ValueError) as error:
            detail = getattr(error, "msg", str(error))
            raise AnchorIRGoldenError(
                "invalid Golden manifest JSON: %s" % detail
            ) from error
        return cls.from_dict(parsed, context=context)


class AnchorIRGoldenBuilder:
    """AnchorIR-specific Stage recorder used by a corpus runner."""

    def __init__(
        self,
        *,
        case_id: str,
        spec_version: str = ANCHOR_IR_SPEC_VERSION,
        normalization_version: str = ANCHOR_IR_NORMALIZATION_VERSION,
        track: Union[AnchorIRTrack, str] = AnchorIRTrack.LINALG,
    ):
        _validate_case_id(case_id)
        self.case_id = case_id
        self.spec_version = spec_version
        self.normalization_version = normalization_version
        self.track = AnchorIRTrack(track)
        self._normalizer = AnchorIRNormalizer()
        self._stages: List[AnchorIRGoldenStage] = []
        self._poisoned_stage: Optional[AnchorIRStageId] = None
        self._poisoned_reason: Optional[str] = None

    def _ensure_usable(self) -> None:
        if self._poisoned_stage is not None:
            raise AnchorIRGoldenError(
                "Golden builder is poisoned after Stage '%s' failed: %s; "
                "create a new builder"
                % (self._poisoned_stage.value, self._poisoned_reason)
            )

    def _poison(self, stage_id: AnchorIRStageId, error: Exception) -> None:
        if self._poisoned_stage is None:
            self._poisoned_stage = stage_id
            self._poisoned_reason = "%s: %s" % (type(error).__name__, error)

    def _append_result(
        self,
        stage_id: AnchorIRStageId,
        extension_dialects: Tuple[str, ...],
        result,
    ) -> AnchorIRGoldenStage:
        if not result.acceptable:
            raise AnchorIRGoldenValidationError(
                stage_id,
                result.validation_report,
            )
        stage = AnchorIRGoldenStage(
            stage_id=stage_id,
            phase=stage_id.phase,
            extension_dialects=extension_dialects,
            sha256=result.sha256,
            normalized_ir=result.normalized_text,
        )
        self._stages.append(stage)
        return stage

    def add_text(
        self,
        stage_id: Union[AnchorIRStageId, str],
        ir_text: str,
        *,
        extension_dialects: Iterable[str] = (),
        context=None,
        source_name: str = "<anchor-ir-golden>",
    ) -> AnchorIRGoldenStage:
        """Validate, normalize and append one textual AnchorIR Stage."""

        self._ensure_usable()
        parsed_stage_id = _coerce_stage_id(stage_id)
        extensions = _validate_extensions(
            extension_dialects,
            phase=parsed_stage_id.phase,
        )
        try:
            result = self._normalizer.normalize_text(
                ir_text,
                normalization_version=self.normalization_version,
                spec_version=self.spec_version,
                track=self.track,
                phase=parsed_stage_id.phase,
                context=context,
                source_name=source_name,
                extension_dialects=extensions,
            )
            return self._append_result(parsed_stage_id, extensions, result)
        except Exception as error:
            self._poison(parsed_stage_id, error)
            raise

    def add_module(
        self,
        stage_id: Union[AnchorIRStageId, str],
        module,
        *,
        extension_dialects: Iterable[str] = (),
    ) -> AnchorIRGoldenStage:
        """Validate, normalize and append one real ModuleOp Stage."""

        self._ensure_usable()
        parsed_stage_id = _coerce_stage_id(stage_id)
        extensions = _validate_extensions(
            extension_dialects,
            phase=parsed_stage_id.phase,
        )
        try:
            result = self._normalizer.normalize_module(
                module,
                normalization_version=self.normalization_version,
                spec_version=self.spec_version,
                track=self.track,
                phase=parsed_stage_id.phase,
                extension_dialects=extensions,
            )
            return self._append_result(parsed_stage_id, extensions, result)
        except Exception as error:
            self._poison(parsed_stage_id, error)
            raise

    def build(self) -> AnchorIRGoldenManifest:
        self._ensure_usable()
        manifest = AnchorIRGoldenManifest(
            manifest_version=ANCHOR_IR_GOLDEN_MANIFEST_VERSION,
            case_id=self.case_id,
            spec_version=self.spec_version,
            normalization_version=self.normalization_version,
            track=self.track,
            stages=tuple(self._stages),
        )
        # Each Stage passed the strict normalizer immediately before append.
        object.__setattr__(manifest, "_payloads_verified", True)
        return manifest


@dataclass(frozen=True)
class AnchorIRGoldenDifference:
    """The first ordered difference between Golden and current Stages."""

    reason: str
    expected_stage_id: Optional[str]
    actual_stage_id: Optional[str]
    old_hash: Optional[str]
    new_hash: Optional[str]
    normalized_ir_diff: str

    def __post_init__(self) -> None:
        reasons = {
            "stage_sequence_mismatch",
            "stage_policy_mismatch",
            "stage_count_mismatch",
            "hash_mismatch",
            "normalized_ir_mismatch",
        }
        if self.reason not in reasons:
            raise AnchorIRGoldenError(
                "unsupported Golden difference reason %r" % self.reason
            )
        if self.expected_stage_id is None and self.actual_stage_id is None:
            raise AnchorIRGoldenError(
                "Golden difference requires an expected or actual Stage ID"
            )
        for field_name in ("expected_stage_id", "actual_stage_id"):
            value = getattr(self, field_name)
            if value is not None:
                AnchorIRStageId(value)
        for field_name in ("old_hash", "new_hash"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
            ):
                raise AnchorIRGoldenError(
                    "%s must be None or a lowercase SHA-256" % field_name
                )
        if not isinstance(self.normalized_ir_diff, str):
            raise TypeError("normalized_ir_diff must be a string")

    @property
    def stage_id(self) -> Optional[str]:
        return self.expected_stage_id or self.actual_stage_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "stage_id": self.stage_id,
            "expected_stage_id": self.expected_stage_id,
            "actual_stage_id": self.actual_stage_id,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "normalized_ir_diff": self.normalized_ir_diff,
        }


@dataclass(frozen=True)
class AnchorIRGoldenComparisonReport:
    """Result of an ordered AnchorIR Stage comparison."""

    case_id: str
    matched_stages: int
    expected_stage_count: int
    actual_stage_count: int
    first_divergence: Optional[AnchorIRGoldenDifference] = None

    def __post_init__(self) -> None:
        _validate_case_id(self.case_id)
        for field_name in (
            "matched_stages",
            "expected_stage_count",
            "actual_stage_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise AnchorIRGoldenError(
                    "%s must be a non-negative integer" % field_name
                )
        if self.matched_stages > min(
            self.expected_stage_count,
            self.actual_stage_count,
        ):
            raise AnchorIRGoldenError("matched_stages exceeds available Stage count")
        if self.first_divergence is None:
            if not (
                self.expected_stage_count
                == self.actual_stage_count
                == self.matched_stages
            ):
                raise AnchorIRGoldenError(
                    "a matched report requires equal, fully matched counts"
                )
        elif not isinstance(
            self.first_divergence,
            AnchorIRGoldenDifference,
        ):
            raise TypeError("first_divergence must be AnchorIRGoldenDifference or None")

    @property
    def matched(self) -> bool:
        return self.first_divergence is None

    @property
    def first_changed_stage(self) -> Optional[str]:
        if self.first_divergence is None:
            return None
        return self.first_divergence.stage_id

    @property
    def old_hash(self) -> Optional[str]:
        if self.first_divergence is None:
            return None
        return self.first_divergence.old_hash

    @property
    def new_hash(self) -> Optional[str]:
        if self.first_divergence is None:
            return None
        return self.first_divergence.new_hash

    @property
    def normalized_ir_diff(self) -> str:
        if self.first_divergence is None:
            return ""
        return self.first_divergence.normalized_ir_diff

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "case_id": self.case_id,
            "matched_stages": self.matched_stages,
            "expected_stage_count": self.expected_stage_count,
            "actual_stage_count": self.actual_stage_count,
            "first_changed_stage": self.first_changed_stage,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "normalized_ir_diff": self.normalized_ir_diff,
            "first_divergence": (
                None
                if self.first_divergence is None
                else self.first_divergence.to_dict()
            ),
        }


def _normalized_diff(
    expected: Optional[AnchorIRGoldenStage],
    actual: Optional[AnchorIRGoldenStage],
) -> str:
    expected_id = "<missing>" if expected is None else expected.stage_id.value
    actual_id = "<missing>" if actual is None else actual.stage_id.value
    expected_lines = [] if expected is None else expected.normalized_ir.splitlines(True)
    actual_lines = [] if actual is None else actual.normalized_ir.splitlines(True)
    return "".join(
        difflib.unified_diff(
            expected_lines,
            actual_lines,
            fromfile="golden:%s" % expected_id,
            tofile="current:%s" % actual_id,
            lineterm="\n",
        )
    )


def _difference(
    reason: str,
    expected: Optional[AnchorIRGoldenStage],
    actual: Optional[AnchorIRGoldenStage],
) -> AnchorIRGoldenDifference:
    return AnchorIRGoldenDifference(
        reason=reason,
        expected_stage_id=(None if expected is None else expected.stage_id.value),
        actual_stage_id=None if actual is None else actual.stage_id.value,
        old_hash=None if expected is None else expected.sha256,
        new_hash=None if actual is None else actual.sha256,
        normalized_ir_diff=_normalized_diff(expected, actual),
    )


def compare_anchor_ir_golden(
    expected: AnchorIRGoldenManifest,
    actual: AnchorIRGoldenManifest,
) -> AnchorIRGoldenComparisonReport:
    """Compare Stage hashes in order and report only the first divergence."""

    if not isinstance(expected, AnchorIRGoldenManifest) or not isinstance(
        actual, AnchorIRGoldenManifest
    ):
        raise TypeError("expected and actual must be AnchorIRGoldenManifest")
    expected._verify_stage_payloads()
    actual._verify_stage_payloads()
    compatibility_fields = (
        "manifest_version",
        "case_id",
        "spec_version",
        "normalization_version",
        "track",
    )
    incompatible = [
        field_name
        for field_name in compatibility_fields
        if getattr(expected, field_name) != getattr(actual, field_name)
    ]
    if incompatible:
        raise AnchorIRGoldenError(
            "incompatible Golden manifests differ in: %s" % ", ".join(incompatible)
        )

    expected_stages = expected.stages
    actual_stages = actual.stages
    common_count = min(len(expected_stages), len(actual_stages))
    for index in range(common_count):
        old = expected_stages[index]
        new = actual_stages[index]
        if old.stage_id != new.stage_id:
            difference = _difference("stage_sequence_mismatch", old, new)
        elif old.extension_dialects != new.extension_dialects:
            difference = _difference("stage_policy_mismatch", old, new)
        elif old.sha256 != new.sha256:
            difference = _difference("hash_mismatch", old, new)
        elif old.normalized_ir != new.normalized_ir:
            difference = _difference("normalized_ir_mismatch", old, new)
        else:
            continue
        return AnchorIRGoldenComparisonReport(
            case_id=expected.case_id,
            matched_stages=index,
            expected_stage_count=len(expected_stages),
            actual_stage_count=len(actual_stages),
            first_divergence=difference,
        )

    if len(expected_stages) != len(actual_stages):
        old = (
            expected_stages[common_count]
            if common_count < len(expected_stages)
            else None
        )
        new = actual_stages[common_count] if common_count < len(actual_stages) else None
        return AnchorIRGoldenComparisonReport(
            case_id=expected.case_id,
            matched_stages=common_count,
            expected_stage_count=len(expected_stages),
            actual_stage_count=len(actual_stages),
            first_divergence=_difference(
                "stage_count_mismatch",
                old,
                new,
            ),
        )

    return AnchorIRGoldenComparisonReport(
        case_id=expected.case_id,
        matched_stages=len(expected_stages),
        expected_stage_count=len(expected_stages),
        actual_stage_count=len(actual_stages),
    )
