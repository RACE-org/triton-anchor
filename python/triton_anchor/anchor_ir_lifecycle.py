"""Mandatory pre-hook / backend-hook / post-hook AnchorIR orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from .anchor_ir_rules import ANCHOR_IR_SPEC_VERSION
from .anchor_ir_normalizer import (
    ANCHOR_IR_NORMALIZATION_VERSION,
    AnchorIRNormalizationResult,
    AnchorIRNormalizer,
)
from .anchor_ir_schema import (
    AnchorIRPhase,
    AnchorIRTrack,
    AnchorIRValidationReport,
)
from .anchor_ir_validator import (
    AnchorIRValidationError,
    StructuredAnchorIRValidator,
)


@runtime_checkable
class AnchorIRBackendHook(Protocol):
    """External backend contract for the AnchorIR boundary.

    ``on_anchor_ir_ready`` may mutate and return the supplied value, or mutate
    it in place and return ``None``.
    """

    def on_anchor_ir_ready(self, anchor_ir: Any) -> Any: ...


@dataclass(frozen=True)
class AnchorIRLifecycleReport:
    """Observable result of one complete AnchorIR boundary lifecycle."""

    output: Any
    pre_hook: AnchorIRValidationReport
    post_hook: Optional[AnchorIRValidationReport]
    hook_executed: bool
    declared_extensions: tuple[str, ...]
    post_hook_snapshot: Optional[AnchorIRNormalizationResult] = None
    lowering_executed: bool = False
    lowered_output: Any = None

    @property
    def valid(self) -> bool:
        return (
            self.pre_hook.valid
            and self.post_hook is not None
            and self.post_hook.valid
            and self.post_hook_snapshot is not None
            and self.post_hook_snapshot.acceptable
            and self.post_hook_snapshot.validation_report == self.post_hook
        )


class AnchorIRLifecycleOrchestrator:
    """Enforce the strict AnchorIR lifecycle for an external backend.

    This is deliberately an explicit component rather than an implicit global
    compiler hook: triton-anchor cannot intercept every out-of-tree backend.
    In-repository Adapters enter through ``ITritonToLinalgAdapter.compile``;
    external backends must invoke ``run_module_or_raise`` (or
    ``run_text_or_raise`` for text IR) before their own lowering begins.
    """

    def __init__(self):
        # This is a production fail-closed boundary.  Accepting a caller-owned
        # validator would let a duck-typed stub approve forbidden IR.
        self._validator = StructuredAnchorIRValidator()
        self._normalizer = AnchorIRNormalizer()

    @staticmethod
    def _raise_if_invalid(
        lifecycle: AnchorIRLifecycleReport,
    ) -> AnchorIRLifecycleReport:
        if not lifecycle.pre_hook.valid:
            raise AnchorIRValidationError(lifecycle.pre_hook)
        if lifecycle.post_hook is None:
            raise RuntimeError("AnchorIR lifecycle ended without post-hook validation")
        if not lifecycle.post_hook.valid:
            raise AnchorIRValidationError(lifecycle.post_hook)
        return lifecycle

    @staticmethod
    def _declared_extensions(
        hook: Optional[AnchorIRBackendHook],
    ) -> tuple[str, ...]:
        if hook is None or not hasattr(hook, "get_allowed_dialects"):
            return ()
        callback = hook.get_allowed_dialects
        if not callable(callback):
            raise TypeError("get_allowed_dialects must be callable")
        declared = callback()
        if declared is None:
            return ()
        if isinstance(declared, str):
            raise TypeError("get_allowed_dialects() must return an iterable of strings")
        try:
            extensions = tuple(sorted(set(declared)))
        except TypeError as error:
            raise TypeError(
                "get_allowed_dialects() must return iterable strings"
            ) from error
        if any(
            not isinstance(dialect, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", dialect) is None
            for dialect in extensions
        ):
            raise TypeError(
                "get_allowed_dialects() must return valid MLIR dialect namespaces"
            )
        return extensions

    @staticmethod
    def _run_hook(hook: Optional[AnchorIRBackendHook], value: Any) -> Any:
        if hook is None:
            return value
        callback = getattr(hook, "on_anchor_ir_ready", None)
        if not callable(callback):
            raise TypeError(
                "backend hook must define callable on_anchor_ir_ready()"
            )
        result = callback(value)
        return value if result is None else result

    def run_module(
        self,
        module: Any,
        *,
        hook: Optional[AnchorIRBackendHook],
        spec_version: str = ANCHOR_IR_SPEC_VERSION,
        track: Union[AnchorIRTrack, str] = AnchorIRTrack.LINALG,
        context: Any = None,
        backend_lowering: Optional[Callable[[Any], Any]] = None,
    ) -> AnchorIRLifecycleReport:
        """Validate a real ModuleOp before and after a backend Hook."""

        if backend_lowering is not None and not callable(backend_lowering):
            raise TypeError("backend_lowering must be callable or None")

        # A Python ModuleOp is a non-owning view of storage in an MLIRContext.
        # Resolve and retain that owner before the first native call so every
        # success or failure report can safely keep exposing ``output``.
        module_context = context
        if module_context is None:
            module_context = getattr(module, "context", None)
        if module_context is None:
            raise TypeError(
                "an owning MLIR context is required for ModuleOp lifecycle; "
                "pass context=... or attach module.context"
            )
        from triton._C.libtriton import anchor

        anchor.check_anchor_ir_module_context(module, module_context)
        module.context = module_context

        pre_hook = self._validator.validate_module(
            module,
            spec_version=spec_version,
            track=track,
            phase=AnchorIRPhase.PRE_HOOK,
        )
        if not pre_hook.valid:
            return AnchorIRLifecycleReport(
                output=module,
                pre_hook=pre_hook,
                post_hook=None,
                hook_executed=False,
                declared_extensions=(),
            )

        declared = self._declared_extensions(hook)
        # Resolve the post policy before executing the Hook.  In particular,
        # merely declaring a core-forbidden namespace is itself an invalid
        # backend contract, even if the Hook would not emit that namespace.
        self._validator._resolve_policy(
            spec_version=spec_version,
            track=track,
            phase=AnchorIRPhase.POST_HOOK,
            extension_dialects=declared,
        )
        output = self._run_hook(hook, module)
        # A Hook may return a new ModuleOp.  Establish its ownership edge before
        # post-hook validation/normalization, otherwise a Hook-local context can
        # disappear while native code is about to inspect the returned module.
        output_context = getattr(output, "context", None)
        if output_context is None:
            output_context = module_context
        anchor.check_anchor_ir_module_context(output, output_context)
        output.context = output_context
        post_hook_snapshot = self._normalizer.normalize_module(
            output,
            normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
            spec_version=spec_version,
            track=track,
            phase=AnchorIRPhase.POST_HOOK,
            extension_dialects=declared,
        )
        post_hook = post_hook_snapshot.validation_report
        lowered_output = None
        lowering_executed = post_hook.valid and backend_lowering is not None
        if lowering_executed:
            # Backend lowering commonly mutates a ModuleOp in place.  Give it a
            # clone so ``report.output`` remains the exact validated post-hook
            # boundary and Golden capture can never observe lowered IR instead.
            lowering_input = anchor.clone_anchor_ir_module(
                output,
                output_context,
            )
            # Triton's Python compiler stages use this dynamic attribute both as
            # their context access contract and as the Python ownership edge.
            lowering_input.context = output_context
            if getattr(output, "context", None) is None:
                output.context = output_context
            lowered_output = backend_lowering(lowering_input)
        return AnchorIRLifecycleReport(
            output=output,
            pre_hook=pre_hook,
            post_hook=post_hook,
            hook_executed=hook is not None,
            declared_extensions=declared,
            post_hook_snapshot=post_hook_snapshot,
            lowering_executed=lowering_executed,
            lowered_output=lowered_output,
        )

    def run_module_or_raise(
        self,
        module: Any,
        *,
        hook: Optional[AnchorIRBackendHook],
        spec_version: str = ANCHOR_IR_SPEC_VERSION,
        track: Union[AnchorIRTrack, str] = AnchorIRTrack.LINALG,
        context: Any = None,
        backend_lowering: Optional[Callable[[Any], Any]] = None,
    ) -> AnchorIRLifecycleReport:
        """Run the ModuleOp lifecycle and raise before an invalid boundary."""

        lifecycle = self.run_module(
            module,
            hook=hook,
            spec_version=spec_version,
            track=track,
            context=context,
            backend_lowering=backend_lowering,
        )
        return self._raise_if_invalid(lifecycle)

    def run_text(
        self,
        anchor_ir: str,
        *,
        hook: Optional[AnchorIRBackendHook],
        spec_version: str = ANCHOR_IR_SPEC_VERSION,
        track: Union[AnchorIRTrack, str] = AnchorIRTrack.LINALG,
        context=None,
        source_name: str = "<anchor-ir>",
        backend_lowering: Optional[Callable[[str], Any]] = None,
    ) -> AnchorIRLifecycleReport:
        """Text Adapter variant; post-hook still parses and checks full IR."""

        if backend_lowering is not None and not callable(backend_lowering):
            raise TypeError("backend_lowering must be callable or None")

        pre_hook = self._validator.validate_text(
            anchor_ir,
            spec_version=spec_version,
            track=track,
            phase=AnchorIRPhase.PRE_HOOK,
            context=context,
            source_name=source_name,
        )
        if not pre_hook.valid:
            return AnchorIRLifecycleReport(
                output=anchor_ir,
                pre_hook=pre_hook,
                post_hook=None,
                hook_executed=False,
                declared_extensions=(),
            )

        declared = self._declared_extensions(hook)
        self._validator._resolve_policy(
            spec_version=spec_version,
            track=track,
            phase=AnchorIRPhase.POST_HOOK,
            extension_dialects=declared,
        )
        output = self._run_hook(hook, anchor_ir)
        if not isinstance(output, str):
            raise TypeError("text backend hook must return MLIR text or None")
        post_hook_snapshot = self._normalizer.normalize_text(
            output,
            normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
            spec_version=spec_version,
            track=track,
            phase=AnchorIRPhase.POST_HOOK,
            context=context,
            source_name=source_name,
            extension_dialects=declared,
        )
        post_hook = post_hook_snapshot.validation_report
        lowered_output = None
        lowering_executed = post_hook.valid and backend_lowering is not None
        if lowering_executed:
            lowered_output = backend_lowering(output)
        return AnchorIRLifecycleReport(
            output=output,
            pre_hook=pre_hook,
            post_hook=post_hook,
            hook_executed=hook is not None,
            declared_extensions=declared,
            post_hook_snapshot=post_hook_snapshot,
            lowering_executed=lowering_executed,
            lowered_output=lowered_output,
        )

    def run_text_or_raise(
        self,
        anchor_ir: str,
        *,
        hook: Optional[AnchorIRBackendHook],
        spec_version: str = ANCHOR_IR_SPEC_VERSION,
        track: Union[AnchorIRTrack, str] = AnchorIRTrack.LINALG,
        context=None,
        source_name: str = "<anchor-ir>",
        backend_lowering: Optional[Callable[[str], Any]] = None,
    ) -> AnchorIRLifecycleReport:
        """Run the text lifecycle and raise before an invalid boundary."""

        lifecycle = self.run_text(
            anchor_ir,
            hook=hook,
            spec_version=spec_version,
            track=track,
            context=context,
            source_name=source_name,
            backend_lowering=backend_lowering,
        )
        return self._raise_if_invalid(lifecycle)
