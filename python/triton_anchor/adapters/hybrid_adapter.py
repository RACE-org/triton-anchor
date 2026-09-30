"""HybridAdapter: Structured first, Router-authorized AxisInfo fallback."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from ..anchor_ir import AnchorIRTrack
from .base import (
    AdapterConversionContext,
    AdapterConversionError,
    ILinalgOptAdapter,
    ITritonToLinalgAdapter,
)

logger = logging.getLogger(__name__)


class HybridAdapter(ILinalgOptAdapter):
    """Hybrid adapter that tries Structured first, then AxisInfo if authorized.

    Fallback authorization is a router decision, not an adapter default.
    """

    def __init__(
        self,
        structured_adapter: Optional[ITritonToLinalgAdapter] = None,
        axis_info_adapter: Optional[ITritonToLinalgAdapter] = None,
    ):
        self._structured_adapter = structured_adapter
        self._axis_info_adapter = axis_info_adapter

    def name(self) -> str:
        return "hybrid"

    def get_supported_tracks(self) -> Tuple[AnchorIRTrack, ...]:
        return (AnchorIRTrack.LINALG,)

    def get_supported_ptr_models(self) -> Tuple[str, ...]:
        return ("hybrid",)

    def get_required_backend_capabilities(self) -> Tuple[str, ...]:
        return ("anchor_ir.linalg",)

    def get_supported_ops(self) -> Tuple[str, ...]:
        return ("*",)

    def supports_internal_fallback(self) -> bool:
        return True

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Attempt Structured conversion, fall back to AxisInfo on failure.

        ``AdapterRouter`` must set ``fallback_authorized`` for the AxisInfo
        path to run.  Without that authorization, the original Structured
        failure is reported through a ``HybridAdapter`` conversion error.
        """
        structured = self._get_structured_adapter()
        axis_info = self._get_axis_info_adapter()
        hybrid_meta = metadata.setdefault(
            "adapter_hybrid",
            {"attempts": [], "fallback_used": False},
        )

        try:
            result = structured.convert(ttir_module, metadata, context)
            hybrid_meta["attempts"].append(
                {"adapter": structured.name(), "result": "selected"}
            )
            hybrid_meta["runtime_adapter"] = structured.name()
            self._mark_router_metadata(
                metadata,
                context,
                fallback=False,
                runtime_adapter=structured.name(),
                fallback_reason=None,
                fallback_chain=(),
            )
            return result
        except AdapterConversionError as structured_error:
            hybrid_meta["attempts"].append(
                {
                    "adapter": structured.name(),
                    "result": "failed",
                    "reason": str(structured_error),
                }
            )
            if not self._fallback_authorized(metadata, context):
                hybrid_meta["fallback_used"] = False
                raise AdapterConversionError(
                    self.name(),
                    kernel_name=metadata.get("name", ""),
                    detail=(
                        "Structured path failed and AdapterRouter did not "
                        f"authorize AxisInfo fallback: {structured_error}"
                    ),
                ) from structured_error

            logger.info("Structured path failed; Router authorized AxisInfo fallback")
            fallback_reason = (
                f"structured adapter '{structured.name()}' failed: "
                f"{structured_error}"
            )
            try:
                result = axis_info.convert(ttir_module, metadata, context)
            except AdapterConversionError as axis_info_error:
                hybrid_meta["attempts"].append(
                    {
                        "adapter": axis_info.name(),
                        "result": "failed",
                        "reason": str(axis_info_error),
                    }
                )
                hybrid_meta["fallback_used"] = True
                self._mark_router_metadata(
                    metadata,
                    context,
                    fallback=True,
                    runtime_adapter=axis_info.name(),
                    fallback_from=structured.name(),
                    fallback_reason=fallback_reason,
                    fallback_chain=(structured.name(), axis_info.name()),
                )
                raise AdapterConversionError(
                    self.name(),
                    kernel_name=metadata.get("name", ""),
                    detail=(
                        "Structured path failed, and authorized AxisInfo "
                        f"fallback also failed: {axis_info_error}"
                    ),
                ) from axis_info_error

            hybrid_meta["attempts"].append(
                {"adapter": axis_info.name(), "result": "selected"}
            )
            hybrid_meta["fallback_used"] = True
            hybrid_meta["runtime_adapter"] = axis_info.name()
            hybrid_meta["fallback_from"] = structured.name()
            self._mark_router_metadata(
                metadata,
                context,
                fallback=True,
                runtime_adapter=axis_info.name(),
                fallback_from=structured.name(),
                fallback_reason=fallback_reason,
                fallback_chain=(structured.name(), axis_info.name()),
            )
            return result

    def _get_structured_adapter(self) -> ITritonToLinalgAdapter:
        if self._structured_adapter is not None:
            return self._structured_adapter
        from .triton_shared_adapter import TritonSharedAdapter

        self._structured_adapter = TritonSharedAdapter()
        return self._structured_adapter

    def _get_axis_info_adapter(self) -> ITritonToLinalgAdapter:
        if self._axis_info_adapter is not None:
            return self._axis_info_adapter
        from .triton_linalg_adapter import TritonLinalgAdapter

        self._axis_info_adapter = TritonLinalgAdapter()
        return self._axis_info_adapter

    def _fallback_authorized(self, metadata: dict, context: Any) -> bool:
        if isinstance(context, AdapterConversionContext):
            return bool(context.fallback_authorized)

        decision = metadata.get("adapter_decision")
        if not isinstance(decision, dict):
            return False
        return bool(decision.get("fallback_authorized"))

    def _mark_router_metadata(
        self,
        metadata: dict,
        context: Any,
        *,
        fallback: bool,
        runtime_adapter: str,
        fallback_from: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        fallback_chain: Tuple[str, ...] = (),
    ) -> None:
        key = (
            context.metadata_key
            if isinstance(context, AdapterConversionContext)
            else "adapter_decision"
        )
        decision = metadata.get(key)
        if not isinstance(decision, dict):
            return
        decision["runtime_adapter"] = runtime_adapter
        decision["fallback"] = bool(fallback)
        decision["fallback_from"] = fallback_from
        decision["fallback_reason"] = fallback_reason
        decision["fallback_chain"] = list(fallback_chain)

    def get_output_dialects(self) -> List[str]:
        return [
            "linalg",
            "linalg_ext",
            "tensor",
            "memref",
            "arith",
            "math",
            "scf",
            "func",
            "aux",
        ]
