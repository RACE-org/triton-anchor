"""
HybridAdapter — Stub for Structured-first, AxisInfo-fallback strategy
======================================================================

Future implementation that tries TritonSharedAdapter first (Structured
pointer analysis, works for regular access patterns), and falls back
to TritonLinalgAdapter (AxisInfo, handles all patterns) on failure.

This provides the best of both worlds:
  - Structured analysis produces cleaner IR for simple patterns
  - AxisInfo is a universal fallback

The Hybrid adapter first runs TritonSharedAdapter in structured mode.  If that
conversion fails and fallback is enabled, it retries with TritonLinalgAdapter
so AxisInfo remains the deterministic fallback path.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

from .base import AdapterConversionError, ITritonToLinalgAdapter
from .triton_shared_adapter import TritonSharedAdapter

logger = logging.getLogger(__name__)


class HybridAdapter(ITritonToLinalgAdapter):
    """Hybrid adapter: tries Structured first, falls back to AxisInfo.

    Fallback can be disabled per-conversion with
    ``metadata["anchor_adapter_fallback_policy"] = "strict"`` or
    ``metadata["anchor_adapter_allow_fallback"] = False``.
    """

    def __init__(
        self,
        structured_adapter: ITritonToLinalgAdapter | None = None,
        axis_info_adapter: ITritonToLinalgAdapter | None = None,
    ):
        self._structured_adapter = structured_adapter
        self._axis_info_adapter = axis_info_adapter

    def name(self) -> str:
        return "hybrid"

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Attempt Structured conversion, fall back to AxisInfo on failure.
        """
        metadata["anchor_adapter_effective"] = self.name()
        metadata.setdefault("anchor_adapter_hybrid_attempts", [])

        structured = self._structured_adapter or TritonSharedAdapter(mode="structured")
        try:
            result = structured.convert(ttir_module, metadata, context)
        except AdapterConversionError as structured_error:
            metadata["anchor_adapter_hybrid_attempts"].append(
                {
                    "adapter": structured.name(),
                    "mode": "structured",
                    "result": "failed",
                    "detail": str(structured_error),
                }
            )
            if not self._fallback_allowed(metadata):
                metadata["anchor_adapter_fallback_blocked"] = True
                metadata["anchor_adapter_fallback_reason"] = (
                    f"strict policy blocked AxisInfo fallback after {structured.name()} "
                    f"failed: {structured_error}"
                )
                raise

            fallback_reason = (
                f"{structured.name()} failed with AdapterConversionError; "
                "falling back to AxisInfo via triton-linalg"
            )
            metadata["anchor_adapter_fallback_reason"] = fallback_reason
            logger.info(
                "Structured adapter failed for kernel '%s'; falling back to AxisInfo",
                metadata.get("name", "<unknown>"),
            )
            axis_info = self._axis_info_adapter or self._make_axis_info_adapter()
            try:
                result = axis_info.convert(ttir_module, metadata, context)
            except AdapterConversionError as axis_info_error:
                metadata["anchor_adapter_hybrid_attempts"].append(
                    {
                        "adapter": axis_info.name(),
                        "mode": "axis_info",
                        "result": "failed",
                        "detail": str(axis_info_error),
                    }
                )
                raise AdapterConversionError(
                    self.name(),
                    kernel_name=metadata.get("name", ""),
                    detail=(
                        "structured conversion failed and AxisInfo fallback "
                        f"also failed: {axis_info_error}"
                    ),
                ) from axis_info_error

            metadata["anchor_adapter_hybrid_attempts"].append(
                {
                    "adapter": axis_info.name(),
                    "mode": "axis_info",
                    "result": "succeeded",
                }
            )
            metadata["anchor_adapter_effective"] = axis_info.name()
            metadata["anchor_adapter_fallback"] = "axis_info"
            return result

        metadata["anchor_adapter_hybrid_attempts"].append(
            {
                "adapter": structured.name(),
                "mode": "structured",
                "result": "succeeded",
            }
        )
        metadata["anchor_adapter_effective"] = structured.name()
        return result

    def _make_axis_info_adapter(self) -> ITritonToLinalgAdapter:
        from .triton_linalg_adapter import TritonLinalgAdapter

        return TritonLinalgAdapter()

    def _fallback_allowed(self, metadata: dict) -> bool:
        if metadata.get("anchor_adapter_fallback_policy") == "strict":
            return False
        if metadata.get("anchor_adapter_allow_fallback") is False:
            return False
        return True

    def supported_routes(self) -> List[Tuple[str, str]]:
        return [("linalg", "hybrid")]

    def get_output_dialects(self) -> List[str]:
        return sorted(
            set(TritonSharedAdapter(mode="structured").get_output_dialects())
            | set(self._make_axis_info_adapter().get_output_dialects())
        )
