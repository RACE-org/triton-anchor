"""
HybridAdapter — Stub for Structured-first, AxisInfo-fallback strategy
======================================================================

Future implementation that tries TritonSharedAdapter first (Structured
pointer analysis, works for regular access patterns), and falls back
to TritonLinalgAdapter (AxisInfo, handles all patterns) on failure.

This provides the best of both worlds:
  - Structured analysis produces cleaner IR for simple patterns
  - AxisInfo is a universal fallback

Status: STUB

At the moment triton-anchor uses a single triton-shared-based lowering path.
The Hybrid adapter is therefore a thin alias over TritonSharedAdapter so call
sites can keep using the symbolic "hybrid" mode.
"""

from __future__ import annotations

from typing import Any, List

from .base import ILinalgOptAdapter
from .triton_shared_adapter import TritonSharedAdapter


class HybridAdapter(ILinalgOptAdapter):
    """Hybrid adapter: tries Structured first, falls back to AxisInfo.

    Status: **STUB** — requires both TritonSharedAdapter and TritonLinalgAdapter
    to be fully functional.
    """

    def name(self) -> str:
        return "hybrid"

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Attempt Structured conversion, fall back to AxisInfo on failure.

        .. note:: Currently a STUB — delegates directly to TritonLinalgAdapter.
        """
        # TODO: When TritonSharedAdapter is fully implemented, try it first:
        #
        # try:
        #     shared = TritonSharedAdapter()
        #     return shared.convert(ttir_module, metadata, context)
        # except AdapterConversionError:
        #     logger.info("Structured analysis failed, falling back to AxisInfo")

        return TritonSharedAdapter(mode="unstructured").convert(
            ttir_module, metadata, context
        )

    def get_output_dialects(self) -> List[str]:
        return TritonSharedAdapter(mode="unstructured").get_output_dialects()
