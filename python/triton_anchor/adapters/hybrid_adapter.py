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
"""

from __future__ import annotations

import logging
from typing import Any, List

from .base import ILinalgOptAdapter, AdapterConversionError

logger = logging.getLogger(__name__)


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

        from .triton_linalg_adapter import TritonLinalgAdapter
        return TritonLinalgAdapter().convert(ttir_module, metadata, context)

    def get_output_dialects(self) -> List[str]:
        return [
            "linalg", "linalg_ext", "tensor", "memref",
            "arith", "math", "scf", "func", "aux",
        ]
