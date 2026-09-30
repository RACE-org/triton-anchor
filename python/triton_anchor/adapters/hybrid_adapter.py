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

    def get_supported_tracks(self) -> List[str]:
        return ["linalg"]

    def get_supported_ptr_models(self) -> List[str]:
        return ["hybrid"]

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Attempt Structured conversion, fall back to AxisInfo on failure.

        The hybrid route is explicit: fallback to AxisInfo is allowed only after
        this adapter has been selected, and the reason is recorded in metadata.
        """
        ptr_features = str(metadata.get("ptr_features", "")).lower()
        has_unstructured_hint = any(
            flag in ptr_features
            for flag in ("unstructured", "dynamic", "irregular", "unknown")
        )
        strict = self._strict_mode(metadata)

        if not has_unstructured_hint:
            from .triton_shared_adapter import TritonSharedAdapter

            try:
                metadata["hybrid_ptr_analysis"] = "structured"
                return TritonSharedAdapter().convert(ttir_module, metadata, context)
            except AdapterConversionError as exc:
                if strict:
                    reason = (
                        "strict policy blocked hybrid fallback after structured "
                        "path failed: " + str(exc)
                    )
                    metadata["adapter_fallback_reason"] = reason
                    metadata.setdefault("adapter_reject_reasons", {})[
                        "triton-shared"
                    ] = str(exc)
                    metadata["adapter_reject_reasons"]["fallback"] = reason
                    raise AdapterConversionError(self.name(), detail=reason)

                logger.info("Hybrid structured path failed; using AxisInfo fallback")
                metadata["hybrid_ptr_analysis"] = "axis_info"
                metadata["adapter_fallback_reason"] = (
                    "hybrid structured path failed: " + str(exc)
                )
                metadata["adapter_fallback_chain"] = [
                    "triton-shared",
                    "triton-linalg",
                ]
        else:
            if strict:
                reason = (
                    "strict policy blocked hybrid fallback requested by "
                    "ptr_features"
                )
                metadata["adapter_fallback_reason"] = reason
                metadata.setdefault("adapter_reject_reasons", {})["fallback"] = reason
                raise AdapterConversionError(self.name(), detail=reason)

            metadata["hybrid_ptr_analysis"] = "axis_info"
            metadata["adapter_fallback_reason"] = (
                "hybrid metadata ptr_features requested axis_info"
            )
            metadata["adapter_fallback_chain"] = [
                "triton-shared",
                "triton-linalg",
            ]

        from .triton_linalg_adapter import TritonLinalgAdapter

        return TritonLinalgAdapter().convert(ttir_module, metadata, context)

    @staticmethod
    def _strict_mode(metadata: dict) -> bool:
        strict = metadata.get("adapter_policy_strict", True)
        if isinstance(strict, str):
            return strict.lower() not in {"0", "false", "no", "off"}
        return bool(strict)

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
