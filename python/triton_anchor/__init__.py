"""
triton-anchor: Unified Triton Compilation Frontend
===================================================

A compilation frontend that converts Triton TTIR to hardware-aware Linalg IR,
serving as the bridge between Triton core and out-of-tree hardware backends.

Architecture:
  Layer 1  — TTIR Pipeline       (core invariant: 7 mandatory passes)
  Layer 2  — Linalg Adapters     (triton-shared / triton-linalg / hybrid)
  Layer 2.5 — AnchorIR Spec      (core invariant: dual-track dialect whitelist)
"""

__version__ = "0.2.0"

from .hw_capability import HWCapability, ComputeParadigm
from .anchor_ir import AnchorIRTrack, AnchorIRValidator
from .pipeline import build_ttir_pipeline


def __getattr__(name):
    # Keep HWCapability/AnchorIR usable in pure-Python environments where the
    # compiled Triton extension is intentionally absent.
    if name in {"AnchorBackendBase", "AnchorCompilationContext"}:
        from .backend import AnchorBackendBase, AnchorCompilationContext

        return {
            "AnchorBackendBase": AnchorBackendBase,
            "AnchorCompilationContext": AnchorCompilationContext,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
