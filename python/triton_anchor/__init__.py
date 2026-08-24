"""
triton-anchor: Unified Triton Compilation Frontend
===================================================

A compilation frontend that keeps the triton_anchor orchestration layer while
embedding the triton-shared frontend/core toolchain.

Architecture:
  Layer 1   — TTIR Pipeline       (core invariant: 7 mandatory passes)
  Layer 2   — Linalg Adapter      (triton-shared subprocess integration)
  Layer 2.5 — AnchorIR Spec       (dual-track dialect whitelist)
"""

from ._version import __version__ as __version__

from .hw_capability import (
    HWCapability as HWCapability,
    ComputeParadigm as ComputeParadigm,
)
from .anchor_ir import (
    AnchorIRTrack as AnchorIRTrack,
    AnchorIRValidator as AnchorIRValidator,
)
from .pipeline import build_ttir_pipeline as build_ttir_pipeline
