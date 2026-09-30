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

from .hw_capability import (
    HWCapability as HWCapability,
    ComputeParadigm as ComputeParadigm,
)
from .anchor_ir import (
    AnchorIRTrack as AnchorIRTrack,
    AnchorIRValidator as AnchorIRValidator,
)
from .pipeline import build_ttir_pipeline as build_ttir_pipeline
from .pipeline import route_adapter as route_adapter
from .pipeline import convert_ttir_to_anchor_ir as convert_ttir_to_anchor_ir
from .adapters.router import (
    AdapterDecision as AdapterDecision,
    AdapterRouter as AdapterRouter,
    AdapterRouterConfig as AdapterRouterConfig,
    AdapterRoutingError as AdapterRoutingError,
    OpCoverage as OpCoverage,
)
