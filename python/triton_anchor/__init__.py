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
from .pipeline import select_adapter as select_adapter
from .adapters import (
    ADAPTER_ROUTING_POLICY_VERSION as ADAPTER_ROUTING_POLICY_VERSION,
    AdapterRouter as AdapterRouter,
    AdapterRoutingPolicy as AdapterRoutingPolicy,
)
