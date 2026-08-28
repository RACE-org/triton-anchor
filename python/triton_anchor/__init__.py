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
from .anchor_ir_golden import (
    ANCHOR_IR_GOLDEN_MANIFEST_VERSION as ANCHOR_IR_GOLDEN_MANIFEST_VERSION,
    AnchorIRGoldenBuilder as AnchorIRGoldenBuilder,
    AnchorIRGoldenComparisonReport as AnchorIRGoldenComparisonReport,
    AnchorIRGoldenDifference as AnchorIRGoldenDifference,
    AnchorIRGoldenError as AnchorIRGoldenError,
    AnchorIRGoldenManifest as AnchorIRGoldenManifest,
    AnchorIRGoldenStage as AnchorIRGoldenStage,
    AnchorIRGoldenValidationError as AnchorIRGoldenValidationError,
    AnchorIRStageId as AnchorIRStageId,
    compare_anchor_ir_golden as compare_anchor_ir_golden,
)
from .anchor_ir_lifecycle import (
    AnchorIRBackendHook as AnchorIRBackendHook,
    AnchorIRLifecycleOrchestrator as AnchorIRLifecycleOrchestrator,
    AnchorIRLifecycleReport as AnchorIRLifecycleReport,
)
from .anchor_ir_normalizer import (
    ANCHOR_IR_NORMALIZATION_VERSION as ANCHOR_IR_NORMALIZATION_VERSION,
    AnchorIRNormalizationResult as AnchorIRNormalizationResult,
    AnchorIRNormalizer as AnchorIRNormalizer,
)
from .anchor_ir_rules import (
    ANCHOR_IR_SPEC_VERSION as ANCHOR_IR_SPEC_VERSION,
    AnchorIRPolicy as AnchorIRPolicy,
    AnchorIRPolicyError as AnchorIRPolicyError,
)
from . import anchor_ir_rules as _anchor_ir_rules
from .anchor_ir_schema import (
    AnchorIRDiagnostic as AnchorIRDiagnostic,
    AnchorIRLocation as AnchorIRLocation,
    AnchorIRObjectKind as AnchorIRObjectKind,
    AnchorIRPhase as AnchorIRPhase,
    AnchorIRValidationReport as AnchorIRValidationReport,
    format_anchor_ir_validation_report as format_anchor_ir_validation_report,
)
from .anchor_ir_validator import (
    AnchorIRValidationError as AnchorIRValidationError,
    StructuredAnchorIRValidator as StructuredAnchorIRValidator,
)
from .pipeline import (
    build_ttir_pipeline as build_ttir_pipeline,
    run_anchor_ir_compilation as run_anchor_ir_compilation,
)

resolve_anchor_ir_policy = _anchor_ir_rules.resolve_policy
validate_anchor_ir_policy_request = _anchor_ir_rules.validate_policy_request
