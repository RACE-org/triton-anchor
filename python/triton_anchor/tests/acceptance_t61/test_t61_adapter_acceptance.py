"""T6.1 acceptance tests for Adapter routing and fallback policy."""

import pytest

from triton_anchor.adapters import AdapterRegistry
from triton_anchor.adapters.base import (
    AdapterConversionContext,
    AdapterConversionError,
    IAnchorAdapter,
)
from triton_anchor.adapters.hybrid_adapter import HybridAdapter
from triton_anchor.adapters.router import (
    AdapterRouter,
    AdapterRouterConfig,
    AdapterRoutingError,
    OpCoverage,
)
from triton_anchor.anchor_ir import AnchorIRError, AnchorIRTrack, AnchorIRValidator
from triton_anchor.hw_capability import (
    ComputeParadigm,
    GPGPUCapability,
    HWCapability,
    MatrixCapability,
    TensorCapability,
)


LINALG_BACKEND_CAPS = ("anchor_ir.linalg",)
TRITON_GPU_BACKEND_CAPS = ("anchor_ir.triton_gpu",)


class FakeAdapter(IAnchorAdapter):
    def __init__(
        self,
        adapter_name,
        *,
        tracks=(AnchorIRTrack.LINALG,),
        ptr_models=("axis_info",),
        required_caps=LINALG_BACKEND_CAPS,
        supported_ops=("*",),
        supports_internal_fallback=False,
        result="converted",
        fail_reason=None,
    ):
        self._adapter_name = adapter_name
        self._tracks = tracks
        self._ptr_models = ptr_models
        self._required_caps = required_caps
        self._supported_ops = supported_ops
        self._supports_internal_fallback = supports_internal_fallback
        self._result = result
        self._fail_reason = fail_reason
        self.calls = 0

    def name(self):
        return self._adapter_name

    def convert(self, ttir_module, metadata, context=None):
        self.calls += 1
        if self._fail_reason:
            raise AdapterConversionError(self.name(), detail=self._fail_reason)
        metadata["converted_by"] = self.name()
        return self._result

    def get_supported_tracks(self):
        return self._tracks

    def get_supported_ptr_models(self):
        return self._ptr_models

    def get_required_backend_capabilities(self):
        return self._required_caps

    def get_supported_ops(self):
        return self._supported_ops

    def supports_internal_fallback(self):
        return self._supports_internal_fallback


def triton_gpu_adapter():
    return FakeAdapter(
        "triton-gpu",
        tracks=(AnchorIRTrack.TRITON_GPU,),
        ptr_models=("gpu",),
        required_caps=TRITON_GPU_BACKEND_CAPS,
    )


def shared_adapter(**kwargs):
    return FakeAdapter(
        "triton-shared",
        ptr_models=("structured",),
        required_caps=LINALG_BACKEND_CAPS,
        **kwargs,
    )


def linalg_adapter(**kwargs):
    return FakeAdapter(
        "triton-linalg",
        ptr_models=("axis_info",),
        required_caps=LINALG_BACKEND_CAPS,
        **kwargs,
    )


def hybrid_route_adapter():
    return FakeAdapter(
        "hybrid",
        ptr_models=("hybrid",),
        required_caps=LINALG_BACKEND_CAPS,
        supports_internal_fallback=True,
    )


def router_with(*adapters):
    return AdapterRouter(adapters={adapter.name(): adapter for adapter in adapters})


def linalg_hw(ptr_model="axis_info", preferred_adapter=None):
    if ptr_model == "structured":
        return HWCapability(
            name="spacemit-x60",
            arch_family="riscv",
            compute_paradigm=ComputeParadigm.AME_MATRIX,
            anchor_ir_track=AnchorIRTrack.LINALG,
            ptr_model=ptr_model,
            preferred_adapter=preferred_adapter,
            matrix_cap=MatrixCapability(),
        )

    return HWCapability(
        name="sophgo-bm1684x",
        arch_family="tpu",
        compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
        anchor_ir_track=AnchorIRTrack.LINALG,
        ptr_model=ptr_model,
        preferred_adapter=preferred_adapter,
        tensor_cap=TensorCapability(num_cores=8),
    )


def gpu_hw(ptr_model="gpu", preferred_adapter=None):
    return HWCapability(
        name="usc-gpu",
        arch_family="gpu",
        compute_paradigm=ComputeParadigm.GPGPU,
        anchor_ir_track=AnchorIRTrack.TRITON_GPU,
        ptr_model=ptr_model,
        preferred_adapter=preferred_adapter,
        gpgpu_cap=GPGPUCapability(num_warps=4, warp_size=32),
    )


def test_router_selects_triton_gpu_for_triton_gpu_track():
    decision = router_with(triton_gpu_adapter()).select(
        gpu_hw(),
        backend_capabilities=TRITON_GPU_BACKEND_CAPS,
    )

    assert decision.selected_adapter == "triton-gpu"
    assert decision.anchor_ir_track == AnchorIRTrack.TRITON_GPU
    assert decision.fallback is False


def test_router_selects_shared_for_structured_linalg():
    decision = router_with(shared_adapter(), linalg_adapter()).select(
        linalg_hw("structured"),
        backend_capabilities=LINALG_BACKEND_CAPS,
    )

    assert decision.selected_adapter == "triton-shared"
    assert decision.ptr_model == "structured"
    assert decision.fallback is False


def test_router_selects_linalg_for_axis_info_linalg():
    decision = router_with(shared_adapter(), linalg_adapter()).select(
        linalg_hw("axis_info"),
        backend_capabilities=LINALG_BACKEND_CAPS,
    )

    assert decision.selected_adapter == "triton-linalg"
    assert decision.ptr_model == "axis_info"
    assert decision.fallback is False


def test_router_selects_hybrid_for_hybrid_linalg():
    decision = router_with(hybrid_route_adapter(), linalg_adapter()).select(
        linalg_hw("hybrid"),
        backend_capabilities=LINALG_BACKEND_CAPS,
    )

    assert decision.selected_adapter == "hybrid"
    assert decision.ptr_model == "hybrid"


def test_preferred_adapter_wins_when_available_and_compatible():
    decision = router_with(shared_adapter(), linalg_adapter()).select(
        linalg_hw("structured", preferred_adapter="triton-shared"),
        backend_capabilities=LINALG_BACKEND_CAPS,
    )

    assert decision.selected_adapter == "triton-shared"
    assert decision.preferred_adapter == "triton-shared"
    assert "preferred adapter" in decision.selected_reason


def test_preferred_adapter_missing_fails():
    with pytest.raises(AdapterRoutingError) as exc_info:
        router_with(shared_adapter(), linalg_adapter()).select(
            linalg_hw("axis_info", preferred_adapter="does-not-exist"),
            backend_capabilities=LINALG_BACKEND_CAPS,
            config=AdapterRouterConfig(allow_fallback=True),
        )

    decision = exc_info.value.decision
    assert decision.selected_adapter is None
    assert decision.fallback is False
    assert decision.rejected_adapters[0].adapter_name == "does-not-exist"
    assert decision.rejected_adapters[0].reason == "adapter is not registered"


def test_preferred_adapter_incompatible_fails():
    with pytest.raises(AdapterRoutingError) as exc_info:
        router_with(shared_adapter(), linalg_adapter(), triton_gpu_adapter()).select(
            linalg_hw("axis_info", preferred_adapter="triton-gpu"),
            backend_capabilities=LINALG_BACKEND_CAPS + TRITON_GPU_BACKEND_CAPS,
            config=AdapterRouterConfig(allow_fallback=True),
        )

    decision = exc_info.value.decision
    assert decision.selected_adapter is None
    assert decision.fallback is False
    assert decision.rejected_adapters[0].adapter_name == "triton-gpu"
    assert decision.rejected_adapters[0].reason == "AnchorIR track mismatch"


def test_no_implicit_registry_order_fallback():
    with pytest.raises(AdapterRoutingError) as exc_info:
        router_with(linalg_adapter()).select(
            linalg_hw("structured"),
            backend_capabilities=LINALG_BACKEND_CAPS,
            config=AdapterRouterConfig(allow_fallback=False),
        )

    decision = exc_info.value.decision
    assert decision.selected_adapter is None
    assert [item.adapter_name for item in decision.rejected_adapters] == [
        "triton-shared",
        "triton-linalg",
    ]
    assert decision.rejected_adapters[-1].reason == (
        "fallback disabled by AdapterRouterConfig"
    )


def test_selection_is_deterministic_across_repeated_runs():
    test_router = router_with(shared_adapter(), linalg_adapter(), hybrid_route_adapter())
    kwargs = {
        "backend_capabilities": LINALG_BACKEND_CAPS,
        "op_coverage": OpCoverage(required_ops=("tt.load", "tt.store")),
    }

    decisions = [
        test_router.select(linalg_hw("axis_info"), **kwargs).to_metadata()
        for _ in range(20)
    ]

    assert all(decision == decisions[0] for decision in decisions)


def test_selection_is_deterministic_across_registration_order():
    order_a = [triton_gpu_adapter(), hybrid_route_adapter(), shared_adapter(), linalg_adapter()]
    order_b = [linalg_adapter(), shared_adapter(), hybrid_route_adapter(), triton_gpu_adapter()]

    decision_a = router_with(*order_a).select(
        linalg_hw("structured"),
        backend_capabilities=LINALG_BACKEND_CAPS,
    )
    decision_b = router_with(*order_b).select(
        linalg_hw("structured"),
        backend_capabilities=LINALG_BACKEND_CAPS,
    )

    assert decision_a.to_metadata() == decision_b.to_metadata()


def test_strict_mode_blocks_fallback():
    with pytest.raises(AdapterRoutingError) as exc_info:
        router_with(linalg_adapter()).select(
            linalg_hw("structured"),
            backend_capabilities=LINALG_BACKEND_CAPS,
            config=AdapterRouterConfig(allow_fallback=False),
        )

    assert exc_info.value.decision.selected_adapter is None
    assert any(
        rejection.reason == "fallback disabled by AdapterRouterConfig"
        for rejection in exc_info.value.decision.rejected_adapters
    )


def test_allowed_fallback_records_reason():
    metadata = {}
    decision = router_with(linalg_adapter()).select(
        linalg_hw("structured"),
        backend_capabilities=LINALG_BACKEND_CAPS,
        config=AdapterRouterConfig(allow_fallback=True),
        metadata=metadata,
    )

    decision_metadata = metadata["adapter_decision"]
    assert decision.selected_adapter == "triton-linalg"
    assert decision_metadata["fallback"] is True
    assert decision_metadata["fallback_from"] == "triton-shared"
    assert decision_metadata.get("fallback_reason"), decision_metadata
    assert decision_metadata.get("fallback_chain") == [
        "triton-shared",
        "triton-linalg",
    ]


def test_disallowed_fallback_fails_before_convert():
    fallback = linalg_adapter()

    with pytest.raises(AdapterRoutingError):
        router_with(fallback).route(
            linalg_hw("structured"),
            backend_capabilities=LINALG_BACKEND_CAPS,
            config=AdapterRouterConfig(allow_fallback=False),
        )

    assert fallback.calls == 0


def test_hybrid_structured_success_does_not_call_linalg():
    structured = shared_adapter(result="structured-ir")
    axis_info = linalg_adapter(result="axis-info-ir")
    hybrid = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {"adapter_decision": {"selected_adapter": "hybrid"}}

    result = hybrid.convert(
        "ttir",
        metadata,
        AdapterConversionContext(fallback_authorized=True),
    )

    assert result == "structured-ir"
    assert structured.calls == 1
    assert axis_info.calls == 0
    assert metadata["adapter_hybrid"]["fallback_used"] is False
    assert metadata["adapter_decision"]["fallback"] is False


def test_hybrid_structured_failure_allowed_fallback_calls_linalg():
    structured = shared_adapter(fail_reason="structured cannot cover kernel")
    axis_info = linalg_adapter(result="axis-info-ir")
    hybrid = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {"adapter_decision": {"selected_adapter": "hybrid"}}

    result = hybrid.convert(
        "ttir",
        metadata,
        AdapterConversionContext(fallback_authorized=True),
    )

    assert result == "axis-info-ir"
    assert structured.calls == 1
    assert axis_info.calls == 1
    assert metadata["adapter_hybrid"]["fallback_used"] is True
    assert metadata["adapter_decision"]["runtime_adapter"] == "triton-linalg"
    assert metadata["adapter_decision"]["fallback_from"] == "triton-shared"


def test_hybrid_structured_failure_strict_mode_fails():
    structured = shared_adapter(fail_reason="structured cannot cover kernel")
    axis_info = linalg_adapter(result="axis-info-ir")
    hybrid = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {"adapter_decision": {"selected_adapter": "hybrid"}}

    with pytest.raises(AdapterConversionError, match="did not authorize"):
        hybrid.convert(
            "ttir",
            metadata,
            AdapterConversionContext(fallback_authorized=False),
        )

    assert structured.calls == 1
    assert axis_info.calls == 0
    assert metadata["adapter_hybrid"]["fallback_used"] is False


def test_decision_metadata_contains_selected_adapter_and_reasons():
    metadata = {}
    router_with(linalg_adapter()).select(
        linalg_hw("structured"),
        backend_capabilities=LINALG_BACKEND_CAPS,
        config=AdapterRouterConfig(allow_fallback=True),
        metadata=metadata,
    )

    decision_metadata = metadata["adapter_decision"]
    required_fields = {
        "selected_adapter",
        "adapter_decision_policy_version",
        "rejected_adapters",
        "fallback_chain",
        "fallback_reason",
        "anchor_ir_track",
        "ptr_model",
        "preferred_adapter",
        "candidate_order",
        "backend_capabilities",
        "required_ops",
        "hw_name",
        "decision_key",
    }

    missing_fields = sorted(required_fields.difference(decision_metadata))
    assert not missing_fields, f"missing AdapterDecision metadata: {missing_fields}"
    assert decision_metadata["selected_adapter"] == "triton-linalg"
    assert decision_metadata["rejected_adapters"][0]["adapter"] == "triton-shared"
    assert decision_metadata["rejected_adapters"][0]["reason"]


def test_linalg_anchor_ir_rejects_forbidden_triton_gpu_dialect():
    ir_text = """
    module {
      func.func @kernel() {
        %0 = triton_gpu.convert_layout %arg0 : tensor<16xf32>
        %1 = tt.load %arg1 : !tt.ptr<f32>
        return
      }
    }
    """
    validator = AnchorIRValidator(track=AnchorIRTrack.LINALG)

    pre_hook_violations = validator.validate_pre_hook(ir_text)
    post_hook_violations = validator.validate_post_hook(ir_text)

    assert {"triton_gpu", "tt"}.issubset(
        {violation.dialect for violation in pre_hook_violations}
    )
    assert {"triton_gpu", "tt"}.issubset(
        {violation.dialect for violation in post_hook_violations}
    )


def test_triton_gpu_anchor_ir_rejects_transition_dialects():
    ir_text = """
    module {
      func.func @kernel() {
        %0 = tts.load %arg0 : tensor<16xf32>
        %1 = tptr.make_tensor_ptr %arg1 : tensor<16xf32>
        return
      }
    }
    """
    validator = AnchorIRValidator(track=AnchorIRTrack.TRITON_GPU)

    pre_hook_violations = validator.validate_pre_hook(ir_text)
    post_hook_violations = validator.validate_post_hook(ir_text)

    assert {"tts", "tptr"}.issubset(
        {violation.dialect for violation in pre_hook_violations}
    )
    assert {"tts", "tptr"}.issubset(
        {violation.dialect for violation in post_hook_violations}
    )


def test_all_builtin_adapters_are_registered_or_have_clear_unavailable_reason():
    registered = AdapterRegistry.list_adapters()
    required = {
        "triton-gpu": "TritonGPUAdapter",
        "triton-shared": "TritonSharedAdapter",
        "triton-linalg": "TritonLinalgAdapter",
        "hybrid": "HybridAdapter",
    }

    missing = sorted(set(required).difference(registered))
    assert not missing, f"missing builtin adapters without unavailable reason: {missing}"
    for adapter_name, class_name in required.items():
        assert registered[adapter_name] == class_name


def test_invalid_track_ptr_model_combination_fails_closed():
    with pytest.raises(AdapterRoutingError) as exc_info:
        router_with(triton_gpu_adapter(), linalg_adapter()).select(
            gpu_hw(ptr_model="axis_info"),
            backend_capabilities=LINALG_BACKEND_CAPS + TRITON_GPU_BACKEND_CAPS,
            config=AdapterRouterConfig(allow_fallback=True),
        )

    assert exc_info.value.decision.selected_adapter is None
    assert "no routing rule" in exc_info.value.decision.selected_reason


def test_gpu_track_does_not_fallback_to_linalg_whitelist():
    with pytest.raises(AdapterRoutingError) as exc_info:
        router_with(linalg_adapter()).select(
            gpu_hw(),
            backend_capabilities=LINALG_BACKEND_CAPS,
            config=AdapterRouterConfig(allow_fallback=True),
        )

    decision = exc_info.value.decision
    assert decision.selected_adapter is None
    assert decision.fallback_candidates == ()


def test_convert_ttir_to_anchor_ir_rejects_invalid_adapter_output(monkeypatch):
    import triton_anchor.pipeline as pipeline

    bad_ir = """
    module {
      func.func @kernel() {
        %0 = tt.load %arg0 : !tt.ptr<f32>
        return
      }
    }
    """
    bad_adapter = linalg_adapter(result=bad_ir)

    class FakeRoute:
        adapter = bad_adapter

        def conversion_context(self, hw):
            return AdapterConversionContext(hw=hw)

    def fake_route_adapter(hw, *, backend_capabilities, metadata=None, **kwargs):
        metadata["adapter_decision"] = {"selected_adapter": "triton-linalg"}
        return FakeRoute()

    monkeypatch.setattr(pipeline, "route_adapter", fake_route_adapter)

    with pytest.raises(AnchorIRError):
        pipeline.convert_ttir_to_anchor_ir(
            "ttir",
            {},
            linalg_hw("axis_info"),
            backend_capabilities=LINALG_BACKEND_CAPS,
        )
