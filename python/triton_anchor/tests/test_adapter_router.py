"""Tests for T6.1 adapter routing policy."""

import pytest

from triton_anchor.adapters.base import AdapterConversionError, IAnchorAdapter
from triton_anchor.adapters.hybrid_adapter import HybridAdapter
from triton_anchor.adapters.router import (
    AdapterRouter,
    AdapterRouterConfig,
    AdapterRoutingError,
    OpCoverage,
)
from triton_anchor.anchor_ir import AnchorIRTrack
from triton_anchor.hw_capability import (
    ComputeParadigm,
    GPGPUCapability,
    HWCapability,
    TensorCapability,
)


class DummyAdapter(IAnchorAdapter):
    def __init__(
        self,
        name,
        *,
        tracks=(AnchorIRTrack.LINALG,),
        ptr_models=("axis_info",),
        required_caps=("anchor_ir.linalg",),
        supported_ops=("*",),
    ):
        self._name = name
        self._tracks = tracks
        self._ptr_models = ptr_models
        self._required_caps = required_caps
        self._supported_ops = supported_ops

    def name(self):
        return self._name

    def convert(self, ttir_module, metadata, context=None):
        metadata["converted_by"] = self._name
        return ttir_module

    def get_supported_tracks(self):
        return self._tracks

    def get_supported_ptr_models(self):
        return self._ptr_models

    def get_required_backend_capabilities(self):
        return self._required_caps

    def get_supported_ops(self):
        return self._supported_ops


class FailingAdapter(DummyAdapter):
    def convert(self, ttir_module, metadata, context=None):
        raise AdapterConversionError(self.name(), detail="forced failure")


def tensor_hw(**overrides):
    kwargs = dict(
        name="sophgo-bm1684x",
        arch_family="tpu",
        compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
        anchor_ir_track=AnchorIRTrack.LINALG,
        ptr_model="axis_info",
        tensor_cap=TensorCapability(num_cores=8),
    )
    kwargs.update(overrides)
    return HWCapability(**kwargs)


def gpu_hw(**overrides):
    kwargs = dict(
        name="usc-gpu",
        arch_family="gpu",
        compute_paradigm=ComputeParadigm.GPGPU,
        anchor_ir_track=AnchorIRTrack.TRITON_GPU,
        ptr_model="gpu",
        gpgpu_cap=GPGPUCapability(num_warps=4, warp_size=32),
    )
    kwargs.update(overrides)
    return HWCapability(**kwargs)


def router(adapters):
    return AdapterRouter(adapters={adapter.name(): adapter for adapter in adapters})


def test_normal_axis_info_selection():
    decision = router([DummyAdapter("triton-linalg")]).select(
        tensor_hw(),
        backend_capabilities=("anchor_ir.linalg",),
    )

    assert decision.selected_adapter == "triton-linalg"
    assert decision.fallback is False
    assert "ptr_model 'axis_info'" in decision.selected_reason


def test_normal_triton_gpu_selection():
    adapter = DummyAdapter(
        "triton-gpu",
        tracks=(AnchorIRTrack.TRITON_GPU,),
        ptr_models=("gpu",),
        required_caps=("anchor_ir.triton_gpu",),
    )
    decision = router([adapter]).select(
        gpu_hw(),
        backend_capabilities=("anchor_ir.triton_gpu",),
    )

    assert decision.selected_adapter == "triton-gpu"
    assert decision.anchor_ir_track == AnchorIRTrack.TRITON_GPU


def test_preferred_adapter_overrides_ptr_model_but_not_track():
    hw = tensor_hw(ptr_model="structured", preferred_adapter="triton-linalg")
    decision = router([DummyAdapter("triton-linalg")]).select(
        hw,
        backend_capabilities=("anchor_ir.linalg",),
    )

    assert decision.selected_adapter == "triton-linalg"
    assert decision.preferred_adapter == "triton-linalg"
    assert "preferred adapter" in decision.selected_reason


def test_capability_mismatch_rejects_adapter():
    with pytest.raises(AdapterRoutingError) as exc_info:
        router([DummyAdapter("triton-linalg")]).select(
            tensor_hw(),
            backend_capabilities=("anchor_ir.triton_gpu",),
        )

    decision = exc_info.value.decision
    assert decision.selected_adapter is None
    assert decision.rejected_adapters[0].adapter_name == "triton-linalg"
    assert decision.rejected_adapters[0].reason == "missing backend capabilities"


def test_fallback_disallowed_records_rejection():
    hw = tensor_hw(ptr_model="structured")

    with pytest.raises(AdapterRoutingError) as exc_info:
        router([DummyAdapter("triton-linalg")]).select(
            hw,
            backend_capabilities=("anchor_ir.linalg",),
            config=AdapterRouterConfig(allow_fallback=False),
        )

    decision = exc_info.value.decision
    assert decision.selected_adapter is None
    assert [r.adapter_name for r in decision.rejected_adapters] == [
        "triton-shared",
        "triton-linalg",
    ]
    assert decision.rejected_adapters[1].reason == (
        "fallback disabled by AdapterRouterConfig"
    )


def test_fallback_allowed_selects_axis_info():
    hw = tensor_hw(ptr_model="structured")
    decision = router([DummyAdapter("triton-linalg")]).select(
        hw,
        backend_capabilities=("anchor_ir.linalg",),
        config=AdapterRouterConfig(allow_fallback=True),
    )

    assert decision.selected_adapter == "triton-linalg"
    assert decision.fallback is True
    assert decision.fallback_from == "triton-shared"
    assert decision.fallback_authorized is True


def test_op_coverage_rejects_missing_ops():
    coverage = OpCoverage(
        required_ops=("tt.load", "tt.store", "tt.dot"),
        adapter_supported_ops={"triton-linalg": ("tt.load", "tt.store")},
    )

    with pytest.raises(AdapterRoutingError) as exc_info:
        router([DummyAdapter("triton-linalg")]).select(
            tensor_hw(),
            backend_capabilities=("anchor_ir.linalg",),
            op_coverage=coverage,
        )

    rejection = exc_info.value.decision.rejected_adapters[0]
    assert rejection.reason == "missing op coverage"
    assert rejection.details == ("tt.dot",)


def test_repeated_selection_is_stable():
    r = router([DummyAdapter("triton-linalg")])
    kwargs = dict(
        backend_capabilities=("anchor_ir.linalg",),
        op_coverage=OpCoverage(required_ops=("tt.load",)),
    )

    first = r.select(tensor_hw(), **kwargs).to_metadata()
    second = r.select(tensor_hw(), **kwargs).to_metadata()

    assert first == second


def test_selection_reason_written_to_metadata():
    metadata = {}
    decision = router([DummyAdapter("triton-linalg")]).select(
        tensor_hw(),
        backend_capabilities=("anchor_ir.linalg",),
        metadata=metadata,
    )

    assert metadata["adapter_decision"]["selected_adapter"] == decision.selected_adapter
    assert metadata["adapter_decision"]["selected_reason"] == decision.selected_reason


def test_registry_compat_entry_does_not_pick_arbitrary_fallback():
    from triton_anchor.adapters.registry import AdapterNotFoundError, AdapterRegistry

    class LocalRegistry(AdapterRegistry):
        _adapters = {}
        _discovered = True

    LocalRegistry.register(DummyAdapter("triton-linalg"))

    with pytest.raises(AdapterNotFoundError):
        LocalRegistry.get_adapter(tensor_hw(ptr_model="structured"))


def test_hybrid_adapter_requires_router_fallback_authorization():
    hybrid = HybridAdapter(
        structured_adapter=FailingAdapter("triton-shared", ptr_models=("structured",)),
        axis_info_adapter=DummyAdapter("triton-linalg"),
    )
    metadata = {
        "adapter_decision": {
            "selected_adapter": "hybrid",
            "fallback_authorized": False,
        }
    }

    with pytest.raises(AdapterConversionError, match="did not authorize"):
        hybrid.convert("ttir", metadata)

    assert metadata["adapter_hybrid"]["fallback_used"] is False


def test_hybrid_adapter_uses_authorized_axis_info_fallback():
    hybrid = HybridAdapter(
        structured_adapter=FailingAdapter("triton-shared", ptr_models=("structured",)),
        axis_info_adapter=DummyAdapter("triton-linalg"),
    )
    metadata = {}
    route = router([hybrid]).route(
        tensor_hw(ptr_model="hybrid"),
        backend_capabilities=("anchor_ir.linalg",),
        config=AdapterRouterConfig(allow_fallback=True),
        metadata=metadata,
    )

    result = hybrid.convert("ttir", metadata, route.conversion_context(tensor_hw()))

    assert result == "ttir"
    assert metadata["converted_by"] == "triton-linalg"
    assert metadata["adapter_hybrid"]["fallback_used"] is True
    assert metadata["adapter_decision"]["runtime_adapter"] == "triton-linalg"
    assert metadata["adapter_decision"]["fallback"] is True
