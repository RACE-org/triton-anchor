"""T6.1 acceptance tests: multi-adapter routing and fallback policy.

These tests intentionally stay pure Python.  They validate that adapter
selection is deterministic and fail-closed without requiring a GPU, a compiled
kernel, or an external ``triton-shared-opt`` binary.
"""

from __future__ import annotations

import itertools

import pytest

from triton_anchor.adapters.base import (
    AdapterConversionError,
    AdapterNotFoundError,
    ITritonToLinalgAdapter,
)
from triton_anchor.adapters.hybrid_adapter import HybridAdapter
from triton_anchor.adapters.registry import AdapterRegistry
from triton_anchor.adapters.router import (
    ADAPTER_ROUTING_POLICY_VERSION,
    AdapterRouter,
    AdapterRoutingPolicy,
)
from triton_anchor.adapters.triton_gpu_adapter import TritonGPUAdapter
from triton_anchor.adapters.triton_linalg_adapter import TritonLinalgAdapter
from triton_anchor.adapters.triton_shared_adapter import TritonSharedAdapter
from triton_anchor.anchor_ir import AnchorIRTrack, AnchorIRValidator
from triton_anchor.hw_capability import (
    ComputeParadigm,
    GPGPUCapability,
    HWCapability,
    MatrixCapability,
    TensorCapability,
)


class FakeAdapter(ITritonToLinalgAdapter):
    def __init__(self, name, tracks, ptr_models, result=None, fail=False):
        self._name = name
        self._tracks = list(tracks)
        self._ptr_models = list(ptr_models)
        self.result = result if result is not None else f"{name}-ir"
        self.fail = fail
        self.convert_calls = 0

    def name(self):
        return self._name

    def get_supported_tracks(self):
        return list(self._tracks)

    def get_supported_ptr_models(self):
        return list(self._ptr_models)

    def convert(self, ttir_module, metadata, context=None):
        self.convert_calls += 1
        if self.fail:
            raise AdapterConversionError(self.name(), detail="fake conversion failure")
        metadata.setdefault("convert_chain", []).append(self.name())
        return self.result


def _builtin_like_adapters():
    return {
        "triton-gpu": FakeAdapter("triton-gpu", ["triton_gpu"], ["gpu"]),
        "triton-shared": FakeAdapter("triton-shared", ["linalg"], ["structured"]),
        "triton-linalg": FakeAdapter("triton-linalg", ["linalg"], ["axis_info"]),
        "hybrid": FakeAdapter("hybrid", ["linalg"], ["hybrid"]),
    }


def _matrix_hw(**overrides):
    values = {
        "name": "acceptance-structured",
        "arch_family": "riscv",
        "compute_paradigm": ComputeParadigm.AME_MATRIX,
        "anchor_ir_track": AnchorIRTrack.LINALG,
        "ptr_model": "structured",
        "matrix_cap": MatrixCapability(),
    }
    values.update(overrides)
    return HWCapability(**values)


def _tensor_hw(**overrides):
    values = {
        "name": "acceptance-axis-info",
        "arch_family": "tpu",
        "compute_paradigm": ComputeParadigm.TENSOR_PROCESSOR,
        "anchor_ir_track": AnchorIRTrack.LINALG,
        "ptr_model": "axis_info",
        "tensor_cap": TensorCapability(num_cores=8),
    }
    values.update(overrides)
    return HWCapability(**values)


def _gpu_hw(**overrides):
    values = {
        "name": "acceptance-gpu",
        "arch_family": "gpu",
        "compute_paradigm": ComputeParadigm.GPGPU,
        "anchor_ir_track": AnchorIRTrack.TRITON_GPU,
        "ptr_model": "gpu",
        "gpgpu_cap": GPGPUCapability(num_warps=4, warp_size=32),
    }
    values.update(overrides)
    return HWCapability(**values)


@pytest.fixture(autouse=True)
def _preserve_adapter_registry():
    original_adapters = dict(AdapterRegistry._adapters)
    original_discovered = AdapterRegistry._discovered
    yield
    AdapterRegistry._adapters.clear()
    AdapterRegistry._adapters.update(original_adapters)
    AdapterRegistry._discovered = original_discovered


def _assert_selected(hw, expected):
    metadata = {}
    decision = AdapterRouter(adapters=_builtin_like_adapters()).select(
        hw, metadata=metadata
    )
    assert decision.selected_adapter == expected
    assert decision.adapter.name() == expected
    assert metadata["selected_adapter"] == expected
    assert metadata["adapter_policy_version"] == ADAPTER_ROUTING_POLICY_VERSION
    return metadata


def test_router_selects_triton_gpu_for_triton_gpu_track():
    _assert_selected(_gpu_hw(), "triton-gpu")


def test_router_selects_shared_for_structured_linalg():
    _assert_selected(_matrix_hw(ptr_model="structured"), "triton-shared")


def test_router_selects_linalg_for_axis_info_linalg():
    _assert_selected(_tensor_hw(ptr_model="axis_info"), "triton-linalg")


def test_router_selects_hybrid_for_hybrid_linalg():
    _assert_selected(_tensor_hw(ptr_model="hybrid"), "hybrid")


def test_preferred_adapter_wins_when_available_and_compatible():
    adapters = _builtin_like_adapters()
    adapters["custom-axis"] = FakeAdapter("custom-axis", ["linalg"], ["axis_info"])
    metadata = {}

    decision = AdapterRouter(adapters=adapters).select(
        _tensor_hw(preferred_adapter="custom-axis"),
        metadata=metadata,
    )

    assert decision.selected_adapter == "custom-axis"
    assert metadata["selected_adapter"] == "custom-axis"
    assert metadata["adapter_requested_adapter"] == "custom-axis"
    assert "preferred adapter matched" in metadata["adapter_selection_reason"]


def test_preferred_adapter_missing_fails():
    metadata = {}

    with pytest.raises(AdapterNotFoundError):
        AdapterRouter(adapters=_builtin_like_adapters()).select(
            _tensor_hw(preferred_adapter="missing-adapter"),
            metadata=metadata,
        )

    assert metadata["selected_adapter"] is None
    assert metadata["adapter_requested_adapter"] == "missing-adapter"
    assert "preferred adapter is not registered" in metadata["adapter_reject_reason"]


def test_preferred_adapter_incompatible_fails():
    metadata = {}

    with pytest.raises(AdapterNotFoundError):
        AdapterRouter(adapters=_builtin_like_adapters()).select(
            _tensor_hw(preferred_adapter="triton-gpu"),
            metadata=metadata,
        )

    assert metadata["selected_adapter"] is None
    assert metadata["adapter_requested_adapter"] == "triton-gpu"
    assert "track 'linalg'" in metadata["adapter_reject_reason"]


def test_no_implicit_registry_order_fallback():
    AdapterRegistry.reset()
    AdapterRegistry.register(FakeAdapter("triton-linalg", ["linalg"], ["axis_info"]))
    AdapterRegistry.register(FakeAdapter("hybrid", ["linalg"], ["hybrid"]))
    AdapterRegistry._discovered = True
    metadata = {}

    with pytest.raises(AdapterNotFoundError):
        AdapterRegistry.get_adapter(_matrix_hw(ptr_model="structured"), metadata)

    assert metadata["selected_adapter"] is None
    assert metadata["adapter_reject_reasons"]["triton-shared"] == (
        "required adapter is not registered"
    )
    assert metadata["adapter_fallback_reason"] == ""


def test_selection_is_deterministic_across_repeated_runs():
    router = AdapterRouter(adapters=_builtin_like_adapters())
    hw = _tensor_hw(ptr_model="axis_info")
    selections = []
    snapshots = []

    for _ in range(20):
        metadata = {}
        decision = router.select(hw, metadata=metadata)
        selections.append(decision.selected_adapter)
        snapshots.append(
            {
                "selected_adapter": metadata["selected_adapter"],
                "adapter_policy_version": metadata["adapter_policy_version"],
                "adapter_reject_reason": metadata["adapter_reject_reason"],
                "adapter_reject_reasons": metadata["adapter_reject_reasons"],
                "adapter_fallback_reason": metadata["adapter_fallback_reason"],
                "adapter_requested_adapter": metadata["adapter_requested_adapter"],
                "adapter_candidates": metadata["adapter_candidates"],
            }
        )

    assert selections == ["triton-linalg"] * 20
    assert snapshots == [snapshots[0]] * 20


def test_selection_is_deterministic_across_registration_order():
    items = list(_builtin_like_adapters().items())
    hw_cases = [
        (_gpu_hw(), "triton-gpu"),
        (_matrix_hw(ptr_model="structured"), "triton-shared"),
        (_tensor_hw(ptr_model="axis_info"), "triton-linalg"),
        (_tensor_hw(ptr_model="hybrid"), "hybrid"),
    ]

    for order in itertools.permutations(items):
        adapters = dict(order)
        for hw, expected in hw_cases:
            metadata = {}
            decision = AdapterRouter(adapters=adapters).select(hw, metadata=metadata)
            assert decision.selected_adapter == expected
            assert metadata["adapter_candidates"] == sorted(adapters)


def test_strict_mode_blocks_fallback():
    metadata = {}
    router = AdapterRouter(
        adapters={"triton-linalg": FakeAdapter("triton-linalg", ["linalg"], ["axis_info"])},
        policy=AdapterRoutingPolicy(
            strict=True,
            fallback_allowlist={"triton-shared": ("triton-linalg",)},
        ),
    )

    with pytest.raises(AdapterNotFoundError):
        router.select(_matrix_hw(ptr_model="structured"), metadata=metadata)

    assert metadata["selected_adapter"] is None
    assert metadata["adapter_reject_reasons"]["fallback"] == (
        "strict policy forbids fallback"
    )


def test_allowed_fallback_records_reason():
    metadata = {}
    router = AdapterRouter(
        adapters={"triton-linalg": FakeAdapter("triton-linalg", ["linalg"], ["axis_info"])},
        policy=AdapterRoutingPolicy(
            strict=False,
            fallback_allowlist={"triton-shared": ("triton-linalg",)},
        ),
    )

    decision = router.select(_matrix_hw(ptr_model="structured"), metadata=metadata)

    assert decision.selected_adapter == "triton-linalg"
    assert metadata["adapter_requested_adapter"] == "triton-shared"
    assert "policy allowlisted fallback" in metadata["adapter_fallback_reason"]
    assert metadata["adapter_reject_reasons"]["triton-shared"] == (
        "required adapter is not registered"
    )


def test_disallowed_fallback_fails_before_convert():
    fallback = FakeAdapter("triton-linalg", ["linalg"], ["axis_info"])
    metadata = {}
    router = AdapterRouter(
        adapters={"triton-linalg": fallback},
        policy=AdapterRoutingPolicy(strict=False, fallback_allowlist={}),
    )

    with pytest.raises(AdapterNotFoundError):
        router.select(_matrix_hw(ptr_model="structured"), metadata=metadata)

    assert fallback.convert_calls == 0
    assert metadata["selected_adapter"] is None
    assert metadata["adapter_reject_reasons"]["fallback"] == (
        "no allowlisted fallback available for 'triton-shared'"
    )


def test_hybrid_structured_success_does_not_call_linalg(monkeypatch):
    calls = {"shared": 0, "linalg": 0}

    class SharedSuccess:
        def convert(self, ttir_module, metadata, context=None):
            calls["shared"] += 1
            metadata["hybrid_ptr_analysis"] = "structured"
            return "structured-ir"

    class LinalgShouldNotRun:
        def convert(self, ttir_module, metadata, context=None):
            calls["linalg"] += 1
            raise AssertionError("linalg fallback must not run after shared success")

    monkeypatch.setattr(
        "triton_anchor.adapters.triton_shared_adapter.TritonSharedAdapter",
        SharedSuccess,
    )
    monkeypatch.setattr(
        "triton_anchor.adapters.triton_linalg_adapter.TritonLinalgAdapter",
        LinalgShouldNotRun,
    )

    metadata = {}
    result = HybridAdapter().convert("ttir", metadata)

    assert result == "structured-ir"
    assert calls == {"shared": 1, "linalg": 0}
    assert metadata["hybrid_ptr_analysis"] == "structured"
    assert metadata.get("adapter_fallback_reason", "") == ""


def test_hybrid_structured_failure_allowed_fallback_calls_linalg(monkeypatch):
    calls = {"shared": 0, "linalg": 0}

    class SharedFailure:
        def convert(self, ttir_module, metadata, context=None):
            calls["shared"] += 1
            raise AdapterConversionError("triton-shared", detail="structured failed")

    class LinalgFallback:
        def convert(self, ttir_module, metadata, context=None):
            calls["linalg"] += 1
            return "axis-info-ir"

    monkeypatch.setattr(
        "triton_anchor.adapters.triton_shared_adapter.TritonSharedAdapter",
        SharedFailure,
    )
    monkeypatch.setattr(
        "triton_anchor.adapters.triton_linalg_adapter.TritonLinalgAdapter",
        LinalgFallback,
    )

    metadata = {"adapter_policy_strict": False}
    result = HybridAdapter().convert("ttir", metadata)

    assert result == "axis-info-ir"
    assert calls == {"shared": 1, "linalg": 1}
    assert metadata["hybrid_ptr_analysis"] == "axis_info"
    assert "hybrid structured path failed" in metadata["adapter_fallback_reason"]


def test_hybrid_structured_failure_strict_mode_fails(monkeypatch):
    calls = {"shared": 0, "linalg": 0}

    class SharedFailure:
        def convert(self, ttir_module, metadata, context=None):
            calls["shared"] += 1
            raise AdapterConversionError("triton-shared", detail="structured failed")

    class LinalgFallback:
        def convert(self, ttir_module, metadata, context=None):
            calls["linalg"] += 1
            return "axis-info-ir"

    monkeypatch.setattr(
        "triton_anchor.adapters.triton_shared_adapter.TritonSharedAdapter",
        SharedFailure,
    )
    monkeypatch.setattr(
        "triton_anchor.adapters.triton_linalg_adapter.TritonLinalgAdapter",
        LinalgFallback,
    )

    metadata = {"adapter_policy_strict": True}
    with pytest.raises(AdapterConversionError):
        HybridAdapter().convert("ttir", metadata)

    assert calls == {"shared": 1, "linalg": 0}


def test_decision_metadata_contains_selected_adapter_and_reasons():
    metadata = {}
    router = AdapterRouter(
        adapters={"triton-linalg": FakeAdapter("triton-linalg", ["linalg"], ["axis_info"])},
        policy=AdapterRoutingPolicy(
            strict=False,
            fallback_allowlist={"triton-shared": ("triton-linalg",)},
        ),
    )

    router.select(_matrix_hw(ptr_model="structured"), metadata=metadata)

    for key in (
        "selected_adapter",
        "adapter_policy_version",
        "adapter_reject_reason",
        "adapter_reject_reasons",
        "adapter_fallback_reason",
        "adapter_selection_reason",
        "adapter_requested_adapter",
        "adapter_candidates",
    ):
        assert key in metadata
    assert metadata["selected_adapter"] == "triton-linalg"
    assert metadata["adapter_policy_version"] == ADAPTER_ROUTING_POLICY_VERSION
    assert metadata["adapter_reject_reasons"]
    assert metadata["adapter_requested_adapter"] == "triton-shared"
    assert metadata["adapter_candidates"] == ["triton-linalg"]
    assert metadata["adapter_fallback_reason"]
    assert metadata.get("adapter_fallback_chain") == [
        "triton-shared",
        "triton-linalg",
    ]


def test_linalg_anchor_ir_rejects_forbidden_triton_gpu_dialect():
    ir_text = """
    func.func @bad(%arg0: tensor<16xf32>) -> tensor<16xf32> {
      %0 = triton_gpu.convert_layout %arg0 : tensor<16xf32> to tensor<16xf32>
      func.return %0 : tensor<16xf32>
    }
    """
    validator = AnchorIRValidator(track=AnchorIRTrack.LINALG)

    pre = validator.validate_pre_hook(ir_text)
    post = validator.validate_post_hook(ir_text)

    assert any(v.dialect == "triton_gpu" for v in pre)
    assert any(v.dialect == "triton_gpu" for v in post)


def test_triton_gpu_anchor_ir_rejects_transition_dialects():
    ir_text = """
    func.func @bad(%arg0: tensor<16xf32>) -> tensor<16xf32> {
      %0 = tptr.reinterpret %arg0 : tensor<16xf32> to tensor<16xf32>
      %1 = tts.structured_load %0 : tensor<16xf32>
      func.return %1 : tensor<16xf32>
    }
    """
    validator = AnchorIRValidator(track=AnchorIRTrack.TRITON_GPU)

    pre = validator.validate_pre_hook(ir_text)
    post = validator.validate_post_hook(ir_text)

    assert {v.dialect for v in pre} >= {"tptr", "tts"}
    assert {v.dialect for v in post} >= {"tptr", "tts"}


def test_all_builtin_adapters_are_registered_or_have_clear_unavailable_reason():
    import triton_anchor.adapters as adapters_module

    AdapterRegistry.reset()
    adapters_module.register_builtin_adapters()

    registered = AdapterRegistry.list_adapters()

    assert registered["triton-gpu"] == TritonGPUAdapter.__name__
    assert registered["triton-shared"] == TritonSharedAdapter.__name__
    assert registered["triton-linalg"] == TritonLinalgAdapter.__name__
    assert registered["hybrid"] == HybridAdapter.__name__
