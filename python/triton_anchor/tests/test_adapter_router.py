"""Tests for T6.1 deterministic adapter routing."""

import pytest

from triton_anchor.adapters.base import AdapterNotFoundError, ITritonToLinalgAdapter
from triton_anchor.adapters.router import AdapterRouter, AdapterRoutingPolicy
from triton_anchor.adapters.triton_gpu_adapter import TritonGPUAdapter
from triton_anchor.anchor_ir import AnchorIRTrack
from triton_anchor.hw_capability import (
    ComputeParadigm,
    GPGPUCapability,
    HWCapability,
    MatrixCapability,
    TensorCapability,
)


class DummyAdapter(ITritonToLinalgAdapter):
    def __init__(self, name, tracks, ptr_models):
        self._name = name
        self._tracks = tracks
        self._ptr_models = ptr_models

    def name(self):
        return self._name

    def convert(self, ttir_module, metadata, context=None):
        return ttir_module

    def get_supported_tracks(self):
        return list(self._tracks)

    def get_supported_ptr_models(self):
        return list(self._ptr_models)


def _adapters():
    return {
        "hybrid": DummyAdapter("hybrid", ["linalg"], ["hybrid"]),
        "triton-gpu": DummyAdapter("triton-gpu", ["triton_gpu"], ["gpu"]),
        "triton-linalg": DummyAdapter("triton-linalg", ["linalg"], ["axis_info"]),
        "triton-shared": DummyAdapter("triton-shared", ["linalg"], ["structured"]),
    }


def _tensor_hw(**kwargs):
    values = dict(
        name="sophgo-bm1684x",
        arch_family="tpu",
        compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
        anchor_ir_track=AnchorIRTrack.LINALG,
        ptr_model="axis_info",
        tensor_cap=TensorCapability(num_cores=8),
    )
    values.update(kwargs)
    return HWCapability(**values)


def _matrix_hw(**kwargs):
    values = dict(
        name="spacemit-x60",
        arch_family="riscv",
        compute_paradigm=ComputeParadigm.AME_MATRIX,
        anchor_ir_track=AnchorIRTrack.LINALG,
        ptr_model="structured",
        matrix_cap=MatrixCapability(),
    )
    values.update(kwargs)
    return HWCapability(**values)


def _gpu_hw(**kwargs):
    values = dict(
        name="usc-gpu",
        arch_family="gpu",
        compute_paradigm=ComputeParadigm.GPGPU,
        anchor_ir_track=AnchorIRTrack.TRITON_GPU,
        ptr_model="gpu",
        gpgpu_cap=GPGPUCapability(num_warps=4, warp_size=32),
    )
    values.update(kwargs)
    return HWCapability(**values)


def test_t61_router_is_deterministic_for_same_inputs():
    router = AdapterRouter(adapters=_adapters())
    cases = (
        (_tensor_hw(), "triton-linalg"),
        (_matrix_hw(), "triton-shared"),
        (_tensor_hw(ptr_model="hybrid"), "hybrid"),
        (_gpu_hw(), "triton-gpu"),
    )

    for hw, expected in cases:
        selected = []
        metadata_snapshots = []
        for _ in range(5):
            metadata = {}
            decision = router.select(hw, metadata=metadata)
            selected.append(decision.selected_adapter)
            metadata_snapshots.append(
                {
                    "selected_adapter": metadata["selected_adapter"],
                    "adapter_policy_version": metadata["adapter_policy_version"],
                    "adapter_reject_reason": metadata["adapter_reject_reason"],
                    "adapter_reject_reasons": metadata["adapter_reject_reasons"],
                    "adapter_fallback_reason": metadata["adapter_fallback_reason"],
                    "adapter_candidates": metadata["adapter_candidates"],
                }
            )

        assert selected == [expected] * 5
        assert metadata_snapshots == [metadata_snapshots[0]] * 5


def test_t61_router_does_not_implicitly_fallback_to_available_adapter():
    router = AdapterRouter(
        adapters={
            "triton-linalg": DummyAdapter(
                "triton-linalg", ["linalg"], ["axis_info"]
            )
        }
    )
    metadata = {}

    with pytest.raises(AdapterNotFoundError):
        router.select(_matrix_hw(), metadata=metadata)

    assert metadata["selected_adapter"] is None
    assert metadata["adapter_fallback_reason"] == ""
    assert metadata["adapter_reject_reasons"]["triton-shared"] == (
        "required adapter is not registered"
    )
    assert metadata["adapter_reject_reasons"]["fallback"] == (
        "strict policy forbids fallback"
    )


def test_t61_router_allows_declared_fallback_only_when_policy_allows_it():
    policy = AdapterRoutingPolicy(
        strict=False,
        fallback_allowlist={"triton-shared": ("triton-linalg",)},
    )
    router = AdapterRouter(
        adapters={
            "triton-linalg": DummyAdapter(
                "triton-linalg", ["linalg"], ["axis_info"]
            )
        },
        policy=policy,
    )
    metadata = {}

    decision = router.select(_matrix_hw(), metadata=metadata)

    assert decision.selected_adapter == "triton-linalg"
    assert metadata["selected_adapter"] == "triton-linalg"
    assert "policy allowlisted fallback" in metadata["adapter_fallback_reason"]
    assert metadata["adapter_reject_reasons"]["triton-shared"] == (
        "required adapter is not registered"
    )


def test_t61_router_rejects_capability_conflict_before_adapter_execution():
    router = AdapterRouter(adapters=_adapters())
    metadata = {}

    with pytest.raises(AdapterNotFoundError):
        router.select(_gpu_hw(ptr_model="axis_info"), metadata=metadata)

    assert metadata["selected_adapter"] is None
    assert "requires ptr_model 'gpu'" in metadata["adapter_reject_reason"]


def test_t61_router_rejects_incompatible_preferred_adapter():
    router = AdapterRouter(adapters=_adapters())
    metadata = {}

    with pytest.raises(AdapterNotFoundError):
        router.select(
            _tensor_hw(preferred_adapter="triton-gpu"),
            metadata=metadata,
        )

    assert metadata["selected_adapter"] is None
    assert metadata["adapter_requested_adapter"] == "triton-gpu"
    assert "track 'linalg'" in metadata["adapter_reject_reason"]


def test_t61_triton_gpu_adapter_formats_backend_targets():
    adapter = TritonGPUAdapter()

    assert adapter._target_name(
        {"target": {"backend": "cuda", "arch": 80}}
    ) == "cuda:80"
    assert adapter._target_name(
        {"target": {"backend": "hip", "arch": "gfx942"}}
    ) == "hip:gfx942"
    assert adapter._target_name({"triton_gpu_target": "cuda:90"}) == "cuda:90"


def test_t61_triton_gpu_adapter_skips_cuda_only_passes_for_generic_target():
    class FakePassModule:
        def __getattr__(self, name):
            def add_pass(*args):
                return None

            return add_pass

    class FakePasses:
        ttgpuir = FakePassModule()
        common = FakePassModule()

    adapter = TritonGPUAdapter()

    generic_passes = adapter._add_ttgpuir_passes(None, FakePasses, 3, "gpu")
    cuda_passes = adapter._add_ttgpuir_passes(None, FakePasses, 3, "cuda:80")

    cuda_only = {
        "add_accelerate_matmul",
        "add_f32_dot_tc",
        "add_optimize_dot_operands",
    }
    assert cuda_only.isdisjoint(generic_passes)
    assert cuda_only.issubset(cuda_passes)
