"""Tests for the Anchor-owned backend stage pipeline."""

from dataclasses import dataclass

import pytest

pytest.importorskip("triton._C.libtriton")

from triton.backends.compiler import GPUTarget
from triton_anchor.backend import AnchorBackendBase
from triton_anchor.anchor_ir import AnchorIRTrack
from triton_anchor.hw_capability import (
    ComputeParadigm,
    GPGPUCapability,
    HWCapability,
    TensorCapability,
)


@dataclass(frozen=True)
class _Options:
    validate_anchor_ir: bool = False

    def hash(self):
        return "test"


class _TestBackend(AnchorBackendBase):
    binary_ext = "bin"

    def __init__(self, track):
        self._track = track
        super().__init__(GPUTarget("test", 0, 32))

    @staticmethod
    def supports_target(target):
        return target.backend == "test"

    def hash(self):
        return "test-anchor-backend"

    def parse_options(self, options):
        return _Options()

    def get_hw_capability(self, options):
        if self._track == AnchorIRTrack.LINALG:
            return HWCapability(
                name="test-tensor",
                arch_family="tpu",
                compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
                anchor_ir_track=self._track,
                ptr_model="axis_info",
                preferred_adapter="triton-linalg",
                tensor_cap=TensorCapability(),
            )
        return HWCapability(
            name="test-gpu",
            arch_family="gpu",
            compute_paradigm=ComputeParadigm.GPGPU,
            anchor_ir_track=self._track,
            ptr_model="gpu",
            gpgpu_cap=GPGPUCapability(),
        )

    def add_vendor_stages(self, stages, options, ctx):
        stages["bin"] = lambda mod, metadata: b"binary"


class _OverwritingBackend(_TestBackend):
    def add_vendor_stages(self, stages, options, ctx):
        stages["ttir"] = lambda mod, metadata: mod
        stages["bin"] = lambda mod, metadata: b"binary"


def test_linalg_track_selects_adapter_and_owns_first_two_stages():
    backend = _TestBackend(AnchorIRTrack.LINALG)
    options = _Options()

    ctx = backend.prepare_anchor_compilation(
        backend.get_hw_capability(options), options
    )
    stages = {}
    backend.add_stages(stages, options)

    assert ctx.track == AnchorIRTrack.LINALG
    assert ctx.adapter is not None
    assert ctx.adapter.name() == "triton-linalg"
    assert list(stages) == ["ttir", "linalg", "bin"]


def test_triton_gpu_track_does_not_select_linalg_adapter():
    backend = _TestBackend(AnchorIRTrack.TRITON_GPU)
    options = _Options()

    ctx = backend.prepare_anchor_compilation(
        backend.get_hw_capability(options), options
    )
    stages = {}
    backend.add_stages(stages, options)

    assert ctx.track == AnchorIRTrack.TRITON_GPU
    assert ctx.adapter is None
    assert list(stages) == ["ttir", "ttgir", "bin"]


def test_vendor_cannot_replace_anchor_owned_stages():
    backend = _OverwritingBackend(AnchorIRTrack.LINALG)

    with pytest.raises(RuntimeError, match="attempted to replace"):
        backend.add_stages({}, _Options())
