"""Tests for HWCapability and ComputeParadigm."""

import pytest
from triton_anchor.hw_capability import (
    HWCapability,
    ComputeParadigm,
    MatrixCapability,
    TensorCapability,
    GPGPUCapability,
)
from triton_anchor.anchor_ir import AnchorIRTrack


class TestComputeParadigm:
    def test_enum_values(self):
        assert ComputeParadigm.AME_MATRIX.value == "ame_matrix"
        assert ComputeParadigm.TENSOR_PROCESSOR.value == "tensor"
        assert ComputeParadigm.GPGPU.value == "gpgpu"


class TestHWCapability:
    def test_sophgo_capability(self):
        hw = HWCapability(
            name="sophgo-bm1684x",
            arch_family="tpu",
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
            anchor_ir_track=AnchorIRTrack.LINALG,
            ptr_model="axis_info",
            tensor_cap=TensorCapability(num_cores=8),
        )
        assert hw.name == "sophgo-bm1684x"
        assert hw.compute_paradigm == ComputeParadigm.TENSOR_PROCESSOR
        assert hw.lowering_path == "linalg"

    def test_spacemit_capability(self):
        hw = HWCapability(
            name="spacemit-x60",
            arch_family="riscv",
            compute_paradigm=ComputeParadigm.AME_MATRIX,
            anchor_ir_track=AnchorIRTrack.LINALG,
            ptr_model="structured",
            matrix_cap=MatrixCapability(
                num_matrix_registers=8,
                tile_shape=(8, 8),
            ),
        )
        assert hw.arch_family == "riscv"
        assert hw.matrix_cap.num_matrix_registers == 8

    def test_gpu_capability(self):
        hw = HWCapability(
            name="usc-gpu",
            arch_family="gpu",
            compute_paradigm=ComputeParadigm.GPGPU,
            anchor_ir_track=AnchorIRTrack.TRITON_GPU,
            ptr_model="gpu",
            gpgpu_cap=GPGPUCapability(num_warps=4, warp_size=32),
        )
        assert hw.gpgpu_cap.num_warps == 4

    def test_to_gpu_target(self):
        hw = HWCapability(
            name="sophgo-bm1684x",
            arch_family="tpu",
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
            anchor_ir_track=AnchorIRTrack.LINALG,
            ptr_model="axis_info",
            tensor_cap=TensorCapability(),
        )
        target = hw.to_gpu_target()
        # When triton is not installed, returns a dict
        if isinstance(target, dict):
            assert target["backend"] == "sophgo"
            assert target["warp_size"] == 0
        else:
            assert target.backend == "sophgo"

    def test_validation_missing_matrix_cap(self):
        with pytest.raises(ValueError, match="matrix_cap"):
            HWCapability(
                name="bad",
                arch_family="riscv",
                compute_paradigm=ComputeParadigm.AME_MATRIX,
                anchor_ir_track=AnchorIRTrack.LINALG,
                ptr_model="structured",
                # Missing matrix_cap!
            )

