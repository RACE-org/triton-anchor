"""Tests for AnchorIR validator."""

import pytest
from triton_anchor.anchor_ir import (
    AnchorIRError,
    AnchorIRTrack,
    AnchorIRValidator,
    LINALG_TRACK_ALLOWED,
    LINALG_TRACK_FORBIDDEN,
    TRITON_GPU_TRACK_ALLOWED,
    TRITON_GPU_TRACK_FORBIDDEN,
)


VALID_LINALG_IR = """
module attributes {hw.name = "test"} {
  func.func @kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c128 step %c1 {
      %val = memref.load %arg0[%i] : memref<128xf32>
      %result = math.exp %val : f32
      memref.store %result, %arg1[%i] : memref<128xf32>
    }
    return
  }
}
"""

INVALID_IR_WITH_TT = """
module {
  func.func @kernel(%arg0: !tt.ptr<f32>) {
    %0 = tt.load %arg0 : !tt.ptr<f32>
    %1 = arith.addf %0, %0 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    return
  }
}
"""

MIXED_IR = """
module {
  func.func @kernel(%arg0: memref<128xf32>) {
    %0 = linalg.generic {indexing_maps = [], iterator_types = []}
         ins(%arg0 : memref<128xf32>) {
      ^bb0(%in: f32):
        linalg.yield %in : f32
    } -> tensor<128xf32>
    %1 = tt.splat %0 : tensor<128xf32>
    return
  }
}
"""


class TestAnchorIRValidator:
    def test_linalg_track_configuration_is_frozen(self):
        assert LINALG_TRACK_ALLOWED == {
            "linalg",
            "linalg_ext",
            "tensor",
            "memref",
            "arith",
            "math",
            "math_ext",
            "scf",
            "func",
            "cf",
            "affine",
            "aux",
            "index",
            "bufferization",
            "vector",
        }
        assert LINALG_TRACK_FORBIDDEN == {
            "tt",
            "triton",
            "tts",
            "tptr",
            "smt",
            "triton_gpu",
            "triton_nvidia_gpu",
        }

    def test_triton_gpu_track_configuration_is_frozen(self):
        assert TRITON_GPU_TRACK_ALLOWED == {
            "triton_gpu",
            "tt",
            "arith",
            "cf",
            "math",
            "scf",
            "func",
            "gpu",
            "nvgpu",
        }
        assert TRITON_GPU_TRACK_FORBIDDEN == {"tts", "tptr", "smt"}

    def test_valid_ir(self):
        v = AnchorIRValidator()
        assert v.is_valid(VALID_LINALG_IR)
        assert v.validate(VALID_LINALG_IR) == []

    def test_invalid_ir_forbidden_dialect(self):
        v = AnchorIRValidator()
        violations = v.validate(INVALID_IR_WITH_TT)
        assert len(violations) > 0
        assert any(viol.dialect == "tt" for viol in violations)

    def test_mixed_ir(self):
        v = AnchorIRValidator()
        violations = v.validate(MIXED_IR)
        assert len(violations) > 0
        tt_violations = [v for v in violations if v.dialect == "tt"]
        assert len(tt_violations) > 0

    def test_validate_and_raise(self):
        v = AnchorIRValidator()
        with pytest.raises(AnchorIRError, match="AnchorIR validation failed"):
            v.validate_and_raise(INVALID_IR_WITH_TT, context="test_kernel")

    def test_extra_allowed_dialects(self):
        v = AnchorIRValidator(extra_allowed={"xsmt", "xsmt_async"})
        ir_with_ext = """
        module {
          func.func @kernel() {
            %0 = xsmt.alloc : memref<128xf32>
            return
          }
        }
        """
        assert v.is_valid(ir_with_ext)

    def test_extra_forbidden_dialects(self):
        v = AnchorIRValidator(extra_forbidden={"custom_bad"})
        ir_with_custom = """
        module {
          func.func @kernel() {
            %0 = custom_bad.evil_op : f32
            return
          }
        }
        """
        violations = v.validate(ir_with_custom)
        assert any(viol.dialect == "custom_bad" for viol in violations)

    def test_comments_ignored(self):
        v = AnchorIRValidator()
        ir_with_comments = """
        // tt.load should be ignored in comments
        # tt.store also ignored
        module {
          func.func @kernel(%arg0: memref<128xf32>) {
            return
          }
        }
        """
        assert v.is_valid(ir_with_comments)

    def test_pre_hook_rejects_extension_and_post_hook_allows_it(self):
        validator = AnchorIRValidator(track=AnchorIRTrack.LINALG)
        ir_with_extension = """
        module {
          func.func @kernel() {
            xsmt.mmt4d
            return
          }
        }
        """

        pre = validator.validate_pre_hook(ir_with_extension)
        post = validator.validate_post_hook(
            ir_with_extension, ext_allowed={"xsmt"}
        )

        assert len(pre) == 1
        assert pre[0].dialect == "xsmt"
        assert post == []

    def test_post_hook_extension_cannot_override_forbidden(self):
        validator = AnchorIRValidator(track=AnchorIRTrack.LINALG)
        ir_with_forbidden = """
        module {
          func.func @kernel() {
            tt.return
          }
        }
        """

        violations = validator.validate_post_hook(
            ir_with_forbidden, ext_allowed={"tt"}
        )

        assert len(violations) == 1
        assert violations[0].dialect == "tt"
        assert "Forbidden dialect" in violations[0].message
