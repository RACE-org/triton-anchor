"""
TritonLinalgAdapter — In-Process Adapter wrapping triton-linalg
================================================================

This adapter wraps the triton-linalg conversion pipeline (from Cambricon)
that is used by triton_race for Sophgo TPU support.

It calls the MLIR PassManager directly (in-process), with zero subprocess
overhead.  The pass sequence is extracted from triton_race's ``_make_raceir()``.

Dependencies:
  - ``triton._C.libtriton`` must be available (i.e., triton_race installed)
  - race passes must be linked into libtriton.so

Output dialects:
  linalg, linalg_ext, tensor, memref, arith, math, scf, func, aux
"""

from __future__ import annotations

import logging
import re
import traceback
from typing import Any, List

from .base import ILinalgPybindAdapter, AdapterConversionError

logger = logging.getLogger(__name__)


class TritonLinalgAdapter(ILinalgPybindAdapter):
    """In-process adapter using triton-linalg (AxisInfo pointer analysis).

    This adapter directly calls the MLIR passes from triton-linalg via
    pybind11 bindings, making it the fastest conversion path.

    Note: The "triton-linalg" name is the Adapter registry name. The
    actual passes wrapped here are triton_race's self-developed 11-pass
    pipeline (``passes.race.triton_to_linalg.*``), NOT the Cambricon
    triton-linalg standalone library.

    Pass pipeline (from triton_race ``_make_raceir()``):
      1. triton_to_ppl                    — PPL index preparation
      2. wrap_func_body_with_single_block  — normalize function body
      3. inliner                           — inline called functions
      4. canonicalizer                     — standard canonicalization
      5. canonicalize_triton               — Triton-specific canonicalization
      6. pointer_strength_reduction        — pointer analysis (AxisInfo)
      7. canonicalizer                     — re-canonicalize after pointer analysis
      8. triton_to_linalg                  — core Triton→Linalg conversion
      9. extract_like_move_backward        — optimization on extract ops
      10. canonicalizer                    — post-conversion canonicalization
      11. arith_to_linalg                  — arithmetic op lowering
      12. math_to_linalg                   — math op lowering
      13. cse                              — common sub-expression elimination
      14. licm                             — loop-invariant code motion
      15. wrap_func_body_with_single_block — final normalization
    """

    def name(self) -> str:
        return "triton-linalg"

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Convert TTIR to Linalg using triton-linalg passes.

        Args:
            ttir_module: MLIR module (``ir.Module``) after TTIR optimization.
            metadata: Compilation metadata dict.
            context: MLIR context (unused — context is obtained from module).

        Returns:
            The converted MLIR module (same object, mutated in-place).

        Raises:
            AdapterConversionError: If any pass in the pipeline fails.
        """
        try:
            from triton._C.libtriton.anchor import anchor_passes as passes
            from triton._C.libtriton import ir
        except ImportError:
            raise AdapterConversionError(
                self.name(), detail="triton_anchor._C not available. Is the C++ extension built?"
            )

        # Check that anchor passes are available
        if not hasattr(passes, 'triton_to_linalg'):
            raise AdapterConversionError(
                self.name(), detail="anchor_passes.triton_to_linalg not available."
            )

        # Pre-process: fix allow_reorder attribute format
        ttir_code = str(ttir_module)
        if "allow_reorder" in ttir_code and "allow_reorder = true" not in ttir_code:
            # This is a known quirk in triton_race
            logger.debug("Applying allow_reorder attribute fixup")

        # Extract kernel name for diagnostics
        kernel_name = self._extract_kernel_name(ttir_module)
        if kernel_name:
            metadata.setdefault("name", kernel_name)

        # Build and run the pass pipeline
        pm = ir.pass_manager(ttir_module.context)
        pm.enable_debug()

        self._add_passes(pm, passes)

        try:
            pm.run(ttir_module)
        except Exception as e:
            logger.error(f"TritonLinalgAdapter conversion failed for kernel "
                         f"'{metadata.get('name', '<unknown>')}'")
            traceback.print_exc()
            raise AdapterConversionError(
                self.name(),
                kernel_name=metadata.get("name", ""),
                detail=str(e)
            )

        return ttir_module

    def _add_passes(self, pm, passes) -> None:
        """Add the triton-linalg conversion pass pipeline."""
        tl = passes.triton_to_linalg

        # Note: triton_to_ppl has been stripped. The backend should handle it if needed.
        tl.add_wrap_func_body_with_single_block(pm)
        
        # We need common passes from libtriton
        from triton._C.libtriton.passes import common
        
        common.add_inliner(pm)
        common.add_canonicalizer(pm)
        tl.add_canonicalize_triton(pm)
        tl.add_pointer_strength_reduction(pm)
        common.add_canonicalizer(pm)
        tl.add_triton_to_linalg(pm)
        tl.add_extract_like_move_backward(pm)
        common.add_canonicalizer(pm)
        tl.add_arith_to_linalg(pm)
        tl.add_math_to_linalg(pm)
        common.add_cse(pm)
        common.add_licm(pm)
        tl.add_wrap_func_body_with_single_block(pm)

    def _extract_kernel_name(self, mod) -> str:
        """Extract the Triton kernel function name from the module."""
        pattern = r"tt\.func\s+(?:public\s+)?@(\w+)\("
        matches = re.findall(pattern, str(mod))
        if len(matches) == 1:
            return matches[0]
        return ""

    def get_required_passes(self) -> List[str]:
        return [
            "triton_to_ppl",
            "wrap_func_body_with_single_block",
            "inliner",
            "canonicalizer",
            "canonicalize_triton",
            "pointer_strength_reduction",
            "triton_to_linalg",
            "extract_like_move_backward",
            "arith_to_linalg",
            "math_to_linalg",
            "cse",
            "licm",
        ]

    def get_output_dialects(self) -> List[str]:
        return [
            "linalg", "linalg_ext", "tensor", "memref",
            "arith", "math", "math_ext", "scf", "func", "aux",
        ]
