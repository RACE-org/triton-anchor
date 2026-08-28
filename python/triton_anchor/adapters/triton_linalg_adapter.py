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

    Pass pipeline:
      1. wrap_func_body_with_single_block  — normalize function body
      2. inliner                           — inline called functions
      3. canonicalizer                     — standard canonicalization
      4. canonicalize_triton               — Triton-specific canonicalization
      5. pointer_strength_reduction        — pointer analysis (AxisInfo)
      6. canonicalizer                     — re-canonicalize after pointer analysis
      7. triton_to_linalg                  — core Triton→Linalg conversion
      8. extract_like_move_backward        — optimization on extract ops
      9. canonicalizer                     — post-conversion canonicalization
      10. arith_to_linalg                  — arithmetic op lowering
      11. math_to_linalg                   — math op lowering
      12. cse                              — common sub-expression elimination
      13. licm                             — loop-invariant code motion
      14. wrap_func_body_with_single_block — final normalization

    ``triton_to_ppl`` is deliberately outside this Adapter boundary.  A backend
    that needs it must run it only after the post-hook validation boundary.
    """

    def name(self) -> str:
        return "triton-linalg"

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Convert TTIR to Linalg using triton-linalg passes.

        Args:
            ttir_module: MLIR module (``ir.Module``) after TTIR optimization.
            metadata: Compilation metadata dict.
            context: Owning MLIR context. Required when the module wrapper does
                not expose one as a Python attribute.

        Returns:
            The converted MLIR module (same object, mutated in-place).

        Raises:
            AdapterConversionError: If any pass in the pipeline fails.
        """
        try:
            from triton._C.libtriton import anchor, ir
            from triton._C.libtriton.anchor import anchor_passes as passes
        except ImportError as error:
            raise AdapterConversionError(
                self.name(),
                detail="triton_anchor._C not available. Is the C++ extension built?",
            ) from error

        # Check that anchor passes are available
        if not hasattr(passes, "triton_to_linalg"):
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

        # Build and run the pass pipeline.  Modules returned directly by
        # ``ir.parse_mlir_module`` do not necessarily expose a Python
        # ``context`` attribute (the upstream compiler attaches it manually).
        # Honour the context supplied by the compilation entry point and only
        # fall back to an attached module context for compatibility.
        module_context = context
        if module_context is None:
            module_context = getattr(ttir_module, "context", None)
        if module_context is None:
            raise AdapterConversionError(
                self.name(),
                kernel_name=metadata.get("name", ""),
                detail=(
                    "an MLIR context is required; pass context=... when the "
                    "module does not expose a context attribute"
                ),
            )
        try:
            # A normal upstream Triton context only has ``ir.load_dialects``.
            # Register the Anchor/Linalg dialects and external interfaces here
            # so the real pass pipeline cannot abort merely because the caller
            # did not know about an additional setup call.
            anchor.load_dialects(module_context)
            pm = ir.pass_manager(module_context)
            pm.enable_debug()
            self._add_passes(pm, passes)
            pm.run(ttir_module)
        except Exception as error:
            logger.exception(
                "TritonLinalgAdapter conversion failed for kernel '%s'",
                metadata.get("name", "<unknown>"),
            )
            raise AdapterConversionError(
                self.name(),
                kernel_name=metadata.get("name", ""),
                detail=str(error),
            ) from error

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
            "linalg",
            "linalg_ext",
            "tensor",
            "memref",
            "arith",
            "math",
            "math_ext",
            "scf",
            "func",
            "aux",
        ]
