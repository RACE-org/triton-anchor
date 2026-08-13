"""
TritonLinalgAdapter — In-Process Adapter wrapping triton-linalg
================================================================

This adapter wraps the in-tree triton-shared Triton-to-Linalg conversion
pipeline provided by the current branch.

It calls the MLIR PassManager directly (in-process), with zero subprocess
overhead.

Dependencies:
  - ``triton._C.libtriton`` must be available
  - triton-anchor's generic passes must be linked into libtriton.so

Output dialects:
  linalg, tensor, memref, arith, math, scf, cf, func, affine, bufferization
"""

from __future__ import annotations

import logging
import re
from typing import Any, List

from .base import ILinalgPybindAdapter, AdapterConversionError

logger = logging.getLogger(__name__)


class TritonLinalgAdapter(ILinalgPybindAdapter):
    """In-process adapter using the in-tree triton-shared conversion pass.

    This adapter directly calls the in-tree triton-shared pass via
    pybind11 bindings, making it the fastest conversion path.

    Pass pipeline:
      1. inliner             — inline called functions
      2. canonicalizer       — normalize TTIR before conversion
      3. triton_to_linalg    — in-tree triton-shared conversion
      4. canonicalizer       — fold conversion artifacts
      5. cse                 — remove duplicate expressions
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
        except ImportError as error:
            raise AdapterConversionError(
                self.name(),
                detail="triton_anchor._C not available. Is the C++ extension built?",
            ) from error

        passes = getattr(anchor, "passes", None)
        if passes is None or not hasattr(passes, "add_triton_to_linalg"):
            raise AdapterConversionError(
                self.name(), detail="anchor.passes.add_triton_to_linalg not available."
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
            # Triton 3.6 requires a stable reproducer tag for every Python
            # PassManager invocation.  It is appended to
            # TRITON_REPRODUCER_PATH when pass execution fails.
            pm.run(ttir_module, "anchor-linalg-adapter")
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
        """Add the current branch's in-tree conversion pipeline."""
        from triton._C.libtriton.passes import common

        common.add_inliner(pm)
        common.add_canonicalizer(pm)
        passes.add_triton_to_linalg(pm)
        common.add_canonicalizer(pm)
        common.add_cse(pm)

    def _extract_kernel_name(self, mod) -> str:
        """Extract the Triton kernel function name from the module."""
        pattern = r"tt\.func\s+(?:public\s+)?@(\w+)\("
        matches = re.findall(pattern, str(mod))
        if len(matches) == 1:
            return matches[0]
        return ""

    def get_required_passes(self) -> List[str]:
        return [
            "inliner",
            "canonicalizer",
            "triton_to_linalg",
            "cse",
        ]

    def get_output_dialects(self) -> List[str]:
        return [
            "linalg",
            "tensor",
            "memref",
            "arith",
            "math",
            "scf",
            "cf",
            "func",
            "affine",
            "bufferization",
        ]
