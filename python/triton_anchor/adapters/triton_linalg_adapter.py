"""
TritonLinalgAdapter — In-Process Adapter wrapping triton-linalg
================================================================

This adapter wraps the in-process Linalg-track conversion path exposed by
``triton._C.libtriton.anchor.passes`` on the 3.3 line.

It calls the MLIR PassManager directly (in-process), with zero subprocess
overhead.

Dependencies:
  - ``triton._C.libtriton`` must be available
  - triton-anchor passes must be linked into libtriton.so

Output dialects:
  linalg, linalg_ext, tensor, memref, arith, math, scf, func, aux
"""

from __future__ import annotations

import logging
import re
import traceback
from typing import Any, List, Optional, Tuple

from ..anchor_ir import AnchorIRTrack
from .base import ILinalgPybindAdapter, AdapterConversionError

logger = logging.getLogger(__name__)


class TritonLinalgAdapter(ILinalgPybindAdapter):
    """In-process adapter using triton-linalg (AxisInfo pointer analysis).

    This adapter directly calls the MLIR passes from triton-linalg via
    pybind11 bindings, making it the fastest conversion path.

    Note: The "triton-linalg" name is the Adapter registry name.  The Python
    binding contract for 3.3 is flat ``anchor.passes.add_*`` functions, not the
    older ``anchor_passes.triton_to_linalg.*`` layout.
    """

    def name(self) -> str:
        return "triton-linalg"

    def get_supported_tracks(self) -> Tuple[AnchorIRTrack, ...]:
        return (AnchorIRTrack.LINALG,)

    def get_supported_ptr_models(self) -> Tuple[str, ...]:
        return ("axis_info",)

    def get_required_backend_capabilities(self) -> Tuple[str, ...]:
        return ("anchor_ir.linalg",)

    def get_supported_ops(self) -> Tuple[str, ...]:
        return ("*",)

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
            from triton._C.libtriton.anchor import passes as anchor_passes
            from triton._C.libtriton import ir
        except ImportError as exc:
            raise AdapterConversionError(
                self.name(),
                detail=(
                    "triton._C.libtriton.anchor.passes is not available. "
                    "Build triton-anchor's C++ extension for the 3.3 line."
                ),
            ) from exc

        if not self._has_supported_pipeline(anchor_passes):
            raise AdapterConversionError(
                self.name(),
                detail=(
                    "triton._C.libtriton.anchor.passes does not expose "
                    "add_triton_to_linalg or add_triton_to_linalg_experimental."
                ),
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

        selected_pipeline = self._add_passes(pm, anchor_passes)

        try:
            pm.run(ttir_module)
        except Exception as e:
            logger.error(
                f"TritonLinalgAdapter conversion failed for kernel "
                f"'{metadata.get('name', '<unknown>')}'"
            )
            traceback.print_exc()
            raise AdapterConversionError(
                self.name(), kernel_name=metadata.get("name", ""), detail=str(e)
            )

        metadata["adapter_pipeline"] = selected_pipeline
        return ttir_module

    def _has_supported_pipeline(self, anchor_passes) -> bool:
        return hasattr(anchor_passes, "add_triton_to_linalg") or hasattr(
            anchor_passes,
            "add_triton_to_linalg_experimental",
        )

    def _add_passes(self, pm, anchor_passes) -> str:
        """Add the 3.3 anchor-bound Linalg conversion pipeline."""
        common = self._load_common_passes()
        self._add_common(pm, common, "add_inliner")
        self._add_common(pm, common, "add_canonicalizer")

        if hasattr(anchor_passes, "add_triton_to_linalg_experimental"):
            anchor_passes.add_triton_to_linalg_experimental(pm)
            pipeline = "anchor.passes.add_triton_to_linalg_experimental"
        else:
            anchor_passes.add_triton_to_linalg(pm)
            pipeline = "anchor.passes.add_triton_to_linalg"

        self._add_common(pm, common, "add_canonicalizer")
        self._add_common(pm, common, "add_cse")
        return pipeline

    def _load_common_passes(self) -> Optional[Any]:
        try:
            from triton._C.libtriton.passes import common
        except ImportError:
            return None
        return common

    def _add_common(self, pm, common: Optional[Any], pass_name: str) -> None:
        if common is None:
            return
        fn = getattr(common, pass_name, None)
        if fn is not None:
            fn(pm)

    def _extract_kernel_name(self, mod) -> str:
        """Extract the Triton kernel function name from the module."""
        pattern = r"tt\.func\s+(?:public\s+)?@(\w+)\("
        matches = re.findall(pattern, str(mod))
        if len(matches) == 1:
            return matches[0]
        return ""

    def get_required_passes(self) -> List[str]:
        return [
            "common.add_inliner",
            "common.add_canonicalizer",
            "anchor.passes.add_triton_to_linalg",
            "common.add_cse",
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
