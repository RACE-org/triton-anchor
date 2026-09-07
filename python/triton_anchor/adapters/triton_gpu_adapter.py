"""TritonGPU adapter for the AnchorIR TRITON_GPU track."""

from __future__ import annotations

import logging
from typing import Any, Tuple

from ..anchor_ir import AnchorIRTrack, AnchorIRValidator
from .base import AdapterConversionContext, AdapterConversionError, IAnchorAdapter

logger = logging.getLogger(__name__)


class TritonGPUAdapter(IAnchorAdapter):
    """In-process TTIR -> TritonGPU conversion adapter.

    This adapter stops at the TritonGPU AnchorIR track.  Backend plugins remain
    responsible for target-specific optimization and final code generation.
    """

    def name(self) -> str:
        return "triton-gpu"

    def get_supported_tracks(self) -> Tuple[AnchorIRTrack, ...]:
        return (AnchorIRTrack.TRITON_GPU,)

    def get_supported_ptr_models(self) -> Tuple[str, ...]:
        return ("gpu",)

    def get_required_backend_capabilities(self) -> Tuple[str, ...]:
        return ("anchor_ir.triton_gpu",)

    def get_supported_ops(self) -> Tuple[str, ...]:
        return ("*",)

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Convert optimized TTIR to TritonGPU IR in-place."""
        try:
            from triton._C.libtriton import ir, passes
        except ImportError as exc:
            raise AdapterConversionError(
                self.name(),
                detail=(
                    "triton._C.libtriton passes are not available. "
                    "Build Triton's Python C++ extension for the 3.3 line."
                ),
            ) from exc

        if not hasattr(passes.ttir, "add_convert_to_ttgpuir"):
            raise AdapterConversionError(
                self.name(),
                detail="passes.ttir.add_convert_to_ttgpuir is not available.",
            )

        target, num_warps, warp_size, num_ctas = self._resolve_options(
            metadata,
            context,
        )

        pm = ir.pass_manager(ttir_module.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(
            pm,
            target,
            num_warps,
            warp_size,
            num_ctas,
        )
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        try:
            pm.run(ttir_module)
        except Exception as exc:
            logger.error("TritonGPUAdapter conversion failed")
            raise AdapterConversionError(
                self.name(),
                kernel_name=metadata.get("name", ""),
                detail=str(exc),
            ) from exc

        metadata["adapter_pipeline"] = "ttir.add_convert_to_ttgpuir"
        metadata["adapter_triton_gpu_options"] = {
            "target": target,
            "num_warps": num_warps,
            "warp_size": warp_size,
            "num_ctas": num_ctas,
        }
        return ttir_module

    def validate_output(self, anchor_ir: Any) -> bool:
        ir_text = str(anchor_ir) if not isinstance(anchor_ir, str) else anchor_ir
        return AnchorIRValidator(track=AnchorIRTrack.TRITON_GPU).is_valid(ir_text)

    def get_required_passes(self) -> list[str]:
        return [
            "ttir.add_convert_to_ttgpuir",
            "common.add_canonicalizer",
            "common.add_cse",
        ]

    def get_output_dialects(self) -> list[str]:
        return ["triton_gpu", "tt", "arith", "math", "scf", "func", "gpu", "nvgpu"]

    def _resolve_options(self, metadata: dict, context: Any) -> tuple[str, int, int, int]:
        hw = context.hw if isinstance(context, AdapterConversionContext) else None
        target_obj = metadata.get("target")

        target = self._target_name(target_obj)
        if target is None and hw is not None:
            target = hw._infer_backend_name()
        if target is None:
            target = metadata.get("hw_arch_family", "gpu")

        gpgpu_cap = getattr(hw, "gpgpu_cap", None) if hw is not None else None
        num_warps = int(metadata.get("num_warps") or getattr(gpgpu_cap, "num_warps", 4))
        warp_size = int(metadata.get("warp_size") or self._target_warp_size(target_obj))
        if not warp_size and gpgpu_cap is not None:
            warp_size = int(gpgpu_cap.warp_size)
        if not warp_size:
            warp_size = 32
        num_ctas = int(metadata.get("num_ctas") or getattr(gpgpu_cap, "num_ctas", 1))
        return str(target), num_warps, warp_size, num_ctas

    def _target_name(self, target_obj: Any) -> str | None:
        if target_obj is None:
            return None
        if isinstance(target_obj, str):
            return target_obj
        if isinstance(target_obj, dict):
            return target_obj.get("backend")
        return getattr(target_obj, "backend", None)

    def _target_warp_size(self, target_obj: Any) -> int:
        if target_obj is None:
            return 0
        if isinstance(target_obj, dict):
            return int(target_obj.get("warp_size", 0) or 0)
        return int(getattr(target_obj, "warp_size", 0) or 0)
