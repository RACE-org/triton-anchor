"""
TritonGPUAdapter — In-process TritonGPU lowering adapter
========================================================

T6.1 GPU route for Triton 3.0.  This adapter wraps the standard
TTIR-to-TritonGPU conversion pass and the TritonGPU optimization passes exposed
by ``triton/python/src/passes.cc`` in the 3.0 line.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, List

from .base import AdapterConversionError, ILinalgPybindAdapter

logger = logging.getLogger(__name__)


class TritonGPUAdapter(ILinalgPybindAdapter):
    """In-process adapter producing AnchorIR's TritonGPU track."""

    def name(self) -> str:
        return "triton-gpu"

    def get_supported_tracks(self) -> List[str]:
        return ["triton_gpu"]

    def get_supported_ptr_models(self) -> List[str]:
        return ["gpu"]

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Convert optimized TTIR to TritonGPU IR with encoding attributes."""
        try:
            from triton._C.libtriton import ir, passes
        except ImportError:
            raise AdapterConversionError(
                self.name(),
                detail="triton._C.libtriton is not available. Is Triton built?",
            )

        if not hasattr(passes.ttir, "add_convert_to_ttgpuir"):
            raise AdapterConversionError(
                self.name(),
                detail="passes.ttir.add_convert_to_ttgpuir is not available.",
            )

        target_name = self._target_name(metadata)
        num_warps = self._int_option(metadata, "num_warps", 4)
        threads_per_warp = self._threads_per_warp(metadata, 32)
        num_ctas = self._int_option(metadata, "num_ctas", 1)
        num_stages = self._int_option(metadata, "num_stages", 3)

        pm = ir.pass_manager(ttir_module.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(
            pm, target_name, num_warps, threads_per_warp, num_ctas
        )
        added_passes = self._add_ttgpuir_passes(pm, passes, num_stages, target_name)

        try:
            pm.run(ttir_module)
        except Exception as exc:
            logger.error("TritonGPUAdapter conversion failed")
            traceback.print_exc()
            raise AdapterConversionError(
                self.name(),
                kernel_name=metadata.get("name", ""),
                detail=str(exc),
            )

        metadata.setdefault("selected_adapter", self.name())
        metadata["triton_gpu_target"] = target_name
        metadata["triton_gpu_num_warps"] = num_warps
        metadata["triton_gpu_threads_per_warp"] = threads_per_warp
        metadata["triton_gpu_num_ctas"] = num_ctas
        metadata["triton_gpu_passes"] = added_passes

        self._validate_encoding(ttir_module)
        return ttir_module

    def _add_ttgpuir_passes(
        self, pm, passes, num_stages: int, target_name: str
    ) -> List[str]:
        ttgpuir = passes.ttgpuir
        added: List[str] = []

        self._try_add(added, ttgpuir, "add_coalesce", pm)
        self._try_add(added, ttgpuir, "add_remove_layout_conversions", pm)
        self._try_add(added, ttgpuir, "add_optimize_thread_locality", pm)
        if target_name.startswith("cuda:"):
            self._try_add(added, ttgpuir, "add_accelerate_matmul", pm)
            self._try_add(added, ttgpuir, "add_f32_dot_tc", pm)
            self._try_add(added, ttgpuir, "add_optimize_dot_operands", pm, True)
        self._try_add(added, ttgpuir, "add_pipeline", pm, num_stages)
        self._try_add(added, ttgpuir, "add_prefetch", pm)
        self._try_add(added, ttgpuir, "add_reorder_instructions", pm)
        self._try_add(added, ttgpuir, "add_reduce_data_duplication", pm)
        self._try_add(added, ttgpuir, "add_remove_layout_conversions", pm)

        common = getattr(passes, "common", None)
        if common is not None:
            self._try_add(added, common, "add_cse", pm)
            self._try_add(added, common, "add_symbol_dce", pm)
        return added

    def _validate_encoding(self, module: Any) -> None:
        text = str(module)
        missing = [
            attr
            for attr in (
                "triton_gpu.num-warps",
                "triton_gpu.threads-per-warp",
                "triton_gpu.num-ctas",
                "triton_gpu.target",
            )
            if attr not in text
        ]
        if missing:
            raise AdapterConversionError(
                self.name(),
                detail="missing TritonGPU encoding attributes: " + ", ".join(missing),
            )
        if not self.validate_output(module):
            raise AdapterConversionError(
                self.name(),
                detail="TritonGPU output violates AnchorIR validation",
            )

    @staticmethod
    def _try_add(added: List[str], module, pass_name: str, pm, *args) -> bool:
        fn = getattr(module, pass_name, None)
        if fn is None:
            return False
        fn(pm, *args) if args else fn(pm)
        added.append(pass_name)
        return True

    @staticmethod
    def _target_name(metadata: dict) -> str:
        explicit = metadata.get("triton_gpu_target") or metadata.get("gpu_target")
        if explicit:
            return str(explicit)

        target = metadata.get("target")
        if isinstance(target, dict):
            backend = target.get("backend")
            arch = target.get("arch")
            if backend:
                return TritonGPUAdapter._format_target(str(backend), arch)
        backend = getattr(target, "backend", None)
        if backend:
            return TritonGPUAdapter._format_target(
                str(backend), getattr(target, "arch", None)
            )

        hw = metadata.get("hw_capability") or metadata.get("hw")
        if hw is not None:
            arch_family = getattr(hw, "arch_family", "")
            if arch_family:
                return str(arch_family)
            name = getattr(hw, "name", "")
            if name:
                return str(name).split("-")[0]

        return "gpu"

    @staticmethod
    def _format_target(backend: str, arch) -> str:
        if ":" in backend:
            return backend
        if backend == "cuda" and arch is not None:
            return f"cuda:{arch}"
        if backend == "hip" and arch is not None:
            return f"hip:{arch}"
        return backend

    @staticmethod
    def _int_option(metadata: dict, name: str, default: int) -> int:
        value = metadata.get(name)
        if value is None:
            hw = metadata.get("hw_capability") or metadata.get("hw")
            gpgpu_cap = getattr(hw, "gpgpu_cap", None)
            value = getattr(gpgpu_cap, name, None)
        return int(value if value is not None else default)

    @staticmethod
    def _threads_per_warp(metadata: dict, default: int) -> int:
        value = metadata.get("threads_per_warp")
        if value is None:
            target = metadata.get("target")
            value = getattr(target, "warp_size", None)
        if value is None:
            hw = metadata.get("hw_capability") or metadata.get("hw")
            gpgpu_cap = getattr(hw, "gpgpu_cap", None)
            value = getattr(gpgpu_cap, "warp_size", None)
        return int(value if value is not None else default)

    def get_required_passes(self) -> List[str]:
        return [
            "convert-triton-to-tritongpu",
            "tritongpu-coalesce",
            "tritongpu-remove-layout-conversions",
            "tritongpu-optimize-thread-locality",
            "tritongpu-accelerate-matmul",
            "tritongpu-pipeline",
            "tritongpu-prefetch",
            "tritongpu-reorder-instructions",
        ]

    def get_output_dialects(self) -> List[str]:
        return ["triton_gpu", "tt", "arith", "math", "scf", "func", "gpu", "nvgpu"]
