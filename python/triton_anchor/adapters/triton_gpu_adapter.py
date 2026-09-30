"""
TritonGPUAdapter — Triton 3.6 GPU-track shim
=============================================

Triton 3.6 in this version line exposes ``passes.ttir``, ``passes.gluon``,
and ``passes.plugin`` but leaves ``passes.ttgpuir`` empty.  This adapter does
not assume the old 3.3 ``add_convert_to_ttgpuir`` binding exists.  Instead it
probes the available in-process pass modules, uses a backend/plugin conversion
pass when one is present, and otherwise fails with a 3.6-specific diagnostic.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import AdapterConversionError, ITritonToLinalgAdapter

logger = logging.getLogger(__name__)


class TritonGPUAdapter(ITritonToLinalgAdapter):
    """In-process adapter for the TritonGPU AnchorIR track on Triton 3.6."""

    _CONVERT_PASS_NAMES = (
        "add_convert_to_ttgpuir",
        "add_convert_triton_to_tritongpu",
        "add_convert_triton_to_triton_gpu",
        "add_convert_triton_to_ttgpuir",
    )
    _TTGPU_CLEANUP_PASSES = (
        "add_coalesce",
        "add_remove_layout_conversions",
        "add_optimize_thread_locality",
        "add_accelerate_matmul",
        "add_optimize_accumulator_init",
        "add_cse",
        "add_symbol_dce",
    )
    _GLUON_PASSES = (
        "add_resolve_auto_encodings",
        "add_infer_coalesced_encodings",
        "add_canonicalizer",
    )

    def name(self) -> str:
        return "triton-gpu"

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Convert TTIR to TritonGPU IR when a 3.6 pass/plugin is available."""
        metadata["anchor_adapter_effective"] = self.name()
        metadata["anchor_adapter_mode"] = "gpu"

        if self._looks_like_ttgpu(ttir_module):
            metadata["anchor_gpu_shim"] = "already-triton-gpu"
            return ttir_module

        try:
            from triton._C.libtriton import ir, passes
        except ImportError as exc:
            raise AdapterConversionError(
                self.name(),
                detail="triton._C.libtriton is required for TritonGPU lowering.",
            ) from exc

        module = self._coerce_module(ttir_module, context, ir)
        kernel_name = self._extract_kernel_name(module)
        if kernel_name:
            metadata.setdefault("name", kernel_name)

        pm = ir.pass_manager(module.context)
        enable_debug = getattr(pm, "enable_debug", None)
        if enable_debug is not None:
            enable_debug()

        selected = self._add_conversion_pass(pm, passes, metadata)
        self._add_available_post_passes(pm, passes, metadata)
        try:
            pm.run(module)
        except Exception as exc:
            raise AdapterConversionError(
                self.name(),
                kernel_name=metadata.get("name", ""),
                detail=f"TritonGPU pass pipeline failed via {selected}: {exc}",
            ) from exc

        metadata["anchor_gpu_shim"] = selected
        return module

    def _coerce_module(self, ttir_module: Any, context: Any, ir: Any) -> Any:
        if not isinstance(ttir_module, str):
            return ttir_module
        if context is None:
            raise AdapterConversionError(
                self.name(),
                detail="string TTIR input requires an MLIR context for GPU lowering.",
            )
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as src:
            src.write(ttir_module)
            src_path = src.name
        try:
            module = ir.parse_mlir_module(src_path, context)
            module.context = context
            return module
        finally:
            try:
                Path(src_path).unlink()
            except OSError:
                pass

    def _add_conversion_pass(self, pm: Any, passes: Any, metadata: dict) -> str:
        attempts: List[str] = []
        arg_sets = self._conversion_arg_sets(metadata)
        for module_name, pass_module in self._pass_modules(passes):
            for pass_name in self._CONVERT_PASS_NAMES:
                fn = getattr(pass_module, pass_name, None)
                if fn is None:
                    continue
                selected = f"passes.{module_name}.{pass_name}"
                for args in arg_sets:
                    try:
                        fn(pm, *args)
                    except TypeError as exc:
                        attempts.append(f"{selected}{args}: {exc}")
                        continue
                    metadata["anchor_gpu_convert_pass"] = selected
                    metadata["anchor_gpu_convert_pass_args"] = list(args)
                    return selected

        available = self._available_pass_summary(passes)
        plugin_path = os.environ.get("TRITON_PASS_PLUGIN_PATH") or "<unset>"
        raise AdapterConversionError(
            self.name(),
            detail=(
                "Triton 3.6 GPU lowering requires a bound or plugin-provided "
                "TTIR->TritonGPU conversion pass. This build exposes "
                f"{available}; TRITON_PASS_PLUGIN_PATH={plugin_path}. "
                "passes.ttgpuir may be empty on 3.6, so provide the conversion "
                "through passes.plugin or a backend-specific binding. "
                f"TypeError attempts: {attempts or '<none>'}"
            ),
        )

    def _pass_modules(self, passes: Any) -> Tuple[Tuple[str, Any], ...]:
        modules = []
        for module_name in ("ttir", "ttgpuir", "plugin"):
            module = getattr(passes, module_name, None)
            if module is not None:
                modules.append((module_name, module))
        return tuple(modules)

    def _conversion_arg_sets(self, metadata: dict) -> Tuple[Tuple[Any, ...], ...]:
        hw = metadata.get("hw") or metadata.get("hw_capability")
        gpgpu_cap = getattr(hw, "gpgpu_cap", None)
        target = metadata.get("target")
        arch = (
            metadata.get("gpu_target")
            or metadata.get("target_name")
            or getattr(hw, "arch_family", None)
            or getattr(target, "backend", None)
            or "gpu"
        )
        num_warps = int(
            metadata.get("num_warps") or getattr(gpgpu_cap, "num_warps", None) or 4
        )
        threads_per_warp = int(
            metadata.get("threads_per_warp")
            or getattr(gpgpu_cap, "warp_size", None)
            or getattr(target, "warp_size", None)
            or 32
        )
        num_ctas = int(
            metadata.get("num_ctas") or getattr(gpgpu_cap, "num_ctas", None) or 1
        )
        return (
            (str(arch), num_warps, threads_per_warp, num_ctas),
            (),
        )

    def _add_available_post_passes(self, pm: Any, passes: Any, metadata: dict) -> None:
        added = []
        common = getattr(passes, "common", None)
        ttgpuir = getattr(passes, "ttgpuir", None)
        for pass_name in self._TTGPU_CLEANUP_PASSES:
            fn = getattr(ttgpuir, pass_name, None) if ttgpuir is not None else None
            if fn is None and common is not None:
                fn = getattr(common, pass_name, None)
            if fn is None:
                continue
            fn(pm)
            added.append(pass_name)

        # 3.6 Gluon is a separate IR path.  Let backend plugins opt in when
        # they know their module contains Gluon IR accepted by these passes.
        if not metadata.get("anchor_gpu_enable_gluon_passes"):
            metadata["anchor_gpu_post_passes"] = added
            return
        gluon = getattr(passes, "gluon", None)
        for pass_name in self._GLUON_PASSES:
            fn = getattr(gluon, pass_name, None) if gluon is not None else None
            if fn is None:
                continue
            try:
                fn(pm)
            except TypeError:
                logger.debug("Skipping incompatible Gluon pass %s", pass_name)
                continue
            added.append(f"gluon.{pass_name}")

        metadata["anchor_gpu_post_passes"] = added

    def _available_pass_summary(self, passes: Any) -> Dict[str, List[str]]:
        summary = {}
        for module_name in ("ttir", "ttgpuir", "gluon", "plugin"):
            module = getattr(passes, module_name, None)
            if module is None:
                summary[module_name] = []
                continue
            summary[module_name] = sorted(
                name for name in dir(module) if name.startswith("add_")
            )
        return summary

    def _looks_like_ttgpu(self, module: Any) -> bool:
        text = module if isinstance(module, str) else str(module)
        return "triton_gpu." in text or "#ttg." in text or "ttg." in text

    def _extract_kernel_name(self, module: Any) -> str:
        import re

        pattern = r"(?:tt|func)\.func\s+(?:public\s+)?@(\w+)\("
        matches = re.findall(pattern, str(module))
        if len(matches) == 1:
            return matches[0]
        return ""

    def cache_key_components(self) -> Dict[str, Any]:
        components = super().cache_key_components()
        components["shim"] = "triton-3.6-plugin-gluon-probe"
        return components

    def get_required_passes(self) -> List[str]:
        return ["convert-triton-to-tritongpu"]

    def supported_routes(self) -> List[Tuple[str, str]]:
        return [("triton_gpu", "*")]

    def get_output_dialects(self) -> List[str]:
        return [
            "ttg",
            "triton_gpu",
            "tt",
            "arith",
            "math",
            "scf",
            "func",
            "gpu",
            "nvgpu",
        ]
