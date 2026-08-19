"""Base backend for hardware plugins using the Anchor frontend.

``AnchorBackendBase`` owns every stage up to the AnchorIR boundary. Vendor
plugins only describe their hardware and append the stages that lower
AnchorIR to a loadable binary.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from triton.backends.compiler import BaseBackend

from .adapters.base import ITritonToLinalgAdapter
from .adapters.registry import AdapterRegistry
from .anchor_ir import AnchorIRError, AnchorIRTrack, AnchorIRValidator
from .hw_capability import HWCapability
from .pipeline import make_ttir


@dataclass(frozen=True)
class AnchorCompilationContext:
    """Frontend decisions derived from one backend's hardware capability.

    This is a compilation configuration, not an MLIR context. Triton creates
    the MLIR context only after ``add_stages`` returns, so stage functions get
    the concrete MLIR context from their input module.
    """

    hw: HWCapability
    track: AnchorIRTrack
    adapter: Optional[ITritonToLinalgAdapter]
    validate_ir: bool = False

    @property
    def frontend_stage_name(self) -> str:
        if self.track == AnchorIRTrack.LINALG:
            return "linalg"
        if self.track == AnchorIRTrack.TRITON_GPU:
            return "ttgir"
        raise ValueError(f"Unsupported AnchorIR track: {self.track}")


class AnchorBackendBase(BaseBackend):
    """Common compiler frontend for out-of-tree Anchor backends."""

    def add_stages(self, stages, options, language=None):
        hw = self.get_hw_capability(options)
        ctx = self.prepare_anchor_compilation(hw, options)

        self.add_anchor_ttir_stage(stages, options, ctx)
        self.add_anchor_frontend_stages(stages, options, ctx)

        # Vendor implementations must not replace frontend-owned stages.
        frontend_stages = dict(stages)
        self.add_vendor_stages(stages, options, ctx)
        for name, stage in frontend_stages.items():
            if stages.get(name) is not stage:
                raise RuntimeError(
                    f"Vendor backend '{type(self).__name__}' attempted to replace "
                    f"Anchor-owned stage '{name}'"
                )
        if len(stages) == len(frontend_stages):
            raise RuntimeError(
                f"Vendor backend '{type(self).__name__}' did not add vendor stages"
            )

    @abstractmethod
    def get_hw_capability(self, options) -> HWCapability:
        """Return the hardware description for the current target/options."""

    @abstractmethod
    def add_vendor_stages(
        self,
        stages,
        options,
        ctx: AnchorCompilationContext,
    ) -> None:
        """Append stages from AnchorIR to the final vendor binary."""

    def prepare_anchor_compilation(
        self,
        hw: HWCapability,
        options,
    ) -> AnchorCompilationContext:
        """Validate hardware capabilities and select the frontend track."""
        if not isinstance(hw, HWCapability):
            raise TypeError(
                f"get_hw_capability() must return HWCapability, got {type(hw).__name__}"
            )
        hw.validate()

        adapter = None
        if hw.anchor_ir_track == AnchorIRTrack.LINALG:
            adapter = AdapterRegistry.get_adapter(hw)
        elif hw.anchor_ir_track != AnchorIRTrack.TRITON_GPU:
            raise ValueError(f"Unsupported AnchorIR track: {hw.anchor_ir_track}")

        validate_ir = bool(getattr(options, "validate_anchor_ir", False))
        validate_ir = validate_ir or os.getenv("TRITON_ANCHOR_VALIDATE_IR", "0") == "1"
        return AnchorCompilationContext(
            hw=hw,
            track=hw.anchor_ir_track,
            adapter=adapter,
            validate_ir=validate_ir,
        )

    def add_anchor_ttir_stage(self, stages, options, ctx: AnchorCompilationContext):
        self._add_frontend_stage(
            stages,
            "ttir",
            lambda mod, metadata: self._make_anchor_ttir(mod, metadata, ctx),
        )

    def add_anchor_frontend_stages(
        self,
        stages,
        options,
        ctx: AnchorCompilationContext,
    ):
        if ctx.track == AnchorIRTrack.LINALG:
            self._add_frontend_stage(
                stages,
                "linalg",
                lambda mod, metadata: self._make_linalg_anchor_ir(
                    mod, metadata, ctx
                ),
            )
            return

        if ctx.track == AnchorIRTrack.TRITON_GPU:
            self._add_frontend_stage(
                stages,
                "ttgir",
                lambda mod, metadata: self._make_triton_gpu_anchor_ir(
                    mod, metadata, options, ctx
                ),
            )
            return

        raise ValueError(f"Unsupported AnchorIR track: {ctx.track}")

    @staticmethod
    def _add_frontend_stage(stages, name, stage):
        if name in stages:
            raise RuntimeError(f"Compilation stage '{name}' is already registered")
        stages[name] = stage

    def _make_anchor_ttir(self, mod, metadata, ctx: AnchorCompilationContext):
        mod = make_ttir(mod, metadata, hw=ctx.hw)
        mod = self._specialize_kernel_name(mod, metadata)
        self._record_anchor_metadata(metadata, ctx)
        return mod

    def _make_linalg_anchor_ir(
        self,
        mod,
        metadata,
        ctx: AnchorCompilationContext,
    ):
        if ctx.adapter is None:
            raise RuntimeError("Linalg AnchorIR track requires an adapter")
        result = ctx.adapter.convert(mod, metadata, context=mod.context)
        result = self._coerce_mlir_module(result, mod.context)
        metadata["anchor_adapter"] = ctx.adapter.name()
        self._validate_anchor_ir(result, metadata, ctx)
        return result

    def _make_triton_gpu_anchor_ir(
        self,
        mod,
        metadata,
        options,
        ctx: AnchorCompilationContext,
    ):
        from triton._C.libtriton import ir, passes

        cap = ctx.hw.gpgpu_cap
        if cap is None:
            raise ValueError("TritonGPU track requires HWCapability.gpgpu_cap")

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(
            pm,
            self.get_triton_gpu_conversion_target(options, ctx.hw),
            cap.num_warps,
            cap.warp_size,
            cap.num_ctas,
        )
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod)
        self._validate_anchor_ir(mod, metadata, ctx)
        return mod

    def get_triton_gpu_conversion_target(self, options, hw: HWCapability) -> str:
        """Return the target string used by Triton-to-TritonGPU conversion."""
        return str(self.target.backend)

    def _validate_anchor_ir(self, mod, metadata, ctx: AnchorCompilationContext):
        metadata["anchor_ir_validated"] = ctx.validate_ir
        if not ctx.validate_ir:
            return
        validator = AnchorIRValidator(track=ctx.track)
        violations = validator.validate_pre_hook(str(mod))
        if violations:
            kernel = metadata.get("name", "<unknown>")
            details = "\n".join(str(violation) for violation in violations)
            raise AnchorIRError(
                f"AnchorIR validation failed for {kernel} ({ctx.track.value}):\n"
                f"{details}"
            )

    @staticmethod
    def _record_anchor_metadata(metadata, ctx: AnchorCompilationContext):
        metadata["anchor_hw_name"] = ctx.hw.name
        metadata["anchor_arch_family"] = ctx.hw.arch_family
        metadata["anchor_compute_paradigm"] = ctx.hw.compute_paradigm.value
        metadata["anchor_ir_track"] = ctx.track.value

    @staticmethod
    def _extract_kernel_name(mod) -> str:
        matches = re.findall(r"tt\.func\s+(?:public\s+)?@(\w+)\(", str(mod))
        return matches[0] if len(matches) == 1 else ""

    def _specialize_kernel_name(self, mod, metadata):
        """Give every specialization a stable symbol and artifact name."""
        kernel_name = self._extract_kernel_name(mod)
        if not kernel_name:
            return mod

        metadata_hash = metadata.get("hash", "")
        suffix = (
            metadata_hash[:8]
            if metadata_hash
            else hashlib.sha256(str(mod).encode("utf-8")).hexdigest()[:8]
        )
        new_name = f"{kernel_name}_{suffix}"
        ttir_code = re.sub(rf"@{re.escape(kernel_name)}(?!\w)", f"@{new_name}", str(mod))

        from triton._C.libtriton import ir

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", encoding="utf-8", delete=False
        ) as file:
            file.write(ttir_code)
            path = file.name
        try:
            renamed = ir.parse_mlir_module(path, mod.context)
        finally:
            os.unlink(path)
        renamed.context = mod.context
        metadata["name"] = new_name
        return renamed

    @staticmethod
    def _coerce_mlir_module(result, context):
        """Parse text returned by out-of-process adapters back into MLIR."""
        if not isinstance(result, str):
            return result

        from triton._C.libtriton import ir

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", encoding="utf-8", delete=False
        ) as file:
            file.write(result)
            path = file.name
        try:
            mod = ir.parse_mlir_module(path, context)
        finally:
            os.unlink(path)
        mod.context = context
        return mod

    def load_dialects(self, context):
        """Load Anchor dialects; vendors may add dialects through the hook."""
        from triton._C.libtriton import anchor

        anchor.load_dialects(context)
        self.load_vendor_dialects(context)

    def load_vendor_dialects(self, context):
        """Optional hook for dialects required only by vendor stages."""

