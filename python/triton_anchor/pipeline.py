"""
Unified TTIR Pipeline
======================

Extracts the 7 mandatory TTIR optimization passes that are 100% shared
across all three projects (spine-triton, triton_race, fantasy-triton).

This is a **core invariant** — the pass list is append-only and
synchronized with upstream Triton.

The pipeline also supports conditional passes controlled by ``HWCapability``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    from .adapters.base import ITritonToLinalgAdapter
    from .anchor_ir_lifecycle import (
        AnchorIRBackendHook,
        AnchorIRLifecycleReport,
    )
    from .hw_capability import HWCapability

from .anchor_ir_rules import ANCHOR_IR_SPEC_VERSION
from .anchor_ir_rules import validate_policy_request
from .anchor_ir_schema import AnchorIRPhase, AnchorIRTrack


def build_ttir_pipeline(pm, hw: Optional[HWCapability] = None):
    """Build the standard TTIR optimization pipeline.

    This is extracted from triton_race's ``_make_ttir()`` and is identical
    to the 7 mandatory passes used by all three projects.

    Args:
        pm: An ``mlir.PassManager`` instance.
        hw: Optional ``HWCapability``.  If provided, conditional passes
            are added based on hardware capabilities.

    Usage::

        from triton._C.libtriton import ir, passes

        mod = ...  # TTIR module
        pm = ir.pass_manager(mod.context)
        build_ttir_pipeline(pm, hw=my_hw_capability)
        pm.run(mod)

    Note:
        This function requires ``triton._C.libtriton`` to be available.
        It will raise ``ImportError`` if Triton is not installed.
    """
    from triton._C.libtriton import passes

    # ═══════════════════════════════════════════════════════════════════
    # Mandatory Passes (7) — shared 100% across all projects
    # Order matters: inliner → combine → canonicalize → reorder → cse → licm → dce
    # ═══════════════════════════════════════════════════════════════════
    passes.common.add_inliner(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)

    # ═══════════════════════════════════════════════════════════════════
    # Conditional Passes — controlled by HWCapability
    # ═══════════════════════════════════════════════════════════════════
    if hw is not None:
        from .hw_capability import ComputeParadigm

        # GPU path needs tensor pointer rewriting (CRITICAL — must not silently skip)
        if hw.compute_paradigm == ComputeParadigm.GPGPU:
            _require_pass(passes.ttir, "add_rewrite_tensor_pointer", pm)

        # Optional loop unrolling (safe to skip if unavailable)
        if hw.enable_loop_unroll:
            _try_add_pass(passes.ttir, "add_loop_unroll", pm)

        # FlagTree extra optimization (optional, auto-probe)
        _try_add_pass(passes.ttir, "add_expression_restructing", pm)


def _try_add_pass(module, pass_name, pm, **kwargs):
    """Safely try to add a pass. Silently skip if not available.

    For optional passes (e.g., add_expression_restructing, add_loop_unroll)
    whose absence does not affect compilation correctness.
    """
    fn = getattr(module, pass_name, None)
    if fn is not None:
        fn(pm, **kwargs) if kwargs else fn(pm)
        return True
    return False


def _require_pass(module, pass_name, pm, **kwargs):
    """Add a critical-path pass. Raise if not available.

    For passes on the critical compilation path (e.g., GPU's
    add_rewrite_tensor_pointer, add_convert_to_ttgpuir) whose
    absence would cause incorrect compilation results.
    """
    fn = getattr(module, pass_name, None)
    if fn is None:
        mod_name = getattr(module, "__name__", str(module))
        raise RuntimeError(
            f"Required pass '{pass_name}' not found in module '{mod_name}'. "
            f"This pass is critical for the current compilation path. "
            f"Check your Triton version and backend installation."
        )
    fn(pm, **kwargs) if kwargs else fn(pm)
    return True


def make_ttir(mod, metadata: dict, hw: Optional[HWCapability] = None):
    """Convenience function: build pipeline + run it on a module.

    This mirrors the signature of triton_race's ``_make_ttir(mod, metadata, options)``.

    Args:
        mod: An MLIR module (``ir.Module``).
        metadata: Compilation metadata dict (mutated in-place).
        hw: Optional ``HWCapability``.

    Returns:
        The optimized MLIR module (same object, mutated in-place).
    """
    from triton._C.libtriton import ir

    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    build_ttir_pipeline(pm, hw=hw)
    pm.run(mod)
    return mod


def inject_hw_attributes(mod, hw: HWCapability, metadata: dict):
    """将硬件能力信息注入 MLIR module 属性和编译元数据中。

    在 TTIR 优化之后、硬件感知 IR 降级之前调用。
    后端插件可通过 ``on_ttir_ready()`` hook 注入额外属性。

    Args:
        mod: MLIR module。
        hw: 目标硬件能力描述。
        metadata: 编译元数据 dict（就地更新）。
    """
    try:
        from triton._C.libtriton import ir

        builder = ir.builder(mod.context)

        # 整型硬件属性注入到 MLIR module（供下游 C++ pass 使用）
        if hw.arch_family == "riscv" and hw.matrix_cap:
            mod.set_attr("hw.num_threads", builder.get_int32_attr(hw.num_cores))
        elif hw.arch_family == "tpu" and hw.tensor_cap:
            mod.set_attr("hw.core_num", builder.get_int32_attr(hw.tensor_cap.num_cores))
        elif hw.arch_family == "gpu" and hw.gpgpu_cap:
            mod.set_attr(
                "hw.num_warps",
                builder.get_int32_attr(hw.gpgpu_cap.num_warps),
            )

    except ImportError:
        pass

    # 硬件描述信息通过 metadata dict 传递给下游 Python 代码
    metadata["hw_name"] = hw.name
    metadata["hw_paradigm"] = hw.compute_paradigm.value
    metadata["hw_arch_family"] = hw.arch_family


def run_anchor_ir_compilation(
    adapter: "ITritonToLinalgAdapter",
    ttir_module: Any,
    metadata: dict,
    *,
    hook: Optional["AnchorIRBackendHook"],
    backend_lowering: Optional[Callable[[Any], Any]],
    spec_version: str = ANCHOR_IR_SPEC_VERSION,
    track: Union[AnchorIRTrack, str] = AnchorIRTrack.LINALG,
    context: Any = None,
    source_name: Optional[str] = None,
) -> "AnchorIRLifecycleReport":
    """Run the mandatory Adapter -> AnchorIR -> Backend compilation boundary.

    This is the fail-closed production entry point.  ``adapter.convert`` is a
    low-level conversion primitive; callers entering a backend should use this
    function (or ``ITritonToLinalgAdapter.compile``) so invalid Adapter or Hook
    output raises ``AnchorIRValidationError`` before backend lowering.
    """

    from .anchor_ir_lifecycle import AnchorIRLifecycleOrchestrator
    from .anchor_ir_validator import AnchorIRValidationError

    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")
    if backend_lowering is not None and not callable(backend_lowering):
        raise TypeError("backend_lowering must be callable or None")
    request = validate_policy_request(
        spec_version=spec_version,
        track=track,
        phase=AnchorIRPhase.PRE_HOOK,
    )
    if not request.valid:
        raise AnchorIRValidationError(request)
    if request.track is not AnchorIRTrack.LINALG:
        raise ValueError(
            "ITritonToLinalgAdapter output requires track='linalg'; "
            "use AnchorIRLifecycleOrchestrator directly for TritonGPU output"
        )
    if source_name is None:
        adapter_name = adapter.name()
        source_name = "<adapter:%s>" % adapter_name

    adapter_output = adapter.convert(ttir_module, metadata, context)
    # The production entry deliberately constructs the strict orchestrator
    # itself.  Accepting a caller-supplied lookalike here would make it possible
    # to bypass both validation boundaries while still calling the API that is
    # documented as fail-closed.
    lifecycle = AnchorIRLifecycleOrchestrator()
    if isinstance(adapter_output, str):
        return lifecycle.run_text_or_raise(
            adapter_output,
            hook=hook,
            spec_version=spec_version,
            track=track,
            context=context,
            source_name=source_name,
            backend_lowering=backend_lowering,
        )
    return lifecycle.run_module_or_raise(
        adapter_output,
        hook=hook,
        spec_version=spec_version,
        track=track,
        context=context,
        backend_lowering=backend_lowering,
    )
