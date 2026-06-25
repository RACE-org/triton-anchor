"""
Unified TTIR Pipeline
======================

Extracts the 7 mandatory TTIR optimization passes that are 100% shared
across all three projects (triton-shared, triton_race, fantasy-triton).

This is a **core invariant** — the pass list is append-only and
synchronized with upstream Triton.

The pipeline also supports conditional passes controlled by ``HWCapability``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .hw_capability import HWCapability


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
    """Inject hardware attributes into an MLIR module.

    Called after TTIR optimization, before lowering to hardware-aware IR.
    Backend plugins can use ``on_ttir_ready()`` hook to inject additional
    attributes.

    Args:
        mod: An MLIR module.
        hw: The target hardware capability.
        metadata: Compilation metadata dict (updated in-place).
    """
    try:
        from triton._C.libtriton import ir

        builder = ir.builder(mod.context)

        mod.set_attr("hw.name", builder.get_string_attr(hw.name))
        mod.set_attr("hw.paradigm", builder.get_string_attr(hw.compute_paradigm.value))
        mod.set_attr("hw.arch_family", builder.get_string_attr(hw.arch_family))

        if hw.arch_family == "riscv" and hw.matrix_cap:
            mod.set_attr("hw.num_threads", builder.get_int32_attr(hw.num_cores))
        elif hw.arch_family == "tpu" and hw.tensor_cap:
            mod.set_attr("hw.core_num", builder.get_int32_attr(hw.tensor_cap.num_cores))
        elif hw.arch_family == "gpu" and hw.gpgpu_cap:
            mod.set_attr("hw.num_warps", builder.get_int32_attr(hw.gpgpu_cap.num_warps))

        if hw.anchor_ir_track.value == "linalg" and hw.ptr_model != "gpu":
            num_threads = hw.num_threads or hw.num_cores
            mod.set_attr("tt.num_threads", builder.get_int32_attr(num_threads))
            mod.set_attr(
                "tt.force_vector_interleave",
                builder.get_int32_attr(hw.force_vector_interleave),
            )
            if hw.arch_id:
                mod.set_attr("tt.arch_id", builder.get_string_attr(hw.arch_id))
            metadata["num_threads"] = num_threads
            metadata["force_vector_interleave"] = hw.force_vector_interleave
            if hw.arch_id:
                metadata["arch_id"] = hw.arch_id

    except ImportError:
        # Triton not installed — keep metadata-only path for pure Python tests.
        pass

    metadata["hw"] = hw
    metadata["hw_capability"] = hw
    metadata["hw_name"] = hw.name
    metadata["hw_paradigm"] = hw.compute_paradigm.value
    metadata["hw_arch_family"] = hw.arch_family
