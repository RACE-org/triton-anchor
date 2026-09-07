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

import inspect
from typing import Any, TYPE_CHECKING, Optional

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


def extract_hw_capability(
    backend: Any = None,
    options: Any = None,
    metadata: Optional[dict] = None,
) -> Optional[HWCapability]:
    """Best-effort extraction of ``HWCapability`` from 3.6 compile objects.

    Triton 3.6 core still passes a ``GPUTarget`` through the compiler, so
    triton-anchor backend plugins may publish ``HWCapability`` either on the
    parsed options object, the backend object, or through an optional hook.
    """
    for container in (metadata, options, backend):
        hw = _extract_hw_from_container(container)
        if hw is not None:
            return hw

    for hook_name in ("get_anchor_hw_capability", "get_hw_capability"):
        hook = getattr(backend, hook_name, None) if backend is not None else None
        if hook is None:
            continue
        hw = _call_hw_capability_hook(hook, options)
        if hw is not None:
            return hw
    return None


def resolve_adapter_route_for_compile(
    backend: Any = None,
    options: Any = None,
    metadata: Optional[dict] = None,
    hw: Optional[HWCapability] = None,
):
    """Resolve the Anchor adapter route before Triton's cache key is built."""
    hw = hw or extract_hw_capability(backend=backend, options=options, metadata=metadata)
    if hw is None:
        return None

    from .adapters.registry import AdapterRegistry

    return AdapterRegistry.get_route(hw, metadata=metadata)


def make_anchor_ir(
    mod: Any,
    metadata: dict,
    hw: Optional[HWCapability] = None,
    context: Any = None,
):
    """Route TTIR through the selected adapter and return AnchorIR."""
    hw = hw or extract_hw_capability(metadata=metadata)
    if hw is None:
        raise ValueError(
            "make_anchor_ir requires HWCapability via hw or metadata['hw_capability']"
        )

    from .adapters.registry import AdapterRegistry

    adapter, _route = AdapterRegistry.resolve(hw, metadata=metadata)
    anchor_ir = adapter.convert(mod, metadata, context)
    _validate_anchor_ir_output(anchor_ir, hw, metadata)
    return anchor_ir


def _validate_anchor_ir_output(anchor_ir: Any, hw: HWCapability, metadata: dict) -> None:
    from .anchor_ir import AnchorIRError, AnchorIRValidator

    ir_text = anchor_ir if isinstance(anchor_ir, str) else str(anchor_ir)
    validator = AnchorIRValidator(track=hw.anchor_ir_track)
    pre_violations = validator.validate_pre_hook(ir_text)
    metadata["anchor_ir_pre_hook_validation"] = {
        "track": hw.anchor_ir_track.value,
        "violations": [str(violation) for violation in pre_violations],
    }
    if pre_violations:
        details = "\n".join(str(violation) for violation in pre_violations)
        raise AnchorIRError(f"AnchorIR pre-hook validation failed:\n{details}")

    ext_allowed = _extract_extension_dialects(metadata)
    post_violations = validator.validate_post_hook(ir_text, ext_allowed=ext_allowed)
    metadata["anchor_ir_post_hook_validation"] = {
        "track": hw.anchor_ir_track.value,
        "ext_allowed": sorted(ext_allowed),
        "violations": [str(violation) for violation in post_violations],
    }
    if post_violations:
        details = "\n".join(str(violation) for violation in post_violations)
        raise AnchorIRError(f"AnchorIR post-hook validation failed:\n{details}")


def _extract_extension_dialects(metadata: dict) -> set[str]:
    for key in (
        "anchor_ir_extra_allowed_dialects",
        "extra_allowed_dialects",
        "allowed_dialects",
    ):
        values = metadata.get(key)
        if values:
            return {str(value) for value in values}

    backend = metadata.get("backend") or metadata.get("backend_plugin")
    hook = getattr(backend, "get_allowed_dialects", None) if backend is not None else None
    if hook is None:
        return set()
    return {str(value) for value in hook()}


def _extract_hw_from_container(container: Any) -> Optional[HWCapability]:
    if container is None:
        return None
    keys = ("hw_capability", "anchor_hw_capability", "hw")
    if isinstance(container, dict):
        for key in keys:
            hw = container.get(key)
            if _looks_like_hw_capability(hw):
                return hw
        return None
    for key in keys:
        hw = getattr(container, key, None)
        if _looks_like_hw_capability(hw):
            return hw
    return None


def _looks_like_hw_capability(value: Any) -> bool:
    return (
        value is not None
        and hasattr(value, "ptr_model")
        and hasattr(value, "anchor_ir_track")
        and hasattr(value, "compute_paradigm")
    )


def _call_hw_capability_hook(hook: Any, options: Any) -> Optional[HWCapability]:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        hw = hook(options) if required else hook()
    else:
        try:
            hw = hook(options)
        except TypeError:
            hw = hook()

    if hw is None or _looks_like_hw_capability(hw):
        return hw
    raise TypeError(
        "HWCapability hook must return a triton_anchor.hw_capability.HWCapability "
        f"compatible object, got {type(hw).__name__}"
    )
