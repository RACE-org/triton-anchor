"""
ITritonToLinalgAdapter — Adapter Pattern Interface
====================================================

The adapter pattern allows the unified frontend to support two
fundamentally different pointer analysis / lowering strategies:

  1. **triton-shared** — Structured/Unstructured dual-path pointer analysis
     (used by triton-shared, from Microsoft)
  2. **triton-linalg** — AxisInfo unified analysis
     (used by triton_race, from Cambricon)

Both adapters must produce output that conforms to the **AnchorIR** spec.

ABI Isolation Strategy (v0.1.3):
  Two adapter base classes provide clean ABI separation:
  - ``ILinalgOptAdapter``:     subprocess-based, calls external MLIR opt tools
  - ``ILinalgPybindAdapter``:  in-process, calls pybind11-bound MLIR passes

  This prevents C++ ABI collisions between different MLIR builds — e.g.,
  triton-shared's opt tool uses its own libMLIR, while triton_race's passes
  are compiled into the host libtriton.so.

Future extensibility:
  - HybridAdapter: tries Structured first, falls back to AxisInfo
  - Custom adapters: new analysis methods via plugin
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class ITritonToLinalgAdapter(ABC):
    """Abstract interface for TTIR → AnchorIR conversion adapters.

    Each adapter wraps a specific pointer analysis + conversion pipeline
    (e.g., triton-shared, triton-linalg, or TritonGPU) and must produce
    AnchorIR-compliant output.

    Subclass Contract:
        1. ``name()`` must return a unique string identifier
        2. ``convert()`` must produce valid AnchorIR from an optimized TTIR module
        3. ``validate_output()`` should check AnchorIR compliance (optional override)
    """

    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this adapter (e.g., 'triton-linalg')."""
        ...

    @abstractmethod
    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Convert an optimized TTIR module to Linalg IR (AnchorIR).

        Args:
            ttir_module: The MLIR module after TTIR optimization.
                For in-process adapters: an ``ir.Module`` object.
                For out-of-process adapters: the MLIR text as ``str``.
            metadata: Compilation metadata dict (mutated in-place).
            context: Optional MLIR context.

        Returns:
            The converted module.
                For in-process adapters: the same ``ir.Module`` (mutated).
                For out-of-process adapters: MLIR text as ``str``.

        Raises:
            AdapterConversionError: If the conversion fails.
        """
        ...

    def validate_output(self, linalg_ir: Any) -> bool:
        """Validate that the adapter output conforms to AnchorIR.

        Default implementation uses the AnchorIRValidator.
        Subclasses may override for custom validation.

        Args:
            linalg_ir: The converted MLIR module (text or object).

        Returns:
            True if valid, False otherwise.
        """
        from ..anchor_ir import AnchorIRValidator

        validator = AnchorIRValidator()
        ir_text = str(linalg_ir) if not isinstance(linalg_ir, str) else linalg_ir
        return validator.is_valid(ir_text)

    def get_required_passes(self) -> List[str]:
        """List of MLIR pass names this adapter requires.

        Used for documentation and diagnostic purposes.
        """
        return []

    def cache_key_components(self) -> Dict[str, Any]:
        """Stable adapter identity included in Router metadata/cache keys."""
        adapter_type = type(self)
        return {
            "adapter": self.name(),
            "adapter_type": f"{adapter_type.__module__}.{adapter_type.__qualname__}",
        }

    def supported_routes(self) -> List[Tuple[str, str]]:
        """Return supported ``(anchor_ir_track, ptr_model)`` pairs.

        ``ptr_model="*"`` means every pointer model on that track is accepted.
        Third-party adapters that predate this hook are treated as compatible
        unless they override ``supports`` or ``supported_routes``.
        """
        return []

    def supports(self, hw: Any) -> bool:
        """Return whether this adapter can handle the given capability."""
        routes = self.supported_routes()
        if not routes:
            return True
        track = getattr(hw, "anchor_ir_track", "")
        track_value = getattr(track, "value", track)
        ptr_model = getattr(hw, "ptr_model", "")
        return any(
            route_track == track_value
            and (route_ptr_model == "*" or route_ptr_model == ptr_model)
            for route_track, route_ptr_model in routes
        )

    def get_output_dialects(self) -> List[str]:
        """List of MLIR dialects this adapter may produce in its output.

        Used for AnchorIR extension validation — if an adapter produces
        a dialect not in the AnchorIR whitelist, it must be registered
        as a DSL extension.
        """
        return ["linalg", "tensor", "memref", "arith", "math", "scf", "func"]


# ═══════════════════════════════════════════════════════════════════════
# ABI-Isolated Adapter Base Classes (v0.1.3)
# ═══════════════════════════════════════════════════════════════════════


class ILinalgOptAdapter(ITritonToLinalgAdapter, ABC):
    """Adapter variant using out-of-process MLIR opt tool (subprocess).

    ABI Safety:  The external opt tool runs in a separate process,
    so its libMLIR symbols never collide with the host libtriton.so.

    Characteristics:
      - ~200ms subprocess overhead per conversion
      - Input/output via MLIR text files
      - Portable: works with any MLIR opt binary
      - Requires the opt tool to be installed and discoverable

    Used by: TritonSharedAdapter (triton-shared / triton-shared)
    """

    pass


class ILinalgPybindAdapter(ITritonToLinalgAdapter, ABC):
    """Adapter variant using in-process pybind11-bound MLIR passes.

    ABI Safety:  The passes must be compiled into the same libtriton.so
    as the host Triton runtime.  Cross-backend .so loading is NOT safe.

    Characteristics:
      - ~0ms overhead (direct function call)
      - Input/output via ``ir.Module`` objects
      - Fast, but requires matching MLIR ABI
      - Passes must be linked at build time

    Used by: TritonLinalgAdapter (triton_race / Sophgo TPU)
    """

    pass


class AdapterConversionError(Exception):
    """Raised when an Adapter fails to convert TTIR to Linalg IR."""

    def __init__(self, adapter_name: str, kernel_name: str = "", detail: str = ""):
        self.adapter_name = adapter_name
        self.kernel_name = kernel_name
        self.detail = detail
        msg = f"Adapter '{adapter_name}' failed to convert"
        if kernel_name:
            msg += f" kernel '{kernel_name}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
