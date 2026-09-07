"""
Anchor Adapter Interfaces
=========================

The adapter pattern allows the unified frontend to support multiple
post-TTIR AnchorIR tracks:

  1. **triton-gpu** — TritonGPU AnchorIR track for GPGPU backends
  2. **triton-shared** — Structured pointer analysis for Linalg track
  3. **triton-linalg** — AxisInfo-style Linalg track path
  4. **hybrid** — Structured first, Router-authorized AxisInfo fallback

All adapters must produce output that conforms to the **AnchorIR** spec.

ABI Isolation Strategy (v0.1.3):
  Two adapter base classes provide clean ABI separation:
  - ``ILinalgOptAdapter``:     subprocess-based, calls external MLIR opt tools
  - ``ILinalgPybindAdapter``:  in-process, calls pybind11-bound MLIR passes

  This prevents C++ ABI collisions between different MLIR builds — e.g.,
  triton-shared's opt tool uses its own libMLIR, while triton_race's passes
  are compiled into the host libtriton.so.

Future extensibility:
  - Custom adapters: new analysis methods via plugin
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from ..anchor_ir import AnchorIRTrack

PtrModel = str


@dataclass(frozen=True)
class AdapterInfo:
    """Declarative capability summary for one Anchor adapter.

    The registry stores adapter instances; the router reads this immutable
    description to decide whether an adapter can serve a specific AnchorIR
    track, pointer model, backend capability set, and kernel op coverage.
    """

    name: str
    supported_tracks: Tuple[AnchorIRTrack, ...]
    supported_ptr_models: Tuple[PtrModel, ...]
    required_backend_capabilities: Tuple[str, ...] = ()
    supported_ops: Optional[Tuple[str, ...]] = None
    supports_internal_fallback: bool = False


@dataclass
class AdapterConversionContext:
    """Context passed from ``AdapterRouter`` to ``convert()``.

    ``fallback_authorized`` is intentionally explicit.  Adapters such as
    ``HybridAdapter`` may contain multiple conversion paths, but they must not
    enter a weaker path unless the router granted that policy decision.
    """

    hw: Any = None
    decision: Any = None
    fallback_authorized: bool = False
    fallback_candidates: Tuple[str, ...] = ()
    metadata_key: str = "adapter_decision"
    extras: Mapping[str, Any] = field(default_factory=dict)


class IAnchorAdapter(ABC):
    """Abstract interface for TTIR → AnchorIR conversion adapters.

    Each adapter wraps a specific pointer analysis + conversion pipeline
    (e.g., triton-shared or triton-linalg) and must produce AnchorIR-
    compliant output.

    Subclass Contract:
        1. ``name()`` must return a unique string identifier
        2. ``convert()`` must produce valid AnchorIR from an optimized TTIR module
        3. ``describe()`` should expose routable adapter capabilities
        4. ``validate_output()`` should check AnchorIR compliance (optional override)
    """

    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this adapter (e.g., 'triton-linalg')."""
        ...

    @abstractmethod
    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Convert an optimized TTIR module to AnchorIR.

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

    def describe(self) -> AdapterInfo:
        """Return routable capabilities for this adapter.

        Subclasses should override at least ``supported_ptr_models`` and
        ``required_backend_capabilities``.  The default is a conservative
        Linalg-track adapter with no pointer-model match.
        """
        return AdapterInfo(
            name=self.name(),
            supported_tracks=self.get_supported_tracks(),
            supported_ptr_models=self.get_supported_ptr_models(),
            required_backend_capabilities=self.get_required_backend_capabilities(),
            supported_ops=self.get_supported_ops(),
            supports_internal_fallback=self.supports_internal_fallback(),
        )

    def get_supported_tracks(self) -> Tuple[AnchorIRTrack, ...]:
        """AnchorIR tracks this adapter can produce."""
        return (AnchorIRTrack.LINALG,)

    def get_supported_ptr_models(self) -> Tuple[PtrModel, ...]:
        """Pointer models this adapter natively supports."""
        return ()

    def get_required_backend_capabilities(self) -> Tuple[str, ...]:
        """Backend capabilities required to consume this adapter's output."""
        return ()

    def get_supported_ops(self) -> Optional[Tuple[str, ...]]:
        """Triton op coverage for this adapter.

        ``None`` means the adapter does not declare op-level coverage.  ``("*",)``
        means it claims complete coverage for router purposes.
        """
        return None

    def supports_internal_fallback(self) -> bool:
        """Whether the adapter may contain multiple conversion paths."""
        return False

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

    def get_required_passes(self) -> list[str]:
        """List of MLIR pass names this adapter requires.

        Used for documentation and diagnostic purposes.
        """
        return []

    def get_output_dialects(self) -> list[str]:
        """List of MLIR dialects this adapter may produce in its output.

        Used for AnchorIR extension validation — if an adapter produces
        a dialect not in the AnchorIR whitelist, it must be registered
        as a DSL extension.
        """
        return ["linalg", "tensor", "memref", "arith", "math", "scf", "func"]


class ITritonToLinalgAdapter(IAnchorAdapter, ABC):
    """Abstract interface for TTIR → Linalg-track adapters."""

    pass


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

    Used by: TritonSharedAdapter (spine-triton / triton-shared)
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
