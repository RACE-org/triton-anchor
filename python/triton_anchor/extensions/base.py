"""
DSLExtensionPlugin — Layer 0 Plugin Interface
================================================

Hardware vendors use this interface to extend the Triton DSL with
custom builtins (e.g., ``smt.parallel``, ``smt.dot`` for SpacemiT).

A DSL extension consists of three components:
  1. **Python builtins** — registered via ``get_builtins()``
  2. **MLIR dialect .so** — custom dialect definitions (optional)
  3. **Lowering patterns .so** — op lowering rules (optional)

Distribution:
  DSL extensions are published as independent pip packages with
  ``entry_points("triton.dsl_extensions")`` registration::

      # triton-ext-spacemit/pyproject.toml
      [project.entry-points."triton.dsl_extensions"]
      smt = "triton_ext_spacemit.dsl:SmtExtension"

Lifecycle:
  1. Discovery via entry_points at JIT compile time
  2. Builtin registration into Triton's type system
  3. MLIR dialect loading into the compilation context
  4. Lowering pattern registration for Adapter passes

Stability guarantee:
  - ``get_builtins()`` is the only required method
  - New methods always have default implementations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BuiltinSpec:
    """Specification for a single DSL extension builtin.

    Describes how a custom builtin function should be compiled:
      - ``ir_builder``: Python callable that generates MLIR IR for this op
      - ``semantic_fn``: Type inference / semantic analysis function
      - ``arg_types``: Expected argument types (for validation)
      - ``doc``: Documentation string

    Example::

        BuiltinSpec(
            name="dot",
            ir_builder=lambda builder, a, b: builder.create_smt_dot(a, b),
            semantic_fn=lambda a, b: check_matmul_types(a, b),
            arg_types=["tensor", "tensor"],
            ret_type="tensor",
            doc="SMT matrix dot product using AME instructions",
        )
    """
    name: str
    ir_builder: Optional[Callable] = None
    semantic_fn: Optional[Callable] = None
    arg_types: List[str] = field(default_factory=list)
    ret_type: str = "void"
    doc: str = ""


class DSLExtensionPlugin(ABC):
    """Abstract base class for DSL extension plugins.

    Hardware vendors implement this interface to add custom builtins
    to the Triton language.

    Minimal implementation requires only ``name``, ``namespace``,
    and ``get_builtins()``.

    Example::

        class SmtExtension(DSLExtensionPlugin):
            name = "smt"
            namespace = "smt"
            target_backend = "spacemit"

            def get_builtins(self):
                return {
                    "parallel": BuiltinSpec(name="parallel", ...),
                    "dot": BuiltinSpec(name="dot", ...),
                    "alloc": BuiltinSpec(name="alloc", ...),
                    "mbarrier": BuiltinSpec(name="mbarrier", ...),
                }
    """

    # ── Identity ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Extension name (e.g., 'smt', 'tpu')."""
        ...

    @property
    @abstractmethod
    def namespace(self) -> str:
        """MLIR dialect namespace (e.g., 'smt', 'xsmt').

        This namespace is used to:
        1. Prefix Python builtins: ``smt.parallel``, ``smt.dot``
        2. Match MLIR ops: ``smt.parallel_op``, ``smt.dot_op``
        """
        ...

    @property
    def target_backend(self) -> Optional[str]:
        """Target backend name, or None if the extension is universal.

        When set, the frontend validates that the extension is only used
        with the matching backend, raising ``IncompatibleExtensionError``
        otherwise.
        """
        return None

    # ── Builtin Registration ─────────────────────────────────────────

    @abstractmethod
    def get_builtins(self) -> Dict[str, BuiltinSpec]:
        """Return the custom builtin specifications.

        Returns:
            Dict mapping builtin name to BuiltinSpec.
            Keys are the short names (e.g., 'dot'), which will be
            accessible as ``namespace.dot`` in user code.
        """
        ...

    # ── MLIR Dialect & Lowering ──────────────────────────────────────

    def get_dialect_library(self) -> Optional[str]:
        """Path to the MLIR dialect shared library (.so).

        Returns None if no custom dialect is needed (pure-Python extension).
        """
        return None

    def get_lowering_patterns(self) -> Optional[str]:
        """Path to the lowering pattern shared library (.so).

        The library should register ConversionPattern instances that
        lower extension ops to standard Linalg / AnchorIR ops.

        Returns None if lowering is handled differently (e.g., inline).
        """
        return None

    # ── Validation ───────────────────────────────────────────────────

    def validate_kernel_compatibility(
        self, kernel_ir: str, target_backend: str
    ) -> Tuple[bool, str]:
        """Check if this extension is compatible with the target backend.

        Default: compatible if ``target_backend`` is None or matches.
        """
        if self.target_backend and self.target_backend != target_backend:
            return False, (
                f"DSL extension '{self.name}' requires backend "
                f"'{self.target_backend}', but target is '{target_backend}'"
            )
        return True, ""


class IncompatibleExtensionError(Exception):
    """Raised when a DSL extension is incompatible with the target backend."""
    pass
