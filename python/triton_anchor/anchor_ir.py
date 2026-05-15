"""
AnchorIR Specification & Two-Phase Validator
==============================================

AnchorIR is the **contract** between Adapters (Layer 2) and Backend
Plugins (Layer 3).  Regardless of which Adapter produced the IR, the output
must conform to AnchorIR so that any backend can consume it.

Dual-Track Design (v0.1.3):
  - **Linalg Track**: linalg/tensor/memref-centric IR (AME / Tensor backends)
  - **TritonGPU Track**: TritonGPU IR with Encoding attributes (GPU backends)

Two-Phase Validation:
  - ``validate_anchor_ir_pre_hook()``: runs BEFORE ``on_anchor_ir_ready()``
    — checks base whitelist + forbidden list only
  - ``validate_anchor_ir_post_hook()``: runs AFTER ``on_anchor_ir_ready()``
    — checks base + extension whitelist + forbidden list

Key invariants:
  - Each Track has its own whitelist and forbidden list
  - Numerical consistency: same Track, different Adapters → same numerical
    results within tolerance (float rtol≤1e-5, int bitwise)
  - Extension dialects declared by backend via ``get_allowed_dialects()``

Stability guarantee: the allowed dialect whitelist is append-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ═══════════════════════════════════════════════════════════════════════
# AnchorIR Track — the two fundamental output forms
# ═══════════════════════════════════════════════════════════════════════

class AnchorIRTrack(Enum):
    """AnchorIR dual-track output specification.

    Decoupled from ComputeParadigm — backends may freely combine
    compute paradigm and IR track (e.g., a RISC-V GPU with Tensor Core
    could use TRITON_GPU track with RISC-V instructions).
    """
    LINALG = "linalg"           # Linalg Track (AME / Tensor)
    TRITON_GPU = "triton_gpu"   # TritonGPU Track (gpGPU)


class AnchorIRDialectStatus(Enum):
    """Classification of MLIR dialects in AnchorIR."""
    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"
    EXTENSION = "extension"    # Allowed only when declared via get_allowed_dialects()


# ═══════════════════════════════════════════════════════════════════════
# Per-Track Dialect Configuration
# ═══════════════════════════════════════════════════════════════════════

# Linalg Track base whitelist
LINALG_TRACK_ALLOWED: Set[str] = {
    "linalg",           # Core computation
    "linalg_ext",       # Extended ops (scatter, gather, atomic) from triton-linalg
    "tensor",           # Tensor operations
    "memref",           # Memory reference operations
    "arith",            # Arithmetic operations
    "math",             # Math operations (sin, cos, exp, ...)
    "math_ext",         # Extended math (from triton-linalg)
    "scf",              # Structured control flow
    "func",             # Function operations
    "cf",               # Control flow (basic blocks)
    "affine",           # Affine operations
    "aux",              # Auxiliary operations (from triton-linalg)
    "index",            # Index operations
    "bufferization",    # Bufferization operations
    "vector",           # Vector operations
}

# Linalg Track forbidden dialects
LINALG_TRACK_FORBIDDEN: Set[str] = {
    "tt",               # Triton dialect — must be fully lowered
    "triton",           # Alias for tt
    "tts",              # Triton-shared transition dialect
    "tptr",             # Triton pointer transition dialect
    "smt",              # DSL Extension Python namespace — must be lowered to xsmt.*
    "triton_gpu",       # TritonGPU dialect (wrong track)
    "nvidia_gpu",       # NVIDIA-specific
}

# TritonGPU Track base whitelist
TRITON_GPU_TRACK_ALLOWED: Set[str] = {
    "triton_gpu",       # TritonGPU dialect (with Encoding attributes)
    "tt",               # Triton Op retained (with Encoding)
    "arith",            # Arithmetic operations
    "math",             # Math operations
    "scf",              # Structured control flow
    "func",             # Function operations
    "gpu",              # GPU-specific operations (optional)
    "nvgpu",            # NVIDIA GPU operations (optional)
}

# TritonGPU Track forbidden dialects
TRITON_GPU_TRACK_FORBIDDEN: Set[str] = {
    "tts",              # Transition dialects forbidden
    "tptr",             # Transition dialects forbidden
    "smt",              # DSL Extension Python namespace
}


def _get_track_config(track: AnchorIRTrack) -> Tuple[Set[str], Set[str]]:
    """Get base whitelist and forbidden set for a given Track.

    Args:
        track: The AnchorIR track.

    Returns:
        Tuple of (base_allowed, forbidden) dialect sets.
    """
    if track == AnchorIRTrack.LINALG:
        return LINALG_TRACK_ALLOWED.copy(), LINALG_TRACK_FORBIDDEN.copy()
    elif track == AnchorIRTrack.TRITON_GPU:
        return TRITON_GPU_TRACK_ALLOWED.copy(), TRITON_GPU_TRACK_FORBIDDEN.copy()
    else:
        raise ValueError(f"Unknown AnchorIR track: {track}")


# Legacy combined sets for backward compatibility
ALLOWED_DIALECTS: Set[str] = LINALG_TRACK_ALLOWED.copy()
FORBIDDEN_DIALECTS: Set[str] = LINALG_TRACK_FORBIDDEN.copy()


# ═══════════════════════════════════════════════════════════════════════
# AnchorIR Violation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnchorIRViolation:
    """A single violation found during AnchorIR validation."""
    line_number: int
    dialect: str
    op_name: str
    message: str

    def __str__(self):
        return f"  L{self.line_number}: {self.op_name} — {self.message}"


# ═══════════════════════════════════════════════════════════════════════
# Two-Phase AnchorIR Validator
# ═══════════════════════════════════════════════════════════════════════

class AnchorIRValidator:
    """Validates that an MLIR module conforms to the AnchorIR specification.

    Supports two-phase validation (v0.1.3):
      - Phase 1 (pre-hook): base whitelist + forbidden — before ``on_anchor_ir_ready()``
      - Phase 2 (post-hook): base + extension whitelist + forbidden — after Hook injection

    Usage::

        validator = AnchorIRValidator(track=AnchorIRTrack.LINALG)

        # Phase 1: before Hook
        violations = validator.validate_pre_hook(ir_text)

        # ... on_anchor_ir_ready() injects extension ops ...

        # Phase 2: after Hook (with extension whitelist)
        violations = validator.validate_post_hook(ir_text, ext_allowed={"xsmt", "xsmt_async"})

    Legacy single-phase validation is still supported::

        validator = AnchorIRValidator()
        violations = validator.validate(ir_text)
    """

    # Pattern to match MLIR operations: "dialect.op_name"
    _OP_PATTERN = re.compile(
        r'^\s*'                    # leading whitespace
        r'(?:%\w+\s*(?:,\s*%\w+\s*)*=\s*)?'  # optional results: %foo, %bar =
        r'"?'                      # optional quote
        r'(\w+)\.(\w[\w.]*)'      # dialect.op_name
        r'"?',                     # optional quote
        re.MULTILINE
    )

    def __init__(
        self,
        track: Optional[AnchorIRTrack] = None,
        extra_allowed: Optional[Set[str]] = None,
        extra_forbidden: Optional[Set[str]] = None,
    ):
        """Initialize the validator.

        Args:
            track: AnchorIR track. If None, defaults to LINALG for backward compat.
            extra_allowed: Additional allowed dialects (merged with base whitelist).
            extra_forbidden: Additional forbidden dialects.
        """
        self.track = track or AnchorIRTrack.LINALG
        base_allowed, base_forbidden = _get_track_config(self.track)

        self.allowed = base_allowed
        self.forbidden = base_forbidden
        if extra_allowed:
            self.allowed |= extra_allowed
        if extra_forbidden:
            self.forbidden |= extra_forbidden

    def _scan_ops(self, ir_text: str, allowed: Set[str], forbidden: Set[str]) -> List[AnchorIRViolation]:
        """Scan IR text and report violations against given whitelist/forbidden sets."""
        violations: List[AnchorIRViolation] = []
        lines = ir_text.splitlines()

        for line_no, line in enumerate(lines, start=1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("#"):
                continue

            for match in self._OP_PATTERN.finditer(line):
                dialect = match.group(1)
                op_name = f"{dialect}.{match.group(2)}"

                if dialect in forbidden:
                    violations.append(AnchorIRViolation(
                        line_number=line_no,
                        dialect=dialect,
                        op_name=op_name,
                        message=f"Forbidden dialect '{dialect}' must be fully lowered before AnchorIR"
                    ))
                elif dialect not in allowed:
                    violations.append(AnchorIRViolation(
                        line_number=line_no,
                        dialect=dialect,
                        op_name=op_name,
                        message=(
                            f"Unknown dialect '{dialect}'. "
                            f"Register it via backend's get_allowed_dialects()."
                        )
                    ))

        return violations

    # ─── Two-Phase Validation API (v0.1.3) ────────────────────────────

    def validate_pre_hook(self, ir_text: str) -> List[AnchorIRViolation]:
        """Phase 1 (pre-hook): validate against base whitelist only.

        Runs BEFORE ``on_anchor_ir_ready()`` — ensures Adapter output
        does not contain forbidden dialects. Does NOT check extension
        whitelist (backend extension ops not yet injected).

        Args:
            ir_text: The MLIR module as a string.

        Returns:
            A list of ``AnchorIRViolation`` objects. Empty means valid.
        """
        base_allowed, forbidden = _get_track_config(self.track)
        return self._scan_ops(ir_text, base_allowed, forbidden)

    def validate_post_hook(
        self,
        ir_text: str,
        ext_allowed: Optional[Set[str]] = None,
    ) -> List[AnchorIRViolation]:
        """Phase 2 (post-hook): validate against base + extension whitelist.

        Runs AFTER ``on_anchor_ir_ready()`` — ensures backend-injected
        extension ops are declared via ``get_allowed_dialects()``.

        Args:
            ir_text: The MLIR module as a string.
            ext_allowed: Additional dialects declared by the backend plugin
                via ``get_allowed_dialects()``.

        Returns:
            A list of ``AnchorIRViolation`` objects. Empty means valid.
        """
        base_allowed, forbidden = _get_track_config(self.track)
        all_allowed = base_allowed | (ext_allowed or set())
        return self._scan_ops(ir_text, all_allowed, forbidden)

    # ─── Legacy Single-Phase API ──────────────────────────────────────

    def validate(self, ir_text: str) -> List[AnchorIRViolation]:
        """Legacy single-phase validation (backward compatible).

        Uses the allowed/forbidden sets configured at construction time.
        """
        return self._scan_ops(ir_text, self.allowed, self.forbidden)

    def is_valid(self, ir_text: str) -> bool:
        """Quick check — returns True if IR conforms to AnchorIR."""
        return len(self.validate(ir_text)) == 0

    def validate_and_raise(self, ir_text: str, context: str = "") -> None:
        """Validate and raise ``AnchorIRError`` if violations are found."""
        violations = self.validate(ir_text)
        if violations:
            header = f"AnchorIR validation failed"
            if context:
                header += f" for {context}"
            details = "\n".join(str(v) for v in violations)
            raise AnchorIRError(f"{header}:\n{details}")


class AnchorIRError(Exception):
    """Raised when an MLIR module violates the AnchorIR specification."""
    pass

