"""
Hardware Capability & Compute Paradigm
=======================================

Core invariant of the unified frontend. HWCapability replaces the minimal
``GPUTarget(backend, arch, warp_size)`` with a rich, declarative description
of the target hardware.

Three compute paradigms are defined:
  - AME_MATRIX:  CPU-integrated matrix registers + matrix/vector ops
  - TENSOR_PROCESSOR:  Dedicated tensor compute units with own memory
  - GPGPU:  SIMT threads + shared memory + warp execution

Design decisions:
  - ``to_gpu_target()`` provides backward compatibility with existing
    ``GPUTarget``-based compile paths (triton_race, fantasy-triton).
  - Fields are append-only (never removed) to guarantee plugin stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Set, Tuple, Literal, Any, List

if TYPE_CHECKING:
    from .anchor_ir import AnchorIRTrack


# ═══════════════════════════════════════════════════════════════════════
# Compute Paradigm — the three fundamental ISA families
# ═══════════════════════════════════════════════════════════════════════


class ComputeParadigm(Enum):
    """Compute paradigm of the target hardware.

    This enum captures the *essential nature* of the hardware, not just
    a parameter — it determines the entire lowering strategy.
    """

    AME_MATRIX = "ame_matrix"
    """CPU-internal matrix extension (RISC-V AME, ARM SME).
    Characteristics: matrix registers, CPU cache hierarchy, no DMA."""

    TENSOR_PROCESSOR = "tensor"
    """Dedicated tensor processing unit (Sophgo TPU, Google TPU).
    Characteristics: independent memory space (HBM/SRAM), DMA-based data movement."""

    GPGPU = "gpgpu"
    """General-purpose GPU (NVIDIA, AMD, USC).
    Characteristics: SIMT threads, shared memory, warp execution."""


# ═══════════════════════════════════════════════════════════════════════
# Paradigm-Specific Capability Descriptors
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MatrixCapability:
    """Capability descriptor for AME (Advanced Matrix Extension) hardware.

    Used by: SpacemiT X60, 玄铁 AME, ARM SME.
    """

    num_matrix_registers: int = 8
    tile_shape: Tuple[int, int] = (8, 8)
    supported_dtypes: Set[str] = field(default_factory=lambda: {"fp32", "fp16", "int8"})
    has_accumulator_tiles: bool = True
    vector_length: int = 256
    supports_pointwise: bool = True


@dataclass(frozen=True)
class TensorCapability:
    """Capability descriptor for dedicated tensor processor hardware.

    Used by: Sophgo BM1684X, Google TPU.
    """

    num_cores: int = 1
    local_mem_size: int = 0  # bytes, per-core local SRAM
    global_mem_size: int = 0  # bytes, HBM/DDR
    dma_channels: int = 1
    supported_dtypes: Set[str] = field(default_factory=lambda: {"fp32", "fp16", "int8"})
    max_tensor_dims: int = 4


@dataclass(frozen=True)
class GPGPUCapability:
    """Capability descriptor for gpGPU hardware.

    Used by: NVIDIA GPU, AMD GPU, USC GPU.
    """

    num_warps: int = 4
    warp_size: int = 32
    shared_mem_size: int = 49152  # bytes
    num_stages: int = 2
    num_ctas: int = 1
    cluster_dims: Tuple[int, int, int] = (1, 1, 1)
    supported_dtypes: Set[str] = field(
        default_factory=lambda: {"fp32", "fp16", "bf16", "int8"}
    )


# ═══════════════════════════════════════════════════════════════════════
# HWCapability — the unified hardware descriptor
# ═══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class _Diagnostic:
    status: Literal["ok", "warning", "error"]
    check: str
    message: str

@dataclass
class HWCapability:
    """Declarative hardware capability descriptor.

    This is the **core invariant** of the unified frontend.  Backend plugins
    declare their hardware's capabilities through this dataclass, and the
    frontend uses it to drive compilation decisions:

    - ``compute_paradigm`` selects the lowering path (linalg vs triton_gpu)
    - ``ptr_model`` selects the pointer analysis adapter
    - ``preferred_adapter`` overrides automatic adapter selection
    - paradigm-specific caps (``matrix_cap``, ``tensor_cap``, ``gpgpu_cap``)
      provide fine-grained hardware parameters

    Stability guarantee: fields are append-only, never removed.

    Example::

        hw = HWCapability(
            name="sophgo-bm1684x",
            arch_family="tpu",
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
            lowering_path="linalg",
            ptr_model="axis_info",
            tensor_cap=TensorCapability(num_cores=8, local_mem_size=16*1024*1024),
        )

    """

    # ── Identity ─────────────────────────────────────────────────────
    name: str  # e.g. "spacemit-x60", "sophgo-bm1684x"
    arch_family: str  # "riscv", "tpu", "gpu"

    # ── Compilation Strategy ─────────────────────────────────────────
    compute_paradigm: ComputeParadigm
    anchor_ir_track: "AnchorIRTrack"  # Decoupled from paradigm; backend controls
    ptr_model: Literal["structured", "axis_info", "hybrid", "gpu"]

    # ── Adapter Override ─────────────────────────────────────────────
    preferred_adapter: Optional[str] = None  # e.g. "triton-shared"

    # triton-shared lowering metadata for spine-style CPU/tensor backends.
    arch_id: Optional[str] = None
    force_vector_interleave: int = 2
    num_threads: Optional[int] = None

    # ── Paradigm-Specific Capabilities (mutually exclusive) ──────────
    matrix_cap: Optional[MatrixCapability] = None  # AME
    tensor_cap: Optional[TensorCapability] = None  # Tensor
    gpgpu_cap: Optional[GPGPUCapability] = None  # gpGPU

    # ── Optional Flags ───────────────────────────────────────────────
    enable_loop_unroll: bool = False
    num_cores: int = 1

    # ── Compatibility ────────────────────────────────────────────────

    def to_gpu_target(self):
        """Convert to a ``GPUTarget`` for backward compatibility.

        This allows HWCapability to be used in existing triton compilation
        paths that expect ``GPUTarget(backend, arch, warp_size)``.

        Returns:
            A ``GPUTarget``-compatible object.  If ``triton`` is not
            installed, returns a plain ``dict`` with the same fields.
        """
        backend = self._infer_backend_name()
        arch = self._infer_arch()
        warp_size = self._infer_warp_size()

        try:
            from triton.backends.compiler import GPUTarget

            return GPUTarget(backend=backend, arch=arch, warp_size=warp_size)
        except ImportError:
            # Fallback when triton is not installed (e.g., in tests)
            return {"backend": backend, "arch": arch, "warp_size": warp_size}

    def _infer_backend_name(self) -> str:
        """Infer the backend name string for GPUTarget compatibility."""
        # Map known hardware families to backend names
        _family_to_backend = {
            "tpu": "sophgo",
            "riscv": "spacemit",
            "gpu": "usc",
        }
        return _family_to_backend.get(self.arch_family, self.name.split("-")[0])

    def _infer_arch(self):
        """Infer architecture identifier for GPUTarget compatibility."""
        if self.gpgpu_cap:
            return 0  # Placeholder; real backends override
        return 0

    def _infer_warp_size(self) -> int:
        """Infer warp size for GPUTarget compatibility."""
        if self.gpgpu_cap:
            return self.gpgpu_cap.warp_size
        # Non-GPU paradigms don't have warps; use 0 as sentinel
        return 0

    # ── Validation ───────────────────────────────────────────────────

    def validate(self) -> None:
        """Validate that capability fields are self-consistent.

        Raises:
            ValueError: If paradigm-specific cap doesn't match compute_paradigm,
                or if lowering_path is inconsistent.
        """
        errors = [
            diagnostic
            for diagnostic in self._collect_diagnostics(include_successes=False)
            if diagnostic.status == "error"
        ]
        if errors:
            raise ValueError(self._format_diagnostics(errors))

    def __post_init__(self):
        """Validate capability fields and resolve AnchorIRTrack.

        Design decision: compute_paradigm and anchor_ir_track are decoupled.
        Default mapping: AME/Tensor → LINALG, GPGPU → TRITON_GPU,
        but backends may override (e.g., a RISC-V GPU with Tensor Core).
        """
        # Resolve string → enum if needed (backward compat)
        if isinstance(self.anchor_ir_track, str):
            from .anchor_ir import AnchorIRTrack

            object.__setattr__(
                self, "anchor_ir_track", AnchorIRTrack(self.anchor_ir_track)
            )

        self.validate()

    @property
    def lowering_path(self) -> str:
        """Backward-compatible lowering_path string.

        Returns:
            'linalg' or 'triton_gpu' based on anchor_ir_track.
        """
        return self.anchor_ir_track.value

    def diagnose(self) -> str:
        """Return a human-readable configuration check report.

        The report uses the same checks as ``validate()``, but keeps successful
        checks so backend authors can see the full configuration surface that
        was inspected.
        """
        diagnostics = self._collect_diagnostics(include_successes=True)
        errors = [item for item in diagnostics if item.status == "error"]
        warnings = [item for item in diagnostics if item.status == "warning"]

        lines = [
            f"HWCapability diagnose: {self.name}",
            f"status: {'FAIL' if errors else 'PASS'}",
            f"errors: {len(errors)}",
            f"warnings: {len(warnings)}",
            "",
            "configuration:",
            f"  arch_family: {self.arch_family}",
            f"  compute_paradigm: {self.compute_paradigm.value}",
            f"  anchor_ir_track: {self.anchor_ir_track.value}",
            f"  ptr_model: {self.ptr_model}",
            f"  preferred_adapter: {self.preferred_adapter or '<auto>'}",
            f"  arch_id: {self.arch_id or '<auto>'}",
            f"  force_vector_interleave: {self.force_vector_interleave}",
            f"  num_threads: {self.num_threads if self.num_threads is not None else '<auto>'}",
            f"  lowering_path: {self.lowering_path}",
            f"  num_cores: {self.num_cores}",
            "",
            "checks:",
        ]

        for diagnostic in diagnostics:
            prefix = {
                "ok": "OK",
                "warning": "WARN",
                "error": "ERROR",
            }[diagnostic.status]
            lines.append(f"  - {prefix} {diagnostic.check}: {diagnostic.message}")

        return "\n".join(lines)

    def _collect_diagnostics(self, include_successes: bool) -> List[_Diagnostic]:
        diagnostics: List[_Diagnostic] = []

        def add(status: Literal["ok", "warning", "error"], check: str, message: str) -> None:
            if status != "ok" or include_successes:
                diagnostics.append(_Diagnostic(status, check, message))

        self._check_required_capability(add)
        self._check_capability_values(add)
        self._check_preferred_adapter(add)
        return diagnostics

    def _check_required_capability(self, add) -> None:
        required_caps = {
            ComputeParadigm.AME_MATRIX: ("matrix_cap", self.matrix_cap),
            ComputeParadigm.TENSOR_PROCESSOR: ("tensor_cap", self.tensor_cap),
            ComputeParadigm.GPGPU: ("gpgpu_cap", self.gpgpu_cap),
        }

        if self.compute_paradigm not in required_caps:
            add("error", "compute_paradigm", f"unsupported compute paradigm: {self.compute_paradigm!r}")
            return

        cap_name, cap_value = required_caps[self.compute_paradigm]
        if cap_value is None:
            add("error", "paradigm_capability", f"{self.compute_paradigm.name} requires {cap_name}")
            return

        add("ok", "paradigm_capability", f"{self.compute_paradigm.name} uses {cap_name}")

    def _check_capability_values(self, add) -> None:
        self._check_positive_int(add, "num_cores", self.num_cores)
        self._check_optional_non_empty_string(add, "arch_id", self.arch_id)
        self._check_positive_int(add, "force_vector_interleave", self.force_vector_interleave)
        self._check_optional_positive_int(add, "num_threads", self.num_threads)

        if self.matrix_cap is not None:
            cap = self.matrix_cap
            self._check_positive_int(add, "matrix_cap.num_matrix_registers", cap.num_matrix_registers)
            self._check_positive_tuple(add, "matrix_cap.tile_shape", cap.tile_shape)
            self._check_positive_int(add, "matrix_cap.vector_length", cap.vector_length)
            self._check_supported_dtypes(add, "matrix_cap.supported_dtypes", cap.supported_dtypes)

        if self.tensor_cap is not None:
            cap = self.tensor_cap
            self._check_positive_int(add, "tensor_cap.num_cores", cap.num_cores)
            self._check_non_negative_int(add, "tensor_cap.local_mem_size", cap.local_mem_size)
            self._check_non_negative_int(add, "tensor_cap.global_mem_size", cap.global_mem_size)
            self._check_positive_int(add, "tensor_cap.dma_channels", cap.dma_channels)
            self._check_supported_dtypes(add, "tensor_cap.supported_dtypes", cap.supported_dtypes)
            self._check_positive_int(add, "tensor_cap.max_tensor_dims", cap.max_tensor_dims)

        if self.gpgpu_cap is not None:
            cap = self.gpgpu_cap
            self._check_positive_int(add, "gpgpu_cap.num_warps", cap.num_warps)
            self._check_positive_int(add, "gpgpu_cap.warp_size", cap.warp_size)
            self._check_non_negative_int(add, "gpgpu_cap.shared_mem_size", cap.shared_mem_size)
            self._check_positive_int(add, "gpgpu_cap.num_stages", cap.num_stages)
            self._check_positive_int(add, "gpgpu_cap.num_ctas", cap.num_ctas)
            self._check_positive_tuple(add, "gpgpu_cap.cluster_dims", cap.cluster_dims, expected_len=3)
            self._check_supported_dtypes(add, "gpgpu_cap.supported_dtypes", cap.supported_dtypes)

    def _check_preferred_adapter(self, add) -> None:
        if self.preferred_adapter is None:
            add("ok", "preferred_adapter", "not set; adapter will be selected from ptr_model")
            return

        try:
            from .adapters import AdapterRegistry
        except Exception as exc:
            add("error", "preferred_adapter", f"cannot import AdapterRegistry: {exc}")
            return

        adapter = AdapterRegistry.get(self.preferred_adapter)
        available = sorted(AdapterRegistry.list_adapters())
        if adapter is None:
            add(
                "error",
                "preferred_adapter",
                f"'{self.preferred_adapter}' is not registered; available adapters: {available}",
            )
            return

        add("ok", "preferred_adapter", f"'{self.preferred_adapter}' is registered")

        output_track = self._infer_adapter_output_track(adapter)
        if output_track is None:
            add(
                "warning",
                "adapter_output_track",
                f"cannot infer output track for adapter '{self.preferred_adapter}'",
            )
            return

        if output_track != self.anchor_ir_track:
            add(
                "error",
                "adapter_output_track",
                f"adapter '{self.preferred_adapter}' outputs {output_track.value}, "
                f"but anchor_ir_track is {self.anchor_ir_track.value}",
            )
            return

        add(
            "ok",
            "adapter_output_track",
            f"adapter '{self.preferred_adapter}' output track matches {self.anchor_ir_track.value}",
        )

    def _infer_adapter_output_track(self, adapter: Any):
        from .anchor_ir import AnchorIRTrack

        output_track = getattr(adapter, "get_output_track", None)
        if callable(output_track):
            track = output_track()
            if isinstance(track, AnchorIRTrack):
                return track
            if isinstance(track, str):
                return AnchorIRTrack(track)

        get_output_dialects = getattr(adapter, "get_output_dialects", None)
        if not callable(get_output_dialects):
            return None

        dialects = set(get_output_dialects() or [])
        linalg_dialects = {"linalg", "linalg_ext", "tensor", "memref"}
        gpu_dialects = {"triton_gpu", "ttg", "gpu", "nvgpu"}
        has_linalg = bool(dialects & linalg_dialects)
        has_gpu = bool(dialects & gpu_dialects)

        if has_linalg and not has_gpu:
            return AnchorIRTrack.LINALG
        if has_gpu and not has_linalg:
            return AnchorIRTrack.TRITON_GPU
        return None

    @staticmethod
    def _check_positive_int(add, name: str, value: int) -> None:
        if type(value) is not int or value <= 0:
            add("error", name, f"expected a positive integer, got {value!r}")
            return
        add("ok", name, f"{value}")

    @staticmethod
    def _check_optional_positive_int(add, name: str, value: Optional[int]) -> None:
        if value is None:
            add("ok", name, "<auto>")
            return
        HWCapability._check_positive_int(add, name, value)

    @staticmethod
    def _check_non_negative_int(add, name: str, value: int) -> None:
        if type(value) is not int or value < 0:
            add("error", name, f"expected a non-negative integer, got {value!r}")
            return
        add("ok", name, f"{value}")

    @staticmethod
    def _check_optional_non_empty_string(add, name: str, value: Optional[str]) -> None:
        if value is None:
            add("ok", name, "<auto>")
            return
        if not isinstance(value, str) or not value:
            add("error", name, f"expected a non-empty string or None, got {value!r}")
            return
        add("ok", name, value)

    @staticmethod
    def _check_positive_tuple(add, name: str, value: Tuple[int, ...], expected_len: Optional[int] = None) -> None:
        if not isinstance(value, tuple):
            add("error", name, f"expected a tuple of positive integers, got {value!r}")
            return
        if expected_len is not None and len(value) != expected_len:
            add("error", name, f"expected {expected_len} elements, got {value!r}")
            return
        if not value:
            add("error", name, "expected at least one element")
            return
        bad_values = [dim for dim in value if type(dim) is not int or dim <= 0]
        if bad_values:
            add("error", name, f"all elements must be positive integers, got {value!r}")
            return
        add("ok", name, f"{value}")

    @staticmethod
    def _check_supported_dtypes(add, name: str, value: Set[str]) -> None:
        if not isinstance(value, set) or not value:
            add("error", name, f"expected a non-empty set of dtype names, got {value!r}")
            return
        bad_values = [dtype for dtype in value if not isinstance(dtype, str) or not dtype]
        if bad_values:
            add("error", name, f"dtype names must be non-empty strings, got {value!r}")
            return
        add("ok", name, f"{sorted(value)}")

    @staticmethod
    def _format_diagnostics(diagnostics: List[_Diagnostic]) -> str:
        lines = ["HWCapability validation failed:"]
        for diagnostic in diagnostics:
            lines.append(f"  - {diagnostic.check}: {diagnostic.message}")
        return "\n".join(lines)
