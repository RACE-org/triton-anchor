"""
TritonSharedAdapter — Stub for triton-shared (Microsoft)
=========================================================

Placeholder for the triton-shared conversion path that uses
Structured / Unstructured dual-path pointer analysis.

Used by: spine-triton (SpacemiT RISC-V)
Source:  https://github.com/microsoft/triton-shared

This is a **stub** — full implementation requires:
  1. triton-shared-opt binary installed and on PATH
  2. triton-shared Python bindings (or subprocess invocation)

When implemented, this adapter will:
  - Use out-of-process ``triton-shared-opt`` tool for conversion
  - Support both Structured and Unstructured pointer analysis modes
  - Produce AnchorIR-compliant output (memref-based)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from .base import ILinalgOptAdapter, AdapterConversionError

logger = logging.getLogger(__name__)


class TritonSharedAdapter(ILinalgOptAdapter):
    """Out-of-process adapter using triton-shared (Structured pointer analysis).

    This adapter invokes the ``triton-shared-opt`` external tool to convert
    TTIR to Linalg IR.

    Status: **STUB** — raises NotImplementedError until triton-shared is integrated.

    Future implementation will support two modes:
      - Structured:   ``--triton-to-structured`` + ``--triton-to-linalg``
      - Unstructured:  ``--triton-to-linalg-experimental``
    """

    def __init__(self, opt_path: Optional[str] = None, mode: str = "structured"):
        """Initialize the adapter.

        Args:
            opt_path: Path to ``triton-shared-opt`` binary.
                Defaults to env var ``TRITON_SHARED_OPT_PATH`` or PATH lookup.
            mode: Pointer analysis mode, one of "structured" or "unstructured".
        """
        self._opt_path = opt_path
        self._mode = mode

    def name(self) -> str:
        return "triton-shared"

    def _find_opt_tool(self) -> str:
        """Locate the triton-shared-opt binary."""
        # 1. Explicit path
        if self._opt_path:
            return self._opt_path
        # 2. Environment variable
        env_path = os.environ.get("TRITON_SHARED_OPT_PATH")
        if env_path and os.path.isfile(env_path):
            return env_path
        # 3. PATH lookup
        which = shutil.which("triton-shared-opt")
        if which:
            return which
        return ""

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        """Convert TTIR to Linalg using triton-shared-opt.

        .. note:: This is currently a STUB.  Full implementation pending
           triton-shared integration.

        Raises:
            AdapterConversionError: Always, until triton-shared is integrated.
        """
        opt_path = self._find_opt_tool()
        if not opt_path:
            raise AdapterConversionError(
                self.name(),
                detail=(
                    "triton-shared-opt not found. "
                    "Install triton-shared and set TRITON_SHARED_OPT_PATH, "
                    "or ensure triton-shared-opt is on PATH.\n"
                    "  pip install triton-shared  (when available)\n"
                    "  export TRITON_SHARED_OPT_PATH=/path/to/triton-shared-opt"
                )
            )

        # ── Out-of-process conversion ────────────────────────────────
        ttir_text = str(ttir_module) if not isinstance(ttir_module, str) else ttir_module

        flags = self._get_pipeline_flags()

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "tt.mlir"
            dst = Path(tmpdir) / "linalg.mlir"
            src.write_text(ttir_text)

            cmd = [opt_path, str(src), *flags, "-o", str(dst)]
            logger.info(f"Running: {' '.join(cmd)}")

            try:
                subprocess.check_call(cmd, timeout=60)
            except subprocess.CalledProcessError as e:
                raise AdapterConversionError(
                    self.name(),
                    kernel_name=metadata.get("name", ""),
                    detail=f"triton-shared-opt failed with exit code {e.returncode}"
                )
            except FileNotFoundError:
                raise AdapterConversionError(
                    self.name(),
                    detail=f"triton-shared-opt not found at: {opt_path}"
                )

            return dst.read_text()

    def _get_pipeline_flags(self) -> List[str]:
        """Get the command-line flags for triton-shared-opt."""
        if self._mode == "structured":
            return [
                "--triton-to-structured",
                "--triton-to-linalg",
            ]
        elif self._mode == "unstructured":
            return [
                "--triton-to-linalg-experimental",
            ]
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def get_required_passes(self) -> List[str]:
        if self._mode == "structured":
            return ["triton-to-structured", "triton-to-linalg"]
        return ["triton-to-linalg-experimental"]

    def get_output_dialects(self) -> List[str]:
        return [
            "linalg", "tensor", "memref", "arith", "math",
            "scf", "func", "tptr", "triton_structured",
        ]
