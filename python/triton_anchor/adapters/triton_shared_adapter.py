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

Uses the embedded ``triton-shared-opt`` tool from the packaged triton-shared frontend
build to lower TTIR to Linalg IR out-of-process.
"""

from __future__ import annotations

import importlib.resources as resources
import logging
import os
import re
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

    def __init__(self, opt_path: Optional[str] = None, mode: str = "unstructured"):
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
        if self._opt_path and os.path.isfile(self._opt_path):
            return self._opt_path
        # 2. Environment variable
        env_path = os.environ.get("TRITON_SHARED_OPT_PATH")
        if env_path and os.path.isfile(env_path):
            return env_path
        # 3. PATH lookup
        try:
            packaged = resources.files("triton").joinpath("bin/triton-shared-opt")
            if packaged.is_file():
                return str(packaged)
        except Exception:
            pass
        which = shutil.which("triton-shared-opt")
        return which or ""

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
                    "triton-shared-opt not found. Set TRITON_SHARED_OPT_PATH or "
                    "use a triton-anchor wheel that packages the triton-shared frontend toolchain."
                ),
            )

        ttir_text = (
            str(ttir_module) if not isinstance(ttir_module, str) else ttir_module
        )
        ttir_text = self._ensure_target_attrs(ttir_text, metadata)
        flags = self._get_pipeline_flags()

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "tt.mlir"
            dst = Path(tmpdir) / "linalg.mlir"
            src.write_text(ttir_text)
            cmd = [opt_path, str(src), *flags, "-o", str(dst)]
            logger.info("Running: %s", " ".join(cmd))
            try:
                subprocess.check_call(cmd, timeout=60)
            except subprocess.CalledProcessError as e:
                raise AdapterConversionError(
                    self.name(),
                    kernel_name=metadata.get("name", ""),
                    detail=f"triton-shared-opt failed with exit code {e.returncode}",
                )
            except FileNotFoundError:
                raise AdapterConversionError(
                    self.name(), detail=f"triton-shared-opt not found at: {opt_path}"
                )
            return dst.read_text()

    def _ensure_target_attrs(self, ttir_text: str, metadata: dict) -> str:
        attrs = {
            "tt.num_threads": f"{int(self._resolve_num_threads(metadata))} : i32",
            "tt.arch_id": f'"{self._resolve_arch_id(metadata)}"',
            "tt.force_vector_interleave": f"{int(self._resolve_force_vector_interleave(metadata))} : i32",
        }
        if all(key in ttir_text for key in attrs):
            return ttir_text

        if "module attributes {" in ttir_text:
            match = re.search(
                r"module\s+attributes\s*\{([^}]*)\}\s*\{", ttir_text, re.S
            )
            if not match:
                return ttir_text
            attr_block = match.group(1).strip()
            entries = [attr_block] if attr_block else []
            for key, value in attrs.items():
                if key not in ttir_text:
                    entries.append(f"{key} = {value}")
            replacement = "module attributes {" + ", ".join(entries) + "} {"
            return ttir_text[: match.start()] + replacement + ttir_text[match.end() :]

        insertion = (
            "module attributes {"
            + ", ".join(f"{k} = {v}" for k, v in attrs.items())
            + "} {"
        )
        return re.sub(r"module\s*\{", insertion, ttir_text, count=1)

    def _resolve_arch_id(self, metadata: dict) -> str:
        hw = metadata.get("hw") or metadata.get("hw_capability")
        return (
            metadata.get("arch_id")
            or getattr(hw, "arch_id", None)
            or os.environ.get("TRITON_SHARED_ARCH_ID")
            or "0xF000"
        )

    def _resolve_num_threads(self, metadata: dict) -> int:
        hw = metadata.get("hw") or metadata.get("hw_capability")
        return int(
            metadata.get("num_threads")
            or getattr(hw, "num_threads", None)
            or getattr(hw, "num_cores", None)
            or os.environ.get("TRITON_SHARED_NUM_THREADS")
            or 32
        )

    def _resolve_force_vector_interleave(self, metadata: dict) -> int:
        hw = metadata.get("hw") or metadata.get("hw_capability")
        return int(
            metadata.get("force_vector_interleave")
            or getattr(hw, "force_vector_interleave", None)
            or os.environ.get("TRITON_SHARED_FORCE_VECTOR_INTERLEAVE")
            or 2
        )

    def _get_pipeline_flags(self) -> List[str]:
        if self._mode == "structured":
            return ["--triton-to-structured", "--triton-to-linalg"]
        if self._mode == "unstructured":
            return ["--triton-to-linalg-experimental"]
        raise ValueError(f"Unknown mode: {self._mode}")

    def get_required_passes(self) -> List[str]:
        return self._get_pipeline_flags()

    def get_output_dialects(self) -> List[str]:
        return [
            "linalg",
            "tensor",
            "memref",
            "arith",
            "math",
            "scf",
            "func",
            "xsmt",
            "xsmt_async",
            "tle",
        ]
