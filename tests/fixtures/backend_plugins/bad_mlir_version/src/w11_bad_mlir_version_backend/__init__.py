"""Backend that must be rejected before its MLIR-version import."""

import os
from pathlib import Path


marker_dir = os.environ.get("TRITON_ANCHOR_W11_FULL_MARKER_DIR")
if marker_dir:
    marker = Path(marker_dir) / "bad_mlir_version.imported"
    marker.write_text("imported\n", encoding="utf-8")
raise RuntimeError("bad_mlir_version reached EntryPoint.load unexpectedly")
