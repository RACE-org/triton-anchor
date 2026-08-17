"""Backend that must be rejected before its Triton-commit import."""

import os
from pathlib import Path


marker_dir = os.environ.get("TRITON_ANCHOR_W11_FULL_MARKER_DIR")
if marker_dir:
    marker = Path(marker_dir) / "bad_triton_commit.imported"
    marker.write_text("imported\n", encoding="utf-8")
raise RuntimeError("bad_triton_commit reached EntryPoint.load unexpectedly")
