"""Collision fixture B must be rejected before its entry point is imported."""

import os
from pathlib import Path


marker_dir = os.environ.get("TRITON_ANCHOR_W10_NATIVE_MARKER_DIR")
if marker_dir:
    marker = Path(marker_dir) / "collision_b.imported"
    marker.write_text("imported\n", encoding="utf-8")

raise RuntimeError("native collision fixture B reached EntryPoint.load()")
