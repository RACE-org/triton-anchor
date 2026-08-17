"""Backend that must be rejected before its protocol-mismatched import."""

import os
from pathlib import Path


marker_dir = os.environ.get("TRITON_ANCHOR_W11_FULL_MARKER_DIR")
if marker_dir:
    marker = Path(marker_dir) / "bad_protocol.imported"
    marker.write_text("imported\n", encoding="utf-8")
raise RuntimeError("bad_protocol reached EntryPoint.load unexpectedly")
