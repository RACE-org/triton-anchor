"""This module must never be imported because its ABI fingerprint is wrong."""

import os
from pathlib import Path


marker_dir = os.environ.get("TRITON_ANCHOR_W10_NATIVE_MARKER_DIR")
if marker_dir:
    marker = Path(marker_dir) / "bad_abi.imported"
    marker.write_text("imported\n", encoding="utf-8")

raise RuntimeError("bad-ABI native fixture reached EntryPoint.load()")
