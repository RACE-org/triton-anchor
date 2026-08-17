"""First independent native plugin in the AC2 coexistence proof."""

import os
from pathlib import Path

TARGET = "ac2_native_alpha"


class AC2NativeAlphaCompiler:

    @classmethod
    def supports_target(cls, target):
        return getattr(target, "backend", None) == TARGET

    def __init__(self, target):
        self.target = target


class AC2NativeAlphaDriver:

    @classmethod
    def is_active(cls):
        return False


compiler_cls = AC2NativeAlphaCompiler
driver_cls = AC2NativeAlphaDriver

marker_dir = os.environ.get("TRITON_ANCHOR_AC2_MARKER_DIR")
if marker_dir:
    with (Path(marker_dir) / "alpha.imported").open(
        "a", encoding="utf-8"
    ) as marker:
        marker.write("imported\n")
