"""Second independent native plugin in the AC2 coexistence proof."""

import os
from pathlib import Path

TARGET = "ac2_native_beta"


class AC2NativeBetaCompiler:

    @classmethod
    def supports_target(cls, target):
        return getattr(target, "backend", None) == TARGET

    def __init__(self, target):
        self.target = target


class AC2NativeBetaDriver:

    @classmethod
    def is_active(cls):
        return False


compiler_cls = AC2NativeBetaCompiler
driver_cls = AC2NativeBetaDriver

marker_dir = os.environ.get("TRITON_ANCHOR_AC2_MARKER_DIR")
if marker_dir:
    with (Path(marker_dir) / "beta.imported").open(
        "a", encoding="utf-8"
    ) as marker:
        marker.write("imported\n")
