"""Compatible native fixture; importing this module never opens its ELF."""

import os
from pathlib import Path


TARGET = "w10_native_compatible"


class W10NativeCompatibleCompiler:

    @classmethod
    def supports_target(cls, target):
        return getattr(target, "backend", None) == TARGET

    def __init__(self, target):
        self.target = target


class W10NativeCompatibleDriver:

    @classmethod
    def is_active(cls):
        return False


compiler_cls = W10NativeCompatibleCompiler
driver_cls = W10NativeCompatibleDriver

marker_dir = os.environ.get("TRITON_ANCHOR_W10_NATIVE_MARKER_DIR")
if marker_dir:
    marker = Path(marker_dir) / "compatible.imported"
    marker.write_text("imported\n", encoding="utf-8")
