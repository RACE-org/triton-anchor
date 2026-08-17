"""Valid backend used by the W11 full-profile real-wheel probe."""

import os
from pathlib import Path

from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.driver import DriverBase


TARGET = "w11_full_mock"


def _mark_import():
    marker_dir = os.environ.get("TRITON_ANCHOR_W11_FULL_MARKER_DIR")
    if marker_dir:
        marker = Path(marker_dir) / "full_compatible.imported"
        marker.write_text("imported\n", encoding="utf-8")


_mark_import()


class W11FullCompatibleCompiler(BaseBackend):

    @classmethod
    def supports_target(cls, target):
        return getattr(target, "backend", None) == TARGET

    def hash(self):
        return "w11-full-compatible"

    def parse_options(self, options):
        return dict(options)

    def add_stages(self, stages, options):
        return None

    def load_dialects(self, context):
        return None


class W11FullCompatibleDriver(DriverBase):

    @classmethod
    def is_active(cls):
        return True

    def get_current_target(self):
        return GPUTarget(TARGET, 1, 1)


compiler_cls = W11FullCompatibleCompiler
driver_cls = W11FullCompatibleDriver
