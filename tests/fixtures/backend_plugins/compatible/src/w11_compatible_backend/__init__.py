"""Valid W11 backend used by the real-wheel Triton-version probe."""

import os
from pathlib import Path

from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.driver import DriverBase


TARGET = "w11_mock"


def _mark_import() -> None:
    marker_dir = os.environ.get("TRITON_ANCHOR_W11_MARKER_DIR")
    if marker_dir:
        marker = Path(marker_dir) / "compatible.imported"
        marker.write_text("imported\n", encoding="utf-8")


_mark_import()


class W11CompatibleCompiler(BaseBackend):

    @classmethod
    def supports_target(cls, target):
        return getattr(target, "backend", None) == TARGET

    def hash(self):
        return "w11-compatible"

    def parse_options(self, options):
        return dict(options)

    def add_stages(self, stages, options):
        return None

    def load_dialects(self, context):
        return None


class W11CompatibleDriver(DriverBase):

    @classmethod
    def is_active(cls):
        return True

    def get_current_target(self):
        return GPUTarget(TARGET, 1, 1)


compiler_cls = W11CompatibleCompiler
driver_cls = W11CompatibleDriver
