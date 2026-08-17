"""W11 backend that must be rejected before this module is imported."""

import os
from dataclasses import dataclass
from pathlib import Path


TARGET = "w11_mock"


@dataclass(frozen=True)
class W11Target:
    backend: str = TARGET
    arch: int = 1
    warp_size: int = 1


class W11IncompatibleCompiler:

    @classmethod
    def supports_target(cls, target):
        return getattr(target, "backend", None) == TARGET

    def __init__(self, target):
        self.target = target


class W11IncompatibleDriver:

    @classmethod
    def is_active(cls):
        return True

    def get_current_target(self):
        return W11Target()


compiler_cls = W11IncompatibleCompiler
driver_cls = W11IncompatibleDriver

marker_dir = os.environ.get("TRITON_ANCHOR_W11_MARKER_DIR")
if marker_dir:
    marker = Path(marker_dir) / "incompatible.imported"
    marker.write_text("imported\n", encoding="utf-8")
raise RuntimeError(
    "W11 incompatible backend reached EntryPoint.load unexpectedly"
)
