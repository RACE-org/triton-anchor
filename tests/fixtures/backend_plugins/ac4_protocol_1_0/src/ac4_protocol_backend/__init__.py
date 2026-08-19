import os
from pathlib import Path


class Compiler:
    pass


class Driver:
    pass


compiler_cls = Compiler
driver_cls = Driver


def _mark():
    marker_dir = os.environ.get("TRITON_ANCHOR_AC4_MARKER_DIR")
    if marker_dir:
        Path(marker_dir).mkdir(parents=True, exist_ok=True)
        (Path(marker_dir) / "diagnostics.called").write_text("called\n", encoding="utf-8")


def diagnostics():
    _mark()
    return {"healthy": True}
