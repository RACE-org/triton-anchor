"""Minimal Gluon symbols required by Triton's shared AST generator."""


def _unsupported(*args, **kwargs):
    raise RuntimeError(
        "Gluon is not available in the hardware-independent triton-anchor Wheel"
    )


static_assert = _unsupported
static_print = _unsupported

__all__ = ["static_assert", "static_print"]
