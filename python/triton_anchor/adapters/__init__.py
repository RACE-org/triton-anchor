"""Adapter package — TTIR → Linalg/TritonGPU conversion adapters."""

from .base import (
    ITritonToLinalgAdapter,
    ILinalgOptAdapter,
    ILinalgPybindAdapter,
    AdapterConversionError,
)
from .registry import AdapterRegistry, get_adapter

__all__ = [
    "ITritonToLinalgAdapter",
    "ILinalgOptAdapter",
    "ILinalgPybindAdapter",
    "AdapterConversionError",
    "AdapterRegistry",
    "get_adapter",
]

# 注册默认自带的 in-process adapter
try:
    from .triton_linalg_adapter import TritonLinalgAdapter

    AdapterRegistry.register(TritonLinalgAdapter())
except ImportError:
    pass
