"""Adapter package — TTIR → Linalg/TritonGPU conversion adapters."""

from .base import (
    ITritonToLinalgAdapter as ITritonToLinalgAdapter,
    ILinalgOptAdapter as ILinalgOptAdapter,
    ILinalgPybindAdapter as ILinalgPybindAdapter,
    AdapterConversionError as AdapterConversionError,
)
from .registry import AdapterRegistry as AdapterRegistry, get_adapter as get_adapter

try:
    from .triton_shared_adapter import TritonSharedAdapter

    AdapterRegistry.register(TritonSharedAdapter())
except ImportError:
    pass
