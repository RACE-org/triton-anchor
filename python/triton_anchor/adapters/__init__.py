"""Adapter package — TTIR → Linalg/TritonGPU conversion adapters."""

from .base import (
    ITritonToLinalgAdapter as ITritonToLinalgAdapter,
    ILinalgOptAdapter as ILinalgOptAdapter,
    ILinalgPybindAdapter as ILinalgPybindAdapter,
    AdapterConversionError as AdapterConversionError,
)
from .registry import AdapterRegistry as AdapterRegistry, get_adapter as get_adapter
from .router import AdapterRouter as AdapterRouter, AdapterRoute as AdapterRoute

AdapterRegistry.register_builtins()
