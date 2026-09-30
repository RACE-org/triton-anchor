"""Adapter package — TTIR → Linalg/TritonGPU conversion adapters."""

from .base import (
    IAnchorAdapter as IAnchorAdapter,
    ITritonToLinalgAdapter as ITritonToLinalgAdapter,
    ILinalgOptAdapter as ILinalgOptAdapter,
    ILinalgPybindAdapter as ILinalgPybindAdapter,
    AdapterConversionError as AdapterConversionError,
    AdapterConversionContext as AdapterConversionContext,
    AdapterInfo as AdapterInfo,
)
from .registry import AdapterRegistry as AdapterRegistry, get_adapter as get_adapter
from .router import (
    AdapterDecision as AdapterDecision,
    AdapterRejection as AdapterRejection,
    AdapterRoute as AdapterRoute,
    AdapterRouter as AdapterRouter,
    AdapterRouterConfig as AdapterRouterConfig,
    AdapterRoutingError as AdapterRoutingError,
    OpCoverage as OpCoverage,
)

# Register built-in adapters.  Imports are intentionally lightweight; C++
# bindings are loaded only when convert() runs.
for _adapter_import in (
    (".triton_linalg_adapter", "TritonLinalgAdapter"),
    (".triton_shared_adapter", "TritonSharedAdapter"),
    (".triton_gpu_adapter", "TritonGPUAdapter"),
    (".hybrid_adapter", "HybridAdapter"),
):
    try:
        from importlib import import_module

        _module = import_module(_adapter_import[0], package=__name__)
        _adapter_cls = getattr(_module, _adapter_import[1])
        AdapterRegistry.register(_adapter_cls())
    except ImportError:
        pass
