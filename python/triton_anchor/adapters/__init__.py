"""Adapter package — TTIR → Linalg/TritonGPU conversion adapters."""

from .base import (
    ITritonToLinalgAdapter as ITritonToLinalgAdapter,
    ILinalgOptAdapter as ILinalgOptAdapter,
    ILinalgPybindAdapter as ILinalgPybindAdapter,
    AdapterConversionError as AdapterConversionError,
    AdapterSelectionError as AdapterSelectionError,
    AdapterNotFoundError as AdapterNotFoundError,
)
from .registry import AdapterRegistry as AdapterRegistry, get_adapter as get_adapter
from .router import (
    ADAPTER_ROUTING_POLICY_VERSION as ADAPTER_ROUTING_POLICY_VERSION,
    AdapterDecision as AdapterDecision,
    AdapterRouter as AdapterRouter,
    AdapterRoutingPolicy as AdapterRoutingPolicy,
)

def register_builtin_adapters() -> None:
    """Register built-in adapters shipped with triton-anchor."""
    from .hybrid_adapter import HybridAdapter
    from .triton_gpu_adapter import TritonGPUAdapter
    from .triton_linalg_adapter import TritonLinalgAdapter
    from .triton_shared_adapter import TritonSharedAdapter

    for adapter in (
        TritonLinalgAdapter(),
        TritonSharedAdapter(),
        HybridAdapter(),
        TritonGPUAdapter(),
    ):
        AdapterRegistry.register(adapter)


register_builtin_adapters()
