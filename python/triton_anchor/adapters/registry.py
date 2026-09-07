"""
Adapter Registry
=================

Manages discovery and selection of TTIR → Linalg adapters.
Selection is driven by ``HWCapability.ptr_model`` and optional user override.

Discovery order:
  1. Explicit registration via ``AdapterRegistry.register()``
  2. ``entry_points("triton.adapters")`` discovery (pip-installed adapters)

"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from .base import ITritonToLinalgAdapter
from .router import AdapterNotFoundError, AdapterRoute, AdapterRouter

if TYPE_CHECKING:
    from ..hw_capability import HWCapability

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for TTIR → Linalg conversion adapters.

    Usage::

        # Registration
        AdapterRegistry.register(TritonLinalgAdapter())

        # Auto-discovery from entry_points
        AdapterRegistry.discover()

        # Selection by hardware capability
        adapter = AdapterRegistry.get_adapter(hw_capability)
    """

    _adapters: Dict[str, ITritonToLinalgAdapter] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, adapter: ITritonToLinalgAdapter) -> None:
        """Explicitly register an adapter instance."""
        name = adapter.name()
        if name in cls._adapters:
            logger.warning(f"Adapter '{name}' already registered, overwriting")
        cls._adapters[name] = adapter

    @classmethod
    def register_builtins(cls) -> None:
        """Register triton-anchor's built-in adapters."""
        builtin_factories = (
            (
                "triton-shared",
                lambda: __import__(
                    "triton_anchor.adapters.triton_shared_adapter",
                    fromlist=["TritonSharedAdapter"],
                ).TritonSharedAdapter(mode="structured"),
            ),
            (
                "triton-linalg",
                lambda: __import__(
                    "triton_anchor.adapters.triton_linalg_adapter",
                    fromlist=["TritonLinalgAdapter"],
                ).TritonLinalgAdapter(),
            ),
            (
                "hybrid",
                lambda: __import__(
                    "triton_anchor.adapters.hybrid_adapter",
                    fromlist=["HybridAdapter"],
                ).HybridAdapter(),
            ),
            (
                "triton-gpu",
                lambda: __import__(
                    "triton_anchor.adapters.triton_gpu_adapter",
                    fromlist=["TritonGPUAdapter"],
                ).TritonGPUAdapter(),
            ),
        )
        for name, factory in builtin_factories:
            if name in cls._adapters:
                continue
            try:
                cls.register(factory())
            except ImportError as exc:
                logger.debug("Built-in adapter '%s' unavailable: %s", name, exc)

    @classmethod
    def discover(cls) -> None:
        """Auto-discover adapters from ``entry_points("triton.adapters")``."""
        if cls._discovered:
            return
        cls._discovered = True
        cls.register_builtins()
        try:
            eps = importlib.metadata.entry_points(group="triton.adapters")
        except TypeError:
            # Python 3.8/3.9 compatibility
            eps = importlib.metadata.entry_points().get("triton.adapters", [])
        for ep in eps:
            try:
                adapter_cls = ep.load()
                cls.register(adapter_cls())
            except Exception as e:
                logger.warning(f"Failed to load adapter entry_point '{ep.name}': {e}")

    @classmethod
    def get(cls, name: str) -> Optional[ITritonToLinalgAdapter]:
        """Get a specific adapter by name."""
        cls.discover()
        return cls._adapters.get(name)

    @classmethod
    def resolve(
        cls, hw: HWCapability, metadata: Optional[dict] = None
    ) -> Tuple[ITritonToLinalgAdapter, AdapterRoute]:
        """Resolve an adapter and record a deterministic Router decision.

        Selection is intentionally fail-closed: unsupported ``ptr_model``
        values or missing adapters raise ``AdapterNotFoundError`` instead of
        falling back to whichever adapter happened to be registered first.
        Hybrid conversion fallback is handled inside ``HybridAdapter``.
        """
        cls.discover()
        return AdapterRouter(cls._adapters).resolve(hw, metadata=metadata)

    @classmethod
    def get_route(cls, hw: HWCapability, metadata: Optional[dict] = None) -> AdapterRoute:
        """Return only the deterministic route metadata for ``hw``."""
        _, route = cls.resolve(hw, metadata=metadata)
        return route

    @classmethod
    def get_adapter(
        cls, hw: HWCapability, metadata: Optional[dict] = None
    ) -> ITritonToLinalgAdapter:
        """Select the best adapter for the given hardware capability.

        Selection logic:
          1. If ``hw.preferred_adapter`` is set, use that adapter
          2. Otherwise, select by ``hw.ptr_model``:
             - "structured" → TritonSharedAdapter
             - "axis_info"  → TritonLinalgAdapter
             - "hybrid"     → HybridAdapter
             - "gpu"        → TritonGPUAdapter

        Args:
            hw: The target hardware capability.

        Returns:
            The selected adapter instance.

        Raises:
            AdapterNotFoundError: If no suitable adapter is found.
        """
        adapter, _route = cls.resolve(hw, metadata=metadata)
        return adapter

    @classmethod
    def list_adapters(cls) -> Dict[str, str]:
        """List all registered adapters: {name: class_name}."""
        cls.discover()
        return {name: type(adapter).__name__ for name, adapter in cls._adapters.items()}

    @classmethod
    def reset(cls) -> None:
        """Reset registry state (for testing)."""
        cls._adapters.clear()
        cls._discovered = False


def get_adapter(
    hw: HWCapability, metadata: Optional[dict] = None
) -> ITritonToLinalgAdapter:
    """Shortcut for ``AdapterRegistry.get_adapter(hw)``."""
    return AdapterRegistry.get_adapter(hw, metadata=metadata)
