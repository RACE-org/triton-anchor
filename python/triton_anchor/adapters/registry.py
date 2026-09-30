"""
Adapter Registry
=================

Manages registration and discovery of TTIR -> AnchorIR adapters.
Final adapter selection is owned by ``AdapterRouter``.

Discovery order:
  1. Explicit registration via ``AdapterRegistry.register()``
  2. ``entry_points("triton.adapters")`` discovery (pip-installed adapters)
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Dict, Optional, TYPE_CHECKING

from .base import AdapterNotFoundError, ITritonToLinalgAdapter

if TYPE_CHECKING:
    from ..hw_capability import HWCapability

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for TTIR -> AnchorIR conversion adapters.

    Usage::

        # Registration
        AdapterRegistry.register(TritonLinalgAdapter())

        # Auto-discovery from entry_points
        AdapterRegistry.discover()

        # Compatibility selection entry point, delegated to AdapterRouter
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
        logger.debug(f"Registered adapter: {name}")

    @classmethod
    def discover(cls) -> None:
        """Auto-discover adapters from ``entry_points("triton.adapters")``."""
        if cls._discovered:
            return
        cls._discovered = True

        try:
            eps = importlib.metadata.entry_points(group="triton.adapters")
        except TypeError:
            # Python 3.8/3.9 compatibility
            eps = importlib.metadata.entry_points().get("triton.adapters", [])

        for ep in eps:
            try:
                adapter_cls = ep.load()
                adapter = adapter_cls()
                cls.register(adapter)
                logger.info(f"Discovered adapter from entry_point: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load adapter entry_point '{ep.name}': {e}")

    @classmethod
    def get(cls, name: str) -> Optional[ITritonToLinalgAdapter]:
        """Get a specific adapter by name."""
        cls.discover()
        return cls._adapters.get(name)

    @classmethod
    def get_adapter(
        cls, hw: HWCapability, metadata: Optional[dict] = None
    ) -> ITritonToLinalgAdapter:
        """Compatibility wrapper: delegate final selection to AdapterRouter."""
        from .router import AdapterRouter

        return AdapterRouter(registry=cls).get_adapter(hw, metadata=metadata)

    @classmethod
    def list_adapters(cls) -> Dict[str, str]:
        """List all registered adapters: {name: class_name}."""
        cls.discover()
        return {name: type(adapter).__name__ for name, adapter in cls._adapters.items()}

    @classmethod
    def snapshot(cls) -> Dict[str, ITritonToLinalgAdapter]:
        """Return discovered adapters without making a route decision."""
        cls.discover()
        return dict(cls._adapters)

    @classmethod
    def reset(cls) -> None:
        """Reset registry state (for testing)."""
        cls._adapters.clear()
        cls._discovered = False


# ── Convenience function ─────────────────────────────────────────────


def get_adapter(
    hw: HWCapability, metadata: Optional[dict] = None
) -> ITritonToLinalgAdapter:
    """Shortcut for deterministic adapter routing through the registry."""
    return AdapterRegistry.get_adapter(hw, metadata=metadata)
