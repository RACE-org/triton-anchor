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
from typing import Dict, Optional, TYPE_CHECKING

from .base import ITritonToLinalgAdapter

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
    def get_adapter(cls, hw: HWCapability) -> ITritonToLinalgAdapter:
        """Select the best adapter for the given hardware capability.

        Selection logic:
          1. If ``hw.preferred_adapter`` is set, use that adapter
          2. Otherwise, select by ``hw.ptr_model``:
             - "structured" → TritonSharedAdapter
             - "axis_info"  → TritonLinalgAdapter
             - "hybrid"     → HybridAdapter
             - "gpu"        → None (GPU path doesn't use Linalg adapters)

        Args:
            hw: The target hardware capability.

        Returns:
            The selected adapter instance.

        Raises:
            AdapterNotFoundError: If no suitable adapter is found.
        """
        cls.discover()

        # 1. Explicit preference
        if hw.preferred_adapter:
            adapter = cls._adapters.get(hw.preferred_adapter)
            if adapter:
                return adapter
            raise AdapterNotFoundError(
                f"Preferred adapter '{hw.preferred_adapter}' not found. "
                f"Available: {list(cls._adapters.keys())}"
            )

        # 2. Automatic selection by ptr_model
        _model_to_adapter = {
            "structured": "triton-shared",
            "axis_info": "triton-linalg",
            "hybrid": "hybrid",
        }
        adapter_name = _model_to_adapter.get(hw.ptr_model)
        if adapter_name and adapter_name in cls._adapters:
            return cls._adapters[adapter_name]

        # 3. Fallback: return any available adapter
        if cls._adapters:
            fallback = next(iter(cls._adapters.values()))
            logger.warning(
                f"No adapter matching ptr_model='{hw.ptr_model}', "
                f"falling back to '{fallback.name()}'"
            )
            return fallback

        raise AdapterNotFoundError(
            f"No adapters available for ptr_model='{hw.ptr_model}'. "
            f"Install a Linalg adapter package."
        )

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


class AdapterNotFoundError(Exception):
    """Raised when no suitable adapter is found."""
    pass


# ── Convenience function ─────────────────────────────────────────────

def get_adapter(hw: HWCapability) -> ITritonToLinalgAdapter:
    """Shortcut for ``AdapterRegistry.get_adapter(hw)``."""
    return AdapterRegistry.get_adapter(hw)
