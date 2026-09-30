"""Adapter registration and discovery.

The registry deliberately does not own routing policy.  T6.1 adapter choice is
handled by ``AdapterRouter`` so fallback and rejection reasons are explicit.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Dict, Optional, TYPE_CHECKING

from .base import IAnchorAdapter

if TYPE_CHECKING:
    from ..hw_capability import HWCapability

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for TTIR → AnchorIR conversion adapters.

    Usage::

        # Registration
        AdapterRegistry.register(TritonLinalgAdapter())

        # Auto-discovery from entry_points
        AdapterRegistry.discover()
    """

    _adapters: Dict[str, IAnchorAdapter] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, adapter: IAnchorAdapter) -> None:
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
    def get(cls, name: str) -> Optional[IAnchorAdapter]:
        """Get a specific adapter by name."""
        cls.discover()
        return cls._adapters.get(name)

    @classmethod
    def items(cls) -> Dict[str, IAnchorAdapter]:
        """Return discovered adapters as a deterministic name -> adapter map."""
        cls.discover()
        return dict(sorted(cls._adapters.items()))

    @classmethod
    def get_adapter(cls, hw: "HWCapability") -> IAnchorAdapter:
        """Compatibility shortcut that delegates selection to ``AdapterRouter``.

        New code should call ``AdapterRouter.select`` or ``AdapterRouter.route``
        directly and pass explicit backend capabilities, user config, and op
        coverage.  This legacy shortcut infers only the basic AnchorIR track
        capability and still performs no registry-owned fallback.
        """
        from ..anchor_ir import AnchorIRTrack
        from .router import AdapterRouter, AdapterRoutingError

        backend_capability = (
            "anchor_ir.triton_gpu"
            if hw.anchor_ir_track == AnchorIRTrack.TRITON_GPU
            else "anchor_ir.linalg"
        )
        try:
            return AdapterRouter(registry=cls).route(
                hw,
                backend_capabilities=(backend_capability,),
            ).adapter
        except AdapterRoutingError as exc:
            raise AdapterNotFoundError(str(exc)) from exc

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


def get_adapter(hw: "HWCapability") -> IAnchorAdapter:
    """Shortcut for ``AdapterRegistry.get_adapter(hw)``."""
    return AdapterRegistry.get_adapter(hw)
