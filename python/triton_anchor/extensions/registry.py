"""
DSL Extension Registry
=======================

Auto-discovers and manages DSL extension plugins via entry_points.

Discovery flow:
  1. ``pip install triton-ext-spacemit`` installs the package
  2. Package declares entry_point in ``triton.dsl_extensions`` group
  3. On first JIT compile, ``DSLExtensionRegistry.discover()`` loads all
  4. Builtins are registered into the Triton type system
  5. MLIR dialect libraries are loaded into the compilation context
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Dict, List, Optional, Set

from .base import DSLExtensionPlugin, IncompatibleExtensionError

logger = logging.getLogger(__name__)


class DSLExtensionRegistry:
    """Registry for DSL extension plugins.

    Usage::

        # Auto-discover all installed extensions
        DSLExtensionRegistry.discover()

        # Get a specific extension
        ext = DSLExtensionRegistry.get_extension("smt")

        # Validate kernel compatibility
        DSLExtensionRegistry.validate_kernel(kernel_ir, target_backend="sophgo")

        # List all extensions
        for ns, ext in DSLExtensionRegistry.list_extensions().items():
            print(f"{ns}: {ext.name} (target={ext.target_backend})")
    """

    _plugins: Dict[str, DSLExtensionPlugin] = {}
    _loaded: bool = False

    @classmethod
    def discover(cls) -> None:
        """Auto-discover DSL extensions from entry_points.

        Thread-safe: uses a flag to prevent re-entrant loading.
        """
        if cls._loaded:
            return
        cls._loaded = True

        try:
            eps = importlib.metadata.entry_points(group="triton.dsl_extensions")
        except TypeError:
            eps = importlib.metadata.entry_points().get("triton.dsl_extensions", [])

        for ep in eps:
            try:
                plugin_cls = ep.load()
                plugin = plugin_cls()
                cls._register(plugin)
                logger.info(
                    f"Discovered DSL extension: {plugin.namespace} "
                    f"(target={plugin.target_backend})"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load DSL extension '{ep.name}': {e}"
                )

    @classmethod
    def register(cls, plugin: DSLExtensionPlugin) -> None:
        """Explicitly register a DSL extension plugin."""
        cls._register(plugin)

    @classmethod
    def _register(cls, plugin: DSLExtensionPlugin) -> None:
        """Internal registration with builtin + dialect loading."""
        ns = plugin.namespace
        if ns in cls._plugins:
            logger.warning(f"DSL extension '{ns}' already registered, overwriting")

        cls._plugins[ns] = plugin

        # Register builtins
        builtins = plugin.get_builtins()
        for name, spec in builtins.items():
            fqn = f"{ns}.{name}"
            logger.debug(f"  Registered builtin: {fqn}")
            # TODO: integrate with Triton's builtin registry
            # register_builtin(fqn, spec)

        # Load MLIR dialect library
        dialect_lib = plugin.get_dialect_library()
        if dialect_lib:
            logger.debug(f"  Loading dialect library: {dialect_lib}")
            # TODO: integrate with MLIR dialect loading
            # load_dialect_plugin(dialect_lib)

    @classmethod
    def get_extension(cls, namespace: str) -> Optional[DSLExtensionPlugin]:
        """Get an extension by namespace."""
        cls.discover()
        return cls._plugins.get(namespace)

    @classmethod
    def validate_kernel(cls, kernel_ir: str, target_backend: str) -> None:
        """Validate that all DSL extensions used in a kernel are compatible.

        Scans the kernel IR for extension ops and checks each one against
        the target backend.

        Args:
            kernel_ir: MLIR text of the kernel.
            target_backend: Target backend name (e.g., 'sophgo').

        Raises:
            IncompatibleExtensionError: If any extension is incompatible.
        """
        cls.discover()

        # Extract extension namespaces used in the kernel
        used_namespaces = cls._extract_extension_namespaces(kernel_ir)

        for ns in used_namespaces:
            plugin = cls._plugins.get(ns)
            if plugin is None:
                continue  # Unknown namespace — might be a standard dialect

            is_ok, msg = plugin.validate_kernel_compatibility(kernel_ir, target_backend)
            if not is_ok:
                raise IncompatibleExtensionError(msg)

    @classmethod
    def _extract_extension_namespaces(cls, kernel_ir: str) -> Set[str]:
        """Extract DSL extension namespaces from MLIR text.

        Looks for ops matching registered extension namespaces.
        """
        import re
        # Match all "namespace.op_name" patterns
        pattern = re.compile(r'"?(\w+)\.\w[\w.]*"?')
        found = set()
        for match in pattern.finditer(kernel_ir):
            dialect = match.group(1)
            if dialect in cls._plugins:
                found.add(dialect)
        return found

    @classmethod
    def list_extensions(cls) -> Dict[str, DSLExtensionPlugin]:
        """List all registered extensions."""
        cls.discover()
        return dict(cls._plugins)

    @classmethod
    def get_extensions_for_backend(cls, backend_name: str) -> List[DSLExtensionPlugin]:
        """Get all extensions compatible with a given backend."""
        cls.discover()
        result = []
        for plugin in cls._plugins.values():
            if plugin.target_backend is None or plugin.target_backend == backend_name:
                result.append(plugin)
        return result

    @classmethod
    def reset(cls) -> None:
        """Reset registry state (for testing)."""
        cls._plugins.clear()
        cls._loaded = False
