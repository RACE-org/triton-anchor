"""
DSL Extension Namespace — Reserved
====================================

This package provides the ``triton.language.ext`` namespace where
DSL extensions are auto-discovered and made available to users::

    from triton.language.ext import smt
    from triton.language.ext import tpu

Currently contains only stubs.  Actual extensions are loaded
dynamically from ``entry_points("triton.dsl_extensions")``.
"""

# Lazy-loading proxy for DSL extensions
# When a user does `from triton_anchor.language.ext import smt`,
# we look up the registered extension and return its builtins.

# import importlib
import sys
from types import ModuleType


class _ExtensionProxy(ModuleType):
    """Lazy-loading module proxy for DSL extensions.

    Intercepts attribute access to look up registered extensions.
    """

    def __getattr__(self, name):
        from ..extensions.registry import DSLExtensionRegistry

        ext = DSLExtensionRegistry.get_extension(name)
        if ext is not None:
            # Create a proxy module for this extension's builtins
            proxy = ModuleType(f"triton_anchor.language.ext.{name}")
            proxy.__doc__ = f"DSL extension: {ext.name} (namespace: {ext.namespace})"
            for builtin_name, spec in ext.get_builtins().items():
                setattr(proxy, builtin_name, spec)
            return proxy

        raise AttributeError(
            f"DSL extension '{name}' not found. "
            f"Install it: pip install triton-ext-{name}"
        )


# Replace this module with the proxy
sys.modules[__name__] = _ExtensionProxy(__name__)
