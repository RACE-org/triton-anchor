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

# Lazy attribute loading for DSL extensions.  Keep this module's native package
# identity so Python can still import the physical ``language.ext`` package.
from types import ModuleType


def __getattr__(name: str) -> ModuleType:
    """Look up a registered DSL extension without replacing this package."""
    from ..extensions.registry import DSLExtensionRegistry

    extension = DSLExtensionRegistry.get_extension(name)
    if extension is not None:
        proxy = ModuleType(f"triton_anchor.language.ext.{name}")
        proxy.__doc__ = (
            f"DSL extension: {extension.name} (namespace: {extension.namespace})"
        )
        for builtin_name, specification in extension.get_builtins().items():
            setattr(proxy, builtin_name, specification)
        return proxy

    raise AttributeError(
        f"DSL extension '{name}' not found. Install it: pip install triton-ext-{name}"
    )
