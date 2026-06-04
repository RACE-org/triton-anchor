"""Extensions sub-package — Layer 0: DSL Extensions."""

from .base import DSLExtensionPlugin, BuiltinSpec
from .registry import DSLExtensionRegistry

__all__ = [
    "DSLExtensionPlugin",
    "BuiltinSpec",
    "DSLExtensionRegistry",
]
