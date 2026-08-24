"""Authoritative public versions for triton-anchor.

Keep this module dependency-free: ``setup.py`` reads it without importing the
``triton_anchor`` package.
"""

CORE_VERSION = "0.2.0"
BACKEND_PLUGIN_PROTOCOL_VERSION = "1.0"
BACKEND_MANIFEST_SCHEMA_VERSION = "1.0"

__version__ = CORE_VERSION
