"""Adapter modules for extraction roles.

Importing a module here triggers its decorator-based registration.
GPU-dependent modules (added in later stages) are wrapped in
try/except ImportError so CPU-only installs do not break.
"""
from __future__ import annotations

from . import (
    noop,  # noqa: F401
    pymupdf_renderer,  # noqa: F401
)
