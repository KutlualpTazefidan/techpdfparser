"""Adapter modules for extraction roles.

Importing a module here triggers its decorator-based registration.
GPU-dependent modules are wrapped in try/except ImportError so CPU-only
installs do not break at CLI startup — the adapter simply does not
register, and the registry's `get_*` functions will raise a clear
KeyError listing the available (CPU) adapters instead.
"""
from __future__ import annotations

from . import noop, pymupdf_renderer, pymupdf_text_segmenter  # noqa: F401

try:
    from . import mineru25  # noqa: F401
except ImportError:
    # mineru25.py is built to keep mineru imports lazy (inside methods),
    # so this try/except is defensive — it only triggers if a future edit
    # accidentally adds a top-level `import mineru` or `import torch`
    # to that module.
    pass
