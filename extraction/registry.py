"""Adapter registry for config-driven tool switching.

Same pattern as src/core/registry.py — decorators register adapters
by name, get_* functions instantiate them.
"""
from __future__ import annotations

from typing import Any, Callable

AdapterRegistry = dict[str, type[Any]]
RegisterDecorator = Callable[[type[Any]], type[Any]]


def _make_register(registry: AdapterRegistry) -> Callable[[str], RegisterDecorator]:
    def register(name: str) -> RegisterDecorator:
        def decorator(cls: type[Any]) -> type[Any]:
            registry[name] = cls
            return cls
        return decorator
    return register


def _make_get(registry: AdapterRegistry, category: str) -> Callable[..., Any]:
    def get(name: str, **kwargs: Any) -> Any:
        if name not in registry:
            available = sorted(registry.keys())
            raise KeyError(
                f"Unknown {category} adapter '{name}'. Available: {available}"
            )
        return registry[name](**kwargs)
    return get


_RENDERERS: AdapterRegistry = {}
_SEGMENTERS: AdapterRegistry = {}
_TEXT_EXTRACTORS: AdapterRegistry = {}
_TABLE_EXTRACTORS: AdapterRegistry = {}
_FORMULA_EXTRACTORS: AdapterRegistry = {}
_FIGURE_DESCRIPTORS: AdapterRegistry = {}

register_renderer = _make_register(_RENDERERS)
register_segmenter = _make_register(_SEGMENTERS)
register_text_extractor = _make_register(_TEXT_EXTRACTORS)
register_table_extractor = _make_register(_TABLE_EXTRACTORS)
register_formula_extractor = _make_register(_FORMULA_EXTRACTORS)
register_figure_descriptor = _make_register(_FIGURE_DESCRIPTORS)

get_renderer = _make_get(_RENDERERS, "renderer")
get_segmenter = _make_get(_SEGMENTERS, "segmenter")
get_text_extractor = _make_get(_TEXT_EXTRACTORS, "text_extractor")
get_table_extractor = _make_get(_TABLE_EXTRACTORS, "table_extractor")
get_formula_extractor = _make_get(_FORMULA_EXTRACTORS, "formula_extractor")
get_figure_descriptor = _make_get(_FIGURE_DESCRIPTORS, "figure_descriptor")
