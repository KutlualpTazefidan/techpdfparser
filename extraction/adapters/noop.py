"""No-op adapters. Return empty content; used to skip roles in a config."""
from __future__ import annotations

from typing import Any

from PIL.Image import Image

from ..models import ElementContent
from ..registry import (
    register_figure_descriptor,
    register_formula_extractor,
    register_table_extractor,
    register_text_extractor,
)


@register_text_extractor("noop")
@register_table_extractor("noop")
@register_formula_extractor("noop")
class NoopExtractor:
    """Extractor that always returns empty ElementContent.

    Stacked decorators register the same class under three roles.
    The registry holds classes, not instances, so get_text_extractor('noop'),
    get_table_extractor('noop'), and get_formula_extractor('noop') each
    construct a separate instance of this class.
    """

    def __init__(self, **_kwargs: Any) -> None:
        """Accept arbitrary kwargs from YAML adapter config; ignore all."""
        pass

    @property
    def tool_name(self) -> str:
        return "noop"

    def extract(self, image: Image, page_number: int) -> ElementContent:
        return ElementContent()


@register_figure_descriptor("noop")
class NoopDescriber:
    """Figure descriptor that always returns an empty string."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    @property
    def tool_name(self) -> str:
        return "noop"

    def describe(self, image: Image) -> str:
        return ""
