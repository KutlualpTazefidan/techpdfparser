"""Tests for the noop adapter.

Provides empty-content implementations of the text, table, formula,
and figure roles so a config can selectively skip any of them.
"""
from __future__ import annotations

from PIL import Image as PILImage

from extraction.models import ElementContent
from extraction.registry import (
    get_figure_descriptor,
    get_formula_extractor,
    get_table_extractor,
    get_text_extractor,
)


def test_noop_text_extractor_tool_name() -> None:
    assert get_text_extractor("noop").tool_name == "noop"


def test_noop_text_extractor_returns_empty_content() -> None:
    adapter = get_text_extractor("noop")
    blank = PILImage.new("RGB", (10, 10))
    result = adapter.extract(blank, page_number=0)
    assert isinstance(result, ElementContent)
    assert not (result.text or "").strip()


def test_noop_table_extractor_tool_name() -> None:
    assert get_table_extractor("noop").tool_name == "noop"


def test_noop_table_extractor_returns_empty_content() -> None:
    adapter = get_table_extractor("noop")
    blank = PILImage.new("RGB", (10, 10))
    result = adapter.extract(blank, page_number=0)
    assert isinstance(result, ElementContent)
    assert not (result.markdown or "").strip()


def test_noop_formula_extractor_tool_name() -> None:
    assert get_formula_extractor("noop").tool_name == "noop"


def test_noop_formula_extractor_returns_empty_content() -> None:
    adapter = get_formula_extractor("noop")
    blank = PILImage.new("RGB", (10, 10))
    result = adapter.extract(blank, page_number=0)
    assert isinstance(result, ElementContent)
    assert not (result.latex or "").strip()


def test_noop_figure_descriptor_tool_name() -> None:
    assert get_figure_descriptor("noop").tool_name == "noop"


def test_noop_figure_descriptor_returns_empty_string() -> None:
    adapter = get_figure_descriptor("noop")
    blank = PILImage.new("RGB", (10, 10))
    assert adapter.describe(blank) == ""
