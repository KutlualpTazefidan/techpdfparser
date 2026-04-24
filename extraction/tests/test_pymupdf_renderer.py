"""Tests for the pymupdf page renderer."""
from __future__ import annotations

from pathlib import Path

from PIL.Image import Image

from extraction.registry import get_renderer


def test_pymupdf_renderer_tool_name() -> None:
    assert get_renderer("pymupdf").tool_name == "pymupdf"


def test_pymupdf_renderer_page_count_positive(sample_text_pdf: Path) -> None:
    assert get_renderer("pymupdf").page_count(sample_text_pdf) >= 1


def test_pymupdf_renderer_render_page_returns_image(sample_text_pdf: Path) -> None:
    img = get_renderer("pymupdf").render_page(sample_text_pdf, 0)
    assert isinstance(img, Image)
    assert img.size[0] > 0 and img.size[1] > 0


def test_pymupdf_renderer_render_all_matches_page_count(sample_text_pdf: Path) -> None:
    r = get_renderer("pymupdf")
    images = r.render_all(sample_text_pdf)
    count = r.page_count(sample_text_pdf)
    assert len(images) == count
    for img in images:
        assert isinstance(img, Image)


def test_pymupdf_renderer_higher_dpi_larger_image(sample_text_pdf: Path) -> None:
    low = get_renderer("pymupdf", dpi=72).render_page(sample_text_pdf, 0)
    high = get_renderer("pymupdf", dpi=300).render_page(sample_text_pdf, 0)
    assert high.size[0] > low.size[0]
    assert high.size[1] > low.size[1]
