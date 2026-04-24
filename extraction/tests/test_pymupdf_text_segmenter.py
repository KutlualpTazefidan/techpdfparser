"""Tests for the pymupdf_text segmenter + text_extractor adapter.

pymupdf_text registers under two roles. When used as both the
segmenter AND text_extractor, the pipeline's tool_match optimization
reuses segmenter output directly — so extract() is normally not called.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage

from extraction.models import ElementType, Region
from extraction.registry import get_segmenter, get_text_extractor


def test_pymupdf_text_segmenter_tool_name() -> None:
    assert get_segmenter("pymupdf_text").tool_name == "pymupdf_text"


def test_pymupdf_text_registers_as_text_extractor() -> None:
    assert get_text_extractor("pymupdf_text").tool_name == "pymupdf_text"


def test_pymupdf_text_segmenter_returns_text_regions(sample_text_pdf: Path) -> None:
    regions = get_segmenter("pymupdf_text").segment(sample_text_pdf)
    assert len(regions) > 0
    for region in regions:
        assert isinstance(region, Region)
        assert region.region_type == ElementType.TEXT
        assert region.confidence == 1.0
        assert region.content is not None
        assert (region.content.text or "").strip() != ""


def test_pymupdf_text_segmenter_bbox_well_formed(sample_text_pdf: Path) -> None:
    regions = get_segmenter("pymupdf_text").segment(sample_text_pdf)
    for region in regions:
        x0, y0, x1, y1 = region.bbox
        assert x0 <= x1
        assert y0 <= y1


def test_pymupdf_text_segmenter_pages_within_doc(sample_text_pdf: Path) -> None:
    # Confirm page indices in regions are nonnegative and bounded.
    import pymupdf  # local import to avoid polluting module namespace

    regions = get_segmenter("pymupdf_text").segment(sample_text_pdf)
    doc = pymupdf.open(str(sample_text_pdf))
    try:
        total = doc.page_count
    finally:
        doc.close()
    for region in regions:
        assert 0 <= region.page < total


def test_pymupdf_text_extract_satisfies_protocol() -> None:
    # extract() is normally bypassed by tool_match; still needs to return
    # a valid ElementContent so the Protocol is satisfied.
    adapter = get_text_extractor("pymupdf_text")
    blank = PILImage.new("RGB", (10, 10))
    content = adapter.extract(blank, page_number=0)
    assert content is not None
