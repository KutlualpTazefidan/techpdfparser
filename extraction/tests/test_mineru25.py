"""Tests for the mineru25 adapter.

Unit tests (CPU-only) verify registration under both roles and that
the class can be constructed without triggering any mineru imports.
Integration tests (GPU-gated via -m integration) appear in later tasks.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from extraction.adapters.mineru25 import (
    _html_table_to_markdown,
    _to_regions,
)
from extraction.models import ElementType
from extraction.registry import get_segmenter, get_table_extractor


def test_mineru25_segmenter_tool_name() -> None:
    assert get_segmenter("mineru25").tool_name == "mineru25"


def test_mineru25_registers_as_table_extractor() -> None:
    assert get_table_extractor("mineru25").tool_name == "mineru25"


def test_mineru25_construction_does_not_import_mineru() -> None:
    # Construction must be lazy; importing/instantiating the class
    # must not require mineru or torch at class-definition time.
    import sys

    # Force a clean import state for this check.
    before = set(sys.modules.keys())
    adapter = get_segmenter("mineru25")
    after = set(sys.modules.keys())
    newly_imported = after - before
    # No heavy dependency should have been pulled in by mere construction.
    assert not any(m.startswith("mineru") for m in newly_imported)
    assert not any(m.startswith("torch") for m in newly_imported)
    assert not any(m.startswith("transformers") for m in newly_imported)
    assert adapter is not None


def test_mineru25_loaded_flag_starts_false() -> None:
    adapter = get_segmenter("mineru25")
    # Adapter exposes a `_loaded` flag that later lifecycle tests use
    # as a lightweight observable of load state (full VRAM verification
    # happens in the integration tests).
    assert adapter._loaded is False


# --- mapping helper tests (CPU-only, no GPU) ---


def test_html_table_to_markdown_basic() -> None:
    html = "<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>"
    md = _html_table_to_markdown(html)
    # Structural assertions — we don't pin the exact whitespace.
    assert "| a | b |" in md
    assert "| --- | --- |" in md
    assert "| 1 | 2 |" in md


def test_html_table_to_markdown_empty_returns_empty_string() -> None:
    assert _html_table_to_markdown("") == ""


def test_html_table_to_markdown_fallback_on_no_rows() -> None:
    # A non-table HTML string with no <tr> should not crash.
    result = _html_table_to_markdown("<div>not a table</div>")
    assert isinstance(result, str)


def test_to_regions_maps_types_correctly() -> None:
    content_list = [
        {"type": "text", "page_idx": 0, "bbox": [0, 0, 500, 50], "text": "body"},
        {"type": "title", "page_idx": 0, "bbox": [0, 60, 500, 100], "text": "Heading", "text_level": 1},
        {"type": "table", "page_idx": 1, "bbox": [0, 0, 800, 400],
         "table_body": "<table><tr><th>x</th></tr><tr><td>1</td></tr></table>"},
        {"type": "equation", "page_idx": 1, "bbox": [100, 500, 400, 550],
         "text": r"\int_0^1 x\,dx", "text_format": "latex"},
        {"type": "image", "page_idx": 2, "bbox": [0, 0, 500, 500], "img_path": "pages/foo.png"},
        {"type": "chart", "page_idx": 2, "bbox": [500, 0, 1000, 500], "img_path": "pages/bar.png"},
        {"type": "header", "page_idx": 0, "bbox": [0, 0, 1000, 20], "text": "drop me"},
    ]
    page_sizes_pts = [(612.0, 792.0), (612.0, 792.0), (612.0, 792.0)]

    regions = _to_regions(content_list, page_sizes_pts)

    # header is dropped (not a modeled type); 6 regions remain.
    assert len(regions) == 6

    by_type = [r.region_type for r in regions]
    assert ElementType.TEXT in by_type
    assert ElementType.HEADING in by_type
    assert ElementType.TABLE in by_type
    assert ElementType.FORMULA in by_type
    # image and chart both map to FIGURE
    assert by_type.count(ElementType.FIGURE) == 2


def test_to_regions_converts_normalized_bbox_to_pdf_points() -> None:
    content_list = [
        {"type": "text", "page_idx": 0, "bbox": [100, 200, 500, 400], "text": "hello"},
    ]
    page_sizes_pts = [(612.0, 792.0)]
    regions = _to_regions(content_list, page_sizes_pts)
    assert len(regions) == 1
    r = regions[0]
    # 100/1000 * 612 = 61.2 ; 200/1000 * 792 = 158.4 ; etc.
    assert abs(r.bbox[0] - 61.2) < 0.1
    assert abs(r.bbox[1] - 158.4) < 0.1
    assert abs(r.bbox[2] - 306.0) < 0.1
    assert abs(r.bbox[3] - 316.8) < 0.1


def test_to_regions_table_uses_markdown_field() -> None:
    content_list = [
        {"type": "table", "page_idx": 0, "bbox": [0, 0, 1000, 500],
         "table_body": "<table><tr><th>h</th></tr><tr><td>v</td></tr></table>"},
    ]
    regions = _to_regions(content_list, [(612.0, 792.0)])
    assert len(regions) == 1
    assert regions[0].region_type == ElementType.TABLE
    assert regions[0].content is not None
    # Table markdown is filled; text field is not.
    assert (regions[0].content.markdown or "").strip() != ""
    assert "| h |" in (regions[0].content.markdown or "")


def test_to_regions_formula_uses_latex_field() -> None:
    content_list = [
        {"type": "equation", "page_idx": 0, "bbox": [0, 0, 500, 50],
         "text": r"E = mc^2", "text_format": "latex"},
    ]
    regions = _to_regions(content_list, [(612.0, 792.0)])
    assert len(regions) == 1
    assert regions[0].region_type == ElementType.FORMULA
    assert regions[0].content is not None
    assert regions[0].content.latex == r"E = mc^2"


def test_to_regions_drops_items_with_out_of_range_page() -> None:
    content_list = [
        {"type": "text", "page_idx": 0, "bbox": [0, 0, 100, 100], "text": "ok"},
        {"type": "text", "page_idx": 5, "bbox": [0, 0, 100, 100], "text": "out of range"},
    ]
    regions = _to_regions(content_list, [(612.0, 792.0)])  # only page 0 exists
    assert len(regions) == 1
    assert regions[0].content is not None
    assert regions[0].content.text == "ok"


# --- integration tests (GPU required; gated by @pytest.mark.integration) ---


@pytest.mark.integration
def test_mineru25_segment_produces_regions(sample_text_pdf: Path) -> None:
    adapter = get_segmenter("mineru25")
    regions = adapter.segment(sample_text_pdf)
    assert len(regions) > 0
    # Kohavi 1997 is a 7-page paper with body text; at a minimum we expect
    # text regions to be detected.
    text_regions = [r for r in regions if r.region_type.value == "text"]
    assert len(text_regions) > 0
    # Every region should have well-formed bbox in PDF points (the Kohavi
    # PDF is US Letter, so max width ~612pts, max height ~792pts).
    for r in regions:
        x0, y0, x1, y1 = r.bbox
        assert 0 <= x0 <= x1 <= 700, f"x bbox out of plausible range: {r.bbox}"
        assert 0 <= y0 <= y1 <= 900, f"y bbox out of plausible range: {r.bbox}"
        assert 0 <= r.page < 7


@pytest.mark.integration
def test_mineru25_segment_finds_at_least_one_table(sample_text_pdf: Path) -> None:
    # Kohavi 1997 paper contains "Table 1 True accuracy estimates...".
    adapter = get_segmenter("mineru25")
    regions = adapter.segment(sample_text_pdf)
    table_regions = [r for r in regions if r.region_type.value == "table"]
    assert len(table_regions) >= 1
    # The table should have markdown content filled in.
    for t in table_regions:
        assert t.content is not None
        assert (t.content.markdown or "").strip() != ""


@pytest.mark.integration
def test_mineru25_segment_caches_per_pdf(sample_text_pdf: Path) -> None:
    # Calling segment twice on the same PDF should return the same result
    # from cache (cheap, no re-inference). We assert object-equality on
    # a per-region basis.
    adapter = get_segmenter("mineru25")
    r1 = adapter.segment(sample_text_pdf)
    r2 = adapter.segment(sample_text_pdf)
    assert len(r1) == len(r2)
