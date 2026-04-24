"""MinerU 2.5+ adapter.

Registers under both segmenter and table_extractor roles so the pipeline's
tool_match optimization reuses segmenter output for tables without a
second model invocation.

All mineru / torch imports are deferred into method bodies so this module
can be imported on CPU-only installs (where `mineru` is not present).
Registration decorators only need the class object, not the heavy deps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL.Image import Image

from ..models import ElementContent, ElementType, Region
from ..registry import register_segmenter, register_table_extractor

_MINERU_TO_ELEMENT_TYPE = {
    "text": "text",
    "title": "heading",
    "table": "table",
    "equation": "formula",
    "image": "figure",
    "chart": "figure",
    # Everything else (header, footer, page_number, aside_text, page_footnote,
    # algorithm, ref_text, phonetic, code, list, seal) is intentionally dropped
    # at the Region level — these types are not modeled in our schema.
}


def _html_table_to_markdown(html: str) -> str:
    """Convert a MinerU HTML <table> string into a GitHub-flavored markdown table.

    Keeps the logic small: extract rows of cells, emit pipe-delimited lines
    with a separator after the header. Falls back to the input string if
    no rows are present (non-table HTML).
    """
    if not html:
        return ""
    from bs4 import BeautifulSoup  # lazy; bs4 is in the [gpu] extra

    soup = BeautifulSoup(html, "html.parser")
    rows: list[list[str]] = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    if not rows:
        return html  # caller gets something useful back
    num_cols = max(len(r) for r in rows)
    # Pad short rows so every row has num_cols cells.
    rows = [r + [""] * (num_cols - len(r)) for r in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * num_cols) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, separator, *body])


def _to_regions(
    content_list: list[dict[str, Any]],
    page_sizes_pts: list[tuple[float, float]],
) -> list[Region]:
    """Convert a MinerU content_list.json payload to our Region list.

    page_sizes_pts: one (width_pts, height_pts) tuple per page index,
    in PDF points. Used to convert MinerU's 0-1000 normalized bboxes
    to the PDF-point coordinates the rest of the pipeline expects.
    """
    regions: list[Region] = []
    for item in content_list:
        mineru_type = str(item.get("type", ""))
        if mineru_type not in _MINERU_TO_ELEMENT_TYPE:
            continue
        page_idx = int(item.get("page_idx", -1))
        if page_idx < 0 or page_idx >= len(page_sizes_pts):
            continue
        raw_bbox = item.get("bbox")
        if not (isinstance(raw_bbox, list) and len(raw_bbox) == 4):
            continue

        w_pts, h_pts = page_sizes_pts[page_idx]
        x0 = float(raw_bbox[0]) / 1000.0 * w_pts
        y0 = float(raw_bbox[1]) / 1000.0 * h_pts
        x1 = float(raw_bbox[2]) / 1000.0 * w_pts
        y1 = float(raw_bbox[3]) / 1000.0 * h_pts

        element_type = ElementType(_MINERU_TO_ELEMENT_TYPE[mineru_type])
        content = _build_content_for(mineru_type, item)

        regions.append(
            Region(
                page=page_idx,
                bbox=[x0, y0, x1, y1],
                region_type=element_type,
                confidence=1.0,  # MinerU does not expose per-region confidence; use 1.0
                content=content,
            )
        )
    return regions


def _build_content_for(mineru_type: str, item: dict[str, Any]) -> ElementContent:
    if mineru_type in ("text", "title"):
        return ElementContent(text=str(item.get("text", "")).strip() or None)
    if mineru_type == "table":
        md = _html_table_to_markdown(str(item.get("table_body", "")))
        return ElementContent(markdown=md.strip() or None)
    if mineru_type == "equation":
        return ElementContent(latex=str(item.get("text", "")).strip() or None)
    if mineru_type in ("image", "chart"):
        return ElementContent()  # image_path is filled by the pipeline later
    return ElementContent()


@register_segmenter("mineru25")
@register_table_extractor("mineru25")
class MinerU25:
    """Wrap MinerU 2.5+ for segmentation and table extraction.

    Construction is cheap: no mineru import, no model load. Model state
    is materialized on the first call to `segment()` / `extract()` and
    torn down on `unload()` (called by the pipeline's
    release_runtime_resources helper between roles).
    """

    TOOL_NAME = "mineru25"

    def __init__(self, **adapter_config: Any) -> None:
        self._config = adapter_config
        self._loaded: bool = False  # observable flag for lifecycle tests

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def segment(self, pdf_path: Path) -> list[Region]:
        # Implemented in Task 4.
        raise NotImplementedError("MinerU25.segment implemented in Task 4")

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        # Implemented in Task 5. Normally bypassed by the pipeline's
        # tool_match optimization when segmenter and table_extractor
        # share tool_name.
        raise NotImplementedError("MinerU25.extract implemented in Task 5")

    def unload(self) -> None:
        # Implemented in Task 5.
        raise NotImplementedError("MinerU25.unload implemented in Task 5")
