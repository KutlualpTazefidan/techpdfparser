"""PyMuPDF-based CPU segmenter. Emits one TEXT region per text block.

Registers under two roles:
  - segmenter: produces Region list with per-block text filled in.
  - text_extractor: the pipeline's tool_match optimization bypasses this
    when segmenter and text_extractor share tool_name — extract() is
    implemented to satisfy the Protocol but is not expected to be called
    in the Stage-1 default config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
from PIL.Image import Image

from ..models import ElementContent, ElementType, Region
from ..registry import register_segmenter, register_text_extractor

# get_text("blocks") returns 7-tuples. Block type 0 means text, 1 means image.
_BLOCK_TYPE_TEXT = 0


@register_segmenter("pymupdf_text")
@register_text_extractor("pymupdf_text")
class PyMuPDFTextSegmenter:
    """Emit one TEXT region per PyMuPDF text block; fill region.content.text."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    @property
    def tool_name(self) -> str:
        return "pymupdf_text"

    def segment(self, pdf_path: Path) -> list[Region]:
        regions: list[Region] = []
        doc = pymupdf.open(str(pdf_path))
        try:
            for page_number, page in enumerate(doc):
                for block in page.get_text("blocks"):
                    # block: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(block) < 7:
                        continue
                    x0, y0, x1, y1, text, _block_no, block_type = block
                    if block_type != _BLOCK_TYPE_TEXT:
                        continue
                    cleaned = (text or "").strip()
                    if not cleaned:
                        continue
                    regions.append(
                        Region(
                            page=page_number,
                            bbox=[float(x0), float(y0), float(x1), float(y1)],
                            region_type=ElementType.TEXT,
                            confidence=1.0,
                            content=ElementContent(text=cleaned),
                        )
                    )
        finally:
            doc.close()
        return regions

    def extract(self, page_image: Image, page_number: int) -> ElementContent:
        # Not normally called: the pipeline's tool_match reuses segment()
        # output when text_extractor and segmenter share a tool_name.
        # Return empty content to satisfy the Protocol if tool_match ever
        # does not apply.
        return ElementContent()
