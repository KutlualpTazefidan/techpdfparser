"""PyMuPDF-based page renderer. CPU-only baseline for Stage 1."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image as PILImage
from PIL.Image import Image

from ..registry import register_renderer

_POINTS_PER_INCH = 72.0


@register_renderer("pymupdf")
class PyMuPDFRenderer:
    """Render PDF pages to PIL images at a configurable DPI."""

    def __init__(self, dpi: int = 150, **_kwargs: Any) -> None:
        self._dpi = int(dpi)

    @property
    def tool_name(self) -> str:
        return "pymupdf"

    def page_count(self, pdf_path: Path) -> int:
        doc = pymupdf.open(str(pdf_path))
        try:
            return int(doc.page_count)
        finally:
            doc.close()

    def render_page(self, pdf_path: Path, page_number: int) -> Image:
        doc = pymupdf.open(str(pdf_path))
        try:
            return self._render(doc[page_number])
        finally:
            doc.close()

    def render_all(self, pdf_path: Path) -> list[Image]:
        doc = pymupdf.open(str(pdf_path))
        try:
            return [self._render(doc[i]) for i in range(doc.page_count)]
        finally:
            doc.close()

    def _render(self, page: Any) -> Image:
        zoom = self._dpi / _POINTS_PER_INCH
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pix.tobytes("png")
        # .copy() detaches from the BytesIO so the caller can use it freely.
        return PILImage.open(io.BytesIO(img_bytes)).copy()
