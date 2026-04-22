"""Protocol classes for all swappable extraction components.

Each adapter type has a Protocol that defines the interface.
Concrete adapters register via decorators in extraction/registry.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL.Image import Image

from .models import ElementContent, Region


class PageRenderer(Protocol):
    @property
    def tool_name(self) -> str: ...

    def page_count(self, pdf_path: Path) -> int: ...

    def render_page(self, pdf_path: Path, page_number: int) -> Image: ...

    def render_all(self, pdf_path: Path) -> list[Image]: ...


class Segmenter(Protocol):
    @property
    def tool_name(self) -> str: ...

    def segment(self, pdf_path: Path) -> list[Region]:
        """Analyze layout and return typed regions for all pages."""
        ...


class TextExtractor(Protocol):
    @property
    def tool_name(self) -> str: ...

    def extract(self, page_image: Image, page_number: int) -> ElementContent:
        """Extract text from a page image. Returns content with text field set."""
        ...


class TableExtractor(Protocol):
    @property
    def tool_name(self) -> str: ...

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        """Extract table content from a cropped region. Returns markdown + text."""
        ...


class FormulaExtractor(Protocol):
    @property
    def tool_name(self) -> str: ...

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        """Extract formula from a cropped region. Returns latex + text."""
        ...


class FigureDescriptor(Protocol):
    @property
    def tool_name(self) -> str: ...

    def describe(self, image: Image) -> str:
        """Generate a text description of a figure/diagram image."""
        ...
