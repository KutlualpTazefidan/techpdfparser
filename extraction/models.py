"""Data models for the extraction block output format.

Defines the schema for content_list.json, document_rich.json,
and the intermediate segmentation regions.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FORMULA = "formula"
    FIGURE = "figure"
    DIAGRAM = "diagram"
    TECHNICAL_DRAWING = "technical_drawing"


class ElementContent(BaseModel):
    """Content fields vary by element type. All are optional; presence depends on type."""

    text: str | None = None
    markdown: str | None = None
    latex: str | None = None
    image_path: str | None = None
    description: str | None = None
    caption: str | None = None


class Element(BaseModel):
    """One extractable unit from the PDF — a text block, table, formula, figure, etc."""

    element_id: str
    type: ElementType
    page: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    reading_order_index: int
    section_path: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    extractor: str
    content: ElementContent


class PageInfo(BaseModel):
    """Per-page metadata linking the page image to its elements."""

    page: int
    image_path: str
    element_ids: list[str] = Field(default_factory=list)


class ContentList(BaseModel):
    """Root model for content_list.json — flat element list in reading order."""

    doc_id: str
    source_file: str
    total_pages: int
    schema_version: str = "1.0"
    segmentation_tool: str
    pages: list[PageInfo] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)


class Section(BaseModel):
    """A section in the document hierarchy."""

    heading: str
    level: int
    page_start: int
    children: list[str] = Field(default_factory=list)
    subsections: list[Section] = Field(default_factory=list)


class Relation(BaseModel):
    """A reference between two elements (e.g. text refers to a table)."""

    source: str
    target: str
    type: str
    evidence: str = ""


class DocumentRich(BaseModel):
    """Root model for document_rich.json — hierarchical structure + relations."""

    doc_id: str
    source_file: str
    total_pages: int
    schema_version: str = "1.0"
    segmentation_tool: str
    sections: list[Section] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class Region(BaseModel):
    """A detected region from layout segmentation, before extraction."""

    page: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    region_type: ElementType
    confidence: float = Field(ge=0.0, le=1.0)
    content: ElementContent | None = None
