"""Write extraction results to the spec output format.

Produces content_list.json, per-element JSON sidecars, segmentation.json,
and image crops inside an output directory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL.Image import Image

from .models import ContentList, DocumentRich, Element, ElementType, PageInfo

_SIDECAR_TYPES = "|".join(t.value for t in ElementType)
_SIDECAR_RE = re.compile(rf"^.+_({_SIDECAR_TYPES})\.json$")


class OutputWriter:
    """Manages writing extraction output to a directory."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_content_list(self, content_list: ContentList) -> Path:
        path = self.output_dir / "content_list.json"
        data = content_list.model_dump(mode="json", exclude_none=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def write_document_rich(self, document_rich: DocumentRich) -> Path:
        path = self.output_dir / "document_rich.json"
        data = document_rich.model_dump(mode="json", exclude_none=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def write_segmentation(self, regions: list) -> Path:
        """Write raw segmentation regions for later inspection."""
        path = self.output_dir / "segmentation.json"
        data = [r.model_dump(mode="json", exclude_none=True) for r in regions]
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def save_page_image(self, page: int, image: Image) -> Path:
        page_dir = self.output_dir / "pages" / str(page)
        page_dir.mkdir(parents=True, exist_ok=True)
        path = page_dir / "page.png"
        image.save(path)
        return path

    def save_element_crop(
        self, page: int, element_id: str, element_type: str, image: Image
    ) -> Path:
        page_dir = self.output_dir / "pages" / str(page)
        page_dir.mkdir(parents=True, exist_ok=True)
        path = page_dir / f"{element_id}_{element_type}.png"
        image.save(path)
        return path

    def crop_region(
        self, page_image: Image, bbox: list[float], dpi: int = 72
    ) -> Image:
        """Crop a region given in PDF-points from a page image rendered at `dpi`."""
        scale = dpi / 72.0
        x0 = max(0, int(bbox[0] * scale))
        y0 = max(0, int(bbox[1] * scale))
        x1 = min(page_image.width, int(bbox[2] * scale + 0.999))
        y1 = min(page_image.height, int(bbox[3] * scale + 0.999))
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, page_image.width, page_image.height
        return page_image.crop((x0, y0, x1, y1))

    def write_element_sidecar(self, element: Element) -> Path:
        """Write one element to pages/<page>/<el_id>_<type>.json (source of truth)."""
        page_dir = self.output_dir / "pages" / str(element.page)
        page_dir.mkdir(parents=True, exist_ok=True)
        path = page_dir / f"{element.element_id}_{element.type.value}.json"
        data = element.model_dump(mode="json", exclude_none=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def read_all_sidecars(self) -> list[Element]:
        """Load every <el_id>_<type>.json from pages/*/ into Element instances."""
        pages_dir = self.output_dir / "pages"
        if not pages_dir.exists():
            return []
        elements: list[Element] = []
        for page_dir in sorted(pages_dir.iterdir()):
            if not page_dir.is_dir():
                continue
            for json_path in sorted(page_dir.glob("*.json")):
                if not _SIDECAR_RE.match(json_path.name):
                    continue
                data = json.loads(json_path.read_text(encoding="utf-8"))
                elements.append(Element.model_validate(data))
        return elements

    def build_content_list(
        self,
        doc_id: str,
        source_file: str,
        total_pages: int,
        segmentation_tool: str,
    ) -> ContentList:
        """Deterministically rebuild content_list from sidecars.

        Sorts by (page, reading_order_index, element_id) and re-numbers
        reading_order_index globally across all pages.
        """
        elements = self.read_all_sidecars()
        elements.sort(key=lambda e: (e.page, e.reading_order_index, e.element_id))
        for idx, el in enumerate(elements):
            el.reading_order_index = idx
        pages = [
            PageInfo(
                page=p,
                image_path=f"pages/{p}/page.png",
                element_ids=[e.element_id for e in elements if e.page == p],
            )
            for p in range(total_pages)
        ]
        return ContentList(
            doc_id=doc_id,
            source_file=source_file,
            total_pages=total_pages,
            segmentation_tool=segmentation_tool,
            pages=pages,
            elements=elements,
        )
