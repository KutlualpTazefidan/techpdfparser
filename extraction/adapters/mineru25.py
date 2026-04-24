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
        self._pdf_cache: dict[str, list[Region]] = {}

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def _ensure_loaded(self) -> None:
        # MinerU's ModelSingleton loads lazily on the first do_parse call.
        # We set our own flag so lifecycle tests can observe state transitions.
        self._loaded = True

    def segment(self, pdf_path: Path) -> list[Region]:
        self._ensure_loaded()
        resolved = pdf_path.resolve()
        key = str(resolved)
        if key in self._pdf_cache:
            return self._pdf_cache[key]

        content_list, page_sizes_pts = self._run_do_parse_and_read(resolved)
        regions = _to_regions(content_list, page_sizes_pts)
        self._pdf_cache[key] = regions
        return regions

    def _run_do_parse_and_read(
        self, pdf_path: Path
    ) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
        """Invoke mineru's do_parse into a scratch dir; read content_list + page sizes.

        Heavy imports (mineru, pymupdf) are deferred to this method so the
        module stays importable on CPU-only installs.
        """
        import json
        import tempfile

        import pymupdf  # for page sizes
        from mineru.cli.common import do_parse

        pdf_bytes = pdf_path.read_bytes()
        stem = pdf_path.stem

        with tempfile.TemporaryDirectory(prefix="mineru25_") as tmp:
            tmp_dir = Path(tmp)
            do_parse(
                output_dir=str(tmp_dir),
                pdf_file_names=[stem],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[self._config.get("lang", "en")],
                backend=self._config.get("backend", "pipeline"),
                parse_method=self._config.get("parse_method", "auto"),
                formula_enable=bool(self._config.get("formula_enable", True)),
                table_enable=bool(self._config.get("table_enable", True)),
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=False,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=True,
            )

            # MinerU writes content_list under <tmp>/<stem>/<parse_method>/<stem>_content_list.json
            parse_method_dir = self._config.get("parse_method", "auto")
            content_list_candidates = list(
                (tmp_dir / stem / parse_method_dir).glob("*_content_list.json")
            )
            if not content_list_candidates:
                raise RuntimeError(
                    f"mineru25: do_parse produced no content_list.json under "
                    f"{tmp_dir / stem / parse_method_dir} (directory listing: "
                    f"{list((tmp_dir / stem).rglob('*'))[:20]})"
                )
            content_list_path = content_list_candidates[0]
            content_list = json.loads(content_list_path.read_text(encoding="utf-8"))

        # Open the PDF with PyMuPDF to read page sizes in points.
        doc = pymupdf.open(str(pdf_path))
        try:
            page_sizes_pts = [
                (float(doc[i].rect.width), float(doc[i].rect.height))
                for i in range(doc.page_count)
            ]
        finally:
            doc.close()

        return content_list, page_sizes_pts

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        # Normally bypassed by the pipeline's tool_match when segmenter and
        # table_extractor share tool_name. Return empty content so the
        # Protocol is satisfied if tool_match ever does not apply.
        return ElementContent()

    def unload(self) -> None:
        """Release MinerU's held VRAM and drop all model references.

        Called by `extraction._runtime.release_runtime_resources` between
        pipeline roles. Safe to call multiple times and before any load.
        """
        # Clear pipeline backend's ModelSingleton (if the module was ever imported).
        try:
            from mineru.backend.pipeline.pipeline_analyze import (
                ModelSingleton as _PipelineSingleton,
            )

            _PipelineSingleton()._models.clear()
        except ImportError:
            pass

        # Shut down VLM backend if it was used (not by default, but defensive).
        try:
            from mineru.backend.vlm.vlm_analyze import shutdown_cached_models

            shutdown_cached_models()
        except ImportError:
            pass

        import gc

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._pdf_cache.clear()
        self._loaded = False
