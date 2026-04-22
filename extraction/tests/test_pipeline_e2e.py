"""End-to-end CPU pipeline smoke test.

Runs the full ExtractionPipeline with Stage 1 adapters against the
fixture PDF. Asserts structural properties only — no exact-content
checks, since that would be fragile across fixture changes.
"""
from __future__ import annotations

import json
from pathlib import Path

from extraction.config import ExtractionConfig
from extraction.models import ContentList, ElementType
from extraction.pipeline import ExtractionPipeline
from extraction.registry import (
    get_figure_descriptor,
    get_formula_extractor,
    get_renderer,
    get_segmenter,
    get_table_extractor,
    get_text_extractor,
)


def _stage1_cpu_config() -> ExtractionConfig:
    return ExtractionConfig(
        renderer="pymupdf",
        segmenter="pymupdf_text",
        text_extractor="pymupdf_text",
        table_extractor="noop",
        formula_extractor="noop",
        figure_descriptor="noop",
    )


def test_cpu_pipeline_runs_and_emits_valid_output(
    sample_text_pdf: Path, tmp_output_dir: Path
) -> None:
    cfg = _stage1_cpu_config()

    pipeline = ExtractionPipeline(
        renderer=get_renderer(cfg.renderer, dpi=cfg.dpi),
        segmenter=get_segmenter(cfg.segmenter),
        text_extractor=get_text_extractor(cfg.text_extractor),
        table_extractor=get_table_extractor(cfg.table_extractor),
        formula_extractor=get_formula_extractor(cfg.formula_extractor),
        figure_descriptor=get_figure_descriptor(cfg.figure_descriptor),
        output_dir=tmp_output_dir,
        confidence_threshold=cfg.confidence_threshold,
        dpi=cfg.dpi,
    )

    result = pipeline.run(sample_text_pdf)

    assert isinstance(result, ContentList)
    assert result.total_pages >= 1
    assert result.segmentation_tool == "pymupdf_text"
    assert len(result.elements) >= 1

    text_elements = [e for e in result.elements if e.type == ElementType.TEXT]
    assert len(text_elements) >= 1
    # Every text element should have real text content
    for el in text_elements:
        assert (el.content.text or "").strip() != ""

    # content_list.json was written and re-validates against the schema
    cl_path = tmp_output_dir / "content_list.json"
    assert cl_path.exists()
    raw = json.loads(cl_path.read_text(encoding="utf-8"))
    ContentList.model_validate(raw)

    # Segmentation debug file was also written
    seg_path = tmp_output_dir / "segmentation.json"
    assert seg_path.exists()


def test_cpu_pipeline_reading_order_is_sequential(
    sample_text_pdf: Path, tmp_output_dir: Path
) -> None:
    cfg = _stage1_cpu_config()
    pipeline = ExtractionPipeline(
        renderer=get_renderer(cfg.renderer, dpi=cfg.dpi),
        segmenter=get_segmenter(cfg.segmenter),
        text_extractor=get_text_extractor(cfg.text_extractor),
        table_extractor=get_table_extractor(cfg.table_extractor),
        formula_extractor=get_formula_extractor(cfg.formula_extractor),
        figure_descriptor=get_figure_descriptor(cfg.figure_descriptor),
        output_dir=tmp_output_dir,
        confidence_threshold=cfg.confidence_threshold,
        dpi=cfg.dpi,
    )
    result = pipeline.run(sample_text_pdf)

    indices = [e.reading_order_index for e in result.elements]
    # reading_order_index is assigned from the original region ordering
    assert indices == sorted(indices)
