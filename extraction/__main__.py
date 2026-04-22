"""CLI entrypoint: python -m extraction {extract, rebuild} ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import extraction.adapters  # noqa: F401 — trigger adapter registration

from .config import ExtractionConfig, load_extraction_config
from .output import OutputWriter
from .pipeline import ExtractionPipeline
from .registry import (
    get_figure_descriptor,
    get_formula_extractor,
    get_renderer,
    get_segmenter,
    get_table_extractor,
    get_text_extractor,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="extraction")
    sub = parser.add_subparsers(dest="command")

    extract = sub.add_parser("extract", help="Extract structured content from a PDF")
    extract.add_argument("pdf", type=Path, help="Path to the PDF file")
    extract.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    extract.add_argument("--output", type=Path, default=None, help="Output directory")

    rebuild = sub.add_parser(
        "rebuild",
        help="Rebuild content_list.json from per-element JSON sidecars",
    )
    rebuild.add_argument("output_dir", type=Path)
    rebuild.add_argument("--doc-id", type=str, default=None)
    rebuild.add_argument("--source", type=str, default=None)
    rebuild.add_argument("--pages", type=int, default=None)
    rebuild.add_argument("--segmenter", type=str, default=None)

    return parser.parse_args()


def _resolve_renderer_dpi(cfg: ExtractionConfig) -> int:
    """Top-level cfg.dpi unless the renderer adapter block overrides it."""
    adapter_cfg = cfg.get_adapter_config(cfg.renderer)
    if "dpi" in adapter_cfg:
        return int(adapter_cfg["dpi"])
    return int(cfg.dpi)


def _load_cfg(config_path: Path | None) -> ExtractionConfig:
    if config_path is not None:
        return load_extraction_config(config_path)
    default = Path("extraction_config.yaml")
    if default.exists():
        return load_extraction_config(default)
    return ExtractionConfig()


def _run_extract(pdf_path: Path, cfg: ExtractionConfig, output_dir: Path | None) -> None:
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    out = output_dir or Path(cfg.output_dir)

    dpi = _resolve_renderer_dpi(cfg)
    renderer_kwargs = dict(cfg.get_adapter_config(cfg.renderer))
    renderer_kwargs.setdefault("dpi", dpi)
    renderer = get_renderer(cfg.renderer, **renderer_kwargs)
    segmenter = get_segmenter(cfg.segmenter, **cfg.get_adapter_config(cfg.segmenter))
    text_extractor = get_text_extractor(
        cfg.text_extractor, **cfg.get_adapter_config(cfg.text_extractor)
    )
    table_extractor = get_table_extractor(
        cfg.table_extractor, **cfg.get_adapter_config(cfg.table_extractor)
    )
    formula_extractor = get_formula_extractor(
        cfg.formula_extractor, **cfg.get_adapter_config(cfg.formula_extractor)
    )
    figure_descriptor = get_figure_descriptor(
        cfg.figure_descriptor, **cfg.get_adapter_config(cfg.figure_descriptor)
    )

    pipeline = ExtractionPipeline(
        renderer=renderer,
        segmenter=segmenter,
        text_extractor=text_extractor,
        table_extractor=table_extractor,
        formula_extractor=formula_extractor,
        figure_descriptor=figure_descriptor,
        output_dir=out,
        confidence_threshold=cfg.confidence_threshold,
        dpi=dpi,
    )

    print(f"Extracting {pdf_path.name}...")
    content_list = pipeline.run(pdf_path)
    print(f"  Elements: {len(content_list.elements)}")
    print(f"  Pages:    {content_list.total_pages}")
    print(f"  Output:   {out}")


def _run_rebuild(
    output_dir: Path,
    doc_id: str | None,
    source: str | None,
    pages: int | None,
    segmenter: str | None,
) -> None:
    if not output_dir.exists():
        print(f"Error: output dir not found: {output_dir}")
        sys.exit(1)

    existing = output_dir / "content_list.json"
    meta: dict[str, Any] = {}
    if existing.exists():
        meta = json.loads(existing.read_text(encoding="utf-8"))

    resolved_doc_id = doc_id or str(meta.get("doc_id", ""))
    resolved_source = source or str(meta.get("source_file", ""))
    resolved_pages = pages if pages is not None else int(meta.get("total_pages", 0))
    resolved_seg = segmenter or str(meta.get("segmentation_tool", ""))

    missing = [
        name
        for name, val in [
            ("doc-id", resolved_doc_id),
            ("source", resolved_source),
            ("pages", resolved_pages),
            ("segmenter", resolved_seg),
        ]
        if not val
    ]
    if missing:
        print(
            "Error: rebuild needs these values (pass as flags or keep an existing "
            f"content_list.json): {', '.join(missing)}"
        )
        sys.exit(1)

    writer = OutputWriter(output_dir)
    cl = writer.build_content_list(
        doc_id=resolved_doc_id,
        source_file=resolved_source,
        total_pages=resolved_pages,
        segmentation_tool=resolved_seg,
    )
    writer.write_content_list(cl)
    print(f"Rebuilt content_list.json with {len(cl.elements)} elements in {output_dir}")


def main() -> None:
    args = _parse_args()

    if args.command is None:
        print("Usage: python -m extraction {extract, rebuild} ...")
        sys.exit(1)

    if args.command == "extract":
        cfg = _load_cfg(getattr(args, "config", None))
        _run_extract(args.pdf, cfg, getattr(args, "output", None))
        return

    if args.command == "rebuild":
        _run_rebuild(
            args.output_dir, args.doc_id, args.source, args.pages, args.segmenter
        )
        return

    print(f"Unknown command: {args.command}")
    sys.exit(1)


if __name__ == "__main__":
    main()
