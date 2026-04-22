"""Extraction block configuration.

Load from YAML or use defaults. Each adapter has its own config section.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ExtractionConfig(BaseModel):
    """Configuration for the extraction pipeline."""

    renderer: str = "pymupdf"
    segmenter: str = "mineru25"
    text_extractor: str = "olmocr2"
    table_extractor: str = "mineru25"
    # MinerU liefert interline_equation-LaTeX direkt aus middle_json mit.
    # Ein dedizierter ppformulanet-Port folgt, wenn ein Segmenter ohne LaTeX
    # hinzukommt oder der OCR-Benchmark ihn fordert.
    formula_extractor: str = "noop"
    figure_descriptor: str = "qwen25vl"
    output_dir: str = "outputs"
    confidence_threshold: float = 0.3
    dpi: int = 150
    adapters: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_adapter_config(self, adapter_name: str) -> dict[str, Any]:
        return self.adapters.get(adapter_name, {})


def load_extraction_config(path: str | Path) -> ExtractionConfig:
    """Load extraction config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    extraction_raw = raw.get("extraction", {})
    extraction_raw["adapters"] = raw.get("adapters", {})
    return ExtractionConfig.model_validate(extraction_raw)
