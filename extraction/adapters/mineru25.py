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

from ..models import ElementContent, Region
from ..registry import register_segmenter, register_table_extractor


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
