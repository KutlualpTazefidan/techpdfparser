# techpdfparser

Technical-PDF parsing into a unified structured output: text, tables,
formulas, figures, diagrams, drawings. Output format is a stable contract
(see `docs/extraction_output.md`). Tools are swappable per role via YAML.

## Install

CPU-only (no segmenter):

    pip install -e .

Full GPU stack (MinerU 2.5, OlmOCR, Qwen2.5-VL):

    pip install -e .[gpu]

PyTorch: CUDA builds depend on your system. Follow
https://pytorch.org/get-started/locally/ and install a matching
`torch` wheel before or alongside `-e .[gpu]`.

## CLI

Extract a PDF:

    python -m extraction extract path/to/document.pdf --config config.yaml --output outputs/run1/

Rebuild `content_list.json` from existing sidecars (no re-extraction):

    python -m extraction rebuild outputs/run1/

## CPU-only config example

    extraction:
      renderer: pymupdf
      segmenter: pymupdf_text
      text_extractor: noop
      table_extractor: noop
      formula_extractor: noop
      figure_descriptor: noop
      output_dir: outputs
      dpi: 150

## Quality gates

    pytest -q
    ruff check extraction
    mypy

Integration tests (MinerU / GPU) are marker-gated:

    pytest -m integration
