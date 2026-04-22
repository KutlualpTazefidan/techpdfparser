# Architecture Overview

> Companion document for newcomers. Explains the idea behind `techpdfparser`,
> the layered design, the current state of the extraction block, and walks
> through adding a new tool end-to-end.
>
> For the output format spec, see `docs/extraction_output.md`.
> For the adapter-author checklist, see `docs/writing_adapters.md`.
> For behavioral principles, see `docs/principles.md`.
> Code is authoritative — where this document and code disagree, trust the code.

---

## 1. Vision

The long-term goal is a system that **verifies technical claims in PDFs**
and traces them back to the supporting evidence — think of safety-critical
documentation where a sentence like *"the container was tested at 3 bar"*
must be grounded in a measurement table, a test procedure, and a signed
report.

Reaching that goal means going through three stages, each a separate,
independently replaceable component:

1. **Extraction** — turn the PDF (text, tables, figures, formulas,
   drawings) into a uniform, structured representation.
2. **Embedding + Storage** — index that representation into a knowledge
   graph and/or vector store.
3. **Agent / Verification** — retrieve evidence for a claim, check
   consistency, surface contradictions.

Only stage 1 is in scope for this repository right now. It must produce
a **stable output contract** so stages 2 and 3 can be built without
constantly chasing format changes.

---

## 2. System Architecture — Four Layers

```
┌────────────────────────────────────────────────────────────────┐
│  Layer 4 — Agent / Routing                                     │
│  Claim verification, retrieval orchestration, evidence check   │
└────────────────────────────────────────────────────────────────┘
                              ▲
┌────────────────────────────────────────────────────────────────┐
│  Layer 3 — Storage                                             │
│  Vector DB + graph DB; uniform query surface                   │
└────────────────────────────────────────────────────────────────┘
                              ▲
┌────────────────────────────────────────────────────────────────┐
│  Layer 2 — Embedding                                           │
│  Chunk, embed, build relations from structured output          │
└────────────────────────────────────────────────────────────────┘
                              ▲
┌────────────────────────────────────────────────────────────────┐
│  Layer 1 — Extraction   ◀── YOU ARE HERE                       │
│  PDF → structured elements (text, table, formula, figure, …)   │
└────────────────────────────────────────────────────────────────┘
```

**Why strict separation?** Because each layer has a different replacement
cadence. Extraction tools (OCR, layout models) evolve fast. Embedding
models change every few months. Storage backends rarely change. The
agent layer is where product evolves. Coupling them means every small
extraction improvement drags the whole stack.

**The contract between layers is the output format**, not the code.
Layer 2 reads `content_list.json` (defined in `docs/extraction_output.md`)
and does not know what tools produced it.

---

## 3. Extraction Block — Current State

### 3.1 Flow

```
PDF
 │
 ▼
┌───────────┐   page images   ┌────────────┐   regions   ┌────────┐
│  Renderer │ ──────────────► │ Segmenter  │ ──────────► │ Router │
└───────────┘                 └────────────┘             └────┬───┘
                                                              │
                                         per role (text /     │
                                         table / formula /    ▼
                                         figure)         ┌──────────┐
                                                         │ Extractor│
                                                         │   pool   │
                                                         └────┬─────┘
                                                              │
                                                   merge rule │
                                                              ▼
                                                     ┌────────────────┐
                                                     │ OutputWriter   │
                                                     │ sidecars + .png│
                                                     │ content_list   │
                                                     └────────────────┘
```

Orchestrated by `ExtractionPipeline` in `extraction/pipeline.py`:

1. **Render** all pages into PIL images (one resolution, from config `dpi`).
2. **Segment** the PDF into typed `Region`s — bbox, page, type, confidence,
   optionally content (some segmenters like MinerU already carry table
   markdown or formula LaTeX).
3. **Route** each region to the role tool configured for its type.
4. **Extract** content with the role tool, unless the segmenter is the same
   tool as the role tool — then its content is reused (tool-match
   optimization).
5. **Merge** segmenter layout fields (caption) onto the extractor content.
6. **Drop** regions that ended up empty on a required field.
7. **Write** per-element JSON sidecars; then build `content_list.json`
   deterministically from those sidecars.

### 3.2 Six Roles

The pipeline depends on six swappable components, each defined as a
`Protocol` in `extraction/interfaces.py`:

| Role              | Input                       | Output              |
|-------------------|-----------------------------|---------------------|
| `PageRenderer`    | PDF path                    | `list[Image]`       |
| `Segmenter`       | PDF path                    | `list[Region]`      |
| `TextExtractor`   | page image                  | `ElementContent`    |
| `TableExtractor`  | region crop                 | `ElementContent`    |
| `FormulaExtractor`| region crop                 | `ElementContent`    |
| `FigureDescriptor`| region crop                 | `str` (description) |

Concrete adapters for these roles live under `extraction/adapters/`.
The registry maps a **name** (the string used in YAML) to the adapter
class. See `extraction/registry.py` for the mechanism, and
`extraction/config.py` for the default names.

### 3.3 Config Dictates, Pipeline Obeys

This is the single most important design rule in the extraction block.

> The YAML config assigns exactly **one tool per role**. The pipeline
> never second-guesses that choice. It does not fall back, heuristically
> pick a "better" tool, or skip the role when the segmenter already has
> content.

What the pipeline does do, per region:

1. Look up the role tool from config (e.g., `table_extractor: mineru25`).
2. Compare `role_tool.tool_name` against `segmenter.tool_name`. If equal
   *and* the segmenter already produced content for this region, keep
   that content. Otherwise crop the page and invoke the role tool.
3. Overlay the segmenter's `caption` (layout-side field) onto the
   resulting content.
4. Drop the region if its required content field is empty.

Full data-source table and the merge rule rewrite: see
`docs/extraction_output.md` → Merge-Regeln.

### 3.4 Registry and Adapter Loading

Adapters register themselves at import time via decorator:

```python
# extraction/adapters/stubs.py
@register_text_extractor("noop")
class NoopTextExtractor:
    TOOL_NAME = "noop"
    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME
    def extract(self, page_image, page_number):
        return ElementContent(text="")
```

`extraction/adapters/__init__.py` imports each adapter module. Heavy
adapters (MinerU, OlmOCR, Qwen-VL) are wrapped in `try/except ImportError`
so the package imports cleanly on CPU-only hosts. Their decorator only
fires when their dependencies are installed; if the user names an
unavailable adapter in the config, `get_*` raises a clear `KeyError`
listing what *is* available.

### 3.5 Output Format — One Contract, Two Views

Every element ends up in two places:

- **Per-element sidecar** `pages/<page>/<element_id>_<type>.json`
  — this is the **source of truth**.
- **Merged flat list** `content_list.json` — a deterministic, sorted
  rebuild from the sidecars. Can be regenerated anytime via
  `python -m extraction rebuild outputs/<run>/`.

Visual element types (`table`, `formula`, `figure`, `diagram`,
`technical_drawing`) additionally get a PNG crop alongside their JSON.

Coordinate system: all `bbox` values are in **PDF points** (origin
top-left, DPI-independent). The only place points are scaled to pixels
is `OutputWriter.crop_region` (scale = `dpi / 72`), with overflow
clamped. Segmenters stay DPI-agnostic.

See `docs/extraction_output.md` for the full schema, directory layout,
reading-order rules, and ID formulas.

### 3.6 Fail-Safes

A few cross-cutting invariants the pipeline enforces:

- **Output dir isolation**: `Pipeline.run()` refuses to write into a
  directory that already contains `content_list.json`, `segmentation.json`,
  or a non-empty `pages/`. Use a fresh `--output` directory per run.
- **Confidence filter**: every element with `confidence < threshold`
  (default `0.3`, from `ExtractionConfig.confidence_threshold`) is
  dropped *after* extraction but before writing. This is why segmenters
  must return **real** scores — hard-coded `1.0` silently disables the
  filter (see `docs/writing_adapters.md`).
- **Empty-content drop**: text/heading without non-empty text, tables
  without markdown or text, formulas without LaTeX, and visuals with
  neither `image_path` nor `description` are dropped rather than
  persisted as empty records.
- **Path-independent IDs**: `element_id` is a hash of
  `"<doc_id>:<page>:<type>:<rounded bbox>"`. Same PDF bytes → same IDs,
  no matter where the file lives.

---

## 4. Data Model

All models live in `extraction/models.py` as Pydantic classes.

```
Region             ◀── what the segmenter returns
 ├─ page, bbox, region_type, confidence
 └─ content: ElementContent | None

Element            ◀── what the pipeline writes
 ├─ element_id, type, page, bbox
 ├─ reading_order_index, section_path
 ├─ confidence, extractor (tool name)
 └─ content: ElementContent

ElementContent
 ├─ text, markdown, latex       ◀── per-type content fields
 ├─ image_path, description
 └─ caption                     ◀── layout field (from segmenter)

ContentList        ◀── merged view, root of content_list.json
 ├─ doc_id, source_file, total_pages, schema_version
 ├─ segmentation_tool
 ├─ pages: [PageInfo]
 └─ elements: [Element]
```

`DocumentRich`, `Section`, and `Relation` are defined but **not yet
produced** in phase 1 — they come with the phase-2 `document_rich.json`.

---

## 5. What's Implemented vs Planned

**Implemented (phase 1):**

- Full render → segment → route → extract → merge → write pipeline
- Six-role config-driven adapter system
- `content_list.json` + sidecars + page/element PNGs
- MinerU 2.5 segmenter with real confidence scores
- OlmOCR 2 text extractor, Qwen2.5-VL figure descriptor
- PyMuPDF CPU-only baseline (renderer + rule-based text segmenter)
- `rebuild` CLI for regenerating `content_list.json` from sidecars

**Planned (phase 2, not in this iteration):**

- `document_rich.json` with hierarchical sections and relations
  (`captioned_by`, `refers_to`)
- `section_path` populated per element
- `mentions` extraction ("Table 3", "Eq. 5", …)
- `extractor_version` on each element

Phase 2 is computable *from* the phase-1 output — no re-parsing of the
PDF is required. See `docs/extraction_output.md` → "Was Phase 2 bringt"
and `backlog.md`.

---

## 6. Adding a New Tool — End-to-End Example

Suppose you want to evaluate a new table extractor — say, a hypothetical
`tabulite` library that promises better merged-cell handling. Here's
every step, in order.

### Step 1 — Pick the role, find the Protocol

Tables are handled by `TableExtractor`. The Protocol:

```python
# extraction/interfaces.py
class TableExtractor(Protocol):
    @property
    def tool_name(self) -> str: ...
    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        """Extract table content from a cropped region."""
```

A compliant adapter needs a `tool_name` property and an `extract`
method that returns `ElementContent` with `markdown` and/or `text` set.

### Step 2 — Write the adapter

Create `extraction/adapters/tabulite_table.py`:

```python
"""Tabulite table extractor.

Runs tabulite on a pre-cropped table region and returns markdown.
"""
from __future__ import annotations

from typing import Any

from PIL.Image import Image

from ..models import ElementContent
from ..registry import register_table_extractor


@register_table_extractor("tabulite")
class TabuliteTableExtractor:
    TOOL_NAME = "tabulite"

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._engine: Any = None

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def _load(self) -> None:
        if self._engine is not None:
            return
        try:
            import tabulite  # heavy import — keep lazy
        except ImportError as exc:
            raise ImportError("tabulite not installed. Run: pip install tabulite") from exc
        self._engine = tabulite.load(self._model_path)

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        self._load()
        result = self._engine.analyze(region_image)
        return ElementContent(
            markdown=result.markdown or None,
            text=result.plain_text or None,
        )
```

Points worth noting — all mandated by `docs/writing_adapters.md`:

- `tool_name` property value and the decorator string are **identical**
  (`"tabulite"`). The merge rule relies on string equality.
- The heavy import is lazy (inside `_load`), so the package still imports
  on a CPU-only host without tabulite installed.
- The adapter only returns **content fields**. Layout fields like
  `caption` belong to the segmenter and are overlaid by the pipeline.

### Step 3 — Register the import

Open `extraction/adapters/__init__.py` and add a guarded import:

```python
try:
    from . import tabulite_table  # noqa: F401
except ImportError as exc:
    log.debug("tabulite_table not registered: %s", exc)
```

This is what triggers the `@register_table_extractor("tabulite")`
decorator at package-load time when its dependencies are available.

### Step 4 — Configure it

In your YAML config:

```yaml
extraction:
  segmenter: mineru25
  table_extractor: tabulite     # ← swapped in
  text_extractor: olmocr2
  figure_descriptor: qwen25vl
  formula_extractor: noop

adapters:
  tabulite:
    model_path: /opt/models/tabulite-v2.bin
```

The `adapters.tabulite` block becomes the keyword arguments passed to
`TabuliteTableExtractor.__init__`. Config keys must match parameter
names exactly.

### Step 5 — Understand the merge interaction

Because `segmenter: mineru25` and `table_extractor: tabulite` are
**different** tool names, the merge rule in `ExtractionPipeline` takes
the *extractor* branch: it crops each table region and calls
`TabuliteTableExtractor.extract`. MinerU's own table markdown is
discarded. The segmenter-provided `caption` still survives — it's
layout, not content.

Compare with the default config (`table_extractor: mineru25`), where
tool-match short-circuits the crop-and-call and reuses MinerU's
built-in markdown. Switching extractors is one config line; the
pipeline adapts automatically.

### Step 6 — Write tests

Follow the pattern of existing adapter tests
(e.g. `extraction/tests/test_mineru25_segmenter.py`). A lightweight
unit test with mocked dependencies belongs in the default suite.
Anything requiring the real model weights or GPU goes behind a marker:

```python
# extraction/tests/test_tabulite_table.py
import pytest
from PIL import Image

from extraction.adapters.tabulite_table import TabuliteTableExtractor


def test_tool_name_matches_registration():
    adapter = TabuliteTableExtractor.__new__(TabuliteTableExtractor)
    assert adapter.tool_name == "tabulite"


@pytest.mark.integration
def test_tabulite_extracts_simple_table():
    adapter = TabuliteTableExtractor()
    image = Image.open("extraction/tests/fixtures/simple_table.png")
    result = adapter.extract(image, page_number=0)
    assert result.markdown
    assert "|" in result.markdown
```

Integration tests are opt-in:

    pytest -m integration

### Step 7 — Verify the quality gates

    pytest -q
    ruff check extraction
    mypy

All three must stay green. The `pyproject.toml` `[tool.mypy]` section
lists which paths are type-checked; any new adapter file inside
`extraction/adapters/` is covered automatically.

### Step 8 — Smoke-test against a real PDF

    python -m extraction extract docs/fixtures/sample.pdf \
        --config configs/with_tabulite.yaml \
        --output outputs/tabulite_smoke/

Inspect `outputs/tabulite_smoke/content_list.json`. For every `"type":
"table"` element, check that `"extractor": "tabulite"` and that
`content.markdown` is populated. Spot-check `pages/<n>/<id>_table.png`
to confirm the crop lined up with the extracted content.

### Step 9 — Document the decision

If the new adapter becomes a default, update the relevant row in
`extraction/config.py` (one default value change) and mention the
rationale in `backlog.md`. **Do not** list adapter names in
`CLAUDE.md` or other docs — the config file is the single source of
truth; duplicate lists rot fast.

---

## 7. Where to Look

| Concern                            | File                                    |
|------------------------------------|-----------------------------------------|
| Pipeline orchestration             | `extraction/pipeline.py`                |
| Role interfaces (Protocols)        | `extraction/interfaces.py`              |
| Registry mechanics                 | `extraction/registry.py`                |
| Config model + defaults            | `extraction/config.py`                  |
| Data models (Region, Element, …)   | `extraction/models.py`                  |
| Output writer and crop scaling     | `extraction/output.py`                  |
| CLI (`extract`, `rebuild`)         | `extraction/__main__.py`                |
| Adapter list + lazy imports        | `extraction/adapters/__init__.py`       |
| Output format spec                 | `docs/extraction_output.md`             |
| Adapter-author checklist           | `docs/writing_adapters.md`              |
| Behavioral principles              | `docs/principles.md`                    |
| Open work items                    | `backlog.md`                            |

When in doubt, read the code — it is authoritative.
