# Stage 1 — CPU Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the CPU foundation for `techpdfparser` — three adapters (pymupdf renderer, pymupdf_text segmenter + text_extractor, noop), a test scaffold with a CPU smoke E2E, GitHub Actions CI, a generated lock file, a `Makefile` target for regenerating it, and an `.github/pull_request_template.md`. At the end, `python -m extraction extract <fixture.pdf> --config config_cpu.yaml --output outputs/t1/` runs to completion on pure CPU and emits a schema-valid `content_list.json` containing real text.

**Architecture:** Three small adapter modules live under `extraction/adapters/`. Each registers with the existing decorator-based registry in `extraction/registry.py` and satisfies one or more of the Protocols in `extraction/interfaces.py`. An `extraction/adapters/__init__.py` imports the CPU modules unconditionally and wraps future GPU modules in `try/except ImportError`. The `pymupdf_text` class registers under BOTH `segmenter` and `text_extractor` so the pipeline's `tool_match` optimization in `extraction/pipeline.py:222-236` reuses segmenter output for the text role with no second extract call.

**Tech stack:** Python 3.10+, PyMuPDF, Pydantic v2, pytest, ruff, mypy, `uv` (for generating `requirements/cpu.lock`), GitHub Actions (CPU only).

**Parent spec:** `docs/superpowers/specs/2026-04-22-gpu-adapters-design.md`

---

## Prerequisites

Before starting:

- Working directory is `DocumentAnalysis/techpdfparser/`, on the `dev` branch.
- `git config user.email` returns `ktazefidan@hotmail.com` (local override already set during earlier session). Confirm with:
  ```bash
  git -C /home/ktazefid/Documents/projects/DocumentAnalysis/techpdfparser config user.email
  ```
- A real, text-heavy PDF is placed at `extraction/tests/fixtures/sample_text.pdf` **before Task 4** so the TDD tests have something to run against. Small is better (ideally under 1 MB). Provided by the user.
- `uv` is available on PATH. If not:
  ```bash
  pip install uv
  ```
  (Task 9 adds it to the project's `[dev]` extras, but initial lock generation needs it available before the lock itself is committed.)

## Task overview

| # | Task | Creates TDD cycle? |
|---|---|---|
| 1 | Package scaffolding (`extraction/tests/`, `extraction/adapters/`, `conftest.py`) | No |
| 2 | `.gitignore` update for fixture PDFs | No |
| 3 | `noop` adapter (TDD) | Yes |
| 4 | `pymupdf` renderer (TDD) | Yes |
| 5 | `pymupdf_text` segmenter + text_extractor (TDD) | Yes |
| 6 | End-to-end CPU pipeline smoke test | Yes (partial) |
| 7 | `pyproject.toml` cleanup (exclude pattern, add `uv`, pymupdf mypy override) | No |
| 8 | `config_cpu.yaml` at repo root | No |
| 9 | `Makefile` lock target + generate `requirements/cpu.lock` | No |
| 10 | GitHub Actions CI workflow | No |
| 11 | `.github/pull_request_template.md` | No |
| 12 | Final verification + push + open PR | No |

---

## Task 1: Package scaffolding

**Files:**
- Create: `extraction/adapters/__init__.py`
- Create: `extraction/tests/__init__.py`
- Create: `extraction/tests/conftest.py`
- Create: `extraction/tests/fixtures/` (directory, no files yet)

- [ ] **Step 1: Create the empty fixtures dir and adapters package marker**

Run:
```bash
mkdir -p extraction/tests/fixtures
```

Create `extraction/adapters/__init__.py`:

```python
"""Adapter modules for extraction roles.

Importing a module here triggers its decorator-based registration.
GPU-dependent modules (added in later stages) are wrapped in
try/except ImportError so CPU-only installs do not break.
"""
from __future__ import annotations
```

- [ ] **Step 2: Create the tests package marker**

Create `extraction/tests/__init__.py`:

```python
"""Test suite for the extraction package."""
```

- [ ] **Step 3: Create `conftest.py` with shared fixtures**

Create `extraction/tests/conftest.py`:

```python
"""Shared pytest fixtures for the extraction test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

import extraction.adapters  # noqa: F401 — trigger adapter registration

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_text_pdf() -> Path:
    """Path to the small text-heavy fixture PDF.

    Provided out-of-band by the maintainer (see plan Prerequisites).
    Tests that require it will skip if it is missing, so CI without the
    fixture stays green.
    """
    path = FIXTURES_DIR / "sample_text.pdf"
    if not path.exists():
        pytest.skip(f"Fixture PDF not found: {path}")
    return path


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Per-test isolated output dir; avoids pipeline._assert_output_dir_clean."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out
```

- [ ] **Step 4: Run pytest to confirm scaffolding imports cleanly**

```bash
pytest -q
```

Expected:
```
no tests ran in ...s
```

(Zero tests collected. Important: **no collection errors** — if `pytest` complains that `extraction.adapters` can't be imported, fix before continuing.)

- [ ] **Step 5: Commit**

```bash
git add extraction/adapters/__init__.py \
        extraction/tests/__init__.py \
        extraction/tests/conftest.py
git commit -m "scaffolding: add adapters package and tests package with shared fixtures"
```

---

## Task 2: Allow fixture PDFs in `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Read the current file to locate the `*.pdf` line**

```bash
grep -n "^\*\.pdf" .gitignore
```

Expected output: a line number showing `*.pdf` is in the file.

- [ ] **Step 2: Add an un-ignore line immediately after `*.pdf`**

Open `.gitignore` and add a new line immediately after the existing `*.pdf` line so the section reads:

```
# Test PDFs
*.pdf
!extraction/tests/fixtures/*.pdf
```

- [ ] **Step 3: Verify with git that the fixture path would now be trackable**

```bash
echo "placeholder" > extraction/tests/fixtures/.gitkeep
git check-ignore -v extraction/tests/fixtures/sample_text.pdf || echo "not ignored — good"
rm extraction/tests/fixtures/.gitkeep
```

Expected: the `echo "not ignored — good"` line prints (exit code 1 from `git check-ignore` means the path is NOT ignored, which is what we want).

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "gitignore: allow fixture PDFs in extraction/tests/fixtures/"
```

---

## Task 3: Implement the `noop` adapter (TDD)

**Files:**
- Test: `extraction/tests/test_noop.py`
- Create: `extraction/adapters/noop.py`
- Modify: `extraction/adapters/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `extraction/tests/test_noop.py`:

```python
"""Tests for the noop adapter.

Provides empty-content implementations of the text, table, formula,
and figure roles so a config can selectively skip any of them.
"""
from __future__ import annotations

from PIL import Image as PILImage

from extraction.models import ElementContent
from extraction.registry import (
    get_figure_descriptor,
    get_formula_extractor,
    get_table_extractor,
    get_text_extractor,
)


def test_noop_text_extractor_tool_name() -> None:
    assert get_text_extractor("noop").tool_name == "noop"


def test_noop_text_extractor_returns_empty_content() -> None:
    adapter = get_text_extractor("noop")
    blank = PILImage.new("RGB", (10, 10))
    result = adapter.extract(blank, page_number=0)
    assert isinstance(result, ElementContent)
    assert not (result.text or "").strip()


def test_noop_table_extractor_tool_name() -> None:
    assert get_table_extractor("noop").tool_name == "noop"


def test_noop_table_extractor_returns_empty_content() -> None:
    adapter = get_table_extractor("noop")
    blank = PILImage.new("RGB", (10, 10))
    result = adapter.extract(blank, page_number=0)
    assert isinstance(result, ElementContent)
    assert not (result.markdown or "").strip()


def test_noop_formula_extractor_tool_name() -> None:
    assert get_formula_extractor("noop").tool_name == "noop"


def test_noop_formula_extractor_returns_empty_content() -> None:
    adapter = get_formula_extractor("noop")
    blank = PILImage.new("RGB", (10, 10))
    result = adapter.extract(blank, page_number=0)
    assert isinstance(result, ElementContent)
    assert not (result.latex or "").strip()


def test_noop_figure_descriptor_tool_name() -> None:
    assert get_figure_descriptor("noop").tool_name == "noop"


def test_noop_figure_descriptor_returns_empty_string() -> None:
    adapter = get_figure_descriptor("noop")
    blank = PILImage.new("RGB", (10, 10))
    assert adapter.describe(blank) == ""
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
pytest extraction/tests/test_noop.py -v
```

Expected: every test fails with `KeyError: Unknown text_extractor adapter 'noop'. Available: []` (or similar per-role messages).

- [ ] **Step 3: Implement the `noop` adapter**

Create `extraction/adapters/noop.py`:

```python
"""No-op adapters. Return empty content; used to skip roles in a config."""
from __future__ import annotations

from typing import Any

from PIL.Image import Image

from ..models import ElementContent
from ..registry import (
    register_figure_descriptor,
    register_formula_extractor,
    register_table_extractor,
    register_text_extractor,
)


@register_text_extractor("noop")
@register_table_extractor("noop")
@register_formula_extractor("noop")
class NoopExtractor:
    """Extractor that always returns empty ElementContent.

    Stacked decorators register the same class under three roles.
    The registry holds classes, not instances, so get_text_extractor('noop'),
    get_table_extractor('noop'), and get_formula_extractor('noop') each
    construct a separate instance of this class.
    """

    def __init__(self, **_kwargs: Any) -> None:
        """Accept arbitrary kwargs from YAML adapter config; ignore all."""
        pass

    @property
    def tool_name(self) -> str:
        return "noop"

    def extract(self, image: Image, page_number: int) -> ElementContent:
        return ElementContent()


@register_figure_descriptor("noop")
class NoopDescriber:
    """Figure descriptor that always returns an empty string."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    @property
    def tool_name(self) -> str:
        return "noop"

    def describe(self, image: Image) -> str:
        return ""
```

- [ ] **Step 4: Register the module in `extraction/adapters/__init__.py`**

Replace the file contents with:

```python
"""Adapter modules for extraction roles.

Importing a module here triggers its decorator-based registration.
GPU-dependent modules (added in later stages) are wrapped in
try/except ImportError so CPU-only installs do not break.
"""
from __future__ import annotations

from . import noop  # noqa: F401
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
pytest extraction/tests/test_noop.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Lint and type-check**

```bash
ruff check extraction
mypy
```

Expected: both pass with no errors.

- [ ] **Step 7: Commit**

```bash
git add extraction/adapters/noop.py \
        extraction/adapters/__init__.py \
        extraction/tests/test_noop.py
git commit -m "feat(adapters): add noop adapter for text/table/formula/figure roles"
```

---

## Task 4: Implement the `pymupdf` renderer (TDD)

> Requires `extraction/tests/fixtures/sample_text.pdf` to be placed — see Prerequisites. Tests will SKIP (not fail) if it is missing, but the point of this task is to verify the renderer actually works.

**Files:**
- Test: `extraction/tests/test_pymupdf_renderer.py`
- Create: `extraction/adapters/pymupdf_renderer.py`
- Modify: `extraction/adapters/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `extraction/tests/test_pymupdf_renderer.py`:

```python
"""Tests for the pymupdf page renderer."""
from __future__ import annotations

from pathlib import Path

from PIL.Image import Image

from extraction.registry import get_renderer


def test_pymupdf_renderer_tool_name() -> None:
    assert get_renderer("pymupdf").tool_name == "pymupdf"


def test_pymupdf_renderer_page_count_positive(sample_text_pdf: Path) -> None:
    assert get_renderer("pymupdf").page_count(sample_text_pdf) >= 1


def test_pymupdf_renderer_render_page_returns_image(sample_text_pdf: Path) -> None:
    img = get_renderer("pymupdf").render_page(sample_text_pdf, 0)
    assert isinstance(img, Image)
    assert img.size[0] > 0 and img.size[1] > 0


def test_pymupdf_renderer_render_all_matches_page_count(sample_text_pdf: Path) -> None:
    r = get_renderer("pymupdf")
    images = r.render_all(sample_text_pdf)
    count = r.page_count(sample_text_pdf)
    assert len(images) == count
    for img in images:
        assert isinstance(img, Image)


def test_pymupdf_renderer_higher_dpi_larger_image(sample_text_pdf: Path) -> None:
    low = get_renderer("pymupdf", dpi=72).render_page(sample_text_pdf, 0)
    high = get_renderer("pymupdf", dpi=300).render_page(sample_text_pdf, 0)
    assert high.size[0] > low.size[0]
    assert high.size[1] > low.size[1]
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest extraction/tests/test_pymupdf_renderer.py -v
```

Expected: `test_pymupdf_renderer_tool_name` fails with `KeyError: Unknown renderer adapter 'pymupdf'`. If `sample_text.pdf` is present, the rest also fail with the same error. If it's missing, they skip.

- [ ] **Step 3: Implement the renderer**

Create `extraction/adapters/pymupdf_renderer.py`:

```python
"""PyMuPDF-based page renderer. CPU-only baseline for Stage 1."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image as PILImage
from PIL.Image import Image

from ..registry import register_renderer

_POINTS_PER_INCH = 72.0


@register_renderer("pymupdf")
class PyMuPDFRenderer:
    """Render PDF pages to PIL images at a configurable DPI."""

    def __init__(self, dpi: int = 150, **_kwargs: Any) -> None:
        self._dpi = int(dpi)

    @property
    def tool_name(self) -> str:
        return "pymupdf"

    def page_count(self, pdf_path: Path) -> int:
        doc = pymupdf.open(str(pdf_path))
        try:
            return int(doc.page_count)
        finally:
            doc.close()

    def render_page(self, pdf_path: Path, page_number: int) -> Image:
        doc = pymupdf.open(str(pdf_path))
        try:
            return self._render(doc[page_number])
        finally:
            doc.close()

    def render_all(self, pdf_path: Path) -> list[Image]:
        doc = pymupdf.open(str(pdf_path))
        try:
            return [self._render(page) for page in doc]
        finally:
            doc.close()

    def _render(self, page: Any) -> Image:
        zoom = self._dpi / _POINTS_PER_INCH
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pix.tobytes("png")
        # .copy() detaches from the BytesIO so the caller can use it freely.
        return PILImage.open(io.BytesIO(img_bytes)).copy()
```

- [ ] **Step 4: Register the module**

Update `extraction/adapters/__init__.py` to:

```python
"""Adapter modules for extraction roles.

Importing a module here triggers its decorator-based registration.
GPU-dependent modules (added in later stages) are wrapped in
try/except ImportError so CPU-only installs do not break.
"""
from __future__ import annotations

from . import noop  # noqa: F401
from . import pymupdf_renderer  # noqa: F401
```

- [ ] **Step 5: Run the tests and confirm they pass**

```bash
pytest extraction/tests/test_pymupdf_renderer.py -v
```

Expected: with `sample_text.pdf` in place, all 5 tests PASS. Without the fixture, `test_pymupdf_renderer_tool_name` still passes; the other 4 SKIP with "Fixture PDF not found".

- [ ] **Step 6: Lint and type-check**

```bash
ruff check extraction
mypy
```

Expected: both pass. If mypy complains about missing type stubs for `pymupdf`, proceed — Task 7 adds the override.

- [ ] **Step 7: Commit**

```bash
git add extraction/adapters/pymupdf_renderer.py \
        extraction/adapters/__init__.py \
        extraction/tests/test_pymupdf_renderer.py
git commit -m "feat(adapters): add pymupdf renderer"
```

---

## Task 5: Implement the `pymupdf_text` segmenter + text_extractor (TDD)

**Files:**
- Test: `extraction/tests/test_pymupdf_text_segmenter.py`
- Create: `extraction/adapters/pymupdf_text_segmenter.py`
- Modify: `extraction/adapters/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `extraction/tests/test_pymupdf_text_segmenter.py`:

```python
"""Tests for the pymupdf_text segmenter + text_extractor adapter.

pymupdf_text registers under two roles. When used as both the
segmenter AND text_extractor, the pipeline's tool_match optimization
reuses segmenter output directly — so extract() is normally not called.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage

from extraction.models import ElementType, Region
from extraction.registry import get_segmenter, get_text_extractor


def test_pymupdf_text_segmenter_tool_name() -> None:
    assert get_segmenter("pymupdf_text").tool_name == "pymupdf_text"


def test_pymupdf_text_registers_as_text_extractor() -> None:
    assert get_text_extractor("pymupdf_text").tool_name == "pymupdf_text"


def test_pymupdf_text_segmenter_returns_text_regions(sample_text_pdf: Path) -> None:
    regions = get_segmenter("pymupdf_text").segment(sample_text_pdf)
    assert len(regions) > 0
    for region in regions:
        assert isinstance(region, Region)
        assert region.region_type == ElementType.TEXT
        assert region.confidence == 1.0
        assert region.content is not None
        assert (region.content.text or "").strip() != ""


def test_pymupdf_text_segmenter_bbox_well_formed(sample_text_pdf: Path) -> None:
    regions = get_segmenter("pymupdf_text").segment(sample_text_pdf)
    for region in regions:
        x0, y0, x1, y1 = region.bbox
        assert x0 <= x1
        assert y0 <= y1


def test_pymupdf_text_segmenter_pages_within_doc(sample_text_pdf: Path) -> None:
    # Confirm page indices in regions are nonnegative and bounded.
    import pymupdf  # local import to avoid polluting module namespace

    regions = get_segmenter("pymupdf_text").segment(sample_text_pdf)
    doc = pymupdf.open(str(sample_text_pdf))
    try:
        total = doc.page_count
    finally:
        doc.close()
    for region in regions:
        assert 0 <= region.page < total


def test_pymupdf_text_extract_satisfies_protocol() -> None:
    # extract() is normally bypassed by tool_match; still needs to return
    # a valid ElementContent so the Protocol is satisfied.
    adapter = get_text_extractor("pymupdf_text")
    blank = PILImage.new("RGB", (10, 10))
    content = adapter.extract(blank, page_number=0)
    assert content is not None
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest extraction/tests/test_pymupdf_text_segmenter.py -v
```

Expected: tool_name and registration tests fail with `KeyError: Unknown segmenter adapter 'pymupdf_text'`. Others either fail or skip depending on fixture availability.

- [ ] **Step 3: Implement the adapter**

Create `extraction/adapters/pymupdf_text_segmenter.py`:

```python
"""PyMuPDF-based CPU segmenter. Emits one TEXT region per text block.

Registers under two roles:
  - segmenter: produces Region list with per-block text filled in.
  - text_extractor: the pipeline's tool_match optimization bypasses this
    when segmenter and text_extractor share tool_name — extract() is
    implemented to satisfy the Protocol but is not expected to be called
    in the Stage-1 default config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
from PIL.Image import Image

from ..models import ElementContent, ElementType, Region
from ..registry import register_segmenter, register_text_extractor

# get_text("blocks") returns 7-tuples. Block type 0 means text, 1 means image.
_BLOCK_TYPE_TEXT = 0


@register_segmenter("pymupdf_text")
@register_text_extractor("pymupdf_text")
class PyMuPDFTextSegmenter:
    """Emit one TEXT region per PyMuPDF text block; fill region.content.text."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    @property
    def tool_name(self) -> str:
        return "pymupdf_text"

    def segment(self, pdf_path: Path) -> list[Region]:
        regions: list[Region] = []
        doc = pymupdf.open(str(pdf_path))
        try:
            for page_number, page in enumerate(doc):
                for block in page.get_text("blocks"):
                    # block: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(block) < 7:
                        continue
                    x0, y0, x1, y1, text, _block_no, block_type = block
                    if block_type != _BLOCK_TYPE_TEXT:
                        continue
                    cleaned = (text or "").strip()
                    if not cleaned:
                        continue
                    regions.append(
                        Region(
                            page=page_number,
                            bbox=[float(x0), float(y0), float(x1), float(y1)],
                            region_type=ElementType.TEXT,
                            confidence=1.0,
                            content=ElementContent(text=cleaned),
                        )
                    )
        finally:
            doc.close()
        return regions

    def extract(self, page_image: Image, page_number: int) -> ElementContent:
        # Not normally called: the pipeline's tool_match reuses segment()
        # output when text_extractor and segmenter share a tool_name.
        # Return empty content to satisfy the Protocol if tool_match ever
        # does not apply.
        return ElementContent()
```

- [ ] **Step 4: Register the module**

Update `extraction/adapters/__init__.py` to:

```python
"""Adapter modules for extraction roles.

Importing a module here triggers its decorator-based registration.
GPU-dependent modules (added in later stages) are wrapped in
try/except ImportError so CPU-only installs do not break.
"""
from __future__ import annotations

from . import noop  # noqa: F401
from . import pymupdf_renderer  # noqa: F401
from . import pymupdf_text_segmenter  # noqa: F401
```

- [ ] **Step 5: Run tests and confirm pass**

```bash
pytest extraction/tests/test_pymupdf_text_segmenter.py -v
```

Expected: tool_name + registration + extract tests PASS. Segment tests PASS if `sample_text.pdf` is in place; SKIP otherwise.

- [ ] **Step 6: Lint and type-check**

```bash
ruff check extraction
mypy
```

Expected: both pass.

- [ ] **Step 7: Commit**

```bash
git add extraction/adapters/pymupdf_text_segmenter.py \
        extraction/adapters/__init__.py \
        extraction/tests/test_pymupdf_text_segmenter.py
git commit -m "feat(adapters): add pymupdf_text segmenter + text_extractor"
```

---

## Task 6: End-to-end CPU pipeline smoke test

**Files:**
- Test: `extraction/tests/test_pipeline_e2e.py`

- [ ] **Step 1: Write the E2E test**

Create `extraction/tests/test_pipeline_e2e.py`:

```python
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
```

- [ ] **Step 2: Run the E2E test**

```bash
pytest extraction/tests/test_pipeline_e2e.py -v
```

Expected: both tests PASS with a fixture PDF in place; SKIP otherwise.

- [ ] **Step 3: Lint and type-check**

```bash
ruff check extraction
mypy
```

Expected: both pass.

- [ ] **Step 4: Commit**

```bash
git add extraction/tests/test_pipeline_e2e.py
git commit -m "test: add CPU pipeline end-to-end smoke test"
```

---

## Task 7: `pyproject.toml` cleanup

**Files:**
- Modify: `pyproject.toml`

Three changes:
1. Exclude `extraction.tests*` from the wheel (currently bundled).
2. Add `uv` to the `[dev]` extras for lock regeneration.
3. Add `pymupdf.*` to the mypy `ignore_missing_imports` list.

- [ ] **Step 1: Edit `pyproject.toml`**

Apply these edits (shown as diffs for clarity; use the Edit tool or hand-edit):

Change 1 — the `[tool.setuptools.packages.find]` section:

```toml
# before
exclude = ["outputs*", "_archive*", "embedding*", "indexing*", "tests*", "tasks*"]

# after
exclude = [
    "outputs*",
    "_archive*",
    "embedding*",
    "indexing*",
    "extraction.tests*",
    "tests*",
    "tasks*",
]
```

Change 2 — the `dev` extras:

```toml
# before
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
    "mypy",
]

# after
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
    "mypy",
    "uv",
]
```

Change 3 — the mypy override block for third-party libs missing stubs:

```toml
# before
[[tool.mypy.overrides]]
module = ["mineru.*", "transformers.*", "torch.*", "bs4.*", "fitz.*"]
ignore_missing_imports = true

# after
[[tool.mypy.overrides]]
module = ["mineru.*", "transformers.*", "torch.*", "bs4.*", "fitz.*", "pymupdf.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Run the full test + lint + type-check gate**

```bash
pytest -q
ruff check extraction
mypy
```

Expected: all pass.

- [ ] **Step 3: Verify wheel-build does not bundle tests**

```bash
pip wheel --no-deps -w /tmp/wheelhouse .
unzip -l /tmp/wheelhouse/techpdfparser-0.1.0-py3-none-any.whl | grep -E "tests|conftest" || echo "no tests in wheel — good"
rm -rf /tmp/wheelhouse
```

Expected: `no tests in wheel — good` prints. Grep returns no matches (exit code 1), so the `|| echo` fires.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "pyproject: exclude tests from wheel, add uv dev dep, pymupdf mypy override"
```

---

## Task 8: `config_cpu.yaml` at repo root

**Files:**
- Create: `config_cpu.yaml`

- [ ] **Step 1: Create the config file**

Create `config_cpu.yaml` at the repo root (same directory as `pyproject.toml`):

```yaml
# Stage-1 CPU-only config. No GPU adapters required.
# Reuses the segmenter for text extraction via tool_match; other roles are noop.
extraction:
  renderer: pymupdf
  segmenter: pymupdf_text
  text_extractor: pymupdf_text   # tool_match: pipeline reuses segmenter content
  table_extractor: noop
  formula_extractor: noop
  figure_descriptor: noop
  output_dir: outputs
  confidence_threshold: 0.3
  dpi: 150
```

- [ ] **Step 2: Verify the config loads cleanly**

```bash
python -c "from pathlib import Path; from extraction.config import load_extraction_config; cfg = load_extraction_config(Path('config_cpu.yaml')); print(cfg.model_dump())"
```

Expected output (order may differ):
```
{'renderer': 'pymupdf', 'segmenter': 'pymupdf_text', 'text_extractor': 'pymupdf_text', 'table_extractor': 'noop', 'formula_extractor': 'noop', 'figure_descriptor': 'noop', 'output_dir': 'outputs', 'confidence_threshold': 0.3, 'dpi': 150, 'adapters': {}}
```

- [ ] **Step 3: Commit**

```bash
git add config_cpu.yaml
git commit -m "config: add CPU-only extraction config"
```

---

## Task 9: Lock file and `Makefile`

**Files:**
- Create: `Makefile`
- Create: `requirements/cpu.lock`

- [ ] **Step 1: Create the `Makefile`**

Create `Makefile` at the repo root:

```makefile
# Regenerate lock files from pyproject.toml.
# Run after editing dependency declarations; commit the resulting *.lock files.

.PHONY: lock lock-cpu lock-gpu test lint type check

lock: lock-cpu

lock-cpu:
	uv pip compile pyproject.toml --extra dev -o requirements/cpu.lock

lock-gpu:
	uv pip compile pyproject.toml --extra dev --extra gpu -o requirements/gpu.lock

test:
	pytest -q

lint:
	ruff check extraction

type:
	mypy

check: lint type test
```

- [ ] **Step 2: Generate `requirements/cpu.lock`**

```bash
mkdir -p requirements
uv pip compile pyproject.toml --extra dev -o requirements/cpu.lock
```

Expected: a file `requirements/cpu.lock` is created, containing pinned versions for `pydantic`, `pyyaml`, `pillow`, `pymupdf`, `pytest`, `pytest-cov`, `ruff`, `mypy`, `uv`, and their transitive dependencies. Each line looks like `package==X.Y.Z`.

- [ ] **Step 3: Verify an install from the lock works**

In a clean temporary venv:

```bash
python -m venv /tmp/lockcheck-venv
/tmp/lockcheck-venv/bin/pip install -q -r requirements/cpu.lock
/tmp/lockcheck-venv/bin/pip install -q -e .
/tmp/lockcheck-venv/bin/python -c "import extraction.adapters; print(sorted(extraction.adapters.__dict__.keys()))"
rm -rf /tmp/lockcheck-venv
```

Expected: final line prints a list including `'noop'`, `'pymupdf_renderer'`, `'pymupdf_text_segmenter'` (among dunder attributes).

- [ ] **Step 4: Commit**

```bash
git add Makefile requirements/cpu.lock
git commit -m "build: add Makefile lock target and initial requirements/cpu.lock"
```

---

## Task 10: GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

      - name: Install uv
        run: python -m pip install uv

      - name: Install from cpu.lock + editable project
        run: uv pip install --system -r requirements/cpu.lock -e .

      - name: Lint
        run: ruff check extraction

      - name: Type check
        run: mypy

      - name: Tests (CPU, integration marker excluded)
        run: pytest -q
```

- [ ] **Step 2: Verify YAML syntax locally**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

Expected: exits cleanly, no output.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions CI (CPU tests, lint, type-check on Python 3.10 and 3.12)"
```

---

## Task 11: Pull request template

**Files:**
- Create: `.github/pull_request_template.md`

- [ ] **Step 1: Create the template**

Create `.github/pull_request_template.md`:

```markdown
## Summary

<what this PR changes and why>

## Test plan

- [ ] `pytest -q` passes locally
- [ ] `ruff check extraction` passes
- [ ] `mypy` passes
- [ ] `pytest -m integration` passes locally (if touching GPU adapters)
- [ ] CI is green on the PR

## Related

<spec / plan / issue references, if any>
```

- [ ] **Step 2: Commit**

```bash
git add .github/pull_request_template.md
git commit -m "chore: add pull request template"
```

---

## Task 12: Final verification, push, and PR

- [ ] **Step 1: Run the full local gate**

```bash
pytest -q
ruff check extraction
mypy
```

Expected: all pass.

- [ ] **Step 2: Manual CLI verification (requires fixture PDF in place)**

```bash
rm -rf outputs/t1
python -m extraction extract extraction/tests/fixtures/sample_text.pdf \
    --config config_cpu.yaml \
    --output outputs/t1
```

Expected stdout (numbers vary with the fixture):

```
Extracting sample_text.pdf...
  Elements: <N>
  Pages:    <M>
  Output:   outputs/t1
```

Confirm artifacts:

```bash
ls outputs/t1/
cat outputs/t1/content_list.json | python -c "import json, sys; d = json.load(sys.stdin); print('schema_version=', d['schema_version'], 'elements=', len(d['elements']))"
```

Expected: `content_list.json`, `segmentation.json`, and a `pages/` directory exist; the Python one-liner prints `schema_version= 1.0 elements= <N>`.

Clean up the verification output:

```bash
rm -rf outputs/t1
```

- [ ] **Step 3: Push dev**

```bash
git push origin dev
```

Expected: push succeeds; GitHub Actions CI starts automatically for the `dev` branch.

- [ ] **Step 4: Verify CI passes**

Watch the Actions tab on GitHub (or):

```bash
gh run list --branch dev --limit 1
gh run watch <run-id-from-above>
```

Expected: workflow completes with conclusion `success` for both Python 3.10 and 3.12 jobs.

- [ ] **Step 5: Open a PR from `dev` to `main`**

```bash
gh pr create \
    --base main --head dev \
    --title "Stage 1: CPU foundation adapters + CI + lock" \
    --body "$(cat <<'EOF'
## Summary

Implements Stage 1 of the GPU-adapters plan
(`docs/superpowers/specs/2026-04-22-gpu-adapters-design.md`,
 `docs/superpowers/plans/2026-04-22-stage-1-cpu-foundation.md`):

- `pymupdf` renderer, `pymupdf_text` segmenter + text_extractor, and
  `noop` adapters for the other three roles.
- Test scaffolding with shared fixtures and a CPU-only end-to-end smoke test.
- GitHub Actions CI running lint, type-check, and CPU tests on Python 3.10 and 3.12.
- Initial `requirements/cpu.lock` plus a `Makefile` target for regeneration.
- PR template and pyproject.toml cleanup (exclude tests from wheel, add `uv` to [dev]).

After merge, the CLI runs end-to-end on CPU and emits a schema-valid
content_list.json. GPU stages (mineru25, olmocr2, qwen25vl) follow in
separate PRs per the plan.

## Test plan

- [x] \`pytest -q\` passes locally
- [x] \`ruff check extraction\` passes
- [x] \`mypy\` passes
- [x] Manual CLI run succeeds on fixture PDF
- [x] CI is green on this PR
EOF
)"
```

- [ ] **Step 6: In the parent `DocumentAnalysis` repo, bump the submodule pointer after PR merge**

(After the PR is merged into `techpdfparser`'s `main`.)

```bash
cd /home/ktazefid/Documents/projects/DocumentAnalysis
cd techpdfparser
git checkout main
git pull --ff-only origin main
cd ..
git add techpdfparser
git commit -m "Bump techpdfparser to Stage 1 (CPU foundation)"
git push origin main
```

Expected: the parent `DocumentAnalysis` repo records the new submodule commit on its `main` branch.

---

## Self-review

### Spec coverage check

Walked through `docs/superpowers/specs/2026-04-22-gpu-adapters-design.md` section by section for Stage 1 deliverables:

| Spec item | Implemented in |
|---|---|
| `extraction/adapters/__init__.py` with GPU import guards | Task 1 initial skeleton; Task 5 adds all three CPU modules. (GPU `try/except` blocks will be added in Stages 2–4 when those modules exist.) |
| `pymupdf_renderer.py` | Task 4 |
| `pymupdf_text_segmenter.py` (registers as both segmenter and text_extractor) | Task 5 |
| `noop.py` for table, formula, figure | Task 3 |
| `extraction/tests/` scaffolding + `conftest.py` | Task 1 |
| `fixtures/sample_text.pdf` (user-supplied) | Prerequisites; used in Tasks 4–6 |
| `test_pipeline_e2e.py` (CPU config) | Task 6 |
| `test_pymupdf_renderer.py`, `test_pymupdf_text_segmenter.py`, `test_noop.py` | Tasks 3–5 |
| `config_cpu.yaml` at repo root | Task 8 |
| `requirements/cpu.lock` | Task 9 |
| `.github/workflows/ci.yml` | Task 10 |
| `.github/pull_request_template.md` | Task 11 |
| `.gitignore` update for fixture PDFs | Task 2 |
| `pyproject.toml` cleanup: exclude tests from wheel | Task 7 |
| `pyproject.toml`: add `uv` to `[dev]` | Task 7 |
| `pyproject.toml`: add `pymupdf.*` to mypy ignore | Task 7 |

Exit criteria from the spec:
- [x] `python -m extraction extract ... --config config_cpu.yaml ...` runs to completion — Task 12 Step 2.
- [x] `content_list.json` validates against `ContentList` schema and has ≥1 text element with non-empty text — Tasks 6 and 12.
- [x] `pytest -q` passes without GPU — Task 12 Step 1.
- [x] `ruff check extraction` and `mypy` pass — Task 12 Step 1.
- [x] GitHub Actions CI passes — Task 12 Step 4.

### Placeholder scan

No `TBD`, `TODO`, `implement later`, or "similar to Task N" references. Every code step has full code; every command step has the exact command and expected output.

One intentional user-action in the plan: the fixture PDF placement is listed under Prerequisites, not as a task step. If it is missing when Tasks 4–6 run, their fixture-dependent tests SKIP (not fail) so the gate stays green.

### Type consistency

Names, signatures, and types cross-referenced between tasks:

- `PyMuPDFRenderer.__init__(self, dpi: int = 150, **_kwargs)` — same default 150 appears in `config_cpu.yaml`.
- `PyMuPDFTextSegmenter.segment(self, pdf_path: Path) -> list[Region]` — matches `Segmenter` Protocol in `extraction/interfaces.py:31`.
- `PyMuPDFTextSegmenter.extract(self, page_image: Image, page_number: int) -> ElementContent` — matches `TextExtractor` Protocol in `extraction/interfaces.py:40`.
- `NoopExtractor.extract(self, image: Image, page_number: int) -> ElementContent` — parameter name `image` rather than `page_image`/`region_image` because the class stands in for three different Protocols. The Protocol is structural; parameter names differ across the Text/Table/Formula Protocols anyway (all are `Image, int -> ElementContent`).
- `NoopDescriber.describe(self, image: Image) -> str` — matches `FigureDescriptor` Protocol in `extraction/interfaces.py:67`.
- Registry keys used in tests match registration strings: `"noop"`, `"pymupdf"`, `"pymupdf_text"`.
- `config_cpu.yaml` names match the strings used in registration decorators and in `_stage1_cpu_config()` in the E2E test.

### Scope check

Stage 1 is a single cohesive unit: make the CLI run end-to-end on CPU with CI and a lock file. Stages 2–4 are explicitly deferred to their own plans, to be drafted just before each stage starts. This plan contains everything needed for one mergeable PR and nothing more.
