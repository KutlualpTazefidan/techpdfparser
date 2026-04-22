# GPU-first extraction adapters for techpdfparser

**Date:** 2026-04-22
**Status:** Approved — ready for implementation planning
**Author:** ktazefid

---

## Summary

`techpdfparser` has a complete pipeline architecture (registry, Protocols, VRAM-aware
lifecycle, config system) but no adapter implementations — `extraction/adapters/` does
not exist, so `python -m extraction extract` fails at startup with
`ModuleNotFoundError`. This spec designs the adapter layer: six adapter modules
covering all six role Protocols, delivered in four incremental PRs against `dev`. Each
stage lands a fully runnable system; no stage leaves the repository broken. GPU
adapters use a lazy-load + `unload()` pattern that plugs into the pipeline's existing
`release_runtime_resources()` calls, ensuring at most one GPU model occupies VRAM at
any time on the user's 24 GB RTX PRO 6000 Blackwell.

## Context

### What exists in `extraction/`

- `pipeline.py` — orchestrates per-role extraction; explicitly releases each role's
  GPU state after it finishes (`pipeline.py:84` for segmenter;
  `pipeline.py:197` for each extractor role).
- `interfaces.py` — six `Protocol` classes defining adapter contracts
  (`PageRenderer`, `Segmenter`, `TextExtractor`, `TableExtractor`,
  `FormulaExtractor`, `FigureDescriptor`).
- `registry.py` — decorator-based `register_*` / `get_*` factory pairs for each role.
- `models.py` — Pydantic schemas for `Region`, `Element`, `ElementContent`,
  `ContentList`.
- `config.py` — YAML loading, per-adapter config forwarded as `**kwargs`.
- `_runtime.py` — `release_runtime_resources()` helper that calls
  `adapter.unload()` if present, then `gc.collect()` + `torch.cuda.empty_cache()`.
- `__main__.py` — CLI with `extract` and `rebuild` subcommands. Imports
  `extraction.adapters` on line 10 to trigger adapter registration (currently
  the source of the `ModuleNotFoundError`).
- `docs/architecture.md`, `docs/extraction_output.md`, `docs/principles.md`,
  `docs/writing_adapters.md`.

### What is missing

- `extraction/adapters/` — does not exist. No renderer, segmenter, or extractor
  implementations registered.
- `extraction/tests/` — does not exist.

### Hardware and constraints

- NVIDIA RTX PRO 6000 Blackwell, 24 GB VRAM visible (likely vGPU slice), CUDA 13.0.
- All models must run locally — no external APIs.
- At most one GPU model may be resident in VRAM at any moment (user constraint).
- A vLLM process may be running outside of extraction jobs; it is stopped before
  `python -m extraction extract` is invoked, so the full 24 GB is available.

## Goals

- Implement the six-role adapter set so `python -m extraction extract <pdf>` runs
  end-to-end and produces a complete, schema-valid `content_list.json`.
- Honor the existing lifecycle: lazy-load + `unload()` called between roles by the
  pipeline.
- Deliver in four incremental PRs against `dev`, each merging a working system.
- Ship GitHub Actions CI (CPU tests only) from Stage 1.
- Ship lock files (`requirements/cpu.lock`, `requirements/gpu.lock`) generated
  with `uv pip compile` for reproducible installs across machines.
- Final state supports swapping any adapter in place by changing the YAML config —
  no pipeline code changes needed to adopt future / alternative adapter
  implementations.

## Non-goals

- Refactoring existing `extraction/*.py` modules beyond the minimum required
  (config / dependency updates; no interface changes).
- Fixing the design flaw in `TextExtractor.extract(page_image, page_number)` (no
  region bbox passed; redundant inference across text regions). Papered over with
  per-page caching inside the adapter; a proper refactor is deferred.
- CI on GPU runners. Requires self-hosted infrastructure; scoped out.
- Adapter quality audits per `backlog.md`. Separate, later work.
- Implementing `embedding/` or `indexing/` layers. Stubs remain stubs.

## Design

### 1. Directory and file structure

```
extraction/
├── adapters/
│   ├── __init__.py                   # imports each module below; GPU modules
│   │                                 # wrapped in try/except ImportError
│   ├── pymupdf_renderer.py           # "pymupdf" — CPU, stateless
│   ├── pymupdf_text_segmenter.py     # "pymupdf_text" — CPU, stateless;
│   │                                 # registers as both segmenter AND
│   │                                 # text_extractor so tool_match reuses
│   │                                 # output for free
│   ├── noop.py                       # "noop" — stateless; registers for
│   │                                 # table, formula, and figure roles
│   ├── mineru25.py                   # "mineru25" — GPU; one class, two
│   │                                 # decorators (segmenter + table_extractor)
│   ├── olmocr2.py                    # "olmocr2" — GPU, text_extractor
│   └── qwen25vl.py                   # "qwen25vl" — GPU, figure_descriptor
└── tests/
    ├── __init__.py
    ├── conftest.py                   # shared fixtures (cuda_clean, tmp_output_dir)
    ├── fixtures/
    │   ├── sample_text.pdf           # real PDF, text-heavy, small
    │   └── sample_with_tables.pdf    # real PDF with tables/figures
    ├── test_pymupdf_renderer.py
    ├── test_pymupdf_text_segmenter.py
    ├── test_noop.py
    ├── test_mineru25.py              # @pytest.mark.integration
    ├── test_olmocr2.py               # @pytest.mark.integration
    ├── test_qwen25vl.py              # @pytest.mark.integration
    └── test_pipeline_e2e.py          # full pipeline smoke, CPU config
```

**Key structural decisions and rationale:**

- **`mineru25.py` registers two roles from one class.** One `MinerU25` class
  carries both `@register_segmenter("mineru25")` and
  `@register_table_extractor("mineru25")` decorators. This preserves the
  existing `tool_match` optimization in `pipeline._extract_region` — when
  segmenter and table extractor share a `tool_name`, segmenter output is reused
  for tables without a second model load.
- **`__init__.py` wraps GPU-dependent imports in `try/except ImportError`.**
  A user installing only the base package (no `[gpu]` extra) gets working CPU
  adapters; attempts to use a GPU adapter raise a clear error from the registry
  ("adapter not found, did you install `[gpu]`?") instead of an opaque
  `ImportError` at CLI startup.
- **Integration tests are gated by `pytest -m integration`.** Matches the
  existing configuration in `pyproject.toml` (already has the marker defined
  and `-m 'not integration'` as the default `addopts`).
- **`conftest.py` provides `cuda_clean`** — a fixture that calls
  `torch.cuda.empty_cache()` between tests to prevent VRAM bleed from affecting
  subsequent tests when the full `pytest -m integration` suite runs.

### 2. Adapter module skeleton

Every GPU adapter follows the same shape. Template using `mineru25` as the example:

```python
# extraction/adapters/mineru25.py
from __future__ import annotations
from pathlib import Path
from typing import Any

from ..models import ElementContent, ElementType, Region
from ..registry import register_segmenter, register_table_extractor


@register_segmenter("mineru25")
@register_table_extractor("mineru25")
class MinerU25:
    TOOL_NAME = "mineru25"

    def __init__(self, **adapter_config: Any) -> None:
        self._config = adapter_config
        self._model: Any | None = None                # lazy — not loaded in __init__
        self._pdf_cache: dict[str, list[Region]] = {} # avoid re-running on same PDF

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    # --- lifecycle ---------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Heavy imports happen HERE, not at module import.
        import torch  # noqa: F401
        from mineru.backend.pipeline import PipelineAnalyze
        self._model = PipelineAnalyze(**self._config)

    def unload(self) -> None:
        """Drop the model reference; pipeline calls this between roles."""
        self._model = None
        self._pdf_cache.clear()

    # --- role methods ------------------------------------------------
    def segment(self, pdf_path: Path) -> list[Region]:
        self._ensure_loaded()
        key = str(pdf_path.resolve())
        if key in self._pdf_cache:
            return self._pdf_cache[key]
        raw = self._model.analyze(str(pdf_path))
        regions = self._to_regions(raw)
        self._pdf_cache[key] = regions
        return regions

    def extract(self, region_image, page_number: int) -> ElementContent:
        # Called for tables. Pipeline's tool_match optimization bypasses this
        # when segmenter and table_extractor share a tool_name. Implemented
        # for the case where they don't.
        self._ensure_loaded()
        ...

    def _to_regions(self, raw: Any) -> list[Region]:
        ...
```

**Properties enforced by this shape:**

- **Lazy loading.** `__init__` is cheap; no VRAM touched until the first real
  call. Instantiating an adapter you never use costs ~nothing.
- **Heavy imports deferred to `_ensure_loaded`.** `extraction/adapters/__init__.py`
  does not blow up at import time on a machine without PyTorch.
- **`unload()` is the pipeline's contract.** `extraction/_runtime.py:9-18`
  already calls `adapter.unload()` duck-typed; no Protocol extension needed.
- **Per-PDF / per-page caching.** Papers over the `TextExtractor.extract`
  design flaw without refactoring the pipeline interface.

**CPU adapters** (`pymupdf_renderer`, `pymupdf_text_segmenter`, `noop`) follow
the same module shape minus the lazy load and `unload` — they are lightweight
enough to do real work in `__init__`.

### 3. Stage-by-stage delivery plan

Each stage is its own PR into `dev`, landing a working system. After each merge
to `main` in the `techpdfparser` repo, the parent `DocumentAnalysis` submodule
pointer is bumped in a follow-up commit.

#### Stage 1 — CPU foundation, CI, and lock file (~½ day)

Ships:
- `extraction/adapters/__init__.py` (with GPU import guards)
- `extraction/adapters/pymupdf_renderer.py`
- `extraction/adapters/pymupdf_text_segmenter.py` (registers as segmenter AND
  text_extractor — the `tool_match` optimization means text role reuses
  segmenter output for free)
- `extraction/adapters/noop.py` (table / formula / figure roles)
- `extraction/tests/` scaffolding (`__init__.py`, `conftest.py`,
  `fixtures/sample_text.pdf` — provided by user)
- `extraction/tests/test_pipeline_e2e.py` (full pipeline, CPU config)
- `extraction/tests/test_pymupdf_renderer.py`,
  `test_pymupdf_text_segmenter.py`, `test_noop.py`
- `config_cpu.yaml` at repo root
- `requirements/cpu.lock` generated via `uv pip compile`
- `.github/workflows/ci.yml`
- `.github/pull_request_template.md`
- `.gitignore` update adding `!extraction/tests/fixtures/*.pdf`
- `pyproject.toml` cleanup: exclude pattern change so tests are not bundled
  into the wheel (see §4).

Exit criteria:
- `python -m extraction extract extraction/tests/fixtures/sample_text.pdf
  --config config_cpu.yaml --output outputs/t1/` runs to completion.
- `outputs/t1/content_list.json` validates against `ContentList` schema and
  contains one or more text elements with non-empty `content.text`.
- `pytest -q` passes (no GPU needed).
- `ruff check extraction` and `mypy` pass.
- GitHub Actions CI passes on a pushed branch.

What you have after Stage 1: complete CPU-only pipeline for text-only PDFs;
full CI gate; all Stage 2–4 work builds on verified plumbing.

#### Stage 2 — MinerU 2.5 segmenter + table extractor (~1–2 days)

Ships:
- `extraction/adapters/mineru25.py`
- `extraction/tests/test_mineru25.py` (marked `@pytest.mark.integration`)
- `extraction/tests/fixtures/sample_with_tables.pdf` (provided by user)
- `requirements/gpu.lock` (new file, first GPU stage)
- `config_s2.yaml` (mineru25 segmenter + mineru25 tables; noop for
  text/formula/figure)
- `pyproject.toml` — verify `mineru>=2.5` and `torch>=2.3` pins against
  MinerU's current requirements; tighten if needed.

Exit criteria:
- With `config_s2.yaml`, pipeline produces regions including tables with
  non-empty `content.markdown`.
- Integration test `test_mineru25.py` passes on the GPU machine.
- Integration test asserts `adapter._model is not None` during use and
  `is None` after `release_runtime_resources(adapter)`.

#### Stage 3 — olmOCR-2 text extractor (~1–2 days)

Ships:
- `extraction/adapters/olmocr2.py` (per-page caching inside)
- `extraction/tests/test_olmocr2.py` (marked `@pytest.mark.integration`)
- `config_s3.yaml` (mineru25 segmenter/tables + olmocr2 text)
- `pyproject.toml` — bump `transformers>=4.49`; add `accelerate>=0.30`.
- `requirements/gpu.lock` regenerated.

Exit criteria:
- With `config_s3.yaml`, text regions contain real OCR text via olmOCR-2.
- VRAM trace in integration test shows: mineru loaded → unloaded → olmocr
  loaded → unloaded, never both simultaneously.

#### Stage 4 — Qwen2.5-VL figure descriptor, polish (~1–2 days)

Ships:
- `extraction/adapters/qwen25vl.py` (default
  `Qwen/Qwen2.5-VL-7B-Instruct`; 3B variant configurable via YAML)
- `extraction/tests/test_qwen25vl.py` (marked `@pytest.mark.integration`)
- `config_gpu.yaml` — final full-GPU config
- `extraction/__main__.py` — new `prefetch` subcommand that calls
  `_ensure_loaded()` on every configured adapter without running extraction
  (for first-time setup / air-gapped machines).
- `docs/troubleshooting.md` — documented failure modes and recovery steps.
- `README.md` — updated installation / runtime-expectation section.
- `pyproject.toml` — add `qwen-vl-utils>=0.0.8`, possibly `torchvision>=0.18`.
- `requirements/gpu.lock` regenerated.

Exit criteria:
- Full stack (`pymupdf` + `mineru25` + `olmocr2` + `qwen25vl`) runs
  end-to-end on `sample_with_tables.pdf` within 24 GB VRAM without OOM.
- Figures/diagrams emerge with non-empty `content.description`.
- `python -m extraction prefetch --config config_gpu.yaml` completes
  offline-safely after one online run.

### 4. Dependencies and lock files

Current `pyproject.toml` is well-organized. Stage 1 adds no runtime deps; later
stages tighten pins and add a few helpers.

#### Final `[project.optional-dependencies]` after Stage 4

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
    "mypy",
    "uv",                # for regenerating lock files
]
gpu = [
    "mineru>=2.5",
    "transformers>=4.49",
    "torch>=2.3",
    "torchvision>=0.18",
    "accelerate>=0.30",
    "qwen-vl-utils>=0.0.8",
    "beautifulsoup4>=4.12",
]
```

#### Stage 1 `pyproject.toml` cleanup

`[tool.setuptools.packages.find]` currently excludes `tests*` but not
`extraction.tests*`, so test files get bundled into the wheel. Fix:

```toml
[tool.setuptools.packages.find]
include = ["extraction*"]
exclude = [
    "outputs*", "_archive*", "embedding*", "indexing*",
    "extraction.tests*", "tests*", "tasks*",
]
```

#### Lock files

Two files, generated with `uv pip compile`:

- `requirements/cpu.lock` — base + `[dev]` extras. Covers Stage 1
  development and CI; anyone without a GPU installs from this.
- `requirements/gpu.lock` — base + `[dev]` + `[gpu]` extras. Full stack;
  pins CUDA-specific torch wheel to the user's environment (CUDA 13.0).

Regeneration (manual, not in CI):

```bash
uv pip compile pyproject.toml --extra dev --output-file requirements/cpu.lock
uv pip compile pyproject.toml --extra dev --extra gpu --output-file requirements/gpu.lock
```

A `Makefile` target or a short shell script wraps this; details in Stage 1 PR.

#### Install recipes (documented in README)

```bash
# CPU-only — for Stage 1 development or collaborators without a GPU
uv pip install -r requirements/cpu.lock -e .

# Full GPU — for running the full pipeline
uv pip install -r requirements/gpu.lock -e .
# Also install a CUDA-matched torch wheel:
# https://pytorch.org/get-started/locally/
```

**CUDA caveat:** `gpu.lock` pins the torch wheel to CUDA 13.0 (the user's
environment). Collaborators on different CUDA versions must regenerate from
their own environment.

#### Pin strategy

Lower-bound (`>=X.Y`) only in `pyproject.toml`; exact pins live in the lock
files. Upgrades are explicit and batched (edit `pyproject.toml`, regenerate
both locks).

### 5. Model weights

Three dimensions to pin down: weight sources, cache location, download strategy.

#### Sources and disk footprint

| Adapter | Model | Fetched from | Approx disk |
|---|---|---|---|
| `mineru25` | Bundled by `mineru` package | Auto-downloaded by `mineru` on first use | ~5–8 GB |
| `olmocr2` | `allenai/olmOCR-2-7B-*` (exact revision pinned in adapter config) | HuggingFace Hub | ~14 GB fp16 |
| `qwen25vl` | `Qwen/Qwen2.5-VL-7B-Instruct` (default) or `-3B-Instruct` (configurable) | HuggingFace Hub | 7B: ~16 GB / 3B: ~7 GB |
| **Total** | | | **~35–40 GB** |

#### Cache location

HuggingFace models go to the standard HF cache:

```
$HF_HOME/hub/    (defaults to ~/.cache/huggingface/hub)
```

MinerU uses its own cache (per that tool's convention). Not overridden — keeps
each tool's `huggingface-cli`/`mineru` inspection commands working as expected.

The user is recommended (not required) to set `HF_HOME` in `~/.bashrc` if
weights should live on a specific fast disk (e.g., an NVMe separate from
`~/.cache`). Documented in README.

#### Download strategy

**Lazy, on first use.** No download at `pip install`. Weights fetch on the
first `_ensure_loaded()` call, with visible progress bars. Subsequent loads
are cache hits.

#### Offline / pre-staged mode

- Standard HuggingFace env var `HF_HUB_OFFLINE=1` works — adapters raise a
  clear error if weights are not cached and offline mode is forced.
- Stage 4 adds `python -m extraction prefetch --config <yaml>` — calls
  `_ensure_loaded()` on every configured adapter without running extraction.
  Good for first-time machine setup or warming a cache before air-gapping.

#### Revision pinning

Default model IDs in adapter config are pinned to exact HuggingFace revisions:

```python
# extraction/adapters/qwen25vl.py
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_REVISION = "<pinned-hash>"   # exact git commit on HF Hub
```

Overridable per adapter via YAML (`adapters.qwen25vl.model_id`,
`adapters.qwen25vl.revision`). The model equivalent of the lock file.

### 6. Testing strategy

#### Test kinds

| Kind | Runs on | What it checks |
|---|---|---|
| Unit (per adapter) | `pytest -q` (no GPU) | Protocol conformance, registration, `tool_name`, lazy-load behavior (mocked). |
| CPU smoke E2E | `pytest -q` | Full `ExtractionPipeline` with Stage 1 adapters against a fixture PDF. Validates `content_list.json` schema, text element presence, role-order of `_extract_by_role` calls. |
| GPU integration | `pytest -m integration` | Actually loads model, runs on fixture, asserts non-empty output. Asserts `_model is not None` mid-run and `is None` after `release_runtime_resources()`. |
| VRAM / lifecycle | `pytest -m integration` (could add `vram` marker) | Measures `torch.cuda.memory_allocated()` before/during/after unload; asserts it drops by at least the expected model size. |
| Schema | `pytest -q` | Every fixture sidecar and `content_list.json` validates against Pydantic models. |

#### Fixture PDFs

User-supplied, committed to `extraction/tests/fixtures/`:

- `sample_text.pdf` — small text-heavy PDF (<~1 MB) for Stage 1 smoke E2E and
  Stage 3 olmOCR integration tests.
- `sample_with_tables.pdf` — contains tables and figures for Stage 2 and
  Stage 4 integration tests.

`!extraction/tests/fixtures/*.pdf` added to `.gitignore` to unblock commit.

#### Determinism

OCR and VLM outputs are not byte-deterministic across runs or hardware.
Integration tests assert **structural** properties only:

- Element count in plausible range.
- Expected element types present (text, table, figure — depending on PDF).
- Text / markdown / description fields non-empty.
- Schema valid.

**Golden-file snapshot testing is deliberately avoided.** Fragile across model
versions and GPU hardware; high maintenance cost for little value here.

#### `conftest.py` fixtures

- `cuda_clean` — session-level, calls `torch.cuda.empty_cache()` between
  tests to prevent VRAM bleed.
- `fixture_pdf` — parameterized over available fixture PDFs.
- `tmp_output_dir` — isolated per-test output directory; avoids
  `_assert_output_dir_clean` failures across consecutive tests.

### 7. Error handling, CI workflow, config examples

#### Error handling

No central error-handling machinery; per-adapter clear messages at boundaries.

| Failure | Where | Message shape |
|---|---|---|
| GPU extras not installed, user tries GPU adapter | `adapters/__init__.py` `try/except ImportError` skips registration; registry's `get_*` raises | `KeyError: segmenter 'mineru25' not found. Available: [...]. Did you install the [gpu] extra?` |
| Model weights missing / offline | `_ensure_loaded()` raises | `RuntimeError: Cannot load weights for 'olmocr2'. Run with internet to cache, or 'python -m extraction prefetch' first. HF_HUB_OFFLINE=1 enforces offline.` |
| CUDA OOM during inference | Adapter wraps inference, catches `is_cuda_oom()` (from `_runtime.py`), calls `release_runtime_resources(self)` + `torch.cuda.empty_cache()`, retries once | On retry failure: `RuntimeError: OOM on <adapter> even after cache clear. Free VRAM or use a smaller variant.` |
| Unknown adapter name in YAML | Registry's `get_*` already raises with available list | No code change needed. |
| Malformed PDF | PyMuPDF raises | Bubbled unchanged. |
| Output dir dirty | `_assert_output_dir_clean` | Already informative. |

Troubleshooting doc (`docs/troubleshooting.md`) ships with Stage 4.

#### GitHub Actions workflow (`.github/workflows/ci.yml`, Stage 1)

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
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install uv
        run: python -m pip install uv
      - name: Install (cpu.lock + editable)
        run: uv pip install --system -r requirements/cpu.lock -e .
      - name: Lint
        run: ruff check extraction
      - name: Type check
        run: mypy
      - name: Tests (CPU, no integration)
        run: pytest -q
```

- Runs on GitHub-hosted runners — no GPU. Integration tests not executed here.
- Matrix covers the minimum supported Python (3.10, per `requires-python`) and
  current stable (3.12).
- Future: add a manually-dispatched workflow for GPU integration tests on a
  self-hosted runner (out of scope for this spec).

#### Pull-request template (`.github/pull_request_template.md`, Stage 1)

```markdown
## Summary
<what this PR changes>

## Test plan
- [ ] `pytest -q` passes locally
- [ ] `ruff check extraction` passes
- [ ] `mypy` passes
- [ ] `pytest -m integration` passes locally (if touching GPU adapters)
```

#### Config examples

**Stage 1 — `config_cpu.yaml`:**

```yaml
extraction:
  renderer: pymupdf
  segmenter: pymupdf_text
  text_extractor: pymupdf_text   # tool_match — reuses segmenter output
  table_extractor: noop
  formula_extractor: noop
  figure_descriptor: noop
  output_dir: outputs
  confidence_threshold: 0.3
  dpi: 150
```

**Stage 4 — `config_gpu.yaml`:**

```yaml
extraction:
  renderer: pymupdf
  segmenter: mineru25
  text_extractor: olmocr2
  table_extractor: mineru25      # tool_match — reuses segmenter tables
  formula_extractor: noop
  figure_descriptor: qwen25vl
  output_dir: outputs
  confidence_threshold: 0.3
  dpi: 150

adapters:
  olmocr2:
    model_id: "allenai/olmOCR-2-7B-1025"
    revision: "<pinned-sha>"
    dtype: "float16"
  qwen25vl:
    model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
    revision: "<pinned-sha>"
    dtype: "float16"
```

Stages 2 and 3 each ship their own intermediate config (`config_s2.yaml`,
`config_s3.yaml`) so any past stage is easily reproducible.

## Open items flagged for follow-up

- **Fixture PDFs** — user to provide 1–2 real PDFs covering text plus
  tables/figures. Must be non-sensitive (committed to the repo, visible to
  collaborators).
- **MinerU dependency pins** — verify `mineru>=2.5` transitive requirements
  (torch / transformers / opencv versions) at Stage 2 implementation time and
  tighten `pyproject.toml` accordingly.
- **`TextExtractor.extract` interface quirk** — papered over with per-page
  caching in olmocr2. A proper refactor of the interface (pass a region bbox
  or restructure to `extract_page -> dict[region_id, ElementContent]`) is a
  worthwhile follow-up, not part of this work.
- **CUDA-version-coupled `gpu.lock`** — `torch==X+cu130` binds the lock file
  to the user's current environment. Collaborators regenerate locally if their
  CUDA differs.
- **GPU CI** — not in scope. Revisit when a self-hosted runner with a GPU is
  available, or when an external service is acceptable.

## Implementation order summary

| PR | Stage | Adapters added | GPU needed | Runnable after merge |
|---|---|---|---|---|
| 1 | CPU foundation + CI + lock + fixtures | pymupdf (renderer + text_segmenter), noop | No | Text-only CPU pipeline |
| 2 | MinerU 2.5 | mineru25 (segmenter + table) | Yes | Layout + tables via GPU; text via CPU fallback |
| 3 | olmOCR-2 | olmocr2 (text_extractor) | Yes | Full layout + tables + real OCR text |
| 4 | Qwen2.5-VL + docs polish | qwen25vl (figure_descriptor) | Yes | Full production stack |
