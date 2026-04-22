# Backlog

---

## 2026-04-19 — Audit Segmentation Output Against Input PDFs

**Priority:** Urgent

### User Story
As a developer, I want to compare the segmenter output against the input PDF so that layout-detection weaknesses are caught before they propagate into every downstream extraction stage.

### Tasks
- [ ] Curate a reference set of technical PDFs covering multi-column layouts, dense tables, formulas, figures, and technical drawings
- [ ] Run extraction with each available segmenter (`mineru25`, `pymupdf_text`) and persist `segmentation.json` plus page images per run
- [ ] Overlay segmentation bboxes on the rendered page images and diff them against the source PDF visually
- [ ] Record weaknesses (missed regions, wrong `region_type`, inaccurate bbox, wrong reading order, unrealistic confidence) in a review report
- [ ] Fix prioritised weaknesses in the segmenter adapters or via pre-/post-processing and re-run for verification

### Acceptance Criteria
- [ ] A documented reference PDF set exists for segmentation evaluation
- [ ] Segmentation output is compared page-by-page with the source and captured in a written review
- [ ] Every identified weakness is triaged (fix now / defer / accept) with rationale
- [ ] Critical segmentation issues (missed regions, wrong bboxes) are fixed and re-verified on the reference set

---

## 2026-04-19 — Audit Table Extraction Quality

**Priority:** Important

### User Story
As a developer, I want to compare extracted table content against the source tables so that table accuracy is verified and regressions get caught early.

### Tasks
- [ ] Curate PDFs with representative tables (simple grids, merged cells, nested headers, rotated, dense technical data)
- [ ] Run extraction with the configured `table_extractor` and persist per-element sidecars plus PNG crops
- [ ] Compare each extracted `markdown`/`text` against the rendered source table visually and textually
- [ ] Log extraction weaknesses (cell misalignment, lost content, wrong structure, missed tables, insufficient crop padding)
- [ ] Fix priority weaknesses via adapter parameters, crop padding, or by evaluating an alternative table tool

### Acceptance Criteria
- [ ] A reference set of table-heavy PDFs is curated and documented
- [ ] Every table in the set has a recorded extraction-vs-source comparison
- [ ] Identified weaknesses are triaged and tracked
- [ ] Critical table-extraction defects are fixed and re-verified on the reference set

---

## 2026-04-19 — Audit Formula Extraction Quality

**Priority:** Important

### User Story
As a developer, I want to compare extracted formula LaTeX against the source formulas so that mathematical content is correctly preserved for downstream claim verification.

### Tasks
- [ ] Assemble PDFs containing inline and display formulas of varying complexity
- [ ] Run extraction with the active formula path (currently `noop`; LaTeX arrives via MinerU's `interline_equation` segmenter output)
- [ ] Render extracted LaTeX back to images and diff against the source formula crops
- [ ] Record weaknesses (wrong symbols, lost super-/subscripts, missed formulas, mis-typed regions)
- [ ] Fix priority weaknesses or, if the default path is insufficient, scope a dedicated formula-extractor adapter

### Acceptance Criteria
- [ ] A reference PDF set with formulas is curated
- [ ] Extracted LaTeX is compared against every source formula and recorded
- [ ] Identified weaknesses are triaged; any decision on a dedicated extractor is documented with rationale
- [ ] Critical formula-extraction issues are fixed and re-verified

---

## 2026-04-19 — Audit Figure and Diagram Descriptions

**Priority:** Important

### User Story
As a developer, I want to compare figure and diagram descriptions against their source images so that VLM-generated descriptions are accurate enough for downstream retrieval and claim verification.

### Tasks
- [ ] Curate a PDF set covering diagrams, photographs, technical drawings, and charts
- [ ] Run extraction with the configured `figure_descriptor` (default `qwen25vl`) and persist crops plus descriptions
- [ ] Review each description against its crop for factual accuracy, completeness, and domain terminology
- [ ] Record weaknesses (hallucinations, missed elements, generic descriptions, wrong technical terms)
- [ ] Tune prompts or adapter parameters, or evaluate an alternative descriptor, for priority issues

### Acceptance Criteria
- [ ] A reference set of figure-bearing PDFs exists
- [ ] Every figure has a recorded description-vs-image review
- [ ] Weaknesses are triaged with written rationale
- [ ] Critical description defects are addressed and re-verified

---

## 2026-04-19 — Audit Merge Output (content_list.json)

**Priority:** Important

### User Story
As a developer, I want to verify the merged `content_list.json` against the per-element sidecars so that the stable output contract actually holds across representative extraction runs.

### Tasks
- [ ] Run end-to-end extractions and inspect `content_list.json` against the sidecar set for field completeness, ordering, and ID stability
- [ ] Verify merge rules: caption always from segmenter, content routed per configured role, `extractor` recorded correctly, confidence filter applied
- [ ] Check that `reading_order_index` is globally consecutive and stable after `python -m extraction rebuild`
- [ ] Document any merge defects (mis-routed fields, lost captions, wrong extractor attribution, non-deterministic ordering)
- [ ] Fix identified merge-layer issues in `pipeline.py` / `output.py` and add regression tests

### Acceptance Criteria
- [ ] A review document compares `content_list.json` against sidecars for the reference PDFs
- [ ] All merge-rule invariants (caption source, content source, confidence filter, ID stability) are explicitly verified
- [ ] Weaknesses are triaged and critical ones are fixed with accompanying regression tests
- [ ] `python -m extraction rebuild` produces a byte-identical `content_list.json` for the audited runs
