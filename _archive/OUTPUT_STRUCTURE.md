# Output-Struktur — techpdfparser

Erzeugt durch: `python -m src ingest test-pdf.pdf --config config.yaml`  
Dokument: `test-pdf.pdf` (9 Seiten, gescannt → kein Textlayer)  
`doc_id`: `42de1bdf4b766d81`

---

## Verzeichnisbaum

```
outputs/
└── documents/
    └── 42de1bdf4b766d81/               ← ein Ordner pro Dokument (doc_id = SHA256 des Dateipfads)
        │
        ├── document.json               ← Index aller Runs für dieses Dokument
        │
        └── runs/
            │
            ├── visual_clip_<ts>/       ← Visual-Pipeline-Run
            │   ├── manifest.json
            │   ├── elements.jsonl      ← 1 VisualPage pro Zeile (mit Embedding)
            │   └── pages/
            │       ├── p0000.png       ← gerendertes Seitenbild (DPI 150)
            │       ├── p0001.png
            │       └── ...             ← p0000–p0008 (9 Seiten)
            │
            ├── text_pymupdf_text_minilm_<ts>/   ← Text-Pipeline-Run
            │   ├── manifest.json
            │   ├── raw_blocks.jsonl    ← Rohausgabe des Extraktors (vor Chunking)
            │   └── chunks.jsonl        ← Chunks mit Embedding
            │
            └── structured_pdfplumber_pix2tex_noop_<ts>/   ← Structured-Pipeline-Run
                ├── manifest.json
                ├── tables.jsonl
                ├── formulas.jsonl
                ├── figures.jsonl       ← 1 Figure pro Zeile (mit Embedding + Bildpfad)
                └── figures/
                    ├── 42de1bdf4b766d81_p0_f0.png   ← aus tempdir verschoben
                    ├── 42de1bdf4b766d81_p1_f0.png
                    └── ...             ← 1 Bild pro Seite (9 Bilder)
```

---

## document.json

Wird nach jedem Run aktualisiert. Enthält einen Eintrag pro Run.

```json
{
  "doc_id": "42de1bdf4b766d81",
  "source_file": "test-pdf.pdf",
  "runs": [
    {
      "run_id": "visual_clip_20260411_131732",
      "pipeline": "visual",
      "recorded_at": "2026-04-11T13:17:43.525278+00:00"
    },
    {
      "run_id": "text_pymupdf_text_minilm_20260411_131743",
      "pipeline": "text",
      "recorded_at": "2026-04-11T13:17:47.305519+00:00"
    },
    {
      "run_id": "structured_pdfplumber_pix2tex_noop_20260411_131747",
      "pipeline": "structured",
      "recorded_at": "2026-04-11T13:17:50.548225+00:00"
    }
  ]
}
```

---

## manifest.json — Visual

```json
{
  "run_id": "visual_clip_20260411_131732",
  "pipeline": "visual",
  "doc_id": "42de1bdf4b766d81",
  "source_file": "test-pdf.pdf",
  "started_at": "2026-04-11T13:17:32.051689+00:00",
  "finished_at": "2026-04-11T13:17:43.524627+00:00",
  "duration_seconds": 11.473,
  "tools": { "renderer": "pymupdf", "embedder": "clip" },
  "tool_versions": { "clip": "vit-base-patch32" },
  "counts": { "pages": 9, "elements": 9 },
  "qdrant": { "collection": "visual_pages", "upserted": 9 }
}
```

---

## manifest.json — Text

```json
{
  "run_id": "text_pymupdf_text_minilm_20260411_131743",
  "pipeline": "text",
  "doc_id": "42de1bdf4b766d81",
  "source_file": "test-pdf.pdf",
  "started_at": "2026-04-11T13:17:43.525852+00:00",
  "finished_at": "2026-04-11T13:17:47.299399+00:00",
  "duration_seconds": 3.774,
  "tools": { "extractor": "pymupdf_text", "chunker": "fixed_size", "embedder": "minilm" },
  "tool_versions": { "pymupdf_text": "1.24" },
  "counts": { "raw_blocks": 0, "chunks": 0 },
  "qdrant": { "collection": "text_chunks", "upserted": 0 }
}
```

> **Hinweis:** 0 Blöcke/Chunks weil `test-pdf.pdf` ein gescanntes PDF ohne Textlayer ist.  
> Mit einem PDF mit echtem Text werden hier Daten erscheinen.

---

## manifest.json — Structured

```json
{
  "run_id": "structured_pdfplumber_pix2tex_noop_20260411_131747",
  "pipeline": "structured",
  "doc_id": "42de1bdf4b766d81",
  "source_file": "test-pdf.pdf",
  "started_at": "2026-04-11T13:17:47.306095+00:00",
  "finished_at": "2026-04-11T13:17:50.539266+00:00",
  "duration_seconds": 3.233,
  "tools": { "parser": "pdfplumber", "formula_extractor": "pix2tex", "figure_descriptor": "noop" },
  "counts": { "tables": 0, "formulas": 0, "figures": 9 },
  "qdrant": { "collection": "tables,formulas,figures", "upserted": 9 }
}
```

---

## elements.jsonl — Schema eines VisualPage-Eintrags

```json
{
  "object_id": "2f94f6e2ece5d056",
  "doc_id": "42de1bdf4b766d81",
  "source_file": "test-pdf.pdf",
  "page_number": 0,
  "object_type": "visual_page",
  "bbox": null,
  "tool_name": "clip",
  "tool_version": "vit-base-patch32",
  "extraction_timestamp": "2026-04-11T...",
  "raw_output_path": "outputs/documents/.../pages/p0000.png",
  "image_path": "outputs/documents/.../pages/p0000.png",
  "embedding": [[0.021, -0.013, ...]]   ← shape: 1 × 512 (CLIP, single-vector wrapped)
}
```

---

## Namenskonventionen

| Element | Muster | Beispiel |
|---|---|---|
| `doc_id` | SHA256 des absoluten Pfads, 16 Hex-Zeichen | `42de1bdf4b766d81` |
| Run-Verzeichnis | `<pipeline>_<tools>_<YYYYmmdd_HHMMSS>` | `visual_clip_20260411_131732` |
| Seiten-PNG | `p<####>.png` | `p0003.png` |
| Figure-PNG | `<doc_id>_p<n>_f<n>.png` | `42de1bdf4b766d81_p2_f0.png` |
| JSONL-Dateien | ein Element pro Zeile, UTF-8 | `elements.jsonl`, `chunks.jsonl` |

---

## Wiederholte Runs

Jeder `python -m src ingest` erzeugt **neue** Run-Verzeichnisse mit neuem Timestamp.  
`document.json` wächst um einen Eintrag pro Run — so lassen sich Runs mit verschiedenen  
Tools oder Einstellungen direkt vergleichen:

```bash
diff runs/visual_clip_20260411_130755/elements.jsonl \
     runs/visual_clip_20260411_131732/elements.jsonl
```
