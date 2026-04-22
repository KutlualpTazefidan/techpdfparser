# techpdfparser

Local-first PDF-Ingest für technische Dokumente mit modularen Pipelines, austauschbaren Embeddings und austauschbarer Vector-DB-Anbindung. Das System priorisiert Robustheit und Nachvollziehbarkeit: Konfigurationsfehler werden früh validiert, bestehende Collections werden gegen ihr Schema geprüft, und Modell-/Backend-Wechsel erzeugen standardmäßig einen neuen Namespace statt still dieselben Daten weiterzuverwenden.

## Überblick

```text
PDF
 ├── Visual Pipeline     PyMuPDF -> Visual Embedder -> visual_pages
 ├── Text Pipeline       Extractor -> Chunker -> Text Embedder -> text_chunks
 └── Structured Pipeline Parser -> Enrichment -> Text Embedder -> tables/formulas/figures
```

Wichtige Laufzeitregeln:

- Adapter werden über die Registry und `config.yaml` gewählt, nicht im Pipeline-Code.
- Collections werden intern aus `index_namespace + base_collection` aufgelöst.
- `index_namespace: auto` erzeugt einen deterministischen Namespace aus Backend und Embedder-Signaturen.
- `index_namespace: legacy` nutzt die alten, unnamespaced Collection-Namen.
- Retrieval läuft bewusst sequentiell und robust, nicht parallel optimiert.
- `section_aware` ist nur mit `pymupdf_structured` erlaubt.

## Architektur

Die wichtigsten Bausteine:

- `src/core/config.py`: YAML -> Pydantic-Konfiguration inklusive Cross-Checks.
- `src/core/indexing.py`: `VectorSchema`, `adapter_signature`, Namespace-Auflösung.
- `src/core/pipelines/`: Visual-, Text- und Structured-Pipeline.
- `src/adapters/`: konkrete Implementierungen für Renderer, Extractor, Embedder, Parser, Fusion und Vector-DB.
- `src/adapters/vectordb/qdrant.py`: produktiver Backendpfad.
- `src/adapters/vectordb/memory.py`: test-only Backend zum Verifizieren der Backend-Agnostik.
- `src/utils/manifest.py`: Manifeste mit Backend, Namespace, Signaturen und Schema.

Mehr Kontext steht in [ARCHITECTURE.md](ARCHITECTURE.md) und [docs/ARCHITEKTUR.md](docs/ARCHITEKTUR.md).

## Setup

Voraussetzungen:

- Python 3.10+
- Für produktives Retrieval typischerweise Qdrant, z. B. `docker run -p 6333:6333 qdrant/qdrant`

Die Requirements behalten ihre bisherige Struktur, werden aber jetzt immer mit `constraints-common.txt` installiert, damit fragile Transitiv-Abhängigkeiten reproduzierbar bleiben.

### CPU / Apple Silicon / allgemeines lokales Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt -r requirements-amd.txt
pip install -e .
```

### NVIDIA GPU

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -r requirements-dev.txt -r requirements-gpu.txt
pip install -e .
```

Hinweis: Die `requirements*.txt` binden automatisch `constraints-common.txt` ein.

## Konfiguration

```bash
cp config.example.yaml config.yaml
```

Zentrale Schalter:

```yaml
pipelines:
  visual:
    embedder: "colqwen25"
  text:
    extractor: "pymupdf_structured"
    chunker: "section_aware"
    embedder: "bge_m3"
  structured:
    parser: "mineru25"

retrieval:
  retrieval_engine: "qdrant"
  fusion_engine: "rrf"
  index_namespace: "auto"      # "auto" | "legacy" | eigener String
  validate_on_start: true
  fail_on_schema_mismatch: true
```

Namespace-Verhalten:

- `auto`: neuer Namespace pro Backend-/Embedder-Kombination
- `legacy`: bisherige Collection-Namen ohne Namespace
- eigener String: stabiler, expliziter Namespace

Reindex-Modell:

1. Config ändern
2. `python -m src doctor --config config.yaml`
3. PDF erneut mit `ingest` in den neuen Namespace schreiben

Es gibt keine stille In-Place-Migration vorhandener Vektordaten.

## Doctor

`doctor` ist der Preflight für neue oder geänderte Setups.

```bash
python -m src doctor --config config.yaml
```

Geprüft werden:

- aktive Adapter und ihre Python-Imports
- Config-Konsistenz
- Backend-Erreichbarkeit
- aufgelöster Namespace
- bestehende Collection-Schemata gegen `VectorSchema`

## Ingest

```bash
python -m src ingest path/to/document.pdf --config config.yaml
```

Die CLI zeigt dabei unter anderem den aufgelösten Namespace und die wichtigsten Collection-Namen an.

Die erzeugten Manifest-Dateien enthalten jetzt:

- aktives Backend
- Namespace
- Adapter-Signaturen
- validierte Vector-Schemata

## Query API

Wenn `index_namespace` nicht auf `legacy` steht, sollten Collection-Namen nicht mehr hart verdrahtet werden. Verwende stattdessen die Layout-Auflösung:

```python
from src.core.config import load_config
from src.core.indexing import resolve_index_layout
from src.core.registry import (
    get_fusion_engine,
    get_retrieval_engine,
    get_text_embedder,
    get_visual_embedder,
)
from src.core.retrieval import UnifiedRetriever

cfg = load_config("config.yaml")
visual_embedder = get_visual_embedder(
    cfg.pipelines.visual.embedder,
    **cfg.adapters.get(cfg.pipelines.visual.embedder, {}),
)
text_embedder = get_text_embedder(
    cfg.pipelines.text.embedder,
    **cfg.adapters.get(cfg.pipelines.text.embedder, {}),
)
layout = resolve_index_layout(
    cfg,
    visual_embedder=visual_embedder,
    text_embedder=text_embedder,
)

retriever = UnifiedRetriever(
    retrieval_engine=get_retrieval_engine(
        cfg.retrieval.retrieval_engine,
        **cfg.adapters.get(cfg.retrieval.retrieval_engine, {}),
    ),
    visual_embedder=visual_embedder,
    text_embedder=text_embedder,
    fusion_engine=get_fusion_engine(
        cfg.retrieval.fusion_engine,
        **cfg.adapters.get(cfg.retrieval.fusion_engine, {}),
    ),
    visual_collection=layout.collections["visual"],
    text_collection=layout.collections["text"],
    tables_collection=layout.collections["tables"],
    formulas_collection=layout.collections["formulas"],
    figures_collection=layout.collections["figures"],
)

results = retriever.query("heat dissipation in multilayer PCBs", top_k=10)
```

## Entwicklung

```bash
pytest -q
ruff check src
mypy
```

Die Quality-Gates sind jetzt bewusst auf den stabilen Kern zugeschnitten:

- `ruff` fokussiert auf Import-/Fehlerklassen im Produktivcode.
- `mypy` prüft den stabilen Kernbereich (`src/core`, `src/utils`, `src/adapters/chunkers`, `src/adapters/vectordb`, `src/__main__.py`) mit pragmatischen Regeln statt globalem `strict`.

## Aktueller Fokus

- Qdrant ist der produktive Backendpfad.
- `memory` dient als test-only Backend zum Nachweis der Modularität.
- Robustheit und explizite Validierung haben Vorrang vor Geschwindigkeit.
