# Architektur

Diese Datei beschreibt den aktuellen Runtime-Stand des Projekts: modulare Adapter, robuste Schema-Prüfung, deterministische Namespaces und ein expliziter Preflight über `doctor`.

## Ziele

- Embedding-Modelle und Vector-DB-Backends sollen per Konfiguration austauschbar sein.
- Modell- oder Backend-Wechsel dürfen keine bestehenden Collections still wiederverwenden.
- Fehler sollen früh und präzise sichtbar werden: Config-Fehler, Import-Probleme, Backend-Ausfälle und Schema-Mismatches.
- Geschwindigkeit ist zweitrangig; Retrieval bleibt absichtlich sequentiell.

## Modulgrenzen

```text
src/
  core/
    config.py        -> Pydantic-Konfiguration + Cross-Checks
    indexing.py      -> VectorSchema, adapter_signature, Namespace-Auflösung
    interfaces/      -> Protokolle für alle austauschbaren Komponenten
    pipelines/       -> Visual, Text, Structured
    retrieval.py     -> sequenzielles Multi-Collection-Retrieval + Fusion
  adapters/
    ...              -> konkrete Implementierungen
    vectordb/
      qdrant.py      -> produktiver Backendpfad
      memory.py      -> test-only Backend
  utils/
    manifest.py      -> Run-Metadaten
    storage.py       -> Dateisystem-Layout und document.json
```

Die Pipelines kennen nur Interfaces, keine konkreten Tools. Welcher Adapter instanziiert wird, entscheidet die Registry über Namen aus der Config.

## Die drei Pipelines

```text
PDF
 ├── Visual Pipeline
 │    render_all -> embed_pages -> upsert_visual
 ├── Text Pipeline
 │    extract -> chunk -> embed -> upsert_text
 └── Structured Pipeline
      parse -> enrich -> embed -> upsert_tables/formulas/figures
```

Structured nutzt denselben Text-Embedder wie die Text-Pipeline für Tabellen, Formeln und Figure-Beschreibungen. Dadurch ist das Schema der drei Structured-Collections identisch zum Text-Schema.

## Modulare Index-Architektur

Der Kern dafür steckt in [src/core/indexing.py](src/core/indexing.py).

### `VectorSchema`

Jede Collection wird durch ein backend-agnostisches Schema beschrieben:

```python
VectorSchema(dim=1024, distance="cosine", multi_vector=False)
```

Das Schema ist die Vertragsbasis zwischen Embeddern und Vector-DB-Backend.

### `adapter_signature`

Jeder Embedder liefert eine deterministische Signatur aus:

- Adaptername
- Modellname
- Vektordimension
- Distanzmetrik
- Single- oder Multi-Vector-Modus

Die Signatur landet in:

- Namespace-Auflösung
- Manifesten
- `document.json`-Metadaten
- Fehlermeldungen bei Schema-Konflikten

### Namespace-Auflösung

Das Laufzeitlayout wird aus Config + aktiven Embeddern aufgelöst:

```text
index_namespace = auto
-> <backend>__<visual-signature>__<text-signature>
```

Beispiele:

- `auto`: neuer Namespace pro Backend-/Embedder-Kombination
- `legacy`: kein Namespace, alte Collection-Namen bleiben erhalten
- `"kundenprojekt-a"`: expliziter, stabiler Namespace

Die tatsächlichen Collection-Namen entstehen immer aus:

```text
resolved_collection = namespace + "__" + base_collection
```

Nur `legacy` überspringt dieses Präfix.

## Reindex-Modell

Das System migriert bestehende Vektordaten nicht in-place. Der vorgesehene Ablauf ist:

1. Config ändern
2. `python -m src doctor --config config.yaml`
3. PDF erneut mit `python -m src ingest ...` indexieren

Dadurch bleiben alte Namespaces als Vergleichs- oder Fallback-Datenbestand erhalten.

## Backend-Verhalten

### Qdrant

[src/adapters/vectordb/qdrant.py](src/adapters/vectordb/qdrant.py) ist der produktive Pfad.

- `ensure_collection()` erstellt Collections nicht nur, sondern validiert vorhandene Collections gegen `VectorSchema`.
- Bei Mismatch gibt es einen harten Fehler mit erwartetem und gefundenem Schema.
- Multi-Vector-Seiten nutzen Qdrants `MAX_SIM`.

### Memory Backend

[src/adapters/vectordb/memory.py](src/adapters/vectordb/memory.py) dient nur der Verifikation in Tests und beim lokalen Preflight.

- gleiches `IndexWriter`-/`RetrievalEngine`-Interface
- gleiches Schema-Verhalten
- keine produktive Persistenz

## `doctor`

Die CLI in [src/__main__.py](src/__main__.py) hat jetzt zwei Subcommands:

- `ingest`
- `doctor`

`doctor` prüft:

1. aktive Adapter und deren Python-Imports
2. Config-Validierung
3. Runtime-Initialisierung
4. Backend-Erreichbarkeit
5. bestehende Collections gegen das erwartete Layout

Damit ist `doctor` der Standard-Preflight vor jedem Modell- oder Backend-Wechsel.

## Section-Aware Chunking

[src/adapters/chunkers/section_aware.py](src/adapters/chunkers/section_aware.py) ist jetzt wirklich sectionsensitiv:

- benachbarte Blöcke mit gleichem `section_path` werden auf derselben Seite zusammengeführt
- Section-Wechsel trennt hart
- Seitengrenzen werden nie überschritten
- `child_ids` enthalten bei zusammengeführten Chunks alle Ursprungselemente
- `bbox` bleibt bei Merge-Chunks `None`
- `parent_id` wird nur für echte Splits eines einzelnen langen Blocks genutzt

Zusätzlich erzwingt [src/core/config.py](src/core/config.py), dass `section_aware` nur mit `pymupdf_structured` konfiguriert werden darf.

## Manifest- und Storage-Metadaten

Jeder Pipeline-Run schreibt ein Manifest mit:

- Tools und Tool-Versionen
- Element-Anzahlen
- Backend
- Namespace
- Collection-Namen
- Adapter-Signaturen
- Vector-Schemata

`document.json` bekommt dieselben Layout-Metadaten, damit man später nachvollziehen kann, welcher Run in welchen Namespace geschrieben wurde.

## Retrieval

[src/core/retrieval.py](src/core/retrieval.py) ruft alle fünf Collections sequentiell ab:

1. Visual
2. Text
3. Tables
4. Formulas
5. Figures

Danach fusioniert der Fusion-Adapter die Result-Listen. Die Architektur priorisiert hier robustes, leicht nachvollziehbares Verhalten statt Parallelisierung.

## Qualitätsgrenzen

Die Projektkonfiguration in [pyproject.toml](pyproject.toml) ist auf den stabilen Kern zugeschnitten:

- `ruff` prüft funktionale Lint-Klassen im stabilen Kern
- `mypy` fokussiert auf Kernmodule statt auf alle ML-lastigen Adapter

## Relevante Dateien

- [src/core/indexing.py](src/core/indexing.py)
- [src/core/config.py](src/core/config.py)
- [src/__main__.py](src/__main__.py)
- [src/adapters/vectordb/qdrant.py](src/adapters/vectordb/qdrant.py)
- [src/adapters/vectordb/memory.py](src/adapters/vectordb/memory.py)
- [src/adapters/chunkers/section_aware.py](src/adapters/chunkers/section_aware.py)
- [config.example.yaml](config.example.yaml)
