# Writing Extraction Adapters

Checkliste für alle neuen Adapter (Renderer, Segmenter, Extractor,
Figure-Descriptor) im Extraction-Block.

## Pflicht

1. **Echte Confidence liefern (Segmenter).**
   Das ML-Modell liefert fast immer pro Region einen Score. Diesen in
   `Region.confidence` eintragen. **Nicht** hart auf `1.0` setzen — damit
   wird `confidence_threshold` in `ExtractionConfig` wirkungslos. Regelbasierte
   Segmenter (z. B. `PyMuPDFTextSegmenter`) dürfen `1.0` verwenden, müssen
   diesen Umstand aber im Adapter kommentieren.

2. **bbox in PDF-Points (Origin top-left).**
   Niemand im Projekt rechnet einen Pixel-bbox zurück in Points. Wenn das
   Tool Pixel liefert, muss der Adapter selbst umrechnen. Das Cropping auf
   Pixel passiert an genau einer Stelle (`OutputWriter.crop_region`) mit
   `scale = dpi / 72`.

3. **`tool_name` konsistent.**
   Property `tool_name` und Registry-Name (`@register_xxx("NAME")`) müssen
   identisch sein. Die Pipeline vergleicht diese Strings für die
   Tool-Match-Optimierung beim Merge.

4. **Merge-Regel beachten (Extractor-Rollen).**
   Ein Extractor gibt nur Content-Felder zurück (`text`, `markdown`,
   `latex`, `description`). Layout-Felder (`caption`) gehören dem Segmenter
   und werden vom Extractor-Output überschrieben, falls er sie setzt.

## Empfohlen

- Lazy Imports für schwere Dependencies (Torch, Transformers, MinerU):
  Import im `__init__` oder in einem `_load()`-Helper, nicht auf Modul-Ebene.
  Die Pipeline registriert Adapter beim Import — schwere Importe dort
  machen `python -m extraction` auf CPU-Hosts unnötig langsam/fragil.
- Tests in `extraction/tests/test_<adapter>.py`. Integration-Tests, die
  GPU oder Modell-Downloads brauchen, mit `@pytest.mark.integration`
  markieren.

## Beispiele

- Typischer Content-Extractor (CPU, keine ML-Confidence):
  `extraction/adapters/stubs.py`
- Rich Segmenter (ML, echte Scores):
  `extraction/adapters/mineru25_segmenter.py`
- Regelbasierter Segmenter (keine ML-Confidence):
  `extraction/adapters/pymupdf_text_segmenter.py`
