📌 Projektbeschreibung: Intelligentes PDF-Analyse- und Verifikationssystem
🧭 Ziel des Systems
Entwicklung eines Systems zur Analyse technischer Dokumente (PDFs)
Extraktion strukturierter Informationen aus unstrukturierten Quellen
Identifikation und Strukturierung von Aussagen (Claims)
Nachvollziehbarkeit von Informationen innerhalb eines Dokuments
Vorbereitung für eine spätere inhaltliche Verifikation von Aussagen
Bereitstellung von Antworten mit klarer Quellenzuordnung
📄 Input
PDF-Dokumente (insbesondere technische Berichte, Verträge, Studien, Spezifikationen)
Dokumente können enthalten:
Fließtext
Tabellen
Diagramme
Formeln
Abschnittsstrukturen
⚙️ Hauptfunktionale Anforderungen
1. Dokumentenverarbeitung
Extraktion von Text aus PDFs
Erhalt von Strukturinformationen (z. B. Abschnitte, Überschriften)
Trennung von Dokumentbestandteilen:
Text
Tabellen
ggf. Bilder/Diagramme
Speicherung in einem weiterverarbeitbaren Format
2. Dokumentstrukturierung
Erkennung von:
Kapiteln
Unterkapiteln
Abschnitten
Zuordnung von Inhalten zu strukturellen Einheiten
Beibehaltung der logischen Dokumenthierarchie
3. Chunking / Segmentierung
Aufteilung von Dokumenten in kleinere Einheiten (Chunks)
Berücksichtigung von:
semantischen Grenzen
Abschnittsstruktur
Optional:
Overlap zwischen Chunks
Jeder Chunk enthält:
Textinhalt
Positionsinformationen
Referenz zum Ursprungsdokument
4. Metadaten-Management
Speicherung von Metadaten auf mehreren Ebenen:
Dokumentebene
Abschnittsebene
Chunk-Ebene
Beispiele:
Dokument-ID
Abschnittsnummer
Titel
Seitenzahl
Position im Dokument
5. Semantische Repräsentation
Umwandlung von Text in Vektorrepräsentationen (Embeddings)
Speicherung in einer Vektor-Datenbank
Unterstützung von:
semantischer Suche
Ähnlichkeitsabfragen
6. Retrieval-System
Verarbeitung von Nutzeranfragen
Finden relevanter Dokumentteile
Ranking von Ergebnissen
Rückgabe:
relevante Chunks
zugehörige Metadaten
7. Question Answering (RAG)
Generierung von Antworten basierend auf gefundenen Dokumentinhalten
Nutzung eines Language Models
Antworten enthalten:
Textliche Antwort
Verweise auf Quellen (Chunks/Abschnitte)
🧠 Erweiterte Anforderungen (fortgeschrittene Stufe)
8. Claim Extraction
Identifikation von Aussagen im Text
Extraktion strukturierter Informationen aus Aussagen
Beispiele für Claims:
Messwerte
technische Eigenschaften
Leistungsangaben
Struktur eines Claims:
Beschreibung der Aussage
beteiligte Entitäten
Werte und Einheiten
Kontext (Abschnitt/Quelle)
9. Strukturierte Datenerfassung aus Tabellen
Erkennung von Tabellen
Umwandlung in strukturierte Formate (z. B. JSON)
Verknüpfung mit zugehörigen Textstellen
10. Verknüpfung von Informationen
Herstellung von Beziehungen zwischen:
Claims
Tabellen
Abschnitten
Ziel:
Nachvollziehbarkeit von Aussagen
Beispiel:
Claim basiert auf Tabelle
Tabelle basiert auf Messung
11. Aufbau eines Wissensmodells
Repräsentation von:
Entitäten
Beziehungen
Abhängigkeiten
Struktur ähnlich einem Knowledge Graph
Speicherung von:
Knoten (z. B. Claims, Tabellen)
Kanten (z. B. „basiert auf“)
12. Traceability (Nachvollziehbarkeit)
Rückverfolgung von Aussagen zu ihren Quellen
Mehrstufige Verlinkung:
Claim → Abschnitt → Tabelle → Ursprung
Darstellung von Herkunft und Kontext
13. Verifikationslogik
Überprüfung von Aussagen auf Konsistenz
Vergleich von:
Textaussagen
Tabellenwerten
Erkennung von:
Widersprüchen
Inkonsistenzen
Optional:
Regelbasierte oder modellbasierte Prüfung

Systemarchitektur (konzeptionell)
Pipeline-basierter Aufbau
Verarbeitungsschritte:
Dokumenteingang
Parsing / Extraktion
Strukturierung
Segmentierung
Speicherung
Embedding / Indexierung
Retrieval
Antwortgenerierung
(optional) Analyse / Verifikation
🔄 Datenfluss
PDF → strukturierter Text → Chunks → Embeddings → Index
Query → Retrieval → relevante Chunks → Antwort
Erweiterung:
Chunks → Claims → Beziehungen → Wissensstruktur
📦 Datenformate (konzeptionell)
Textdaten (z. B. Markdown oder Plain Text)
Strukturierte Daten (z. B. JSON)
Vektorrepräsentationen (Embeddings)
Graphstruktur (optional)
🧪 Nicht-funktionale Anforderungen
Modularität:
einzelne Komponenten austauschbar
Erweiterbarkeit:
neue Verarbeitungsschritte integrierbar
Nachvollziehbarkeit:
Ergebnisse müssen erklärbar sein
Robustheit:
unterschiedliche PDF-Formate verarbeiten können
Testbarkeit:
einzelne Pipeline-Schritte isoliert testbar
🚫 Nicht-Ziele (vorerst)
Perfekte Extraktionsgenauigkeit
Vollständige Diagramm-Interpretation
Echtzeit-Verarbeitung großer Datenmengen
Benutzeroberfläche
Vollautomatische wissenschaftliche Verifikation auf höchstem Niveau