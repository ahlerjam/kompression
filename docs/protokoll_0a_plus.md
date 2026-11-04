# Protokoll 0a+ B1: „erst lernen, dann komprimieren“ (ENTWURF, Version 1)

Stand: 2026-11-04 12:00. Freigabe der Variante B1 durch Jonas (2026-11-04). Wird nach der Kalibrierung (Seeds 100–102) eingefroren; erst dann Hauptlauf (Seeds 20–29).

## Frage

Verbessert Kompression NACH dem Lernen (Nachtraining ab dem Endstand von Weight Decay) die prequentielle Codelänge der Additionstabelle? Und ist unsere Kompression (variationale Codelänge mit GMM-Prior) dabei besser als gängige Standardverfahren?

**Einordnung (ehrlich):** „Erst trainieren, dann komprimieren“ ist eine etablierte Methode. Unser mdl-Arm entspricht im Kern Soft Weight-Sharing (Ullrich, Meeds & Welling 2017), ursprünglich ein Nachtrainings-Kompressionsverfahren; verwandt: Deep Compression (Han, Mao & Dally 2016: Pruning, Quantisierung), Bayesian Compression (Louizos, Ullrich & Welling 2017), quantisierungsbewusstes Nachtraining (BitNet-artig). Neu ist hier nicht das Verfahren, sondern die Messung: ob das Nachkomprimieren die *prequentielle Codelänge* (Vorhersage noch ungesehener Daten) verbessert, nicht nur die Modellgröße bei gleicher Genauigkeit.

## Aufbau

- Aufgabe, Modell, Optimierer wie Protokoll 0a (Addition mod 97, ganze Tabelle, Blöcke ab 64, Neustart je Block, Self-Switch über Trainingsstände × Temperaturen, bezahlt).
- Phase 1 (für alle Bedingungen identisch): `wd` λ = 1, 3.000 Schritte je Block; Kandidaten c0 … c3000.
- Phase 2 (Zweig vom Endstand der Phase 1, gleiches Compute für alle): 1.000 Schritte Nachtraining; zusätzliche Kandidaten p30, p100, p300, p1000. Alle Bedingungen zahlen dieselben Wahl-Bits.
- Bedingungen (Phase 2):
  - `wd` weiter (Standard ohne Kompression, Referenz)
  - `mdl` (unser Verfahren, Soft Weight-Sharing), β aus Kalibrierung, Warmup 300
  - `bitnet` (ternäres QAT-Nachtraining, Standard-Quantisierung)
  - `prune` (Magnitude-Pruning je Matrix mit Nachtraining unter wd, Standard nach Han et al.), Anteil aus Kalibrierung
- Code: `stages/s0a_grokking/prequential.py --post …` (neue Option, `--post none` bitgenau wie vorher geprüft), Start `scripts/b1_0a_plus.sh`.

## Kalibrierung (Seeds 100–102, kein Befund)

mdl β ∈ {0,03; 0,1}, prune Anteil ∈ {0,5; 0,9}; je ein Wert wird für den Hauptlauf festgelegt (kleinste mittlere Codelänge). [[Ergebnis eintragen]]

## Hypothesen (Hauptlauf, Seeds 20–29, gepaart)

| | Hypothese | Endpunkt |
|---|---|---|
| H1 (primär) | Nachtraining mit `mdl` codiert kürzer als `wd` weiter | prequentielle Codelänge |
| H2 | `mdl` kürzer als `bitnet`-Nachtraining | prequentielle Codelänge |
| H3 | `mdl` kürzer als `prune`-Nachtraining | prequentielle Codelänge |
| H4 (sekundär) | `mdl` präziser bei viel Daten als `wd` weiter | Bits der letzten beiden Blöcke |

- Wilcoxon gepaart (10 Seeds), Bootstrap-95-%-KI, Holm über H1–H4. Beschreibend zusätzlich: bitnet und prune gegen wd weiter.
- Grenzen: eine Aufgabe, eine Modellgröße, ein Phase-1-Verfahren; Wahl von β und Pruning-Anteil auf anderen Seeds kalibriert.
