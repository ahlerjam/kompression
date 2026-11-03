# Protokoll Stufe 0a (Version 2, EINGEFROREN 2026-11-03)

Stand: 2026-11-03. Eingefroren mit Start des Hauptlaufs (Messphase). Entscheidung für Version 2 durch Claude im Auftrag von Jonas („data driven“), Begründung im Logbuch.
Grundlage: Kalibrierungen im Logbuch (Seeds 0–4 und 100–102; vom Hauptlauf ausgeschlossen).

**Frage:** Führt Kompressionsdruck auf die Gewichte zu besserer Generalisierung als Weight Decay, auch wenn man die Gewichtsnorm kontrolliert?

## Aufbau

- Aufgabe: (a + b) mod 97, Eingabe `[a, op, b, =]`.
- Modell: dichter Transformer, 2 Layer, d = 128, 4 Köpfe, MLP 512, float32, StableMax.
- Optimierer: AdamW, lr = 1e-3, β = (0,9; 0,98), Full-Batch.
- **Seeds: 10–19** (nie in einer Kalibrierung benutzt), gepaart über alle Arme (gleicher Split, gleiche Initialisierung).
- Code-Stand: Git-SHA beim Start, `git_dirty: false` Pflicht.

## Arme

| Arm | Beschreibung | Grundlage |
|---|---|---|
| `wd` | Weight Decay λ = 1 | Kalibrierung: groked, instabil |
| `mdl` | variationale Codelänge, GMM-Prior K = 16, **β = 0,1**, Warmup 2.000 | Kalibrierung: groked stabil; β ≤ 0,01 lernt auswendig, β = 1 lernt nicht |
| `wd_normproj` | Weight Decay λ = 1, Gewichtsnorm je Seed auf die Trajektorie von `mdl` projiziert (E10) | Kontrolle: Ist es nur die Norm? |
| `bitnet` | ternäre Gewichte (QAT), keine weitere Regularisierung | Kalibrierung: groked schnell und stabil |
| `none` | keine Regularisierung | Kalibrierung: groked nicht (Untergrenze) |

Nicht im Hauptlauf: `subspace` (r16 lernt nicht, r64 lernt nur auswendig), bleibt Forschungszweig (E27).

## Teil A: Trainingsläufe (Trainingsanteil 0,4, 40.000 Schritte)

Endpunkte je Seed:
1. **Stabiler Grok-Schritt:** erster Schritt, ab dem Test-Accuracy ≥ 0,99 für mindestens 2.000 Schritte ununterbrochen gilt; Nicht-Grokking = zensiert bei 40.000.
2. Anteil der Evaluationen mit Test ≥ 0,99 in der zweiten Hälfte.
3. L(W) in Bit, Rezept `uq-entropy-v4` (Minimum aus absolut, Seed+Differenz, Low-Rank, geprüften strukturellen Beschreibungen), Toleranz ε = 0,05 nats/Beispiel auf dem Trainingsloss.
4. Finale Test-Accuracy.

## Teil B: Prequentielle Codelänge (primärer Endpunkt)

- Ganze Tabelle (9.409 Paare), Reihenfolge je Seed fest, Blöcke 64, 128, … 9.409.
- **Neustart je Block** (Blier & Ollivier): für jeden Block frisch von derselben Initialisierung, bis 3.000 Schritte auf dem Präfix. Grund: Warmstart ließ MDL auf kleinen Präfixen irreversibel kollabieren (Logbuch 2026-11-02/03).
- **Self-Switch je Block:** Wahl aus Trainingsständen {0, 30, 100, 300, 1.000, 3.000} × Temperaturen {1; 1,5; 2; 3; 5; 8; 16; 32} oder uniform, bezahlt mit log2(Anzahl Optionen) Bit je Block. (Mit Neustart ersetzt das ein Gitter über Schritte je Block.)
- **Hyperparameter-Gitter je Arm, Wahl bezahlt:** `mdl` β ∈ {0,03; 0,1; 0,3} (Warmup 2.000), `wd` λ ∈ {0,3; 1; 3}, `bitnet` und `none` ohne Gitter. Codelänge eines Arms je Seed = Minimum über sein Gitter + log2(Gittergröße) Bit.
- Arme in Teil B: `mdl`, `wd`, `bitnet`, `none`.
- **Primärer Endpunkt:** prequentielle Codelänge in Bit.
- **Sekundärer Endpunkt (vorab benannt):** Bits der letzten beiden Blöcke ([4.096, 9.409)) beim jeweils gewählten Hyperparameter — „Präzision bei viel Daten“.

## Hypothesen und Tests (vorab festgelegt)

| | Hypothese | Endpunkt | Richtung |
|---|---|---|---|
| **H1 (primär)** | `mdl` codiert die Daten kürzer als `wd` | prequential (Gitter) | mdl < wd |
| **H4 (primär)** | `bitnet` codiert die Daten kürzer als `wd` | prequential (Gitter) | bitnet < wd |
| H2 | `mdl` groked nicht nur wegen der Norm | stabiler Grok-Schritt, Anteil ≥ 0,99 (Teil A) | mdl besser als wd_normproj |
| H3 | `mdl` findet kürzere Gewichte als `wd` und `wd_normproj` | L(W) (Teil A) | mdl < beide |
| H5 (sekundär) | Kompression ist bei viel Daten präziser | Bits der letzten beiden Blöcke | mdl < wd, bitnet < wd |

- Test: zweiseitiger Wilcoxon-Vorzeichen-Rang-Test der gepaarten Differenzen (10 Paare), Bootstrap-95-%-KI der mittleren Differenz.
- Holm über H1 und H4 (primär); getrennt Holm über H2, H3, H5 (sekundär).
- Zensierte Grok-Schritte gehen mit 40.000 in den Rangtest ein.
- **Gate 0a:** H1 oder H4 mit Holm-adjustiertem p < 0,05 in erwarteter Richtung, **oder** sauber dokumentierter Negativbefund.

## Kalibrierung, die zu Version 2 führte (Seeds 100–102, Neustart, β = 0,1 / λ = 1)

`wd` 26,7–27,6 kbit · `bitnet` 26,7–26,8 · `mdl` β 0,1: 29,5–29,6 · β 0,03: 52,0–53,9 · `none` (läuft). Erwartung daher: H4 eher als H1; H1 könnte negativ ausfallen. Das Protokoll wird trotzdem so festgelegt.

## Rechenzeit (Schätzung aus den Kalibrierungen)

- Teil A: ca. 2,5 h je Arm (10 Seeds × 40k), `mdl` ca. doppelt so lang → zusammen ca. 15 h.
- Teil B: 8 prequentielle Läufe (3 mdl, 3 wd, bitnet, none) à ca. 1,5–2 h mit 10 Seeds → ca. 14 h.
- Reihenfolge: `mdl` (Teil A) zuerst, weil `wd_normproj` dessen Normtrajektorie braucht. Insgesamt etwa 1,5 Tage.

## Bekannte Grenzen

- Ein Datensatz, eine Modellgröße; die Aussage gilt nur für dieses Regime.
- β und λ wurden auf anderen Seeds kalibriert, nicht gegeneinander optimiert („kalibriert gegen kalibriert“).
- L(W) bleibt eine obere Schranke; deshalb ist die prequentielle Codelänge primär.
- Prequential-Wert für `mdl` aus der Kalibrierung steht noch aus (läuft); er ändert das Protokoll nicht, nur die Erwartung.
