# Protokoll Stufe 0a (Version 1, ENTWURF zur Freigabe)

Stand: 2026-11-02. Wird mit dem Start des Hauptlaufs eingefroren (Messphase, AGENTS.md).
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
- Schritte je Block S ∈ {100, 300, 1.000, 3.000} als **globaler Hyperparameter**: je S ein eigener Durchlauf mit Warmstart; Codelänge eines Arms = Minimum über S + log2(4) Bit (Wahl ehrlich bezahlt). Grund: 0b-Kalibrierung zeigte, dass langes Training auf kleinen Präfixen Struktur zerstört.
- Innerhalb eines Durchlaufs **Self-Switch (v2):** je Block Wahl aus Trainingsständen {0, 30, 100, 300, 1.000, 3.000} ≤ S × Temperaturen {1; 1,5; 2; 3; 5; 8; 16; 32} oder uniform, bezahlt mit log2(Anzahl Optionen) Bit je Block.
- Arme: `wd`, `mdl`, `bitnet`, `none` (Norm-Projektion ist hier nicht definiert; die Norm-Kontrolle läuft über Teil A).
- **Primärer Endpunkt:** prequentielle Codelänge in Bit (kleiner = bessere Kompression der Daten durch den Lerner).

## Hypothesen und Tests (vorab festgelegt)

| | Hypothese | Endpunkt | Richtung |
|---|---|---|---|
| H1 (primär) | `mdl` codiert die Daten kürzer als `wd` | prequential | mdl < wd |
| H2 | `mdl` groked nicht nur wegen der Norm | stabiler Grok-Schritt, Anteil ≥ 0,99 | mdl besser als wd_normproj |
| H3 | `mdl` findet kürzere Gewichte als `wd` und `wd_normproj` | L(W) | mdl < beide |
| H4 | harte Kompression (`bitnet`) generalisiert besser als `wd` | stabiler Grok-Schritt, prequential | bitnet besser |

- Test: zweiseitiger Wilcoxon-Vorzeichen-Rang-Test der gepaarten Differenzen (10 Paare), Bootstrap-95-%-KI der mittleren Differenz.
- Mehrfachtests: Holm über alle Kontraste von H1–H4.
- Zensierte Grok-Schritte gehen mit dem Wert 40.000 in den Rangtest ein (konservativ).
- **Gate 0a:** H1 mit Holm-adjustiertem p < 0,05, **oder** sauber dokumentierter Negativbefund.

## Rechenzeit (Schätzung aus den Kalibrierungen)

- Teil A: ca. 2,5 h je Arm (10 Seeds × 40k), `mdl` ca. doppelt so lang → zusammen ca. 15 h.
- Teil B: ca. 3–4 h je Arm (alle vier S zusammen; kurze S sind billig) → zusammen ca. 14 h.
- Reihenfolge: `mdl` (Teil A) zuerst, weil `wd_normproj` dessen Normtrajektorie braucht. Insgesamt etwa 1,5 Tage.

## Bekannte Grenzen

- Ein Datensatz, eine Modellgröße; die Aussage gilt nur für dieses Regime.
- β und λ wurden auf anderen Seeds kalibriert, nicht gegeneinander optimiert („kalibriert gegen kalibriert“).
- L(W) bleibt eine obere Schranke; deshalb ist die prequentielle Codelänge primär.
- Prequential-Wert für `mdl` aus der Kalibrierung steht noch aus (läuft); er ändert das Protokoll nicht, nur die Erwartung.
