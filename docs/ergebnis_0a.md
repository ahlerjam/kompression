# Ergebnis Stufe 0a (Hauptlauf, Protokoll Version 2)

Stand: 2026-11-03 22:25. Seeds 10–19, gepaart. Code-Stand: Tag `hauptlauf-0a-v2`, alle Läufe mit `git_dirty: false`. Auswertung: `uv run python stages/s0a_grokking/analyse.py` (Rohausgabe unten unverändert; Kopie lokal in `runs/analyse_0a_hauptlauf.txt`). Keine Warnung zu fehlenden Gittern.

## Kurzfassung

**Gate 0a: Negativbefund für beide primären Hypothesen** (Protokoll: Gate gilt auch bei sauber dokumentiertem Negativbefund).

- **H1 (mdl codiert die Daten kürzer als wd): nein, umgekehrt.** Prequentielle Codelänge mdl 32,5 kbit, wd 26,5 kbit (Mittel; Differenz +6,0 kbit, KI95 [4,4; 7,7] kbit; Wilcoxon p = 0,002, Holm 0,004). wd ist signifikant besser.
- **H4 (bitnet codiert kürzer als wd): nein, knapp umgekehrt.** bitnet 27,1 kbit, wd 26,5 kbit (Differenz +0,54 kbit, KI95 [0,28; 0,80]; p = 0,006, Holm 0,006).
- **Sekundär, in erwarteter Richtung und nach Holm signifikant:**
  - H3: mdl findet die mit Abstand kürzesten Gewichte: L(W) 143 kbit gegenüber 1.288 kbit (wd) und 813 kbit (wd mit gleicher Normtrajektorie). Etwa 9× bzw. 5,7× kürzer.
  - H2b: mdl bleibt stabiler gegrokt als die normgleiche Kontrolle (Anteil Evaluationen ≥ 0,99 in der zweiten Hälfte: 1,0 gegenüber 0,975; Holm 0,014). Der stabile Grok-Schritt selbst (H2a) ist nicht signifikant verschieden (mdl 8.730, wd_normproj 7.230).
  - H5 bitnet: Bits der letzten beiden Blöcke 137 gegenüber 8.599 (wd), Holm 0,014 → bei viel Daten codiert bitnet die Tabelle praktisch exakt.
- **Sekundär, nicht signifikant:** H5 mdl (5.749 gegenüber 8.599 Bit, p = 0,027, Holm 0,082); H4a stabiler Grok bitnet < wd (Holm 0,23).

## Deutung (vorsichtig)

Kompressionsdruck (mdl, ternäre Gewichte) liefert hier kürzere bzw. exaktere Modelle bei viel Daten, zahlt aber auf kleinen Präfixen: Die Netze lernen später, und die prequentielle Summe wird von den frühen Blöcken dominiert. In der Summe gewinnt Weight Decay. Damit ist die These „Kompression der Gewichte führt zu kürzerer Codierung der Daten“ in diesem Regime (eine Aufgabe, ein kleines Modell, kalibrierte Hyperparameter) **nicht bestätigt**. Das Muster passt zum „catch-up“-Phänomen (van Erven, Grünwald, de Rooij 2007): ein komplexeres Modell gewinnt spät, verliert aber die Summe; ein Switch-Code könnte das adressieren (Folgeexperiment 0a+, Entscheidung mit Jonas).

Grenzen: ein Datensatz (Addition mod 97), eine Modellgröße; β und λ auf anderen Seeds kalibriert; L(W) ist eine obere Schranke.

## Rohausgabe

```
# Endpunkte je Arm (Mittel über Seeds)
mdl {'stable_grok': 8730.0, 'frac99': 1.0, 'lw_bits': 143339.845, 'final_test': 1.0} prequential: 32549
wd {'stable_grok': 12010.0, 'frac99': 0.915, 'lw_bits': 1288366.912, 'final_test': 0.998} prequential: 26534
wd_normproj {'stable_grok': 7230.0, 'frac99': 0.975, 'lw_bits': 813312.732, 'final_test': 0.998} prequential: -
bitnet {'stable_grok': 4400.0, 'frac99': 1.0, 'lw_bits': 675211.059, 'final_test': 0.999} prequential: 27075
none {'stable_grok': 40000.0, 'frac99': 0.0, 'lw_bits': 2109588.98, 'final_test': 0.049} prequential: 57110

# Primäre Tests (H1, H4; Holm)
{"kontrast": "H1 prequential mdl < wd", "n": 10, "mittel_a": 32549.16536506047, "mittel_b": 26534.08632915114, "diff_ki95": [4367.098719751524, 7746.916393680798], "p": 0.001953125, "richtung_wie_erwartet": false, "p_holm": 0.00390625}
{"kontrast": "H4 prequential bitnet < wd", "n": 10, "mittel_a": 27075.200066815334, "mittel_b": 26534.08632915114, "diff_ki95": [283.0668183882493, 801.5899762694913], "p": 0.005859375, "richtung_wie_erwartet": false, "p_holm": 0.005859375}

Gate 0a: nicht bestanden (Negativbefund dokumentieren)

# Sekundäre Tests (H2, H3, H4a, H5; Holm)
{"kontrast": "H2a stabiler Grok mdl < wd_normproj", "n": 10, "mittel_a": 8730.0, "mittel_b": 7230.0, "diff_ki95": [-10.0, 2860.0], "p": 0.11328125, "richtung_wie_erwartet": false, "p_holm": 0.2265625}
{"kontrast": "H2b Anteil≥0,99 mdl > wd_normproj", "n": 10, "mittel_a": 1.0, "mittel_b": 0.9749999999999999, "diff_ki95": [0.019000000000000017, 0.031000000000000028], "p": 0.001953125, "richtung_wie_erwartet": true, "p_holm": 0.013671875}
{"kontrast": "H3 L(W) mdl < wd", "n": 10, "mittel_a": 143339.8454165187, "mittel_b": 1288366.9122518233, "diff_ki95": [-1417151.5535969795, -929878.8868068984], "p": 0.001953125, "richtung_wie_erwartet": true, "p_holm": 0.013671875}
{"kontrast": "H3 L(W) mdl < wd_normproj", "n": 10, "mittel_a": 143339.8454165187, "mittel_b": 813312.7324094584, "diff_ki95": [-740621.097443131, -599136.8766753381], "p": 0.001953125, "richtung_wie_erwartet": true, "p_holm": 0.013671875}
{"kontrast": "H4a stabiler Grok bitnet < wd", "n": 10, "mittel_a": 4400.0, "mittel_b": 12010.0, "diff_ki95": [-15490.0, -1320.0], "p": 0.130859375, "richtung_wie_erwartet": true, "p_holm": 0.2265625}
{"kontrast": "H5 letzte Blöcke mdl < wd", "n": 10, "mittel_a": 5749.191289849533, "mittel_b": 8599.144546080435, "diff_ki95": [-4521.4935030329925, -1103.9379224728962], "p": 0.02734375, "richtung_wie_erwartet": true, "p_holm": 0.08203125}
{"kontrast": "H5 letzte Blöcke bitnet < wd", "n": 10, "mittel_a": 137.29028979388366, "mittel_b": 8599.144546080435, "diff_ki95": [-9380.865543659185, -6791.278546400181], "p": 0.001953125, "richtung_wie_erwartet": true, "p_holm": 0.013671875}
```
