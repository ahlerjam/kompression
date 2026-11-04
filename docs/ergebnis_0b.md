# Ergebnis Stufe 0b (Transfer in Bit): Subtraktion und Multiplikation

Stand: 2026-11-04 04:57. Hauptlauf nach `docs/protokoll_0b.md` Version 2 (eingefroren 2026-11-03 22:30), Seeds 10–19, A-Gewichte aus Hauptlauf 0a, Lerner `wd` λ = 1, Neustart je Block, S ∈ {30, 100, 300, 1.000, 3.000}, Codelänge = Minimum über S + log2(5) Bit. Teil Multiplikation ergänzt 2026-11-04 11:27 (gleiches Protokoll, gleiche Seeds).

## Kurz

**Gate 0b: BESTANDEN.** Alle vier vorab festgelegten Hypothesen für die Subtraktion sind in erwarteter Richtung signifikant (je Holm p = 0,0078; alle 10 Seeds in derselben Richtung, kleinstmögliches Wilcoxon-p bei n = 10).

| Bedingung (Start auf B = Subtraktion) | mittlere Codelänge |
|---|---|
| Zufall | 29,3 kbit |
| A(wd) permutiert | 54,4 kbit |
| A(wd) Norm-skaliert | 54,4 kbit |
| **A(wd)** | **12,5 kbit** |
| **A(mdl)** | **7,4 kbit** |
| A(bitnet) (beschreibend, keine Hypothese) | 6,9 kbit |
| uniform (Referenz) | 62,1 kbit |

- H1 Transfer: A(wd) spart gegenüber Zufall 16,9 kbit (KI95 der Differenz [−18,3; −15,6] kbit).
- H2 Struktur: A(wd) < A(wd) permutiert, Differenz KI95 [−43,6; −40,2] kbit. Permutiert ist schlechter als Zufall.
- H3 nicht nur Norm: A(wd) < A(wd) Norm-skaliert, KI95 [−43,5; −40,2] kbit.
- H4 komprimierte A-Lösungen übertragen besser: A(mdl) < A(wd), KI95 [−6,9; −3,3] kbit; Transfer-Gewinn T(mdl) 22,0 vs. T(wd) 16,9 kbit.

## Deutung

- Übertragen wird Struktur, nicht Gewichtsstatistik oder Norm: Permutierte oder normgleiche Zufallsgewichte sind sogar schlechter als ein normaler Zufallsstart.
- Zusammen mit 0a: `mdl` verliert auf der eigenen Aufgabe (prequential, H1 0a negativ), seine Lösung überträgt aber besser und ist ~9× kürzer (L(W) 143 vs. 1.288 kbit). bitnet zeigt beschreibend dasselbe Bild. Das ist der erste Befund in diesem Projekt, der zur Kernthese „Kompression → Transfer“ passt.

## Grenzen

- Zwei Aufgabenpaare (Addition → Subtraktion, Addition → Multiplikation mod 97), eine Modellgröße.
- Gemessen ist die bedingte Codelänge von B; L(A-Gewichte) ist nicht mitgezählt (im Gesamtziel des Konzepts nötig). Die Aussage „mdl-Gewichte sind kürzer UND übertragen besser“ ist eine Zusammenschau zweier Befunde, kein eigener Test.
- Lerner auf B ist immer `wd`; H4 vergleicht nur die Herkunft der Startgewichte.
- Bei n = 10 und vollständiger Richtungsgleichheit ist p nach unten begrenzt; die Effektgrößen (KI) sind aussagekräftiger.

## Teil Multiplikation (zweite Zielaufgabe, Holm über H1–H4): NEGATIV

| Bedingung (Start auf B = Multiplikation) | mittlere Codelänge |
|---|---|
| **Zufall** | **27,0 kbit** |
| A(wd) | 40,6 kbit |
| A(wd) permutiert | 40,7 kbit |
| A(wd) Norm-skaliert | 42,5 kbit |
| A(mdl) | 43,5 kbit |
| A(bitnet) (beschreibend) | 42,9 kbit |
| uniform (Referenz) | 62,1 kbit |

- H1 Transfer: **Gegenrichtung signifikant.** A(wd) codiert Multiplikation 13,6 kbit LÄNGER als Zufall (KI95 der Differenz [+9,3; +17,4] kbit, Holm p = 0,0078, alle Seeds). Negativer Transfer.
- H2 Struktur: nicht bestätigt; A(wd) ≈ A(wd) permutiert (KI95 [−5,2; +5,1] kbit, Holm p = 1,0).
- H3 nicht nur Norm: nicht bestätigt; A(wd) vs. Norm-skaliert KI95 [−6,2; +1,9] kbit, Holm p = 1,0.
- H4: nicht bestätigt, Richtung sogar umgekehrt (A(mdl) 2,9 kbit länger als A(wd), KI95 [−1,1; +7,6], Holm p = 1,0).
- Alle Starts aus A-Gewichten (auch permutiert und Norm-skaliert) liegen bei 40–44 kbit, also gleich schlecht. Das passt zur vorab im Logbuch (09:40) notierten explorativen Deutung: Die Strafe kommt von der großen Gewichtsnorm der A-Lösungen (die das Grokking auf neuer Struktur verzögert), nicht von übertragener Struktur. Beide dort notierten Vorhersagen trafen ein: A(bitnet) bei mul schlecht (42,9 kbit; höchste A-Norm) und wd_permuted ≈ wd_scaled (40,7 vs. 42,5). Das bleibt eine Deutung, kein Test.

## Deutung über beide Zielaufgaben

- Transfer in Bit ist messbar, und die Messung trennt Struktur von Statistik: bei Subtraktion (Struktur geteilt) großer, struktureller Gewinn; bei Multiplikation (andere Struktur mod 97) ein Verlust, den auch strukturlose Kontrollen mit gleicher Norm zeigen.
- „Komprimierte Lösungen übertragen besser“ (H4) gilt für die verwandte Aufgabe, nicht für die entfernte. Die Kernthese bekommt damit eine Bedingung: Kompression hilft beim Transfer, wenn die Struktur geteilt wird; sie schützt nicht vor negativem Transfer.
- Explorativ (Switch-Code, `runs/explor_switch_mul.txt`, Logbuch): Darf der Code je Block zwischen Zufallsstart und A-Start wechseln, wird A-Start in frühen Blöcken gelegentlich gewählt und spart 0,4–1,0 kbit gegenüber Zufall allein (10/10 Seeds); der negative Transfer betrifft die späten, datenreichen Blöcke.

## Rohausgabe `analyse_0b.py` (unverändert)

```
## Zielaufgabe sub
random n = 10 Mittel kbit: 29.34
wd n = 10 Mittel kbit: 12.46
wd_scaled n = 10 Mittel kbit: 54.36
wd_permuted n = 10 Mittel kbit: 54.4
mdl n = 10 Mittel kbit: 7.35
bitnet n = 10 Mittel kbit: 6.94
Transfer-Gewinn T(wd) Mittel kbit: 16.88
Transfer-Gewinn T(mdl) Mittel kbit: 21.99
Transfer-Gewinn T(bitnet) Mittel kbit: 22.4
{"kontrast": "H1 Transfer: wd-A < Zufall", "n": 10, "mittel_a": 12461.751651472263, "mittel_b": 29342.232404052767, "diff_ki95": [-18268.000382159164, -15563.594588281834], "p": 0.001953125, "richtung_wie_erwartet": true, "p_holm": 0.0078125}
{"kontrast": "H2 Struktur: wd-A < wd-A permutiert", "n": 10, "mittel_a": 12461.751651472263, "mittel_b": 54398.684007004915, "diff_ki95": [-43563.512724500004, -40204.64419864393], "p": 0.001953125, "richtung_wie_erwartet": true, "p_holm": 0.0078125}
{"kontrast": "H3 nicht nur Norm: wd-A < wd-A Norm-skaliert", "n": 10, "mittel_a": 12461.751651472263, "mittel_b": 54359.29753321502, "diff_ki95": [-43459.681727482464, -40240.43847979493], "p": 0.001953125, "richtung_wie_erwartet": true, "p_holm": 0.0078125}
{"kontrast": "H4 komprimiert überträgt besser: mdl-A < wd-A", "n": 10, "mittel_a": 7352.287451280116, "mittel_b": 12461.751651472263, "diff_ki95": [-6853.972952065235, -3310.3666921916215], "p": 0.001953125, "richtung_wie_erwartet": true, "p_holm": 0.0078125}
Gate 0b: BESTANDEN

## Zielaufgabe mul
random n = 10 Mittel kbit: 26.97
wd n = 10 Mittel kbit: 40.56
wd_scaled n = 10 Mittel kbit: 42.48
wd_permuted n = 10 Mittel kbit: 40.66
mdl n = 10 Mittel kbit: 43.47
bitnet n = 10 Mittel kbit: 42.93
Transfer-Gewinn T(wd) Mittel kbit: -13.59
Transfer-Gewinn T(mdl) Mittel kbit: -16.5
Transfer-Gewinn T(bitnet) Mittel kbit: -15.96
{"kontrast": "H1 Transfer: wd-A < Zufall", "n": 10, "mittel_a": 40564.44732181549, "mittel_b": 26971.84436168612, "diff_ki95": [9289.197897676255, 17429.02510490346], "p": 0.001953125, "richtung_wie_erwartet": false, "p_holm": 0.0078125}
{"kontrast": "H2 Struktur: wd-A < wd-A permutiert", "n": 10, "mittel_a": 40564.44732181549, "mittel_b": 40656.55225612241, "diff_ki95": [-5184.412657943839, 5115.978212333352], "p": 0.76953125, "richtung_wie_erwartet": true, "p_holm": 1.0}
{"kontrast": "H3 nicht nur Norm: wd-A < wd-A Norm-skaliert", "n": 10, "mittel_a": 40564.44732181549, "mittel_b": 42476.019238977104, "diff_ki95": [-6246.784440564514, 1850.5829184634188], "p": 1.0, "richtung_wie_erwartet": true, "p_holm": 1.0}
{"kontrast": "H4 komprimiert überträgt besser: mdl-A < wd-A", "n": 10, "mittel_a": 43469.620886270175, "mittel_b": 40564.44732181549, "diff_ki95": [-1080.1167551144401, 7643.16455856056], "p": 0.6953125, "richtung_wie_erwartet": false, "p_holm": 1.0}
```
