# Protokoll Stufe 0b (Version 2, ENTWURF; wird mit dem Start des Hauptlaufs 0b eingefroren)

Stand: 2026-11-03 22:20. Version 2 ersetzt Version 1 (unten unverändert erhalten). Entscheidung datenbasiert durch Claude im Auftrag von Jonas.

## Änderungen gegenüber Version 1 (Grundlage: Kalibrierung `0b4_*`, Seeds 0–2, Subtraktion, A = `kalib_wd`)

**Neustart je Block: ja** (wie im eingefrorenen Protokoll 0a). Prequentielle Codelänge in kbit (uniform 62,1):

| S | Zufall (Neustart) | A-Init ohne Neustart (`0b3`) | **A-Init mit Neustart (`0b4`)** | A-permutiert mit Neustart |
|---|---|---|---|---|
| 30 | 62,1 | 35,0 / 53,7 / 43,8 | 61,8 / 61,3 / 53,3 | 62,1 |
| 100 | 62,1 | 45,8 / 55,7 / 54,0 | 47,8 / 44,7 / 24,0 | 62,1 |
| 300 | 62,1 | 56,0 / 61,0 / 60,3 | **19,8 / 22,1 / 17,7** | 57,4 / 57,9 / 57,7 |

Minimum über S je Seed: mit Neustart 17,7–22,1 kbit, ohne 35,0–53,7. Ohne Neustart driftet das Modell über die Blöcke von der A-Lösung weg (bei größerem S schlechter); mit Neustart startet jeder Block wieder von A. Die permutierte Kontrolle spart bei S = 300 etwas (≈ 4,5 kbit, Gewichtsstatistik), bleibt aber weit über A-Init. Kalibrierung ist kein Befund.

**Weitere Festlegungen:**
- Seeds 10–19; A-Gewichte aus `H_wd`, `H_mdl`, `H_bitnet` des Hauptlaufs 0a (Endgewichte, gleiche Seeds).
- Lerner auf B: `wd` λ = 1, lr 1e-3, Neustart je Block, S ∈ {30, 100, 300, 1.000, 3.000}, Codelänge = Minimum über S + log2(5) Bit.
- Bedingungen: Zufall · A(wd) · A(wd)-permutiert · A(wd)-Norm-skaliert · A(mdl) · A(bitnet). Kontrollen nur für A(wd) (Rechenzeit); H4 vergleicht A(mdl) mit A(wd).
- Aufgaben: Subtraktion (Gate), danach Multiplikation (zweites Ziel); Holm je Zielaufgabe über H1–H4.
- Auswertung: `stages/s0a_grokking/analyse_0b.py` (Tests `tests/test_analyse_0b.py`); Start: `NEUSTART=1 scripts/hauptlauf_0b.sh 1` und `… 2`.
- Gate 0b unverändert: H1 und H2 Holm-adjustiert p < 0,05 für Subtraktion.

---

# Protokoll Stufe 0b (Version 1, ENTWURF, ersetzt durch Version 2 oben)

Stand: 2026-11-02. Wird mit dem Start des Hauptlaufs eingefroren.

**Frage:** Lässt sich Transfer zwischen Aufgaben als eingesparte Bits messen, und unterscheidet die Messung echte Struktur von bloßer Gewichtsstatistik?

## Kalibrierung (Grundlage, kein Befund)

Lerner `wd`, Aufgabe A = Addition mod 97 (Gewichte aus `kalib_wd`), Aufgabe B = Subtraktion mod 97, Seeds 0–2, prequential v2 (`runs/s0a_preq/*0b*`):

| S (Schritte/Block) | Zufall | A-Init | A-permutiert |
|---|---|---|---|
| 30 | 62,1 kbit | 35,0 / 53,7 / 43,8 | 62,1 |
| 100 | 62,1 | 45,8 / 55,7 / 54,0 | 62,1 |
| 300 | 62,1 / 62,1 / 54,8 | 56,0 / 61,0 / 60,3 | 62,0 |
| 3.000 | 61,2–61,4 | 62,0 | 61,4–61,9 |

Folgerungen: Transfer ist messbar und strukturell (die permutierte Kontrolle spart nichts); er hängt stark von S ab → S ist globaler Hyperparameter mit bezahlter Wahl.

## Aufbau Hauptlauf

- A-Gewichte: aus dem Hauptlauf 0a (Arme `wd`, `mdl`, `bitnet`; Seeds 10–19), jeweils Endgewichte.
- Aufgabe B: Subtraktion mod 97; zusätzlich Multiplikation mod 97 (zweite Zielaufgabe, weiter entfernt).
- Lerner auf B: `wd` (λ = 1, lr 1e-3), für alle Bedingungen gleich.
- Bedingungen je Seed: Zufallsstart · A-Init · A-Norm-skaliert · A-permutiert.
- Prequential v2, ganze B-Tabelle, Blöcke ab 64; S ∈ {30, 100, 300, 1.000, 3.000}, Codelänge = Minimum über S + log2(5) Bit.

## Endpunkte und Hypothesen

- **Transfer-Gewinn** T = L_preq(B | Zufall) − L_preq(B | A-Init) in Bit, je Seed.
- H1: T > 0 (A-Init codiert B kürzer als Zufall).
- H2: L_preq(B | A-Init) < L_preq(B | A-permutiert) (Struktur, nicht Statistik).
- H3: L_preq(B | A-Init) < L_preq(B | A-Norm-skaliert) (nicht nur Norm).
- H4 (Brücke zu Stufe 1): T(A aus `mdl`) > T(A aus `wd`) — komprimiertere A-Lösungen übertragen besser.
- Tests: Wilcoxon gepaart (10 Seeds), Bootstrap-KI, Holm über H1–H4 je Zielaufgabe.
- **Gate 0b:** H1 und H2 Holm-adjustiert p < 0,05 für Subtraktion.

## Grenzen

- Ein Paar verwandter Aufgaben; Multiplikation als zweites, weiter entferntes Ziel.
- Im Gesamtziel des Konzepts muss L(A-Gewichte) mitgezählt werden; hier wird nur der B-Teil gemessen (bedingte Codelänge).
