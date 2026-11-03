# Protokoll Stufe 0b (Version 1, ENTWURF zur Freigabe)

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
