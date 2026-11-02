# Stufe-1-Pilot (Entwurf)

**Frage:** Wählt ein Proposer, der für Kompressionsfortschritt belohnt wird, bessere Aufgaben als einer, der für Learnability belohnt wird, gemessen an Transfer-Bits und Pass-Rate auf zurückgehaltenen Familien, bei gleichem Compute?

## Bausteine

| Teil | Umsetzung | Stand |
|---|---|---|
| Welten | Reasoning Gym, 16 Trainings- und 8 zurückgehaltene Familien (`s1/worlds.py`), immer frisch per Seed erzeugt | gebaut |
| Sandbox | Subprozess mit Zeit- und CPU-Limit (`s1/sandbox.py`); echte Isolation offen (E17) | Pilot |
| Bibliothek | Python-Funktionen, L(B) = Tokens × log2(V) (`s1/library.py`) | gebaut |
| Modell | eingefroren, klein für den Piloten (Qwen, 4 Bit, mlx-lm); Durchsatz selbst messen | offen |
| Solver | Modell schreibt ein Programm, das die Antwort druckt; Bibliothek steht im Prompt (Prefix-Cache) | offen |
| L(y \| x, B) | Logprob des verifizierten Programms unter dem Modell, Bibliothek im Prompt, in Bit | offen |
| Schlafphase | Modell extrahiert Hilfsfunktionen aus gelösten Programmen (ReGAL-artig); Annahme nur, wenn L_gesamt auf dem Probe-Set sinkt | offen |
| Proposer | im Piloten eine **Auswahlpolitik** über (Familie, Schwierigkeit), kein Textgenerator; Bandit mit Belohnung A oder B | offen |

## Ziel und Belohnungen

- L_gesamt(B) = L(B) + Σ über das Probe-Set von L(y | x, B), y = kürzeste verifizierte Lösung.
- **A (Kompression):** Belohnung einer Aufgabenwahl = Abnahme von L_gesamt auf dem versiegelten Probe-Set nach der Schlafphase, mit Vorzeichen, ohne Clipping (E13).
- **B (Learnability):** 1 − r̄ bei 0 < r̄ < 1, sonst 0 (Absolute Zero).
- Gleiches Budget an Modell-Tokens für A und B (E15).

## Kontrollen (E14)

1. Leere Bibliothek, gleiches Token-Budget.
2. Self-Consistency: gleiches Budget für mehr Samples statt Bibliothek.
3. Zufällige Aufgabenwahl.

## Messung

- Transfer: L_gesamt auf zurückgehaltenen Familien mit gelernter Bibliothek minus ohne.
- Pass-Rate auf zurückgehaltenen Familien.
- Wiederverwendung: pro Funktion Anzahl verschiedener Lösungen auf zurückgehaltenen Aufgaben, die sie aufrufen.
