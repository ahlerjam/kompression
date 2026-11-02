# STATUS

**Phase: BAUPHASE** (Regeln in AGENTS.md)

Zuletzt aktualisiert: 2026-11-02 (Loop-Durchgang 1)

## Wo wir stehen

- Stufe 0a: alle Arme gebaut (`none`, `wd`, `mdl`, `bitnet`, `subspace`), L(W) mit Seed+Differenz (E27), 27 Tests grün.
- Stufe-1-Pilot: Welten (Reasoning Gym), Sandbox, Bibliothek mit L(B) gebaut; Entwurf in `docs/stufe1_pilot.md`.

## Läuft gerade

| tmux | Lauf | Stand |
|---|---|---|
| `s0a-kalib` | Kalibrierung `none`, 5 Seeds × 20k | bei 12k kein Grokking (Test ≈ 2 %, Norm ≈ 470) |
| `s0a-wd` | Kalibrierung `wd` (λ = 1), 5 Seeds × 20k | bei 6,1k: 4/5 Seeds groked (Test ≥ 0,97), Seed 2 noch nicht |
| `rc` | Remote Control „mac-ai“ | verbunden |

Hinweis: `s0a-kalib` würde nach `none` noch einmal `wd` starten; ein Wächter beendet die Sitzung nach `none` (Dublette).

## Nächste Schritte (Loop)

1. Kalibrierung `none`/`wd` auswerten → Schrittbudget und λ ins Protokoll.
2. GPU: Kalibrierung `mdl` über β ∈ {1; 0,1; 0,01; 0,001} (Seeds 100–102), dann `subspace` (Rang 4/16/64) und `bitnet`.
3. Pilot: Modell wählen und Durchsatz messen (wenn GPU frei), Solver mit Logprob-Messung, Schlafphase, Bandit-Proposer.

## Wartet auf Jonas

- **Hauptlauf 0a starten** (Wechsel in die Messphase), sobald Kalibrierungen ausgewertet und Protokoll ausgefüllt sind.
- Optional: Cloud-GPU für Stufe 1 (Kosten grob 50–200 €).

## Bekannte Befunde Engineering

- Metal ist nicht bitgenau deterministisch (ca. 6e-8).
- GPU ist mit einem Lauf fast ausgelastet; zwei parallele Läufe bringen nur ca. 15 % mehr Durchsatz.
- Echtes MDL (β = 1) lernt bei 0a praktisch nicht: Die Daten tragen nur ca. 25 kbit, das Modell hat ca. 400k Gewichte.
