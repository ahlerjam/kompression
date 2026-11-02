# STATUS

**Phase: BAUPHASE** (Regeln in AGENTS.md)

Zuletzt aktualisiert: 2026-11-02 (Loop-Durchgang 1)

## Wo wir stehen

- Stufe 0a: Arme `none`, `wd`, `mdl`, `bitnet`, `subspace` und Norm-Projektion (E10) gebaut; 31 Tests grün.
- Kalibrierung: `none` groked in 20k nicht; `wd` groked bei ~2,3k, aber instabil bis ~17–20k → Budget 40k, Endpunkt „stabil gegrokt“ (Protokoll angepasst).
- Stufe-1-Pilot: Welten, Sandbox (isolierte VM), Bibliothek, eingefrorenes LM, Solver mit Prefill und Bits; erste Messung mit Qwen2.5-Coder-1.5B im Logbuch.

## Läuft gerade

| tmux | Lauf |
|---|---|
| `s0a-mdl` | Kalibrierung `mdl`, β = 0,001 → 0,01 → 0,1 → 1 (je 3 Seeds × 20k, je ca. 1 h) |
| `s0a-struct` | Kalibrierung `subspace` Rang 16 → 4 → 64, dann `bitnet` (je 3 Seeds × 20k) |
| `dl` | Download Qwen3.5-4B und Qwen3.6-35B-A3B |
| `rc` | Remote Control „mac-ai“ |

## Nächste Schritte (Loop)

1. Kalibrierungen auswerten → β, Rang, ε für L(W) ins Protokoll; `wd_normproj` gegen `mdl`-Trajektorie kalibrieren.
2. Protokoll 0a vervollständigen → Freigabe Hauptlauf (10 Seeds × 40k je Arm) bei Jonas anfragen.
3. Pilot: Schlafphase (Funktionen aus gelösten Programmen extrahieren, Annahme nur bei sinkendem L_gesamt), L(B) zusätzlich als −log2 p_Modell(B) messen, Bandit-Proposer A/B, Kontrollen.

## Wartet auf Jonas

- **Hauptlauf 0a starten** (Wechsel in die Messphase), sobald Kalibrierungen ausgewertet und Protokoll ausgefüllt sind.
- Optional: Cloud-GPU für Stufe 1 (Kosten grob 50–200 €).

## Bekannte Befunde Engineering

- Metal ist nicht bitgenau deterministisch (ca. 6e-8).
- GPU ist mit einem Lauf fast ausgelastet; zwei parallele Läufe bringen nur ca. 15 % mehr Durchsatz.
- Echtes MDL (β = 1) lernt bei 0a praktisch nicht: Die Daten tragen nur ca. 25 kbit, das Modell hat ca. 400k Gewichte.
