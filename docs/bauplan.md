# Bauplan Stufe 0

Ziel: in 3–5 Wochen beantworten, ob ein MDL-Ziel beim Generalisieren mehr hilft als Weight Decay (0a) und ob sich Transfer in Bits messen lässt (0b). Alles auf dem MacBook Pro M4 Pro, 48 GB.

## Repo-Struktur

```
kompression/
  core/                 unantastbarer Kern (nur Mensch ändert; Prüfsummen)
    codelength.py       Bits aus Logprobs, prequentielle Codelänge (numpy, float64)
    weight_bits.py      L(W)-Rezept: Low-Rank + Quantisierung + Entropiecodierung
    stats.py            gepaarte Tests, Bootstrap-KI, Holm
    budget.py           Compute-Budgets pro Bedingung
    MANIFEST.sha256     Prüfsummen über core/
  kompression/          lernbarer Code
    model_mlx.py        dichter Transformer, StableMax
    model_torch.py      Spiegel für Paritätstest und Cloud
    train.py            vmap über Seeds, compile, Checkpoints
    objectives.py       Weight Decay, variationales MDL (GMM-Prior), Norm-Projektion
    runlog.py           run_dir, config/metrics/meta
  stages/s0a_grokking/  Configs, Protokoll, Auswertung
  stages/s0b_transfer/
  tests/                pytest
  runs/                 Ergebnisse (nicht in Git)
  docs/                 Entscheidungen, Recherche, Protokolle
  CLAUDE.md             Regeln für Claude
```

## Schritte

| # | Schritt | Fertig, wenn |
|---|---|---|
| 1 | Werkzeuge: `uv`, Python 3.12, `tmux`; Pakete mlx, mlx-lm, numpy, torch (CPU), pytest, tyro, matplotlib, polars | `uv run pytest` läuft leer durch |
| 2 | `core/codelength.py` + Tests (uniform = log2 V; prequential von Hand nachgerechnet) | Tests grün |
| 3 | Modell + Training in MLX, StableMax, vmap über Seeds | Overfit-one-batch, Determinismus, vmap-Äquivalenz grün |
| 4 | Paritätstest MLX ↔ PyTorch | Logits gleich bis 1e-4 |
| 5 | Smoke-Test: p = 97, 1 Seed, Arm „ohne Regularisierung“ | groked, Pipeline < 30 s im Kurzmodus |
| 6 | `core/weight_bits.py` (L(W)-Rezept), ε und Suchgitter festgelegt | stabil über Wiederholungen |
| 7 | MDL-Ziel (GMM-Prior) + Norm-Projektion | Arme (iii) und (iv) laufen |
| 8 | **Protokoll 0a einfrieren** (Arme, Endpunkte, Tests), dann ≥ 10 Seeds × 5 Arme | Gate 0a entschieden |
| 9 | 0b: prequentielle Codelänge mit Blöcken 64 … 9409, vier Init-Bedingungen | Gate 0b entschieden |
| 10 | Bericht: Workshop-Beitrag, positiv oder negativ | |

## Arme für 0a

Alle mit StableMax, AdamW, gleicher Lernrate, gleichen Daten-Splits und Init pro Seed.

1. ohne Regularisierung
2. Weight Decay (λ per kleinem Sweep auf separatem Seed)
3. variationales MDL mit GMM-Prior
4. Weight Decay, Norm auf die Trajektorie von 3 projiziert
5. optional: MDL, Norm auf die Trajektorie von 2 projiziert

Endpunkte: Grok-Schritt (log-skaliert; Nicht-Grokking als zensiert), L(W) in Bit nach festem Rezept, Test-Accuracy.

## Betrieb auf dem Mac

- Läufe in `tmux`, Checkpoints mit RNG- und Optimizer-Zustand, damit sie nach Unterbrechung weiterlaufen.
- Remote Control als `launchd`-Dienst; Weiterlaufen bei geschlossenem Deckel über `pmset` (braucht sudo, separat).
- Unter ~40 GB aktivem Speicher bleiben, damit nichts swappt.
