# Antrag Google TPU Research Cloud (ENTWURF, noch nicht einreichen)

Stand: 2026-11-04 06:00. 0a und 0b-Subtraktion eingetragen (Zahlen aus `docs/ergebnis_0a.md`, `docs/ergebnis_0b.md`); offen: Multiplikation, Folgeexperiment, Code-Veröffentlichung. Nur eine Einreichung pro Person → Jonas reicht erst ein, wenn Claude den fertigen Text meldet.

Formularfelder: Email `account@ahler.org` · First/Last Name · Affiliation (independent researcher) · Job title · Country · Project Description (unten) · How did you hear.

## Project Description (Entwurf v2, englisch; Stand 2026-11-04 06:00)

**Compression-driven learning: measuring generalization and transfer in bits.**

We test whether pressure toward short descriptions (Minimum Description Length, MDL) makes neural networks generalize and transfer better. Every claim is measured as a code length in bits, not as accuracy alone, and all protocols are frozen before the main runs.

*Results so far (one Apple-silicon laptop, MLX, 2-layer transformer, modular arithmetic mod 97, 10 paired seeds, Wilcoxon tests with Holm correction):*

1. **Learning one task (pre-registered, partly negative).** We compared a variational MDL objective, weight decay, a norm-matched weight-decay control, ternary (BitNet-style) weights and no regularization. Primary endpoint: prequential code length of the full addition table (Blier & Ollivier 2018, restart per block, paid hyperparameter selection). Both primary hypotheses failed: MDL coded the data *longer* than weight decay (32.5 vs. 26.5 kbit, Holm p = 0.004), ternary weights slightly longer (27.1 vs. 26.5 kbit, p = 0.006). Secondary results held: MDL found ~9× shorter weights (143 vs. 1,288 kbit; also shorter than the norm-matched control, so it is not just the norm), grokked more stably than the norm-matched control, and ternary weights coded the last, data-rich blocks almost exactly (137 vs. 8,599 bits). Compression loses early (little data) and wins late.

2. **Transfer in bits (pre-registered, positive).** Starting from addition weights, we measured the prequential code length of subtraction. Addition weights from weight decay cut the code length from 29.3 to 12.5 kbit; permuted and norm-matched controls were worse than random (54.4 kbit), so the transfer is structural. **Weights from the MDL run transferred better (7.4 kbit) than weights from weight decay (12.5 kbit)**; all four hypotheses held in all 10 seeds (Holm p = 0.008 each). [[Multiplikation: Ergebnis eintragen]]

Together: the MDL solution is worse at coding its own task's data stream but is much shorter and transfers better — the first evidence in this project for "compression → transfer". The result is limited to one task family and one model size.

*What TPUs would enable:* [[Folgeexperiment nach Entscheidung Jonas, z. B.: (a) scaling p = 97/197/397 and model width to test whether the late advantage of compression grows with scale; (b) data-dependent compression pressure and switching codes (van Erven et al. 2007) to remove the early-phase penalty; (c) more task pairs and seeds for transfer.]] These runs are compute-bound (full-batch training, many seeds × hyperparameter grids × restarts per block; one condition takes 2–3 h on our laptop, the full study ~2 days) and do not scale on one laptop GPU. The model is functional (parameters as dicts, training vmapped over seeds), so a JAX port is direct.

*Openness:* protocols are frozen before main runs and versioned; negative results are reported alongside positive ones; [[Code-Veröffentlichung nach Entscheidung Jonas]].

## Offene Punkte vor dem Einreichen

- [x] Ergebnisse 0a eingetragen (aus `docs/ergebnis_0a.md`, unverändert)
- [ ] Ergebnisse 0b eingetragen (Subtraktion erledigt, Multiplikation offen)
- [ ] Folgeexperiment mit Jonas festgelegt
- [ ] Code-Veröffentlichung geklärt (Repo ist privat)
