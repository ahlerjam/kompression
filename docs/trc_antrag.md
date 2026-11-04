# Antrag Google TPU Research Cloud (ENTWURF, noch nicht einreichen)

Stand: 2026-11-03 19:56. Gerüst ohne Ergebnisse; die Platzhalter `[[…]]` werden erst nach der Auswertung von 0a (`docs/ergebnis_0a.md`) und 0b gefüllt. Nur eine Einreichung pro Person → Jonas reicht erst ein, wenn Claude den fertigen Text meldet.

Formularfelder: Email `account@ahler.org` · First/Last Name · Affiliation (independent researcher) · Job title · Country · Project Description (unten) · How did you hear.

## Project Description (Entwurf, englisch)

**Compression-driven self-improvement: measuring generalization and transfer in bits.**

We test whether pressure toward short descriptions (Minimum Description Length) makes neural networks generalize and transfer better, and we measure every claim as a code length in bits rather than as accuracy alone.

*What we have done (single Apple-silicon machine, MLX):* a pre-registered experiment on grokking (modular arithmetic, p = 97, 10 paired seeds) comparing a variational MDL objective, weight decay, a norm-matched weight-decay control, ternary (BitNet-style) weights and no regularization. Primary endpoint: prequential code length of the full task (Blier & Ollivier 2018) with per-block restart and paid hyperparameter selection. [[Ergebnis 0a: H1/H4 mit Effektgröße, p (Holm), auch wenn negativ]]. Transfer stage: [[Ergebnis 0b: bits saved on subtraction/multiplication from addition weights vs. permuted and norm-scaled controls]].

*What TPUs would enable:* [[Folgeexperiment nach Entscheidung Jonas: z. B. Skalierung p = 97/197/397, mehr Seeds, Switch-Code/Konsolidierung, β wachsend mit Datenmenge]]. These runs are compute-bound (full-batch, many seeds × hyperparameter grids × restarts per block) and do not fit on one laptop GPU. We will port the code to JAX (the model is a 2-layer transformer; the training step is already vmapped over seeds).

*Openness:* protocols are frozen before main runs, negative results are reported, code and results will be published.

## Offene Punkte vor dem Einreichen

- [ ] Ergebnisse 0a eingetragen (aus `docs/ergebnis_0a.md`, unverändert)
- [ ] Ergebnisse 0b eingetragen
- [ ] Folgeexperiment mit Jonas festgelegt
- [ ] Code-Veröffentlichung geklärt (Repo ist privat)
