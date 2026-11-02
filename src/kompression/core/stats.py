"""Gepaarte Statistik für vorab registrierte Vergleiche (Entscheidung E11)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as _st


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    """Zweiseitiger Wilcoxon-Vorzeichen-Rang-Test auf a - b (gepaart über Seeds)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("gepaarte Arrays brauchen gleiche Form")
    d = a - b
    res = _st.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"n": int(d.size), "median_diff": float(np.median(d)), "p": float(res.pvalue)}


def bootstrap_ci(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Perzentil-Bootstrap-KI für den Mittelwert der gepaarten Differenz."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni-adjustierte p-Werte (gleiche Reihenfolge wie Eingabe)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
