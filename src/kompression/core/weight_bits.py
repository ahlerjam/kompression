"""L(W): Beschreibungslänge der Gewichte nach festem Rezept (Entscheidung E8).

Rezept (unabhängig vom Trainingsziel, in allen Armen identisch):
1. Alle Gewichte gleichförmig quantisieren mit Schrittweite delta (Indizes round(w/delta)).
2. Kosten = empirische Entropie der Indizes * Anzahl + Codebuch (je Symbol 32 Bit
   für den Indexwert + 32 Bit für die Häufigkeit) + 64 Bit für delta.
3. delta wird aus einem festen Gitter gewählt: das größte delta, bei dem der Loss
   des quantisierten Modells höchstens um `tolerance` steigt. Dann minimale Bits.

Low-Rank-Vergröberung (DeMoss et al.) ist bewusst noch nicht enthalten; sie kommt
als eigene, versionierte Rezept-Variante, falls nötig.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

RECIPE_VERSION = "uq-entropy-v1"
DEFAULT_DELTAS: tuple[float, ...] = tuple(float(x) for x in np.geomspace(1e-4, 1.0, 41))


def quantize(weights: Sequence[np.ndarray], delta: float) -> list[np.ndarray]:
    return [np.round(np.asarray(w, dtype=np.float64) / delta).astype(np.int64) for w in weights]


def dequantize(indices: Sequence[np.ndarray], delta: float) -> list[np.ndarray]:
    return [idx.astype(np.float64) * delta for idx in indices]


def entropy_bits(indices: Sequence[np.ndarray]) -> float:
    flat = np.concatenate([i.ravel() for i in indices])
    if flat.size == 0:
        return 0.0
    _, counts = np.unique(flat, return_counts=True)
    p = counts / flat.size
    payload = float(-(counts * np.log2(p)).sum())
    codebook = 64.0 * len(counts)
    return payload + codebook + 64.0


def weight_bits_at(weights: Sequence[np.ndarray], delta: float) -> float:
    return entropy_bits(quantize(weights, delta))


def weight_bits(
    weights: Sequence[np.ndarray],
    loss_fn: Callable[[list[np.ndarray]], float],
    tolerance: float,
    deltas: Sequence[float] = DEFAULT_DELTAS,
) -> dict:
    """Minimale Bits über das Gitter unter der Bedingung loss(quant) <= loss(orig) + tolerance."""
    base = float(loss_fn([np.asarray(w, dtype=np.float64) for w in weights]))
    best = None
    for delta in sorted(deltas):
        q = dequantize(quantize(weights, delta), delta)
        loss_q = float(loss_fn(q))
        if loss_q <= base + tolerance:
            bits = weight_bits_at(weights, delta)
            if best is None or bits < best["bits"]:
                best = {"bits": bits, "delta": delta, "loss_quant": loss_q}
    if best is None:
        raise RuntimeError("Kein delta im Gitter erfüllt die Toleranz")
    best.update({"loss_base": base, "tolerance": tolerance, "recipe": RECIPE_VERSION})
    return best
