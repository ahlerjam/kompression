"""Codelängen in Bit. Reines numpy, float64.

Konventionen:
- Eingaben sind natürliche Log-Wahrscheinlichkeiten (nats) der *korrekten* Symbole.
- Ausgaben sind Bit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

LN2 = np.log(2.0)


def bits_from_logprobs(logp_nat: np.ndarray | Sequence[float]) -> float:
    """Summe der Codelänge -log2 p über alle Symbole."""
    arr = np.asarray(logp_nat, dtype=np.float64)
    if np.any(arr > 1e-9):
        raise ValueError("Log-Wahrscheinlichkeiten müssen <= 0 sein")
    return float(-arr.sum() / LN2)


def uniform_bits(n_symbols: int, alphabet_size: int) -> float:
    """Codelänge von n Symbolen unter dem gleichförmigen Code."""
    if alphabet_size < 1 or n_symbols < 0:
        raise ValueError("ungültige Argumente")
    return float(n_symbols * np.log2(alphabet_size))


def doubling_blocks(n_total: int, first: int) -> list[int]:
    """Blockgrenzen 0 < first < 2*first < ... < n_total (letzte Grenze = n_total)."""
    if not 0 < first <= n_total:
        raise ValueError("first muss in (0, n_total] liegen")
    bounds = [first]
    while bounds[-1] < n_total:
        bounds.append(min(2 * bounds[-1], n_total))
    return bounds


def prequential_codelength(
    n_total: int,
    alphabet_size: int,
    block_bounds: Sequence[int],
    encode_block: Callable[[int, int], np.ndarray],
) -> dict:
    """Prequentielle Codelänge (Blier & Ollivier 2018).

    Der erste Block wird gleichförmig codiert. Für jeden weiteren Block
    [start, end) liefert `encode_block(start, end)` die Log-Wahrscheinlichkeiten
    (nats) der korrekten Labels unter einem Modell, das nur auf [0, start)
    trainiert wurde. Die Verantwortung für dieses "nur auf dem Präfix"
    liegt beim Aufrufer; hier wird nur gezählt.
    """
    bounds = list(block_bounds)
    if bounds[-1] != n_total or any(b <= a for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Blockgrenzen müssen streng steigen und bei n_total enden")
    per_block = [uniform_bits(bounds[0], alphabet_size)]
    for start, end in zip(bounds, bounds[1:]):
        logp = np.asarray(encode_block(start, end), dtype=np.float64)
        if logp.shape != (end - start,):
            raise ValueError(f"Block [{start},{end}) liefert Form {logp.shape}")
        per_block.append(bits_from_logprobs(logp))
    total = float(sum(per_block))
    return {
        "total_bits": total,
        "per_block_bits": per_block,
        "bounds": bounds,
        "uniform_bits": uniform_bits(n_total, alphabet_size),
    }
