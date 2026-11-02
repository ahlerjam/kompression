"""L(W): Beschreibungslänge der Gewichte nach festem Rezept (Entscheidungen E8, E27).

Rezept `uq-entropy-v3` (unabhängig vom Trainingsziel, in allen Armen identisch):
1. Zwei Beschreibungen werden geprüft:
   - absolut:  W selbst wird quantisiert.
   - Differenz: W = W0(seed) + D; nur D wird quantisiert. W0 ist aus dem Seed
     reproduzierbar und kostet SEED_BITS.
2. Gleichförmige Quantisierung mit Schrittweite delta (Indizes round(x/delta)).
   Kosten = empirische Entropie der Indizes * Anzahl + Codebuch (64 Bit je Symbol)
   + 64 Bit für delta.
3. delta aus festem Gitter; zulässig, wenn loss(rekonstruiert) <= loss(original) + tolerance.
   Pro Beschreibung das Minimum der Bits; L(W) = Minimum aller Beschreibungen.
4. (v3) Zusätzliche strukturelle Beschreibungen eines Arms (z. B. Unterraum: nur Z;
   BitNet: ternäre Indizes) werden wie oben quantisiert und codiert. Der Kern
   rekonstruiert die Gewichte selbst über `decode` und prüft den Loss; die Kosten
   des Decoders selbst (Seed, Skalen) trägt der Arm als `overhead_bits`.

Low-Rank-Vergröberung (DeMoss et al.) ist bewusst nicht enthalten; sie käme als
eigene, versionierte Rezept-Variante.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

RECIPE_VERSION = "uq-entropy-v3"
SEED_BITS = 32.0
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


def _search(
    target: list[np.ndarray],
    decode: Callable[[list[np.ndarray]], list[np.ndarray]],
    loss_fn: Callable[[list[np.ndarray]], float],
    base: float,
    tolerance: float,
    deltas: Sequence[float],
    overhead: float,
) -> dict | None:
    best = None
    for delta in sorted(deltas):
        q = dequantize(quantize(target, delta), delta)
        loss_q = float(loss_fn(decode(q)))
        if loss_q <= base + tolerance:
            bits = weight_bits_at(target, delta) + overhead
            if best is None or bits < best["bits"]:
                best = {"bits": bits, "delta": delta, "loss_quant": loss_q}
    return best


def weight_bits(
    weights: Sequence[np.ndarray],
    loss_fn: Callable[[list[np.ndarray]], float],
    tolerance: float,
    deltas: Sequence[float] = DEFAULT_DELTAS,
    init: Sequence[np.ndarray] | None = None,
    structural: Sequence[dict] = (),
) -> dict:
    """L(W) in Bit als Minimum über alle gültigen Beschreibungen.

    `init` (= W0 aus dem Seed) aktiviert die Differenz-Beschreibung.
    `structural`: Liste von {"name", "payload": [arrays], "decode": f(payload) -> Gewichte,
    "overhead_bits": float}. Der Kern quantisiert die Payload und prüft decode(payload).
    """
    w = [np.asarray(x, dtype=np.float64) for x in weights]
    base = float(loss_fn(w))
    found: dict[str, dict | None] = {}
    found["absolute"] = _search(w, lambda q: q, loss_fn, base, tolerance, deltas, 0.0)
    if init is not None:
        w0 = [np.asarray(x, dtype=np.float64) for x in init]
        d = [a - b for a, b in zip(w, w0, strict=True)]
        found["diff"] = _search(
            d, lambda q: [o + x for o, x in zip(w0, q, strict=True)],
            loss_fn, base, tolerance, deltas, SEED_BITS,
        )
    for cand in structural:
        payload = [np.asarray(x, dtype=np.float64) for x in cand["payload"]]
        found[cand["name"]] = _search(
            payload, cand["decode"], loss_fn, base, tolerance, deltas, float(cand["overhead_bits"])
        )
    valid = {k: v for k, v in found.items() if v is not None}
    if not valid:
        raise RuntimeError("Kein delta im Gitter erfüllt die Toleranz")
    name = min(valid, key=lambda k: valid[k]["bits"])
    return {
        "bits": valid[name]["bits"],
        "description": name,
        "candidates": found,
        "absolute": found.get("absolute"),
        "diff": found.get("diff"),
        "loss_base": base,
        "tolerance": tolerance,
        "recipe": RECIPE_VERSION,
    }
