"""Brücke zwischen trainierten MLX-Gewichten und dem Kern-Rezept für L(W)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from kompression.core.weight_bits import SEED_BITS, weight_bits
from kompression.model_mlx import forward, label_logprobs
from kompression.train import TrainConfig, unstack_tree


def structural_candidates(cfg: TrainConfig, tr: dict, const: dict, p: dict) -> list[dict]:
    """Strukturelle Beschreibungen je Arm; der Kern prüft sie per Rekonstruktion."""
    keys = [k for k, _ in tree_flatten(p)]
    if cfg.arm == "subspace":
        w0 = dict((k, np.array(v, dtype=np.float64)) for k, v in tree_flatten(const["w0"]))
        A = {k: np.array(v, dtype=np.float64) for k, v in const["A"].items()}
        B = {k: np.array(v, dtype=np.float64) for k, v in const["B"].items()}
        zk = sorted(tr["z"])
        payload = [np.array(tr["z"][k]) for k in zk]

        def decode(q, keys=keys, zk=zk, w0=w0, A=A, B=B):
            z = dict(zip(zk, q, strict=True))
            return [w0[k] + A[k] @ z[k] @ B[k].T if k in z else w0[k] for k in keys]

        return [{"name": "subspace", "payload": payload, "decode": decode,
                 "overhead_bits": SEED_BITS}]
    if cfg.arm == "bitnet":
        flat = [np.array(v, dtype=np.float64) for _, v in tree_flatten(p)]
        scales, payload = [], []
        for w in flat:
            if w.ndim == 2:
                nz = np.abs(w[w != 0])
                a = float(nz.mean()) if nz.size else 1.0
                scales.append(a)
                payload.append(np.round(w / a))
            else:
                scales.append(None)
                payload.append(w)

        def decode(q, scales=scales):
            return [x * a if a is not None else x for x, a in zip(q, scales, strict=True)]

        n_mat = sum(a is not None for a in scales)
        return [{"name": "ternary", "payload": payload, "decode": decode,
                 "overhead_bits": 32.0 * n_mat}]
    return []


def weight_bits_per_seed(cfg: TrainConfig, result: dict, data: dict, tolerance: float) -> dict:
    """L(W) je Seed; Toleranz bezieht sich auf den mittleren Trainings-Loss (nats/Beispiel)."""
    mcfg = cfg.model()
    out = {}
    for i, seed in enumerate(cfg.seeds):
        p = unstack_tree(result["params"], i)
        w0 = unstack_tree(result["w0"], i)
        keys = [k for k, _ in tree_flatten(p)]
        x = mx.array(data["x_train"][i])
        y = mx.array(data["y_train"][i])

        def loss_fn(ws, keys=keys, x=x, y=y):
            tree = tree_unflatten([(k, mx.array(np.asarray(w, dtype=np.float32)))
                                   for k, w in zip(keys, ws, strict=True)])
            lp = label_logprobs(forward(tree, x, mcfg), y, cfg.use_stablemax)
            return float(-lp.mean())

        ws = [np.array(v) for _, v in tree_flatten(p)]
        w0s = [np.array(v) for _, v in tree_flatten(w0)]
        struct = []
        if "trainable" in result:
            struct = structural_candidates(
                cfg, unstack_tree(result["trainable"], i), unstack_tree(result["const"], i), p)
        r = weight_bits(ws, loss_fn, tolerance, init=w0s, structural=struct)
        out[seed] = {
            "bits": r["bits"],
            "description": r["description"],
            **{f"{k}_bits": (v["bits"] if v else None) for k, v in r["candidates"].items()},
            "recipe": r["recipe"],
            "tolerance": tolerance,
        }
    return out
