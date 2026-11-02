"""Brücke zwischen trainierten MLX-Gewichten und dem Kern-Rezept für L(W)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from kompression.core.weight_bits import weight_bits
from kompression.model_mlx import forward, label_logprobs
from kompression.train import TrainConfig, unstack_tree


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
            tree = tree_unflatten([(k, mx.array(w.astype(np.float32))) for k, w in zip(keys, ws, strict=True)])
            lp = label_logprobs(forward(tree, x, mcfg), y, cfg.use_stablemax)
            return float(-lp.mean())

        ws = [np.array(v) for _, v in tree_flatten(p)]
        w0s = [np.array(v) for _, v in tree_flatten(w0)]
        r = weight_bits(ws, loss_fn, tolerance, init=w0s)
        out[seed] = {"bits": r["bits"], "description": r["description"],
                     "absolute_bits": r["absolute"]["bits"] if r["absolute"] else None,
                     "diff_bits": r["diff"]["bits"] if r["diff"] else None,
                     "recipe": r["recipe"], "tolerance": tolerance}
    return out
