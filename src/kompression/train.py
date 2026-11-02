"""Training vieler Seeds gleichzeitig per `mx.vmap` (Entscheidung: Bauplan Schritt 3).

Jeder Seed hat eigene Initialisierung und eigenen Daten-Split. Für einen Seed
sind Split und Init in allen Armen identisch (gepaartes Design).
Arme werden über `reg` gewählt; neue Arme kommen in `objectives.py`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from kompression.data import ModTask, split
from kompression.model_mlx import ModelConfig, forward, init_params, label_logprobs


@dataclass(frozen=True)
class TrainConfig:
    p: int = 97
    op: str = "add"
    train_frac: float = 0.4
    seeds: tuple[int, ...] = (0,)
    steps: int = 10_000
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    weight_decay: float = 0.0
    use_stablemax: bool = True
    eval_every: int = 100
    grok_threshold: float = 0.99
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 512
    tags: tuple[str, ...] = field(default_factory=tuple)

    def task(self) -> ModTask:
        return ModTask(p=self.p, op=self.op)

    def model(self) -> ModelConfig:
        t = self.task()
        return ModelConfig(
            vocab_size=t.vocab_size,
            n_out=t.p,
            seq_len=t.seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_mlp=self.d_mlp,
        )


def stack_trees(trees: list):
    return tree_map(lambda *xs: mx.stack(xs), *trees)


def unstack_tree(tree, i: int):
    return tree_map(lambda a: a[i], tree)


def sq_norm(params) -> mx.array:
    return sum((v * v).sum() for _, v in tree_flatten(params))


def build_fns(cfg: TrainConfig):
    mcfg = cfg.model()

    def loss_fn(params, x, y):
        return -label_logprobs(forward(params, x, mcfg), y, cfg.use_stablemax).mean()

    def step(params, m, v, t, x, y):
        loss, grads = mx.value_and_grad(loss_fn)(params, x, y)
        b1, b2 = cfg.beta1, cfg.beta2
        m = tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
        v = tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
        c1, c2 = 1 - b1**t, 1 - b2**t

        def upd(p, m_, v_):
            p = p - cfg.lr * cfg.weight_decay * p  # entkoppelter Weight Decay (AdamW)
            return p - cfg.lr * (m_ / c1) / (mx.sqrt(v_ / c2) + cfg.eps)

        params = tree_map(upd, params, m, v)
        return params, m, v, loss

    def evaluate(params, x, y):
        logits = forward(params, x, mcfg)
        loss = -label_logprobs(logits, y, cfg.use_stablemax).mean()
        acc = (mx.argmax(logits, axis=-1) == y).astype(mx.float32).mean()
        return loss, acc

    vstep = mx.compile(mx.vmap(step, in_axes=(0, 0, 0, None, 0, 0)))
    veval = mx.compile(mx.vmap(evaluate, in_axes=(0, 0, 0)))
    vnorm = mx.compile(mx.vmap(lambda p: mx.sqrt(sq_norm(p))))
    return vstep, veval, vnorm, loss_fn


def prepare(cfg: TrainConfig):
    task = cfg.task()
    splits = [split(task, cfg.train_frac, s) for s in cfg.seeds]
    data = {k: mx.array(np.stack([sp[k] for sp in splits])) for k in splits[0]}
    params = stack_trees([init_params(cfg.model(), s) for s in cfg.seeds])
    return data, params


def train(cfg: TrainConfig, log=None) -> dict:
    """Trainiert alle Seeds parallel. `log(**row)` bekommt einen Messpunkt pro Seed."""
    data, params = prepare(cfg)
    m = tree_map(mx.zeros_like, params)
    v = tree_map(mx.zeros_like, params)
    vstep, veval, vnorm, _ = build_fns(cfg)
    S = len(cfg.seeds)
    grok_step = [None] * S
    t0 = time.time()
    for step_i in range(1, cfg.steps + 1):
        params, m, v, loss = vstep(params, m, v, mx.array(float(step_i)), data["x_train"], data["y_train"])
        if step_i % cfg.eval_every == 0 or step_i == cfg.steps:
            tr_loss, tr_acc = veval(params, data["x_train"], data["y_train"])
            te_loss, te_acc = veval(params, data["x_test"], data["y_test"])
            norm = vnorm(params)
            mx.eval(tr_loss, tr_acc, te_loss, te_acc, norm)
            for i, seed in enumerate(cfg.seeds):
                row = {
                    "step": step_i,
                    "seed": seed,
                    "train_loss": tr_loss[i].item(),
                    "train_acc": tr_acc[i].item(),
                    "test_loss": te_loss[i].item(),
                    "test_acc": te_acc[i].item(),
                    "weight_norm": norm[i].item(),
                    "elapsed_s": time.time() - t0,
                }
                if grok_step[i] is None and row["test_acc"] >= cfg.grok_threshold:
                    grok_step[i] = step_i
                if log:
                    log(**row)
            if any(math.isnan(x) for x in np.array(te_loss).tolist()):
                raise FloatingPointError(f"NaN im Test-Loss bei Schritt {step_i}")
    mx.eval(params)
    return {
        "params": params,
        "grok_step": dict(zip(cfg.seeds, grok_step)),
        "final_test_acc": dict(zip(cfg.seeds, np.array(te_acc).tolist())),
        "final_train_acc": dict(zip(cfg.seeds, np.array(tr_acc).tolist())),
        "final_weight_norm": dict(zip(cfg.seeds, np.array(norm).tolist())),
        "seconds": time.time() - t0,
    }
