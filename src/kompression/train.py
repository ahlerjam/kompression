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
from kompression.objectives import Arm, make_arm


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
    arm: str = "none"
    mdl_beta: float = 1.0
    beta_warmup: int = 0
    mdl_k: int = 16
    subspace_rank: int = 16
    norm_target_run: str = ""  # Run-Ordner, dessen Normtrajektorie erzwungen wird (E10)
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


_NO_NOISE = mx.zeros((1,))  # Platzhalter: vmap akzeptiert kein None


def stack_trees(trees: list):
    return tree_map(lambda *xs: mx.stack(xs), *trees)


def unstack_tree(tree, i: int):
    return tree_map(lambda a: a[i], tree)


def sq_norm(params) -> mx.array:
    return sum((v * v).sum() for _, v in tree_flatten(params))


def build_fns(cfg: TrainConfig, arm: Arm, n_train: int):
    mcfg = cfg.model()

    def loss_fn(tr, const, noise, beta, x, y):
        p = arm.effective(tr, const, noise)
        nll = -label_logprobs(forward(p, x, mcfg), y, cfg.use_stablemax).mean()
        pen = arm.penalty(tr, const, noise)
        return nll + beta * pen / n_train, nll, pen

    def step(tr, const, m, v, t, beta, noise, x, y):
        (_, nll, pen), grads = mx.value_and_grad(loss_fn)(tr, const, noise, beta, x, y)
        b1, b2 = cfg.beta1, cfg.beta2
        m = tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
        v = tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
        c1, c2 = 1 - b1**t, 1 - b2**t

        def upd(p, m_, v_):
            p = p - cfg.lr * cfg.weight_decay * p  # entkoppelter Weight Decay (AdamW)
            return p - cfg.lr * (m_ / c1) / (mx.sqrt(v_ / c2) + cfg.eps)

        tr = tree_map(upd, tr, m, v)
        return tr, m, v, nll, pen

    def evaluate(tr, const, x, y):
        p = arm.deterministic(tr, const)
        logits = forward(p, x, mcfg)
        loss = -label_logprobs(logits, y, cfg.use_stablemax).mean()
        acc = (mx.argmax(logits, axis=-1) == y).astype(mx.float32).mean()
        dnorm = mx.sqrt(sum(((a - b) ** 2).sum() for (_, a), (_, b) in
                            zip(tree_flatten(p), tree_flatten(const["w0"]), strict=True)))
        return loss, acc, mx.sqrt(sq_norm(p)), dnorm

    noise_axis = 0 if arm.needs_noise else None
    vstep = mx.compile(mx.vmap(step, in_axes=(0, 0, 0, 0, None, None, noise_axis, 0, 0)))
    veval = mx.compile(mx.vmap(evaluate, in_axes=(0, 0, 0, 0)))
    vdet = mx.vmap(arm.deterministic, in_axes=(0, 0))
    return vstep, veval, vdet


def load_norm_targets(run_dir: str, seeds: tuple[int, ...], steps: int) -> np.ndarray:
    """Normtrajektorie [S, steps] aus metrics.jsonl eines anderen Laufs (linear interpoliert)."""
    import json
    from pathlib import Path

    rows = [json.loads(x) for x in (Path(run_dir) / "metrics.jsonl").read_text().splitlines()]
    out = np.zeros((len(seeds), steps), dtype=np.float32)
    grid = np.arange(1, steps + 1)
    for i, s in enumerate(seeds):
        pts = sorted((r["step"], r["weight_norm"]) for r in rows if r["seed"] == s)
        if not pts:
            raise ValueError(f"Seed {s} fehlt im Ziel-Lauf {run_dir}")
        xs, ys = zip(*pts, strict=True)
        out[i] = np.interp(grid, xs, ys)
    return out


def prepare(cfg: TrainConfig, arm: Arm):
    task = cfg.task()
    splits = [split(task, cfg.train_frac, s) for s in cfg.seeds]
    data = {k: mx.array(np.stack([sp[k] for sp in splits])) for k in splits[0]}
    pairs = [arm.init(init_params(cfg.model(), s), s) for s in cfg.seeds]
    tr = stack_trees([p[0] for p in pairs])
    const = stack_trees([p[1] for p in pairs])
    return data, tr, const


def train(cfg: TrainConfig, log=None, arm: Arm | None = None) -> dict:
    """Trainiert alle Seeds parallel. `log(**row)` bekommt einen Messpunkt pro Seed."""
    arm = arm or make_arm(cfg.arm, subspace_rank=cfg.subspace_rank, mdl_k=cfg.mdl_k)
    data, tr, const = prepare(cfg, arm)
    n_train = int(data["y_train"].shape[1])
    m = tree_map(mx.zeros_like, tr)
    v = tree_map(mx.zeros_like, tr)
    vstep, veval, vdet = build_fns(cfg, arm, n_train)
    mx.random.seed(1_000_003 * (1 + cfg.seeds[0]) + len(cfg.seeds))
    S = len(cfg.seeds)
    targets = None
    if cfg.norm_target_run:
        if cfg.arm not in ("none", "wd"):
            raise ValueError("Norm-Projektion nur für Arme none/wd")
        targets = mx.array(load_norm_targets(cfg.norm_target_run, cfg.seeds, cfg.steps))
        vnorm = mx.compile(mx.vmap(lambda p: mx.sqrt(sq_norm(p))))
        vscale = mx.vmap(lambda p, c: tree_map(lambda a: a * c, p), in_axes=(0, 0))
    grok_step = [None] * S
    t0 = time.time()
    for step_i in range(1, cfg.steps + 1):
        beta = min(1.0, step_i / cfg.beta_warmup) * cfg.mdl_beta if cfg.beta_warmup else cfg.mdl_beta
        noise = arm.noise_like(tr) if arm.needs_noise else _NO_NOISE
        tr, m, v, nll, pen = vstep(
            tr, const, m, v, mx.array(float(step_i)), mx.array(beta), noise,
            data["x_train"], data["y_train"],
        )
        if targets is not None:
            tr = vscale(tr, targets[:, step_i - 1] / (vnorm(tr) + 1e-12))
        if step_i % cfg.eval_every == 0 or step_i == cfg.steps:
            tr_loss, tr_acc, norm, dnorm = veval(tr, const, data["x_train"], data["y_train"])
            te_loss, te_acc, _, _ = veval(tr, const, data["x_test"], data["y_test"])
            mx.eval(tr_loss, tr_acc, te_loss, te_acc, norm, dnorm, nll, pen)
            for i, seed in enumerate(cfg.seeds):
                row = {
                    "step": step_i,
                    "seed": seed,
                    "train_loss": tr_loss[i].item(),
                    "train_acc": tr_acc[i].item(),
                    "test_loss": te_loss[i].item(),
                    "test_acc": te_acc[i].item(),
                    "weight_norm": norm[i].item(),
                    "delta_norm": dnorm[i].item(),
                    "penalty_bits": pen[i].item() / math.log(2),
                    "beta": beta,
                    "elapsed_s": time.time() - t0,
                }
                if grok_step[i] is None and row["test_acc"] >= cfg.grok_threshold:
                    grok_step[i] = step_i
                if log:
                    log(**row)
            if any(math.isnan(x) for x in np.array(te_loss).tolist()):
                raise FloatingPointError(f"NaN im Test-Loss bei Schritt {step_i}")
    params = vdet(tr, const)
    mx.eval(params)
    return {
        "params": params,
        "w0": const["w0"],
        "grok_step": dict(zip(cfg.seeds, grok_step)),
        "final_test_acc": dict(zip(cfg.seeds, np.array(te_acc).tolist())),
        "final_train_acc": dict(zip(cfg.seeds, np.array(tr_acc).tolist())),
        "final_weight_norm": dict(zip(cfg.seeds, np.array(norm).tolist())),
        "final_delta_norm": dict(zip(cfg.seeds, np.array(dnorm).tolist())),
        "final_penalty_bits": dict(zip(cfg.seeds, (np.array(pen) / math.log(2)).tolist())),
        "seconds": time.time() - t0,
    }
