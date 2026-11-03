"""Prequentielle Codelänge der ganzen Additionstabelle für einen Arm (primärer MDL-Endpunkt).

Pro Seed: feste Zufallsreihenfolge aller p*p Paare, Blöcke 64, 128, ... bis p*p.
Block 1 uniform codiert; für jeden weiteren Block wird auf dem Präfix trainiert
(Warmstart aus dem vorigen Block, `steps_per_block` Schritte, Full-Batch) und dann
der Block mit dem Modell codiert, BEVOR es ihn sieht. Self-Switch (v2): pro Block wählt
der Codierer einen Trainingsstand (nach 0, 30, 100, 300, 1.000, 3.000 Schritten auf dem
Präfix) und eine Temperatur T (oder uniform) und zahlt dafür log2(Anzahl Optionen) Bit
(zweiteiliger Code, ehrlich mitgezählt; „Early Stopping als Codewahl“, Blier & Ollivier).
Weitertrainiert wird immer vom Endstand. Gezählt wird im Kern
(`core.codelength.prequential_codelength`).

  uv run python stages/s0a_grokking/prequential.py --arm wd --seeds 100 101 102

Stufe 0b (Transfer): `--op sub --init-from runs/s0a/<A-Lauf>` startet von den Gewichten eines
auf Aufgabe A trainierten Laufs (gleiche Seeds). Kontrollen: `--init-control scaled`
(Zufallsinit, je Tensor auf die Norm der A-Gewichte skaliert) und `permuted` (A-Gewichte je
Tensor zufällig permutiert: gleiche Werteverteilung, Struktur zerstört). Nur für Arme none/wd.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np
import tyro
from mlx.utils import tree_map

from kompression.core.codelength import doubling_blocks, prequential_codelength
from kompression.model_mlx import forward, init_params, label_logprobs
from kompression.objectives import make_arm
from kompression.runlog import Run
from kompression.train import TrainConfig, build_fns, stack_trees


@dataclass(frozen=True)
class Args:
    arm: Literal["none", "wd", "mdl", "bitnet", "subspace"] = "wd"
    seeds: tuple[int, ...] = (100, 101, 102)
    p: int = 97
    first_block: int = 64
    steps_per_block: int = 3000
    weight_decay: float = 1.0
    lr: float = 1e-3
    mdl_beta: float = 0.1
    penalty_n_min: int = 0  # mdl: Strafterm durch max(Präfixgröße, penalty_n_min) teilen
    beta_warmup: int = 2000  # mdl: beta linear über die ersten Schritte hochfahren (wie im Training)
    subspace_rank: int = 64
    op: Literal["add", "sub", "mul"] = "add"
    init_from: str = ""
    init_control: Literal["none", "scaled", "permuted"] = "none"
    name: str = ""


def load_inits(run_dir: str, seeds, random_inits, control: str):
    """A-Gewichte je Seed laden (gestapelt in params*.npz); optional Kontrolle anwenden."""
    import json
    from pathlib import Path

    from mlx.utils import tree_flatten, tree_unflatten

    d = Path(run_dir)
    run_seeds = json.loads((d / "config.json").read_text())
    run_seeds = (run_seeds.get("train") or run_seeds.get("args"))["seeds"]
    npz = d / ("params.npz" if (d / "params.npz").exists() else "params_final.npz")
    flat = dict(np.load(npz))
    out = []
    for s, rnd in zip(seeds, random_inits, strict=True):
        i = run_seeds.index(s)
        a_w = {k: v[i] for k, v in flat.items()}
        res = []
        rng = np.random.default_rng(10_000 + s)
        for k, r in tree_flatten(rnd):
            w = a_w[k]
            if control == "scaled":
                r = np.array(r)
                w = r * (np.linalg.norm(w) / (np.linalg.norm(r) + 1e-12))
            elif control == "permuted":
                w = rng.permutation(w.ravel()).reshape(w.shape)
            res.append((k, mx.array(np.asarray(w, dtype=np.float32))))
        out.append(tree_unflatten(res))
    return out


def main(a: Args) -> None:
    cfg = TrainConfig(p=a.p, op=a.op, seeds=a.seeds, steps=a.steps_per_block, lr=a.lr,
                      weight_decay=a.weight_decay if a.arm == "wd" else 0.0, arm=a.arm,
                      mdl_beta=a.mdl_beta, subspace_rank=a.subspace_rank, tags=(a.arm, "preq"))
    run = Run("s0a_preq", a.name or a.arm, {"args": dataclasses.asdict(a)})
    print(f"Lauf: {run.dir}", flush=True)
    task, mcfg = cfg.task(), cfg.model()
    arm = make_arm(a.arm, subspace_rank=a.subspace_rank)
    x_all, y_all = task.all_pairs()
    n = len(y_all)
    perms = [np.random.default_rng(s).permutation(n) for s in a.seeds]
    X = mx.array(np.stack([x_all[pm] for pm in perms]))
    Y = mx.array(np.stack([y_all[pm] for pm in perms]))

    inits = [init_params(mcfg, s) for s in a.seeds]
    if a.init_from:
        if a.arm not in ("none", "wd"):
            raise ValueError("Transfer-Init nur für Arme none/wd")
        inits = load_inits(a.init_from, a.seeds, inits, a.init_control)
    pairs = [arm.init(p0, s) for p0, s in zip(inits, a.seeds, strict=True)]
    tr = stack_trees([q[0] for q in pairs])
    const = stack_trees([q[1] for q in pairs])
    m = tree_map(mx.zeros_like, tr)
    v = tree_map(mx.zeros_like, tr)
    t = 0

    temps = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 16.0, 32.0)
    checkpoints = tuple(c for c in (0, 30, 100, 300, 1000, 3000, 10000) if c <= a.steps_per_block)
    if a.steps_per_block not in checkpoints:
        checkpoints = (*checkpoints, a.steps_per_block)
    switch_bits = math.log2(len(checkpoints) * len(temps) + 1)  # + uniform

    def block_logp(tr_, const_, x, y):
        p_ = arm.deterministic(tr_, const_)
        logits = forward(p_, x, mcfg)
        return mx.stack([label_logprobs(logits / tt, y, cfg.use_stablemax) for tt in temps])

    vlogp = mx.vmap(block_logp, in_axes=(0, 0, 0, 0))
    bounds = doubling_blocks(n, a.first_block)
    per_seed_logp: dict[tuple[int, int], np.ndarray] = {}
    t0 = time.time()
    for start, end in zip(bounds, bounds[1:], strict=False):
        vstep, _, _ = build_fns(cfg, arm, n_train=max(start, a.penalty_n_min))
        noise_dummy = mx.zeros((1,))
        cands = []  # je Checkpoint: [S, len(temps), block]
        for step_in_block in range(a.steps_per_block + 1):
            if step_in_block in checkpoints:
                lp_c = vlogp(tr, const, X[:, start:end], Y[:, start:end])
                mx.eval(lp_c)
                cands.append(np.array(lp_c, dtype=np.float64))
            if step_in_block == a.steps_per_block:
                break
            t += 1
            noise = arm.noise_like(tr) if arm.needs_noise else noise_dummy
            beta_t = a.mdl_beta * min(1.0, t / a.beta_warmup) if a.beta_warmup else a.mdl_beta
            tr, m, v, _, _ = vstep(tr, const, m, v, mx.array(float(t)), mx.array(beta_t),
                                   noise, X[:, :start], Y[:, :start])
            if t % 50 == 0:  # Graph regelmäßig auswerten (sonst Metal-Pufferlimit, v. a. bei mdl)
                mx.eval(tr, m, v)
        lp = np.concatenate(cands, axis=1)  # [S, len(checkpoints)*len(temps), block]
        uni = np.full(lp.shape[-1], -math.log(task.p))
        chosen, bits = [], []
        for i in range(len(a.seeds)):
            options = [*lp[i], uni]
            costs = [-o.sum() for o in options]
            k = int(np.argmin(costs))
            # Wahl der Temperatur wird als Teil des Codes bezahlt (switch_bits)
            per_seed_logp[(i, start)] = options[k] - switch_bits * math.log(2) / len(uni)
            chosen.append("uniform" if k == lp.shape[1] else
                          f"c{checkpoints[k // len(temps)]}/T{temps[k % len(temps)]:g}")
            bits.append(costs[k] / math.log(2) + switch_bits)
        bits = np.array(bits)
        run.log(block_start=start, block_end=end, bits_per_seed=bits.tolist(),
                bits_per_label=(bits / (end - start)).tolist(), temperature=chosen,
                elapsed_s=time.time() - t0)
        print(f"Block [{start},{end}) Bits/Label je Seed: {np.round(bits / (end - start), 3)} T={chosen}",
              flush=True)

    results = {}
    for i, s in enumerate(a.seeds):
        r = prequential_codelength(n, task.p, bounds, lambda st, en, i=i: per_seed_logp[(i, st)])
        results[s] = {"total_bits": r["total_bits"], "uniform_bits": r["uniform_bits"],
                      "per_block_bits": r["per_block_bits"]}
    run.finish({"prequential": results, "bounds": bounds, "seconds": time.time() - t0})
    print({s: round(v["total_bits"]) for s, v in results.items()}, "uniform",
          round(results[a.seeds[0]]["uniform_bits"]), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
