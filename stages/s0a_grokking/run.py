"""Stufe 0a: ein Arm, viele Seeds parallel.

Beispiele:
  uv run python stages/s0a_grokking/run.py --arm none --seeds 0 1 2 --steps 20000
  uv run python stages/s0a_grokking/run.py --arm wd --weight-decay 1.0
  uv run python stages/s0a_grokking/run.py --arm mdl --mdl-beta 0.01 --beta-warmup 2000
  uv run python stages/s0a_grokking/run.py --arm subspace --subspace-rank 16
  uv run python stages/s0a_grokking/run.py --arm bitnet

Arme: siehe docs/protokoll_0a.md und src/kompression/objectives.py.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
from mlx.utils import tree_flatten

from kompression.data import split
from kompression.measure import weight_bits_per_seed
from kompression.runlog import Run
from kompression.train import TrainConfig, train


@dataclass(frozen=True)
class Args:
    arm: Literal["none", "wd", "mdl", "bitnet", "subspace"] = "none"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    steps: int = 20_000
    weight_decay: float = 1.0
    lr: float = 1e-3
    train_frac: float = 0.4
    p: int = 97
    use_stablemax: bool = True
    eval_every: int = 100
    mdl_beta: float = 1.0
    beta_warmup: int = 0
    mdl_k: int = 16
    subspace_rank: int = 16
    lw_tolerance: float = 0.05
    norm_target_run: str = ""
    name: str = ""


def main(a: Args) -> None:
    cfg = TrainConfig(
        p=a.p,
        train_frac=a.train_frac,
        seeds=a.seeds,
        steps=a.steps,
        lr=a.lr,
        weight_decay=a.weight_decay if a.arm == "wd" else 0.0,
        use_stablemax=a.use_stablemax,
        eval_every=a.eval_every,
        arm=a.arm,
        mdl_beta=a.mdl_beta,
        beta_warmup=a.beta_warmup,
        mdl_k=a.mdl_k,
        subspace_rank=a.subspace_rank,
        norm_target_run=a.norm_target_run,
        tags=(a.arm,),
    )
    run = Run("s0a", a.name or a.arm, {"args": dataclasses.asdict(a), "train": dataclasses.asdict(cfg)})
    print(f"Lauf: {run.dir}", flush=True)
    res = train(cfg, log=run.log)
    splits = [split(cfg.task(), cfg.train_frac, s) for s in cfg.seeds]
    data = {k: np.stack([sp[k] for sp in splits]) for k in splits[0]}
    res["weight_bits"] = weight_bits_per_seed(cfg, res, data, a.lw_tolerance)
    for key in ("params", "w0"):
        flat = {k: np.array(v) for k, v in tree_flatten(res.pop(key))}
        np.savez_compressed(run.dir / f"{key}.npz", **flat)
    run.finish(res)
    print(res, flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
