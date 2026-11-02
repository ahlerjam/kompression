"""Stufe 0a: ein Arm, viele Seeds parallel.

Beispiele:
  uv run python stages/s0a_grokking/run.py --arm none --seeds 0 1 2 --steps 20000
  uv run python stages/s0a_grokking/run.py --arm wd --weight-decay 1.0

Arme (siehe docs/protokoll_0a.md): none, wd. Weitere (mdl, wd_normproj, bitnet)
kommen in kompression/objectives.py hinzu.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
from mlx.utils import tree_flatten

from kompression.runlog import Run
from kompression.train import TrainConfig, train


@dataclass(frozen=True)
class Args:
    arm: Literal["none", "wd"] = "none"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    steps: int = 20_000
    weight_decay: float = 1.0
    lr: float = 1e-3
    train_frac: float = 0.4
    p: int = 97
    use_stablemax: bool = True
    eval_every: int = 100
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
        tags=(a.arm,),
    )
    run = Run("s0a", a.name or a.arm, {"args": dataclasses.asdict(a), "train": dataclasses.asdict(cfg)})
    print(f"Lauf: {run.dir}")
    res = train(cfg, log=run.log)
    flat = {k: np.array(v) for k, v in tree_flatten(res.pop("params"))}
    np.savez_compressed(run.dir / "params_final.npz", **flat)
    run.finish(res)
    print({k: v for k, v in res.items()})


if __name__ == "__main__":
    main(tyro.cli(Args))
