"""Pilot-Bench: Pass-Rate, Bits und Durchsatz des eingefrorenen Modells ohne Bibliothek.

  uv run python stages/s1_pilot/bench.py --repo mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tyro

from kompression.runlog import Run
from kompression.s1.library import Library
from kompression.s1.lm import FrozenLM
from kompression.s1.solver import solve
from kompression.s1.worlds import World


@dataclass(frozen=True)
class Args:
    repo: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    families: tuple[str, ...] = ("gcd", "base_conversion", "word_sorting", "count_bits",
                                 "letter_counting", "prime_factorization")
    tasks_per_family: int = 4
    samples: int = 4
    max_tokens: int = 384
    seed: int = 7
    name: str = "bench"


def main(a: Args) -> None:
    run = Run("s1", a.name, a)
    lm, world = FrozenLM(a.repo), World()
    rows = []
    t0 = time.time()
    for fam in a.families:
        tasks = world.generate(fam, seed=a.seed, n=a.tasks_per_family)
        t1 = time.time()
        res = solve(lm, world, tasks, Library(), n=a.samples, max_tokens=a.max_tokens)
        dt = time.time() - t1
        scores = [at.score for atts in res for at in atts]
        bits = [at.bits for atts in res for at in atts if at.bits is not None]
        row = {"family": fam, "pass_rate": float(np.mean(scores)),
               "solved_tasks": sum(any(at.score >= 1 for at in atts) for atts in res),
               "mean_bits_verified": float(np.mean(bits)) if bits else None, "seconds": dt}
        run.log(**row)
        rows.append(row)
        print(row, flush=True)
    run.finish({"rows": rows, "total_seconds": time.time() - t0, "repo": a.repo})


if __name__ == "__main__":
    main(tyro.cli(Args))
