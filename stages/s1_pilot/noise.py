"""Wie stark schwankt L_gesamt auf dem Probe-Set bei wiederholter Messung (leere Bibliothek)?

  uv run python stages/s1_pilot/noise.py --repo mlx-community/Qwen3.5-4B-4bit --reps 3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tyro

from kompression.runlog import Run
from kompression.s1.library import Library
from kompression.s1.lm import FrozenLM
from kompression.s1.probe import evaluate
from kompression.s1.worlds import World

FAMS = ("gcd", "lcm", "prime_factorization", "count_primes", "number_sorting",
        "letter_counting", "word_sorting", "power_function")


@dataclass(frozen=True)
class Args:
    repo: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    reps: int = 3
    probe_samples: int = 4
    probe_per_family: int = 2
    name: str = "noise"


def main(a: Args) -> None:
    run = Run("s1", a.name, a)
    lm, world = FrozenLM(a.repo), World()
    probe = [t for f in FAMS for t in world.generate(f, seed=1000, n=a.probe_per_family)]
    totals, solved = [], []
    for i in range(a.reps):
        r = evaluate(lm, world, probe, Library(), n=a.probe_samples)
        totals.append(r.total_bits)
        solved.append(r.solved)
        run.log(rep=i, total_bits=r.total_bits, solved=r.solved, task_bits=r.task_bits)
        print(f"rep {i}: L={r.total_bits:.0f} Bit, gelöst {r.solved}/{len(probe)}", flush=True)
    res = {"repo": a.repo, "mean": float(np.mean(totals)), "sd": float(np.std(totals, ddof=1)),
           "solved": solved}
    run.finish(res)
    print(res, flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
