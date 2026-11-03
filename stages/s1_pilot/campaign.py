"""Stufe-1-Kampagne: Proposer-Politik × Kontrolle bei gleichem Budget, Transfer auf ungesehene Familien.

  uv run python stages/s1_pilot/campaign.py --policy compression --seed 0
  uv run python stages/s1_pilot/campaign.py --policy learnability --seed 0
  uv run python stages/s1_pilot/campaign.py --policy random --control empty_library --seed 0

Budget je Runde: `families_per_round` × `tasks_per_family` × `samples` Generierungen in der
Wachphase, gleich für alle Bedingungen. Kontrollen:
- `empty_library`: keine Schlafphase (Bibliothek bleibt leer).
- `self_consistency`: keine Schlafphase; dafür doppelt so viele Samples bei der Endmessung
  (verbraucht das gesparte Budget der Schlafphase grob gleich).
Endmessung: L_gesamt und Lösungszahl auf Probe (Trainingsfamilien) und auf zurückgehaltenen Familien.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

from kompression.runlog import Run
from kompression.s1.library import Library
from kompression.s1.lm import FrozenLM
from kompression.s1.probe import evaluate
from kompression.s1.proposer import (
    RandomPolicy,
    SoftmaxBandit,
    compression_credit,
    learnability_reward,
)
from kompression.s1.sleep import sleep_cycle
from kompression.s1.solver import solve
from kompression.s1.worlds import HARD_CONFIGS, World

TRAIN = ("shortest_path", "graph_color", "sudoku", "n_queens", "game_of_life", "knights_knaves")
HELDOUT = ("maze", "largest_island", "course_schedule", "spiral_matrix")


@dataclass(frozen=True)
class Args:
    policy: Literal["compression", "learnability", "random"] = "compression"
    control: Literal["none", "empty_library", "self_consistency"] = "none"
    repo: str = "mlx-community/Qwen3.6-35B-A3B-4bit"
    rounds: int = 6
    families_per_round: int = 2
    tasks_per_family: int = 3
    samples: int = 4
    sleep_every: int = 2
    probe_per_family: int = 2
    probe_samples: int = 2
    max_tokens: int = 1024
    min_uses: int = 2
    amortize: bool = True  # L_gesamt über alle gesehenen Aufgaben hochrechnen (Konzept-Ziel)
    seed: int = 0
    name: str = ""


def main(a: Args) -> None:
    run = Run("s1_campaign", a.name or f"{a.policy}_{a.control}_s{a.seed}", a)
    lm, world = FrozenLM(a.repo), World(HARD_CONFIGS)
    probe = [t for f in TRAIN for t in world.generate(f, seed=5000, n=a.probe_per_family)]
    heldout = [t for f in HELDOUT for t in world.generate(f, seed=6000, n=a.probe_per_family)]
    pol = RandomPolicy(TRAIN, seed=a.seed) if a.policy == "random" else SoftmaxBandit(TRAIN, seed=a.seed)
    lib = Library()
    current = evaluate(lm, world, probe, lib, n=a.probe_samples, max_tokens=a.max_tokens)
    pool: list[tuple] = []
    pool_fams: Counter = Counter()
    gens = 0
    n_seen = 0
    t0 = time.time()
    for r in range(1, a.rounds + 1):
        fams = pol.select(a.families_per_round)
        for f in fams:
            tasks = world.generate(f, seed=10_000 * (a.seed + 1) + 100 * r + TRAIN.index(f),
                                   n=a.tasks_per_family)
            res = solve(lm, world, tasks, lib, n=a.samples, max_tokens=a.max_tokens)
            gens += len(tasks) * a.samples
            n_seen += len(tasks)
            per_task = [np.mean([x.score >= 1 for x in atts]) for atts in res]
            for atts in res:
                ok = [x for x in atts if x.score >= 1 and x.bits is not None]
                if ok:
                    best = min(ok, key=lambda x: x.bits)
                    pool.append((best.task, best.code))
                    pool_fams[f] += 1
            if a.policy == "learnability":
                for pr in per_task:
                    pol.update(f, learnability_reward(float(pr)))
            run.log(round=r, family=f, pass_rate=float(np.mean(per_task)), gens=gens,
                    t=time.time() - t0)
        if a.control == "none" and r % a.sleep_every == 0 and pool:
            ns = n_seen if a.amortize else 0
            before = current.amortized(ns)
            lib, current, log = sleep_cycle(lm, world, probe, lib, pool, current,
                                            min_uses=a.min_uses, max_tokens=a.max_tokens,
                                            probe_samples=a.probe_samples, n_seen=ns)
            evaluated = [p["amortized_bits"] for p in log if "amortized_bits" in p]
            delta = before - min(evaluated) if evaluated else 0.0  # mit Vorzeichen, ohne Clipping
            if a.policy == "compression":
                for f, rew in compression_credit(delta, dict(pool_fams)).items():
                    pol.update(f, rew)
            run.log(round=r, sleep=True, delta_bits=delta, accepted=sorted(
                k for k in lib.functions if k != "__imports__"), total_bits=current.total_bits,
                solved=current.solved, proposals=log, t=time.time() - t0)
            pool, pool_fams = [], Counter()
    n_final = a.probe_samples * (2 if a.control == "self_consistency" else 1)
    fin_probe = evaluate(lm, world, probe, lib, n=n_final, max_tokens=a.max_tokens)
    fin_held = evaluate(lm, world, heldout, lib, n=n_final, max_tokens=a.max_tokens)
    res = {"probe_total_bits": fin_probe.total_bits, "probe_solved": fin_probe.solved,
           "heldout_total_bits": fin_held.total_bits, "heldout_solved": fin_held.solved,
           "heldout_reuse": fin_held.reuse, "library": sorted(lib.functions),
           "library_bits": fin_probe.library_bits, "wake_generations": gens,
           "bandit_values": getattr(pol, "values", None), "seconds": time.time() - t0}
    (run.dir / "library.py").write_text(lib.source())
    run.finish(res)
    print(res, flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
