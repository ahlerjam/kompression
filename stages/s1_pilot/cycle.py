"""Ein oder mehrere Wach-Schlaf-Zyklen ohne Proposer (feste Aufgabenwahl), zum Prüfen der Mechanik.

  uv run python stages/s1_pilot/cycle.py --cycles 2
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import tyro

from kompression.runlog import Run
from kompression.s1.library import Library
from kompression.s1.lm import FrozenLM
from kompression.s1.probe import evaluate, library_bits
from kompression.s1.sleep import sleep_cycle
from kompression.s1.solver import solve
from kompression.s1.worlds import HARD_CONFIGS, World

FAMS = ("gcd", "lcm", "prime_factorization", "count_primes", "number_sorting",
        "letter_counting", "word_sorting", "power_function")


@dataclass(frozen=True)
class Args:
    repo: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    families: tuple[str, ...] = FAMS
    probe_per_family: int = 2
    wake_per_family: int = 4
    wake_samples: int = 4
    probe_samples: int = 4
    proposals: int = 3
    cycles: int = 1
    lib_mode: str = "lm"
    hard: bool = False
    min_uses: int = 2
    max_tokens: int = 1024
    name: str = "cycle"


def main(a: Args) -> None:
    run = Run("s1", a.name, a)
    lm, world = FrozenLM(a.repo), World(HARD_CONFIGS if a.hard else None)
    probe = [t for f in a.families for t in world.generate(f, seed=1000, n=a.probe_per_family)]
    lib = Library()
    t0 = time.time()
    current = evaluate(lm, world, probe, lib, n=a.probe_samples, lib_mode=a.lib_mode,
                       max_tokens=a.max_tokens)
    run.log(cycle=0, total_bits=current.total_bits, library_bits=current.library_bits,
            solved=current.solved, n_probe=len(probe), functions=[], t=time.time() - t0)
    print(f"Start: L_gesamt={current.total_bits:.0f} Bit, gelöst {current.solved}/{len(probe)}",
          flush=True)
    for c in range(1, a.cycles + 1):
        wake = [t for f in a.families
                for t in world.generate(f, seed=2000 + c, n=a.wake_per_family)]
        res = solve(lm, world, wake, lib, n=a.wake_samples, max_tokens=a.max_tokens)
        programs = []
        for atts in res:
            ok = [x for x in atts if x.score >= 1.0 and x.bits is not None]
            if ok:
                best_att = min(ok, key=lambda x: x.bits)
                programs.append((best_att.task, best_att.code))
        lib, current, log = sleep_cycle(lm, world, probe, lib, programs, current,
                                        n_proposals=a.proposals, probe_samples=a.probe_samples,
                                        lib_mode=a.lib_mode, min_uses=a.min_uses,
                                        max_tokens=a.max_tokens)
        row = dict(cycle=c, wake_solved=len(programs), n_wake=len(wake),
                   total_bits=current.total_bits, library_bits=current.library_bits,
                   library_bits_token=library_bits(lm, lib, "token"), solved=current.solved,
                   functions=sorted(k for k in lib.functions if k != "__imports__"),
                   reuse=current.reuse, proposals=log, t=time.time() - t0)
        run.log(**row)
        print({k: v for k, v in row.items() if k != "proposals"}, flush=True)
        for p in log:
            print("  Vorschlag:", p, flush=True)
    (run.dir / "library.py").write_text(lib.source())
    run.finish({"final_total_bits": current.total_bits, "functions": sorted(lib.functions)})


if __name__ == "__main__":
    main(tyro.cli(Args))
