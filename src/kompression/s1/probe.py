"""L_gesamt auf einem versiegelten Probe-Set (Stufe-1-Ziel).

L_gesamt(B) = L(B) + Σ_probe L(x_i),  L(x_i) = min( Bits der kürzesten verifizierten Lösung
unter dem Modell mit B im Prompt ,  Rückfall: Antwort direkt übertragen ).
Rückfall-Code (v2): −log2 p_Modell(Antwort | Frage) — gültig ohne Lösung, glatter als
8 Bit/Zeichen (v1 erzeugte Schwankungen von ±600 Bit durch einzelne ungelöste Aufgaben).
L(B): `token` = Tokens × log2(V) (uniform) oder `lm` = −log2 p_Modell(B | Bibliotheks-Prompt).
"""

from __future__ import annotations

from dataclasses import dataclass

from kompression.s1.library import Library, token_bits
from kompression.s1.solver import solve
from kompression.s1.worlds import Task, World

LIB_SYSTEM = "You write small, reusable, well-named Python helper functions with docstrings."
LIB_USER = "Write a Python library of helper functions."


ANSWER_SYSTEM = "Answer with only the final answer, nothing else."


def fallback_bits(answer: str, lm=None, question: str | None = None) -> float:
    if lm is None or question is None:
        return 8.0 * (len(answer) + 1)
    return lm.completion_bits(lm.chat(ANSWER_SYSTEM, question), answer)


def library_bits(lm, lib: Library, mode: str = "lm") -> float:
    src = lib.source()
    if not src:
        return 0.0
    if mode == "token":
        return token_bits(src, lm.tokenizer, lm.vocab_size)
    return lm.completion_bits(lm.chat(LIB_SYSTEM, LIB_USER), src)


@dataclass
class ProbeResult:
    total_bits: float
    library_bits: float
    task_bits: list[float]
    solved: int
    reuse: dict[str, int]


def evaluate(lm, world: World, probe: list[Task], lib: Library, n: int = 4,
             lib_mode: str = "lm", backend: str = "container") -> ProbeResult:
    res = solve(lm, world, probe, lib, n=n, backend=backend)
    task_bits, solved, reuse = [], 0, {}
    for t, atts in zip(probe, res, strict=True):
        ok = [a for a in atts if a.score >= 1.0 and a.bits is not None]
        if ok:
            best = min(ok, key=lambda a: a.bits)
            solved += 1
            task_bits.append(min(best.bits, fallback_bits(t.answer, lm, t.question)))
            for f in best.lib_calls:
                reuse[f] = reuse.get(f, 0) + 1
        else:
            task_bits.append(fallback_bits(t.answer, lm, t.question))
    lb = library_bits(lm, lib, lib_mode)
    return ProbeResult(lb + sum(task_bits), lb, task_bits, solved, reuse)
