"""Schlafphase: Hilfsfunktionen aus gelösten Programmen extrahieren (ReGAL-artig).

Annahmeregel (Konzept): neue Bibliothek nur, wenn L_gesamt auf dem Probe-Set sinkt.
"""

from __future__ import annotations

from kompression.s1.library import Library
from kompression.s1.probe import ProbeResult, evaluate
from kompression.s1.sandbox import execute
from kompression.s1.solver import PREFILL, extract_code

SLEEP_SYSTEM = (
    "You refactor solved programs into a small library of reusable helper functions. "
    "Write general functions (with short docstrings) that would make many of these programs "
    "shorter. Do not include task-specific constants. Reply with one ```python block that "
    "contains only function definitions and imports."
)


def sleep_prompt(programs: list[str], lib: Library) -> str:
    shown = "\n\n".join(f"# Program {i + 1}\n{p}" for i, p in enumerate(programs))
    existing = f"Existing library:\n```python\n{lib.source()}\n```\n\n" if lib.functions else ""
    return f"{existing}Solved programs:\n```python\n{shown}\n```"


def propose_functions(lm, programs: list[str], lib: Library, n: int = 2,
                      temperature: float = 0.7, max_tokens: int = 768,
                      backend: str = "container") -> list[dict[str, str]]:
    prefill = lm.tokenizer.encode(PREFILL, add_special_tokens=False)
    prompt = lm.chat(SLEEP_SYSTEM, sleep_prompt(programs, lib)) + prefill
    out = []
    for text in lm.generate([prompt] * n, max_tokens, temperature):
        code = extract_code(PREFILL + text)
        if not code:
            continue
        try:
            fns = Library.parse(code)
        except SyntaxError:
            continue
        fns = {k: v for k, v in fns.items() if k not in lib.functions}
        if not fns:
            continue
        candidate = lib.with_functions(fns)
        imports = "\n".join(ln for ln in code.splitlines() if ln.startswith(("import ", "from ")))
        if imports:
            fns["__imports__"] = imports
            candidate = lib.with_functions(fns)
        if execute("print('ok')", candidate.source(), backend=backend).ok:
            out.append(fns)
    return out


def sleep_cycle(lm, world, probe, lib: Library, programs: list[str], current: ProbeResult,
                n_proposals: int = 2, probe_samples: int = 4, lib_mode: str = "lm",
                backend: str = "container"):
    """Gibt (neue Bibliothek, ProbeResult, Protokoll) zurück; unverändert, wenn nichts L senkt."""
    log = []
    best_lib, best = lib, current
    for fns in propose_functions(lm, programs, lib, n=n_proposals, backend=backend):
        cand = lib.with_functions(fns)
        r = evaluate(lm, world, probe, cand, n=probe_samples, lib_mode=lib_mode, backend=backend)
        log.append({"functions": sorted(k for k in fns if k != "__imports__"),
                    "total_bits": r.total_bits, "library_bits": r.library_bits,
                    "solved": r.solved, "reuse": r.reuse})
        if r.total_bits < best.total_bits:
            best_lib, best = cand, r
    return best_lib, best, log
