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
                      temperature: float = 0.7, max_tokens: int = 1536,
                      backend: str = "container", diag: list | None = None) -> list[dict[str, str]]:
    prefill = lm.tokenizer.encode(PREFILL, add_special_tokens=False)
    prompt = lm.chat(SLEEP_SYSTEM, sleep_prompt(programs, lib)) + prefill
    out = []
    for text in lm.generate([prompt] * n, max_tokens, temperature):
        code = extract_code(PREFILL + text)
        if not code:
            _diag(diag, "kein Code")
            continue
        try:
            fns = Library.parse(code)
        except SyntaxError:
            _diag(diag, "Syntaxfehler (abgeschnitten?)" if not text.rstrip().endswith("```") else "Syntaxfehler")
            continue
        fns = {k: v for k, v in fns.items() if k not in lib.functions}
        if not fns:
            _diag(diag, "keine neuen Funktionen")
            continue
        candidate = lib.with_functions(fns)
        imports = "\n".join(ln for ln in code.splitlines() if ln.startswith(("import ", "from ")))
        if imports:
            fns["__imports__"] = imports
            candidate = lib.with_functions(fns)
        if execute("print('ok')", candidate.source(), backend=backend).ok:
            out.append(fns)
        else:
            _diag(diag, "Bibliothek nicht ausführbar")
    return out


def _diag(diag: list | None, reason: str) -> None:
    if diag is not None:
        diag.append(reason)


REWRITE_SYSTEM = (
    "Rewrite the given program so that it uses the library functions wherever they fit. "
    "The rewritten program must print exactly the same output. Library functions are already "
    "defined; do not redefine them. Reply with one ```python block."
)


def filter_by_verified_reuse(lm, world, solved: list[tuple], fns: dict[str, str], lib: Library,
                             min_uses: int = 2, backend: str = "container") -> tuple[dict, dict]:
    """ReGAL-artig: gelöste Programme mit den neuen Funktionen umschreiben, neu verifizieren
    und nur Funktionen behalten, die in >= min_uses verifizierten Umschriften vorkommen.

    solved: Liste (Task, Programm). Gibt (behaltene Funktionen, Nutzungszählung) zurück.
    """
    cand = lib.with_functions(fns)
    prefill = lm.tokenizer.encode(PREFILL, add_special_tokens=False)
    prompts = [lm.chat(REWRITE_SYSTEM,
                       f"Library:\n```python\n{cand.source()}\n```\n\nProgram:\n```python\n{p}\n```")
               + prefill for _, p in solved]
    texts = lm.generate(prompts, 1536, 0.2) if prompts else []
    uses: dict[str, int] = {}
    for (task, _), text in zip(solved, texts, strict=True):
        code = extract_code(PREFILL + text)
        if not code:
            continue
        res = execute(code, cand.source(), backend=backend)
        if res.ok and world.verify(task, res.answer) >= 1.0:
            for name in cand.called_in(code) & set(fns):
                uses[name] = uses.get(name, 0) + 1
    keep = {k: v for k, v in fns.items() if uses.get(k, 0) >= min_uses}
    if keep and "__imports__" in fns:
        keep["__imports__"] = fns["__imports__"]
    return keep, uses


def sleep_cycle(lm, world, probe, lib: Library, programs: list, current: ProbeResult,
                n_proposals: int = 2, probe_samples: int = 4, lib_mode: str = "lm",
                backend: str = "container", min_uses: int = 0, max_tokens: int = 1024,
                n_seen: int = 0):
    """Gibt (neue Bibliothek, ProbeResult, Protokoll) zurück; unverändert, wenn nichts L senkt.

    programs: Liste von Programmen oder von (Task, Programm). Mit min_uses > 0 und Paaren
    werden Vorschläge erst per verifizierter Umschrift gefiltert (ReGAL-artig).
    Annahme: L_gesamt sinkt UND die Zahl gelöster Probe-Aufgaben sinkt nicht.
    Mit n_seen > 0 wird L_gesamt über alle gesehenen Aufgaben hochgerechnet (Konzept-Ziel),
    sodass sich eine Bibliothek über viele Aufgaben amortisieren kann.
    """
    log = []
    best_lib, best = lib, current
    pairs = [p for p in programs if isinstance(p, tuple)]
    codes = [p[1] if isinstance(p, tuple) else p for p in programs]
    diag: list[str] = []
    for fns in propose_functions(lm, codes, lib, n=n_proposals, backend=backend, diag=diag):
        uses = None
        if min_uses and pairs:
            fns, uses = filter_by_verified_reuse(lm, world, pairs, fns, lib, min_uses, backend)
            if not [k for k in fns if k != "__imports__"]:
                log.append({"functions": [], "verified_uses": uses, "rejected": "keine Wiederverwendung"})
                continue
        cand = lib.with_functions(fns)
        r = evaluate(lm, world, probe, cand, n=probe_samples, lib_mode=lib_mode, backend=backend,
                     max_tokens=max_tokens)
        log.append({"functions": sorted(k for k in fns if k != "__imports__"),
                    "total_bits": r.total_bits, "amortized_bits": r.amortized(n_seen),
                    "amortized_before": best.amortized(n_seen), "library_bits": r.library_bits,
                    "solved": r.solved, "reuse": r.reuse, "verified_uses": uses})
        if r.amortized(n_seen) < best.amortized(n_seen) and r.solved >= best.solved:
            best_lib, best = cand, r
    if diag:
        log.append({"verworfene_vorschlaege": diag})
    return best_lib, best, log
