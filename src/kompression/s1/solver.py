"""Solver: Modell schreibt ein Programm, das die Antwort druckt; Bibliothek im Prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kompression.s1.library import Library
from kompression.s1.sandbox import execute
from kompression.s1.worlds import Task, World

SYSTEM = (
    "You solve tasks by writing a short Python program that prints ONLY the final answer, "
    "exactly in the requested format (it may span several lines), and nothing else. Library functions shown to you are already defined; call them, "
    "do not redefine them. Reply with a single ```python code block and nothing else."
)


LIB_VIEW = "signatures"  # "signatures" (kompakt) oder "source" (voller Code im Prompt)


def user_prompt(task: Task, lib: Library, view: str | None = None) -> str:
    view = view or LIB_VIEW
    if not [k for k in lib.functions if k != "__imports__"]:
        return f"Task:\n{task.question}"
    shown = lib.signatures() if view == "signatures" else lib.source()
    return f"Library (already defined):\n```python\n{shown}\n```\n\nTask:\n{task.question}"


PREFILL = "```python\n"  # erzwingt Code; Bits zählen nur ab hier (Prefill ist für alle gleich)

_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    m = _CODE.findall(text)
    if m:
        return m[0].strip()
    if text.startswith(PREFILL):  # abgeschnitten ohne schließende Backticks
        return text[len(PREFILL):].strip() or None
    return None


@dataclass
class Attempt:
    task: Task
    code: str | None
    answer: str | None
    score: float
    bits: float | None = None  # L(code | task, lib) unter dem Modell
    lib_calls: frozenset[str] = frozenset()


def solve(lm, world: World, tasks: list[Task], lib: Library, n: int = 4,
          max_tokens: int = 512, temperature: float = 0.7, backend: str = "container",
          score_bits: bool = True) -> list[list[Attempt]]:
    """n Versuche je Aufgabe; Bits nur für verifizierte Lösungen."""
    prefill = lm.tokenizer.encode(PREFILL, add_special_tokens=False) if hasattr(lm, "tokenizer") else []
    prompts = [lm.chat(SYSTEM, user_prompt(t, lib)) + prefill for t in tasks]
    texts = [PREFILL + x for x in
             lm.generate([p for p in prompts for _ in range(n)], max_tokens, temperature)]
    out = []
    for i, t in enumerate(tasks):
        attempts = []
        for text in texts[i * n : (i + 1) * n]:
            code = extract_code(text)
            res = execute(code, lib.source(), backend=backend) if code else None
            ans = res.answer if res and res.ok else None
            score = world.verify(t, ans)
            bits = None
            if score >= 1.0 and score_bits:
                bits = lm.completion_bits(prompts[i], f"{code}\n```")
            attempts.append(Attempt(t, code, ans, score, bits,
                                    frozenset(lib.called_in(code or ""))))
        out.append(attempts)
    return out
