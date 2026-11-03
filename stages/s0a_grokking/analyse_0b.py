"""Auswertung Hauptlauf 0b (Transfer in Bit) nach docs/protokoll_0b.md.

Läufe heißen `H0b_<op>_<bedingung>_s<S>` (scripts/hauptlauf_0b.sh). Je Seed und Bedingung:
Codelänge = Minimum über die S-Werte + log2(Anzahl S) Bit (Wahl von S bezahlt).
Bedingungen: random, wd, wd_scaled, wd_permuted, mdl, bitnet (A-Init aus dem jeweiligen 0a-Arm).

  uv run python stages/s0a_grokking/analyse_0b.py [--prefix H0b_]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import RUNS, compare

from kompression.core.stats import holm

S_GRID = (30, 100, 300, 1000, 3000)
CONDITIONS = ("random", "wd", "wd_scaled", "wd_permuted", "mdl", "bitnet")


def run_name(prefix: str, op: str, cond: str, s: int) -> str:
    return f"{prefix}{op}_{cond}_s{s}"


def latest_complete(pattern: str) -> Path | None:
    hits = [p for p in sorted(RUNS.glob(pattern)) if (p / "results.json").exists()]
    return hits[-1] if hits else None


def min_over_s(per_s: dict[int, dict[int, float]], n_grid: int = len(S_GRID)) -> dict[int, float]:
    """per_s: S -> {seed: bits}. Nur Seeds, die für alle S vorliegen; + log2(n_grid) Bit."""
    if len(per_s) < n_grid:
        return {}
    seeds = set.intersection(*(set(d) for d in per_s.values()))
    return {s: min(d[s] for d in per_s.values()) + math.log2(n_grid) for s in seeds}


def load_condition(prefix: str, op: str, cond: str) -> dict[int, float]:
    per_s = {}
    for s in S_GRID:
        run = latest_complete(f"s0a_preq/*_{run_name(prefix, op, cond, s)}")
        if run is None:
            continue
        res = json.loads((run / "results.json").read_text())
        per_s[s] = {int(k): v["total_bits"] for k, v in res["prequential"].items()}
    return min_over_s(per_s)


def hypotheses(L: dict[str, dict[int, float]]) -> list[dict]:
    """H1 A-Init < Zufall, H2 A-Init < permutiert, H3 A-Init < Norm-skaliert, H4 mdl-A < wd-A
    (gleichbedeutend mit T(mdl) > T(wd), da beide gegen denselben Zufallsstart gemessen)."""
    pairs = [("H1 Transfer: wd-A < Zufall", "wd", "random"),
             ("H2 Struktur: wd-A < wd-A permutiert", "wd", "wd_permuted"),
             ("H3 nicht nur Norm: wd-A < wd-A Norm-skaliert", "wd", "wd_scaled"),
             ("H4 komprimiert überträgt besser: mdl-A < wd-A", "mdl", "wd")]
    out = [compare(n, L[a], L[b]) for n, a, b in pairs if L.get(a) and L.get(b)]
    for t, p in zip(out, holm([t["p"] for t in out]), strict=True):
        t["p_holm"] = p
    return out


def gate(tests: list[dict]) -> bool:
    ok = {t["kontrast"][:2]: t["p_holm"] < 0.05 and t["richtung_wie_erwartet"] for t in tests}
    return ok.get("H1", False) and ok.get("H2", False)


def main(prefix: str = "H0b_") -> None:
    for op in ("sub", "mul"):
        L = {c: load_condition(prefix, op, c) for c in CONDITIONS}
        print(f"\n## Zielaufgabe {op}")
        for c, d in L.items():
            if d:
                print(c, "n =", len(d), "Mittel kbit:", round(sum(d.values()) / len(d) / 1000, 2))
            else:
                print(c, "fehlt/unvollständig")
        if L["random"]:
            for c in ("wd", "mdl", "bitnet"):
                if L[c]:
                    t = [L["random"][s] - L[c][s] for s in sorted(set(L["random"]) & set(L[c]))]
                    print(f"Transfer-Gewinn T({c}) Mittel kbit:", round(sum(t) / len(t) / 1000, 2))
        tests = hypotheses(L)
        for t in tests:
            print(json.dumps(t, ensure_ascii=False))
        if op == "sub" and tests:
            print("Gate 0b:", "BESTANDEN" if gate(tests) else "nicht bestanden (Negativbefund dokumentieren)")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "H0b_")
