"""Auswertung Stufe 0a nach Protokoll (Version 1): Endpunkte je Arm, gepaarte Tests, Holm.

  uv run python stages/s0a_grokking/analyse.py [--prefix H_]
Sucht die jeweils neuesten Läufe runs/s0a/*_<prefix><arm> und runs/s0a_preq/*_<prefix>preq_<arm>.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from kompression.core.stats import bootstrap_ci, holm, paired_wilcoxon

RUNS = Path(__file__).resolve().parents[2] / "runs"
BUDGET = 40_000


def latest(pattern: str) -> Path | None:
    hits = sorted(RUNS.glob(pattern))
    return hits[-1] if hits else None


def train_endpoints(run: Path) -> dict[int, dict]:
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    res = json.loads((run / "results.json").read_text())
    budget = max(r["step"] for r in rows)
    out = {}
    for seed in sorted({r["seed"] for r in rows}):
        tr = [(r["step"], r["test_acc"]) for r in rows if r["seed"] == seed]
        stable = budget
        for st, _ in tr:
            win = [a for s2, a in tr if st <= s2 < st + 2000]
            if len(win) >= 20 and min(win) >= 0.99:
                stable = st
                break
        half = [a for s2, a in tr if s2 > budget / 2]
        lw = res.get("weight_bits", {}).get(str(seed), {}).get("bits")
        out[seed] = {"stable_grok": stable, "frac99": float(np.mean([a >= 0.99 for a in half])),
                     "lw_bits": lw, "final_test": tr[-1][1]}
    return out


def preq_endpoints(runs: list[Path]) -> tuple[dict[int, float], dict[int, float]]:
    """Primär: Minimum über das Hyperparameter-Gitter + log2(K) Bit (Protokoll v2).
    Sekundär: Bits der letzten beiden Blöcke beim jeweils gewählten Gitterpunkt."""
    import math

    per_seed: dict[int, list[tuple[float, float]]] = {}
    for run in runs:
        res = json.loads((run / "results.json").read_text())
        for s, v in res["prequential"].items():
            per_seed.setdefault(int(s), []).append((v["total_bits"], sum(v["per_block_bits"][-2:])))
    k = len(runs)
    prim, sec = {}, {}
    for s, v in per_seed.items():
        if len(v) == k:
            best = min(v)
            prim[s] = best[0] + math.log2(k)
            sec[s] = best[1]
    return prim, sec


def preq_patterns(prefix: str) -> dict[str, str]:
    """Suchmuster der prequentiellen Läufe. Hauptlauf heißt `Hpreq_*` (Präfix `H_` ohne `_`),
    Kalibrierung `preq_*` (Präfix leer)."""
    pp = prefix.rstrip("_") + "preq_"
    return {"mdl": f"s0a_preq/*_{pp}mdl_b*", "wd": f"s0a_preq/*_{pp}wd_l*",
            "bitnet": f"s0a_preq/*_{pp}bitnet", "none": f"s0a_preq/*_{pp}none"}


def compare(name: str, a: dict, b: dict, lower_is_better: bool = True) -> dict:
    seeds = sorted(set(a) & set(b))
    x = [a[s] for s in seeds]
    y = [b[s] for s in seeds]
    w = paired_wilcoxon(x, y)
    lo, hi = bootstrap_ci(x, y)
    better = (np.mean(x) < np.mean(y)) if lower_is_better else (np.mean(x) > np.mean(y))
    return {"kontrast": name, "n": len(seeds), "mittel_a": float(np.mean(x)),
            "mittel_b": float(np.mean(y)), "diff_ki95": (lo, hi), "p": w["p"],
            "richtung_wie_erwartet": bool(better)}


def main(prefix: str = "H_") -> None:
    arms = ["mdl", "wd", "wd_normproj", "bitnet", "none"]
    tr = {a: train_endpoints(r) for a in arms if (r := latest(f"s0a/*_{prefix}{a}"))}
    grids = preq_patterns(prefix)
    pq, pq_late = {}, {}
    for a_, pat in grids.items():
        runs = sorted(RUNS.glob(pat))
        if runs:
            pq[a_], pq_late[a_] = preq_endpoints(runs)
        else:
            print(f"WARNUNG: keine prequentiellen Läufe für {a_} ({pat})")
    print("# Endpunkte je Arm (Mittel über Seeds)")
    for a, d in tr.items():
        v = list(d.values())
        print(a, {k: round(float(np.mean([x[k] for x in v if x[k] is not None])), 3)
                  for k in ("stable_grok", "frac99", "lw_bits", "final_test")},
              "prequential:", round(float(np.mean(list(pq[a].values())))) if a in pq else "-")

    def col(arm, key):
        return {s: v[key] for s, v in tr[arm].items() if v[key] is not None}

    primary = []
    if "mdl" in pq and "wd" in pq:
        primary.append(compare("H1 prequential mdl < wd", pq["mdl"], pq["wd"]))
    if "bitnet" in pq and "wd" in pq:
        primary.append(compare("H4 prequential bitnet < wd", pq["bitnet"], pq["wd"]))
    tests = []
    if "mdl" in tr and "wd_normproj" in tr:
        tests.append(compare("H2a stabiler Grok mdl < wd_normproj", col("mdl", "stable_grok"),
                             col("wd_normproj", "stable_grok")))
        tests.append(compare("H2b Anteil≥0,99 mdl > wd_normproj", col("mdl", "frac99"),
                             col("wd_normproj", "frac99"), lower_is_better=False))
    for other in ("wd", "wd_normproj"):
        if "mdl" in tr and other in tr:
            tests.append(compare(f"H3 L(W) mdl < {other}", col("mdl", "lw_bits"), col(other, "lw_bits")))
    if "bitnet" in tr and "wd" in tr:
        tests.append(compare("H4a stabiler Grok bitnet < wd", col("bitnet", "stable_grok"),
                             col("wd", "stable_grok")))
    for arm in ("mdl", "bitnet"):
        if arm in pq_late and "wd" in pq_late:
            tests.append(compare(f"H5 letzte Blöcke {arm} < wd", pq_late[arm], pq_late["wd"]))
    if primary:
        for t, p_adj in zip(primary, holm([t["p"] for t in primary]), strict=True):
            t["p_holm"] = p_adj
        print("\n# Primäre Tests (H1, H4; Holm)")
        for t in primary:
            print(json.dumps(t, ensure_ascii=False))
        ok = any(t["p_holm"] < 0.05 and t["richtung_wie_erwartet"] for t in primary)
        print("\nGate 0a:", "BESTANDEN" if ok else "nicht bestanden (Negativbefund dokumentieren)")
    if tests:
        for t, p_adj in zip(tests, holm([t["p"] for t in tests]), strict=True):
            t["p_holm"] = p_adj
        print("\n# Sekundäre Tests (H2, H3, H4a, H5; Holm)")
        for t in tests:
            print(json.dumps(t, ensure_ascii=False))


if __name__ == "__main__":
    pre = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "H_"
    main(pre)
