"""Auswertung Hauptlauf 0a+ B1-Zyklus (Wach-Schlaf) nach docs/protokoll_0a_plus.md (Version 2).

Läufe heißen `B1_<bedingung>` (scripts/b1_0a_plus.sh 1/2), Seeds 20–29. Endpunkt je Seed:
prequentielle Gesamt-Codelänge (`total_bits`, Self-Switch-Bits enthalten). Keine Gitterwahl im
Hauptlauf (β auf Seeds 100–102 kalibriert), daher keine zusätzlichen Wahl-Bits.

  uv run python stages/s0a_grokking/analyse_b1.py [--prefix B1_]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import RUNS, compare

from kompression.core.stats import holm

CONDITIONS = ("wd", "mdl", "wd_l10", "mdlnorm", "bitnet", "prune", "free")
LATE_BLOCKS = (4096, 8192)  # H4: Bits der letzten beiden Blöcke


def latest_complete(pattern: str) -> Path | None:
    hits = [p for p in sorted(RUNS.glob(pattern)) if (p / "results.json").exists()]
    return hits[-1] if hits else None


def load(run: Path) -> tuple[dict[int, float], dict[int, float]]:
    """(Gesamtbits je Seed, Bits der letzten Blöcke je Seed)."""
    res = json.loads((run / "results.json").read_text())["prequential"]
    total = {int(k): v["total_bits"] for k, v in res.items()}
    seeds = json.loads((run / "config.json").read_text())["args"]["seeds"]
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines() if x.strip()]
    late = {int(s): 0.0 for s in seeds}
    found = set()
    for r in rows:
        if "bits_per_seed" in r and r["block_start"] in LATE_BLOCKS:
            found.add(r["block_start"])
            for s, b in zip(seeds, r["bits_per_seed"], strict=True):
                late[int(s)] += b
    if found != set(LATE_BLOCKS):
        raise ValueError(f"{run.name}: letzte Blöcke fehlen ({sorted(found)})")
    return total, late


def hypotheses(L: dict[str, dict[int, float]], late: dict[str, dict[int, float]]) -> list[dict]:
    pairs = [("H1 mdl-Schlaf < nur lernen (wd)", L, "mdl", "wd"),
             ("H2 mdl-Schlaf < wd λ10-Schlaf (Norm-Kontrolle)", L, "mdl", "wd_l10"),
             ("H3 mdl-Schlaf < Norm-Zwilling (mdl_norm)", L, "mdl", "mdlnorm"),
             ("H4 letzte Blöcke: mdl-Schlaf < nur lernen", late, "mdl", "wd")]
    out = [compare(n, d[a], d[b]) for n, d, a, b in pairs if d.get(a) and d.get(b)]
    for t, p in zip(out, holm([t["p"] for t in out]), strict=True):
        t["p_holm"] = p
    return out


def descriptive(L: dict[str, dict[int, float]]) -> list[dict]:
    return [compare(f"beschreibend: {c} gegen wd", L[c], L["wd"])
            for c in ("bitnet", "prune", "free", "wd_l10", "mdlnorm") if L.get(c) and L.get("wd")]


def main(prefix: str = "B1_") -> None:
    L, late = {}, {}
    for c in CONDITIONS:
        run = latest_complete(f"s0a_preq/*_{prefix}{c}")
        if run is None:
            print(c, "fehlt/unvollständig")
            continue
        L[c], late[c] = load(run)
        n = len(L[c])
        print(f"{c:8s} n = {n}  Mittel kbit: {sum(L[c].values()) / n / 1000:7.2f}"
              f"  letzte Blöcke Mittel bit: {sum(late[c].values()) / n:7.0f}  ({run.name})")
    print("\n# Hypothesen (Holm über H1–H4)")
    for t in hypotheses(L, late):
        print(json.dumps(t, ensure_ascii=False))
    print("\n# Beschreibend (ohne Test-Anspruch)")
    for t in descriptive(L):
        print(json.dumps(t, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "B1_")
