"""0a+ Teil A (EXPLORATIV, nicht vorab registriert): Switch-Code über vorhandene Blockwerte.

Idee (van Erven, Grünwald & de Rooij 2007): Statt sich für die ganze Datenfolge auf EIN Modell
festzulegen, darf der Codierer an Blockgrenzen zwischen Modellen („Experten“) wechseln. Hier als
Fixed-Share-Mischung auf Blockebene (Herbster & Warmuth 1998), ein echter, decodierbarer Code:

    w_k ∝ Gewicht von Experte k;  Block b kostet  -log2 Σ_k w_k 2^(-bits_k,b)
    danach Bayes-Update w_k ← w_k 2^(-bits_k,b) / Σ, dann Fixed-Share w ← (1-α) w + α/K.

Jeder Experte codiert jeden Block bereits mit einem gültigen Code für den Block gegeben das Präfix
(prequentiell, inkl. eigener Self-Switch-Bits), daher ist die Mischung wieder ein gültiger Code.
α wird aus einem festen Gitter gewählt und mit log2(|Gitter|) Bit bezahlt. Block [0, 64) ist für
alle Experten gleich (uniform) und wird unverändert übernommen.

Keine neue GPU-Rechnung: liest nur `metrics.jsonl` der Hauptläufe 0a (`Hpreq_*`) und 0b (`H0b_*`).

  uv run python stages/s0a_grokking/explor_switch.py [--op sub|mul|add]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from kompression.core.stats import bootstrap_ci, paired_wilcoxon

RUNS = Path(__file__).resolve().parents[2] / "runs"
ALPHAS = (0.0, 0.01, 0.05, 0.125, 0.25)
SEEDS = tuple(range(10, 20))
P = 97


def latest(pattern: str) -> Path:
    """Neuester fertiger Lauf (mit results.json) zum Suchmuster."""
    c = sorted(d for d in RUNS.glob(pattern) if (d / "results.json").exists())
    if not c:
        raise FileNotFoundError(pattern)
    return c[-1]


def block_bits(run_dir: Path) -> np.ndarray:
    """[Seeds, Blöcke] Bits je Block (ab Block 64, Self-Switch-Bits enthalten)."""
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            if line.strip()]
    rows = [r for r in rows if "bits_per_seed" in r]
    rows.sort(key=lambda r: r["block_start"])
    seeds = json.loads((run_dir / "config.json").read_text())["args"]["seeds"]
    if list(seeds) != list(SEEDS):
        raise ValueError(f"{run_dir.name}: Seeds {seeds}")
    b = np.array([r["bits_per_seed"] for r in rows], dtype=np.float64).T
    # Gegenprobe gegen results.json: uniformer erster Block + Summe = total_bits
    res = json.loads((run_dir / "results.json").read_text())["prequential"]
    first = rows[0]["block_start"] * math.log2(P)
    for i, s in enumerate(seeds):
        tot = res[str(s)]["total_bits"]
        if abs(first + b[i].sum() - tot) > 1e-3 * tot:
            raise ValueError(f"{run_dir.name} Seed {s}: {first + b[i].sum():.1f} vs {tot:.1f}")
    return b


def fixed_share(bits: np.ndarray, alpha: float) -> float:
    """bits: [Experten, Blöcke] → Codelänge in Bit der Fixed-Share-Mischung (gleicher Prior)."""
    k = bits.shape[0]
    logw = np.full(k, -math.log(k))
    total = 0.0
    for col in bits.T:
        lj = logw - col * math.log(2)
        m = np.max(lj)
        lz = m + math.log(np.exp(lj - m).sum())
        total += -lz / math.log(2)
        post = np.exp(lj - lz)
        w = (1 - alpha) * post + alpha / k
        with np.errstate(divide="ignore"):
            logw = np.log(w)
    return total


def switch_code(experts: dict[str, np.ndarray], first_bits: float) -> np.ndarray:
    """Je Seed: min über α-Gitter + log2|Gitter| + erster Block."""
    names = list(experts)
    n_seeds = experts[names[0]].shape[0]
    out = []
    for i in range(n_seeds):
        m = np.stack([experts[nm][i] for nm in names])
        out.append(min(fixed_share(m, a) for a in ALPHAS) + math.log2(len(ALPHAS)) + first_bits)
    return np.array(out)


def single_best(experts: dict[str, np.ndarray], first_bits: float) -> np.ndarray:
    """Je Seed: bestes einzelnes Modell + log2(Anzahl) (wie die eingefrorenen Endpunkte)."""
    m = np.stack([e.sum(axis=1) for e in experts.values()])
    return m.min(axis=0) + math.log2(len(experts)) + first_bits


def oracle(experts: dict[str, np.ndarray], first_bits: float) -> np.ndarray:
    """Untere Schranke OHNE Codekosten: je Block das beste Modell (kein gültiger Code)."""
    m = np.stack(list(experts.values()))
    return m.min(axis=0).sum(axis=1) + first_bits


def report(title: str, experts: dict[str, np.ndarray], ref: np.ndarray, ref_name: str) -> dict:
    first = 64 * math.log2(P)
    sw = switch_code(experts, first)
    sb = single_best(experts, first)
    orc = oracle(experts, first)
    w = paired_wilcoxon(list(sw), list(ref))
    lo, hi = bootstrap_ci(list(sw), list(ref))
    r = {"vergleich": title, "experten": list(experts), "switch_kbit": round(sw.mean() / 1e3, 2),
         "bestes_einzeln_kbit": round(sb.mean() / 1e3, 2), "orakel_kbit": round(orc.mean() / 1e3, 2),
         f"{ref_name}_kbit": round(ref.mean() / 1e3, 2),
         "diff_switch_minus_ref_ki95_kbit": (round(lo / 1e3, 2), round(hi / 1e3, 2)),
         "p_wilcoxon": w["p"], "seeds_switch_besser": int((sw < ref).sum())}
    print(json.dumps(r, ensure_ascii=False))
    return r


def grid(pattern_prefix: str, names: list[str]) -> dict[str, np.ndarray]:
    return {n: block_bits(latest(f"s0a_preq/*_{pattern_prefix}{n}")) for n in names}


def main_0a() -> None:
    print("== 0a Addition: Switch über Arme (Hpreq_*) ==")
    wd = grid("Hpreq_", ["wd_l0.3", "wd_l1.0", "wd_l3.0"])
    mdl = grid("Hpreq_", ["mdl_b0.03", "mdl_b0.1", "mdl_b0.3"])
    bit = grid("Hpreq_", ["bitnet"])
    first = 64 * math.log2(P)
    ref_wd = single_best(wd, first)  # = eingefrorener Endpunkt wd (min Gitter + log2 3)
    report("wd-Gitter allein (Switch nur innerhalb wd)", wd, ref_wd, "wd")
    report("wd + mdl", {**wd, **mdl}, ref_wd, "wd")
    report("wd + bitnet", {**wd, **bit}, ref_wd, "wd")
    report("wd + mdl + bitnet", {**wd, **mdl, **bit}, ref_wd, "wd")
    # fairer Vergleich: Referenz darf ebenfalls wechseln (Switch nur über das wd-Gitter)
    sw_wd = switch_code(wd, first)
    report("wd + mdl vs. Switch(wd)", {**wd, **mdl}, sw_wd, "switch_wd")
    report("wd + bitnet vs. Switch(wd)", {**wd, **bit}, sw_wd, "switch_wd")
    report("mdl allein (Switch über mdl-Gitter) vs. Switch(wd)", mdl, sw_wd, "switch_wd")


def main_0b(op: str) -> None:
    print(f"== 0b {op}: Switch über S und Startpunkte (H0b_{op}_*) ==")
    s_vals = (30, 100, 300, 1000, 3000)

    def cond(c: str) -> dict[str, np.ndarray]:
        return {f"{c}_s{s}": block_bits(latest(f"s0a_preq/*_H0b_{op}_{c}_s{s}")) for s in s_vals}

    rnd, awd, amdl, abit = cond("random"), cond("wd"), cond("mdl"), cond("bitnet")
    first = 64 * math.log2(P)
    ref_r = single_best(rnd, first)  # = eingefrorener Endpunkt Zufall (min S + log2 5)
    ref_a = single_best(awd, first)
    report("Zufall allein (Switch über S)", rnd, ref_r, "random")
    report("A(wd) allein (Switch über S)", awd, ref_a, "A_wd")
    report("Zufall + A(wd) vs. Zufall", {**rnd, **awd}, ref_r, "random")
    report("Zufall + A(wd) vs. A(wd)", {**rnd, **awd}, ref_a, "A_wd")
    sw_r = switch_code(rnd, first)
    report("Zufall + A(wd) vs. Switch(Zufall)", {**rnd, **awd}, sw_r, "switch_random")
    report("Zufall + A(mdl) vs. Switch(Zufall)", {**rnd, **amdl}, sw_r, "switch_random")
    report("Zufall + A(mdl) vs. Zufall", {**rnd, **amdl}, ref_r, "random")
    report("Zufall + A(bitnet) vs. Zufall", {**rnd, **abit}, ref_r, "random")
    report("Zufall + A(wd) + A(mdl) + A(bitnet) vs. Zufall", {**rnd, **awd, **amdl, **abit},
           ref_r, "random")


if __name__ == "__main__":
    op = sys.argv[sys.argv.index("--op") + 1] if "--op" in sys.argv else "add"
    if op == "add":
        main_0a()
    else:
        main_0b(op)
