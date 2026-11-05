"""Engineering-Prüfstand (KEIN Befund): schnelle prequentielle Vergleiche auf kleinem p.

Rechnet Varianten nacheinander (ein MLX-Prozess), eigene Seeds 200–202 (nie in Kalibrierung
oder Hauptlauf), und gibt eine Tabelle der Gesamt-Codelänge je Seed aus. Dient nur dazu,
Ideen für Training/Architektur zu sichten und Code auf Plausibilität zu prüfen, bevor etwas
in ein Protokoll geht. Läufe landen unter runs/s0a_preq/*_bench_*.

  uv run python stages/s0a_grokking/eng_bench.py <satz> [--p 23] [--n-seeds 5]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ("200", "201", "202")

CYC = ["--steps-per-block", "250", "--cycles", "4", "--post-steps", "100",
       "--post-beta-warmup", "30"]

# Einstellung wie Hauptlauf 0a+ (750 Wach + 250 Schlaf, 4 Zyklen)
HL = ["--steps-per-block", "750", "--cycles", "4", "--post-steps", "250", "--post-beta-warmup", "50"]
MDL = [*HL, "--post", "mdl", "--post-mdl-beta", "0.1"]

SETS: dict[str, dict[str, list[str]]] = {
    # Strategie E30: Verbesserungen gegen die stärkste Kontrolle (wd λ10 im Schlaf)
    "v2": {
        "wd": [*HL, "--post", "wd"],
        "wd_l10": [*HL, "--post", "wd", "--post-weight-decay", "10"],
        "mdl": MDL,
        "hyb3": [*MDL, "--post-mdl-wd", "3"],
        "hyb10": [*MDL, "--post-mdl-wd", "10"],
        "prior0": [*MDL, "--post-mdl-pi0", "3", "--post-mdl-s0", "0.01"],
        "prior0_hyb3": [*MDL, "--post-mdl-pi0", "3", "--post-mdl-s0", "0.01", "--post-mdl-wd", "3"],
    },
    # Feinabstimmung Hybrid (nach v2 p = 23: hyb3 vorn)
    "v3": {
        "hyb1": [*MDL, "--post-mdl-wd", "1"],
        "hyb5": [*MDL, "--post-mdl-wd", "5"],
        "hyb3_b0.3": [*HL, "--post", "mdl", "--post-mdl-beta", "0.3", "--post-mdl-wd", "3"],
        "hyb3_b0.03": [*HL, "--post", "mdl", "--post-mdl-beta", "0.03", "--post-mdl-wd", "3"],
    },
    # Selbst gewählter Schlaf (Holdout-Codelänge je Seed); faire Kontrolle: gleiche Auswahl nur über L2
    "v4": {
        "sel_mdl": [*HL, "--post", "wd", "--post-select", "mdl:0.03", "mdl:0.1", "mdl:0.3", "mdl:1"],
        "sel_l2": [*HL, "--post", "wd", "--post-select", "wd:1", "wd:3", "wd:10", "wd:30"],
        "sel_mix": [*HL, "--post", "wd", "--post-select", "mdl:0.1", "mdl:0.3", "wd:10", "wd:30"],
    },
    # Varianten der gemischten Auswahl (nach v4 p = 47: sel_mix knapp vor wd λ 10, 4/5 Seeds)
    "v5": {
        "mix2": [*HL, "--post", "wd", "--post-select", "mdl:0.1", "wd:10"],
        "mix_h25": [*HL, "--post", "wd", "--post-select", "mdl:0.1", "mdl:0.3", "wd:10", "wd:30",
                    "--post-select-holdout", "0.25"],
    },
    # Fester Schlafplan je Zyklus (starkes L2 hilft früh beim Groken, mdl ist spät präziser)
    "v6": {
        "plan_lllm": [*HL, "--post", "wd", "--post-schedule", "wd:10", "wd:10", "wd:10", "mdl:0.1"],
        "plan_llmm": [*HL, "--post", "wd", "--post-schedule", "wd:10", "wd:10", "mdl:0.1", "mdl:0.1"],
        "plan_lmlm": [*HL, "--post", "wd", "--post-schedule", "wd:10", "mdl:0.1", "wd:10", "mdl:0.1"],
    },
    # Ist der mdl-Schlaf echte Kompression oder ein Norm-Effekt?
    "norm": {
        "wd": [*CYC, "--post", "wd"],
        "free": [*CYC, "--post", "free"],
        "wd_l10": [*CYC, "--post", "wd", "--post-weight-decay", "10"],
        "mdl_b0.3": [*CYC, "--post", "mdl", "--post-mdl-beta", "0.3"],
        "mdlnorm_b0.3": [*CYC, "--post", "mdl_norm", "--post-mdl-beta", "0.3"],
        "bitnet": [*CYC, "--post", "bitnet"],
    },
}


def run(name: str, extra: list[str], p: int, seeds: tuple[str, ...] = SEEDS) -> list[float]:
    tag = f"bench_p{p}_{name}"
    cmd = ["uv", "run", "python", "stages/s0a_grokking/prequential.py", "--arm", "wd", "--restart",
           "--p", str(p), "--first-block", "64", "--seeds", *seeds, *extra, "--name", tag]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    d = sorted(ROOT.glob(f"runs/s0a_preq/*_{tag}"))[-1]
    res = json.loads((d / "results.json").read_text())["prequential"]
    return [res[s]["total_bits"] for s in seeds]


def main() -> None:
    which = sys.argv[1]
    p = int(sys.argv[sys.argv.index("--p") + 1]) if "--p" in sys.argv else 23
    n = int(sys.argv[sys.argv.index("--n-seeds") + 1]) if "--n-seeds" in sys.argv else len(SEEDS)
    seeds = tuple(str(200 + i) for i in range(n))
    rows = {}
    for name, extra in SETS[which].items():
        rows[name] = run(name, extra, p, seeds)
        print(f"{name:14s} " + " ".join(f"{b:8.0f}" for b in rows[name])
              + f"  | Mittel {np.mean(rows[name]):8.0f}", flush=True)
    out = ROOT / "runs" / f"bench_{which}_p{p}.json"
    out.write_text(json.dumps(rows, indent=1))
    print("gespeichert:", out)


if __name__ == "__main__":
    main()
