"""Endpunkt 0b: Minimum über S + log2(5), Gate H1 und H2."""

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stages/s0a_grokking"))
spec = importlib.util.spec_from_file_location("analyse_0b", ROOT / "stages/s0a_grokking/analyse_0b.py")
a0b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a0b)


def test_min_ueber_s_bezahlt_wahl():
    per_s = {s: {10: 1000.0 + s, 11: 5000.0 - s} for s in a0b.S_GRID}
    r = a0b.min_over_s(per_s)
    assert r[10] == 1030.0 + math.log2(5)
    assert r[11] == 2000.0 + math.log2(5)


def test_unvollstaendiges_gitter_zaehlt_nicht():
    per_s = {s: {10: 1.0} for s in a0b.S_GRID[:4]}
    assert a0b.min_over_s(per_s) == {}


def test_gate_braucht_h1_und_h2():
    seeds = range(10, 20)
    rnd = {s: 60_000.0 + s for s in seeds}
    wd = {s: 40_000.0 + 7 * s for s in seeds}
    perm = {s: 61_000.0 for s in seeds}
    tests = a0b.hypotheses({"random": rnd, "wd": wd, "wd_permuted": perm, "wd_scaled": perm,
                            "mdl": {}, "bitnet": {}})
    assert a0b.gate(tests)
    tests = a0b.hypotheses({"random": rnd, "wd": wd, "wd_permuted": {s: 30_000.0 for s in seeds},
                            "wd_scaled": perm, "mdl": {}, "bitnet": {}})
    assert not a0b.gate(tests)


def test_laufname():
    assert a0b.run_name("H0b_", "sub", "wd_permuted", 30) == "H0b_sub_wd_permuted_s30"
