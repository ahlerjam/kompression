"""Auswertung 0a+ B1: Laden der Endpunkte und Hypothesen mit Holm."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stages/s0a_grokking"))
spec = importlib.util.spec_from_file_location("analyse_b1", ROOT / "stages/s0a_grokking/analyse_b1.py")
ab1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab1)


def _fake_run(tmp: Path, seeds, blocks: dict[int, list[float]], total: list[float]) -> Path:
    tmp.mkdir(parents=True)
    (tmp / "config.json").write_text(json.dumps({"args": {"seeds": seeds}}))
    (tmp / "results.json").write_text(json.dumps(
        {"prequential": {str(s): {"total_bits": t} for s, t in zip(seeds, total, strict=True)}}))
    rows = [{"block_start": b, "bits_per_seed": v} for b, v in blocks.items()]
    (tmp / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp


def test_load_summiert_letzte_bloecke(tmp_path):
    run = _fake_run(tmp_path / "r", [20, 21], {2048: [5.0, 6.0], 4096: [1.0, 2.0], 8192: [3.0, 4.0]},
                    [100.0, 200.0])
    total, late = ab1.load(run)
    assert total == {20: 100.0, 21: 200.0}
    assert late == {20: 4.0, 21: 6.0}


def test_load_verlangt_letzte_bloecke(tmp_path):
    run = _fake_run(tmp_path / "r", [20], {2048: [5.0], 4096: [1.0]}, [100.0])
    with pytest.raises(ValueError):
        ab1.load(run)


def test_hypothesen_holm_und_richtung():
    seeds = range(20, 30)
    mdl = {s: 10_000.0 + s for s in seeds}
    worse = {s: 20_000.0 + 3 * s for s in seeds}
    L = {"mdl": mdl, "wd": worse, "wd_l10": worse, "mdlnorm": {s: 5_000.0 for s in seeds}}
    tests = ab1.hypotheses(L, {"mdl": mdl, "wd": worse})
    assert [t["kontrast"][:2] for t in tests] == ["H1", "H2", "H3", "H4"]
    assert all("p_holm" in t for t in tests)
    assert tests[0]["richtung_wie_erwartet"] and not tests[2]["richtung_wie_erwartet"]
    assert tests[0]["p_holm"] >= tests[0]["p"]
