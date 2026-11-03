"""Suchmuster der 0a-Auswertung müssen die Laufnamen aus scripts/hauptlauf_0a.sh treffen."""

import fnmatch
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyse", ROOT / "stages/s0a_grokking/analyse.py")
analyse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyse)


def test_hauptlauf_namen_werden_gefunden():
    pats = analyse.preq_patterns("H_")
    names = {"mdl": "20261103-023705_Hpreq_mdl_b0.1", "wd": "20261103-100908_Hpreq_wd_l1.0",
             "bitnet": "20261104-000000_Hpreq_bitnet", "none": "20261104-000000_Hpreq_none"}
    for arm, name in names.items():
        assert fnmatch.fnmatch("s0a_preq/" + name, pats[arm]), arm
    assert not fnmatch.fnmatch("s0a_preq/20261103-0_Hpreq_wd_l1.0", pats["mdl"])


def test_kalibrierung_ohne_praefix():
    assert analyse.preq_patterns("")["wd"] == "s0a_preq/*_preq_wd_l*"
