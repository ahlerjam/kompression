import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "core_manifest", Path(__file__).resolve().parents[1] / "scripts" / "core_manifest.py"
)
core_manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core_manifest)


def test_core_matches_manifest():
    assert core_manifest.check(), (
        "src/kompression/core weicht von MANIFEST.sha256 ab. Der Kern darf nur nach "
        "menschlicher Freigabe geändert werden (siehe AGENTS.md)."
    )
