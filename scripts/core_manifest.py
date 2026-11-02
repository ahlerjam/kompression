"""Prüfsummen über den unantastbaren Kern.

  uv run python scripts/core_manifest.py          prüft (Exit 1 bei Abweichung)
  uv run python scripts/core_manifest.py --write  schreibt neu (NUR nach menschlicher Freigabe)
"""

import hashlib
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "kompression" / "core"
MANIFEST = CORE / "MANIFEST.sha256"


def current() -> str:
    lines = []
    for f in sorted(CORE.glob("*.py")):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
    return "\n".join(lines) + "\n"


def check() -> bool:
    return MANIFEST.exists() and MANIFEST.read_text() == current()


if __name__ == "__main__":
    if "--write" in sys.argv:
        MANIFEST.write_text(current())
        print(f"geschrieben: {MANIFEST}")
    elif check():
        print("Kern unverändert")
    else:
        print("KERN WEICHT VOM MANIFEST AB", file=sys.stderr)
        sys.exit(1)
