"""Lauf-Protokoll, local-first (Entscheidung E18).

runs/<stage>/<zeitstempel>_<name>/
  config.json    vollständige Konfiguration
  meta.json      Git-SHA, dirty-Flag, Diff-Hash, uv.lock-Hash, Versionen, Hardware
  metrics.jsonl  eine Zeile pro Messpunkt (append + flush)
  results.json   Endergebnis
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return ""


def environment_meta() -> dict:
    import mlx.core as mx
    import numpy as np

    diff = _git("diff", "HEAD")
    lock = REPO / "uv.lock"
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "chip": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip(),
        "started": dt.datetime.now().isoformat(timespec="seconds"),
    }


class Run:
    def __init__(self, stage: str, name: str, config, root: Path | None = None):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = (root or REPO / "runs") / stage / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        cfg = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        (self.dir / "meta.json").write_text(json.dumps(environment_meta(), indent=2))
        if diff := _git("diff", "HEAD"):
            (self.dir / "uncommitted.diff").write_text(diff)
        self._metrics = open(self.dir / "metrics.jsonl", "a")

    def log(self, **row) -> None:
        self._metrics.write(json.dumps(row, default=float) + "\n")
        self._metrics.flush()

    def finish(self, results: dict) -> None:
        (self.dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
        self._metrics.close()
