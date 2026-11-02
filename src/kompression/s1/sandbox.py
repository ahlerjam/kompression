"""Ausführung von Lösungsprogrammen in einem Subprozess mit Zeit- und CPU-Limit.

PILOT-STAND: kein Netz- und Dateisystem-Schutz. Für unbeaufsichtigte Läufe ist
eine echte Isolation nötig (E17, apple/container), siehe STATUS „Wartet auf Jonas“.
Konvention: Das Programm gibt die Antwort als letzte Zeile auf stdout aus.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class ExecResult:
    ok: bool
    answer: str | None
    stdout: str
    stderr: str
    timed_out: bool


def _limits(cpu_s: int, mem_mb: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (mem_mb << 20, mem_mb << 20))
        except (ValueError, OSError):
            pass
        os.setsid()

    return apply


def run_program(code: str, library_source: str = "", timeout_s: float = 5.0,
                cpu_s: int = 5, mem_mb: int = 512) -> ExecResult:
    program = (library_source + "\n\n" if library_source else "") + code
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            p = subprocess.run(
                [sys.executable, "-I", path], cwd=tmp, capture_output=True, text=True,
                timeout=timeout_s, preexec_fn=_limits(cpu_s, mem_mb),
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(False, None, e.stdout or "", e.stderr or "", True)
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
    return ExecResult(p.returncode == 0 and bool(lines), lines[-1] if lines else None,
                      p.stdout, p.stderr, False)
