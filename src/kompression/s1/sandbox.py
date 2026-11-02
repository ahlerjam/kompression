"""Ausführung von Lösungsprogrammen.

Zwei Backends:
- `container` (Standard, E17): dauerhaft laufende Linux-VM über apple/container,
  ohne Netz, Speicher- und CPU-begrenzt; Programme laufen per `container exec`.
- `subprocess`: lokaler Subprozess mit Zeit- und CPU-Limit (nur Tests/Notfall,
  kein Netz- und Dateischutz).
Konvention: Das Programm gibt die Antwort als letzte Zeile auf stdout aus.
"""

from __future__ import annotations

import os
import shutil
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


CONTAINER_NAME = "kompression-sandbox"
CONTAINER_IMAGE = "docker.io/library/python:3.12-slim"


def container_available() -> bool:
    return shutil.which("container") is not None


def ensure_container(cpus: int = 4, memory: str = "2g") -> None:
    """Startet die Sandbox-VM einmalig (ohne Netz). Idempotent."""
    ls = subprocess.run(["container", "ls", "--quiet"], capture_output=True, text=True)
    if CONTAINER_NAME in ls.stdout.split():
        return
    subprocess.run(["container", "rm", CONTAINER_NAME], capture_output=True)
    subprocess.run(
        ["container", "run", "-d", "--name", CONTAINER_NAME, "--network", "none",
         "-c", str(cpus), "-m", memory, CONTAINER_IMAGE, "sleep", "infinity"],
        check=True, capture_output=True, text=True,
    )


def run_in_container(code: str, library_source: str = "", timeout_s: float = 5.0) -> ExecResult:
    program = (library_source + "\n\n" if library_source else "") + code
    ensure_container()
    try:
        p = subprocess.run(
            ["container", "exec", "-i", CONTAINER_NAME, "timeout", "-s", "KILL",
             str(max(1, int(timeout_s))), "python", "-I", "-"],
            input=program, capture_output=True, text=True, timeout=timeout_s + 5,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(False, None, e.stdout or "", e.stderr or "", True)
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
    timed_out = p.returncode in (124, 137, -9)
    return ExecResult(p.returncode == 0 and bool(lines), lines[-1] if lines else None,
                      p.stdout, p.stderr, timed_out)


def execute(code: str, library_source: str = "", timeout_s: float = 5.0,
            backend: str = "container") -> ExecResult:
    if backend == "container":
        return run_in_container(code, library_source, timeout_s)
    return run_program(code, library_source, timeout_s)
