from kompression.s1.library import Library, token_bits
import pytest

from kompression.s1.sandbox import container_available, run_in_container, run_program
from kompression.s1.worlds import HELDOUT_FAMILIES, TRAIN_FAMILIES, World


def test_families_disjoint():
    assert not set(TRAIN_FAMILIES) & set(HELDOUT_FAMILIES)


def test_world_generate_and_verify():
    w = World()
    t = w.generate("gcd", seed=1, n=2)[0]
    assert w.verify(t, t.answer) == 1.0
    assert w.verify(t, "nonsense") == 0.0
    assert w.verify(t, None) == 0.0


def test_sandbox_runs_with_library_and_times_out():
    lib = Library(Library.parse("def double(x):\n    return 2 * x\n"))
    r = run_program("print(double(21))", lib.source())
    assert r.ok and r.answer == "42"
    r = run_program("while True: pass", timeout_s=1.0, cpu_s=1)
    assert not r.ok
    r = run_program("raise SystemExit(3)")
    assert not r.ok


def test_library_reuse_and_bits():
    lib = Library(Library.parse("def f(x):\n    return x\n\ndef g(x):\n    return f(x)\n"))
    assert lib.called_in("print(g(1))") == {"g"}
    assert lib.called_in("syntax error (") == set()

    class Tok:
        def encode(self, s):
            return s.split()

    assert token_bits("a b c d", Tok(), 1024) == 40.0
    assert token_bits("", Tok(), 1024) == 0.0


@pytest.mark.skipif(not container_available(), reason="apple/container fehlt")
def test_container_sandbox_isolated():
    lib = Library(Library.parse("def double(x):\n    return 2 * x\n"))
    r = run_in_container("print(double(21))", lib.source())
    assert r.ok and r.answer == "42"
    r = run_in_container(
        "import urllib.request\nprint(urllib.request.urlopen('http://example.com', timeout=2).status)"
    )
    assert not r.ok  # kein Netz
    r = run_in_container("while True: pass", timeout_s=1.0)
    assert not r.ok and r.timed_out


def test_multiline_answer_is_kept():
    r = run_program("print('1 2')\nprint('3 4')")
    assert r.ok and r.answer == "1 2\n3 4"


def test_library_signatures_view_is_compact():
    lib = Library(Library.parse(
        'def bfs(grid, start):\n    """Breitensuche, gibt Distanzen zurück."""\n    return {}\n'))
    sig = lib.signatures()
    assert "def bfs(grid, start):" in sig and "Breitensuche" in sig and "return {}" not in sig
