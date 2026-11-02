from kompression.s1.library import Library, token_bits
from kompression.s1.sandbox import run_program
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
