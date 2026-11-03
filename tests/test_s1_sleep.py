from kompression.s1.library import Library
from kompression.s1.probe import UNSOLVED_BITS, evaluate
from kompression.s1.sleep import sleep_cycle
from kompression.s1.worlds import World


class Tok:
    def encode(self, s, add_special_tokens=False):
        return list(range(len(s.split())))


class FakeLM:
    """Löst gcd nur, wenn die Bibliothek `g` enthält; dann kostet die Lösung 2 Bit statt 40."""
    vocab_size = 1000
    tokenizer = Tok()

    def __init__(self, answer):
        self.answer = answer

    def chat(self, system, user):
        return [hash(user) % 997]

    def generate(self, prompts, max_tokens, temperature):
        out = []
        for _ in prompts:
            out.append(f"def g():\n    return {self.answer!r}\n```")
        return out

    def completion_bits(self, prompt, completion):
        return 5.0 * len(completion.split())


def test_fallback_and_sleep_accepts_only_if_total_drops():
    w = World()
    probe = w.generate("gcd", seed=3, n=2)
    lm = FakeLM(probe[0].answer)
    empty = evaluate(lm, w, probe, Library(), n=1, backend="subprocess")
    assert empty.library_bits == 0.0
    assert all(b <= UNSOLVED_BITS for b in empty.task_bits)
    assert empty.total_bits == sum(empty.task_bits)
    lib, res, log = sleep_cycle(lm, w, probe, Library(), ["print(1)"], empty, n_proposals=1,
                                probe_samples=1, backend="subprocess")
    assert log, "Vorschlag wurde nicht bewertet"
    assert (res.total_bits < empty.total_bits) == bool(lib.functions)


def test_verified_reuse_filter_keeps_only_reused_functions():
    from kompression.s1.sleep import filter_by_verified_reuse

    w = World()
    tasks = w.generate("gcd", seed=5, n=2)

    class RewriteLM(FakeLM):
        def generate(self, prompts, max_tokens, temperature):
            return [f"print(echo({t.answer!r}))\n```" for t in tasks][: len(prompts)]

    fns = {"echo": "def echo(x):\n    return x", "unused": "def unused():\n    return 0"}
    keep, uses = filter_by_verified_reuse(RewriteLM(tasks[0].answer), w,
                                          [(t, "print(1)") for t in tasks], fns, Library(),
                                          min_uses=2, backend="subprocess")
    assert set(keep) == {"echo"} and uses == {"echo": 2}
