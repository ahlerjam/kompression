from kompression.s1.library import Library
from kompression.s1.probe import evaluate, fallback_bits
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
    assert empty.task_bits[1] == fallback_bits(probe[1].answer, lm, probe[1].question) or empty.solved >= 1
    lib, res, log = sleep_cycle(lm, w, probe, Library(), ["print(1)"], empty, n_proposals=1,
                                probe_samples=1, backend="subprocess")
    assert log, "Vorschlag wurde nicht bewertet"
    assert (res.total_bits < empty.total_bits) == bool(lib.functions)
