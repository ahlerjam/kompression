from kompression.s1.library import Library
from kompression.s1.solver import extract_code, solve
from kompression.s1.worlds import World


def test_extract_code():
    assert extract_code("bla\n```python\nprint(1)\n```\n") == "print(1)"
    assert extract_code("no code") is None


class FakeLM:
    vocab_size = 100

    def chat(self, system, user):
        return [1, 2, 3]

    def generate(self, prompts, max_tokens, temperature):
        return ["```python\nprint(gcd_all())\n```", "```python\nprint('x')\n```"] * (len(prompts) // 2)

    def completion_bits(self, prompt, completion):
        return float(len(completion))


def test_solve_with_fake_lm_scores_and_reuse():
    w = World()
    t = w.generate("gcd", seed=1, n=1)[0]
    lib = Library({"gcd_all": f"def gcd_all():\n    return {t.answer!r}"})
    (attempts,) = solve(FakeLM(), w, [t], lib, n=2, backend="subprocess")
    assert attempts[0].score == 1.0 and attempts[0].bits > 0
    assert attempts[0].lib_calls == {"gcd_all"}
    assert attempts[1].score == 0.0 and attempts[1].bits is None
