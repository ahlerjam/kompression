import numpy as np
import pytest

from kompression.core import codelength as cl
from kompression.core import stats, weight_bits


def test_uniform_logprobs_give_log2_v_bits():
    V, n = 97, 50
    logp = np.full(n, -np.log(V))
    assert cl.bits_from_logprobs(logp) == pytest.approx(n * np.log2(V))
    assert cl.uniform_bits(n, V) == pytest.approx(n * np.log2(V))


def test_positive_logprob_rejected():
    with pytest.raises(ValueError):
        cl.bits_from_logprobs([0.1])


def test_doubling_blocks():
    assert cl.doubling_blocks(100, 8) == [8, 16, 32, 64, 100]
    assert cl.doubling_blocks(8, 8) == [8]


def test_prequential_by_hand():
    # Block 1: 2 Symbole uniform über 4 -> 4 Bit.
    # Block 2: 2 Symbole mit p=1/2 -> 2 Bit. Block 3: 4 Symbole mit p=1 -> 0 Bit.
    def enc(start, end):
        return np.full(end - start, -np.log(2.0)) if start == 2 else np.zeros(end - start)

    r = cl.prequential_codelength(8, 4, [2, 4, 8], enc)
    assert r["per_block_bits"] == pytest.approx([4.0, 2.0, 0.0])
    assert r["total_bits"] == pytest.approx(6.0)
    assert r["uniform_bits"] == pytest.approx(16.0)


def test_prequential_rejects_bad_shapes():
    with pytest.raises(ValueError):
        cl.prequential_codelength(8, 4, [2, 4, 8], lambda s, e: np.zeros(1))


def test_weight_bits_prefers_coarse_delta_when_loss_flat():
    rng = np.random.default_rng(0)
    w = [rng.normal(size=(64, 64))]
    r = weight_bits.weight_bits(w, loss_fn=lambda ws: 0.0, tolerance=0.0)
    assert r["absolute"]["delta"] == pytest.approx(max(weight_bits.DEFAULT_DELTAS))
    fine = weight_bits.weight_bits_at(w, 1e-3)
    assert r["bits"] < fine


def test_weight_bits_respects_tolerance():
    w = [np.linspace(-1, 1, 1000)]
    target = w[0].copy()

    def loss(ws):
        return float(np.mean((ws[0] - target) ** 2))

    r = weight_bits.weight_bits(w, loss_fn=loss, tolerance=1e-4)
    assert r["absolute"]["loss_quant"] <= 1e-4 + 1e-12
    assert r["absolute"]["delta"] < 1.0


def test_holm_monotone_and_bounded():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert all(0 <= p <= 1 for p in adj)


def test_paired_tests_detect_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(size=12)
    b = a - 1.0 + 0.01 * rng.normal(size=12)
    assert stats.paired_wilcoxon(a, b)["p"] < 0.01
    lo, hi = stats.bootstrap_ci(a, b)
    assert lo > 0.9 and hi < 1.1


def test_weight_bits_diff_description_wins_for_small_update():
    """W = W0 + kleines D: Seed + Differenz muss billiger sein als W selbst (E27)."""
    rng = np.random.default_rng(2)
    w0 = [rng.normal(size=(64, 64))]
    w = [w0[0] + 0.001 * rng.normal(size=(64, 64))]

    def loss(ws):
        return float(np.mean((ws[0] - w[0]) ** 2))

    r = weight_bits.weight_bits(w, loss_fn=loss, tolerance=1e-5, init=w0)
    assert r["description"] == "diff"
    assert r["diff"]["bits"] < r["absolute"]["bits"]
    assert r["recipe"] == "uq-entropy-v2"
