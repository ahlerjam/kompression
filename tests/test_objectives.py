import math

import mlx.core as mx
import numpy as np
import pytest

from kompression.model_mlx import ModelConfig, init_params
from kompression.objectives import BitNet, Subspace, VariationalMDL
from kompression.train import TrainConfig, train

CFG = ModelConfig(vocab_size=13, n_out=11, d_model=32, n_heads=2, d_mlp=64)
SMALL = dict(p=11, d_model=32, n_heads=2, d_mlp=64, eval_every=50, seeds=(0,))


def test_bitnet_weights_are_ternary():
    w = BitNet.ternary(mx.array(np.random.default_rng(0).normal(size=(8, 8)), dtype=mx.float32))
    vals = np.unique(np.round(np.array(w) / np.abs(np.array(w)).max(), 5))
    assert set(vals.tolist()) <= {-1.0, 0.0, 1.0}


def test_subspace_starts_at_init_and_counts_params():
    p0 = init_params(CFG, 0)
    arm = Subspace(rank=4)
    tr, const = arm.init(p0, 0)
    eff = arm.effective(tr, const, None)
    np.testing.assert_allclose(np.array(eff["unemb"]), np.array(p0["unemb"]))
    n_train = sum(z.size for z in tr["z"].values())
    assert n_train == sum(min(4, a.shape[0]) * min(4, a.shape[1])
                          for a in [p0["tok_emb"], p0["pos_emb"], p0["unemb"]]) + 2 * (
        4 * 16 + 2 * 16)  # wq,wk,wv,wo (4x4) + w1,w2 (4x4) je Layer


def test_mdl_penalty_is_kl_for_gaussian_prior():
    """Mit einer einzigen Prior-Komponente ist die MC-KL im Mittel die analytische KL."""
    arm = VariationalMDL(k=1, init_log_sigma=math.log(0.5))
    p = {"w": mx.zeros((20000,))}
    tr, const = arm.init(p, 0)
    tr["prior"]["log_s"] = mx.array([0.0])
    mx.random.seed(0)
    kl = float(arm.penalty(tr, const, arm.noise_like(tr))) / 20000
    expected = math.log(1 / 0.5) + (0.25) / 2 - 0.5
    assert kl == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize("arm", ["bitnet", "subspace", "mdl"])
def test_arms_train_without_nan(arm):
    r = train(TrainConfig(steps=60, arm=arm, beta_warmup=50, **SMALL))
    assert all(np.isfinite(v) for v in r["final_delta_norm"].values())


def test_pruned_masks_fraction_and_keeps_1d():
    from kompression.objectives import Pruned

    p0 = init_params(CFG, 0)
    arm = Pruned(frac=0.75)
    tr, const = arm.init(p0, 0)
    eff = arm.effective(tr, const, None)
    w = np.array(eff["unemb"])
    assert abs((w == 0).mean() - 0.75) < 0.01
    one_d = [k for k, v in p0.items() if hasattr(v, "ndim") and v.ndim == 1]
    for k in one_d:
        np.testing.assert_allclose(np.array(eff[k]), np.array(p0[k]))


def test_mdl_snap_uses_prior_means():
    p0 = init_params(CFG, 0)
    arm = VariationalMDL(k=4)
    tr, _ = arm.init(p0, 0)
    snapped = arm.snap(tr)
    means = np.concatenate([[0.0], np.array(tr["prior"]["mean"])]).astype(np.float32)
    assert np.isin(np.unique(np.array(snapped["unemb"])), means).all()
    one_d = [k for k, v in p0.items() if hasattr(v, "ndim") and v.ndim == 1]
    for k in one_d:
        np.testing.assert_allclose(np.array(snapped[k]), np.array(p0[k]))
