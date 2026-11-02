import mlx.core as mx
import numpy as np
import pytest

from kompression import model_torch
from kompression.data import ModTask, split
from kompression.model_mlx import (
    ModelConfig,
    forward,
    init_params,
    log_stablemax,
    params_to_numpy,
)
from kompression.train import TrainConfig, train, unstack_tree

SMALL = dict(p=11, d_model=32, n_heads=2, d_mlp=64, eval_every=50)


def test_split_is_deterministic_and_disjoint():
    t = ModTask(p=11)
    a, b = split(t, 0.5, 3), split(t, 0.5, 3)
    np.testing.assert_array_equal(a["x_train"], b["x_train"])
    tr = {tuple(r) for r in a["x_train"]}
    te = {tuple(r) for r in a["x_test"]}
    assert not tr & te and len(tr) + len(te) == 121


def test_stablemax_is_a_distribution():
    logits = mx.array(np.random.default_rng(0).normal(size=(5, 7)) * 10, dtype=mx.float32)
    p = mx.exp(log_stablemax(logits))
    np.testing.assert_allclose(np.array(p.sum(-1)), 1.0, atol=1e-5)
    assert float(p.min()) > 0


def test_parity_mlx_torch():
    cfg = ModelConfig(vocab_size=13, n_out=11, d_model=32, n_heads=2, d_mlp=64)
    params = init_params(cfg, seed=7)
    x = ModTask(p=11).all_pairs()[0][:64]
    out_mlx = np.array(forward(params, mx.array(x), cfg))
    out_torch = model_torch.forward(params_to_numpy(params), x, cfg)
    np.testing.assert_allclose(out_mlx, out_torch, atol=1e-4, rtol=1e-4)


def test_overfit_small_task():
    r = train(TrainConfig(seeds=(0,), steps=300, train_frac=0.5, **SMALL))
    assert r["final_train_acc"][0] == pytest.approx(1.0)


def test_determinism():
    """Gleicher Seed -> gleiches Ergebnis bis auf GPU-Rundung (Metal summiert nicht bitgenau)."""
    cfg = TrainConfig(seeds=(0,), steps=50, **SMALL)
    a, b = train(cfg), train(cfg)
    np.testing.assert_allclose(
        np.array(a["params"]["unemb"]), np.array(b["params"]["unemb"]), atol=1e-5
    )


def test_vmap_equivalence():
    """Seed i im gestapelten Lauf == Einzellauf mit Seed i."""
    batch = train(TrainConfig(seeds=(0, 1), steps=50, **SMALL))
    single = train(TrainConfig(seeds=(1,), steps=50, **SMALL))
    np.testing.assert_allclose(
        np.array(unstack_tree(batch["params"], 1)["unemb"]),
        np.array(unstack_tree(single["params"], 0)["unemb"]),
        atol=1e-5,
    )


def test_norm_projection_follows_target(tmp_path):
    import json

    steps = 60
    rows = [{"step": st, "seed": 0, "weight_norm": 20.0 + st / 10} for st in (10, 60)]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    r = train(TrainConfig(seeds=(0,), steps=steps, arm="wd", weight_decay=1.0,
                          norm_target_run=str(tmp_path), **SMALL))
    assert r["final_weight_norm"][0] == pytest.approx(26.0, rel=1e-3)
