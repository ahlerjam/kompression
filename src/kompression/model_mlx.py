"""Kleiner dichter Transformer, rein funktional (Parameter als dict-Pytree).

Funktional, damit `mx.vmap` über einen Seed-Stapel von Parametern laufen kann.
Der PyTorch-Spiegel in `model_torch.py` muss exakt dieselbe Rechnung machen.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    n_out: int
    seq_len: int = 4
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 512


def init_params(cfg: ModelConfig, seed: int) -> dict:
    key = mx.random.key(seed)
    keys = iter(mx.random.split(key, 4 + 6 * cfg.n_layers))
    d = cfg.d_model

    def normal(shape, std):
        return mx.random.normal(shape, key=next(keys)) * std

    params = {
        "tok_emb": normal((cfg.vocab_size, d), 1.0 / np.sqrt(d)),
        "pos_emb": normal((cfg.seq_len, d), 1.0 / np.sqrt(d)),
        "lnf_g": mx.ones((d,)),
        "lnf_b": mx.zeros((d,)),
        "unemb": normal((d, cfg.n_out), 1.0 / np.sqrt(d)),
        "layers": [],
    }
    for _ in range(cfg.n_layers):
        params["layers"].append(
            {
                "ln1_g": mx.ones((d,)),
                "ln1_b": mx.zeros((d,)),
                "wq": normal((d, d), 1.0 / np.sqrt(d)),
                "wk": normal((d, d), 1.0 / np.sqrt(d)),
                "wv": normal((d, d), 1.0 / np.sqrt(d)),
                "wo": normal((d, d), 1.0 / np.sqrt(d)),
                "ln2_g": mx.ones((d,)),
                "ln2_b": mx.zeros((d,)),
                "w1": normal((d, cfg.d_mlp), 1.0 / np.sqrt(d)),
                "b1": mx.zeros((cfg.d_mlp,)),
                "w2": normal((cfg.d_mlp, d), 1.0 / np.sqrt(cfg.d_mlp)),
                "b2": mx.zeros((d,)),
            }
        )
    next(keys)  # Reserve, damit sich die Schlüsselfolge bei Erweiterung nicht verschiebt
    return params


def _layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps) * g + b


def forward(params: dict, x: mx.array, cfg: ModelConfig) -> mx.array:
    """x: int [B, T] -> Logits [B, n_out] an der letzten Position."""
    B, T = x.shape
    h = params["tok_emb"][x] + params["pos_emb"][:T]
    H, dh = cfg.n_heads, cfg.d_model // cfg.n_heads
    mask = mx.triu(mx.full((T, T), -1e9), k=1)
    for lp in params["layers"]:
        a = _layer_norm(h, lp["ln1_g"], lp["ln1_b"])
        q = (a @ lp["wq"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = (a @ lp["wk"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = (a @ lp["wv"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        att = mx.softmax((q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh) + mask, axis=-1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg.d_model)
        h = h + o @ lp["wo"]
        m = _layer_norm(h, lp["ln2_g"], lp["ln2_b"])
        h = h + mx.maximum(m @ lp["w1"] + lp["b1"], 0.0) @ lp["w2"] + lp["b2"]
    h = _layer_norm(h[:, -1], params["lnf_g"], params["lnf_b"])
    return h @ params["unemb"]


def log_softmax(logits: mx.array) -> mx.array:
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def log_stablemax(logits: mx.array) -> mx.array:
    """StableMax (Prieto et al. 2025): s(x) = x+1 für x>=0, sonst 1/(1-x)."""
    s = mx.where(logits >= 0, logits + 1.0, 1.0 / (1.0 - mx.minimum(logits, 0.0)))
    return mx.log(s) - mx.log(s.sum(axis=-1, keepdims=True))


def label_logprobs(logits: mx.array, y: mx.array, use_stablemax: bool) -> mx.array:
    lp = log_stablemax(logits) if use_stablemax else log_softmax(logits)
    return mx.take_along_axis(lp, y[:, None], axis=-1)[:, 0]


def params_to_numpy(params) -> dict:
    from mlx.utils import tree_map

    return tree_map(lambda a: np.array(a, dtype=np.float32), params)
