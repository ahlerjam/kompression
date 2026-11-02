"""PyTorch-Spiegel von `model_mlx.forward` (für Paritätstest und Cloud).

Nimmt denselben Parameter-Pytree als numpy-Arrays entgegen.
"""

from __future__ import annotations

import numpy as np
import torch

from kompression.model_mlx import ModelConfig


def _t(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32))


def _layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
    return (x - mu) * torch.rsqrt(var + eps) * g + b


@torch.no_grad()
def forward(params_np: dict, x: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    x = torch.as_tensor(np.asarray(x, dtype=np.int64))
    B, T = x.shape
    h = _t(params_np["tok_emb"])[x] + _t(params_np["pos_emb"])[:T]
    H, dh = cfg.n_heads, cfg.d_model // cfg.n_heads
    mask = torch.triu(torch.full((T, T), -1e9), diagonal=1)
    for lp in params_np["layers"]:
        lp = {k: _t(v) for k, v in lp.items()}
        a = _layer_norm(h, lp["ln1_g"], lp["ln1_b"])
        q = (a @ lp["wq"]).reshape(B, T, H, dh).transpose(1, 2)
        k = (a @ lp["wk"]).reshape(B, T, H, dh).transpose(1, 2)
        v = (a @ lp["wv"]).reshape(B, T, H, dh).transpose(1, 2)
        att = torch.softmax((q @ k.transpose(-1, -2)) / np.sqrt(dh) + mask, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, cfg.d_model)
        h = h + o @ lp["wo"]
        m = _layer_norm(h, lp["ln2_g"], lp["ln2_b"])
        h = h + torch.relu(m @ lp["w1"] + lp["b1"]) @ lp["w2"] + lp["b2"]
    h = _layer_norm(h[:, -1], _t(params_np["lnf_g"]), _t(params_np["lnf_b"]))
    return (h @ _t(params_np["unemb"])).numpy()
