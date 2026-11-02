"""L(W) nachträglich für einen Lauf (Arme none/wd/mdl) aus gespeicherten Gewichten berechnen.

  uv run python scripts/lw_posthoc.py runs/s0a/<lauf> [--tolerance 0.05]
W0 wird aus dem Seed neu erzeugt (init_params ist deterministisch). Ergebnis: <lauf>/weight_bits_posthoc.json
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from kompression.data import split
from kompression.measure import weight_bits_per_seed
from kompression.model_mlx import init_params
from kompression.train import TrainConfig, stack_trees


def main(run_dir: str, tolerance: float = 0.05) -> None:
    d = Path(run_dir)
    cfg_raw = json.loads((d / "config.json").read_text())
    tc = cfg_raw.get("train", cfg_raw)
    fields = TrainConfig.__dataclass_fields__
    tc = {k: (tuple(v) if isinstance(v, list) else v) for k, v in tc.items() if k in fields}
    cfg = TrainConfig(**tc)
    npz = d / ("params.npz" if (d / "params.npz").exists() else "params_final.npz")
    flat = dict(np.load(npz))
    params = tree_unflatten([(k, mx.array(v)) for k, v in flat.items()])
    w0 = stack_trees([init_params(cfg.model(), s) for s in cfg.seeds])
    # Schlüsselreihenfolge an die gespeicherten Gewichte angleichen
    assert {k for k, _ in tree_flatten(w0)} == set(flat), "Parameterstruktur passt nicht"
    splits = [split(cfg.task(), cfg.train_frac, s) for s in cfg.seeds]
    data = {k: np.stack([sp[k] for sp in splits]) for k in splits[0]}
    res = weight_bits_per_seed(cfg, {"params": params, "w0": w0}, data, tolerance)
    (d / "weight_bits_posthoc.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({s: (round(v["bits"] / 1e3, 1), v["description"]) for s, v in res.items()}))


if __name__ == "__main__":
    tol = float(sys.argv[sys.argv.index("--tolerance") + 1]) if "--tolerance" in sys.argv else 0.05
    main(sys.argv[1], tol)
