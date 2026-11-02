"""Modulare Arithmetik als Sequenzaufgabe: [a, op, b, =] -> c."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OPS = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
}


@dataclass(frozen=True)
class ModTask:
    p: int = 97
    op: str = "add"

    @property
    def vocab_size(self) -> int:
        return self.p + 2  # Zahlen, Operator-Token, Gleich-Token

    @property
    def seq_len(self) -> int:
        return 4

    def all_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        a, b = np.meshgrid(np.arange(self.p), np.arange(self.p), indexing="ij")
        a, b = a.ravel(), b.ravel()
        x = np.stack([a, np.full_like(a, self.p), b, np.full_like(a, self.p + 1)], axis=1)
        y = OPS[self.op](a, b, self.p)
        return x.astype(np.int32), y.astype(np.int32)


def split(task: ModTask, train_frac: float, seed: int) -> dict[str, np.ndarray]:
    """Fester Zufalls-Split pro Seed (identisch über alle Arme, Entscheidung E11)."""
    x, y = task.all_pairs()
    perm = np.random.default_rng(seed).permutation(len(y))
    n_train = int(round(train_frac * len(y)))
    tr, te = perm[:n_train], perm[n_train:]
    return {"x_train": x[tr], "y_train": y[tr], "x_test": x[te], "y_test": y[te]}
