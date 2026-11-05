"""Arme für Stufe 0a. Jeder Arm legt fest:

- `init(params0, seed)` -> (trainable, const). `const["w0"]` ist immer die Initialisierung.
- `effective(trainable, const, noise)` -> Gewichte für den Forward-Pass im Training.
- `deterministic(trainable, const)` -> Gewichte für Evaluation und L(W)-Messung.
- `penalty(trainable, const, noise)` -> Strafterm in nats (Summe, nicht pro Beispiel).

Alle Funktionen arbeiten auf EINEM Seed; `train.py` vmappt darüber.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten

LOG_2PI = math.log(2 * math.pi)


def _is_matrix(path: str, a: mx.array) -> bool:
    return a.ndim == 2


class Arm:
    name = "base"
    needs_noise = False

    def init(self, params0: dict, seed: int):
        return params0, {"w0": params0}

    def effective(self, tr, const, noise):
        return tr

    def deterministic(self, tr, const):
        return tr

    def penalty(self, tr, const, noise):
        return mx.array(0.0)

    def noise_like(self, tr):
        return None


class Plain(Arm):
    """`none` und `wd` (Weight Decay steckt im Optimierer, nicht hier)."""

    def __init__(self, name: str):
        self.name = name


class BitNet(Arm):
    """Ternäre Gewichte (BitNet b1.58) für alle Matrizen, Straight-Through-Estimator.
    1D-Parameter (LayerNorm, Bias) bleiben float. Skala pro Matrix = mean|W|."""

    name = "bitnet"

    @staticmethod
    def ternary(w: mx.array) -> mx.array:
        alpha = mx.abs(w).mean() + 1e-8
        q = mx.clip(mx.round(w / alpha), -1, 1) * alpha
        return w + mx.stop_gradient(q - w)

    def effective(self, tr, const, noise):
        flat = [(k, self.ternary(v) if v.ndim == 2 else v) for k, v in tree_flatten(tr)]
        return tree_unflatten(flat)

    def deterministic(self, tr, const):
        return self.effective(tr, const, None)


class Pruned(Arm):
    """Magnitude-Pruning (Han et al. 2015/2016) für alle Matrizen: Beim `init` werden je Matrix
    die betragskleinsten `frac` der Gewichte auf 0 gesetzt und bleiben dort (feste Maske);
    der Rest wird weitertrainiert. 1D-Parameter bleiben unmaskiert. Gedacht als Nachtraining
    nach einer ersten Lernphase (0a+ B1), nicht als Training von Grund auf."""

    name = "prune"

    def __init__(self, frac: float = 0.9):
        if not 0.0 <= frac < 1.0:
            raise ValueError("frac muss in [0, 1) liegen")
        self.frac = frac

    def _mask(self, w: mx.array) -> mx.array:
        if w.ndim != 2 or self.frac == 0.0:
            return mx.ones_like(w)
        a = mx.abs(w).reshape(-1)
        k = int(self.frac * a.size)
        thr = mx.sort(a)[k]
        return (mx.abs(w) >= thr).astype(w.dtype)

    def init(self, params0, seed):
        mask = tree_map(self._mask, params0)
        tr = tree_map(lambda w, m: w * m, params0, mask)
        return tr, {"w0": params0, "mask": mask}

    def effective(self, tr, const, noise):
        return tree_map(lambda w, m: w * m, tr, const["mask"])

    def deterministic(self, tr, const):
        return self.effective(tr, const, None)


class Subspace(Arm):
    """Zufallsbasis-Unterraum (Idee E27, verwandt mit Li et al. 2018 und VeRA):
    W = W0(seed) + A Z B^T je Matrix; A, B fest aus dem Seed, nur Z (r x r) wird trainiert.
    1D-Parameter bleiben auf ihrer Initialisierung. Trainierbare Zahlen = Summe r_m * r_n."""

    name = "subspace"

    def __init__(self, rank: int):
        self.rank = rank

    def init(self, params0, seed):
        key = mx.random.key(10_000 + seed)
        flat = tree_flatten(params0)
        keys = mx.random.split(key, 2 * len(flat))
        basis, z = {}, {}
        for i, (k, v) in enumerate(flat):
            if v.ndim != 2:
                continue
            m, n = v.shape
            rm, rn = min(self.rank, m), min(self.rank, n)
            basis[k] = (
                mx.random.normal((m, rm), key=keys[2 * i]) / math.sqrt(rm),
                mx.random.normal((n, rn), key=keys[2 * i + 1]) / math.sqrt(rn),
            )
            z[k] = mx.zeros((rm, rn))
        const = {"w0": params0, "A": {k: a for k, (a, _) in basis.items()},
                 "B": {k: b for k, (_, b) in basis.items()}}
        return {"z": z}, const

    def effective(self, tr, const, noise):
        out = []
        for k, v in tree_flatten(const["w0"]):
            if k in tr["z"]:
                v = v + const["A"][k] @ tr["z"][k] @ const["B"][k].T
            out.append((k, v))
        return tree_unflatten(out)

    def deterministic(self, tr, const):
        return self.effective(tr, const, None)


class VariationalMDL(Arm):
    """Variationale Codelänge (Bits-Back) mit lernbarem GMM-Prior (Soft-Weight-Sharing).

    Posterior q(w) = N(mu, sigma^2) je Gewicht; Prior p(w) = sum_k pi_k N(m_k, s_k^2),
    Komponente 0 fest bei m_0 = 0. Strafterm = Monte-Carlo-Schätzung von KL(q || p) in nats.
    Gesamtcodelänge = NLL(Daten) + KL. Mit beta = 1 ist das das echte MDL-Ziel.
    """

    name = "mdl"
    needs_noise = True

    def __init__(self, k: int = 16, init_log_sigma: float = -6.0, min_log_s: float = -9.0,
                 pi0_logit: float = 0.0, s0: float = 0.05):
        self.k = k
        self.init_log_sigma = init_log_sigma
        self.min_log_s = min_log_s
        self.pi0_logit = pi0_logit  # Startgewicht der Null-Komponente (Logit; 0 = wie bisher)
        self.s0 = s0  # Startbreite der Null-Komponente (0,05 = wie bisher)

    def init(self, params0, seed):
        k = self.k
        logit_pi = mx.zeros((k,))
        log_s = mx.full((k,), math.log(0.05))
        if self.pi0_logit or self.s0 != 0.05:
            logit_pi = mx.concatenate([mx.array([self.pi0_logit]), logit_pi[1:]])
            log_s = mx.concatenate([mx.array([math.log(self.s0)]), log_s[1:]])
        prior = {"logit_pi": logit_pi, "mean": mx.linspace(-0.3, 0.3, k - 1), "log_s": log_s}
        tr = {
            "mu": params0,
            "rho": tree_map(lambda a: mx.full(a.shape, self.init_log_sigma), params0),
            "prior": prior,
        }
        return tr, {"w0": params0}

    def noise_like(self, tr):
        return tree_map(lambda a: mx.random.normal(a.shape), tr["mu"])

    def effective(self, tr, const, noise):
        return tree_map(lambda mu, rho, e: mu + mx.exp(rho) * e, tr["mu"], tr["rho"], noise)

    def deterministic(self, tr, const):
        return tr["mu"]

    def _log_prior(self, w: mx.array, prior) -> mx.array:
        means = mx.concatenate([mx.zeros((1,)), prior["mean"]])
        log_s = mx.maximum(prior["log_s"], self.min_log_s)
        log_pi = prior["logit_pi"] - mx.logsumexp(prior["logit_pi"])
        z = (w[..., None] - means) / mx.exp(log_s)
        comp = log_pi - 0.5 * z * z - log_s - 0.5 * LOG_2PI
        return mx.logsumexp(comp, axis=-1)

    def snap(self, tr) -> dict:
        """Soft Weight-Sharing-Abschluss (Ullrich et al. 2017): jedes Matrix-Gewicht auf das Mittel
        der Prior-Komponente mit größter Verantwortlichkeit setzen. 1D-Parameter bleiben."""
        means = mx.concatenate([mx.zeros((1,)), tr["prior"]["mean"]])
        log_s = mx.maximum(tr["prior"]["log_s"], self.min_log_s)
        log_pi = tr["prior"]["logit_pi"] - mx.logsumexp(tr["prior"]["logit_pi"])

        def one(w):
            if w.ndim != 2:
                return w
            z = (w[..., None] - means) / mx.exp(log_s)
            k = mx.argmax(log_pi - 0.5 * z * z - log_s, axis=-1)
            return means[k]

        return tree_map(one, tr["mu"])

    def penalty(self, tr, const, noise):
        total = mx.array(0.0)
        leaves = zip(
            tree_flatten(tr["mu"]), tree_flatten(tr["rho"]), tree_flatten(noise), strict=True
        )
        for (_, mu), (_, rho), (_, e) in leaves:
            w = mu + mx.exp(rho) * e
            log_q = -0.5 * e * e - rho - 0.5 * LOG_2PI
            total = total + (log_q - self._log_prior(w, tr["prior"])).sum()
        return total


def make_arm(name: str, subspace_rank: int = 16, mdl_k: int = 16, prune_frac: float = 0.9,
             mdl_init_log_sigma: float = -6.0, mdl_pi0_logit: float = 0.0,
             mdl_s0: float = 0.05) -> Arm:
    if name == "prune":
        return Pruned(prune_frac)
    if name in ("none", "wd"):
        return Plain(name)
    if name == "bitnet":
        return BitNet()
    if name == "subspace":
        return Subspace(subspace_rank)
    if name == "mdl":
        return VariationalMDL(k=mdl_k, init_log_sigma=mdl_init_log_sigma, pi0_logit=mdl_pi0_logit,
                              s0=mdl_s0)
    raise ValueError(f"unbekannter Arm: {name}")
