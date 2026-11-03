"""Proposer als Auswahlpolitik über Aufgabenfamilien (Entscheidung E12).

Der Proposer ist im Piloten eingefroren; die Belohnung wirkt über die AUSWAHL
(Bandit), nicht über Gradienten. Beide Belohnungen nutzen exakt denselben Mechanismus.

Belohnungen:
- `compression`: Δ L_gesamt auf dem versiegelten Probe-Set nach einer Schlafphase,
  mit Vorzeichen und ohne Clipping (E13), anteilig auf die Familien verteilt, deren
  Programme in diese Schlafphase eingingen.
- `learnability`: 1 − r̄ für 0 < r̄ < 1, sonst 0 (Absolute Zero), sofort nach der Wachrunde.
- `random`: gleichverteilte Auswahl (Kontrolle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SoftmaxBandit:
    arms: tuple[str, ...]
    temperature: float = 0.5
    prior: float = 0.0
    seed: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        for a in self.arms:
            self.counts.setdefault(a, 0)
            self.values.setdefault(a, self.prior)

    def select(self, k: int = 1) -> list[str]:
        # Unbesuchte Arme zuerst (jede Familie mindestens einmal), dann Softmax über Mittelwerte.
        unseen = [a for a in self.arms if self.counts[a] == 0]
        if unseen:
            return list(self.rng.choice(unseen, size=min(k, len(unseen)), replace=False))
        v = np.array([self.values[a] for a in self.arms], dtype=np.float64)
        scale = np.std(v) if np.std(v) > 0 else 1.0
        p = np.exp((v - v.max()) / (self.temperature * scale))
        p /= p.sum()
        return list(self.rng.choice(self.arms, size=k, replace=True, p=p))

    def update(self, arm: str, reward: float) -> None:
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


@dataclass
class RandomPolicy:
    arms: tuple[str, ...]
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def select(self, k: int = 1) -> list[str]:
        return list(self.rng.choice(self.arms, size=k, replace=True))

    def update(self, arm: str, reward: float) -> None:
        pass


def learnability_reward(pass_rate: float) -> float:
    return 1.0 - pass_rate if 0.0 < pass_rate < 1.0 else 0.0


def compression_credit(delta_bits: float, family_counts: dict[str, int]) -> dict[str, float]:
    """Verteilt die Verbesserung (positive Zahl = L_gesamt gesunken) anteilig auf Familien."""
    total = sum(family_counts.values())
    if total == 0 or math.isnan(delta_bits):
        return {}
    return {f: delta_bits * c / total for f, c in family_counts.items()}
