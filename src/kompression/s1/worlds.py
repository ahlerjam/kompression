"""Prüfbare Welten aus Reasoning Gym.

Trainings-Familien sieht das System; zurückgehaltene Familien nur der Prüfstand
(Transfer). Aufgaben werden immer frisch mit Seeds erzeugt, nie aus festen
Benchmarks (keine Kontamination des eingefrorenen Modells).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

TRAIN_FAMILIES: tuple[str, ...] = (
    "gcd", "lcm", "base_conversion", "prime_factorization", "count_primes",
    "number_sorting", "letter_counting", "word_sorting", "basic_arithmetic",
    "chain_sum", "bitwise_arithmetic", "power_function", "fraction_simplification",
    "string_manipulation", "spell_backward", "calendar_arithmetic",
)
HELDOUT_FAMILIES: tuple[str, ...] = (
    "count_bits", "number_filtering", "decimal_arithmetic", "isomorphic_strings",
    "ransom_note", "palindrome_generation", "word_sequence_reversal", "time_intervals",
)


@dataclass
class Task:
    family: str
    seed: int
    index: int
    question: str
    answer: str
    entry: dict = field(repr=False)


class World:
    def __init__(self):
        import reasoning_gym as rg

        self._rg = rg
        self._cache: dict[tuple[str, int, int], object] = {}

    def _dataset(self, family: str, seed: int, size: int):
        key = (family, seed, size)
        if key not in self._cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._cache[key] = self._rg.create_dataset(family, size=size, seed=seed)
        return self._cache[key]

    def generate(self, family: str, seed: int, n: int) -> list[Task]:
        ds = self._dataset(family, seed, n)
        return [
            Task(family, seed, i, ds[i]["question"], str(ds[i]["answer"]), ds[i])
            for i in range(n)
        ]

    def verify(self, task: Task, answer: str | None) -> float:
        if answer is None:
            return 0.0
        ds = self._dataset(task.family, task.seed, max(task.index + 1, 1))
        try:
            return float(ds.score_answer(answer.strip(), task.entry))
        except Exception:  # Bewertungsfehler zählen als falsch
            return 0.0
