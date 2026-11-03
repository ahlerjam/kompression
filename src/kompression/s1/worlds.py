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


# Schweres Profil für das 35B-Modell: Aufgaben, die echte Algorithmen (Suche, Backtracking,
# Simulation) verlangen. Name in Task.family bleibt die Reasoning-Gym-Familie.
HARD_CONFIGS: dict[str, dict] = {
    "maze": {"min_dist": 15, "max_dist": 30, "min_grid_size": 12, "max_grid_size": 20},
    "shortest_path": {"min_rows": 10, "max_rows": 16, "min_cols": 10, "max_cols": 16},
    "largest_island": {"min_rows": 15, "max_rows": 25, "min_cols": 15, "max_cols": 25,
                       "max_num_islands": 12, "max_island_size": 30},
    "graph_color": {"min_num_vertices": 20, "max_num_vertices": 30, "edge_probability": 0.15},
    "course_schedule": {"min_num_courses": 15, "max_num_courses": 25,
                        "min_num_prerequisites": 2, "max_num_prerequisites": 4},
    "sudoku": {"min_empty": 40, "max_empty": 55},
    "n_queens": {"n": 8, "min_remove": 5, "max_remove": 7},
    "spiral_matrix": {"min_n": 12, "max_n": 20},
    "game_of_life": {"grid_size_x": 15, "grid_size_y": 15, "filled_cells": 30,
                     "simulation_steps": 3},
    "knights_knaves": {"n_people": 4, "depth_constraint": 3, "width_constraint": 3},
}


@dataclass
class Task:
    family: str
    seed: int
    index: int
    question: str
    answer: str
    entry: dict = field(repr=False)


class World:
    def __init__(self, configs: dict[str, dict] | None = None):
        import reasoning_gym as rg

        self._rg = rg
        self.configs = configs or {}
        self._cache: dict[tuple[str, int, int], object] = {}

    def _dataset(self, family: str, seed: int, size: int):
        key = (family, seed, size)
        if key not in self._cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._cache[key] = self._rg.create_dataset(
                    family, size=size, seed=seed, **self.configs.get(family, {}))
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
