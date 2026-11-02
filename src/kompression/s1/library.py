"""Lernende Python-Bibliothek (Ebene 2) und ihre Länge L(B) in Bit.

L(B) = Anzahl Tokens des Bibliotheksquelltexts unter dem festen Tokenizer
des eingefrorenen Modells * log2(Vokabulargröße) (Konzept: „Token-Länge unter
einem festen Tokenizer, umgerechnet in Bit“). PILOT: soll nach Freigabe in den Kern.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field


@dataclass
class Library:
    functions: dict[str, str] = field(default_factory=dict)  # Name -> Quelltext
    version: int = 0

    def source(self) -> str:
        return "\n\n".join(self.functions[k] for k in sorted(self.functions))

    def with_functions(self, new: dict[str, str]) -> "Library":
        merged = dict(self.functions)
        merged.update(new)
        return Library(merged, self.version + 1)

    def without(self, names: set[str]) -> "Library":
        return Library({k: v for k, v in self.functions.items() if k not in names},
                       self.version + 1)

    @staticmethod
    def parse(source: str) -> dict[str, str]:
        """Top-Level-Funktionen aus Quelltext extrahieren (ungültiger Code -> ValueError)."""
        tree = ast.parse(source)
        out = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                out[node.name] = ast.get_source_segment(source, node)
        return out

    def called_in(self, program: str) -> set[str]:
        """Welche Bibliotheksfunktionen ruft ein Programm auf (für ehrliche Wiederverwendung, E14)."""
        try:
            tree = ast.parse(program)
        except SyntaxError:
            return set()
        names = {n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        return names & set(self.functions)


def token_bits(text: str, tokenizer, vocab_size: int) -> float:
    if not text:
        return 0.0
    return len(tokenizer.encode(text)) * math.log2(vocab_size)
