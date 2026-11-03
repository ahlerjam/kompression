"""Eingefrorenes Sprachmodell (mlx-lm): Generierung und Logprob-Bits.

L(y | x, B) = -log2 p(y | prompt(x, B)) über die Tokens von y plus Ende-Token.
"""

from __future__ import annotations

import math

import mlx.core as mx

LN2 = math.log(2.0)


class FrozenLM:
    def __init__(self, repo: str):
        from mlx_lm import load

        self.repo = repo
        self.model, self.tokenizer = load(repo)
        self.vocab_size = self._vocab_size()
        self.end_id = self.tokenizer.eos_token_id

    def _vocab_size(self) -> int:
        args = getattr(self.model, "args", None)
        for obj in (args, getattr(args, "text_config", None)):
            if obj is None:
                continue
            v = obj.get("vocab_size") if isinstance(obj, dict) else getattr(obj, "vocab_size", None)
            if v:
                return int(v)
        return len(self.tokenizer.vocab)

    def chat(self, system: str, user: str) -> list[int]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

    max_batch: int = 8  # begrenzt den GPU-Speicher (35B + parallele Läufe: OOM bei großen Batches)

    def generate(self, prompts: list[list[int]], max_tokens: int = 512,
                 temperature: float = 0.7) -> list[str]:
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler

        out: list[str] = []
        for i in range(0, len(prompts), self.max_batch):
            r = batch_generate(self.model, self.tokenizer, prompts[i : i + self.max_batch],
                               max_tokens=max_tokens, sampler=make_sampler(temp=temperature))
            out.extend(r.texts)
            mx.clear_cache()
        return out

    def completion_bits(self, prompt: list[int], completion: str) -> float:
        ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if self.end_id is not None:
            ids = [*ids, self.end_id]
        full = mx.array([*prompt, *ids])[None]
        logits = self.model(full)[0, len(prompt) - 1 : -1].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picked = mx.take_along_axis(logp, mx.array(ids)[:, None], axis=-1)
        return float(-picked.sum()) / LN2
