#!/bin/zsh
# Stufe 0b Kalibrierung: Transfer add -> sub per prequentieller Codelänge.
# Wartet, bis die 0a-Prequential-Warteschlange (tmux s0a-preq) fertig ist.
cd ~/ai/kompression
while tmux has-session -t s0a-preq 2>/dev/null; do sleep 60; done
A=$(ls -d runs/s0a/*_kalib_wd | head -1)
uv run python stages/s0a_grokking/prequential.py --arm wd --seeds 0 1 2 --op sub \
  --name 0b_random 2>&1 | tee runs/0b_random.log
for c in none scaled permuted; do
  uv run python stages/s0a_grokking/prequential.py --arm wd --seeds 0 1 2 --op sub \
    --init-from $A --init-control $c --name 0b_from_wd_$c 2>&1 | tee runs/0b_from_wd_$c.log
done
