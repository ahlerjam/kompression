#!/bin/zsh
# 0b mit Prequential v2: Transfer add -> sub, Lerner wd, Seeds 0-2
cd ~/ai/kompression
A=$(ls -d runs/s0a/*_kalib_wd | head -1)
uv run python stages/s0a_grokking/prequential.py --arm wd --seeds 0 1 2 --op sub \
  --name 0b2_random 2>&1 | tee runs/0b2_random.log
for c in none scaled permuted; do
  uv run python stages/s0a_grokking/prequential.py --arm wd --seeds 0 1 2 --op sub \
    --init-from $A --init-control $c --name 0b2_from_wd_$c 2>&1 | tee runs/0b2_from_wd_$c.log
done
