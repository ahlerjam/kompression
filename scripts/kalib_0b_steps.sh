#!/bin/zsh
# 0b: Trainingsschritte je Block als globaler Hyperparameter (Wahl kostet log2(5) Bit).
# Wartet auf das Ende der laufenden 0b-Warteschlange (tmux s0b).
cd ~/ai/kompression
while tmux has-session -t s0b 2>/dev/null; do sleep 60; done
A=$(ls -d runs/s0a/*_kalib_wd | head -1)
for S in 30 100 300; do
  uv run python stages/s0a_grokking/prequential.py --arm wd --seeds 0 1 2 --op sub --steps-per-block $S \
    --name 0b3_random_s$S 2>&1 | tee runs/0b3_random_s$S.log
  for c in none permuted; do
    uv run python stages/s0a_grokking/prequential.py --arm wd --seeds 0 1 2 --op sub --steps-per-block $S \
      --init-from $A --init-control $c --name 0b3_from_wd_${c}_s$S 2>&1 | tee runs/0b3_from_wd_${c}_s$S.log
  done
done
