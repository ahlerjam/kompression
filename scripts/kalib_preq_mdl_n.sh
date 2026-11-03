#!/bin/zsh
# MDL prequential mit Strafterm-Normierung >= 3764 (Regime der erfolgreichen Kalibrierung)
cd ~/ai/kompression
while tmux has-session -t s0a-preq 2>/dev/null; do sleep 60; done
for S in 3000 1000; do
  uv run python stages/s0a_grokking/prequential.py --arm mdl --mdl-beta 0.1 --penalty-n-min 3764 \
    --steps-per-block $S --seeds 100 101 102 --name preq3_mdl_n3764_s$S 2>&1 | tee runs/preq3_mdl_n3764_s$S.log
done
