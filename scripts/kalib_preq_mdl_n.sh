#!/bin/zsh
# MDL prequential mit Warmup (2000 Schritte) und Strafterm-Normierung >= 3764; S = 3000, 1000, 300
cd ~/ai/kompression
for S in 3000 1000 300; do
  uv run python stages/s0a_grokking/prequential.py --arm mdl --mdl-beta 0.1 --penalty-n-min 3764 \
    --beta-warmup 2000 --steps-per-block $S --seeds 100 101 102 --name preq4_mdl_wu_s$S 2>&1 | tee runs/preq4_mdl_wu_s$S.log
done
