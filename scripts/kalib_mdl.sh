#!/bin/zsh
# Kalibrierung mdl über beta (Seeds 100-102, nicht die Hauptlauf-Seeds 0-9)
cd ~/ai/kompression
for b in 0.001 0.01 0.1 1.0; do
  uv run python stages/s0a_grokking/run.py --arm mdl --mdl-beta $b --beta-warmup 2000 \
    --seeds 100 101 102 --steps 20000 --name kalib_mdl_b$b 2>&1 | tee runs/kalib_mdl_b$b.log
done
