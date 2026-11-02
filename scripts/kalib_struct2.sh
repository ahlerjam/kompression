#!/bin/zsh
# Rang 4 übersprungen (Rang 16 lernte nicht, siehe Logbuch). Rang 64, dann bitnet.
cd ~/ai/kompression
uv run python stages/s0a_grokking/run.py --arm subspace --subspace-rank 64 --seeds 100 101 102 \
  --steps 20000 --name kalib_subspace_r64 2>&1 | tee runs/kalib_subspace_r64.log
uv run python stages/s0a_grokking/run.py --arm bitnet --seeds 100 101 102 --steps 20000 \
  --name kalib_bitnet 2>&1 | tee runs/kalib_bitnet.log
