#!/bin/zsh
# Kalibrierung subspace (Ränge 4/16/64) und bitnet, Seeds 100-102
cd ~/ai/kompression
for r in 16 4 64; do
  uv run python stages/s0a_grokking/run.py --arm subspace --subspace-rank $r --seeds 100 101 102 \
    --steps 20000 --name kalib_subspace_r$r 2>&1 | tee runs/kalib_subspace_r$r.log
done
uv run python stages/s0a_grokking/run.py --arm bitnet --seeds 100 101 102 --steps 20000 \
  --name kalib_bitnet 2>&1 | tee runs/kalib_bitnet.log
