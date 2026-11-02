#!/bin/zsh
# Prequentielle Codelänge der ganzen Tabelle je Arm (Seeds 100-102), Kalibrierung
cd ~/ai/kompression
for arm in wd none subspace mdl bitnet; do
  uv run python stages/s0a_grokking/prequential.py --arm $arm --seeds 100 101 102 \
    --name preq_$arm 2>&1 | tee runs/preq_$arm.log
done
