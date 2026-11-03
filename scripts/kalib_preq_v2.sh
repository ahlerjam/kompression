#!/bin/zsh
# Prequential v2 (Self-Switch über Trainingsstände) für 0a-Arme, Seeds 100-102
cd ~/ai/kompression
for arm in wd mdl bitnet none; do
  uv run python stages/s0a_grokking/prequential.py --arm $arm --seeds 100 101 102 \
    --name preq2_$arm 2>&1 | tee runs/preq2_$arm.log
done
