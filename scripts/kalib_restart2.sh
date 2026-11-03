#!/bin/zsh
# Neustart je Block, S = 3000, Seeds 100-102: bitnet, mdl beta 0.03, mdl beta 0.01 (Vorbereitung Option B)
cd ~/ai/kompression
uv run python stages/s0a_grokking/prequential.py --arm bitnet --restart --steps-per-block 3000 \
  --seeds 100 101 102 --name preq6_bitnet_restart_s3000 2>&1 | tee runs/preq6_bitnet_restart_s3000.log
for b in 0.03 0.01; do
  uv run python stages/s0a_grokking/prequential.py --arm mdl --mdl-beta $b --beta-warmup 2000 --restart \
    --steps-per-block 3000 --seeds 100 101 102 --name preq6_mdl_b${b}_restart_s3000 2>&1 | tee runs/preq6_mdl_b${b}_restart_s3000.log
done
