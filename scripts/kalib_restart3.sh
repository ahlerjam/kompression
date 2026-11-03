#!/bin/zsh
# Neustart je Block, S = 3000, Seeds 100-102: mdl beta 0.3 (stärkerer Druck), none (Untergrenze)
cd ~/ai/kompression
uv run python stages/s0a_grokking/prequential.py --arm mdl --mdl-beta 0.3 --beta-warmup 2000 --restart \
  --steps-per-block 3000 --seeds 100 101 102 --name preq6_mdl_b0.3_restart_s3000 2>&1 | tee runs/preq6_mdl_b0.3_restart_s3000.log
uv run python stages/s0a_grokking/prequential.py --arm none --restart \
  --steps-per-block 3000 --seeds 100 101 102 --name preq6_none_restart_s3000 2>&1 | tee runs/preq6_none_restart_s3000.log
