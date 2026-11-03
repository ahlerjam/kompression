#!/bin/zsh
# Prequential mit Neustart je Block (Blier & Ollivier): mdl (Warmup) und wd, S = 3000; Seeds 100-102
cd ~/ai/kompression
uv run python stages/s0a_grokking/prequential.py --arm mdl --mdl-beta 0.1 --beta-warmup 2000 --restart \
  --steps-per-block 3000 --seeds 100 101 102 --name preq6_mdl_restart_s3000 2>&1 | tee runs/preq6_mdl_restart_s3000.log
uv run python stages/s0a_grokking/prequential.py --arm wd --restart \
  --steps-per-block 3000 --seeds 100 101 102 --name preq6_wd_restart_s3000 2>&1 | tee runs/preq6_wd_restart_s3000.log
