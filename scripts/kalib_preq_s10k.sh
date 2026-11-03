#!/bin/zsh
# Prequential mit S = 10000 (langsame Lerner): mdl (Warmup, Normierung) und wd; Seeds 100-102
cd ~/ai/kompression
while tmux has-session -t '=s0a-preq-mdl' 2>/dev/null; do sleep 60; done
uv run python stages/s0a_grokking/prequential.py --arm mdl --mdl-beta 0.1 --penalty-n-min 3764 \
  --beta-warmup 2000 --steps-per-block 10000 --seeds 100 101 102 --name preq5_mdl_wu_s10000 2>&1 | tee runs/preq5_mdl_wu_s10000.log
uv run python stages/s0a_grokking/prequential.py --arm wd --steps-per-block 10000 --seeds 100 101 102 \
  --name preq5_wd_s10000 2>&1 | tee runs/preq5_wd_s10000.log
