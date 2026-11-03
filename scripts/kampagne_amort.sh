#!/bin/zsh
# Kampagnen mit amortisiertem Ziel (Seed 1): Kompression vs. Learnability vs. Zufall
cd ~/ai/kompression
while tmux has-session -t s1-pilot 2>/dev/null; do sleep 60; done
for pol in compression learnability random; do
  uv run python -W ignore stages/s1_pilot/campaign.py --policy $pol --seed 1 \
    --name KA_${pol}_s1 2>&1 | grep -v -iE "warn|fetching" | tee runs/KA_${pol}_s1.log
done
