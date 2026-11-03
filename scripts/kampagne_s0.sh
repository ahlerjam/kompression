#!/bin/zsh
# Erste Stufe-1-Kampagne (Seed 0): Kompression vs. Learnability, gleiches Budget
cd ~/ai/kompression
for pol in compression learnability; do
  uv run python -W ignore stages/s1_pilot/campaign.py --policy $pol --seed 0 \
    --name K_${pol}_s0 2>&1 | grep -v -iE "warn|fetching" | tee runs/K_${pol}_s0.log
done
