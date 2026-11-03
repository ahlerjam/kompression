#!/bin/zsh
# Kampagnen mit amortisiertem Ziel (Seed 1): Kompression, Learnability, Zufall; danach Learnability Seed 0 (alte Regel) nachholen
cd ~/ai/kompression
for pol in compression learnability random; do
  uv run python -W ignore stages/s1_pilot/campaign.py --policy $pol --seed 1 \
    --name KA_${pol}_s1 2>&1 | grep -v -iE "warn|fetching" | tee runs/KA_${pol}_s1.log
done
uv run python -W ignore stages/s1_pilot/campaign.py --policy learnability --seed 0 --no-amortize \
  --name K_learnability_s0 2>&1 | grep -v -iE "warn|fetching" | tee runs/K_learnability_s0.log
