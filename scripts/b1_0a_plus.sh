#!/bin/zsh
# 0a+ B1 „erst lernen, dann komprimieren“ nach docs/protokoll_0a_plus.md.
# Phase 1: wd λ = 1, Neustart je Block, 3.000 Schritte; Phase 2 (Zweig): 1.000 Schritte Nachtraining.
#   scripts/b1_0a_plus.sh CAL1 | CAL2   Kalibrierung, Seeds 100–102 (zwei Stränge)
#   scripts/b1_0a_plus.sh 1 | 2         Hauptlauf, Seeds 20–29 (erst nach Einfrieren des Protokolls)
set -e
cd ~/ai/kompression
P="uv run python stages/s0a_grokking/prequential.py --arm wd --weight-decay 1.0 --restart --steps-per-block 3000 --post-steps 1000"
case "$1" in
  CAL1)
    SE="100 101 102"
    eval $P --seeds $SE --post wd --name kalB1_wd
    eval $P --seeds $SE --post mdl --post-mdl-beta 0.03 --post-beta-warmup 300 --name kalB1_mdl_b0.03
    eval $P --seeds $SE --post prune --prune-frac 0.9 --name kalB1_prune_f0.9 ;;
  CAL2)
    SE="100 101 102"
    eval $P --seeds $SE --post bitnet --name kalB1_bitnet
    eval $P --seeds $SE --post mdl --post-mdl-beta 0.1 --post-beta-warmup 300 --name kalB1_mdl_b0.1
    eval $P --seeds $SE --post prune --prune-frac 0.5 --name kalB1_prune_f0.5 ;;
  *) echo "Hauptlauf erst nach Einfrieren; Befehle stehen dann im Protokoll"; exit 1 ;;
esac
