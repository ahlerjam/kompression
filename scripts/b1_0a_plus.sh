#!/bin/zsh
# 0a+ B1-Zyklus (Wach-Schlaf) nach docs/protokoll_0a_plus.md.
# Je Block (Neustart): 4 Zyklen aus Wach (wd λ = 1, 750 Schritte) und Schlaf (Verfahren --post, 250 Schritte);
# die nächste Wachphase lernt vom komprimierten Stand weiter. Kontrolle: Schlaf = wd (nur lernen, gleiche Schritte).
#   scripts/b1_0a_plus.sh CAL1 | CAL2   Kalibrierung, Seeds 100–102 (zwei Stränge)
#   scripts/b1_0a_plus.sh 1 | 2         Hauptlauf, Seeds 20–29 (erst nach Einfrieren des Protokolls)
set -e
cd ~/ai/kompression
P() { uv run python stages/s0a_grokking/prequential.py --arm wd --weight-decay 1.0 --restart \
  --steps-per-block 750 --cycles 4 --post-steps 250 --post-beta-warmup 50 "$@" }
SE=(100 101 102)
case "$1" in
  CAL1)
    P --seeds $SE --post wd --name kalZ_wd
    P --seeds $SE --post mdl --post-mdl-beta 0.03 --name kalZ_mdl_b0.03
    P --seeds $SE --post prune --prune-frac 0.9 --name kalZ_prune_f0.9 ;;
  CAL2)
    P --seeds $SE --post bitnet --name kalZ_bitnet
    P --seeds $SE --post mdl --post-mdl-beta 0.1 --name kalZ_mdl_b0.1
    P --seeds $SE --post prune --prune-frac 0.5 --name kalZ_prune_f0.5 ;;
  *) echo "Hauptlauf erst nach Einfrieren; Befehle kommen dann hierher"; exit 1 ;;
esac
