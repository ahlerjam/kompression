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
H=(20 21 22 23 24 25 26 27 28 29)
case "$1" in
  CAL1)
    P --seeds $SE --post wd --name kalZ_wd
    P --seeds $SE --post mdl --post-mdl-beta 0.1 --name kalZ_mdl_b0.1
    P --seeds $SE --post free --name kalZ_free
    P --seeds $SE --post prune --prune-frac 0.9 --name kalZ_prune_f0.9 ;;
  CAL2)
    P --seeds $SE --post mdl --post-mdl-beta 0.3 --name kalZ_mdl_b0.3
    P --seeds $SE --post wd --post-weight-decay 10 --name kalZ_wd_l10
    P --seeds $SE --post mdl_norm --post-mdl-beta 0.3 --name kalZ_mdlnorm_b0.3
    P --seeds $SE --post bitnet --name kalZ_bitnet ;;
  # Hauptlauf (Protokoll Version 2, eingefroren): β = 0,1 aus Kalibrierung; wichtigste Bedingungen zuerst
  1)
    P --seeds $H --post mdl --post-mdl-beta 0.1 --name B1_mdl
    P --seeds $H --post wd --name B1_wd
    P --seeds $H --post prune --prune-frac 0.9 --name B1_prune
    P --seeds $H --post free --name B1_free ;;
  2)
    P --seeds $H --post wd --post-weight-decay 10 --name B1_wd_l10
    P --seeds $H --post mdl_norm --post-mdl-beta 0.1 --name B1_mdlnorm
    P --seeds $H --post bitnet --name B1_bitnet ;;
  # Testdurchlauf der Hauptlauf-Befehle auf kleinem p (kein Befund)
  T)
    for a in "mdl --post-mdl-beta 0.1" "wd" "wd --post-weight-decay 10" "mdl_norm --post-mdl-beta 0.1" \
             "bitnet" "prune --prune-frac 0.9" "free"; do
      P --p 23 --first-block 64 --seeds 200 201 --post ${=a} --name testB1_${a%% *}
    done ;;
  *) echo "unbekannt: $1"; exit 1 ;;
esac
