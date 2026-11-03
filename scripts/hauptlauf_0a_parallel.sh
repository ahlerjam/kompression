#!/bin/zsh
# Hauptlauf 0a, Teil B auf weitere GPU-Stränge verteilt (2026-11-03, Wunsch Jonas: alle GPU an das Gate).
# Gleiche Befehle und Parameter wie scripts/hauptlauf_0a.sh B; nur die Reihenfolge/Parallelität ändert sich.
#   C: wd-Gitter   D: bitnet, none
#   STOPB: wartet, bis Hpreq_mdl_b0.3 fertig ist, beendet dann die alte Warteschlange H-B,
#          damit sie wd/bitnet/none nicht doppelt rechnet; angefangene Doppel werden umbenannt.
set -e
cd ~/ai/kompression
S=(10 11 12 13 14 15 16 17 18 19)
P="uv run python stages/s0a_grokking/prequential.py --seeds $S --restart --steps-per-block 3000"
if [ "$1" = "C" ]; then
  for l in 0.3 1.0 3.0; do eval $P --arm wd --weight-decay $l --name Hpreq_wd_l$l; done
elif [ "$1" = "D" ]; then
  eval $P --arm bitnet --name Hpreq_bitnet
  eval $P --arm none --name Hpreq_none
elif [ "$1" = "STOPB" ]; then
  until ls runs/s0a_preq/*_Hpreq_mdl_b0.3/results.json >/dev/null 2>&1; do sleep 30; done
  sleep 5
  tmux kill-session -t '=H-B' 2>/dev/null || true
  sleep 5
  for n in Hpreq_wd_l0.3 Hpreq_wd_l1.0 Hpreq_wd_l3.0 Hpreq_bitnet Hpreq_none; do
    d=$(ls -d runs/s0a_preq/*_$n 2>/dev/null | tail -1)
    if [ -n "$d" ] && [ "$(ls -d runs/s0a_preq/*_$n | wc -l)" -gt 1 ] && [ ! -f "$d/results.json" ]; then
      mv "$d" "$(dirname $d)/abgebrochen_doppel_$(basename $d)"
    fi
  done
  echo "H-B beendet $(date)"
else
  echo "Aufruf: $0 C|D|STOPB"; exit 1
fi
