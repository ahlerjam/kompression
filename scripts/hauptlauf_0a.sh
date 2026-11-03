#!/bin/zsh
# HAUPTLAUF Stufe 0a nach docs/protokoll_0a.md (Version 2, eingefroren 2026-11-03).
# Aufruf: scripts/hauptlauf_0a.sh A   (Trainingsläufe)   |   scripts/hauptlauf_0a.sh B   (prequential)
set -e
cd ~/ai/kompression
S=(10 11 12 13 14 15 16 17 18 19)
if [ "$1" = "A" ]; then
  R="uv run python stages/s0a_grokking/run.py --seeds $S --steps 40000"
  eval $R --arm mdl --mdl-beta 0.1 --beta-warmup 2000 --name H_mdl
  MDL=$(ls -d runs/s0a/*_H_mdl | tail -1)
  eval $R --arm wd --name H_wd
  eval $R --arm wd --norm-target-run $MDL --name H_wd_normproj
  eval $R --arm bitnet --name H_bitnet
  eval $R --arm none --name H_none
elif [ "$1" = "B" ]; then
  P="uv run python stages/s0a_grokking/prequential.py --seeds $S --restart --steps-per-block 3000"
  for b in 0.03 0.1 0.3; do eval $P --arm mdl --mdl-beta $b --beta-warmup 2000 --name Hpreq_mdl_b$b; done
  for l in 0.3 1.0 3.0; do eval $P --arm wd --weight-decay $l --name Hpreq_wd_l$l; done
  eval $P --arm bitnet --name Hpreq_bitnet
  eval $P --arm none --name Hpreq_none
else
  echo "Aufruf: $0 A|B"; exit 1
fi
