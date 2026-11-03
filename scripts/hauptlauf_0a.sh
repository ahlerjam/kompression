#!/bin/zsh
# HAUPTLAUF Stufe 0a nach docs/protokoll_0a.md (Version 1). NUR nach Freigabe durch Jonas starten.
# Startet die Messphase: vorher STATUS.md auf MESSPHASE setzen, sauberer Git-Stand Pflicht.
set -e
cd ~/ai/kompression
[ -z "$(git status --porcelain)" ] || { echo "Git-Stand nicht sauber"; exit 1; }
S=(10 11 12 13 14 15 16 17 18 19)
R="uv run python stages/s0a_grokking/run.py --seeds $S --steps 40000"
eval $R --arm mdl --mdl-beta 0.1 --beta-warmup 2000 --name H_mdl
MDL=$(ls -d runs/s0a/*_H_mdl | tail -1)
eval $R --arm wd --name H_wd
eval $R --arm wd --norm-target-run $MDL --name H_wd_normproj
eval $R --arm bitnet --name H_bitnet
eval $R --arm none --name H_none
P="uv run python stages/s0a_grokking/prequential.py --seeds $S"
for arm in mdl wd bitnet none; do
  for S in 100 300 1000 3000; do eval $P --arm $arm --steps-per-block $S --name Hpreq_${arm}_s$S; done
done
uv run python stages/s0a_grokking/analyse.py > runs/H_0a_auswertung.txt
