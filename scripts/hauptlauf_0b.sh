#!/bin/zsh
# Stufe 0b (Transfer in Bit) nach docs/protokoll_0b.md.
#   scripts/hauptlauf_0b.sh CAL      Kalibrierung mit Neustart je Block (Seeds 0–2, sub, S 30/100/300)
#   scripts/hauptlauf_0b.sh 1 | 2    Hauptlauf, Strang 1 bzw. 2 (Seeds 10–19); erst sub (Gate), dann mul
# NEUSTART=1 setzt --restart (Entscheidung nach der Kalibrierung, im Protokoll v2 festgehalten).
set -e
cd ~/ai/kompression
R=""; [ "$NEUSTART" = "1" ] && R="--restart"
P="uv run python stages/s0a_grokking/prequential.py --arm wd $R"
if [ "$1" = "CAL" ]; then
  A=$(ls -d runs/s0a/*_kalib_wd | head -1)
  for S in 30 100 300; do
    eval $P --seeds 0 1 2 --op sub --restart --steps-per-block $S --name 0b4_random_s$S
    for c in none permuted; do
      eval $P --seeds 0 1 2 --op sub --restart --steps-per-block $S --init-from $A --init-control $c \
        --name 0b4_from_wd_${c}_s$S
    done
  done
  exit 0
fi
S10=(10 11 12 13 14 15 16 17 18 19)
src() { ls -d runs/s0a/*_H_$1 | tail -1 }
cond() {  # cond <op> <name> [init-args...]
  local op=$1 name=$2; shift 2
  for S in 30 100 300 1000 3000; do
    eval $P --seeds $S10 --op $op --steps-per-block $S "$@" --name H0b_${op}_${name}_s$S
  done
}
for op in sub mul; do
  if [ "$1" = "1" ]; then
    cond $op random
    cond $op wd --init-from $(src wd) --init-control none
    cond $op wd_permuted --init-from $(src wd) --init-control permuted
  elif [ "$1" = "2" ]; then
    cond $op wd_scaled --init-from $(src wd) --init-control scaled
    cond $op mdl --init-from $(src mdl) --init-control none
    cond $op bitnet --init-from $(src bitnet) --init-control none
  else
    echo "Aufruf: $0 CAL|1|2"; exit 1
  fi
done
