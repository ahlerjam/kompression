#!/bin/zsh
# Wartet, bis für mindestens einen der Namensteile ein results.json existiert.
# Aufruf: scripts/wait_results.sh kalib_mdl_b0.01 kalib_subspace_r64
cd ~/ai/kompression
while true; do
  for n in "$@"; do
    for f in runs/s0a/*_"$n"/results.json(N); do echo "fertig: $f"; exit 0; done
  done
  sleep 30
done
