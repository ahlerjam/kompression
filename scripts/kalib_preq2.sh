#!/bin/zsh
# Rest der 0a-Prequential-Kalibrierung: mdl (beta 0.1, Kalibrierung) und bitnet
cd ~/ai/kompression
uv run python stages/s0a_grokking/prequential.py --arm mdl --mdl-beta 0.1 --seeds 100 101 102 \
  --name preq_mdl 2>&1 | tee runs/preq_mdl.log
uv run python stages/s0a_grokking/prequential.py --arm bitnet --seeds 100 101 102 \
  --name preq_bitnet 2>&1 | tee runs/preq_bitnet.log
