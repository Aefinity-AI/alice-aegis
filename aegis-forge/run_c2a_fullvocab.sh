#!/usr/bin/env bash
set -euo pipefail
REPO="/home/justinbrianthompson/projects/alice-aegis-cm-c2a"
# Wait for the pruned-vocab n=119 run (c2a-mc.service) to finish before
# starting the much larger full-vocab n=570 run — keeps peak RAM to one
# model resident at a time on this shared dev box.
while systemctl --user is-active --quiet c2a-mc.service; do
  sleep 60
done
exec "$REPO/aegis-eval/target/release/aegis-eval" \
  "$REPO/aegis-forge/fullvocab_out/MODEL.SAF" \
  "$REPO/aegis-forge/fullvocab_out/EMBED.BIN" \
  "$REPO/aegis-forge/fullvocab_out/VOCAB.BIN" \
  --mc "$REPO/model-lab/data/evals/arc_easy/arc_easy_val_n570_seed42_bitnet_fullvocab.jsonl" \
  --mc-out "$REPO/model-lab/data/evals/arc_easy/results/bitnet2b_float_n570_fullvocab.jsonl" \
  --mc-cis-full \
  --mc-cis-full-out "$REPO/model-lab/data/evals/arc_easy/results/bitnet2b_fullint_n570_fullvocab.jsonl"
