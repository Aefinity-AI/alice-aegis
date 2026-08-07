#!/usr/bin/env bash
# seed_retractions.sh — put the already-known-dead numbers into claims.jsonl so
# that from this moment on they cannot re-enter a document silently.
#
# Every entry below is a number this program has ALREADY retracted, superseded,
# or found unlogged, per the 2026-07-29 forensic sweep and program/RESEARCH_LEDGER.md.
# Nothing here is a new judgment. Run it once.
set -uo pipefail
cd "$(dirname "$0")"
EV="./ev"

add_dead () { # id value unit reason [superseded_by]
  ./tools/ledger.py add --id "$1" --value "$2" --unit "$3" --kind unlogged \
     --statement "$4" --scope "see reason; entered only so the value is recognisable" \
     --force >/dev/null
  ./tools/ledger.py retract --id "$1" --reason "$4" \
     ${5:+--superseded-by "$5"} --value "$2"
}

echo "--- multicore: the 254ba43 unlogged 'clean re-measure' (commit e659a1f retracted it) ---"
add_dead A4.8t   8.25  tok/s "no log anywhere; commit e659a1f says so verbatim. CONTRADICTED by the only record (154f00a: 8t=5.40, slower than 4t=5.61)" A4.sweep2026
add_dead A4.1t   3.64  tok/s "same unlogged 254ba43 run as 8.25; if 8.25 is unlogged so is 3.64" A4.sweep2026
add_dead A4.2t   7.91  tok/s "same unlogged 254ba43 sweep (4-thread arm)" A4.sweep2026
add_dead A4.smt  2.27  x     "2.27x decode speedup derived from the unlogged 254ba43 sweep" A4.sweep2026

echo "--- energy ---"
add_dead A8.old  3.31  J/tok "raw readings never logged; superseded by the 2026-07-14 run (3.99 J incremental)" A8

echo "--- perplexity, pre-tokenizer-fix ---"
add_dead A2.pre  14.488 PPL  "predates the byte-level tokenizer fix; absolute superseded"
add_dead A2.post 12.801 PPL  "superseded AND unreproducible: its dataset file was deleted" B9
add_dead A5.int8 12.738 PPL  "one 1,842-token window, one run, no error bars; the source record says the sign could reverse" B9

echo "--- kernel table: pre-cleanup GMAC/s (ffcdc4c), disowned by 254ba43 itself ---"
add_dead A6.gmac1 9.86  GMAC/s "pre-cleanup measurement taken while a runaway ugrep held a core"
add_dead A6.gmac2 17.13 GMAC/s "pre-cleanup measurement taken while a runaway ugrep held a core"

echo "--- sparsity: six-tensor sample superseded by the full 210-tensor scan ---"
add_dead A6.zeros 40.8 "%"    "six-tensor sample; full scan of all 210 ternary tensors gives 42.21% (f1f8164)" A6.zeros.full

echo "--- bandwidth: no benchmark exists in this repository ---"
add_dead A13.bw  17.3  GB/s  "NO SOURCE FOUND: there is no bandwidth benchmark in this repo; ops.rs:1240 quotes the figure rather than producing it"

echo
echo "--- the live replacements that DO have primary sources ---"
./tools/ledger.py add --id A4.sweep2026 --value 2.14 --unit x --kind commit-only \
  --statement "decode 2.62 -> 5.61 tok/s at 4 threads (the only recorded sweep)" \
  --scope "i5-10210U crosvm guest, BitNet-2B, userspace; PROSE IN commit 154f00a, no log file" \
  --ceiling "commit-message-only: no log file exists. Banned from external documents until thread_sweep emits one." \
  --force >/dev/null
./tools/ledger.py add --id A6.zeros.full --value 42.21 --unit "%" --kind measured \
  --statement "zero fraction of the real BitNet b1.58-2B weights, full scan, 210 tensors, 2.084 G weights" \
  --scope "aegis_pruned_model.safetensors, all ternary tensors" \
  --source aegis-core/benches/ctz_vs_simd.rs \
  --ceiling "static property of the artifact; no run-to-run variance" >/dev/null
./tools/ledger.py add --id A8 --value 3.99 --unit J/tok --kind measured \
  --statement "incremental energy per token, battery-discharge method, parallel+int8_act" \
  --scope "i5-10210U on battery, 385 s decode window, 1901 tokens, 4.94 tok/s" \
  --source docs/hardware_logs/energy_run_i5-10210U_multicore_2026-07-14.log \
  --ceiling "one run; the log's own coulomb cross-check disagrees by 10.8% at idle and 19.3% at load" >/dev/null
./tools/ledger.py add --id B9 --value 16.124 --unit PPL --kind measured \
  --statement "WikiText-2 full test set, post-5989c32 tokenizer, chunked cold-KV" \
  --scope "50,256-token ASCII-pruned vocab, 312,119 scored predictions, ~110 ks run" \
  --source docs/hardware_logs/wikitext2_full_ppl_2026-07-17_newtokenizer_run.log \
  --ceiling "chunked cold-KV biases upward; not comparable across tokenizers" >/dev/null
./tools/ledger.py add --id A3.baremetal --value 726238201 --unit ticks/tok --kind measured \
  --statement "BitNet-2B decode, Dell i5-5200U bare metal, post bd-PROCHOT clear" \
  --scope "Dell Inspiron 15 i5-5200U, no OS, UEFI unikernel, clock 113% of nominal" \
  --source docs/hardware_logs/bitnet_baremetal_postfix_2026-07-29.log \
  --ceiling "single boot; the 0.61->2.85 tok/s pair against 2026-07-12 is CONFOUNDED (8 commits apart)" >/dev/null

echo
./tools/ledger.py list
