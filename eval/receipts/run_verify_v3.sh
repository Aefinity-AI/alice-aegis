#!/usr/bin/env bash
# safe-2c: verify the item-ctx-bound (v3) EVAL-60 T1 receipt set, each item
# checked against its OWN expected item-ctx (--expect-item/--expect-prompt-
# file/--nonce). Parallelized (model load dominates per-invocation cost;
# each item needs its own --expect-item so single-process multi-receipt
# verify, which shares one set of flags, doesn't apply here).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."/..
BIN=aegis-linux/target/release/examples/agent_trace
ART=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts
TABLE=demo/agent-trace/tables/demo.tsv
SUITE=5ecf5fbf616634547206f55edb6ea611a0065d47d569de431459131bc14a5786
NONCE=safe2c-demo-nonce-1
PROMPTS=/home/cm/projects/alice-aegis-cm-safe2/eval/receipts/eval-60-2b-T1-box1/prompts
DIR="${1:?usage: run_verify_v3.sh <receipt-dir> <log-file>}"
LOG="${2:?usage: run_verify_v3.sh <receipt-dir> <log-file>}"

: > "$LOG"
verify_one() {
    r="$1"
    id=$(basename "$r" .txt)
    out=$("$BIN" verify "$ART/aegis_pruned_model.cis.safetensors" "$ART/embed.bin" "$ART/vocab.bin" "$r" \
        --table "$TABLE" --suite-sha256 "$SUITE" \
        --expect-item "$id" --expect-prompt-file "$PROMPTS/$id.txt" --nonce "$NONCE" 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then result=PASS; else result=FAIL; fi
    { echo "$id: $result"; echo "$out" | sed 's/^/  /'; }
}
export -f verify_one
export BIN ART TABLE SUITE NONCE PROMPTS

ls "$DIR"/*.txt | sort | xargs -P4 -I{} bash -c 'verify_one "$@"' _ {} >> "$LOG"

pass=$(grep -c ': PASS$' "$LOG")
fail=$(grep -c ': FAIL$' "$LOG")
echo "TOTAL: $pass PASS / $fail FAIL / $((pass+fail)) items" >> "$LOG"
tail -5 "$LOG"
