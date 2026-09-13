#!/usr/bin/env bash
# safe-2b: verify the tampered EVAL-60 T1 (2B model) receipt set.
# Same invocation as the box1 cross-machine replay leg
# (~/legs/eval-60-replay-box1/run.sh): agent_trace verify with --table and
# --suite-sha256, one receipt at a time, in item_id order.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."/..
BIN=aegis-linux/target/release/examples/agent_trace
ART=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts
TABLE=demo/agent-trace/tables/demo.tsv
SUITE=5ecf5fbf616634547206f55edb6ea611a0065d47d569de431459131bc14a5786
DIR=eval/receipts/eval-60-2b-T1-box1-tampered/receipts
LOG=eval/receipts/eval-60-2b-T1-box1-tampered/verify.log

pass=0
fail=0
: > "$LOG"
for r in $(ls "$DIR"/*.txt | sort); do
    id=$(basename "$r" .txt)
    out=$("$BIN" verify "$ART/aegis_pruned_model.cis.safetensors" "$ART/embed.bin" "$ART/vocab.bin" "$r" --table "$TABLE" --suite-sha256 "$SUITE" 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        pass=$((pass+1))
        result=PASS
    else
        fail=$((fail+1))
        result=FAIL
    fi
    {
        echo "$id: $result"
        echo "$out" | sed 's/^/  /'
    } >> "$LOG"
done
echo "TOTAL: $pass PASS / $fail FAIL / $((pass+fail)) items" >> "$LOG"
cat "$LOG" | tail -5
