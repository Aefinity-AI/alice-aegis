#!/usr/bin/env bash
# demo/agent-trace/live20/tamper.sh — swap-a-tool-result tamper test over a
# subset of the 20 real-2B-model episodes in out/receipts. For each target
# receipt, copies it, replaces exactly one step's `out=<hex>` field with a
# DIFFERENT genuine tool-output value (lifted from another real receipt of
# the same tool kind, so the substituted bytes are themselves well-formed —
# this is a "swap a tool result" attack, not a bit-flip), then runs
# `agent_trace verify` and records PASS/FAIL + the first mismatch reason.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BIN="$ROOT/aegis-linux/target/release/examples/agent_trace"
ART=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts
MODEL="$ART/aegis_pruned_model.cis.safetensors"
EMBED="$ART/embed.bin"
VOCAB="$ART/vocab.bin"
DEMO_TABLE="$ROOT/demo/agent-trace/tables/demo.tsv"
CHAIN_TABLE="$ROOT/demo/agent-trace/tables/chain.tsv"

RECEIPTS="$HERE/out/receipts"
TOUT="$HERE/out/tamper"
mkdir -p "$TOUT"
RESULT="$TOUT/RESULT.tsv"
: > "$RESULT"
echo -e "id\tswapped_step\told_out_hex\tnew_out_hex\tverify_result\treason" >> "$RESULT"

# (receipt_id, step, old_out_hex, new_out_hex, table)
CASES=(
  "calc_02|0|3634|313434|"
  "calc_03|0|313434|3634|"
  "lookup_02|0|426f6c742c2068657820686561642c20312f342d3230207820332f3420696e2e2c206361646d69756d20706c61746564|46696c7465722c206675656c2c20696e6c696e652c203130206d6963726f6e|$DEMO_TABLE"
  "lookup_03|0|46696c7465722c206675656c2c20696e6c696e652c203130206d6963726f6e|426f6c742c2068657820686561642c20332f382d31362078203120696e2e2c206361646d69756d20706c61746564|$DEMO_TABLE"
  "fileread_01|0|436c616d702c20686f73652c20776f726d2d64726976652c20312d312f3420696e2e|426f6c742c2068657820686561642c20332f382d31362078203120696e2e2c206361646d69756d20706c61746564|$DEMO_TABLE"
  "mixed_lookup_calc_01|1|3230|3634|$DEMO_TABLE"
  "chain_lookup_01|1|4761736b65742c204f2d72696e672c206675656c206c696e652c20312f3420696e2e|537570657273656465642c20736565207061727420502d313030|$CHAIN_TABLE"
)

for c in "${CASES[@]}"; do
    IFS='|' read -r id step old new table <<< "$c"
    src="$RECEIPTS/$id.receipt"
    dst="$TOUT/${id}.tampered.receipt"
    if ! grep -q "^step ${step}: .*out=${old} " "$src"; then
        echo "SKIP $id: old out= value not found verbatim as expected" | tee -a "$RESULT"
        continue
    fi
    sed "s/^step ${step}: \(.*out=\)${old} /step ${step}: \1${new} /" "$src" > "$dst"
    table_args=()
    [ -n "$table" ] && table_args=(--table "$table")
    set +e
    out=$("$BIN" verify "$MODEL" "$EMBED" "$VOCAB" "$dst" "${table_args[@]}" 2>&1)
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        result="STILL-VERIFIES(BUG)"
        reason="$(echo "$out" | tail -1)"
    else
        result="FAIL(caught)"
        reason="$(echo "$out" | grep -m1 -E 'MISMATCH|FAIL|mismatch' || echo "$out" | tail -1)"
    fi
    echo -e "${id}\t${step}\t${old}\t${new}\t${result}\t${reason}" >> "$RESULT"
    echo "$out" > "$TOUT/${id}.verify.log"
done

echo "=== tamper matrix ==="
column -t -s $'\t' "$RESULT"
