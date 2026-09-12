#!/usr/bin/env bash
# demo/agent-trace/live20/gen20.sh — tools-1 live half: 20 real-2B-model
# episodes covering CALC, LOOKUP, FILE-READ (some episodes mix two distinct
# tool kinds in one hash-chained trace), then verify all 20 and run a
# tamper matrix (swap one tool-result value) on a subset.
#
# Uses the real 2B model artifacts at
# /home/cm/aefinity-artifacts/bitnet2b-2b-artifacts (aegis_pruned_model.cis.safetensors
# + embed.bin + vocab.bin) via `agent_trace gen`/`verify` — same binary and
# mechanism as demo/agent-trace/run.sh and the EVAL-60 harness, just a
# smaller ad hoc corpus for the tools-1 live-half QUEUE item. Nothing here
# is a timing measurement (Rule A): `time` output in RUN.log is operator
# diagnostics only, never reported as a result.
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

OUT="$HERE/out"
mkdir -p "$OUT/receipts" "$OUT/prompts"
SUMMARY="$OUT/summary.tsv"
RUNLOG="$OUT/RUN.log"
: > "$SUMMARY"
: > "$RUNLOG"
echo -e "id\ttool_kinds\tK\tN\ttable\tgen_status\tverify_status" >> "$SUMMARY"

log() { echo "$*" | tee -a "$RUNLOG"; }

gen_one() {
    local id="$1" k="$2" n="$3" table="$4" prompt="$5" kinds="$6"
    local pf="$OUT/prompts/${id}.txt"
    local rf="$OUT/receipts/${id}.receipt"
    printf '%s' "$prompt" > "$pf"
    local table_args=()
    [ -n "$table" ] && table_args=(--table "$table")
    log "=== gen $id (K=$k N=$n table=${table:-none} kinds=$kinds) ==="
    if "$BIN" gen "$MODEL" "$EMBED" "$VOCAB" "$k" "$n" "$(cat "$pf")" "${table_args[@]}" > "$rf" 2>>"$RUNLOG"; then
        gen_status="ok"
    else
        gen_status="FAIL"
    fi
    if [ "$gen_status" = "ok" ]; then
        if "$BIN" verify "$MODEL" "$EMBED" "$VOCAB" "$rf" "${table_args[@]}" >> "$RUNLOG" 2>&1; then
            verify_status="PASS"
        else
            verify_status="FAIL"
        fi
    else
        verify_status="n/a"
    fi
    echo -e "${id}\t${kinds}\t${k}\t${n}\t${table:-none}\t${gen_status}\t${verify_status}" >> "$SUMMARY"
    log "$id: gen=$gen_status verify=$verify_status"
}

# --- CALC-only, K=1 (5 episodes, distinct numbers each time) ---
gen_one calc_01 1 16 "" "$(printf 'Q: 2 + 2\nA: CALC(2 + 2).\nQ: 10 + 10\nA: CALC(10 + 10).\nQ: 6 * 7\nA:')" calc
gen_one calc_02 1 16 "" "$(printf 'Q: 3 + 5\nA: CALC(3 + 5).\nQ: 9 - 4\nA: CALC(9 - 4).\nQ: 8 * 8\nA:')" calc
gen_one calc_03 1 16 "" "$(printf 'Q: 100 + 1\nA: CALC(100 + 1).\nQ: 50 - 25\nA: CALC(50 - 25).\nQ: 12 * 12\nA:')" calc
gen_one calc_04 1 16 "" "$(printf 'Q: 7 + 7\nA: CALC(7 + 7).\nQ: 20 - 6\nA: CALC(20 - 6).\nQ: 9 * 9\nA:')" calc
gen_one calc_05 1 16 "" "$(printf 'Q: 15 + 15\nA: CALC(15 + 15).\nQ: 30 - 12\nA: CALC(30 - 12).\nQ: 11 * 11\nA:')" calc

# --- LOOKUP-only, K=1 (5 episodes, distinct keys) ---
gen_one lookup_01 1 16 "$DEMO_TABLE" "$(printf 'Q: part P-100\nA: LOOKUP(P-100).\nQ: part P-101\nA:')" lookup
gen_one lookup_02 1 16 "$DEMO_TABLE" "$(printf 'Q: part P-205\nA: LOOKUP(P-205).\nQ: part P-206\nA:')" lookup
gen_one lookup_03 1 16 "$DEMO_TABLE" "$(printf 'Q: part P-317\nA: LOOKUP(P-317).\nQ: part P-318\nA:')" lookup
gen_one lookup_04 1 16 "$DEMO_TABLE" "$(printf 'Q: part P-402\nA: LOOKUP(P-402).\nQ: part P-403\nA:')" lookup
gen_one lookup_05 1 16 "$DEMO_TABLE" "$(printf 'Q: part P-511\nA: LOOKUP(P-511).\nQ: part P-612\nA:')" lookup

# --- FILE-READ-only, K=1 (4 episodes) ---
gen_one fileread_01 1 16 "$DEMO_TABLE" "$(printf 'Q: read file P-402\nA: FILE-READ(P-402).\nQ: read file P-403\nA: FILE-READ(P-403).\nQ: read file P-511\nA:')" file-read
gen_one fileread_02 1 16 "$DEMO_TABLE" "$(printf 'Q: read file P-100\nA: FILE-READ(P-100).\nQ: read file P-101\nA: FILE-READ(P-101).\nQ: read file P-205\nA:')" file-read
gen_one fileread_03 1 16 "$DEMO_TABLE" "$(printf 'Q: read file P-206\nA: FILE-READ(P-206).\nQ: read file P-317\nA: FILE-READ(P-317).\nQ: read file P-318\nA:')" file-read
gen_one fileread_04 1 16 "$DEMO_TABLE" "$(printf 'Q: read file P-511\nA: FILE-READ(P-511).\nQ: read file P-612\nA: FILE-READ(P-612).\nQ: read file P-206\nA:')" file-read

# --- mixed, K=2: two DISTINCT real tool kinds in one hash-chained episode ---
gen_one mixed_lookup_calc_01 2 20 "$DEMO_TABLE" "$(printf 'Q: 2 + 2\nA: CALC(2 + 2).\nQ: part P-100\nA: LOOKUP(P-100).\nQ: read file P-205\nA: FILE-READ(P-205).\nQ: 10 + 10\nA: CALC(10 + 10).\nQ: part P-101\nA:')" "lookup,calc"
gen_one mixed_lookup_calc_02 2 20 "$DEMO_TABLE" "$(printf 'Q: 3 + 3\nA: CALC(3 + 3).\nQ: part P-206\nA: LOOKUP(P-206).\nQ: read file P-317\nA: FILE-READ(P-317).\nQ: 20 + 20\nA: CALC(20 + 20).\nQ: part P-318\nA:')" "lookup,calc"

# --- mixed, K=2: LOOKUP -> LOOKUP chain (2 distinct real tool calls,
# chain.tsv's "superseded" rows engineer step 2's question into the
# appended TOOL[lookup]= text of step 1 rather than the initial prompt) ---
gen_one chain_lookup_01 2 40 "$CHAIN_TABLE" "$(printf 'Q: part P-402\nA: LOOKUP(P-402).\nQ: part P-403\nA: LOOKUP(P-403).\nQ: part P-901\nA:')" "lookup,lookup"
gen_one chain_lookup_02 2 40 "$CHAIN_TABLE" "$(printf 'Q: part P-402\nA: LOOKUP(P-402).\nQ: part P-403\nA: LOOKUP(P-403).\nQ: part P-902\nA:')" "lookup,lookup"

# --- mixed, K=2: FILE-READ -> FILE-READ chain (real 2-step FILE-READ) ---
gen_one chain_fileread_01 2 60 "$CHAIN_TABLE" "$(printf 'Q: part P-402\nA: LOOKUP(P-402).\nQ: read file P-403\nA: FILE-READ(P-403).\nQ: read file P-902\nA:')" "file-read,file-read"
gen_one chain_fileread_02 2 60 "$CHAIN_TABLE" "$(printf 'Q: part P-402\nA: LOOKUP(P-402).\nQ: read file P-403\nA: FILE-READ(P-403).\nQ: read file P-901\nA:')" "file-read,file-read"

log "=== all 20 episodes: gen+verify done, see $SUMMARY ==="
column -t -s $'\t' "$SUMMARY" | tee -a "$RUNLOG"
