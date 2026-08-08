#!/usr/bin/env bash
# thread_sweep.sh — THE MISSING BENCHMARK. Run it first; it closes more
# confirmed DARPA-facing defects than anything else in this tree.
#
# WHAT IT FIXES, BY NAME:
#   [CRITICAL/UNLOGGED]      TECHNICAL_REPORT.md:32  3.64 / 6.00 / 7.91 / 8.25 tok/s, 2.27x
#   [CRITICAL/CONTRADICTED]  TECHNICAL_REPORT.md:207 "SMT is worth ~5%"
#   [CRITICAL/INTERNALLY_INCONSISTENT] :246 the evidentiary inversion
#   [HIGH/UNLOGGED]          :33  293 M -> 165 M cycles/token
#   [HIGH/CONTRADICTED]      :85  "4 threads appear to beat 8 was an artifact"
#   [HIGH/UNLOGGED]          :66  the 8.25 short-burst-peak explanation
#   ledger A4                the whole row, both arms, awaiting instrument output
# Every one of those is the same hole: there has never been a multicore
# thread-sweep benchmark in this repository that wrote a file. The only record
# of any sweep is prose in commit 154f00a; the "clean re-measure" that reversed
# it (254ba43) has no log at all AND silently changed the engine's shipping
# default to logical processors (aegis-core/src/ops.rs:621).
#
# THREE DESIGN POINTS THAT ARE NOT OPTIONAL:
#
# 1. ONE PROCESS PER THREAD COUNT. aegis-core/src/pool.rs:174 caches the pool in
#    a OnceLock and ops.rs:634 caches the count in an AtomicUsize, so
#    AEGIS_THREADS is read exactly once per process. An in-process sweep would
#    silently measure the SAME thread count four times and produce a beautiful,
#    fake "SMT is worth 5%". This is precisely the class of error under audit.
#
# 2. ROUND-ROBIN, NOT BLOCKED. Arms are interleaved (t1,t2,t4,t8, t1,t2,t4,t8, …)
#    so monotone drift — thermal ramp, crosvm balloon, a background sync — cannot
#    load onto one arm. The gauntlet already understands this (PSTATE_run1 vs
#    run2_control is a drift control); the userspace side never did.
#
# 3. RATIOS FROM CYCLES/TOKEN, NEVER FROM tok/s. Printed tok/s carries 2
#    decimals; deriving a ratio from it injects up to ~5% error (this program
#    published 5.08x where the tick-true figure was 4.84x — collect_gauntlet.sh
#    documents the lesson). Cycles/token is an integer from rdtsc.
#
# Usage (always via the wrapper, so a runcard exists):
#   ev run thread_sweep
#   ROUNDS=5 ITERS=3 THREADS="1 2 4 8" MAXNEW=64 ev run thread_sweep
set -uo pipefail

REPO="${ALICE_REPO:-$HOME}"
MODEL="${MODEL:-$REPO/aegis_pruned_model.safetensors}"
EMBED="${EMBED:-$REPO/aegis-forge/embed.bin}"
VOCAB="${VOCAB:-$REPO/aegis-forge/vocab.bin}"
BIN="${BIN:-$REPO/aegis-linux/target/release/examples/inproc_variance}"
THREADS="${THREADS:-1 2 4 8}"
ROUNDS="${ROUNDS:-5}"
ITERS="${ITERS:-3}"
MAXNEW="${MAXNEW:-64}"
PROMPT="${PROMPT:-Write a comprehensive and detailed essay about the future of artificial intelligence in aerospace.}"

for f in "$MODEL" "$EMBED" "$VOCAB"; do
    [ -r "$f" ] || { echo "FATAL: missing artifact $f" >&2; exit 2; }
done
if [ ! -x "$BIN" ]; then
    echo "FATAL: $BIN not built. Run:" >&2
    echo "  CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release -p aegis-linux --example inproc_variance" >&2
    exit 2
fi

CSV="$(mktemp -t thread_sweep.XXXXXX.csv)"
trap 'rm -f "$CSV"' EXIT
echo "round,threads,iter,decode_cyc_per_tok,decode_total_cyc,decode_steps,prefill_cyc,out_hash" | tee "$CSV"

echo "# THREAD SWEEP: threads=[$THREADS] rounds=$ROUNDS iters=$ROUNDS x $ITERS maxnew=$MAXNEW"
echo "# one process per arm (pool.rs:174 OnceLock); arms round-robin to defeat drift"
echo "# nproc logical=$(nproc) physical=$(lscpu -p=core 2>/dev/null | grep -vc '^#' || echo '?')"
echo "# features: whatever aegis-linux was built with — the runcard records the binary sha256"

for r in $(seq 1 "$ROUNDS"); do
    for t in $THREADS; do
        # A fresh process per arm is the whole point; the model reload cost is
        # outside the measured window (inproc_variance times decode only).
        out=$(AEGIS_THREADS="$t" "$BIN" "$MODEL" "$EMBED" "$VOCAB" "$MAXNEW" "$PROMPT" "$ITERS" 2>&1)
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "# ARM FAILED round=$r threads=$t rc=$rc"
            echo "$out" | sed 's/^/#   /'
            continue
        fi
        echo "$out" | awk -F, -v r="$r" -v t="$t" \
            'NR>1 && NF>=6 {print r","t","$0}' | tee -a "$CSV"
    done
done

echo
echo "==== SUMMARY (computed from the rows above, in this run) ===="
python3 "$(dirname "$0")/thread_sweep_parse.py" "$CSV"
