#!/usr/bin/env bash
# demo/agent-trace/run.sh — one-command "verified agent episode" demo.
#
# Produces an AEGIS-TRACE v0 receipt for a small, deterministic K-step agent
# episode (greedy CIS-1 FullInt decode + a scan for one `calc` tool call,
# repeated K times over the M7 tinybit model), then replays and verifies
# that receipt on this machine. A PASS means: given these three artifact
# files and this receipt, any conforming machine reproduces the exact same
# per-step token ids, tool outcomes, and trace chain. See README.md for
# what this does and does NOT prove.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# Artifacts: default to the in-repo M7 tinybit model, same default as
# demo/edge-receipt. Override with AEGIS_ARTIFACTS=<dir> (expects
# MODEL.SAF/EMBED.BIN/VOCAB.BIN) or the three files individually via
# AEGIS_MODEL / AEGIS_EMBED / AEGIS_VOCAB.
ARTIFACTS="${AEGIS_ARTIFACTS:-$ROOT/model-lab/tinybit/m7_final_gate_work/artifacts}"
MODEL="${AEGIS_MODEL:-$ARTIFACTS/MODEL.SAF}"
EMBED="${AEGIS_EMBED:-$ARTIFACTS/EMBED.BIN}"
VOCAB="${AEGIS_VOCAB:-$ARTIFACTS/VOCAB.BIN}"
OUT="$HERE/out"
CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export CARGO_BUILD_JOBS

run() { echo "+ $*" >&2; "$@"; }

need_artifacts() {
    for f in "$MODEL" "$EMBED" "$VOCAB"; do
        if [ ! -f "$f" ]; then
            echo "missing artifact: $f (this demo never downloads anything — it uses the checked-in M7 model)" >&2
            exit 1
        fi
    done
}

cmd_build() {
    need_artifacts
    echo "== build: agent_trace (aegis-linux, release, CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS) ==" >&2
    ( cd "$ROOT/aegis-linux" && run cargo build --release --example agent_trace )
    echo "build done." >&2
}

agent_trace_bin() { echo "$ROOT/aegis-linux/target/release/examples/agent_trace"; }

cmd_gen() {
    need_artifacts
    local prompt="${1:-Once upon a time}"
    local k="${2:-3}"
    local n="${3:-16}"
    mkdir -p "$OUT"
    local hostname_s; hostname_s="$(hostname)"
    local ts; ts="$(date -u +%Y%m%dT%H%M%SZ)"
    local receipt="$OUT/trace-${hostname_s}-${ts}.txt"

    echo "== gen: agent_trace on $hostname_s (K=$k N=$n) ==" >&2
    run "$(agent_trace_bin)" gen "$MODEL" "$EMBED" "$VOCAB" "$k" "$n" "$prompt" > "$receipt"
    echo "wrote $receipt"
    echo "$receipt"
}

cmd_verify() {
    need_artifacts
    local receipt="${1:?usage: run.sh verify <receipt-file>}"
    if [ ! -f "$receipt" ]; then
        echo "no such receipt: $receipt" >&2
        exit 2
    fi
    echo "== verify: agent_trace verify ==" >&2
    "$(agent_trace_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$receipt"
}

# Three adversarial mutations of a known-good receipt, each of which MUST
# make `verify` print VERIFY FAIL and exit 1: flip a token id, flip a
# tool-output byte, drop a step line entirely.
cmd_tamper() {
    need_artifacts
    mkdir -p "$OUT"
    local good
    good="$(cmd_gen "The quick brown fox" 3 16 | tail -1)"
    echo "== tamper baseline: verifying the untouched receipt first (must PASS) ==" >&2
    if ! cmd_verify "$good" >/dev/null 2>&1; then
        echo "baseline receipt does not verify — cannot run tamper tests" >&2
        exit 1
    fi
    echo "baseline PASS" >&2

    local overall=0

    echo ""
    echo "== tamper 1/3: flip a token id ==" >&2
    local t1="$OUT/tamper-flip-token.txt"
    awk '
        BEGIN{done=0}
        /^step 0: / && done==0 {
            match($0, /toks=[0-9]+/)
            tok=substr($0, RSTART+5, RLENGTH-5)
            newtok = tok + 1
            line=$0
            sub("toks=" tok, "toks=" newtok, line)
            print line
            done=1
            next
        }
        {print}
    ' "$good" > "$t1"
    set +e
    "$(agent_trace_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$t1"
    rc1=$?
    set -e
    if [ "$rc1" -ne 0 ]; then echo "tamper 1 (flip token id): FAIL as expected (exit $rc1)"; else echo "tamper 1 (flip token id): DID NOT FAIL — BUG"; overall=1; fi

    echo ""
    echo "== tamper 2/3: flip a tool-output byte ==" >&2
    # A step's tool output is legitimately empty when the tool is "no-tool"
    # (the default prompt rarely provokes the model into emitting a CALC(...)
    # call within N=16 tokens) — there is no byte to flip in an empty field.
    # So: if the first step's out= is non-empty, flip its first hex nibble;
    # if it is empty, inject one fake output byte (0x39, ASCII '9') instead.
    # Either way this mutates the tool-output field the trace chain folds in,
    # which must make verify FAIL.
    local t2="$OUT/tamper-flip-output.txt"
    awk '
        BEGIN{done=0}
        /^step / && done==0 {
            if (match($0, /out=[0-9a-f]*/)) {
                val=substr($0, RSTART+4, RLENGTH-4)
                line=$0
                if (length(val) > 0) {
                    first=substr(val,1,1)
                    if (first=="0") { newfirst="1" } else { newfirst="0" }
                    newval = newfirst substr(val,2)
                } else {
                    newval = "39"
                }
                sub("out=" val, "out=" newval, line)
                print line
                done=1
                next
            }
        }
        {print}
    ' "$good" > "$t2"
    set +e
    "$(agent_trace_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$t2"
    rc2=$?
    set -e
    if [ "$rc2" -ne 0 ]; then echo "tamper 2 (flip tool-output byte): FAIL as expected (exit $rc2)"; else echo "tamper 2 (flip tool-output byte): DID NOT FAIL — BUG"; overall=1; fi

    echo ""
    echo "== tamper 3/3: drop a step ==" >&2
    local t3="$OUT/tamper-drop-step.txt"
    grep -v '^step 1: ' "$good" > "$t3"
    set +e
    "$(agent_trace_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$t3"
    rc3=$?
    set -e
    if [ "$rc3" -ne 0 ]; then echo "tamper 3 (drop a step): FAIL as expected (exit $rc3)"; else echo "tamper 3 (drop a step): DID NOT FAIL — BUG"; overall=1; fi

    echo ""
    if [ "$overall" -eq 0 ]; then
        echo "TAMPER SELFTEST: PASS — all three mutations were rejected"
    else
        echo "TAMPER SELFTEST: FAIL — see BUG lines above" >&2
        exit 1
    fi
}

cmd_all() {
    local prompt="${1:-Once upon a time}"
    local k="${2:-3}"
    local n="${3:-16}"
    cmd_build
    local receipt
    receipt="$(cmd_gen "$prompt" "$k" "$n" | tail -1)"
    cmd_verify "$receipt"
    cmd_tamper
}

case "${1:-}" in
    build)  shift; cmd_build "$@" ;;
    gen)    shift; cmd_gen "$@" ;;
    verify) shift; cmd_verify "$@" ;;
    tamper) shift; cmd_tamper "$@" ;;
    all)    shift; cmd_all "$@" ;;
    *)
        echo "usage: $0 {build|gen [prompt] [K] [N]|verify <receipt>|tamper|all [prompt] [K] [N]}" >&2
        exit 2
        ;;
esac
