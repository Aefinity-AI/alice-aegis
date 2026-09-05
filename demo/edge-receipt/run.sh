#!/usr/bin/env bash
# demo/edge-receipt/run.sh — one-command "verified edge inference" demo.
#
# Produces a CIS-1 witness receipt for a greedy, integer-only decode of the
# checked-in M7 tinybit model, then replays and verifies that receipt with
# TWO independent verifiers (the in-tree cis_witness example and the
# standalone no_std cis-verify binary). A PASS means: given these three
# artifact files and this receipt, any conforming machine reproduces the
# exact same token ids and digest chain. See README.md for what this does
# and does NOT prove.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# Artifacts: default to the in-repo M7 tinybit model. Override with env vars to
# point at another CIS-1 artifact set (e.g. the 2B set): AEGIS_ARTIFACTS=<dir>
# (expects MODEL.SAF/EMBED.BIN/VOCAB.BIN in it) or the three files individually
# via AEGIS_MODEL / AEGIS_EMBED / AEGIS_VOCAB.
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
    echo "== build: cis_witness (aegis-linux, release, CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS) ==" >&2
    ( cd "$ROOT/aegis-linux" && run cargo build --release --example cis_witness )
    echo "== build: cis-verify (standalone verifier, release, features=std) ==" >&2
    ( cd "$ROOT/cis-verify" && run cargo build --release --features std --bin cis-verify )
    echo "== build: cis_decode (text decode helper, release, features=std) ==" >&2
    ( cd "$ROOT/cis-verify" && run cargo build --release --features std --example cis_decode )
    echo "build done." >&2
}

cis_witness_bin() { echo "$ROOT/aegis-linux/target/release/examples/cis_witness"; }
cis_verify_bin()  { echo "$ROOT/cis-verify/target/release/cis-verify"; }
cis_decode_bin()  { echo "$ROOT/cis-verify/target/release/examples/cis_decode"; }

cmd_gen() {
    need_artifacts
    local prompt="${1:?usage: run.sh gen \"<prompt>\" [max_new]}"
    local max_new="${2:-32}"
    mkdir -p "$OUT"
    local hostname_s; hostname_s="$(hostname)"
    local ts; ts="$(date -u +%Y%m%dT%H%M%SZ)"
    local receipt="$OUT/receipt-${hostname_s}-${ts}.txt"
    local meta="${receipt}.meta"

    echo "== gen: cis_witness on $hostname_s ==" >&2
    t0=$(date +%s.%N)
    run "$(cis_witness_bin)" gen "$MODEL" "$EMBED" "$VOCAB" "$max_new" "$prompt" > "$receipt"
    t1=$(date +%s.%N)
    echo "wrote $receipt"

    echo "== decode: cis_decode (text form, same prompt/max_new — not part of the receipt) ==" >&2
    local decode_out
    decode_out="$("$(cis_decode_bin)" "$MODEL" "$EMBED" "$VOCAB" "$max_new" "$prompt" 2>&1 || true)"
    echo "$decode_out" >&2

    local avx2_flags fma_flags
    avx2_flags="$(grep -c avx2 /proc/cpuinfo || true)"; avx2_flags="${avx2_flags:-0}"
    fma_flags="$(grep -c ' fma' /proc/cpuinfo || true)"; fma_flags="${fma_flags:-0}"
    local cpu_model
    cpu_model="$(grep -m1 'model name' /proc/cpuinfo | sed 's/^model name\s*:\s*//' || true)"
    cpu_model="${cpu_model:-unknown}"
    local aegis_commit
    aegis_commit="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
    aegis_commit="${aegis_commit:-unknown}"
    local elapsed
    elapsed="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')"

    {
        echo "hostname $hostname_s"
        echo "uname-m $(uname -m)"
        echo "cpu-model $cpu_model"
        echo "cpuinfo-avx2-flag-count $avx2_flags"
        echo "cpuinfo-fma-flag-count $fma_flags"
        echo "aegis-commit $aegis_commit"
        echo "gen-wallclock-seconds $elapsed (not a benchmark)"
        echo "decoded-text-line $(echo "$decode_out" | grep '^text ' || echo 'NOT PRINTED — cis_decode produced no text line, see raw output above')"
    } > "$meta"
    echo "wrote $meta"
    echo "$receipt"
}

cmd_verify() {
    need_artifacts
    local receipt="${1:?usage: run.sh verify <receipt-file>}"
    if [ ! -f "$receipt" ]; then
        echo "no such receipt: $receipt" >&2
        exit 2
    fi
    local hostname_s; hostname_s="$(hostname)"
    echo "== verify: cis_witness verify ==" >&2
    local w_out w_rc
    set +e
    w_out="$(run "$(cis_witness_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$receipt" 2>&1)"
    w_rc=$?
    set -e
    echo "$w_out"

    echo "== verify: cis-verify (standalone, no_std core) ==" >&2
    local s_out s_rc
    set +e
    s_out="$(run "$(cis_verify_bin)" "$receipt" "$MODEL" "$EMBED" "$VOCAB" 2>&1)"
    s_rc=$?
    set -e
    echo "$s_out"

    local model_sha embed_sha vocab_sha
    model_sha="$(sha256sum "$MODEL" | cut -d' ' -f1)"
    embed_sha="$(sha256sum "$EMBED" | cut -d' ' -f1)"
    vocab_sha="$(sha256sum "$VOCAB" | cut -d' ' -f1)"
    local digest_line; digest_line="$(grep '^cis-digest ' "$receipt" || echo 'cis-digest MISSING')"

    echo ""
    echo "===================== ATTESTATION ====================="
    echo "What was verified: that replaying the greedy, integer-only"
    echo "  (CIS-1 FullInt) decode of the M7 tinybit model on THIS machine,"
    echo "  from the receipt's prompt and artifact hashes, reproduces the"
    echo "  same token ids and digest chain the receipt claims."
    echo "Artifact SHA-256 (local, on $hostname_s):"
    echo "  MODEL.SAF $model_sha"
    echo "  EMBED.BIN $embed_sha"
    echo "  VOCAB.BIN $vocab_sha"
    echo "Receipt   : $digest_line"
    if [ "$w_rc" -eq 0 ]; then echo "cis_witness verify : PASS"; else echo "cis_witness verify : FAIL"; fi
    if [ "$s_rc" -eq 0 ]; then echo "cis-verify (std)    : PASS"; else echo "cis-verify (std)    : FAIL"; fi
    echo "Verifying machine   : $hostname_s ($(uname -m))"
    echo "========================================================="

    if [ "$w_rc" -ne 0 ] || [ "$s_rc" -ne 0 ]; then
        echo "OVERALL: FAIL" >&2
        exit 1
    fi
    echo "OVERALL: PASS"
}

cmd_all() {
    local prompt="${1:?usage: run.sh all \"<prompt>\" [max_new]}"
    local max_new="${2:-32}"
    cmd_build
    local receipt
    receipt="$(cmd_gen "$prompt" "$max_new" | tail -1)"
    cmd_verify "$receipt"
}

case "${1:-}" in
    build)  shift; cmd_build "$@" ;;
    gen)    shift; cmd_gen "$@" ;;
    verify) shift; cmd_verify "$@" ;;
    all)    shift; cmd_all "$@" ;;
    *)
        echo "usage: $0 {build|gen \"<prompt>\" [max_new]|verify <receipt>|all \"<prompt>\" [max_new]}" >&2
        exit 2
        ;;
esac
