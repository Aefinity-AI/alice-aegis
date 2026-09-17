#!/usr/bin/env bash
# demo/reg-art12-cv-triage/run_all.sh — CV triage evidence demo (reg-1).
#
# Runs 10 synthetic candidate profiles (candidates.tsv) through the
# receipt-emitting agent_trace path (K=1, greedy CIS-1 FullInt decode of
# the checked-in M7 tinybit model) against ONE FIXED rubric prompt, for
# every candidate. Each run produces an AEGIS-TRACE v2 receipt
# (out/receipts/), a human-readable decoded-text sidecar via cis_decode
# (out/decoded/ — NOT part of the receipt itself, same convention as
# demo/edge-receipt), and, if a TPM is present, a TPM quote binding the
# receipt's trace-chain digest to the box's PCR state at that moment
# (out/attest/).
#
# This demo does not claim the M7 tinybit model (7 layers, hidden 384,
# vocab 8192, trained on short children's-story-style text — see
# ../../model-lab/tinybit) produces competent CV screening output. It
# almost certainly will not. The point of this demo is the evidence
# plumbing (Art. 12/15 candidate fields), not model capability.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ARTIFACTS="$ROOT/model-lab/tinybit/m7_final_gate_work/artifacts"
MODEL="$ARTIFACTS/MODEL.SAF"
EMBED="$ARTIFACTS/EMBED.BIN"
VOCAB="$ARTIFACTS/VOCAB.BIN"
AGENT_TRACE="$ROOT/aegis-linux/target/release/examples/agent_trace"
CIS_DECODE="$ROOT/cis-verify/target/release/examples/cis_decode"
ATTEST_SH="$ROOT/demo/edge-receipt/attest.sh"
OUT="$HERE/out"
N=64
K=1

# The one FIXED rubric prompt for every candidate in this run (Art. 12/15
# evidence point: the exact instruction each decision was made under must
# be recorded — here it is folded into the receipt's own `prompt-hex`
# field, so it is bound into the trace-chain digest, not just logged
# separately).
RUBRIC='Score this candidate 1 to 5 on three criteria: Experience, Communication, Technical Skill. Give one short justification line per score.
Candidate: '

mkdir -p "$OUT/receipts" "$OUT/decoded" "$OUT/attest"

[ -x "$AGENT_TRACE" ] || { echo "agent_trace not built — run demo/agent-trace/run.sh build first" >&2; exit 1; }
[ -x "$CIS_DECODE" ] || { echo "cis_decode not built — run (cd cis-verify && cargo build --release --features std --example cis_decode)" >&2; exit 1; }

HAVE_TPM=0
if [ -e /dev/tpmrm0 ] && command -v tpm2_quote >/dev/null 2>&1; then
    HAVE_TPM=1
    echo "TPM present (/dev/tpmrm0 + tpm2_quote on PATH) — will attest every receipt." >&2
else
    echo "NO TPM available on this host — falling back to receipt-only evidence (no attest step)." >&2
fi

SUMMARY="$OUT/RUN_SUMMARY.tsv"
echo -e "id\tname\treceipt\tdecoded\ttrace_chain\tverify\tattest" > "$SUMMARY"

tail -n +2 "$HERE/candidates.tsv" | while IFS=$'\t' read -r id name profile; do
    [ -z "$id" ] && continue
    prompt="${RUBRIC}${profile}"
    echo "== $id ($name) ==" >&2

    receipt="$OUT/receipts/${id}.receipt.txt"
    "$AGENT_TRACE" gen "$MODEL" "$EMBED" "$VOCAB" "$K" "$N" "$prompt" > "$receipt"

    decoded="$OUT/decoded/${id}.decoded.txt"
    "$CIS_DECODE" "$MODEL" "$EMBED" "$VOCAB" "$N" "$prompt" > "$decoded" 2>&1 || true

    trace_chain="$(grep '^trace-chain ' "$receipt" | awk '{print $2}')"

    verify_out="$("$AGENT_TRACE" verify "$MODEL" "$EMBED" "$VOCAB" "$receipt" 2>&1)"
    verify_status="FAIL"
    echo "$verify_out" | grep -q 'VERIFY PASS' && verify_status="PASS"

    attest_status="skipped-no-tpm"
    if [ "$HAVE_TPM" = "1" ]; then
        attestdir="$OUT/attest"
        if "$ATTEST_SH" quote "$receipt" "$attestdir" >>"$OUT/attest.log" 2>&1; then
            attest_status="quoted"
        else
            attest_status="FAILED (see out/attest.log)"
        fi
    fi

    echo -e "${id}\t${name}\t$(basename "$receipt")\t$(basename "$decoded")\t${trace_chain}\t${verify_status}\t${attest_status}" >> "$SUMMARY"
done

echo "done. summary: $SUMMARY" >&2
cat "$SUMMARY" >&2
