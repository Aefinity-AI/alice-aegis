#!/usr/bin/env bash
# Phase 0 gate: the perplexity meter must be pinned before anything promotes.
#
#   T0a  --sample mode reproduces the known-good anchor 10.758 +/- 0.01 on
#        the reference file (test.txt, sha256 d790b833…, first 1900 tokens).
#        History: the original 12.80 pin's sample file was deleted Jul 11
#        and is unreproducible; re-based to 10.394 (Jul 12, int8_act+
#        parallel build) then 10.348 (Jul 14 logged run) — that delta was
#        floating-point summation order across thread counts. RE-PINNED
#        10.758 on 2026-07-16 (wikitext2_sample1900_2026-07-16_repin.log)
#        after the whitespace-run pre-tokenizer + id-0 merge-guard fixes:
#        tokenization got DENSER (newline runs merge; T2d vs Falcon-E
#        reference now exact), so per-token PPL rose as a metric artifact —
#        cross-tokenizer PPLs are not comparable; the pin is a regression
#        tripwire, not a quality claim.
#   T0b  the same short prefix produces the SAME number through the
#        --sample path and the default (chunked) path. The prefix is sized
#        to tokenize well under max_tokens: past the cap the two modes
#        truncate differently by design (re-encode-shrunken-text vs
#        slice-token-list), so a truncated comparison would be meaningless —
#        the script detects that and aborts as INCONCLUSIVE rather than
#        failing the meter. If T0a and T0b pass but a full-document chunked
#        run still explodes, the input file (raw WikiText markup?) or the
#        vocab pairing is the culprit — not the meter.
#
# CLI convention (merged evaluator, post-750d377): DEFAULT (flagless) mode
# is chunked full-text; the anchor pin needs the explicit --sample flag.
# (The 2026-07-13 handoff's "flagless is the single-sample pin" describes
# the patch-side evaluator that lost the merge — do not trust it.)
#
# Usage:
#   scripts/phase0_meter_check.sh MODEL.SAF EMBED.BIN VOCAB.BIN reference.txt
#
# Exit code 0 = gate passed. Every number is printed for the ledger row.
set -euo pipefail

if [ $# -ne 4 ]; then
    echo "usage: $0 MODEL.SAF EMBED.BIN VOCAB.BIN reference.txt" >&2
    exit 2
fi
MODEL=$1; EMBED=$2; VOCAB=$3; SAMPLE=$4
EXPECTED=10.758
TOL=0.01

cd "$(dirname "$0")/.."
cargo build --release --manifest-path aegis-eval/Cargo.toml
EVAL=aegis-eval/target/release/aegis-eval

echo "== T0a: regression pin (--sample mode on the reference file) =="
T0A=$("$EVAL" "$MODEL" "$EMBED" "$VOCAB" "$SAMPLE" 1900 --sample \
    | awk '/Perplexity \(teacher-forced/ {print $NF}')
echo "T0a PPL = $T0A (expected $EXPECTED +/- $TOL)"
awk -v p="$T0A" -v e="$EXPECTED" -v t="$TOL" \
    'BEGIN { d = p - e; if (d < 0) d = -d; exit !(d <= t) }' \
    || { echo "T0a FAIL: meter does not reproduce the pin"; exit 1; }
echo "T0a PASS"

echo "== T0b: --sample vs default (chunked) on an identical short prefix =="
PREFIX=$(mktemp)
trap 'rm -f "$PREFIX"' EXIT
# ASCII-only prefix (head -c could split a multi-byte char and break the
# reader) sized so even token-dense text stays under the 1900-token cap.
tr -cd '\11\12\15\40-\176' < "$SAMPLE" | head -c 4000 > "$PREFIX"

T0B_SAMPLE_OUT=$("$EVAL" "$MODEL" "$EMBED" "$VOCAB" "$PREFIX" 1900 --sample)
T0B_CHUNK_OUT=$("$EVAL" "$MODEL" "$EMBED" "$VOCAB" "$PREFIX" 1900)
if echo "$T0B_CHUNK_OUT" | grep -q "^WARNING: chunk"; then
    echo "T0b INCONCLUSIVE: the prefix tokenized past the cap, so the modes"
    echo "truncate differently by design. Shrink the prefix and rerun."
    exit 3
fi
T0B_SAMPLE=$(echo "$T0B_SAMPLE_OUT" | awk '/Perplexity \(teacher-forced/ {print $NF}')
T0B_CHUNK=$(echo "$T0B_CHUNK_OUT" | awk '/Perplexity \(teacher-forced/ {print $NF}')
echo "T0b sample=$T0B_SAMPLE chunked=$T0B_CHUNK"
# Same tokens through the same engine must give the same number.
[ "$T0B_SAMPLE" = "$T0B_CHUNK" ] \
    || { echo "T0b FAIL: the two paths disagree on identical input"; exit 1; }
echo "T0b PASS"

echo "Phase 0 gate PASSED. Ledger row: T0a=$T0A T0b=$T0B_SAMPLE/$T0B_CHUNK"
