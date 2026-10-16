#!/usr/bin/env bash
# ALICE 2B receipt on a fleet host, paired with W2's bitnet.cpp host-class rows (2026-10-16).
# Reproduction of the 2026-09 box2 receipts at a pinned commit; not a new registered bar.
# Correctness only: wall time is recorded "not a benchmark" (no timing on w3/w4).
set -uo pipefail
NAME="${LEG_NAME:-alice-2b-w4}"
LEG_OUT="${LEG_OUT:-$HOME/legs/$NAME}"; mkdir -p "$LEG_OUT"; R="$LEG_OUT/RESULT.txt"
PIN=aa0309d532cfacc749277e332bc9862b29db3f4b
PROMPT="The quick brown fox jumps over the lazy dog near the old river bank."
MAXNEW=8
A="$HOME/aefinity-artifacts/bitnet2b-2b-artifacts"
SRC="$HOME/alice-pin"
export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=2
fail() { echo "RESULT: FAIL $NAME $*" >> "$R"; echo "=== $NAME done $(date -u +%FT%TZ)" >> "$R"; exit 1; }
echo "=== $NAME start $(date -u +%FT%TZ) host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-) avx2=$(grep -c avx2 /proc/cpuinfo)" >> "$R"
( cd "$A" && sha256sum aegis_pruned_model.cis.safetensors embed.bin vocab.bin ) >> "$R" || fail "artifacts missing"
grep -q facb3597665603ba45730cc1f70ba6d82f53473d97f04fd039ca4296a45868db "$R" || fail "model sha mismatch"
URL=$(git -C "$HOME/projects/alice-aegis" remote get-url origin 2>/dev/null) || fail "no local alice-aegis origin"
if [ ! -d "$SRC/.git" ]; then git clone -q "$HOME/projects/alice-aegis" "$SRC" || fail "clone"; fi
git -C "$SRC" fetch -q "$URL" main && git -C "$SRC" checkout -q --detach "$PIN" || fail "checkout $PIN"
( cd "$SRC/aegis-linux" && cargo build --release --example cis_witness ) > "$LEG_OUT/build-witness.log" 2>&1 || fail "build cis_witness"
( cd "$SRC/cis-verify" && cargo build --release --features std --bin cis-verify ) > "$LEG_OUT/build-verify.log" 2>&1 || fail "build cis-verify"
W="$SRC/aegis-linux/target/release/examples/cis_witness"; V="$SRC/cis-verify/target/release/cis-verify"
for run in R1 R2; do
  t0=$(date +%s)
  "$W" gen "$A/aegis_pruned_model.cis.safetensors" "$A/embed.bin" "$A/vocab.bin" "$MAXNEW" "$PROMPT" > "$LEG_OUT/receipt-$run.txt" 2> "$LEG_OUT/gen-$run.err" || fail "gen $run"
  echo "gen $run wall_s=$(( $(date +%s) - t0 )) (not a benchmark) receipt_sha256=$(sha256sum < "$LEG_OUT/receipt-$run.txt" | cut -c1-64)" >> "$R"
done
cmp -s "$LEG_OUT/receipt-R1.txt" "$LEG_OUT/receipt-R2.txt" && REP=SAME || REP=DIFF
"$W" verify "$A/aegis_pruned_model.cis.safetensors" "$A/embed.bin" "$A/vocab.bin" "$LEG_OUT/receipt-R1.txt" > "$LEG_OUT/verify-witness.log" 2>&1; VW=$?
"$V" "$LEG_OUT/receipt-R1.txt" "$A/aegis_pruned_model.cis.safetensors" "$A/embed.bin" "$A/vocab.bin" > "$LEG_OUT/verify-standalone.log" 2>&1; VS=$?
grep -iE 'digest|chain|token' "$LEG_OUT/receipt-R1.txt" | head -20 >> "$R"
echo "RESULT: OK $NAME host=$(hostname) avx2=$(grep -c avx2 /proc/cpuinfo) pin=${PIN:0:12} repeat=$REP verify_witness_rc=$VW verify_standalone_rc=$VS receipt=$(sha256sum < "$LEG_OUT/receipt-R1.txt" | cut -c1-16)" >> "$R"
echo "=== $NAME done $(date -u +%FT%TZ)" >> "$R"
