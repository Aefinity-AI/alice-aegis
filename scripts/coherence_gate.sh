#!/usr/bin/env bash
# THE gate. Run it before believing anything, and before writing any document.
#
# When asked "what is the minimum intervention that would have caught fifteen
# months of fabricated results in week one," the answer was: a coherence test,
# wired to CI, before the first line of documentation is written.
#
# This is that test. It builds the engine, runs it against the real weights, and
# asserts the model can still say Paris. It does not check that a function
# returns the right type. It checks that the machine still thinks.
#
#   ./scripts/coherence_gate.sh        -> exit 0 pass, 1 fail
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODEL=~/aegis_pruned_model.safetensors
EMBED=~/aegis-forge/embed.bin
VOCAB=~/aegis-forge/vocab.bin

fail() { echo "❌ COHERENCE GATE FAILED: $*"; exit 1; }

for f in "$MODEL" "$EMBED" "$VOCAB"; do
    [ -f "$f" ] || fail "missing artifact: $f"
done

echo "[1/4] integrity: no printed number may be uncomputed"
python3 scripts/integrity_check.py aegis-core >/dev/null || fail "aegis-core prints uncomputed metrics"
python3 scripts/integrity_check.py aegis-uefi >/dev/null || fail "aegis-uefi prints uncomputed metrics"
python3 scripts/integrity_check.py aegis-eval >/dev/null || fail "aegis-eval prints uncomputed metrics"
echo "      ok"

echo "[2/4] the kernels agree with themselves"
( cd aegis-core && cargo test --release --quiet --test gemm_equivalence -- --test-threads=1 ) >/dev/null 2>&1 \
    || fail "batched GEMM no longer matches the per-token path"
( cd aegis-core && cargo test --release --quiet --features parallel --test thread_safety -- --test-threads=1 ) >/dev/null 2>&1 \
    || fail "parallel kernels no longer match the serial path"
echo "      ok"

echo "[3/4] the unikernel still emits vector instructions"
( cd aegis-uefi && ./build_hardfloat.sh ) >/dev/null 2>&1 \
    || fail "unikernel build failed, or regressed to soft-float (zero SIMD)"
echo "      ok"

echo "[4/4] the model can still say Paris"
( cd aegis-linux && cargo build --release --features parallel ) >/dev/null 2>&1 \
    || fail "aegis-linux build failed"
OUT=$(cd aegis-linux && timeout 300 ./target/release/aegis-linux "$MODEL" "$EMBED" "$VOCAB" 20 \
      "What is the capital of France?" 2>&1)
echo "$OUT" | grep -qi "paris" || {
    echo "--- actual output ---"; echo "$OUT" | tail -6; echo "---------------------"
    fail "the model did not say Paris. Nothing else matters until it does."
}
echo "      \"$(echo "$OUT" | grep -i 'Final Full Response' | cut -c1-70)\""

echo
echo "✅ COHERENCE GATE PASSED — the engine works, and every number it prints, it computed."
