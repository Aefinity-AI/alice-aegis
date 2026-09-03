#!/bin/bash
# build_scalar_uefi.sh — build aegis-uefi.efi with EVERY AVX2/FMA kernel
# compiled out (aegis-core's `scalar_only` feature), for third-party x86-64
# hardware with no AVX2/FMA — box2, the Celeron N4020 (Gemini Lake) third
# iron leg (E7c). Companion to build_hardfloat.sh, which builds the OPPOSITE
# artifact (AVX2 required, staging-gated to REJECT a binary with zero vector
# instructions). This script's gate is the mirror image: it hard-fails if the
# binary contains ANY vfmadd*/vpmadd*/ymm instruction, because that would mean
# the scalar_only cfg-gating missed a call site and the artifact would SIGILL
# on real non-AVX2 iron instead of just running slow.
#
# Still hard-float + SSE2 baseline (x86_64-uefi-hardfloat.json): UEFI firmware
# initializes SSE2 per spec, so f32 math stays native scalar SSE2 instructions
# (movss/addss/mulss/...) rather than falling back to the stock target's
# soft-float library calls. "Scalar" here means "no AVX2/FMA", not "no SIMD
# registers at all" — see the xmm count in the census below.
#
# Requires: nightly toolchain with rust-src (build-std rebuilds core/alloc
# for the custom target, same as build_hardfloat.sh).
set -e
cd "$(dirname "$0")/../aegis-uefi"

# Default features (gop) stay ON — scalar_only is additive, so this build
# differs from build_hardfloat.sh's output ONLY in the AVX2/FMA kernel gating,
# nothing else (same console path, same qemu-test plumbing).
FEATURES="scalar_only"
if [ "$1" == "--qemu-test" ]; then
    FEATURES="scalar_only,qemu-test"
fi

cargo +nightly build --release \
    --target ./x86_64-uefi-hardfloat.json \
    -Zjson-target-spec \
    -Zbuild-std=core,alloc \
    -Zbuild-std-features=compiler-builtins-mem \
    --features "$FEATURES"

EFI=target/x86_64-uefi-hardfloat/release/aegis-uefi.efi
echo ""
echo "Built: $EFI"

# Gate the build: any AVX2/FMA instruction here means a scalar_only call site
# was missed upstream (aegis-core/src/ops.rs or lib.rs's module gates) and
# this binary would SIGILL the instant it hit that path on real non-AVX2
# hardware. Fail loudly instead of shipping it to box2.
DIS=$(mktemp)
trap 'rm -f "$DIS"' EXIT
objdump -d -M intel "$EFI" > "$DIS" 2>/dev/null

YMM=$(grep -c "ymm" "$DIS" || true)
FMA=$(grep -c "vfmadd" "$DIS" || true)
VPMADD=$(grep -c "vpmadd" "$DIS" || true)
XMM=$(grep -c "xmm" "$DIS" || true)
SIZE=$(stat -c %s "$EFI")

echo "AVX2/FMA sanity check:  ymm=$YMM  vfmadd=$FMA  vpmadd=$VPMADD  (xmm=$XMM, expected >0 — SSE2 baseline)"
if [ "$YMM" -ne 0 ] || [ "$FMA" -ne 0 ] || [ "$VPMADD" -ne 0 ]; then
    echo "FATAL: AVX2/FMA instructions present in a scalar_only build — a call site"
    echo "       was missed. This binary would SIGILL on real non-AVX2 iron (box2)."
    echo "       Check aegis-core/src/ops.rs for an un-cfg-gated simd_on() dispatch."
    exit 1
fi
echo "OK: zero AVX2/FMA instructions confirmed (EFI size: ${SIZE} bytes)."
echo ""
echo "Install into boot image:  scripts/make-kit-image.sh $EFI [out.img]"
