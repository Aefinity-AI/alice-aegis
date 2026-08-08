#!/bin/bash
# Build the A.L.I.C.E. unikernel with HARD-FLOAT + real AVX2 codegen.
#
# WHY THIS EXISTS: the stock `x86_64-unknown-uefi` Rust target is soft-float
# ("-mmx,-sse,+soft-float"). Built that way, every f32 op becomes a software
# library call and ALL AVX2 intrinsics are scalarized — the binary contains
# zero vector instructions and inference runs ~100x slower than intended.
# x86_64-uefi-hardfloat.json is the same target with hard-float + SSE2
# baseline (safe: UEFI x64 firmware initializes SSE per spec). AVX2/FMA stay
# per-function via #[target_feature], and main() enables OSXSAVE/XCR0 before
# any of those functions run.
#
# Requires: nightly toolchain with rust-src (build-std rebuilds core/alloc
# for the custom target).
set -e
cd "$(dirname "$0")"

FEATURES=""
if [ "$1" == "--qemu-test" ]; then
    FEATURES="--features qemu-test"
fi

cargo +nightly build --release \
    --target ./x86_64-uefi-hardfloat.json \
    -Zjson-target-spec \
    -Zbuild-std=core,alloc \
    -Zbuild-std-features=compiler-builtins-mem \
    $FEATURES

EFI=target/x86_64-uefi-hardfloat/release/aegis-uefi.efi
echo ""
echo "Built: $EFI"

# Gate the build: a silent regression to the soft-float target would emit zero
# vector instructions and cost ~100x performance. Fail loudly instead.
FMA=$(objdump -d "$EFI" | grep -c vfmadd231ps || true)
YMM=$(objdump -d "$EFI" | grep -c ymm || true)
echo "AVX2 sanity check:  fma=$FMA  ymm=$YMM"
if [ "$FMA" -eq 0 ] || [ "$YMM" -eq 0 ]; then
    echo "FATAL: no vector instructions in the binary — soft-float regression."
    echo "       Check that --target ./x86_64-uefi-hardfloat.json is being used."
    exit 1
fi
echo "OK: hard-float codegen confirmed."
echo ""
echo "Install into boot image:  mcopy -o -i ~/aegis-boot.img $EFI ::/EFI/BOOT/BOOTX64.EFI"
