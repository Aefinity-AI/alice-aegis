#!/usr/bin/env bash
# check-efi-simd.sh — staging gate for unikernel binaries (ledger A14).
#
# The stock x86_64-unknown-uefi Rust target is soft-float: it compiles every
# f32 op — including #[target_feature] AVX2 kernel bodies — to scalar library
# calls. Such a binary is byte-exact (IEEE) and passes every correctness gate,
# then decodes 120-250x slower on iron (MECH-A1, 2026-08-01). This gate makes
# that failure loud: NO .efi may be staged to a stick unless its disassembly
# contains hardware vector instructions.
#
# Usage: scripts/check-efi-simd.sh path/to/BOOTX64.EFI
set -euo pipefail

EFI="${1:?usage: check-efi-simd.sh <path-to.efi>}"
[ -f "$EFI" ] || { echo "FAIL: no such file: $EFI" >&2; exit 1; }

DIS=$(mktemp)
trap 'rm -f "$DIS"' EXIT
objdump -d -M intel "$EFI" > "$DIS" 2>/dev/null || {
    echo "FAIL: objdump could not disassemble $EFI" >&2; exit 1; }

YMM=$(grep -c "ymm" "$DIS" || true)
FMA=$(grep -c "vfmadd" "$DIS" || true)
XMM=$(grep -c "xmm" "$DIS" || true)

echo "check-efi-simd: $EFI"
echo "  size   : $(stat -c %s "$EFI") bytes  md5: $(md5sum "$EFI" | cut -d' ' -f1)"
echo "  census : xmm=$XMM ymm=$YMM vfmadd=$FMA"

if [ "$YMM" -eq 0 ] || [ "$FMA" -eq 0 ]; then
    echo "  VERDICT: FAIL — soft-float build (stock target?). DO NOT STAGE." >&2
    echo "  Rebuild with ./aegis-uefi/build_hardfloat.sh" >&2
    exit 1
fi
echo "  VERDICT: PASS — hardware vector code present; safe to stage."
