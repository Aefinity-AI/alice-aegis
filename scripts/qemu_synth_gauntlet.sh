#!/usr/bin/env bash
# Weights-free QEMU gate: boot the UEFI unikernel on a SYNTHETIC silu /
# no-SubLN / untied-head model and require the qemu-test success signal.
#
# This proves the whole chain — forge repack, FAT32 artifact loading,
# no_std metadata-config parsing, the generalized graph, real generation —
# on the actual boot target, with no model weights required. It does NOT
# replace the real-model G4b gate before a USB flash; it makes regressions
# in the machinery visible anywhere, CI included.
#
# Needs: qemu-system-x86_64, OVMF, mkfs.fat (dosfstools), mtools, python3.
# No sudo and no loop mounts — the image is built with mtools.
#
#   scripts/qemu_synth_gauntlet.sh        -> exit 0 on pass
set -euo pipefail
cd "$(dirname "$0")/.."

OVMF=""
for c in /usr/share/ovmf/OVMF.fd /usr/share/OVMF/OVMF.fd /usr/share/edk2/x64/OVMF.4m.fd; do
    [ -f "$c" ] && OVMF=$c && break
done
[ -n "$OVMF" ] || { echo "no OVMF firmware found (install ovmf)"; exit 2; }
command -v qemu-system-x86_64 >/dev/null || { echo "qemu-system-x86_64 missing"; exit 2; }
command -v mcopy >/dev/null || { echo "mtools missing"; exit 2; }
command -v mkfs.fat >/dev/null || { echo "dosfstools missing"; exit 2; }

echo "== building unikernel (qemu-test feature) =="
( cd aegis-uefi && cargo build --release --target x86_64-unknown-uefi --features qemu-test )

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "== forging synthetic model =="
python3 aegis-forge/gen_synth_checkpoint.py "$WORK/ckpt" "$WORK/forged"

echo "== building FAT32 boot image =="
IMG=$WORK/aegis-synth.img
truncate -s 64M "$IMG"
mkfs.fat -F 32 "$IMG" >/dev/null
mmd -i "$IMG" ::/EFI ::/EFI/BOOT
mcopy -i "$IMG" aegis-uefi/target/x86_64-unknown-uefi/release/aegis-uefi.efi ::/EFI/BOOT/BOOTX64.EFI
mcopy -i "$IMG" "$WORK/forged/MODEL.SAF" ::/MODEL.SAF
mcopy -i "$IMG" "$WORK/forged/EMBED.BIN" ::/EMBED.BIN
mcopy -i "$IMG" "$WORK/forged/VOCAB.BIN" ::/VOCAB.BIN

echo "== booting QEMU (TCG ok; KVM used if present) =="
KVM=""
[ -w /dev/kvm ] && KVM="-enable-kvm"
set +e
timeout 300 qemu-system-x86_64 $KVM \
    -nographic \
    -bios "$OVMF" \
    -m 1G \
    -machine q35 \
    -cpu max \
    -drive file="$IMG",format=raw \
    -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
    > "$WORK/serial.log" 2>&1
CODE=$?
set -e

# isa-debug-exit: value v exits QEMU with 2v+1. The unikernel writes 0x10
# on success (-> 33) and 0x11 on failure (-> 35).
grep -a "TEST" "$WORK/serial.log" | tr -d '\r' || true
if [ "$CODE" = "33" ]; then
    echo "QEMU synthetic gauntlet PASSED (exit 33 = unikernel success signal)"
    exit 0
fi
echo "QEMU synthetic gauntlet FAILED (exit $CODE); serial tail:"
tail -30 "$WORK/serial.log" | tr -d '\r'
exit 1
