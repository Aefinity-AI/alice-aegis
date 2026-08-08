#!/usr/bin/env bash
# qemu-test.sh — CORRECTNESS-ONLY boot test of the minimal-Linux arm (Rule A:
# no performance number from this run may be recorded anywhere).
#
# Boots a copy of the FAT image (test-console UKI swapped in) as a USB stick
# under OVMF, waits for the gauntlet to power the machine off, then extracts
# BOOTLOG_LINUX_ARM.txt from the image and asserts the run completed.
set -euo pipefail
cd "$(dirname "$0")"

IMG=work/qemu-test.img
cp out/aegis-linux-arm.img "$IMG"
mcopy -o -i "$IMG" out/BOOTX64-QEMUTEST.EFI ::/EFI/BOOT/BOOTX64.EFI

KVM=""
[ -w /dev/kvm ] && KVM="-enable-kvm -cpu host" || KVM="-cpu max"

timeout 900 qemu-system-x86_64 $KVM -m 2G -machine q35 \
    -bios /usr/share/ovmf/OVMF.fd -nographic \
    -device qemu-xhci \
    -drive if=none,id=stick,format=raw,file="$IMG" \
    -device usb-storage,drive=stick \
    > work/qemu-test-serial.log 2>&1 || true

rm -f work/BOOTLOG_LINUX_ARM.txt
mcopy -i "$IMG" ::/BOOTLOG_LINUX_ARM.txt work/BOOTLOG_LINUX_ARM.txt 2>/dev/null || {
    echo "FAIL: no BOOTLOG_LINUX_ARM.txt written to the image; serial tail:"
    tail -30 work/qemu-test-serial.log
    exit 1
}
echo "== BOOTLOG extracted; tail =="
tail -15 work/BOOTLOG_LINUX_ARM.txt
grep -q "GAUNTLET DONE" work/BOOTLOG_LINUX_ARM.txt \
    && echo "QEMU BOOT TEST PASS (correctness only — Rule A: numbers in this log are MEANINGLESS)" \
    || { echo "FAIL: gauntlet did not complete"; exit 1; }
