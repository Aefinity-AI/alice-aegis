#!/bin/bash
# A.L.I.C.E. hardware-compatibility test matrix.
#
# Boots the qemu-test build of the unikernel across CPU generations, firmware
# machine types, and boot-media paths WITHOUT burning USB sticks. KVM masks
# host CPU features down to each model, so old-CPU behavior (e.g. a pre-2013
# Dell with no AVX2) is tested at native speed.
#
#   Nehalem     ~2008-2010 (SSE4.2, no AVX)      -> must use scalar fallback
#   SandyBridge ~2011-2012 (AVX, no AVX2/FMA)    -> must use scalar fallback
#   Haswell     2013+      (AVX2+FMA)            -> must use AVX2 path
#   host        this machine                     -> must use AVX2 path
#
# Media paths:
#   sata = q35 AHCI disk (QEMU default)
#   usb  = XHCI USB mass storage — the exact controller path a real laptop
#          uses when booting from a USB stick (exercises the 64KB DMA
#          bounce-buffer logic against a real XHCI device model)
#
# Success = QEMU exit 33 (isa-debug-exit 0x10: engine generated real tokens).
# Failure = exit 35 (test failed / panic) or timeout.
set -u
cd "$(dirname "$0")"

IMG="${1:-$HOME/aegis-boot.img}"
EFI=target/x86_64-uefi-hardfloat/release/aegis-uefi.efi
LOGDIR="matrix_logs"
mkdir -p "$LOGDIR"

echo "[MATRIX] Building qemu-test EFI..."
./build_hardfloat.sh --qemu-test >/dev/null 2>&1 || { echo "BUILD FAILED"; exit 1; }
mcopy -o -i "$IMG" "$EFI" ::/EFI/BOOT/BOOTX64.EFI

run_case() {
    local cpu="$1" machine="$2" media="$3" mem="${4:-2G}"
    local name="${cpu}_${machine}_${media}_${mem}"
    local log="$LOGDIR/$name.log"
    local media_args
    if [ "$media" = "usb" ]; then
        media_args="-device qemu-xhci,id=xhci -drive if=none,id=stick,file=$IMG,format=raw -device usb-storage,bus=xhci.0,drive=stick"
    else
        media_args="-drive file=$IMG,format=raw"
    fi

    timeout 720 qemu-system-x86_64 \
        -enable-kvm -nographic \
        -bios /usr/share/ovmf/OVMF.fd \
        -m "$mem" -machine "$machine" -cpu "$cpu" \
        $media_args \
        -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
        > "$log" 2>&1
    local code=$?

    local simd
    simd=$(grep -a -o "SIMD level: [^.]*" "$log" | tail -1 | sed 's/SIMD level: //')
    if [ "$code" = "33" ]; then
        printf "  PASS  %-28s simd=%s\n" "$name" "${simd:-?}"
    elif [ "$code" = "124" ]; then
        printf "  TIMEOUT %-26s simd=%s (see %s)\n" "$name" "${simd:-?}" "$log"
    else
        printf "  FAIL(%s) %-25s simd=%s (see %s)\n" "$code" "$name" "${simd:-?}" "$log"
    fi
}

echo "[MATRIX] Running..."
# SIMD dispatch tiers (KVM masks host features down to each CPU model)
run_case Nehalem      q35 sata 2G   # 2008-class: no AVX  -> scalar fallback
run_case SandyBridge  q35 sata 2G   # AVX but no AVX2/FMA -> scalar fallback
run_case Haswell      q35 sata 2G   # first AVX2+FMA      -> vector path
run_case host         q35 sata 2G
# Firmware / boot-media paths
run_case host         pc  sata 2G   # legacy machine type (startup.nsh fallback)
run_case host         q35 usb  2G   # XHCI USB mass storage — the real-stick path
# Memory paths (allocate_huge_pages + init_uefi_alloc_large behavior)
run_case host         q35 sata 8G   # >4GB: weight buffers may land above the 4G line
run_case host         q35 sata 1400M # tight: forces the large-heap chunk-halving loop
echo "[MATRIX] Done. Full serial logs in $LOGDIR/."
echo
echo "NOTE: QEMU's xhci model does not reproduce real-USB short reads, 64KB"
echo "boundary enforcement, or BOT stalls, and OVMF's memory map is far cleaner"
echo "than real firmware. A green matrix is necessary, not sufficient — real"
echo "hardware remains the final check (BOOTLOG.TXT records the failing stage)."
