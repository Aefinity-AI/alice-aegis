#!/usr/bin/env bash
# build.sh — assemble the minimal-Linux arm boot payload (UKI + FAT image).
#
# Produces, under scripts/linux-arm/out/:
#   BOOTX64.EFI          unified kernel image (Debian vmlinuz + initramfs + cmdline)
#   aegis-linux-arm.img  complete bootable FAT image (for dd to a USB stick)
#   esp-tree/            loose files (to copy onto an existing FAT partition)
#
# The initramfs contains: busybox-static, static musl aegis-linux +
# inproc_variance + msrtool, the M7 model artifacts, the ordered module list
# resolved from modules.dep, and init.sh as /init.
#
# No sudo, no loop mounts — FAT image built with mtools (same pattern as
# scripts/qemu_synth_gauntlet.sh).
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
WORK=$PWD/work
OUT=$PWD/out
KVER=6.12.94+deb13-amd64
KDIR=$WORK/kernel/usr/lib/modules/$KVER
VMLINUZ=$WORK/kernel/boot/vmlinuz-$KVER
ARTIFACTS=$REPO/model-lab/tinybit/m7_final_gate_work/artifacts
MUSL=x86_64-unknown-linux-musl

[ -f "$VMLINUZ" ] || { echo "kernel not extracted at $VMLINUZ"; exit 1; }
[ -f "$ARTIFACTS/MODEL.SAF" ] || { echo "M7 artifacts missing"; exit 1; }

echo "== 1. static binaries =="
( cd "$REPO/aegis-linux" &&
  RUSTFLAGS="-C target-feature=+crt-static" cargo build --release --target $MUSL --bin aegis-linux --bin msrtool &&
  RUSTFLAGS="-C target-feature=+crt-static" cargo build --release --target $MUSL \
      --example inproc_variance --example membw --example witness --example cis_selftest )
# kernel-candidate A/B benches (same-binary interleaved, embedded clock-state
# block) — settled on idle bare iron instead of the contended dev box
( cd "$REPO/aegis-core" &&
  RUSTFLAGS="-C target-feature=+crt-static" cargo build --release --target $MUSL --bin fused_vs_sequential --bin bitplane_vs_lut --bin colskip_vs_incumbent --bin cis_vs_float )
BINDIR=$REPO/aegis-linux/target/$MUSL/release
COREBIN=$REPO/aegis-core/target/$MUSL/release

echo "== 2. initramfs tree =="
IR=$WORK/initramfs
rm -rf "$IR"
mkdir -p "$IR"/{bin,sbin,dev,proc,sys,mnt,aegis,modules}

cp /usr/bin/busybox "$IR/bin/busybox"          # busybox-static package
for app in sh mount umount insmod sleep sync tee cat cut grep wc date uname poweroff mkdir dmesg; do
    ln -sf busybox "$IR/bin/$app"
done
cp "$BINDIR/aegis-linux"                 "$IR/bin/aegis-linux"
cp "$BINDIR/msrtool"                     "$IR/bin/msrtool"
cp "$BINDIR/examples/inproc_variance"    "$IR/bin/inproc_variance"
cp "$BINDIR/examples/membw"              "$IR/bin/membw"
cp "$BINDIR/examples/witness"            "$IR/bin/witness"
cp "$BINDIR/examples/cis_selftest"       "$IR/bin/cis_selftest"
cp "$COREBIN/fused_vs_sequential"        "$IR/bin/fused_vs_sequential"
cp "$COREBIN/bitplane_vs_lut"            "$IR/bin/bitplane_vs_lut"
cp "$COREBIN/colskip_vs_incumbent"       "$IR/bin/colskip_vs_incumbent"
cp "$COREBIN/cis_vs_float"               "$IR/bin/cis_vs_float"
cp "$ARTIFACTS"/{MODEL.SAF,EMBED.BIN,VOCAB.BIN} "$IR/aegis/"
# real captured BitNet-2B down_proj input vectors — the colskip bench's
# PRIMARY scenario (synthetic z under-models clustering; ledger A15)
cp "$REPO/artifacts/relu2_down_in_bitnet2b_2026-08-01.av1" "$IR/aegis/relu2_down_in.av1"
cp init.sh "$IR/init" && chmod +x "$IR/init"

echo "== 3. modules: resolve deps from modules.dep, decompress, order =="
[ -f "$KDIR/modules.dep" ] || /usr/sbin/depmod -b "$WORK/kernel/usr" "$KVER"
python3 - "$KDIR" "$IR/modules" << 'PYEOF'
import os, subprocess, sys
kdir, outdir = sys.argv[1], sys.argv[2]
# roots: usb host controllers, usb storage, scsi disk, vfat, nls, msr
ROOTS = ["xhci-pci", "xhci-hcd", "ehci-pci", "ehci-hcd", "usb-storage", "uas",
         "sd_mod", "fat", "vfat", "nls_cp437", "nls_iso8859-1", "nls_ascii", "msr"]
dep = {}
with open(os.path.join(kdir, "modules.dep")) as f:
    for line in f:
        left, _, right = line.partition(":")
        dep[left.strip()] = right.split()
by_name = {os.path.basename(k).split(".ko")[0].replace("_", "-"): k for k in dep}
order, seen = [], set()
def visit(path):
    if path in seen: return
    seen.add(path)
    for d in dep.get(path, []): visit(d)
    order.append(path)
missing = []
for r in ROOTS:
    k = by_name.get(r.replace("_", "-"))
    if k: visit(k)
    else: missing.append(r)
names = []
for rel in order:
    src = os.path.join(kdir, rel)
    base = os.path.basename(rel)
    if base.endswith(".xz"):
        subprocess.run(["xz", "-dk", "-c", src],
                       stdout=open(os.path.join(outdir, base[:-3]), "wb"), check=True)
        names.append(base[:-3])
    else:
        subprocess.run(["cp", src, outdir], check=True)
        names.append(base)
with open(os.path.join(outdir, "insmod.order"), "w") as f:
    f.write("\n".join(names) + "\n")
print(f"   {len(names)} modules, missing roots (built-in or absent): {missing}")
PYEOF

echo "== 4. cpio + UKI =="
mkdir -p "$OUT"
( cd "$IR" && find . | cpio -o -H newc --quiet | gzip -6 ) > "$WORK/initramfs.cpio.gz"
ls -la "$WORK/initramfs.cpio.gz"
# Production: tty0 LAST so the Dell's screen is the primary console.
# Test variant: ttyS0 LAST so QEMU -nographic shows the init script's output.
CMDLINE_BASE="rdinit=/init mitigations=off quiet loglevel=4"
ukify build \
    --linux "$VMLINUZ" \
    --initrd "$WORK/initramfs.cpio.gz" \
    --cmdline "console=ttyS0,115200 console=tty0 $CMDLINE_BASE" \
    --stub /usr/lib/systemd/boot/efi/linuxx64.efi.stub \
    --output "$OUT/BOOTX64.EFI" >/dev/null
ukify build \
    --linux "$VMLINUZ" \
    --initrd "$WORK/initramfs.cpio.gz" \
    --cmdline "console=tty0 console=ttyS0,115200 $CMDLINE_BASE aegis_benchreps=1 aegis_mechv2n=1" \
    --stub /usr/lib/systemd/boot/efi/linuxx64.efi.stub \
    --output "$OUT/BOOTX64-QEMUTEST.EFI" >/dev/null
ls -la "$OUT"/BOOTX64*.EFI

echo "== 5. FAT image + esp tree =="
IMG=$OUT/aegis-linux-arm.img
rm -f "$IMG"
truncate -s 128M "$IMG"
/usr/sbin/mkfs.fat -F 32 -n AEGISLNX "$IMG" >/dev/null
mmd -i "$IMG" ::/EFI ::/EFI/BOOT
mcopy -i "$IMG" "$OUT/BOOTX64.EFI" ::/EFI/BOOT/BOOTX64.EFI
echo "minimal-linux arm marker $(date -u +%F)" > "$WORK/AEGIS_LINUX_ARM.tag"
mcopy -i "$IMG" "$WORK/AEGIS_LINUX_ARM.tag" ::/AEGIS_LINUX_ARM.tag

rm -rf "$OUT/esp-tree"
mkdir -p "$OUT/esp-tree/EFI/BOOT"
cp "$OUT/BOOTX64.EFI" "$OUT/esp-tree/EFI/BOOT/"
cp "$WORK/AEGIS_LINUX_ARM.tag" "$OUT/esp-tree/"

echo "== done =="
echo "  image    : $IMG  (dd to a stick, or use loose files below)"
echo "  esp tree : $OUT/esp-tree/  (copy onto any FAT partition)"
