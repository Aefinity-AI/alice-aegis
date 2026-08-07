#!/bin/bash
set -e

echo "[QEMU] Building AEGIS UEFI Unikernel..."
cd /home/killboxincorporated/aegis-uefi
cargo build --release --target x86_64-unknown-uefi --features qemu-test

echo "[QEMU] Preparing Virtual FAT32 Disk Image (aegis.img)..."
cd /home/killboxincorporated

if [ ! -f aegis.img ]; then
    echo "Creating new 3.0GB FAT32 image..."
    truncate -s 3G aegis.img
    /sbin/mkfs.fat -F 32 aegis.img
    
    echo "Copying massive tensor files to FAT32 image (this happens only once)..."
    mkdir -p qemu_mnt
    sudo mount -o loop aegis.img qemu_mnt
    sudo cp /home/killboxincorporated/aegis-linux/aegis_model.safetensors qemu_mnt/aegis_model.safetensors
    sudo cp /home/killboxincorporated/aegis-forge/embed.bin qemu_mnt/embed.bin
    sudo cp /home/killboxincorporated/aegis-forge/vocab.bin qemu_mnt/vocab.bin
    sudo mkdir -p qemu_mnt/EFI/BOOT
    sudo umount qemu_mnt
fi

echo "Updating BOOTX64.EFI..."
mkdir -p qemu_mnt
sudo mount -o loop aegis.img qemu_mnt
sudo cp aegis-uefi/target/x86_64-unknown-uefi/release/aegis-uefi.efi qemu_mnt/EFI/BOOT/BOOTX64.EFI
sudo umount qemu_mnt

echo "[QEMU] Launching Virtual Machine in Headless Mode..."
set +e
qemu-system-x86_64 \
    -enable-kvm \
    -nographic \
    -bios /usr/share/ovmf/OVMF.fd \
    -m 5G \
    -machine q35 \
    -cpu max \
    -drive file=aegis.img,format=raw \
    -device isa-debug-exit,iobase=0xf4,iosize=0x04
EXIT_CODE=$?
set -e

# QEMU isa-debug-exit math:
# It takes the value written (e.g. 0x10), left-shifts by 1 and adds 1.
# So writing 0x10 (16) results in QEMU exiting with 16 * 2 + 1 = 33.
# Writing 0x11 (17) results in QEMU exiting with 17 * 2 + 1 = 35.

echo ""
echo "=================================================="
if [ "$EXIT_CODE" = "33" ]; then
    echo "[TEST PASSED] Alice gracefully exited and signaled 0x10 (Success)."
    exit 0
elif [ "$EXIT_CODE" = "35" ]; then
    echo "[TEST FAILED] Alice signaled 0x11 (Panic/Abort)."
    exit 1
else
    echo "[CRASH] QEMU exited with unexpected code $EXIT_CODE (Hardware fault / Triple fault)."
    exit 1
fi
