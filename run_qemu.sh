#!/bin/bash
set -e

echo "[1/6] Creating 1.5GB Raw Disk Image..."
dd if=/dev/zero of=aegis-uefi-qemu.img bs=1M count=3000 status=progress

echo "[2/6] Partitioning Image (GPT + ESP)..."
sudo parted -s aegis-uefi-qemu.img mklabel gpt
sudo parted -s aegis-uefi-qemu.img mkpart primary fat32 1MiB 100%
sudo parted -s aegis-uefi-qemu.img set 1 esp on

echo "[3/6] Mapping Loop Device..."
LOOPDEV=$(sudo losetup -fP --show aegis-uefi-qemu.img)
echo "Mapped to $LOOPDEV"
sleep 1

echo "[4/6] Formatting FAT32..."
sudo mkfs.fat -F 32 -n ALICE_UEFI ${LOOPDEV}p1

echo "[5/6] Mounting and Copying Payload..."
sudo mkdir -p /mnt/qemu_usb
sudo mount ${LOOPDEV}p1 /mnt/qemu_usb
sudo cp -r /home/killboxincorporated/aegis-uefi-usb-payload/* /mnt/qemu_usb/
sudo sync
sudo umount /mnt/qemu_usb
sudo losetup -d $LOOPDEV

echo "[6/6] Launching QEMU..."
# Run QEMU in the background, redirecting its output to a log file
qemu-system-x86_64 \
    -bios /usr/share/ovmf/OVMF.fd \
    -drive file=aegis-uefi-qemu.img,format=raw \
    -nographic \
    -m 2G \
    -cpu max > qemu_output.log 2>&1 &
QEMU_PID=$!
echo $QEMU_PID > qemu.pid
echo "QEMU launched with PID $QEMU_PID"
