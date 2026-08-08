#!/bin/bash
set -e
echo "[AEGIS UEFI] Preparing OVMF..."
python3 -c 'open("OVMF.fd", "wb").write(open("/usr/share/OVMF/OVMF_CODE_4M.fd", "rb").read() + open("/usr/share/OVMF/OVMF_VARS_4M.fd", "rb").read())'

echo "[AEGIS UEFI] Launching QEMU..."
qemu-system-x86_64 \
    -bios OVMF.fd \
    -drive file=aegis-boot.img,format=raw \
    -m 4G \
    -smp 4 \
    -cpu max \
    -nographic -serial mon:stdio \
    -machine q35
