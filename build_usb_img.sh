#!/bin/bash
set -e

echo "[AEGIS] Generating empty 3.0GB FAT32 image..."
dd if=/dev/zero of=aegis-boot.img bs=1M count=940 status=progress

echo "[AEGIS] Formatting as FAT32..."
mformat -i aegis-boot.img -F -v "ALICE_UEFI" ::

echo "[AEGIS] Creating EFI/BOOT directories..."
mmd -i aegis-boot.img ::/EFI
mmd -i aegis-boot.img ::/EFI/BOOT

echo "[AEGIS] Copying UEFI Bootloader..."
mcopy -i aegis-boot.img aegis-uefi/target/x86_64-uefi-hardfloat/release/aegis-uefi.efi ::/EFI/BOOT/BOOTX64.EFI

echo "[AEGIS] Copying Tokenizer Vocab..."
mcopy -i aegis-boot.img aegis-forge/vocab.bin ::/VOCAB.BIN

echo "[AEGIS] Copying Neural Weights (1.83GB)..."
mcopy -i aegis-boot.img aegis_pruned_model.safetensors ::/MODEL.SAF

echo "[AEGIS] Copying Embeddings..."
mcopy -i aegis-boot.img aegis-forge/embed.bin ::/EMBED.BIN

# Legacy-firmware fallback: quirky UEFI implementations drop to the EFI shell
# instead of honoring \EFI\BOOT\BOOTX64.EFI; the shell auto-runs this script.
# Omitting it silently breaks the pc-machine matrix case and old real boxes.
echo "[AEGIS] Copying startup.nsh (legacy EFI-shell fallback)..."
mcopy -i aegis-boot.img aegis-uefi/startup.nsh ::/STARTUP.NSH

echo "[AEGIS] Success! aegis-boot.img is ready to be burned to a USB stick."
