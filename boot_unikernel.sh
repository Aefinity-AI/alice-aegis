#!/bin/bash
# boot_unikernel.sh
# AEGIS PROTOCOL: Safe Virtualized Bare-Metal Boot Sequence

cd /home/killboxincorporated/antigravity-aegis || exit 1

echo "[AEGIS] Building A.L.I.C.E. Unikernel (Single-Threaded to protect host RAM)..."
cargo +nightly build -Zbuild-std=std,core,alloc,panic_abort --release -j 1 --target x86_64-unknown-hermit

echo "[AEGIS] Launching QEMU Bare-Metal Emulator..."
# EMULATOR FLAGS EXPLAINED:
# -kernel: Points directly to our compiled RustyHermit payload.
# -m 2G: Strictly restricts the emulator to 2GB RAM. Do not increase this.
# -nographic: Forces all I/O through the current Crostini terminal UI.
# -cpu max: Exposes the host's AVX2 and vector extensions to the guest unikernel.
# -machine q35: Emulates a modern PCIe-based motherboard for advanced memory mapping.

qemu-system-x86_64 \
    -kernel target/x86_64-unknown-hermit/release/antigravity-aegis \
    -m 2G \
    -nographic \
    -cpu max \
    -machine q35 \
    -display none
