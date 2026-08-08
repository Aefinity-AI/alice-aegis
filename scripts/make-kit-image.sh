#!/usr/bin/env bash
# make-kit-image.sh — assemble the Provable AI Kit boot image.
#
# A FAT32 image carrying: the unikernel (BOOTX64.EFI), the M7 model trio,
# the golden witness receipt, and a README for the human holding the stick.
# Booted on any UEFI x86-64 machine, the unikernel sees RECEIPT.TXT and runs
# witness verification instead of the REPL (aegis-uefi/src/verifier.rs).
#
# mtools ONLY — this script never touches a block device. Writing the image
# to a physical stick is a HUMAN step, documented at the end of the output.
#
# usage: make-kit-image.sh <path-to.efi> [out.img]
set -euo pipefail
EFI="${1:?usage: make-kit-image.sh <path-to.efi> [out.img]}"
OUT="${2:-$HOME/aegis-kit.img}"
M="$HOME/model-lab/tinybit/m7_final_gate_work/artifacts"
RECEIPT="$HOME/tests/golden/witness_v1_m7_once64.receipt"

for f in "$EFI" "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN" "$RECEIPT"; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

# 64 MB: M7 trio ~9 MB + .efi + slack; FAT32 floor is 33 MB.
rm -f "$OUT"
truncate -s 64M "$OUT"
mformat -i "$OUT" -F ::
mmd -i "$OUT" ::/EFI ::/EFI/BOOT

mcopy -i "$OUT" "$EFI" ::/EFI/BOOT/BOOTX64.EFI
mcopy -i "$OUT" "$M/MODEL.SAF" ::/MODEL.SAF
mcopy -i "$OUT" "$M/EMBED.BIN" ::/EMBED.BIN
mcopy -i "$OUT" "$M/VOCAB.BIN" ::/VOCAB.BIN
mcopy -i "$OUT" "$RECEIPT" ::/RECEIPT.TXT

TMP_README="$(mktemp)"
cat > "$TMP_README" <<'EOF'
THE PROVABLE AI KIT — Aefinity AI (github.com/Aefinity-AI/alice-aegis)

Boot this stick on any UEFI x86-64 machine (disable Secure Boot). No
operating system will load — the firmware starts a self-contained Rust
inference engine which:

  1. hashes the model files on this stick and checks them against
     RECEIPT.TXT — a signed-by-arithmetic record of a decode performed on
     the author's machine;
  2. replays that decode under CIS-1, a frozen integer semantics; and
  3. recomputes a SHA-256 chain over every step's full logit vector.

VERIFY PASS on your screen means YOUR machine just reproduced the
recorded AI computation bit-for-bit, with no OS underneath. If it prints
FAIL or a different digest: congratulations — read CHALLENGE.md in the
repository, because that finding wins a bounty.

Verification transcript is also written to BOOTLOG.TXT on this stick.
EOF
mcopy -i "$OUT" "$TMP_README" ::/README.TXT
rm -f "$TMP_README"

echo "kit image: $OUT"
md5sum "$OUT" "$EFI" "$RECEIPT"
echo
echo "QEMU check (correctness only, Rule A):"
echo "  qemu-system-x86_64 -enable-kvm -nographic -m 2G -machine q35 \\"
echo "    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \\"
echo "    -drive file=$OUT,format=raw \\"
echo "    -device isa-debug-exit,iobase=0xf4,iosize=0x04"
echo
echo "Physical stick (HUMAN step — never a /dev letter from a script):"
echo "  identify the stick by its tag file, then: mcopy the image contents"
echo "  or dd the image to the CONFIRMED stick device, then md5-verify."
