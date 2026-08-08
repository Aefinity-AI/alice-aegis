# Provenance note — first physical-iron kit verification (Dell), 2026-08-08

Companion to: dell_i5-5200U_kit_iron_verify_bootlog_2026-08-08.txt
(raw BOOTLOG.TXT copied byte-identical off the boot stick, md5
becd7cef9bfac0961d973f99db7c8a04).

## Event
Operator Justin B. Thompson, physically present, booted the Provable AI Kit
stick on the **Dell Inspiron 15 (i5-5200U, Broadwell-U)** via F12 boot menu,
Secure Boot off, 2026-08-08. On-screen result observed by operator and
recorded by firmware to BOOTLOG.TXT on the stick:

    STAGE V: witness verify PASS — VERIFY PASS — this machine reproduced
    all 64 decode steps' full logit vectors bit-for-bit, with no OS underneath

## Stick identity and pre-boot verification (this session, dev host)
- Stick: SanDisk Cruzer Glide, /dev/disk/by-id/usb-SanDisk_Cruzer_Glide_4C530000250302100171-0:0
- Image written: aegis-kit-staging/aegis-kit-iron.img,
  md5 320e1918579dfdf6285ef7620e1c737a (QEMU-proven payload, kit_iron_qemu.log)
- Post-write readback of all 67108864 bytes: md5 identical to image.
- File-level: EFI/BOOT/BOOTX64.EFI 01936a7d54ee07de1798900d79fd3a5f,
  MODEL.SAF 53235f594ca3df50785cda6538d17075,
  EMBED.BIN 297315890a7fa2aa8efcf068240fa2d9,
  VOCAB.BIN 03301400fff883d86b37520cbe135533,
  RECEIPT.TXT 87c45bdd34f4f2bf56ec77c68c6dbadb (= tests/golden/witness_v1_m7_once64.receipt),
  README.TXT 071e2b6049d2824e140dfa1c2537c65b — all OK vs MD5SUMS.txt.

## Attribution of the second BOOTLOG entry to the Dell
The image ships with one baked-in entry (the QEMU OVMF proof run). The
returned stick carries exactly one additional entry, appended after the
baked-in one. Diff against the pristine image isolates it. Corroboration
that it came from different firmware, not a re-run of QEMU: the UEFI
allocator placed EMBED.BIN at 0x3AC000 / VOCAB.BIN at 0x9AC000 in the new
entry vs 0x1780000 / 0x3AC000 in the QEMU entry (different firmware memory
maps).

## Limitation (stated plainly)
The kit binary in verifier mode does not print CPUID/brand string into
BOOTLOG.TXT, so machine attribution rests on (a) operator physical witness
and (b) the firmware memory-map difference above — not on an in-log CPU
identifier. Identity claim only; no timing measured or quotable (Rule A).
