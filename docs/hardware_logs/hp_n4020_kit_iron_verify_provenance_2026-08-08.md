# Provenance note — HP kit verification leg (first unikernel boot on this box), 2026-08-08

Companion to: hp_n4020_kit_iron_verify_bootlog_2026-08-08.txt (raw BOOTLOG.TXT
copied byte-identical off the stick, md5 4fb5fc8ba0ccffad54b89d832fea070f;
contains 3 entries: QEMU baked-in, Dell leg, and this one).

## Event
Operator Justin B. Thompson, physically present, booted the same Provable AI
Kit stick (SanDisk Cruzer Glide 4C530000250302100171, payload unchanged since
the Dell leg — no restaging between boots) on the **HP Stream (Celeron N4020,
Gemini Lake)**, 2026-08-08. This is the FIRST unikernel boot ever attempted
on this box (prior HP legs A25/A26 were minimal-Linux userspace). Firmware
booted it; result appended to BOOTLOG.TXT:

    STAGE V: witness verify PASS — VERIFY PASS — this machine reproduced
    all 64 decode steps' full logit vectors bit-for-bit, with no OS underneath

Diff against the banked Dell-era log (2 entries) isolates exactly one new
entry.

## Attribution limitation (stated plainly)
Unlike the Dell leg, the new entry's UEFI memory map is IDENTICAL to the
Dell entry's (EMBED.BIN @0x3AC000, VOCAB.BIN @0x9AC000), so the in-log
fingerprint does NOT discriminate between the two machines this time.
Attribution to the HP rests on operator physical witness alone. Verifier
mode prints no CPUID.

## Scalar-path corollary (architectural derivation, not an in-log line)
The N4020 does not implement AVX2. The kit binary attempts AVX enablement
and dispatches at runtime; had the AVX2 kernels executed on this CPU the
boot would have died on #UD, not passed. Therefore a PASS on this box
implies the verification ran the SSE2 scalar path — meaning the same golden
receipt (tests/golden/witness_v1_m7_once64.receipt) has now re-derived
bit-for-bit through two disjoint kernel code paths on iron (AVX2 on the
Dell leg, scalar here), conditional on the attribution above.

Identity claim only; no timing measured or quotable (Rule A).
