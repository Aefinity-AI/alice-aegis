# RUNCARD — SUNDOWN BATTERY (2026-08-01 evening): four hands-free boots

Everything below is one stick-swap session. Each boot runs its full battery
unattended and appends everything to its stick. Binary md5s and log offsets
get pinned here at staging time (after QEMU gates pass) — do not boot before
this file says **STAGED**.

**Status: STAGED — both sticks ready for the evening swap session.**

## The drill, in order

| # | Machine | Stick | What runs | Watch for | Est. |
|---|---------|-------|-----------|-----------|------|
| 1 | **Dell** | U (57.8 GB) | MECH v1.1 block + **MECH v2: N=10×3 prompts, QUIET2 conditions** + H3 probe + gauntlet | `==== MECH DONE ====`, then power down | ~5 min |
| 2 | **Dell** | L (14.6 GB) | OS-cost gauntlet + **MECHV2L: N=10×3, console buffered** + membw (G1) + witness v0 + **CIS integer self-test digest** + kernel A/B benches | powers itself off | ~10 min |
| 3 | **HP Stream** | L (same stick) | Same payload, auto-detects the N4020: scalar harness + CIS digest + membw + witness; AVX2 benches self-skip | powers itself off | ~10 min |
| 4 | **HP Stream** | U (same stick) — *optional, exploratory* | The unikernel's runtime dispatch should take the scalar path on the N4020. Never bare-metal-tested on this binary; a crash is a finding, not a failure | either MECH DONE or a wedge — note which | ~5 min |

Then plug both sticks back in.

## What tonight yields if all four land

1. **MECH v2 scored** (prereg: `docs/hardware_logs/mech2_PREREGISTRATION_2026-08-01.md`)
   — the publishable-or-not verdict on "no-OS beats minimal Linux."
2. **FABLE-0 gate G1** — the Dell's real memory-bandwidth ceiling (membw).
3. **Cross-ISA jury preview (E4)** — Dell AVX2 vs HP scalar: same responses
   byte-for-byte? same `CIS_SELFTEST digest=` hex? If yes, the verified-
   inference story has bare-metal evidence on three implementations.
4. **Witness v0 on iron** — hash-chain demo output as a logged artifact.
5. HP scalar decode baselines (the "weakest machine that must still work").

## Staging pins (filled at staging time)

- U stick `EFI/BOOT/BOOTX64.EFI`: **STAGED**, md5
  `397ec75dca6c335b2af427ef02e4a6c1`, 404,480 B (hardfloat, census
  xmm=6850 ymm=433 vfmadd=210 PASS; MECHV2 block verified rendering in the
  QEMU correctness boot, within-boot EXACT=true). `BOOTLOG.TXT` pre-boot
  size **66,743 bytes**. Budget ~30 s extra for the 30 MECHV2 runs.
- L stick `EFI/BOOT/BOOTX64.EFI` (**UKI v4** — supersedes v3, restaged
  ~12:05 after its own qemu-test PASS): md5
  `a761bfc58bcdcdd840b9fc4949237a5f`, 27,406,336 B. v4 adds the
  **colskip_vs_incumbent A/B (3 captures) fed by the REAL captured
  BitNet-2B down_proj vectors** (artifacts .av1 in the initramfs; ordered
  variant byte-identical to the incumbent, so a win = zero-risk drop-in).
  v3 content unchanged (MECHV2L, cis_selftest digest `76985613c965f643`
  — same digest also printed under TCG, witness, membw, fused+bitplane).
  `BOOTLOG_LINUX_ARM.txt` pre-boot size **274,958 bytes**. Unmounted clean.
- Gates required before staging: merged-tree `cargo xtask boot-test` PASS (U),
  `scripts/linux-arm/qemu-test.sh` PASS (L), `scripts/check-efi-simd.sh`
  PASS (U binary), full `cargo test -p aegis-core` green.

## Explicitly not tonight

- Chained auto-reboot "next prototype" sequencing — one boot runs everything;
  reboot chains risk a wedged unattended session.
- Any BitNet-2B on the HP (4 GB RAM; M7 only there).
- E2 integer-vs-float perplexity — that runs on the dev box today, no stick
  needed; its verdict lands in the same evening report.
