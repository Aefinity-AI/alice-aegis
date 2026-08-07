# RUNCARD — MECH boot (H1 console / H2 turbo-bin), Dell i5-5200U

**Purpose:** one hands-off boot that decomposes the Band-3 result
(pooled Δ_OS −39.3%, minimal Linux faster — see
`docs/OSCOST_MECHANISM_REPORT_2026-07-31.md`) into its suspected mechanisms:

- **H1** — synchronous UEFI console writes inside the timed decode loop.
  Measured as LOUD (per-token console, replicates the protocol) vs QUIET
  (tokens buffered, printed after timing stops), same prompts, same boot.
- **H2** — idle core hot-parked by firmware costs the 1-core turbo bin.
  Measured as QUIET vs QUIET2 after the APs are sent to MWAIT-C6
  (`clock %` should rise from ~113% toward ~123% if the bin unlocks).

**Machine:** Dell Inspiron 15 (i5-5200U, Broadwell-U) — same box as the
2026-07-31 protocol. **Nothing else changes:** same stick, same M7 artifacts
(md5s pinned in `oscost_PREREGISTRATION_2026-07-30.md`), same STARTUP.NSH.

**Binary under test:** `aegis-uefi.efi`, **md5 `431ff3a8246559a0b12e6f640dd86c0a`**,
453,120 bytes, built 2026-07-31 from branch `hybrid-engine` (MECH v1 block +
AP-park; production build, no `qemu-test` feature → MECH_MAX=256).
QEMU/OVMF correctness gate: `cargo xtask boot-test` — **correctness only, Rule A;
no number from that run exists.**

---

## 1. Stick prep — **DONE 2026-07-31 (Claude, Chromebook)**

The U stick is the **57.8 GB card**; the boot runtime under test lives on its
**ALICE_M7** partition. Identified by `lsblk -o NAME,SIZE,LABEL` — never by
device letter (letters swap between insertions).

Recorded state:

| Item | Value |
|---|---|
| GATE-1 binary (replaced) | md5 `afa1dd184e4ab8bbfd45d0258bd85af1`, 351,744 B, mtime 2026-07-29 13:57 |
| GATE-1 archive | `artifacts/BOOTX64_GATE1_2026-07-31_afa1dd18.EFI` (md5 verified after copy) |
| MECH binary now on stick | `EFI/BOOT/BOOTX64.EFI` md5 `431ff3a8246559a0b12e6f640dd86c0a`, 453,120 B — verified on-stick after copy |
| MODEL.SAF / EMBED.BIN / VOCAB.BIN | re-verified on-stick == prereg pins (`53235f…`, `2973158…`, `0330140…`) |
| **Pre-boot BOOTLOG.TXT size** | **44,612 bytes** — post-MECH content = everything past this offset |

Nothing else touched; partition unmounted clean. QEMU/OVMF correctness gate
passed the same day (exit 33, all nine MECH passes byte-identical —
correctness only, Rule A).

## 2. The boot (Dell — operator)

1. Insert the U stick. Power on; boot the stick (F12 boot menu if needed).
2. **Hands off.** After the banner, MECH runs by itself:
   LOUD×3 → QUIET×3 → AP-PARK → QUIET2×3. Expect roughly 10–20 minutes
   (up to 9 × 256 tokens at a few tok/s). The screen will show each pass;
   QUIET passes print their text only after their timing window closes.
3. It ends with `==== MECH DONE ====` and drops to the A.L.I.C.E. console.
   Type `/exit`. Power off. Remove the stick.

**If it hangs after the `MECH AP-PARK` lines** (MWAIT misbehavior would show
here): hold power 10 s. The LOUD/QUIET data are already on the stick —
`boot_log` writes through on every line — and the boot is still valid for H1.
Note the hang; do not retry before analysis.

## 3. Data pull (Chromebook — Claude)

1. Mount ALICE_M7 **read-only**. Verify the pre-boot prefix of BOOTLOG.TXT is
   untouched (md5 of first *N* bytes vs step 1.5).
2. Copy the new bytes to `docs/hardware_logs/mech_U_BOOTLOG_2026-08-01.txt`
   (new file — Rule C).

## 4. What the log must contain (analysis keys)

```
==== MECH v1 (2026-08-01): H1 console / H2 cpuidle turbo-bin ====
MECH MSR_TURBO_RATIO_LIMIT raw=0x................ (byte0=1C bin, byte1=2C bin)
MECH LOUD RESPONSE "hello alice": <text>
MECH LOUD "hello alice": N tokens, T ticks, t ticks/token, clock P%
... (×3 prompts, then QUIET ×3)
MECH AP-PARK: 4 logical processors, 4 enabled
MECH AP-PARK: dispatched MWAIT-C6 park to 3 AP(s)
THROTTLE ... (mech-postpark diagnostic block)
... (QUIET2 ×3)
==== MECH DONE ====
```

- **H1 share** = (LOUD − QUIET) / LOUD ticks/token, per prompt.
- **H2 share** = (QUIET − QUIET2) / QUIET ticks/token, per prompt, read
  together with the `clock %` change (113% → ~123% = 1-core bin unlocked;
  no change = parking didn't retire the cores from the bin count).
- **Bit-exactness gate (Rule D):** all nine RESPONSE lines for a given prompt
  must be byte-identical (greedy decode). Any divergence invalidates the run.
- Expected on this part: `MSR_TURBO_RATIO_LIMIT` byte0=0x1B (27), byte1=0x19 (25).
- If the log shows `MECH AP-PARK: skipped/failed ...`, QUIET2 still ran
  unparked: H1 numbers stand, H2 is unanswered this boot.

## 5. Decision rules (set before the run)

| Outcome | Read | Next step |
|---|---|---|
| H1 share ≈ Band-3 gap, H2 small | Gap was measurement construction (console in the stopwatch), not OS magic | Preregister v2 protocol with console excluded on BOTH arms; interleaved 10-boot U/L redo for the publishable Δ_OS |
| H1 + H2 large but gap remains | Real residual OS advantage exists | H3 next: UEFI memory caching attributes (MTRR/PAT audit) before any redo |
| QUIET ≈ LOUD (H1 tiny) | Console hypothesis dead — the gradient evidence was misleading | H3 immediately; treat −39.3% as unexplained, keep unpublishable |

The interleaved 10-boot redo is **deliberately not part of this runcard**: its
protocol depends on what MECH finds and gets its own preregistration first
(prereg discipline — the same rule that made the Band-3 result survivable).

---
*Numbers cited above (−39.3%, 113%, ratio 27/25) trace to
`docs/hardware_logs/oscost_{U,L}_BOOTLOG_2026-07-31.txt` and
`oscost_ANALYSIS_2026-07-31.md`. This runcard adds no measurements.*
