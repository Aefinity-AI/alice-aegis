# IRON_LEG_BOX2.md — booting the scalar (no-AVX2) kit on box2 iron

E7c: box2 (`aefinity-box2`, Celeron N4020, Gemini Lake — the same silicon
model as the HP Stream that produced ledger row A34) is a **third independent
physical x86-64 machine**, distinct from the Dell Inspiron 15 (A33, i5-5200U,
AVX2) and the crosvm/QEMU legs already in the digest jury (A25). It has real
UEFI firmware and no AVX2/FMA. This doc is the procedure to boot the
**scalar-only** kit image on it, for whenever Justin is next at the power
button — it does NOT boot box2 bare-metal itself (that is explicitly out of
scope for this prep task; box2 is a live remote worker, see below).

## Status: QEMU proof done (2026-09-03), iron step open

Prerequisite work (this session, E7c prep) is DONE and measured:
`scripts/build_scalar_uefi.sh` (branch `cm/scalar-build-no-avx2`, commit
`66c91d8`) builds an `aegis-uefi.efi` (347,648 bytes) with every AVX2/FMA
kernel compiled out — objdump-verified `ymm=0 vfmadd=0 vpmadd=0` (`xmm=4275`,
so this is "no AVX2", not "no SIMD" / a soft-float regression). It has been
kit-imaged (`scripts/make-kit-image.sh`, 67,108,864-byte image, same M7 trio
+ golden receipt CA3 used) and booted under QEMU **TCG**, `-cpu
qemu64,+sse4.2` (explicitly no avx2, chosen over `-cpu max`/`host`
specifically to emulate a CPU without AVX2) — `isa-debug-exit` rc=33,
`BOOTLOG.TXT` shows `CPUID: ... feats=<avx2:0,fma:0,sse2:1>` and `STAGE V:
witness verify PASS`. Raw log + provenance:
`docs/hardware_logs/e7c_scalar_uefi_no_avx2_qemu_tcg_box2_2026-09-03.{txt,md}`
(leg: `claudius-maximus/legs/e7c-scalar-uefi.sh`, harvested to
`claudius-maximus/state/legs/box2/e7c-scalar-uefi/`). The CIS-1 selftest
digest was also reproduced on this commit: `CIS_SELFTEST
digest=76985613c965f643 ALL_PASS=true` (expected — the reference chain is
untouched by `scalar_only` — but run, not assumed).

**What remains is the physical step below** — QEMU/TCG proves the binary and
the boot logic; it does not and cannot stand in for real iron (Rule A).

## Why this needs Justin, not a script (defect-#21 class risk)

box2 is not a spare test box sitting idle waiting for a stick — it is a
**live, remotely-managed Claudius Maximus worker** (WiFi, `cm` user, runs
legs). It has no dedicated `ALICE_UEFI` partition the way box1 does
(`state/BOXES.md`: box1 role includes `ALICE_UEFI = sda2`); box2's only disk
is the internal eMMC running Debian (`mmcblk0p1` = `/boot/efi`, `p4` = `/`).
Booting the unikernel means booting from an **external USB stick**, and:

- a one-shot `efibootmgr --bootnext` into a removable USB entry is not
  guaranteed to survive every firmware's quirks unattended (this is exactly
  the class of thing ledger's four hardware-level UEFI bugs, referenced in
  the alice-unikernel skill's `references/uefi-boot.md`, warn about — XHCI
  DMA limits, FAT32 8.3 matching, enumeration order, `AnyPages`);
- the unikernel halts at the end of verifier mode ("Verification complete.
  It is safe to power off.") — nothing brings box2 back to its worker role
  automatically; someone has to pull the stick and power-cycle it back into
  Debian;
- if the boot menu doesn't cooperate (wrong key timing, stick not
  recognized first try), box2 needs a human to intervene, not a remote
  session watching an SSH connection that just dropped because the machine
  is now running firmware, not Linux.

This is the same class of risk the FIRST LIGHT session on box1 already hit
and resolved by requiring Justin at the power button (memory: reset
DEFECT #21 → bare-metal runs need Justin at the power button). Same rule
here: **do not attempt this remotely or unattended.**

## The one-command procedure

Run on box2 itself once Justin is physically present with a spare USB stick
(≥64 MB; the scalar kit image defaults to 64 MB, same as `make-kit-image.sh`'s
default — see A46 for how far that can shrink if it matters later).

1. **Identify the stick — never blindly grab a `/dev/` letter.** Before
   inserting, run `lsblk` to record what's already there; insert the stick;
   run `lsblk` again and diff — the new device is the stick. This is a human
   read-and-confirm step, not a script guess.

2. **Write the image** (already built and sitting in the leg's output,
   `~cm/legs/e7c-scalar-uefi/kit_scalar.img` on box2 from the QEMU-proven
   prep run — reuse that exact file; do not rebuild on the spot):

   ```
   sudo dd if=~cm/legs/e7c-scalar-uefi/kit_scalar.img of=/dev/<CONFIRMED_STICK> bs=4M status=progress conv=fsync
   sync
   sudo dd if=/dev/<CONFIRMED_STICK> bs=4M count=16 2>/dev/null | md5sum   # spot-check readback
   ```

   The image is a complete FAT32 filesystem (`make-kit-image.sh` output) —
   `dd` the whole image to the raw block device, no partitioning step.

3. **One-shot boot, no permanent reorder.** box2's current `efibootmgr -v`
   already carries a generic `Boot2001* EFI USB Device` entry (not first in
   `BootOrder` — `debian` boots first today). Two options, either is a single
   command:

   - **BootNext (preferred, truly one-shot):**
     `sudo efibootmgr --bootnext 2001 && sudo reboot`
     — the firmware boots the USB entry exactly once, then reverts to the
     saved `BootOrder` (`debian` first) on the *next* reboot, without editing
     `BootOrder` at all.
   - **Boot-menu key (fallback if BootNext misbehaves on this firmware):**
     reboot and hit the firmware's boot-menu hotkey (F9/F12/Esc — check the
     splash) to interactively pick the USB stick for this boot only.

   Either way: **Secure Boot must be off** (per `make-kit-image.sh`'s
   README.TXT baked onto the stick) and this is the "one command" — reboot
   itself is the boot.

4. **Read the result off the screen and the stick**, not off SSH — box2's
   network stack is gone the moment firmware takes over. The unikernel's
   console prints `[AEGIS] RECEIPT.TXT found — witness verifier mode` then
   the verdict; the authoritative record is `BOOTLOG.TXT` on the stick
   (`STAGE V` is written there, not to serial/console-only — see
   `aegis-uefi/src/main.rs`).

5. **Power off, pull the stick, power back on** to return box2 to its
   Claudius Maximus worker role. Then, from penguin:

   ```
   dd if=/dev/<CONFIRMED_STICK> of=/tmp/box2_iron_BOOTLOG_readback.img bs=4M count=16   # or mcopy just BOOTLOG.TXT off it on box2 before pulling
   ```

   (In practice: `mcopy -n -i kit_scalar.img ::/BOOTLOG.TXT box2_iron_bootlog.txt`
   run on box2 before removing the stick is simpler than re-imaging it — do
   that instead if convenient.) Copy the raw `BOOTLOG.TXT` into
   `docs/hardware_logs/box2_n4020_scalar_kit_iron_verify_bootlog_<date>.txt`
   (append-only, Rule C — new file, never edit an existing log) and write a
   companion provenance note the way `hp_n4020_kit_iron_verify_provenance_
   2026-08-08.md` did for the original HP leg.

## What RESULT this should produce

Expected `BOOTLOG.TXT` tail, by direct analogy with A34 (the *standard*
build's PASS on this same silicon in 2026-08-08) but with two upgrades this
prep added:

```
STAGE 1: boot volume opened, AVX enable attempted
STAGE 2: sizes OK model=... embed=... vocab=...
STAGE 3: tensor memory allocated
STAGE 4a: MODEL.SAF loaded
STAGE 4b: EMBED.BIN loaded
STAGE 4c: VOCAB.BIN loaded
STAGE 4d: RECEIPT.TXT loaded — verifier mode armed
STAGE 5: working heap online
CPUID: vendor=GenuineIntel brand="Intel(R) Celeron(R) N4020 CPU @ 1.10GHz" family=... model=... stepping=... feats=<avx2:0,fma:0,sse2:1>
STAGE V: witness verify PASS — VERIFY PASS — this machine reproduced all 64 decode steps' full logit vectors bit-for-bit, with no OS underneath
```

Two things this prep changes versus the A34-era procedure:

1. **CPUID line now present.** A34's provenance note had to state plainly
   that "verifier mode prints no CPUID" and fall back to operator physical
   witness for machine attribution. That gap is closed — `main.rs` now logs
   `vendor`/`brand`/`family`/`model`/`stepping`/`feats` right before the
   `STAGE V` line specifically so a future PASS self-identifies the CPU that
   produced it (see the comment at that call site: "Attribution (ledger
   A33/A34): the previous log carried no CPUID at all..."). A box2 PASS
   should show `feats=<avx2:0,fma:0,sse2:1>` — the CPU's own capability
   report, independent of which build produced the binary.
2. **The scalar build makes the "SSE2 path executed" claim load-bearing
   rather than inferential.** A34's PASS on the standard (AVX2-present,
   runtime-dispatched) binary could only *argue* the scalar path ran ("had
   the AVX2 kernels executed... the boot would have died on #UD" — a
   counterfactual, not a fact about the binary). A PASS on the
   `scalar_only` binary built here needs no such argument: `objdump` on the
   exact `.efi` staged to the stick already proves (this session, see
   `RESULT.txt`) that no AVX2/FMA instruction exists anywhere in it. The
   iron PASS becomes evidence about the receipt's portability, with zero
   inference about what path executed.

A FAIL, or any `rc`/console output other than the above, is a real finding —
report it as-is (Rule D / measurement integrity: do not paper over a
divergence). Nothing about this procedure is allowed to produce a number;
it's an identity/correctness result only (Rule A) — no timing is taken or
implied by "how long the boot took."

## How this feeds ledger attribution (A33/A34/A46 lineage)

Once the physical boot happens and the BOOTLOG + provenance note exist under
`docs/hardware_logs/` (Rule C: new files, never edits), the next
`program/RESEARCH_LEDGER.md` row should:

- Cite this as a **third independent physical machine** in the digest/kit
  jury: Dell i5-5200U (A33, AVX2 path) + HP/box2 N4020 (A34, inferred scalar
  path, standard build) + **box2 N4020 again, this time on a build that
  proves zero AVX2 by construction** (this leg) — same silicon as A34, but a
  strictly stronger claim (objdump-provable vs. counterfactual).
- Carry forward A46's box2 provenance chain: A46's footprint bisection ran
  on this same box2 under QEMU **TCG** (not iron); this leg is the iron
  follow-up A46 flagged as open ("Follow-up: repeat on box1 (KVM) or a real
  stick before quoting in reviewer-facing prose").
- State the CPUID line's `feats=<avx2:0,fma:0,sse2:1>` explicitly, closing
  A34's stated attribution limitation for good on this box.
- Link both the BOOTLOG raw log and this doc's procedure as the two legal
  parents (Rule B: instrument + derivation) for any claim like "reviewed the
  CIS-1 paper's third independent x86-64 codegen path, verified on real,
  non-AVX2 iron, from a binary proven by objdump to contain no AVX2
  instructions."
