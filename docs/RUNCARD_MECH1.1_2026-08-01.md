# RUNCARD — MECH v1.1: hardfloat confirmation boot (Dell i5-5200U)

**Why this boot exists:** MECH v1 (ledger A13) was measured on a binary that
turned out to be **soft-float** — the stock `x86_64-unknown-uefi` target
scalarized every f32 op to library calls (ledger A14, census:
`docs/hardware_logs/mecha1_softfloat_census_2026-08-01_v2.log`, xmm=0 ymm=0
vfmadd=0). H1/H2 conclusions were drawn on a 119–250×-slow workload. This boot
re-runs the identical MECH protocol on a **hardfloat** binary to (1) confirm
the MECH-A1 root cause by prediction, (2) re-measure the H1/H2 shares on the
real workload, and (3) capture the first bare-iron H3 MTRR/PAT dump.

**Machine:** Dell Inspiron 15 (i5-5200U, Broadwell-U, AVX2). Same stick, same
M7 artifacts as MECH v1 (md5 pins in `oscost_PREREGISTRATION_2026-07-30.md`).

**Binary under test:** `aegis-uefi.efi` built by `./aegis-uefi/build_hardfloat.sh`
(target `x86_64-uefi-hardfloat.json`, features `-mmx`), from branch
`hybrid-engine` @ `e85e77e` (post-merge of the three gauntlet lanes; the new
aegis-core kernels are dead code in the unikernel — census identical to the
pre-merge build).
md5 `322efa42b24f0f9e42684d5fa120d125`, 402,432 B.
`scripts/check-efi-simd.sh` verdict: **PASS** (xmm=6855 ymm=433 vfmadd=210).
New in this binary vs MECH v1: H3 MTRR/PAT probe block, `H3 buffers:` line,
`STAGE 6 … codegen=hardfloat|softfloat` banner (in-log staging tripwire).
QEMU/OVMF `cargo xtask boot-test`: correctness only, Rule A — no number from
that run exists.

---

## Preregistered predictions (written before the boot)

- **P1 (A14 confirmation, primary):** MECH LOUD/QUIET `process_intent` falls
  from 1.65–1.67G ticks/token to the band implied by A13's gauntlet
  denominators, **~6.6–14M ticks/token**. If it does NOT, A14's mechanism is
  incomplete and MECH-A1 reopens.
- **P2 (H1 revisited):** the absolute console cost (LOUD−QUIET) stays in the
  0.31–0.84M ticks/token range — it is firmware console work, independent of
  decode speed. Against a ~6.6–14M denominator that is now a **2–13% share**:
  H1 was "dead" only on the inflated denominator. A nonzero H1 share here is
  EXPECTED and does not contradict A13 (which already used hardfloat
  denominators for its ≤6.0% bound).
- **P3 (H2 replication):** QUIET2/QUIET ≈ 25/27 = 0.9259 replicates on the
  hardfloat workload (the turbo bin is clock arithmetic; workload-independent
  so long as the path stays compute-bound at these clocks).
- **P4 (H3 first data):** the probe prints MTRRCAP, MTRR_DEF_TYPE, variable
  MTRRs, IA32_PAT, and a per-buffer effective-type verdict for image / MODEL.SAF
  / EMBED.BIN / VOCAB.BIN / heap chunks. Expectation if H3 is FALSE: everything
  WB. Any engine buffer at UC/WT/WC is the Band-3 residual smoking gun.
- **P5 (bit-exactness, Rule D):** within-boot LOUD/QUIET/QUIET2 responses
  byte-identical per prompt (greedy). Cross-build identity vs the MECH v1
  (soft-float) responses is EXPECTED if soft-float lowered
  `_mm256_fmadd_ps` to fused `fmaf` libcalls (single rounding, same as
  vfmadd) — but a cross-build mismatch is DIAGNOSTIC (records that soft-float
  double-rounded), not a failure. Only within-boot mismatches fail the boot.

## Stick prep (Claude, Chromebook) — status: **DONE 2026-08-01 ~05:57**

Staged after BOTH QEMU correctness gates passed on the merged tree (boot-test
exit 33, 51 coherent tokens). Verified on-stick post-copy:
`EFI/BOOT/BOOTX64.EFI` md5 `322efa42b24f0f9e42684d5fa120d125`, 402,432 B
(was `431ff3a8…` = MECH v1 soft-float, archived in `artifacts/`).
`BOOTLOG.TXT` pre-boot size **53,439 bytes**. Unmounted clean.

Original checklist (all items executed as written):

U stick = the **57.8 GB card**, boot runtime on its **ALICE_M7** partition.
Identify by `lsblk -o NAME,SIZE,LABEL`, never by device letter.

1. Mount ALICE_M7; verify current `EFI/BOOT/BOOTX64.EFI` md5 ==
   `431ff3a8…` (the MECH v1 soft-float binary; archived copy already at
   `artifacts/BOOTX64_MECH_softfloat_2026-08-01_431ff3a8.EFI`).
2. Copy the new binary; `md5sum` ON-STICK must read
   `322efa42b24f0f9e42684d5fa120d125`.
3. Verify MODEL.SAF / EMBED.BIN / VOCAB.BIN still match the prereg pins.
4. Pre-boot `BOOTLOG.TXT` size: **53,439 bytes** (verified RO-mount
   2026-08-01 ~05:32) — post-boot content = everything past this offset.
5. `sync`, unmount clean.

## User action

Boot the Dell from the U stick, hands off, wait for `==== MECH DONE ====`
(console goes quiet after the QUIET2 responses), power down, bring the stick
back. One boot is sufficient.

## Analysis plan (after the stick returns)

- Extract post-offset BOOTLOG.TXT → new file
  `docs/hardware_logs/mech11_U_BOOTLOG_<date>.txt` (append-only, new file).
- Score P1–P5. Update ledger A14 (⬜ → verdict) and A13 (H1 share on honest
  denominator now measured, not derived). verify-figures before any ledger edit.
- H3: decode the MTRR/PAT lines with `aegis-uefi/src/mtrr_decode.rs` logic;
  if any engine range is non-WB, H3 graduates from suspect to mechanism and
  gets its own fix experiment (set MTRR/PAT before weight load).

## Explicitly out of scope for the U-stick boot

- Kernel-candidate A/Bs (fused dual/tri, bitplane): hosted benches — they ride
  the **L stick** instead (below). The bitplane AVX2 kernels do not even
  compile for the UEFI soft-float target (LLVM legalization abort — see
  ops_bitplane module docs).
- ReLU² column-skip: prototype not built yet (ledger A15 banks the
  justification; task #27).
- Any tok/s claim for BitNet-2B — this is the M7 MECH protocol.

---

# SECOND BOOT — L stick v2: kernel A/B on idle bare iron

**L stick** = the **14.6 GB ALICE_UEFI stick** (single FAT partition, marker
file `AEGIS_LINUX_ARM.tag`). Its UKI boots a minimal Linux from initramfs,
clears bd-PROCHOT via msrtool, sets performance governors, re-runs the
OS-cost gauntlet (free replicate of the L arm), and NEW in v2: runs each
kernel-candidate bench **3×** on the idle machine —

- `fused_vs_sequential` (default 7 interleaved order-swapped reps/run):
  fused dual (SwiGLU 2×6912×2560), fused tri GQA (2560/640/640×2560), M7
  shapes. Expectation from disassembly: may settle NEGATIVE (GPR spill).
  Either verdict closes the "dual matvec" question with a measured answer.
- `bitplane_vs_lut` (default 5 reps/run): incumbent LUT+FMA vs bitplane
  variants (i) byte-identical and (ii) dual-accumulator. Armchair prior
  0.85–1.0× is printed in the bench header so the result can contradict it.

Both benches print their own clock-state block (TSC nominal +
effective/nominal ratio) and assert byte-identity gates before AND after the
timing loop; everything appends to `BOOTLOG_LINUX_ARM.txt` on the stick.

**UKI to stage:** `scripts/linux-arm/out/BOOTX64.EFI`
md5 `8a70d08ed7b3ff31ec742ff1cf0dcbdd`, 22,937,600 B (v2, benches included,
each self-naming via AEGIS_MACHINE).
Gate before staging: `scripts/linux-arm/qemu-test.sh` **PASS** (2026-08-01
~06:00 — full init incl. both bench sections ran to completion, all identity
gates true, clean poweroff; TCG numbers in that log are Rule-A meaningless).

**Staged 2026-08-01 ~06:01:** on-stick `EFI/BOOT/BOOTX64.EFI` md5 verified
`8a70d08ed7b3ff31ec742ff1cf0dcbdd` (was `68db167e…` = v1 UKI).
Pre-boot `BOOTLOG_LINUX_ARM.txt` size: **242,955 bytes** — post-boot content
= everything past this offset. Unmounted clean.

**User action:** boot the Dell from the L stick after (or before — order
does not matter) the U stick. Hands off; it powers itself off when done.

**Analysis:** extract post-offset BOOTLOG_LINUX_ARM.txt → new
`docs/hardware_logs/` file; ledger rows for fused and bitplane verdicts
(≥3 runs each satisfies the lut_mpgemm-findings admissibility bar);
negative results are deliverables.
