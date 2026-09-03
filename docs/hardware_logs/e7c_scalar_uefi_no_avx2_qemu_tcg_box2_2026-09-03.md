# Provenance note — E7c scalar_only build, QEMU/TCG no-AVX2 verify, 2026-09-03

Companion to: `e7c_scalar_uefi_no_avx2_qemu_tcg_box2_2026-09-03.txt` (raw
`RESULT.txt` from `claudius-maximus` leg `e7c-scalar-uefi`, run via
`bin/cm leg push cm-box2 e7c-scalar-uefi legs/e7c-scalar-uefi.sh`, harvested
to `claudius-maximus/state/legs/box2/e7c-scalar-uefi/`).

## What this is

Prep for E7c (box2 as a third physical x86-64 CIS-1 iron leg), NOT the
bare-metal boot itself — this run is entirely QEMU/TCG, on box2 as compute
host only (no iron boot; see `docs/IRON_LEG_BOX2.md` for that procedure,
which is explicitly deferred to Justin at the power button).

Commit under test: `66c91d8` on branch `cm/scalar-build-no-avx2`
(`aegis-core`'s `scalar_only` feature — every `#[target_feature(enable =
"avx2", ...)]` kernel body in `ops.rs` compiled out; `ops_bitplane`/
`ops_colskip` excluded entirely).

## Results (all from the one leg run, single commit, no cherry-picking)

1. **Build**: `scripts/build_scalar_uefi.sh --qemu-test` succeeded.
   `aegis-uefi.efi` = 347,648 bytes (hardfloat/SSE2-baseline target,
   `scalar_only` feature). For comparison, the same commit's *standard*
   build (AVX2 kernels present, per A46) was 364,544 bytes — the scalar
   binary is ~16,896 bytes smaller, consistent with ten fewer kernel bodies
   compiled in, though byte-count is not the point of this leg.
2. **objdump census** (independent second pass, not just the build script's
   own gate): `ymm=0 vfmadd=0 vpmadd=0` — zero AVX2/FMA instructions
   anywhere in the binary. `xmm=4275` — SSE2-class code is present and
   plentiful, confirming this is "no AVX2" and not "no SIMD/no hardfloat"
   (a soft-float regression would show `xmm=0` too, not just `ymm=0` —
   see `scripts/check-efi-simd.sh`'s inverse gate and the MECH-A1
   regression it exists to catch).
3. **CIS-1 selftest digest**: `aegis-linux/examples/cis_selftest` (host
   build, same commit) printed `CIS_SELFTEST digest=76985613c965f643
   ALL_PASS=true` — the pinned digest (`docs/CIS-1_SPEC_v1.0.md`,
   `CHALLENGE.md`), all 14 A/B sections PASS. Expected and unsurprising:
   `cis.rs`/`cis_infer.rs`/`cis_attn.rs` have zero x86 intrinsics and are
   untouched by `scalar_only` (lib.rs's module-gate doc comment), so this
   is a confirmation that the change is scoped as claimed, not a new
   result — but it is a *run*, not an assertion by inspection.
4. **Kit image**: `scripts/make-kit-image.sh` against the scalar `.efi`,
   default 64 MB (`AEGIS_KIT_SIZE_MB` unset) — 67,108,864 bytes, same M7
   trio + `tests/golden/witness_v1_m7_once64.receipt` payload CA3 used.
5. **QEMU boot, TCG, explicitly no AVX2**: `-machine q35,accel=tcg -cpu
   qemu64,+sse4.2` (NOT `-cpu max`/`host` — chosen specifically to emulate
   a CPU without AVX2, matching the task's ask). `isa-debug-exit` rc=33
   (engine success code 0x10). `BOOTLOG.TXT` on the image:
   ```
   CPUID: vendor=AuthenticAMD brand="QEMU Virtual CPU version 2.5+" family=15 model=107 stepping=1 feats=<avx2:0,fma:0,sse2:1>
   STAGE V: witness verify PASS — VERIFY PASS — this machine reproduced all 64 decode steps' full logit vectors bit-for-bit, with no OS underneath
   ```
   The `CPUID` line's own `feats=<avx2:0,...>` is the emulated CPU
   self-reporting no AVX2 — corroborating the `-cpu qemu64,+sse4.2` choice
   independent of the objdump evidence.

## What this does and does not prove

Proves: a binary provably free of AVX2/FMA instructions (source: objdump,
this run) boots under UEFI and passes full witness verification (64 decode
steps, chained SHA-256 over full logit vectors) on an emulated CPU that
self-reports no AVX2/FMA. This is the load-bearing claim `docs/
IRON_LEG_BOX2.md` builds the iron procedure on top of.

Does NOT prove: anything about real box2 iron. QEMU/TCG models neither cache
hierarchy nor real firmware quirks (four are catalogued in `references/
uefi-boot.md`) — Rule A: this is a correctness/identity result, and Rule A
also means no timing was taken or is quotable from this run (box2 was the
build/QEMU **host**, not a timing subject; nothing here contradicts
`state/BOXES.md`'s "NO timing numbers" note for box2). The physical boot on
box2 iron is still open — procedure in `docs/IRON_LEG_BOX2.md`.
