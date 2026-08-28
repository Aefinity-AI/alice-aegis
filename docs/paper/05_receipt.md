# 5. The decode receipt and bare-metal verification

This section, like §4, carries no timing or throughput number; every figure below is an identity
claim (Rule A).

**Receipt format (A32).** The golden receipt `tests/golden/witness_v1_m7_once64.receipt` binds
SHA-256 hashes of all three model artifacts, the 64 generated token ids, the token digest
`67e8c0a96abc04e1`, and a chained SHA-256 commitment over every decode step's full i64 logit
vector — chain `aee25b770bd7b22e…`. It was minted on the crosvm dev host (i5-10210U). This is not
a summary digest: it is the entire integer state sequence, 64 steps deep, each step's complete
logit vector folded into the chain. Every field the receipt binds is listed above exactly as the
ledger row lists it — the three artifact hashes, the token ids, the token digest, and the
step-by-step logit chain — because the receipt's value is that a verifier can recompute each of
these independently and compare, not that it asserts a conclusion.

**Verification crosses the ISA boundary in public CI (A32).** On 2026-08-08, the same receipt
was replayed from source on a GitHub `ubuntu-24.04-arm` hosted runner, public run `31249589879`,
snapshot `ce93bbb`: `artifacts 3/3`, the token digest and chain `aee25b770bd7b22e…` reproduced
exactly, and the harness printed `VERIFY PASS — replay reproduced 64 tokens, the token digest,
and the full logit chain bit-for-bit`. The same run's x86-64 job verifies the identical receipt,
and `arm-digest.yml` now pins both legs on every push. This upgrades the op-level digest's
cross-ISA claim (§4, A28) to a full decode trajectory — 64 steps times the complete logit vector,
hash-chained — rather than a single summary value. As with the other cloud-runner legs, the
machine is named as precisely as the platform allows; this is an identity artifact only, and no
timing is quoted or quotable from it (A32).

**Physical iron, no operating system: the Dell leg (A33).** On 2026-08-08 the Provable AI Kit
stick booted on the Dell Inspiron 15 (i5-5200U, Broadwell-U) with no OS present and re-derived
the golden receipt bit-for-bit. The operator was physically present; boot was via F12 with
Secure Boot off. The firmware appended to BOOTLOG.TXT: `STAGE V: witness verify PASS — VERIFY
PASS — this machine reproduced all 64 decode steps' full logit vectors bit-for-bit, with no OS
underneath`. The receipt on the stick is md5-identical to the golden (`87c45bdd…`); the boot
payload is the QEMU-proven `aegis-kit-iron.img` (`320e1918…`), and the stick's readback was
verified bit-identical before boot. This completes the chain crosvm-mint → QEMU → public CI
x86-64 → public CI aarch64 (A32) → physical iron, ring 0. We state the attribution limitation as
the ledger states it, not deferred to §8: the new BOOTLOG entry is attributed to the Dell because
it was appended after the baked-in QEMU entry and carries a different firmware memory map
(EMBED.BIN at `0x3AC000` versus QEMU's `0x1780000`) — but verifier mode prints no CPUID, so this
is log-structural evidence, not a hardware identifier (A33).

**Physical iron, no operating system: the HP leg (A34).** On the same date, the identical stick
and identical receipt were booted on the HP Stream (Celeron N4020, Gemini Lake) for the
first-ever unikernel boot on that machine: VERIFY PASS, payload unchanged since the Dell leg.
BOOTLOG.TXT gained exactly one new entry (diffed against the banked Dell-era log): `VERIFY
PASS — this machine reproduced all 64 decode steps' full logit vectors bit-for-bit, with no OS
underneath`. The scalar-path corollary matters: the N4020 lacks AVX2, so a PASS on it implies the
SSE2 scalar path executed, since AVX2 would fault with #UD — the golden receipt has now been
re-derived bit-for-bit through two disjoint kernel code paths on iron. The attribution limitation
is stated as plainly as the ledger states it: the new entry's firmware memory map is identical to
the Dell entry's, so in-log evidence does not discriminate the two boxes, and attribution of this
entry rests on operator witness (A34). The row also records a firmware finding made incidentally
on this leg: that the N4020's UEFI boots the unikernel at all was previously unknown before this
boot (A34).

**A third, standalone implementation verifies the receipt without the engine (A37).** Everything
above shows the receipt can be produced and replayed by machines running `aegis-core` itself, on
either ISA. A receipt that only the reference engine can check is not yet useful to a third party
— the verifier is what makes it useful, and it must not depend on the code being verified.
`cis-verify` is a separate crate, written from the spec and the receipt format rather than as a
fork of `aegis-core`: zero external runtime dependencies, no dependency on `aegis-core`, no
`unsafe`, `no_std`+alloc at its core. On the same dev host, it reproduces the pinned op-level
digest (`CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true`), both pinned table digests (the exp
LUT and RoPE constants), and the token-level decode digest
(`CIS_DECODE digest=67e8c0a96abc04e1 prompt_toks=4 gen_toks=64 mode=fullint`) on first attempt,
then verifies the golden receipt `tests/golden/witness_v1_m7_once64.receipt` end to end — all six
checks (receipt parse, the three artifact hashes, prompt tokenization, the 64-step token-id
sequence, the cis-digest, and the witness chain) — and prints `VERIFY PASS` in about 1.4 seconds.
Its tamper tests fail by naming the corrupted field (token id, chain, model/vocab hash, or
receipt parse), not by silently passing. Honest scope, stated plainly: this was an LM agent
transcribing the spec with the reference source visible, not a clean-room reimplementation in the
sense A31 uses that term — it is evidence that the spec and the receipt format are re-implementable
without the engine's SIMD/dispatch code, not an independent third-party audit (§8, A37).

**Both verifiers cross the ISA boundary, on both receipts (A38, A39).** `cis-verify`'s standalone
verification (A37) now also runs on aarch64: built and run on the GitHub `ubuntu-24.04-arm`
runner, it reproduces `CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true`, passes its full suite
(81 unit + integration tests), and prints `VERIFY PASS` on the x86-minted M7 golden receipt
`tests/golden/witness_v1_m7_once64.receipt`; the x86-64 job in the same run prints the identical
lines. Both are now standing CI gates in `arm-digest.yml` (A38). The same pattern holds at
BitNet-2B production scale: the receipt minted on x86 (`tests/golden/witness_v1_bitnet2b_once64.receipt`,
cis-digest `cab11400d737ac4a`, chain
`917ddf5fea9a848876ddb527d5d5216607637201d6514b94563977009558af32`, bound to artifact
`facb3597…`) verifies bit-for-bit on the GitHub `ubuntu-24.04-arm` runner by two independent
implementations at once: the reference `cis_witness verify` prints `VERIFY PASS`, and the
standalone `cis-verify` (A37/A38) independently prints `VERIFY PASS` on the same receipt; the
x86-64 job prints the identical lines. This is a standing CI gate, `bitnet2b-receipt.yml`, on
every push to `main` (A39). Together, A38 and A39 close the receipt-side counterpart of §4's A39
digest result: the same receipt, at production scale, checked by two independent implementations,
on both ISAs, on every push.

**The 2B receipt, re-derived by the unikernel with no OS, under QEMU (A40).** On 2026-08-27,
`aegis-uefi.efi` booted under QEMU/OVMF (TCG, `-cpu max -m 2048`) from a 1024 MiB FAT32 kit image
carrying the BitNet-2B artifacts — `MODEL.SAF` (522,831,917 B), `EMBED.BIN` (257,310,720 B),
`VOCAB.BIN` (1,759,936 B) — and the golden BitNet-2B receipt. BOOTLOG.TXT records `STAGE 2: sizes
OK model=522831917 embed=257310720 vocab=1759936`, `STAGE 4a`–`STAGE 4d` loading each artifact and
the receipt, `STAGE 5: working heap online`, `CPUID: vendor=AuthenticAMD brand="QEMU TCG CPU
version 2.5+"`, and `STAGE V: witness verify PASS — VERIFY PASS — this machine reproduced all 64
decode steps' full logit vectors bit-for-bit, with no OS underneath`. The serial console shows
`artifacts: 3/3 hashes match`, with receipt and local cis-digest `cab11400d737ac4a` and chain
`917ddf5fea9a8488…` agreeing exactly — the same digest and chain prefix as the x86/aarch64 CI
verifications above (A39). The loader, physical allocator (one contiguous ~732 MB claim), and DMA
bounce path handled 782 MB of assets unchanged — no engine or firmware code change was needed;
only the kit-image packaging script's 64 MiB size constant needed an `AEGIS_KIT_SIZE_MB` override.
This extends A39 (2B receipt cross-ISA in CI) onto the boot path, and extends the M7-scale iron
result (A33, A34) to BitNet-2B — but under QEMU/TCG emulation only: correctness/identity evidence
(Rule A), no timing, and no physical-machine claim. A third physical machine (E7c) is staged for
that leg.

**Where a reviewer should push.** Both physical legs share one structural gap: the verifier
prints no CPU identifier, so the receipt proves what was computed, not unassisted which box
computed it. For the Dell leg, a differing firmware memory map gives log-internal evidence; for
the HP leg, no such internal signal exists, and the claim — including its AVX2/SSE2 corollary —
rests on the operator's physical presence at boot (A33, A34). We record this limitation here,
next to the claims that need it, rather than only in the paper's limitations section.
