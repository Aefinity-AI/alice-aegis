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

**Where a reviewer should push.** Both physical legs share one structural gap: the verifier
prints no CPU identifier, so the receipt proves what was computed, not unassisted which box
computed it. For the Dell leg, a differing firmware memory map gives log-internal evidence; for
the HP leg, no such internal signal exists, and the claim — including its AVX2/SSE2 corollary —
rests on the operator's physical presence at boot (A33, A34). We record this limitation here,
next to the claims that need it, rather than only in the paper's limitations section.
