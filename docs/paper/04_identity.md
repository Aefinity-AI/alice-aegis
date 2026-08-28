# 4. Implementability and cross-ISA identity

This section is correctness evidence only; no throughput or latency figure appears below
(Rule A — identity legs carry no performance claim).

**A four-implementation digest jury (A25).** On 2026-08-02 the op-level self-test printed the
identical digest, `CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true`, on four distinct machines
and codegen paths: the HP Stream (Celeron N4020, Gemini Lake, SSE2-class) running bare iron on
minimal Linux, the Dell i5-5200U running bare iron across two cold boots, the crosvm dev host
(i5-10210U), and QEMU/TCG full-system emulation. The ledger states its own scope exactly, and we
repeat it rather than round it up: "Four distinct microarchitectures + codegen paths, one
bit-exact integer semantics. Correctness leg only; no performance numbers taken from it" (A25).

**The jury crosses its first ISA boundary (A28).** On 2026-08-07 the same op-level digest,
`76985613c965f643`, reproduced on a GitHub `ubuntu-24.04-arm` hosted runner (`uname -m`=aarch64
asserted in-job, CPU part `0xd49`, Neoverse N2 class, Azure southcentralus). The reference
semantics compiled untouched — `cis.rs` has zero imports, and `lib.rs` cfg-gates every
x86-intrinsic module — and reproduced the digest on first execution. The standing CI gate
`arm-digest.yml` now pins this digest on every push; a future mismatch fails the build, with the
ledger's own instruction to preserve the log as a finding, not fix it into silence. Because this
leg runs on a cloud runner, the machine is named as precisely as the platform allows; it stands
as an identity artifact only — no timing is quoted or quotable from it (A28).

Table 2 collects the digest jury; every cell traces to A25 or A28.

**Table 2 — the digest jury**

| Machine | Microarchitecture | Environment | Digest | Row |
|---|---|---|---|---|
| HP Stream (Celeron N4020) | Gemini Lake, SSE2-class | bare iron, minimal Linux | 76985613c965f643 | A25 |
| Dell Inspiron 15 (i5-5200U) | — | bare iron (2 cold boots) | 76985613c965f643 | A25 |
| crosvm dev host (i5-10210U) | — | crosvm (virtualized) | 76985613c965f643 | A25 |
| QEMU/TCG | — | full-system emulation | 76985613c965f643 | A25 |
| GitHub `ubuntu-24.04-arm` (Azure southcentralus) | Neoverse N2 class (CPU part 0xd49) | hosted CI runner | 76985613c965f643 | A28 |

**The specification is implementable from its text (A31).** On 2026-08-08 two independent
implementers — subagents under distinct personas — were given the v1.0.1 spec text as their sole
source, with reference-source access forbidden and, as far as the harness can establish, not
used. Each wrote a from-scratch scalar Rust implementation of the spec §5 operations (400 and 484
lines respectively, verified distinct by diff/md5). Both, linked against the verbatim-ported
public conformance harness, printed exactly `CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true`
— all 14 sections, first run after their own cargo-check. The v1.0.1 spec §5.4 erratum to RMSNORM-I's
Q2.30/2¹⁵ grids is what made this possible: both implementers hit RMSNORM-I's golden on first
attempt from the corrected prose. The ledger states the honest scope plainly, and we do not
soften it: the public conformance suite's golden vectors were visible to the implementers, and
both reports state the A8 goldens were needed to settle `from_ratio` edge cases the text leaves
open — so the proven claim is "implementable from spec text + public conformance suite, without
reference source," not from prose alone. The implementers' ambiguity reports converge on the same
spec gaps, queued as v1.0.2 errata: `from_ratio`'s rejection surface and S-selection bracket are
implicit, its RNE-to-2³¹ renormalization at S>0 is unstated, and QUANT-ACT/RMSNORM-I slice-length
preconditions, empty-ARGMAX, and `rne_div` den≤0 behavior are unspecified. Identity artifact only;
dev host (i5-10210U/crosvm) named, no performance figure exists or is claimed (A31).

**A NEON kernel bit-identical on real ARM silicon (A30).** On 2026-08-07, the same day as A28 and A29 and one day before the clean-room test,
`cis_neon::ternary_matvec_i8_neon` — using `vmulq_s8` as the ternary multiply, per-byte bit-pair
extraction through a `vqtbl1q_s8` code LUT, exact `vpaddlq`/`vpadalq` widening, the shared
precondition check, and the same −128-wrap-to-scalar-fallback contract as the x86 kernel — was
proven on the GitHub `ubuntu-24.04-arm` Neoverse runner, all three suites green in one job:
equivalence 5/5 (bit-identity versus the reference, including tails, code `11`, extremes, and row
independence), contract 6/6 (rejection surface identical to the reference on either side of the
block threshold), and mechanism 3/3 (exhaustive: every byte × 4 pair positions through the LUT,
all 256 activations × 4 codes through `vmulq_s8`, and widening-chain extremes). The `vmulq_s8`
wrap of `-128 × -1` is pinned identical to the x86 `vpsignb` behavior in both directions by test.
Both standing digest pins (A28's op-level digest and A29's token-level digest) held in the same
run, and the kernel is wired into `arm-digest.yml` as a standing gate on every push. The ledger is
explicit that no ARM performance number exists or is claimed here — Rule A requires named
physical hardware, and this runner is shared cloud; the throughput leg stays open until an ARM
board is on the bench (A30).

**Token-level identity across a complete decode (A29).** `cis_decode` runs the full pipeline —
tokenizer, embeddings, all 7 layers, integer attention, integer LM head, `argmax_i64` — on the
tracked M7 model (hidden 384, 8K vocab) and prints one digest over every token id, prompt
included: `CIS_DECODE digest=67e8c0a96abc04e1 prompt_toks=4 gen_toks=64 mode=fullint`. This
reproduced identically on x86_64 and aarch64 (GitHub `ubuntu-24.04` and `ubuntu-24.04-arm`, both
jobs green in one run), deterministic 3/3 on the dev host; the output is coherent TinyStories
text ("…there was a little girl named Lily…") — identical *and* fluent. Both digests are standing
CI gates in `arm-digest.yml`: the x86 job pins the tree, the ARM job pins the ISA, so a future
mismatch is attributable on sight. The pin's design choices are stated in the row and repeated
here rather than left implicit: greedy decoding only, EOS ignored (a fixed 64 tokens, with no
stop-condition dependence), and `CisMode::FullInt` only — the Hybrid arms are explicitly
x86-only, because their f32 legs carry no cross-ISA claim. Not established: BitNet-2B scale — the
2B weights are not in the repository and a 2B decode does not fit in CI; that leg needs a local
ARM board or a paid runner and stays open (A29).

**Token-level identity reaches production scale, x86 only (A36).** `cis_decode` run on the pruned
BitNet-2B artifacts (30 layers, hidden 2560, 50,256-token vocabulary) in the FullInt configuration
prints `CIS_DECODE digest=cab11400d737ac4a prompt_toks=4 gen_toks=64 mode=fullint`, identical
across two sequential runs on the same host (i5-10210U crosvm). The artifact is the A21/A35 model
with `__metadata__.aegis_config` added by `aegis-forge/add_aegis_config.py`; its tensor data blob
is byte-identical to the A21/A35 model, and the same log reproduces the A21/A35 float, hybrid, and
full-integer perplexity figures exactly on this artifact, so the digest and those figures share one
model. The generated text is coherent English ("…in a small town called Greenfield, there lived a
young girl named Lily…"). This is the x86 anchor for the BitNet-2B cross-ISA leg — the 2B
counterpart of A29 — but only the x86 leg exists so far: the aarch64 run has not been performed, so
the cross-ISA claim A29 makes for the M7 model does not yet extend to BitNet-2B. Identity evidence
only; no timing is quoted or quotable from it (A36).
