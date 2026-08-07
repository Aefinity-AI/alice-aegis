# DIRECTION v2 — Verified Inference: computing AI as evidence

**Date:** 2026-07-31 → v2 same night, after the 11-agent prior-art/market
sweep + red team (workflow `new-paradigm-hunt`, 638k tokens, 262 searches).
**Status:** thesis SURVIVED red team with four required pivots, all adopted
below. First measurements banked. This document supersedes v1 (same file,
git history has v1).

---

## 1. The reframe (post-red-team wording)

Every mainstream AI stack computes an inference and asks you to take its
word for it. The output depends on floating-point execution order — we
proved it on our own engine tonight (§3) — and the industry's audit answer
is logs (assertions), TEE certificates (trust Intel/NVIDIA), statistics
(probabilistic fingerprints), or zk-proofs (10⁴–10⁶× overhead).

**The paradigm:** *auditable-by-construction* inference. A system whose
every decision is, by the arithmetic itself, a **re-executable piece of
evidence** — a witness any commodity CPU can replay bit-for-bit, with no
trusted vendor, no network, no reference model, and no statistical
tolerance, verifiable today or in twenty years.

Three components make it a computing paradigm and not a logging feature:

1. **CIS-1 — canonical integer inference semantics.** Ternary weights ×
   integer activations × integer accumulators = associative arithmetic. Any
   ISA, SIMD width, tiling, or thread count produces THE SAME BITS.
   Determinism is a property of the math, not a discipline of kernels.
2. **The witness** — hash chain over model/config hashes, prompt, and every
   step's token + exact i32 logit digest. Verification = re-execution.
3. **The kilobyte root of trust** — the UEFI unikernel as reference
   verifier: firmware auto-measures ONE ~450KB PE image into TPM PCR4 (TCG
   spec-guaranteed), vs a Linux attestation chain of hundreds of fragile
   event-log entries. The Band-3 loss repositioned this asset correctly:
   not a speed play — the smallest auditable inference stack in existence.

**Killed claim (red team, fatal):** ~~"a $50 USB stick audits a
datacenter"~~. Replay verifies only systems that RUN the canonical engine.
We do not audit other people's clouds; we sell deployments that are
*born auditable*. The buyer question is not "can you check OpenAI's
homework" but "field this engine and every decision it ever makes is
bit-reproducible evidence" — a claim no TEE, statistical, or zk vendor
makes, and none of them CAN make on a disconnected 10-year-old laptop.

## 2. The empty cell (verified against the field, 2026-07-31)

Four independent recon sweeps converged: **cross-ISA bit-exact LLM
inference on commodity CPUs via canonical integer semantics, sold as
evidence, with an open spec + conformance suite + minimal-TCB bootable
verifier, is unoccupied.** Every neighbor concedes exactly one axis:

| Neighbor | What they have | The axis they concede |
|---|---|---|
| Thinking Machines / SGLang / vLLM / llama.cpp | batch-invariant / same-box determinism (34–61% overhead on GPU) | docs explicitly disclaim cross-hardware; CPU claims are per-device |
| Gensyn Verde/RepOps (closest mechanism) | cross-hardware bitwise floats by pinning reduction order | curated GPU list (CUDA 12.6+), 200–300% overhead, crypto-compute framing, no spec/conformance |
| EigenAI (EigenLayer) | ~1× byte-equality replay | same-GPU-SKU only + TEE + blockchain; crypto-agent market |
| opML | fixed-point deterministic re-execution | disputes run in an emulated MIPS VM inside a blockchain oracle |
| Cankaya (arXiv 2606.00279) | bit-exact audit of EXISTING float GPU inference | by per-device arithmetic emulation — every GPU/kernel version forever |
| DiFR / TOPLOC | ~1% overhead statistical verification | probabilistic by design; needs reference infrastructure; evidence ≠ opinion |
| EQTY Lab + Intel/NVIDIA | enterprise attestation certificates, EU-AI-Act framing | trust-the-silicon; no third-party re-execution; useless air-gapped |
| zkML (EZKL, Lagrange…) | trustless proofs | 10³–10⁶× proving overhead; offline auditing only |
| **int-llm (June 2026)** | pure-integer Q16.48 C core, byte-identical across arm64/x86-64/32-bit | a feasibility oracle: slow, no spec, no transcript, no verifier, no product — **proves our bet is buildable** |
| **NIGHTRUN (released 2026-07-30)** | Rust no_std UEFI LLM chat, x86_64 + RPi5, MIT, real models | no canonical semantics, no transcript, no attestation framing — **and one feature away; our uniqueness clock is running** |
| bitnet.cpp (Microsoft) | "lossless" ternary kernels | accumulates in FP32 (I2_S) / float epilogues — NOT bit-exact cross-hardware; our lane is open even against Microsoft |
| I-BERT / I-LLM / IntAttention | every hard op integerized in the literature (softmax LUT ≈ 2–3% ppl cost, shrinking with table size) | nobody assembled semantics+transcript+verifier; RoPE is the one open operator (our shipped-table approach covers it) |

Two structural facts from the sweep worth engraving:
- **Integer-for-bit-exactness is already PROVEN practice in neural
  compression** (Ballé lineage: integerized codecs because entropy coding
  *requires* cross-device bit-exactness). We are importing a proven
  principle into a field that hasn't noticed it yet.
- **Nobody anywhere publishes a price for "verify this inference."** Zero
  demonstrated willingness-to-pay for verification as a standalone SaaS.
  Therefore: never sell verification. Sell the RUNTIME into deployments
  that already pay for T&E/forensics/certification, with auditability as
  the property that wins the deal.

## 3. What we proved on our own hardware (tonight, banked)

- **E1 — floats fork stories.**
  `docs/hardware_logs/e1_detprobe_crosspath_m7_2026-07-31.log`: same binary,
  same weights, same machine, greedy; 16/16 runs bit-identical within a
  path; **2 of 4 prompts fork across scalar↔AVX2** (P1 @ token 67, P3 @
  token 77) — narratives genuinely diverge. Correct f32 is not a portable
  fact. (This also quantifies why bitnet.cpp-style float accumulation can
  never be a witness.)
- **Witness v0 — the object works.**
  `docs/hardware_logs/witness_v0_demo_2026-07-31.log`: 9-line witness
  (SHA-256 cross-checked against coreutils); honest replay **PASS**;
  cross-arithmetic replay **FAIL** (the hole CIS-1 closes); 1 byte flipped
  in 2.8MB of weights **FAIL**. Generate → verify → tamper-detect, live.
- **CIS-1 draft spec exists:** `docs/CIS-1_SPEC_DRAFT_v0.1.md` — all decode
  ops as exact integers with headroom proofs; requant (TFLite lineage,
  deliberately); normative LUT softmax/rope-tables/activation; tie-broken
  argmax; transcript format; conformance-by-golden-transcripts.

## 4. Falsifiable program (kill criteria preregistered before advocacy)

| # | Claim | Method | Kill criterion |
|---|---|---|---|
| E1 | f32 cross-path divergence flips real tokens | **DONE, banked** | survived (2/4 prompts) |
| E2 | Integer decode costs acceptable accuracy | CIS mode; `calculate_perplexity` on M7 + BitNet-2B vs f32, same eval set | >5% rel. ppl loss on BitNet-2B with no knob ⇒ revise to A16, retest once, else dies. Literature says 2–3% with 32-entry LUTs; ours are 1024-entry. **This number at ternary-2B scale exists nowhere in the literature — it is the load-bearing unknown and our first publishable figure** |
| E3 | Witness overhead negligible | tok/s ± SHA-256 chain, Dell i5-5200U (Rule A: iron) | >3% decode overhead ⇒ redesign digest granularity |
| E4 | **The jury demo** — cross-ISA bit-identity | one transcript, four verifiers: Dell (AVX2), HP Stream (SSE2 scalar), unikernel (ring 0), QEMU-TCG (emulation-as-verifier) | ANY mismatch ⇒ spec bug; fix or the paradigm dies. No incumbent can run this demo |
| E5 | Hardware-rooted attestation on commodity iron | **HP Stream** (N4020 has fTPM 2.0/PTT; the Dell has NO TPM — recon-corrected): firmware auto-measures unikernel into PCR4; app self-measures weights via EFI_TCG2 into PCR8+; quote | fTPM event-log replay fails on Insyde-class firmware ⇒ ship software-root first, attestation on better iron |
| E6 | A stranger can verify | publish one BitNet-2B witness + verifier; external party reproduces h(T) | non-reproduction by honest third party ⇒ product claim false |

**Standing kill signals (watch, from recon):** Gensyn RepOps ships CPU
coverage at low overhead; bitnet.cpp/llama.cpp ship an "integer-exact"
mode; courts/T&E formally accept statistical fingerprints as evidence;
prEN 18229-1 finalizes logging-only compliance and buyers stop caring;
NIGHTRUN adds transcripts + integer semantics before we publish.

## 5. Strategy: one paradigm, three layers

**L1 — Own the standard (the moat).** Publish CIS-1 as an open spec with
conformance vectors, FAST (red team: spec authorship is the only moat
available to a solo founder, and it decays daily — int-llm and NIGHTRUN are
each one insight away). Reference implementations: aegis-core CIS mode +
a ~500-line obviously-correct scalar auditor's build. We become the
conformance authority the way this program already runs itself: golden
bits or it didn't happen.

**L2 — The wedge product (from the rival that beat us).** The red team's
pivot and Rival 1 ("Cold-Boot Examiner") compose: **forensic AI triage
that boots ON the evidence machine.** Stick in the seized/compromised PC,
ring-0 unikernel + read-only disk access, model triages in place, data
never moves — and every conclusion it emits is a CIS-1 witness. The first
forensic AI whose own output is itself forensic evidence. Tailwinds the
sweep found: DFIR lab backlogs (months-to-years), proposed FRE 707
(machine-generated evidence must show "reliable principles and methods,
reliably applied" — bit-exact replay is the strongest possible showing,
effective as early as Dec 2027), and a practitioner consensus that logs
without tamper-evidence "have zero evidentiary value." Rival 1's honest
technical path: no_std NTFS read-only parsing is the main new engineering.

**L3 — The defense channel.** CDAO's **AI T&E BPA** ($249M ceiling, 44 of
79 vendors are small businesses) and **Tradewinds** (5-minute video →
"Awardable" status → any DoD buyer can procure) are the low-friction
doors. DTE&A's assurance-case framework has NO evidence mechanism for a
specific fielded inference — the witness is exactly that missing evidence
type. Sober note from recon: DOT&E was gutted in 2025 and CDAO is in
churn; sell tools into the T&E ecosystem through existing vehicles, do
not bet the company on a doctrine champion appearing.

**EU (2027+ tailwind, not a 2026 wedge):** Digital Omnibus slipped
high-risk obligations to Dec 2027/Aug 2028; harmonized standards
unfinished. When prEN 18229-1 lands, "exceeds Article 12" is marketing,
not the wedge.

## 6. What we do NOT claim (unchanged in spirit, sharpened by red team)

- We do not audit systems that don't run conforming engines. No datacenter
  claims. Auditable-by-construction only.
- No single primitive is novel — requant is TFLite lineage, integer ops
  are I-BERT lineage, and we say so. The contract, the conformance
  authority, and the bootable verifier are the product.
- Integer accuracy cost is UNKNOWN at our scale until E2 runs. It is a
  kill criterion, not a footnote.
- Verification alone has zero demonstrated willingness-to-pay. The runtime
  is the product; the witness is why it wins.
- MECH/OS-cost series closes first (Dell boot pending, runcard live). That
  discipline is this direction's credibility.

## 7. Build order

1. **CIS-1 ops module in aegis-core + golden vectors** (started: E1 probe,
   witness v0, SHA-256 validated; `cis.rs` next).
2. **E2 perplexity gate** on M7, then BitNet-2B — the field's missing
   number, and our go/no-go.
3. **Witness v1** (i32 logit digests) in aegis-linux + unikernel.
4. **E4 jury demo** across all four verifiers — the video that anchors the
   Tradewinds submission, the SBIR pitch, and the spec announcement.
5. **E5 TPM chain on the HP Stream.**
6. **CIS-1 v1.0 publication** + conformance suite + the two reference
   implementations, announced with the jury video.
7. **Cold-Boot Examiner spike:** no_std NTFS read-only proof-of-concept in
   the unikernel (Rival 1's E1), decision gate after.
