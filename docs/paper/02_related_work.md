# 2. Background and related work

**Ternary-weight and integer-only LLMs.** BitNet b1.58 quantizes every weight
to {−1, 0, +1} (1.58 bits/parameter) and trains directly at that precision
rather than quantizing a float-trained model post hoc; Ma et al. report
perplexity and end-task parity with FP16 LLaMA baselines at matched size and
training tokens, alongside large memory, latency, and energy reductions
(Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58
Bits," arXiv:2402.17764, Feb 2024). Microsoft's follow-up technical report
open-sources a 2B-parameter, 4T-token BitNet b1.58 checkpoint with public
weights and CPU/GPU inference code (Ma et al., "BitNet b1.58 2B4T Technical
Report," arXiv:2504.12285, Apr 2025). A companion systems paper describes
CPU kernels for ternary inference and reports CPU speedups of 2.37–6.17×
on x86 and 1.37–5.07× on ARM over unoptimized baselines (Wang et al.,
"1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on
CPUs," arXiv:2410.16144, Oct 2024); a related report, bitnet.cpp, targets
edge deployment of ternary models (arXiv:2502.11880). None of this line of
work specifies a bit-exact numerical contract for the non-matmul path
(normalization, softmax, rotary embedding, requantization): the object
released is a model and a fast kernel, not a semantics that a second,
independent implementation is obligated to reproduce bit-for-bit. That gap
— fast ternary inference exists; a frozen, checkable numerical contract for
it does not — is the space CIS-1 occupies.

**Non-determinism and reproducibility in DL inference.** Floating-point
addition is not associative, so reduction order — which varies with thread
scheduling, tiling, batch size, and kernel choice — changes results at the
bit level; Shanmugavelu et al. quantify this across HPC and deep-learning
workloads and evaluate deterministic-summation and hardware-level
mitigations, including PyTorch's determinism flags (arXiv:2408.05148, Aug
2024, rev. Oct 2024). Horace He's widely-cited analysis for Thinking
Machines Lab argues the dominant real-world cause of LLM-inference
nondeterminism is not concurrency per se but *batch-size variance*:
individual GPU kernels (matmul, RMSNorm, attention) are themselves
deterministic for a fixed batch shape but change their reduction strategy —
and therefore their bit-level output — as batch size changes; the proposed
fix is batch-invariant kernels, measured at roughly 20% overhead for
matmul/RMSNorm and up to ~2× unoptimized for attention (He, "Defeating
Nondeterminism in LLM Inference," Thinking Machines Lab blog, Sep 2025).
DiFR frames the adjacent problem of verifying a provider's inference under
inherent nondeterminism, comparing generated tokens or compressed
activation fingerprints against a trusted reference to catch, e.g.,
undisclosed quantization (Karvonen et al., arXiv:2511.20621, Nov 2025).
CIS-1 targets a different point in this space: rather than tolerating
float nondeterminism and verifying *around* it statistically, it removes
floats from the decode path so that reduction order provably cannot change
the result, making the DiFR-style tolerance machinery unnecessary for a
conforming implementation.

**Integer and fixed-point transformer inference.** I-BERT performs
end-to-end BERT inference in integer-only arithmetic, replacing GELU,
Softmax, and LayerNorm with polynomial/lookup integer approximations (Kim
et al., arXiv:2101.01321, Jan 2021, rev. Jun 2021). I-ViT extends this to
Vision Transformers with Shiftmax and ShiftGELU, reporting INT8 accuracy
comparable to full precision (Li and Gu, arXiv:2207.01405, Jul 2022, rev.
Aug 2023, ICCV 2023). A 2026 report proposes a clipped-linear softmax
surrogate for int8 edge accelerators (Danopoulos et al., "Taming the
Exponential," arXiv:2604.02292, Apr 2026) — UNVERIFIED beyond its abstract,
not read in full. These works establish that individual nonlinear ops
*can* be made integer-only; none states its integer approximations as a
frozen, versioned specification with pinned rounding and a conformance
digest, nor addresses RoPE. That distinction — approximation technique vs.
normative, checkable semantics — is the same gap noted for BitNet above.

**Verifiable inference.** Platform (not computational) attestation is
mature: TCG measured boot extends per-stage hashes into TPM PCRs, signed
and reported to a verifier (Trusted Computing Group, "Overview of TCG
Technologies for Device Identification and Attestation," v1.0 rev. 1.37,
Feb 2024); NIST's SP 800-155 draft specifies BIOS/firmware integrity
measurement guidelines for the same chain (NIST SP 800-155 (Initial Public
Draft), Dec 2011). This attests *what code ran*, not what it computed.
Cryptographic approaches instead prove computation: zkLLM proves correct
LLM inference in zero-knowledge with a specialized attention argument,
generating a full-inference proof for a 13B model in under 15 minutes with
sub-200 kB proofs (Sun et al., arXiv:2404.16109, Apr 2024, ACM CCS 2024); a
2025–2026 survey catalogs the broader ZKML literature across training,
testing, and inference (Peng et al., arXiv:2502.18535). A parallel line
uses fully homomorphic encryption so a server computes on encrypted inputs
without seeing them, e.g. Orion for FHE deep learning (arXiv:2311.03470);
FHE and ZK proofs both target privacy/correctness against an untrusted
*executor*, at heavy per-inference cost. CIS-1's decode receipt is a
lighter-weight, complementary object: a commitment (SHA-256 chain) over an
already-computed integer transcript, verified by re-derivation rather than
by circuit proof, and — because the underlying arithmetic is
order-invariant by construction — reproducible on different hardware
without the prover/verifier asymmetry ZK and FHE approaches require. It
attests *what was computed*, and is designed to compose with, not replace,
platform attestation.

**Reproducible builds and supply-chain provenance.** SLSA (Supply-chain Levels
for Software Artifacts, slsa.dev) and in-toto (Torres-Arias et al., "in-toto:
Providing farm-to-table guarantees for bits and bytes," USENIX Security 2019)
are the closest prior art for CIS-1's stated goal of "verify what was
built/run without trusting the builder": both attach signed, structured
attestations to a supply chain so a verifier can check that a claimed
sequence of steps (build, test, package) actually produced a given artifact,
without re-executing the whole pipeline. That is provenance about *which
code produced an artifact* — a chain of custody — not a claim about the
bit-exact numerical semantics of what the artifact computes at runtime. A
SLSA/in-toto attestation says "this binary was built from this source by
this build system"; it says nothing about whether re-running that binary's
inference on different hardware reproduces the same output. CIS-1's decode
receipt is a claim of the latter kind: an independent verifier re-derives
the numeric result — token-for-token, bit-for-bit — from the spec and the
artifact hashes, not from trust in the build pipeline that produced the
weights or the binary. The two problems compose (a SLSA-attested build could
still produce a CIS-1 receipt) but are not the same problem, and no
supply-chain-provenance framework we are aware of makes a claim about
numerical reproducibility of the computation itself.

**Verifier independence.** The credibility of a receipt scheme rests on
whether the party checking it can be fooled by the party producing it, so we
state independence explicitly. Two lines of evidence support it for CIS-1:
two from-scratch, spec-only clean-room implementations — no access to the
reference source, 400 and 484 lines, verified distinct by diff/md5 —
independently reproduce the Tier-2 conformance digest
`76985613c965f643`, evidence that the specification text alone, not the
reference engine, determines the answer (RESEARCH_LEDGER.md A31); and a
separate `cis-verify` crate, with zero dependency on `aegis-core` and no
shared runtime code with the engine, ~4,700 lines transcribed from the spec
text, independently reproduces both conformance digests and verifies a
golden decode receipt end-to-end, with tamper tests failing and naming the
corrupted field (A37). `cis-verify`'s transcription had the reference source
visible, so we describe it as evidence of spec-re-implementability rather
than a blind third-party audit (the honest-scope caveat is stated at length
in §8); the clean-room pair (A31) is the stronger independence claim, since
those implementers worked from the spec text alone. Both verifiers, and the
receipts they check, reproduce on the GitHub `aarch64` CI runner as well as
`x86_64`, as a standing CI gate (A38). Taken together, no single codebase
produces both a receipt and its own passing verification.

**Economic and GPU-specific verification.** Two 2026 systems are the
closest prior art to CIS-1's determinism-plus-receipt claim, and both are
narrower in a way worth stating precisely. EigenAI (arXiv:2602.00182) ships
bit-exact GPU inference gated by an *economic* verification layer — stake,
slashing, and a TEE-mediated dispute process — rather than an independent
re-derivation from a public spec; it is also explicitly not
cross-architecture (its own reporting shows 0% output match between an A100
and an H100 for the same model). A second 2026 report on bit-exact
verification of existing floating-point GPU inference (arXiv:2606.00279)
targets NVIDIA hardware only, via CPU-emulator output matching, with a
self-built verifier and no independent third-party re-implementation
reported. Neither crosses instruction-set architectures, neither is
integer-only or ternary, and neither ships a verifier built from a public
specification by a party other than the scheme's own authors. CIS-1/CIS-2
differ on exactly these axes: the numerical contract is a versioned, public
specification rather than a vendor's kernel behavior to be matched, the
arithmetic is integer-only so there is no floating-point reduction order to
emulate, and the receipts are checked by verifiers — the clean-room pair and
`cis-verify` — that are demonstrably independent of the engine, with
CI-green reproduction on both x86_64 and aarch64 (A31, A37, A38).

**Bare-metal / no-OS LLM inference.** Running LLM inference directly in UEFI
boot-services mode, without an operating system, is not unprecedented: a
secondary report describes a from-scratch C tokenizer/weight-loader/tensor-
math/inference stack running as a UEFI application on a Dell E6510, with no
model, precision, or performance details given beyond an interactive chat
demo and an admission that "optimization work has barely been done"
(Insights, marvin-42.com, "Bare-Metal AI: Running LLM Inference Directly in
UEFI, No OS or Kernel Required," Mar 2026 — itself a secondary report on an
unlinked LocalLLaMA/Reddit project, not a primary technical writeup;
UNVERIFIED beyond this secondary source). That project establishes UEFI as
a viable host for LLM inference; it makes no claim of determinism,
bit-exact reproducibility across ISAs, a cryptographic receipt, or
independent third-party verification — the properties CIS-1 adds, and the
ones the unikernel demonstration (§5) is built to exercise rather than the
no-OS environment alone.

**Conformance-by-digest as a design pattern.** Cryptographic standards
have long used exactly this shape: NIST's Cryptographic Algorithm
Validation Program (CAVP) validates a black-box implementation by feeding
it known inputs and checking outputs against known-correct answers (Known
Answer Tests, Multi-block Message Tests, Monte Carlo Tests), most visibly
for AES (NIST CAVP; AESAVS specification, csrc.nist.gov). Conformance is a
match against a pinned test vector, not an audit of source code or
implementation strategy — implementation freedom plus an exactness
contract, verified by anyone who can run the vectors. CIS-1's two
conformance digests (op-level and token-level) are this pattern applied to
transformer inference: a specification document is normative, and any
implementation — regardless of ISA, vector width, or language — either
reproduces the pinned digest or does not conform.

## Sources

- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," arXiv:2402.17764 — https://arxiv.org/abs/2402.17764
- Ma et al., "BitNet b1.58 2B4T Technical Report," arXiv:2504.12285 — https://arxiv.org/abs/2504.12285
- Wang et al., "1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs," arXiv:2410.16144 — https://arxiv.org/abs/2410.16144
- "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs," arXiv:2502.11880 — https://arxiv.org/pdf/2502.11880
- Shanmugavelu et al., "Impacts of floating-point non-associativity on reproducibility for HPC and deep learning applications," arXiv:2408.05148 — https://arxiv.org/abs/2408.05148
- He, "Defeating Nondeterminism in LLM Inference," Thinking Machines Lab — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- Karvonen et al., "DiFR: Inference Verification Despite Nondeterminism," arXiv:2511.20621 — https://arxiv.org/abs/2511.20621
- Kim et al., "I-BERT: Integer-only BERT Quantization," arXiv:2101.01321 — https://arxiv.org/abs/2101.01321
- Li and Gu, "I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference," arXiv:2207.01405 — https://arxiv.org/abs/2207.01405
- Insights (marvin-42.com), "Bare-Metal AI: Running LLM Inference Directly in UEFI, No OS or Kernel Required," Mar 2026 — https://insights.marvin-42.com/articles/bare-metal-ai-running-llm-inference-directly-in-uefi-no-os-or-kernel-required (secondary report, no primary source linked; UNVERIFIED beyond this source)
- Danopoulos et al., "Taming the Exponential: A Fast Softmax Surrogate for Integer-Native Edge Inference," arXiv:2604.02292 — https://arxiv.org/abs/2604.02292 (UNVERIFIED beyond abstract)
- Trusted Computing Group, "Overview of TCG Technologies for Device Identification and Attestation," v1.0 rev. 1.37 — https://trustedcomputinggroup.org/wp-content/uploads/Overview-of-TCG-Technologies-for-Device-Identification-and-Attestation-Version-1.0-Revision-1.37_5Feb24-2.pdf (content summary drawn from search index only; full text not machine-readable via fetch — UNVERIFIED beyond title/version)
- NIST SP 800-155 (Initial Public Draft), "BIOS Integrity Measurement Guidelines" — https://csrc.nist.gov/pubs/sp/800/155/ipd
- Sun, Li, and Zhang, "zkLLM: Zero Knowledge Proofs for Large Language Models," arXiv:2404.16109 — https://arxiv.org/abs/2404.16109
- Peng et al., "A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning," arXiv:2502.18535 — https://arxiv.org/abs/2502.18535
- "Orion: A Fully Homomorphic Encryption Framework for Deep Learning," arXiv:2311.03470 — https://arxiv.org/abs/2311.03470
- NIST, "Cryptographic Algorithm Validation Program (CAVP)" — https://csrc.nist.gov/Projects/cryptographic-algorithm-validation-program
- NIST/CSRC, "The Advanced Encryption Standard Algorithm Validation Suite (AESAVS)" — https://csrc.nist.gov/csrc/media/projects/cryptographic-algorithm-validation-program/documents/aes/aesavs.pdf
- SLSA (Supply-chain Levels for Software Artifacts) — https://slsa.dev
- Torres-Arias et al., "in-toto: Providing farm-to-table guarantees for bits and bytes," USENIX Security 2019 — https://in-toto.io
- "EigenAI: Deterministic AI Inference" (EigenCloud), arXiv:2602.00182 — https://arxiv.org/abs/2602.00182
- "Bit-Exact AI Inference Verification," arXiv:2606.00279 — https://arxiv.org/abs/2606.00279 (details unverified beyond abstract)
