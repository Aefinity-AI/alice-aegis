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
- Danopoulos et al., "Taming the Exponential: A Fast Softmax Surrogate for Integer-Native Edge Inference," arXiv:2604.02292 — https://arxiv.org/abs/2604.02292 (UNVERIFIED beyond abstract)
- Trusted Computing Group, "Overview of TCG Technologies for Device Identification and Attestation," v1.0 rev. 1.37 — https://trustedcomputinggroup.org/wp-content/uploads/Overview-of-TCG-Technologies-for-Device-Identification-and-Attestation-Version-1.0-Revision-1.37_5Feb24-2.pdf (content summary drawn from search index only; full text not machine-readable via fetch — UNVERIFIED beyond title/version)
- NIST SP 800-155 (Initial Public Draft), "BIOS Integrity Measurement Guidelines" — https://csrc.nist.gov/pubs/sp/800/155/ipd
- Sun, Li, and Zhang, "zkLLM: Zero Knowledge Proofs for Large Language Models," arXiv:2404.16109 — https://arxiv.org/abs/2404.16109
- Peng et al., "A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning," arXiv:2502.18535 — https://arxiv.org/abs/2502.18535
- "Orion: A Fully Homomorphic Encryption Framework for Deep Learning," arXiv:2311.03470 — https://arxiv.org/abs/2311.03470
- NIST, "Cryptographic Algorithm Validation Program (CAVP)" — https://csrc.nist.gov/Projects/cryptographic-algorithm-validation-program
- NIST/CSRC, "The Advanced Encryption Standard Algorithm Validation Suite (AESAVS)" — https://csrc.nist.gov/csrc/media/projects/cryptographic-algorithm-validation-program/documents/aes/aesavs.pdf
