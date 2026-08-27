# Paper outline — "Bit-identical transformer inference across ISAs, with a verifiable decode receipt, demonstrated on bare metal"

Drafted 2026-08-27 by Claudius Maximus (Fable) from `program/RESEARCH_LEDGER.md` rows A19–A34,
`docs/CIS-1_SPEC_v1.0.md` (v1.0.2), `CHALLENGE.md`. **Every number below carries its ledger row;
nothing here is a new claim.** Rule B applies to the paper exactly as to the ledger.

Working title alternatives:
- *CIS-1: a canonical integer semantics that makes LLM inference reproducible bit-for-bit across x86 and ARM*
- *Provable AI Kit: a decode receipt that any machine can re-derive with no operating system*

Target: arXiv preprint (cs.LG + cs.CR) first; then a workshop (ML reproducibility / systems-for-ML /
trustworthy-ML). 8–10 pages + appendix. Doubles as the technical core of the DARPA Release 6 and NSF
Phase I narratives — write it so a program manager can read §1 and §7 alone.

---

## 1. Introduction — the problem is stated, not motivated by hype
- Quantized inference is assumed deterministic; it is not (float non-associativity in parallel kernels,
  dequantization paths, norm layers). Two machines running "the same model" disagree at the logit level,
  and there is no standard way to *prove* what a model computed.
- Contribution list (one line each, each pointing to a section):
  1. A frozen integer semantics for transformer inference (CIS-1) whose conformance is a digest (§3).
  2. Evidence it is implementable from the text alone: two clean-room implementations reproduce the
     digest first-run (A31) (§4).
  3. Cross-ISA bit-identity at op level (A25, A28), kernel level on real ARM silicon (A30), and token
     level for a complete greedy decode (A29) (§4).
  4. A decode *receipt* — SHA-256 bindings of artifacts + token digest + chained commitment over every
     step's full i64 logit vector — minted on x86, verified on aarch64 in public CI (A32) (§5).
  5. The receipt re-derived with **no operating system present** on two physical machines spanning
     AVX2 and SSE2-class code paths (A33, A34) (§5).
  6. The cost, measured honestly: integer semantics cost 25% vs scalar float on Broadwell and are 4–14%
     *faster* on Goldmont Plus (A26); at parity SIMD the integer AVX2 kernel is 2.94× faster than the
     float AVX2 incumbent (A27) (§6).
  7. Quality: the all-integer forward pass costs +0.0637% perplexity on M7 and +0.7408% on BitNet-2B
     vs float, inside a preregistered +5% kill line (A19–A21) (§6).
- What is **not** claimed (lift verbatim from spec §10 and the ledger caveats): see §8.

## 2. Background and related work (short, sourced)
- BitNet b1.58 / ternary weights (Microsoft model card in repo). Integer-only inference literature.
  Reproducibility-in-DL work. Measured boot / attestation (TCG, NIST) — position the receipt as
  *computational* attestation, orthogonal to platform attestation.
- One paragraph on why a *spec* (not a library) is the artifact: implementation freedom (§6 of spec) +
  exactness contract.

## 3. CIS-1: the semantics (from spec v1.0.2, normative sections 1–8)
- Axiom, rounding (RNE as the single normative rounding, A19), pinned constants (exp LUT digest
  0x66C2A0EEB8C2DC43, RoPE table 0xD8345EBF01E990FA — generated and cross-checked by an independent
  big-int generator, A20).
- Grids and headroom (hidden 384–2560, head_dim ≤128, seq ≤512; residual <2^50, normq n≤8192 — A20/A21).
- Operations §5.1–5.12: TMV, QUANT-ACT, REQUANT, RMSNORM-I (incl. the v1.0.1 erratum that made the
  spec implementable, A31), NORMQ, SOFTMAX-I (sum-to-one bound |Σp−2^15|≤⌈T/2⌉, monotonicity), ROPE-I,
  ACT-I, ARGMAX.
- Conformance = two digests: op-level `CIS_SELFTEST 76985613c965f643` (14 sections) and token-level
  `CIS_DECODE 67e8c0a96abc04e1` (M7, 64 tokens, full-int).
- Figure 1: pipeline with grid assignments (spec §5.12). Table 1: ops and their pinned goldens.

## 4. Implementability and cross-ISA identity (correctness evidence only — no timing in this section)
- Table 2 — the digest jury (A25): HP N4020 bare iron (SSE2), Dell i5-5200U bare iron (2 boots),
  crosvm i5-10210U, QEMU/TCG; + aarch64 Neoverse N2 CI runner (A28). One semantics, two ISAs,
  ≥5 codegen paths.
- Clean-room test (A31): two implementers, spec text only, 400 and 484 lines, distinct by md5, both
  print the digest first run. State the honest scope from the row (scalar Rust; not a third-party
  audit yet).
- NEON kernel on real ARM silicon (A30): `vmulq_s8` as ternary multiply; equivalence 5/5, contract
  6/6, mechanism 3/3 exhaustive; the −128 wrap pinned both directions.
- Token-level (A29): full pipeline decode digest identical on x86_64 and aarch64; output is coherent
  TinyStories text — identical *and* fluent. CI gates `arm-digest.yml` pin both digests on every push
  ("a future mismatch is a finding, not something to fix into silence" — quote the gate).

## 5. The decode receipt and bare-metal verification
- Receipt format (A32): SHA-256 of MODEL.SAF/EMBED.BIN/VOCAB.BIN, the 64 token ids, token digest,
  chained SHA-256 over every step's full i64 logit vector (chain `aee25b770bd7b22e…`).
- Verification = replay from source and compare. x86-minted, aarch64-verified in public CI run
  31249589879 (snapshot ce93bbb).
- Physical iron, no OS (A33, A34): the Provable AI Kit stick; BOOTLOG.TXT stage V `VERIFY PASS`;
  Dell (AVX2 path) and HP N4020 (SSE2 scalar path — AVX2 would #UD, so the PASS proves the second
  code path). **Attribution limitations stated as in the ledger**: verifier mode prints no CPUID; the
  HP entry's memory map matches the Dell's, so attribution of that entry rests on operator witness.
  This paragraph is where a reviewer will push; write it first and write it plainly.
- Figure 2: receipt structure. Figure 3: photo/BOOTLOG excerpt from the two machines.

## 6. Cost and quality (Rules A and B: physical hardware only, every figure to a log)
- Quality (A19–A21): hybrid +0.3127% → full-int +0.0637% on M7 (why: dropping two f32→fixed
  re-entries beats the table quantization it adds); BitNet-2B +0.7408% (200-token window, pruned
  vocab — carry the row's non-comparability caveats verbatim).
- Cost (A26): C/B 1.248× Dell (Broadwell-U, two FMA units) vs 0.961× HP (Goldmont Plus). The
  decomposition that retired both prior claims: 4.61× absent-SIMD × 1.248× semantics — determinism is
  not the expensive part.
- Kernel (A27): integer AVX2 2.94× faster than float AVX2 at parity SIMD, D/C 16.4× over scalar;
  **include the rejected first run** (arm A +58% drift → order control) as the method note the ledger
  records — it is the finding's provenance.
- Supporting context, clearly labeled as such: ring-0 vs minimal Linux +3.6–9.4% (A22, with its
  P-V2-2 FAIL reported), column-skip 2.88× (A23), bandwidth ceiling 0.62 vs 11 GB/s (A24). These are
  *not* CIS-1 claims; include only if space allows, in an "engine context" subsection.
- Table 3: every number in the paper → ledger row → log path.

## 7. Why it matters (one page, written for a program manager)
- Verifiable inference on commodity hardware with no OS: air-gapped, edge, sovereign deployments;
  audit/regulatory: "prove model X on firmware Y produced output Z".
- The $50 falsification challenge (CHALLENGE.md) as the paper's standing invitation — reviewers can
  try to break it before publication.

## 8. Limitations and what is not claimed (verbatim from spec §10 + ledger caveats)
- No claim of bit-identity for the pre-v0.3 hybrid path across kernel paths; sequence ≤512; the
  pruned-vocab PPL is not comparable cross-tokenizer; clean-room implementers were LLM subagents, not
  third parties; attribution caveats of A33/A34; no token-level integer *throughput* number yet
  (README says so — keep saying so).

## 9. Reproducibility statement
- Repo, commit, CI workflows (`arm-digest.yml`), golden receipt path, the two digests, and the exact
  commands (`cis_selftest`, `cis_decode`, verifier). Rule C: golden files and hardware logs are
  append-only and shipped with the paper.

---

## Work plan (no measurements required; nothing here needs the quiet-terminal rule)
1. Fable: §1, §7, §8 first (the claims and their limits) — these are the parts a reviewer reads.
2. cm-researcher (sonnet): §2 related-work sweep with URLs; no claims about our numbers.
3. cm-builder (sonnet): Table 3 auto-generated from the ledger (row → log path), and Figure 1 from spec §5.12.
4. Adversarial review: run `docs/ADVERSARIAL_REVIEWS_2026-08-01.md`'s protocol against the draft with
   cm-verifier + cm-critic; every objection either changes the text or is answered in §8.
5. Justin: decide venue, author line (sole author), and whether CHALLENGE.md's bounty is cited.

## Open decisions for Justin
- Preprint first, or workshop deadline first? (arXiv is same-day; a workshop gives a review.)
- License line on the paper vs the Apache-2.0/MIT discrepancy on the repo — resolve before submission.
- Whether to run one more physical leg (a third machine) before submitting, to strengthen A33/A34
  attribution. Not required; would need you at the machine.
