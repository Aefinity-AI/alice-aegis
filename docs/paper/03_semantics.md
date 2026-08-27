# 3. CIS-1: the semantics

CIS-1 rests on one axiom (§1): *every reduction in the decode path is a sum of integers whose
worst case provably fits its accumulator.* Integer addition is associative and commutative, so a
conforming implementation may reorder, vectorize, tile, or parallelize any computation — SSE2,
AVX2, NEON, GPU, FPGA, a for-loop in Python — and MUST produce bit-identical results. The spec
explicitly rejects the alternative of a canonical float summation order: pinning one loop
structure would kill SIMD freedom and still break under FMA-vs-separate rounding. Determinism is
therefore a property of the arithmetic itself, not a discipline the kernel author must maintain.

Rounding is fixed at one mode everywhere: round-half-even (RNE), with `rne(num/den)` for `den > 0`
computed exactly in `i128` as the reference primitive (§2). The spec calls tie handling normative,
not a detail, because the motivating token flips traced to near-ties decided by float rounding
noise. Four transcendental constants are pinned as the RNE rounding of published 50-digit decimal
expansions, and the integers themselves — not the decimals — are normative: `TWO_PI_Q62 =
28976077832308491370`, `PI_Q62 = 14488038916154245685`, `PI_2_Q62 = 7244019458077122842`
(independently rounded, not derived from `PI_Q62`), and `LOG2E_Q32 = 6196328019`. Later pinned
artifacts are verified the same way, by digest rather than by printing the value: the exp LUT
(§5.7) carries FNV-1a 64 digest `0x66C2A0EEB8C2DC43`, golden-tested against an independent
big-integer generator, and the RoPE table for the M7 shape 512×32 at base 10000.0 (§5.9) carries
`0xD8345EBF01E990FA`.

Every object that crosses the decode path is assigned a type, a fixed-point grid, and a proven
bound (§3). Quantized activations are symmetric `i8` in `−127..+127`, with `−128` explicitly
forbidden. Dot-product accumulators are `i32`, valid for `dim_in ≤ ⌊i32::MAX/127⌋`. The residual
stream is `i64` in Q.20 with `|h| < 2⁵⁰` and vector length `n ≤ 8192`. q/k/v vectors are `i32`
Q.16 with `|·| < 2²⁹` pre-rotation, enforced by a loud panic rather than saturation. Attention
scores are `i64` Q.24 with `|score| ≤ 2⁵⁹`, accumulated in `i128`. Probabilities live on Q0.15
with `Σp` within `⌈T/2⌉` of `2¹⁵`. exp LUT entries are Q0.31 (§5.7), RoPE tables are `i32` Q1.30
(§5.9), norm gains are `i32` Q.20 in the engine or `i16` in the reference op, and logits are `i64`
from an exact integer dot. The spec states its headroom claim plainly: a ternary dot of length L
over `i8` is bounded by `127·L`, and a sum of squares by `127²·L`; violating a stated bound MUST
be a loud failure, never a silent wrap or saturation.

Weight containers pack row-major, four weights per byte, low bit-pair first, with `dim_in` a
multiple of 4 (§4). The two-bit codes are `00` = 0, `01` = +1, `10` = −1, and — ratified in this
version — `11` decodes to 0 as well (defined-as-zero); a container holding `11` codes is
conforming, and those positions simply contribute nothing to the dot product.

Section 5 defines twelve operations. **TMV** (§5.1), the ternary matrix–vector product, is exact
in `i32` in any summation order; equally normative is its rejection surface — five preconditions
(non-multiple-of-4 `dim_in`, an overflowing `dim_in`, and three length checks) that every
implementation must reject identically regardless of ISA or internal blocking, with a rejected
call writing no output. **QUANT-ACT** (§5.2) is per-token symmetric absmax quantization onto the
`i8` grid, `q_i = rne(x_i·127/absmax)`, with the scale carried forward as the exact rational
`127/absmax`; the clamp that makes `−128` unrepresentable is explicit. **REQUANT** (§5.3) applies
a fixed-point multiplier `y = clamp(rne((acc·M)/2^(31+S)), −127, 127)` with `M ∈ [2³⁰, 2³¹)` and
`S ≤ 62`; the `i64`-input form used by RMSNORM-I is the identical arithmetic widened to `i128`.
The offline generation of `(M,S)` from any rational is itself a normative integer procedure,
including the exact `S`-selection rule and a renormalization case when RNE rounds `M` up to
`2³¹`. **RMSNORM-I** (§5.4) specifies every intermediate grid — sum of squares in `i64`, an exact
floor integer square root `t`, an inverse-RMS in Q2.30, and a final Q.20 output through REQUANT —
because folding these roundings any other mathematically-equivalent way produces different `i8`
outputs; this full procedure is a v1.0.1 erratum, since the original prose omitted the
bit-determining grids. **NORMQ** (§5.5) is the fused engine form: the RMS factor cancels out of
the `i8` codes algebraically and survives only in a carried exact rational scale. §5.6 governs
**container-boundary conversions** — bf16/f32-to-fixed conversions are exact RNE on the float bit
pattern alone, never on accumulated values, and the block-exponent form `fix_f32_vec` MUST panic
rather than saturate when its fractional width would go negative. §5.7 defines the shared **exp
machinery**: a parameter-free floor-isqrt chain generates a 1025-entry, strictly-decreasing Q0.31
LUT, looked up by a monotone interpolation. **SOFTMAX-I** (§5.8) works on the Q.24 score grid,
subtracts the max exactly, and normalizes by exact RNE division rather than a Newton reciprocal,
with the declared bound `|Σp − 2¹⁵| ≤ ⌈T/2⌉`. **ROPE-I** (§5.9) tables are not shipped but
generated at load from `(max_seq, head_dim, f32 bits of rope_theta)` by a normative integer
procedure — quadrant-reduced sin/cos via 9-term Taylor sums, RNE-requantized to Q1.30 and clamped
to `±2³⁰`. **ACT-I** (§5.10) elementwise ops work on the Q.20 grid: relu² is an exact squaring
plus one more RNE requant, and silu evaluates one `exp_neg` call for `σ(g)` in Q0.31 before two
further RNE requants. **ARGMAX** (§5.11) breaks exact-equality ties on the `i64` logits to the
lowest index, making a tie a specified, reproducible event rather than rounding noise.

**§5.12 — pipeline grid assignments** fix the grid at every stage of the forward pass:

| Stage | Grid / accumulator |
|---|---|
| q/k/v (post-rescale) | `i32`, Q.16 (rotation growth ≤ √2 stays inside `i32`) |
| Score dot | `i128` accumulator, scaled by `1/√head_dim` in Q0.30, RNE'd to Q.24 |
| V-mix (`Σ_t p_t·v_t`) | exact `i64`, RNE onto the Q.20 residual grid |
| LM head | exact `i64` dot, `i8`-quantized hidden state × Q.20 embedding table |

Section 6 states the exactness contract: an optimized kernel conforms only if it is byte-identical
to the reference for *every* input, not merely inputs a well-behaved caller would produce. The
worked hazard is `−128`: x86 `vpsignb` and ARM `vmulq_s8` both wrap `−128 × −1` within `i8` where
the reference computes in `i32`, so both shipped kernels detect `i8::MIN` in their preparation
pass and route the whole call to the reference — an unconditional guarantee, not one conditioned
on the caller upholding an invariant. The rejection surface of §5.1 is likewise part of the op:
whether a call is rejected must not depend on which CPU the binary landed on. The shipped vector
kernels, `cis_avx2` and `cis_neon`, are informative rather than normative; their instruction-level
claims are proven by exhaustive enumeration, and whole-kernel equality is checked by test suites —
deliberately described as a test bar, not a proof.

Section 7 defines what a conforming engine computes: `CisMode::FullInt` runs embedding lookup, the
residual stream, every norm, every activation quantization, all ternary matvecs, RoPE, attention
scores, softmax, V-mix, the MLP elementwise ops, the LM head, and argmax such that no float value
exists anywhere in the forward pass. Hybrid modes that keep attention or the MLP in `f32` carry no
cross-ISA claim and sit outside conformance.

Conformance (§8) requires all three tiers to pass: Tier 1 op goldens, Tier 2 the operation-level
digest, and Tier 3 the token-level digest. Tier 2 (`cis_selftest`) must print exactly:

```
CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true
```

Tier 3 (`cis_decode`, prompt `"Once upon a time"`, 64 new tokens, greedy, EOS ignored, `FullInt`) must print
exactly:

```
CIS_DECODE digest=67e8c0a96abc04e1 prompt_toks=4 gen_toks=64 mode=fullint
```

---

**Source map**
- Axiom and rounding/constants paragraphs: spec §1, §2.
- Data types, grids, headroom, and weight packing: spec §3, §4.
- The twelve-operation survey and Table 1: spec §5.1–§5.12.
- Exactness contract and what a conforming engine computes: spec §6, §7.
- Conformance digests: spec §8.
