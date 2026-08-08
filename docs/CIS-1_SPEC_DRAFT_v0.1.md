# CIS-1 — Canonical Inference Semantics, v0.1 (DRAFT)

> **SUPERSEDED (2026-08-07): frozen as [`CIS-1_SPEC_v1.0.md`](CIS-1_SPEC_v1.0.md).**
> This draft and its v0.2/v0.3 notes are retained as design history — the
> record of what was open, what was tried, and why each decision landed where
> it did. Where they disagree with v1.0, v1.0 governs.

**Status:** draft for internal review. Nothing here is normative until the
conformance suite exists.
**Motivating measurement:** `docs/hardware_logs/e1_detprobe_crosspath_m7_2026-07-31.log`
— the same engine, same weights, same machine, greedy decode, flipped real
tokens between its scalar and AVX2 kernels (P1 @ token 67, P3 @ token 77) purely
from f32 accumulation order. **Floating-point inference is not a portable
fact.** Any scheme that wants "re-run it anywhere and get the same bits" must
remove floats from the decode path entirely.

## 0. The axiom

> Every reduction in the decode path is a sum of integers whose worst case
> provably fits its accumulator.

Integer addition is associative and commutative. Therefore a conforming
implementation may reorder, vectorize, tile, or parallelize *any* way it
likes — SSE2, AVX2, NEON, GPU, FPGA, a for-loop in Python — and MUST produce
bit-identical results. Determinism is not an implementation discipline
(fragile, unverifiable); it is a property of the arithmetic (checkable by
anyone). This is the entire trick, and it is only available because the
weights are ternary and the activations can be made integer.

The alternative — specifying a canonical float summation order — was
rejected: it pins every implementation to one loop structure, kills SIMD
freedom, and still breaks on FMA-vs-separate rounding differences.

## 1. Data types

| Object | Type | Notes |
|---|---|---|
| Weights | ternary {−1, 0, +1}, 2-bit codes | existing A.L.I.C.E. packing |
| Activations | `i8`, symmetric, −127..127 (−128 forbidden) | per-tensor scale |
| Accumulators | `i32` (dot products), `i64` (sums of squares) | §2 headroom proof |
| Scales | fixed-point multiplier `(M: i32, S: u8)` | TFLite-style, §3 |
| Logits | `i32`, exact | the witness hashes these |

Headroom: a ternary dot of length L with i8 activations is bounded by
127·L. `i32` overflows only at L > 16.9M; the largest hidden dim in scope
(6912) uses 0.04% of the range. Sum-of-squares for norms is bounded by
127²·L < 2⁶⁴ for any conceivable L.

## 2. Operations (normative once frozen)

**TMV — ternary matvec/matmul.** `acc[j] = Σ_i w[j,i]·a[i]` in i32. No
rounding exists in this op; it is exact in any order. (This is the property
f32 lacks, and the only reason CIS-1 can exist.)

**REQUANT — i32 → i8.** `y = clamp(rne((acc as i64 · M) >> (31+S)), −127, 127)`
with `rne` = round-half-even, `M ∈ [2³⁰, 2³¹)`, per-tensor `(M,S)` computed
offline and shipped in the model container. This is the gemmlowp/TFLite
quantized multiplier — proven op, borrowed deliberately; CIS-1's novelty is
the system contract, not this primitive.

**RMSNORM-I.** Sum of squares in i64 (exact); inverse square root by integer
Newton–Raphson: seed from a 64-entry LUT on the exponent, exactly k=3
iterations in Q2.30, constants normative. Multiply, REQUANT. Same bits
everywhere or non-conforming.

**SOFTMAX-I (attention).** Max-subtract (exact integer compare); `exp` via a
normative Q6.10-indexed LUT of 1024 entries with linear interpolation in
Q0.15; sum in i64; divide via fixed-point reciprocal (Newton, k=2,
normative constants); RNE rounding. Probabilities in Q0.15.

**ROPE-I.** sin/cos as Q0.15 tables *shipped in the model container*,
generated offline by a normative procedure. No runtime trigonometry.

**ACT-I (SiLU).** Normative 256-entry i8→i8 LUT per scale regime, generated
offline by a normative procedure.

**ARGMAX.** Ties (exact i32 equality) break to the LOWEST index. (E1's flip
happened because f32 near-ties are decided by rounding noise; in CIS-1 a tie
is an exact, reproducible event with a specified resolution.)

**SAMPLE (optional mode).** PCG32 seeded from the transcript header; CDF
sampling over Q0.15 probabilities with normative rounding. Deterministic
given (seed, logits). v0.1 conformance requires greedy; seeded sampling is
Appendix A.

## 3. The transcript (the product)

A CIS-1 transcript makes one inference a *portable, checkable object*:

```
header  = {cis_version, sha256(model container), sha256(tokenizer),
           config digest, prompt bytes, mode(greedy|seed), engine id (informative)}
h(0)    = SHA256(header)
h(t)    = SHA256(h(t−1) ‖ token_id(t) ‖ SHA256(logits_i32(t)))
witness = {header, token_ids[1..T], h(T)}
```

Hashing the **exact i32 logit vector** each step means the witness detects
any tamper — weight edit, prompt swap, different model, bugged kernel — even
when the emitted tokens happen to match. Verification is re-execution: any
conforming implementation, on any hardware, recomputes h(T). Scope stated
honestly (red-team corrected): replay verifies systems that RUN a
conforming engine — this is *auditable-by-construction* inference, not an
audit of arbitrary third-party stacks. Within that scope, any commodity
CPU is a verifier, and an emulator is too (emulation is valid for
*identity*, which is precisely what it is invalid for in performance — the
program's Rule A, inverted into an asset).

## 4. Conformance

An implementation is CIS-1 conforming iff it reproduces (a) the per-op golden
vectors and (b) the end-to-end golden transcripts, bit-for-bit. The golden
corpus IS the spec's court of appeal — the same bit-exactness discipline this
program already runs, promoted to a public contract.

## 5. What v0.1 does not claim

- No training, no fine-tuning semantics.
- No claim that i8 activations are accuracy-free: **E2 (preregistered) must
  measure the perplexity delta** on M7 and BitNet-2B. Kill criterion lives in
  the preregistration, not here.
- No cryptographic proof that a transcript came from a specific *machine*
  (that is the TPM/measured-boot layer, separate deliverable).
- No claim of novelty for any single primitive (TFLite requant, I-BERT-style
  integer ops, PCG32 all predate us). The claim is the **contract**:
  order-invariant semantics + re-executable witness + kilobyte-TCB reference
  verifier, as one object.

## 6. Reference implementations (planned)

1. `aegis-core` CIS mode — shares the ternary packing already in production.
2. A deliberately naive, obviously-correct scalar Rust implementation
   (~500 lines) — the *auditor's* implementation; slow is fine, readable is
   the point.
3. The UEFI unikernel as the minimal-TCB verifier appliance: boots from a
   stick, verifies transcripts, no OS in the loop.

---

## v0.2 E2 implementation notes (2026-08-01)

E2 built the first integer-dominant end-to-end forward path
(`aegis-core/src/cis_infer.rs`, unit goldens from the independent big-int
generator `scripts/cis_e2_golden_gen.py`) and measured the teacher-forced
perplexity delta against the float engine on M7's pinned heldout sample
(471 tokens). Result (raw log:
`docs/hardware_logs/cis1_e2_int_vs_float_ppl_m7_i5-10210U_crosvm_2026-08-01.log`):
float PPL 5.639491, integer PPL 5.657126, **relative delta +0.3127% — PASS**
against the preregistered +5% kill line, and bit-identical PPL + argmax
digest across two runs. Where the v0.1 draft was silent, E2 made the
following choices; each needs ratification (or replacement) before v1.0.

### E2-1. Scope actually integerized (the hybrid declaration)

Integer, exact: embedding lookup, the residual stream, every RMSNorm, every
activation quantization, all seven ternary matvecs per layer
(q/k/v/o/gate/up/down), the LM head, and argmax.

Declared f32 hybrid stages (temporary stand-ins for SOFTMAX-I / ROPE-I /
ACT-I, whose normative constants do not exist yet):
1. the attention core — RoPE, score dots, softmax, probability-weighted V
   mix — between the q/k/v projections and the o_proj input quantization;
2. the MLP elementwise stage — `silu(gate) · up` — between the up/gate
   matvecs and the down_proj input quantization.

Values re-enter integer land only through `f32_to_fixed`, an exact
integer-only conversion of the f32 *bit pattern* (RNE): the integer side
never depends on float accumulation order, only on per-element float values.
Consequence stated honestly: until SOFTMAX-I/ROPE-I/ACT-I land, cross-path
(scalar vs AVX2) bit-identity is NOT yet claimed — the hybrid attention
stage still uses the production f32 kernels. Same-machine, same-kernel-path
determinism is claimed and exhibited (identical FNV-1a 64 argmax digest).

### E2-2. Fixed-point conventions

- Residual stream: `i64`, Q.20 (`F = 20`). Embeddings convert BF16 → Q.20
  by exact RNE (`bf16_to_fixed`); the same table serves the tied LM head.
  Whether the conversion happens once at load or per row on the fly is an
  implementation choice with identical produced bits (the 2B-scale
  implementation converts on the fly: a materialized i32 table is ~0.5 GB
  at vocab 50,256 × hidden 2560).
- Norm gains: `i32`, Q.20 (`GQ = 20`), converted from the checkpoint's
  BF16/F32 bytes by exact RNE (element size derived from the byte length,
  same rule as the production `rmsnorm`).
- Weight scales: exact f32 decomposition `(−1)^neg · m · 2^e`, `m` odd —
  no rounding exists in this conversion.

### E2-3. RMSNORM-I + dynamic activation quantization, fused (`normq`)

The per-token absmax i8 codes are `rne(u_i·127 / max|u|)` with
`u_i = h_i·g_i` — the RMS is a common positive factor and **divides out of
the codes entirely**. It survives only in the carried exact rational scale,
via `t = isqrt(s2·n)` (the same exact-isqrt stand-in as `cis::rmsnorm_i`;
`max(t,1)` replaces the float path's epsilon):

    scale = max|u|·n / (127·t·2^GQ)   (real value per code unit)

This resolves cis.rs SPEC GAP "dynamic absmax quantization" and reinforces
the recommendation to ratify exact-isqrt over the unimplementable LUT+Newton
mechanism.

### E2-4. Scale application: `QScale64`

Per-(token, matvec) rescaling onto the Q.20 residual grid uses a 63-bit
fixed-point multiplier `m·2^e`, `m ∈ [2^62, 2^63)`, built from the exact
u128/u128 rational by restoring long division with RNE at 63 significant
bits (no intermediate overflows for any operands ≤ 2^127). This is
`cis::QScale` widened to the i64 domain and computed at runtime from exact
integer scale state; the only rounding introduced is the multiplier's own
63-bit quantization (relative ≤ 2^−63). Needs a spec home next to REQUANT.

### E2-5. LM head and the E2 rounding-mode arbitration

The LM head is an exact `i64` dot of the i8-quantized final-normed hidden
state against the Q.20 embedding table; argmax ties break to the lowest
index on the *integer* logits (spec §2 ARGMAX). Witness digests should hash
these integer logits.

cis.rs flagged that the integer path rounds RNE while the trained-against
float path rounds half-away-from-zero, and deferred to E2. E2's answer:
with RNE everywhere, the whole-path quality cost on M7 is +0.31% PPL —
**RNE is retained** as the single normative rounding mode.

### E2-6. Known deltas vs the float path (accepted, measured)

The +0.31% aggregates: Q.20 residual quantization, i8 quantization of the
LM-head input (the float path dots full-precision f32), exact-isqrt RMS vs
float sqrt+eps, RNE vs half-away rounding, and f32→fixed re-entry rounding
at the two hybrid boundaries. None of these is individually attributed by
E2; the gate only needed the aggregate.

### E2-7. Headroom recalibration at BitNet-2B width (2026-08-01)

The E2 implementation's original guards — `f32_to_fixed` refusing `sh > 20`
(hybrid re-entry values ≥ 2^24 real), `normq`/`quantq` refusing residuals
≥ 2^45 and vectors longer than 4096 — were sized to M7 (hidden 1024-class,
inter ≤ 4096). Both are violated by the real BitNet-2B artifact: inter is
6912, and the relu²-gate MLP product routinely exceeds 2^24 before
`ffn_sub_norm` tames it (this is why BitNet carries SubLN at all). The
`f32_to_fixed` assert fired on the first 2B forward step — a FINDING about
guard calibration, not about Q.20 itself: the format's exactness argument
never depended on the 2^44 bound, only on every intermediate staying inside
i128/u128.

Recalibrated bounds, with the derivation that makes them exact:

- vector length `n ≤ 8192` (covers inter 6912);
- residual / boundary magnitude `|h| < 2^50` — `normq`'s widest
  intermediate is `s2·n ≤ n²·2^2·50 = 2^26·2^100 = 2^126 < 2^128`, and
  `u = h·g < 2^50·2^31 = 2^81` fits i128; `num = a·n ≤ 2^94`,
  `den = 127·(t·2^GQ) ≤ 2^90`;
- `f32_to_fixed`: `sh ≤ 26`, guaranteeing `|v| < 2^24·2^26 = 2^50` (the
  f32 mantissa is < 2^24).

A fixed per-element cap is still not enough: measured BitNet-2B relu² MLP
products exceed 2^30 real, i.e. no static Q.20 window under the 2^50
headroom holds them. The boundary therefore uses a **per-vector dynamic
fractional width** (`fix_f32_vec`, a block exponent): each hybrid vector is
fixed at `G = min(F, 176 − max_exp)` fractional bits, where `max_exp` is
the vector's largest f32 exponent field — the largest width that keeps
every element under 2^50. This is exact (per-element RNE at `G`),
deterministic (a function of the f32 bit patterns only), and scale-correct:
`normq`'s i8 codes are invariant in `G` (exact ratios), its carried
rational agrees across `G` up to the exact-isqrt floor granularity
(relative O(1/t) — that floor is already normative at Q.20), and `quantq`
carries `G` in its denominator (`127·2^G`). Values beyond 2^54
real (`G < 0`) still panic loudly — that is divergence, not headroom.

The recalibration changes NO produced bit for any vector the old guards
admitted (`G = F` whenever every element is < 2^24 real, which covers all
of M7); the M7 A19 digest 0x42E820C2A8A59CD6 must and does reproduce.
Goldens: large-value `f32_to_fixed` cases and a ±2^49 `normq` case
(constants from the independent `scripts/cis_e2_golden_gen.py`), plus
`fix_f32_vec` block-exponent selection and `normq` G-invariance tests.
Downstream `residual_qscale`/`QScale64` paths were already
`checked_*`-guarded and keep their own loud-failure behavior.

---

## v0.3 integer-attention notes (2026-08-01)

v0.3 closes E2's declared hybrid gap: ROPE-I, SOFTMAX-I and ACT-I now exist
(`aegis-core/src/cis_attn.rs`), and `cis_infer` gained a `CisMode::FullInt`
forward path in which **no float value exists anywhere** — attention core and
MLP elementwise included. The hybrid path is retained unchanged for A/B (its
argmax digest on the M7 pinned sample is bit-identical before/after this
change). Measured on M7's pinned 471-token heldout (raw log:
`docs/hardware_logs/cis1_fullint_attention_ppl_m7_i5-10210U_crosvm_2026-08-01.log`):
float PPL 5.639491, hybrid 5.657126 (+0.3127%), **full-integer 5.643085
(+0.0637%) — PASS** against the preregistered +5% kill line, two runs
bit-identical (PPL bits + FNV-1a 64 argmax digest 0xBED4A17A1A5EE296).
Notably the full-integer path is *closer* to float than the hybrid: the two
f32→fixed re-entry quantizations it removes cost more than its own Q0.15
probability / Q1.30 table quantization.

Because every loop in the full-integer path is scalar integer Rust (the
`cis`/`cis_attn` ops have no SIMD dispatch; `ops.rs`'s force_scalar toggle
governs only the float kernels, none of which are called), cross-kernel-path
identity is **true by construction** — there is no second path to diverge.
Cross-ISA identity follows from the arithmetic being exact integer ops; an
actual second-machine replay remains the E4 jury exhibit.

Where the draft was silent or v0.1's §2 sketches were unimplementable
(missing constants), v0.3 made the following choices. Each needs
ratification before v1.0.

### v0.3-1. Rounding and pinned constants

One rounding primitive everywhere: `rne` (round-half-even), as in E2. Two
transcendental constants are pinned as integer literals, defined as the RNE
rounding of published 50-digit decimal expansions:

    TWO_PI_Q62 = 28976077832308491370      (2π, Q2.62)
    PI_Q62     = 14488038916154245685      (π,  Q2.62)
    PI_2_Q62   =  7244019458077122842      (π/2, Q2.62)
    LOG2E_Q32  =  6196328019               (log2 e, Q32.32)

`PI_2_Q62` is independently rounded (not `PI_Q62/2`); the pinned integers
themselves are normative.

### v0.3-2. exp machinery (shared by SOFTMAX-I, ACT-I silu, ROPE-I tables)

- **Chain constants** `C[k] = 2^(−2^(−k))`, k = 1..32, Q0.62, by the
  parameter-free floor-isqrt chain `C[0] = 2^61; C[k] = isqrt(C[k−1]·2^62)`.
- **`exp2_neg_frac(f)`**: product over set bits of a Q0.32 fraction, RNE
  requant to Q0.62 per multiply. Table-generation-time only.
- **The exp LUT** (normative constants, replacing v0.1's unimplemented
  "Q6.10-indexed LUT"): `E[i] = rne(2^(−i/1024)·2^31)` for i = 0..1023 via
  the chain, plus the exact endpoint `E[1024] = 2^30`. 1025 Q0.31 entries;
  FNV-1a 64 digest 0x66C2A0EEB8C2DC43 (u32 LE), golden-tested against the
  independent Python big-int generator.
- **Lookup**: top 10 bits index, low 22 bits linear-interpolate with one RNE
  division. Entries are strictly decreasing, so lookup is provably monotone.
- **`exp_neg(z)`** for z ≥ 0 in Q32.32: `y = z·LOG2E_Q32` exact; split
  `n = ⌊y⌋`, `f = rne(frac(y))` with carry fold; result
  `rne(E-interp(f)/2^n)` in Q0.31; `n ≥ 31` underflows to exactly 0.
  Monotone non-increasing (property-tested), continuous at every seam.
  Relative error is interpolation-dominated, ≈ 2^−24.

### v0.3-3. SOFTMAX-I (revises v0.1 §2)

Input: scores on the Q40.24 grid (i64). Max-subtract exact; per-element
`e_t = exp_neg((m − s_t) << 8)` (the Q.24→Q.32 shift is exact); sum in i64
(≤ T·2^31); probabilities `p_t = rne(e_t·2^15 / Σe)` in **Q0.15** (as v0.1
declared). v0.1's "fixed-point reciprocal (Newton, k=2)" is replaced by the
exact RNE division — simpler, parameter-free, and exact. Declared bounds
(property-tested): `|Σp − 2^15| ≤ ⌈T/2⌉`; monotone (`s_i ≥ s_j ⇒ p_i ≥
p_j`); scores > ~21.49 real units below the max get p = 0 exactly, and that
discarded tail is < 2.4e−7 total mass at T = 512 — below one Q0.15 ulp.

### v0.3-4. ROPE-I (revises v0.1 §2: generated at load, not shipped)

Tables are Q1.30 (not v0.1's Q0.15 — attention quality is table-precision
sensitive and the storage is trivial), and are **generated at load** by a
normative integer-only procedure rather than shipped in the container: the
procedure is bit-reproducible on any platform from (max_seq, head_dim,
f32 bits of rope_theta) alone, and the resulting table is digest-pinned
(M7 shape 512×32, base 10000.0: FNV-1a 64 = 0xD8345EBF01E990FA). Steps:

1. `L = log2(base)` in Q32.32 by shift-and-square on the f32 bit pattern
   (32 squarings in Q2.62, RNE each step; base must be finite, ≥ 2);
2. `a_d = rne(2d·L / head_dim)`; `inv_freq_d = 2^(−a_d)` in Q0.62 via the
   chain (absolute error ≲ 2^−62, so pos·error < 2^−52 rad at pos 512);
3. `θ = pos · inv_freq_d` exact, reduced `mod TWO_PI_Q62`;
4. sin/cos by quadrant reduction against the pinned π constants (a
   reduction landing 1 ulp outside [0, π/2] saturates to the boundary),
   then 9-term Taylor sums in Q0.62 — RNE requant after each multiply,
   exact integer factorial divisors, **zero free coefficients**. Truncation
   < 2^−40; values may exceed 2^62 by ~4.4e−14 at π/2 (absorbed by the
   requant clamp);
5. RNE requant to Q1.30, clamped to ±2^30.

Rotation: same (d, d+half) pairing as the production kernel,
`rne((q0·cos − q1·sin)/2^30)` in i64/i128, on the engine's Q.16 q/k grid.
Property-tested: norm preservation within 2^−13 relative.

### v0.3-5. ACT-I (revises v0.1 §2: no 256-entry i8 LUT)

v0.1 sketched a per-scale-regime 256-entry i8→i8 LUT; that is the wrong
shape for this engine — the MLP elementwise stage runs on the Q.20 grid
*before* re-quantization, where it costs nothing to be accurate. Instead:

- **relu²** (BitNet): exact `rne(max(0,g)²/2^20)·u` with one more RNE
  requant to Q.20. Two RNE roundings total, no approximation.
- **silu**: `σ(g)` in Q0.31 from one `exp_neg(|g| << 12)` evaluation —
  `g ≥ 0: rne(2^62/(2^31+t))`; `g < 0: rne(t·2^31/(2^31+t))`; both give
  exactly 2^30 at g = 0 (seam golden-tested). Then `rne(g·σ/2^31)` and
  `rne(·u/2^20)`.

### v0.3-6. Full-integer engine grids and headroom (hidden 384–2560,
### head_dim ≤ 128, seq ≤ 512)

- q/k/v: i32 on Q.16, from the matvec accumulator by the exact
  rational-rescale machinery (E2's `fixed_qscale`, the F-parameterized
  generalization of `residual_qscale`). The engine asserts |·| < 2^29
  (real magnitude < 8192 — decade-scale headroom over observed
  activations; a violation is a loud conformance panic, not saturation).
  Rotation growth ≤ √2 keeps values < 2^30, inside i32.
- score dot: i128 accumulator (|Σ q·k| ≤ 128·2^60 = 2^67 — no i64 proof
  needed, exact by construction), scaled by `1/sqrt(head_dim)` in Q0.30
  from the parameter-free rule `rne(2^60/isqrt(head_dim·2^60))` (exact
  2^27 for head_dim 64), RNE to the Q.24 score grid: |score| ≤ 2^59 < i64.
- exp argument: `z ≤ 2^60` in Q.24 → `z·2^8·LOG2E ≤ 2^101`, exact in u128.
- V mix: `Σ_t p_t·v_t` ≤ 512·2^15·2^30 = 2^54, exact in i64; RNE by
  2^(16+15−20) onto the Q.20 residual grid, |·| ≤ 2^43 < the 2^45 normq
  bound.
- MLP: gate/up on Q.20 asserted < 2^40; ACT-I intermediates in i128;
  outputs bounded by the normq/quantq 2^45 assert.

### v0.3-7. Memory note

FullInt allocates an integer KV cache (2 × layers·max_pos·kv_dim i32):
3.7 MB for M7. BitNet-2B at 4096 positions would need ~630 MB — full-int
BitNet is deferred (lane B scope) and should shorten the window or store
K/V in i16 after a measured range audit before running on the 6 GB dev box.

### v0.3-8. What v0.3 does not claim

- No cross-ISA replay has been *executed* yet (E4 exhibit); the claim here
  is arithmetic (exact integer ops, no dispatch), not yet demonstrated on
  a second machine.
- The NLL/PPL numbers themselves are f64 *scoring* of exact integer
  logits; the conformance objects are the logits/argmax digests, not the
  PPL float.
- BitNet-2B full-integer quality is unmeasured (M7 only).
