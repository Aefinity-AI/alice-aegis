# CIS-1 — Canonical Integer Semantics for Transformer Inference, v1.0.3

**Status: FROZEN 2026-08-07; v1.0.1 + v1.0.2 errata 2026-08-08** (wording and
provenance corrections from a post-publication adversarial review — §11
itemizes every change; **both conformance digests are unchanged**). This
document is normative. Changes require a new version number; the conformance
digests in §8 are never retroactively edited. The v0.1 draft and its
v0.2/v0.3 implementation notes (`CIS-1_SPEC_DRAFT_v0.1.md`) are retained as
design history; where they disagree with this document, this document
governs.

**What conformance buys:** the semantics are defined so that any conforming
implementation produces bit-identical inference output on any hardware, any
ISA, any vector width; §8's tiers are the machine-checkable evidence bar
this spec ships today (what each tier does and does not pin is stated
there). Measured so far: the operation-level digest in §8 has been
reproduced by one implementation, unmodified, on two ISAs across six
execution environments — physical SSE2-class and AVX2 machines booted into
a minimal-Linux initramfs, KVM, TCG emulation, and x86-64/aarch64 CI hosts;
the token-level digest has been reproduced on x86-64 and aarch64 CI hosts
(evidence, §9).

**Motivation** (informative): the same engine, same weights, same machine
flipped real decoded tokens between its scalar and AVX2 float kernels purely
from f32 accumulation order
(`docs/hardware_logs/e1_detprobe_crosspath_m7_2026-07-31.log`).
Floating-point inference is not a portable fact. Any scheme that needs
"re-run it anywhere, get the same bits" must remove floats from the decode
path entirely.

---

## 1. The axiom (normative)

> Every reduction in the decode path is a sum of integers whose worst case
> provably fits its accumulator.

Integer addition is associative and commutative. A conforming implementation
may therefore reorder, vectorize, tile, or parallelize any computation in any
way — SSE2, AVX2, NEON, GPU, FPGA, a for-loop in Python — and MUST produce
bit-identical results. Determinism is a property of the arithmetic, checkable
by anyone, not a discipline of kernels.

The alternative — a canonical float summation order — is rejected: it pins
every implementation to one loop structure, kills SIMD freedom, and still
breaks on FMA-vs-separate rounding.

## 2. Rounding and pinned constants (normative)

**One rounding mode everywhere: round-half-even (RNE).** Reference primitive:
`rne(num/den)` for `den > 0`, computed exactly in `i128`
(`aegis-core/src/cis.rs::rne_div`). Tie handling is normative, not a detail —
the motivating token flips came precisely from near-ties decided by float
rounding noise.

*Ratified (was v0.1 open question):* the production float path rounds
half-away-from-zero; the measured whole-path quality cost of RNE is +0.31%
perplexity (hybrid) and +0.06% (full-integer) on the M7 pinned heldout — RNE
is retained as the single mode.

Pinned transcendental constants, defined as the RNE rounding of published
50-digit decimal expansions; the integers themselves are normative:

```
TWO_PI_Q62 = 28976077832308491370   (2π,     Q2.62)
PI_Q62     = 14488038916154245685   (π,      Q2.62)
PI_2_Q62   =  7244019458077122842   (π/2,    Q2.62, independently rounded)
LOG2E_Q32  =  6196328019            (log₂ e, Q32.32)
```

## 3. Data types, grids, and headroom (normative)

| Object | Type / grid | Bound |
|---|---|---|
| Weights | ternary {−1, 0, +1}, 2-bit codes | §4 packing |
| Activations (quantized) | `i8`, symmetric | −127..+127; **−128 forbidden** |
| Dot-product accumulators | `i32` | `dim_in ≤ ⌊i32::MAX/127⌋` |
| Residual stream | `i64`, Q.20 | \|h\| < 2⁵⁰; vector length n ≤ 8192 |
| q/k/v vectors | `i32`, Q.16 | \|·\| < 2²⁹ pre-rotation (loud panic, not saturation) |
| Attention scores | `i64`, Q.24 | \|score\| ≤ 2⁵⁹ (i128 dot accumulator) |
| Probabilities | Q0.15 | Σp within ⌈T/2⌉ of 2¹⁵ |
| exp LUT entries | Q0.31 | §5.7 |
| RoPE tables | `i32`, Q1.30 | §5.9 |
| Norm gains | `i32` Q.20 (engine) / `i16` (reference op) | exact RNE from container bytes |
| Logits | `i64`, exact integer dot | conformance objects hash these |

Headroom is proof, not hope: a ternary dot of length L over i8 is bounded by
127·L; sum-of-squares by 127²·L; every op below states its widest
intermediate and the integer type that contains it exactly. Violating a
stated bound MUST be a loud failure (panic/abort), never wrap or saturation.

## 4. Weight container packing (normative)

Row-major, 4 weights per byte, low bit-pair first, `dim_in` a multiple of 4.
Codes: `00` = 0, `01` = +1, `10` = −1, and — **ratified** — `11` decodes to
**0** (defined-as-zero). Every implementation and the golden vectors pin this;
a container containing `11` is conforming and those positions contribute
nothing.

## 5. Operations (normative)

Reference implementations: `aegis-core/src/cis.rs`, `cis_attn.rs`,
`cis_infer.rs`. The reference is definitive; optimized kernels conform by
matching its bits, not by copying its loops.

### 5.1 TMV — ternary matvec

`acc[j] = Σ_i w[j,i]·a[i]` in `i32`, exact in any summation order.

**Preconditions (the rejection surface, itself normative):** an
implementation MUST reject exactly these, independent of ISA, vector width,
or internal blocking thresholds — one binary must never accept on one CPU
what it rejects on another:
1. `dim_in` not a multiple of 4;
2. `dim_in > ⌊i32::MAX/127⌋` (the exactness ceiling);
3. `output.len() < dim_out`;
4. `input.len() < dim_in`;
5. `weights.len() < dim_out·dim_in/4`.

A rejected call MUST NOT write any output element.

### 5.2 QUANT-ACT — dynamic per-token activation quantization

Per-token symmetric absmax onto the i8 grid: `q_i = rne(x_i·127 / absmax)`
with the scale carried forward as the **exact rational** 127/absmax (absmax
as `i64`; an all-zero tensor quantizes to zeros with denominator 0). No float
exists in scale representation or application. `|q_i| ≤ 127` holds by
construction; the clamp making −128 unrepresentable is explicit.

### 5.3 REQUANT and scale application

**QScale (i32 accumulators):** `y = clamp(rne((acc·M) / 2^(31+S)), −127, 127)`
with `M ∈ [2³⁰, 2³¹)` and — **ratified** — `S ≤ 62` (the smallest
representable multiplier is already below one i64 ULP; the bound keeps every
intermediate inside `i128`). The `i64`-input form (required by RMSNORM-I) is
identical arithmetic at `i128` width — **ratified** as part of this op, not a
separate one.

**Offline (M,S) generation — normative procedure:** `QScale::from_ratio`
computes the RNE-nearest `(M,S)` for any rational in (0,1) in pure integer
arithmetic, so any toolchain reproduces identical container bytes.
Made explicit in v1.0.2 (two clean-room implementations had to reconstruct
these from the conformance goldens):
- **S selection**: the unique `S` with `2^−(S+1) ≤ num/den < 2^−S` — i.e.
  maximize `S` subject to `M < 2³¹` — then `M = rne(num·2^(31+S) / den)`.
- **Rejection surface** (returns None): `num = 0`, `den = 0`, `num ≥ den`
  (the domain (0,1) is open — a ratio of exactly 1 rejects, it does not
  clamp), and ratios requiring `S > 62`.
- **Renormalization**: if RNE rounds `M` up to exactly `2³¹` with `S > 0`,
  the result is the value-identical `(2³⁰, S−1)`. Ratios within half an ULP
  of 1.0 (where `S = 0` leaves no headroom) clamp to `(2³¹−1, 0)`.

**QScale64 (runtime, i64/Q.20 domain):** a 63-bit multiplier `m·2^e`,
`m ∈ [2⁶², 2⁶³)`, built from the exact u128/u128 rational by restoring long
division with RNE at 63 significant bits. The only rounding introduced is the
multiplier's own 63-bit quantization (relative ≤ 2⁻⁶³).

### 5.4 RMSNORM-I

The complete normative procedure (every intermediate grid is
bit-determining; v1.0's prose omitted steps 2–4's grids — v1.0.1 erratum):

1. `s2 = Σ x_i²` in `i64`, exact (`n ≤ 2²³` asserted; `s2·n < 2⁶³`).
2. `t = max(isqrt(s2·n), 1)` — the **exact floor integer square root**,
   **ratified, replacing the draft's LUT+Newton sketch**: `isqrt` is a
   unique mathematical function with zero free parameters; any algorithm
   computing the exact floor conforms. `max(·,1)` guards the all-zero
   vector (whose outputs are zero regardless).
3. `inv_rms_q30 = rne(n·2³⁰ / t)`, a Q2.30 intermediate
   (`inv_rms_q30 ≤ n·2³⁰ ≤ 2⁵³`).
4. Per element: `y = rne(x_i·inv_rms_q30 / 2¹⁵)`
   (`|x_i·inv_rms_q30| ≤ 2⁶⁰`, i64-exact), then
   `out_i = REQUANT_i64(y·w_i)` (`|y·w_i| ≤ 2⁶⁰`, inside REQUANT's i128).

Folding these roundings any other way — e.g. the mathematically equivalent
single rounding `rne(x·n·2¹⁵/t)` — produces different i8 outputs and a
Tier-2 digest mismatch: the grids above are normative, not implementation
detail.

### 5.5 NORMQ — fused norm + quantization (engine form)

The per-token i8 codes are `rne(u_i·127 / max|u|)` with `u_i = h_i·g_i` — the
RMS is a common positive factor and **divides out of the codes entirely**,
surviving only in the carried exact rational scale via `t = isqrt(s2·n)`:

```
scale = max|u|·n / (127·t·2^GQ)      (real value per code unit, GQ = 20)
```

### 5.6 Container-boundary conversions

- `bf16_to_fixed` / f32-bit-pattern conversion to Q.20 by exact RNE — the
  integer side depends only on per-element float *values* (bit patterns),
  never on float accumulation. Load-time conversion vs on-the-fly is an
  implementation choice with identical produced bits.
- `f32_to_fixed`: shift bound `sh ≤ 26`, guaranteeing |v| < 2⁵⁰.
- `fix_f32_vec` (block exponent, for any f32-boundary vector): fractional
  width `G = min(F, 176 − max_exp)` where `max_exp` is the vector's largest
  f32 exponent field — exact per-element RNE at G, deterministic from the bit
  patterns alone; i8 codes are G-invariant. `G < 0` (largest element ≥ 2⁵⁰
  real, i.e. f32 exponent field ≥ 177 — v1.0 said "beyond 2⁵⁴", wrong by
  16×, contradicting its own formula; v1.0.1 erratum) MUST panic: that is
  divergence, not headroom.

### 5.7 exp machinery (shared by SOFTMAX-I, ACT-I, ROPE-I generation)

- Chain constants `C[k] = 2^(−2^(−k))`, k = 1..32, Q0.62, by the
  parameter-free floor-isqrt chain `C[0] = 2⁶¹; C[k] = isqrt(C[k−1]·2⁶²)`.
- The exp LUT: `E[i] = rne(2^(−i/1024)·2³¹)` for i = 0..1023 via the chain,
  plus exact endpoint `E[1024] = 2³⁰`. 1025 Q0.31 entries, strictly
  decreasing; **FNV-1a 64 digest `0x66C2A0EEB8C2DC43`** (u32 LE) is
  normative and golden-tested against an independent big-integer generator.
- Lookup: top 10 bits index, low 22 bits linear interpolation with one RNE
  division; monotone by construction.
- `exp_neg(z)`, z ≥ 0 in Q32.32: `y = z·LOG2E_Q32` exact; split integer and
  RNE fraction with carry fold; `rne(E-interp(f)/2ⁿ)` in Q0.31; `n ≥ 31`
  underflows to exactly 0. Monotone non-increasing, continuous at every seam.

### 5.8 SOFTMAX-I

Input on the Q.24 score grid (`i64`). Max-subtract exact;
`e_t = exp_neg((m − s_t) << 8)`; sum in `i64`;
`p_t = rne(e_t·2¹⁵ / Σe)` in Q0.15 by **exact RNE division** — **ratified,
replacing** the draft's Newton reciprocal (simpler, parameter-free, exact).
Declared bounds: `|Σp − 2¹⁵| ≤ ⌈T/2⌉`; monotone; scores more than ~21.49
real units below the max get p = 0 exactly (discarded tail < one Q0.15 ulp
of total mass at T = 512).

### 5.9 ROPE-I

Tables are Q1.30 and — **ratified, revising the draft** — **generated at
load** by a normative integer-only procedure from
`(max_seq, head_dim, f32 bits of rope_theta)` alone, rather than shipped:

1. `L = log₂(base)` in Q32.32 by shift-and-square on the f32 bit pattern
   (32 squarings in Q2.62, RNE each step; base finite, ≥ 2);
2. `a_d = rne(2d·L / head_dim)`; `inv_freq_d = 2^(−a_d)` in Q0.62 via the
   chain;
3. `θ = pos·inv_freq_d` exact, reduced mod `TWO_PI_Q62`;
4. sin/cos by quadrant reduction against the pinned π constants (a reduction
   landing 1 ulp outside [0, π/2] saturates to the boundary), then 9-term
   Taylor sums in Q0.62 — RNE requant after each multiply, exact integer
   factorial divisors, zero free coefficients;
5. RNE requant to Q1.30, clamped to ±2³⁰.

The resulting table is digest-pinned per shape (M7 shape 512×32, base
10000.0: FNV-1a 64 = `0xD8345EBF01E990FA`). Rotation: `(d, d+half)` pairing,
`rne((q0·cos − q1·sin)/2³⁰)` in i64/i128 on the Q.16 q/k grid.

### 5.10 ACT-I — MLP elementwise

On the Q.20 grid before re-quantization (**ratified, replacing** the draft's
256-entry i8 LUT sketch):
- **relu²**: exact `rne(max(0,g)²/2²⁰)·u` with one more RNE requant to Q.20 —
  two roundings total, no approximation.
- **silu**: `σ(g)` in Q0.31 from one `exp_neg(|g| << 12)` evaluation
  (`g ≥ 0: rne(2⁶²/(2³¹+t))`; `g < 0: rne(t·2³¹/(2³¹+t))`; both give exactly
  2³⁰ at g = 0), then `rne(g·σ/2³¹)` and `rne(·u/2²⁰)`.

**Block exponent on the ACT-I output (v1.0.3 erratum, normative).** After the
elementwise product lands on Q.F, a conforming engine MUST apply the §5.6
per-vector block exponent to the ACT-I output vector before it enters NORMQ /
QUANT-ACT: with `b = bits(max|v|)` (bit length of the largest magnitude),
`shift = max(0, b − 49)`, every element is RNE-right-shifted by `shift` and the
vector is carried on grid `G = F − shift`, with `G` passed to the following
NORMQ exactly as at the container boundary (§5.6). When `max|v| < 2⁴⁹` this is
the identity (`G = F`), so every v1.0 golden is unchanged. Rationale (ledger
A35): BitNet-2B relu² products exceed 2⁵⁰ on Q.20 (52–55 bits observed); the
hybrid boundary already carried this contract and the all-integer path did not.

### 5.11 ARGMAX

Ties (exact integer equality on the `i64` logits) break to the **lowest
index**. In CIS-1 a tie is an exact, reproducible event with a specified
resolution — not rounding noise.

### 5.12 Pipeline grid assignments

q/k/v from the matvec accumulator by exact rational rescale onto Q.16;
rotation growth ≤ √2 keeps values inside i32. Score dot in i128, scaled by
`1/sqrt(head_dim)` in Q0.30 from the parameter-free rule
`rne(2⁶⁰/isqrt(head_dim·2⁶⁰))`, RNE to Q.24. V-mix `Σ_t p_t·v_t` exact in
i64, RNE onto the Q.20 residual grid. LM head: exact `i64` dot of the
i8-quantized final-normed hidden state against the Q.20 embedding table.

## 6. Implementation freedom and the exactness contract (normative)

1. **Vectorize freely, match bits unconditionally.** An optimized kernel
   conforms iff byte-identical to the reference **for every input**, not
   merely for inputs a well-behaved caller produces. Known hazard, and both
   shipped vector kernels handle it the same way: x86 `vpsignb` and ARM
   `vmulq_s8` both wrap `−128 × −1` within i8, whereas the reference computes
   in i32. Although §5.2 makes −128 unreachable from the quantizer, the
   kernels detect `i8::MIN` in their (already input-touching) preparation
   pass and route the whole call to the reference — the guarantee is
   unconditional, not conditional on the caller upholding an invariant.
2. **The rejection surface is part of the op** (§5.1): whether an illegal
   call is rejected must not depend on which CPU the binary landed on or
   which side of an internal blocking threshold the shape fell.
3. Shipped vector kernels (informative): `cis_avx2` (x86-64 AVX2 —
   `vpsignb` as ternary multiply) and `cis_neon` (aarch64 — `vmulq_s8`).
   Their instruction-level claims are proven by exhaustive enumeration
   (every activation × every weight code; every byte × every pair
   position); their whole-kernel equality is verified by equivalence and
   contract test suites in `aegis-core/tests/` — a test bar, deliberately
   not called proof (v1.0.1 wording).

## 7. What a conforming engine computes (normative)

`CisMode::FullInt` (`cis_infer.rs`): embedding lookup, residual stream,
every norm, every activation quantization, all ternary matvecs, RoPE,
attention scores, softmax, V-mix, MLP elementwise, LM head, argmax — **no
float value exists anywhere in the forward pass**. Hybrid modes (f32
attention/MLP stand-ins) carry no cross-ISA claim and are outside
conformance.

## 8. Conformance (normative)

An implementation is CIS-1 v1.0 conforming iff all three tiers pass.
Emulation is valid for identity (exactly what it is invalid for in
performance), so any host can verify. (A bootable no-OS verifier appliance
is planned, not built — §10; v1.0 cited it as if it existed, v1.0.1
erratum.)

Scope of the tiers, stated plainly (v1.0.1): Tier 3's digest covers token
ids (prompt included), not per-step logits — logit-level witnessing is the
transcript format of §10, deferred; and no tier feeds `−128` activations to
the vector kernels (that contract is pinned by the kernel test suites of
§6.3, which conforming implementations should replicate, not by these
digests).

**Tier 1 — op goldens.** The golden vectors embedded in the reference test
suite (`cis.rs` unit goldens, generated by the independent big-integer
generator `scripts/cis_e2_golden_gen.py`; exp-LUT and RoPE table digests of
§5.7/§5.9).

**Tier 2 — the operation-level digest.** Run `cis_selftest`
(`aegis-linux/examples/`): 14 sections — golden vectors A1–A8 and
deterministic sweeps B1–B6 over rne_div, REQUANT, TMV shapes, QUANT-ACT,
RMSNORM-I, ARGMAX — folded LE into one FNV-1a 64 digest. It MUST print
exactly:

```
CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true
```

**Tier 3 — the token-level digest.** Run `cis_decode` on the in-repo M7
artifacts (`model-lab/tinybit/m7_final_gate_work/artifacts/`), prompt
`"Once upon a time"`, 64 new tokens, greedy, EOS ignored, `FullInt`. One
FNV-1a 64 digest over the LE bytes of every token id, prompt included (so
the tokenizer is inside the pin). It MUST print exactly:

```
CIS_DECODE digest=67e8c0a96abc04e1 prompt_toks=4 gen_toks=64 mode=fullint
```

Both digests are re-proven in CI on every commit (`arm-digest.yml`) on
x86-64 and aarch64; the workflow's failure text instructs preserving a
mismatch as a finding, never weakening the gate.

## 9. Evidence (informative)

Every claim traces to an append-only raw log in `docs/hardware_logs/` and a
row in the public research ledger:

- **One implementation, two ISAs, six execution environments, one digest**
  (op-level) — physical SSE2-class (HP Celeron N4020) and AVX2
  (Dell i5-5200U) machines booted into a minimal-Linux initramfs, KVM, TCG
  emulation (A25); aarch64 Neoverse and x86-64 CI hosts, unmodified code,
  first execution (A28). Independent *implementations* of this spec: none
  yet — that is what §4 of the roadmap (conformance) exists to invite.
- **Token-level cross-ISA identity** — complete greedy decode bit-identical
  on x86-64 and aarch64 CI hosts (A29); not yet run on physical iron.
- **Vector kernels** — AVX2 kernel 2.94× faster than the float AVX2 kernel
  at parity SIMD on the Dell i5-5200U, order-controlled same-binary A/B
  (A27); NEON kernel bit-identical on Neoverse silicon, exhaustive
  instruction-level suites (A30).
- **Cost is microarchitectural, not semantic** — scalar integer vs scalar
  float: 1.248× — slower — (Dell i5-5200U), 0.961× — faster —
  (HP Celeron N4020) (A26). Both directions occur; the semantics do not
  dictate either.
- **Quality** — M7 pinned heldout: float PPL 5.639491, full-integer 5.643085
  (+0.06%), two runs bit-identical
  (`cis1_fullint_attention_ppl_m7_i5-10210U_crosvm_2026-08-01.log`);
  production-scale ternary model (BitNet-2B) +0.74% in the
  integer-dominant *hybrid* configuration (f32 attention — outside §7
  conformance; ledger A21). 2B full-integer quality MEASURED (2026-08-27, ledger A35): +0.1239% vs float, inside the +5% line, two runs identical — after the v1.0.3 ACT-I block-exponent erratum; before it the all-integer path panicked on the NORMQ residual headroom (A35, negative half).

## 10. Not claimed by v1.0

- **Seeded sampling.** Conformance is greedy-only; the PCG32 appendix is
  deferred to a future version (no reference implementation exists — the
  spec does not freeze what code has not proven).
- **Production-scale (2B) full-integer conformance artifacts** — the
  token-level pin is small-model; the 2B leg needs hardware this program
  does not yet have.
- **The witness/transcript format** (v0.1 §3): a v0 prototype exists; the
  format freezes separately once independent verification is implemented.
- **Machine attestation** (TPM/measured boot): separate layer.
- **Training/fine-tuning semantics.**
- **Novelty of any single primitive** (TFLite-style requant, I-BERT-style
  integer ops predate this work). The claim is the contract: order-invariant
  semantics + machine-checkable conformance + a kilobyte-TCB verifier path,
  as one object.

## 11. Version history

- **v1.0.3 (2026-08-27)** — erratum from the first production-scale
  full-integer run (ledger A35): §5.10 ACT-I output now carries the §5.6
  per-vector block exponent (`shift = max(0, bits(max|v|) − 49)`, `G = F −
  shift`, `G` threaded to NORMQ). Identity at M7 ranges. Found by E1/E1b:
  BitNet-2B relu² products reach 52–55 bits on Q.20 and tripped the NORMQ
  `|h| < 2⁵⁰` precondition; attention re-entry was exact. §9 updated.
  **Both conformance digests unchanged.**
- **v1.0.2 (2026-08-08)** — errata from the clean-room sufficiency test
  (ledger A31: two independent implementations, spec text as sole source,
  both reproduce the Tier-2 digest — the v1.0.1 §5.4 procedure held on
  first attempt). Both implementers converged on the same implicit spots,
  now explicit: §5.3 `from_ratio` S-selection bracket, rejection surface
  (open domain — exact 1.0 rejects), and the `2³¹`-renormalization at
  `S > 0`. Known residual under-specifications, recorded not yet ratified:
  QUANT-ACT/RMSNORM-I slice-length preconditions, ARGMAX on empty input,
  `rne_div` outside `den > 0` (none reachable by the conformance tiers).
  **Both conformance digests unchanged.**
- **v1.0.1 (2026-08-08)** — errata from a post-publication adversarial
  review (three hostile lenses, every finding independently
  re-verified against code and logs before acceptance). **Both conformance
  digests unchanged.** Corrections: (1) §5.4 now states RMSNORM-I's full
  normative procedure — the Q2.30 inverse-RMS grid and 2¹⁵ downshift were
  bit-determining but unstated, so a clean-room implementation could fail
  Tier 2 undiagnosably; (2) §5.6 panic threshold corrected to 2⁵⁰ (was
  "2⁵⁴", a constant copied from an erroneous assert message, wrong by 16×);
  (3) header and §9 re-scoped: "six implementations" → one implementation
  across six execution environments; each digest now carries only its own
  provenance (op-level: physical machines under minimal-Linux initramfs +
  VMs + CI; token-level: CI hosts only, not yet on iron); (4) §8 no longer
  cites the bootable verifier appliance as existing, and states plainly
  that Tier 3 digests token ids (not logits) and that no tier exercises the
  −128 kernel hazard; (5) §9 quality bullet: +0.74% correctly attributed to
  ledger A21 and labeled as the hybrid (f32-attention) configuration;
  (6) "proven implementations/kernels" → proven at instruction level by
  exhaustive enumeration, verified at kernel level by test suites; machine
  named on the A27 figure. Ratified list addition: RMSNORM-I intermediate
  grids (Q2.30 inverse-RMS, 2¹⁵ downshift).
- **v1.0 (2026-08-07)** — first frozen release. Ratified from v0.1 + E2/v0.3
  notes: RNE as sole rounding mode; `11` defined-as-zero; `S ≤ 62`;
  `from_ratio` as the normative offline procedure; i64 REQUANT form; dynamic
  absmax QUANT-ACT; exact floor-isqrt (replacing LUT+Newton); exact-division
  SOFTMAX-I (replacing Newton reciprocal); load-generated Q1.30 ROPE-I
  (replacing shipped Q0.15 tables); Q.20-grid ACT-I (replacing i8 LUT);
  QScale64; block-exponent boundary conversion; the −128 kernel contract and
  ISA-independent rejection surface; conformance digests
  `76985613c965f643` / `67e8c0a96abc04e1`.
- v0.1 (2026-07-31) — draft; v0.2/v0.3 implementation notes (2026-08-01).
