//! CIS-1 — Canonical Inference Semantics, reference ops.
//!
//! Spec: `docs/CIS-1_SPEC_v1.0.md` (FROZEN 2026-08-07). Every `SPEC GAP`
//! marked below is now ratified in v1.0 exactly as implemented here — the
//! markers are retained as the record of what was once open. Motivation:
//! `docs/hardware_logs/e1_detprobe_crosspath_m7_2026-07-31.log` — the same
//! engine flipped real tokens between its scalar and AVX2 paths purely from
//! f32 accumulation order. Floating-point inference is not a portable fact.
//!
//! The axiom (spec §0): every reduction in the decode path is a sum of
//! integers whose worst case provably fits its accumulator. Integer addition
//! is associative and commutative, so a conforming implementation may
//! reorder, vectorize, tile, or parallelize any way it likes and MUST produce
//! bit-identical results. Determinism is a property of the arithmetic, not a
//! discipline of kernels. This module is the deliberately simple reference:
//! readable, exact, and the source of the unit golden vectors. Optimized
//! kernels conform by matching its bits, not by copying its loops.
//!
//! Everything here is `core`-only integer arithmetic — no floats, no `libm`,
//! no `unsafe`.

/// Round-to-nearest-even division: `rne(num / den)` for `den > 0`, exact.
///
/// This is the single rounding primitive of the module — REQUANT, dynamic
/// activation quantization, and RMSNORM-I all round through this function,
/// so "rne" means one thing everywhere (spec §2 names round-half-even; E1's
/// token flips came precisely from near-ties decided by float rounding
/// noise, so tie handling is normative, not a detail).
///
/// `i128` keeps every caller's intermediate exact: the largest product any
/// op forms is |acc| · M < 2^63 · 2^31 = 2^94.
pub fn rne_div(num: i128, den: i128) -> i128 {
    debug_assert!(den > 0, "rne_div: denominator must be positive");
    let q = num.div_euclid(den);
    let r = num.rem_euclid(den); // 0 <= r < den, exact
    // Round up when the fraction exceeds 1/2, or equals 1/2 and q is odd
    // (so the result lands on the even neighbor).
    if 2 * r > den || (2 * r == den && q & 1 != 0) {
        q + 1
    } else {
        q
    }
}

/// Round-half-to-even step given an exact floor quotient `q` and exact
/// remainder `r` (`0 <= r < den`) of `num / den`, i.e. the second half of
/// what `rne_div` computes — shared here so `normq`, `quantq`, and
/// `rne_shr_i128` can supply `(q, r)` from a division-free path (or a
/// cheaper narrower division) and still apply the identical, normative
/// rounding rule.
#[inline]
pub fn rne_round(q: i128, r: i128, den: i128) -> i128 {
    debug_assert!(den > 0 && r >= 0 && r < den, "rne_round: r out of range");
    if 2 * r > den || (2 * r == den && q & 1 != 0) {
        q + 1
    } else {
        q
    }
}

/// Exact RNE division of a signed i128 by 2^k (1 ≤ k ≤ 126) in pure
/// shift/mask arithmetic — bit-identical to `rne_div(p, 1i128 << k)` for
/// ANY sign of `p` (asserted exhaustively/randomly in tests). Unlike
/// `rne_shr` (nonnegative i64 only, `cis_infer`), this covers the signed
/// i128 products that `QScale64::rescale`, `f32_to_fixed`, `fix_q_vec`, and
/// the FullInt attention score dot / V-mix requants divide by a runtime
/// power of two — `rne_div`'s i128/i128 division there compiles to
/// `compiler_builtins::u128_div_rem` (~100 cycles) and dominates both the
/// FullInt Act phase (~5 G ticks / 15% of a 2B verify) and the Attn phase.
/// An arithmetic right shift is `div_euclid` by `2^k` (floor), and masking
/// off the low `k` bits is `rem_euclid` — so the exact quotient/remainder
/// pair `rne_div` computes via `i128::div_euclid`/`rem_euclid` is available
/// here without a divide; `rne_round` then applies the single normative
/// rounding rule to it.
#[inline]
pub fn rne_shr_i128(p: i128, k: u32) -> i128 {
    debug_assert!((1..=126).contains(&k));
    let floor = p >> k; // arithmetic shift == div_euclid(p, 2^k) for any sign
    let rem = p & ((1i128 << k) - 1); // == rem_euclid(p, 2^k), 0 <= rem < 2^k
    rne_round(floor, rem, 1i128 << k)
}

/// Fixed-point requantization multiplier `(M, S)`: value = M / 2^(31+S),
/// with `M ∈ [2^30, 2^31)` (spec §1, TFLite/gemmlowp lineage — borrowed
/// deliberately; CIS-1's novelty is the contract, not this primitive).
/// Computed offline, shipped in the model container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QScale {
    m: i32,
    s: u8,
}

impl QScale {
    /// Construct a validated multiplier. Panics on an out-of-range `M`
    /// rather than degrading: a non-normalized multiplier silently changes
    /// the produced bits, which is a conformance break, not a warning.
    ///
    /// SPEC GAP: the draft leaves `S: u8` unbounded. This implementation
    /// requires `S <= 62` (smallest representable multiplier 2^30/2^93 =
    /// 2^-63): even an i64 accumulator times 2^-63 is below 1, so larger S
    /// can only ever produce 0, and the bound keeps every intermediate
    /// provably inside `i128`. Needs ratification.
    pub const fn new(m: i32, s: u8) -> Self {
        assert!(m >= 1 << 30, "CIS-1 spec §2: M must be in [2^30, 2^31)");
        assert!(s <= 62, "CIS-1 (this impl): S bounded to 62 — see SPEC GAP");
        QScale { m, s }
    }

    pub const fn m(self) -> i32 {
        self.m
    }

    pub const fn s(self) -> u8 {
        self.s
    }

    /// Offline generator: the `(M, S)` pair whose value M / 2^(31+S) is the
    /// round-to-nearest-even representation of `num / den`, for ratios in
    /// (0, 1). Integer-only, so any toolchain reproduces the same container
    /// bytes. Returns `None` for ratios >= 1, ratios too small for `S <= 62`,
    /// and degenerate inputs.
    ///
    /// A ratio within half an ULP of 1.0 (unrepresentable: max value is
    /// (2^31−1)/2^31) clamps to `(2^31−1, 0)`, the nearest representable.
    ///
    /// SPEC GAP: the draft says `(M, S)` are "computed offline" but does not
    /// give the procedure. This is a candidate normative procedure.
    pub fn from_ratio(num: u64, den: u64) -> Option<QScale> {
        if num == 0 || den == 0 || num >= den {
            return None;
        }
        // Smallest e with floor(num·2^e / den) >= 2^30; then M = rne at that
        // e is in [2^30, 2^31] (floor at e−1 below 2^30 caps floor at e
        // below 2^31). num·2^e stays under 2^31·den < 2^95: exact in u128.
        let mut e = 31u32;
        while ((num as u128) << e) / (den as u128) < (1 << 30) {
            e += 1;
            if e - 31 > 62 {
                return None; // multiplier below 2^-63: out of bounded-S range
            }
        }
        let m = rne_div((num as i128) << e, den as i128);
        let (m, e) = if m == 1i128 << 31 {
            if e == 31 {
                // Ratio rounds to 1.0 exactly: clamp to the largest
                // representable multiplier.
                return Some(QScale::new(i32::MAX, 0));
            }
            (1i128 << 30, e - 1)
        } else {
            (m, e)
        };
        Some(QScale::new(m as i32, (e - 31) as u8))
    }
}

/// REQUANT (spec §2): `y = clamp(rne((acc · M) / 2^(31+S)), −127, 127)`.
///
/// −128 never appears (spec §1 forbids it: a future `_mm256_sign_epi8`
/// kernel would saturate −(−128) to +127 and silently flip a sign).
pub fn requant_i32(acc: i32, q: QScale) -> i8 {
    requant_i64(acc as i64, q)
}

/// REQUANT over an `i64` accumulator (RMSNORM-I's weighted products exceed
/// i32). Exact: |acc| · M < 2^63 · 2^31 = 2^94 fits `i128`, and the divisor
/// 2^(31+S) <= 2^93 by the `S <= 62` bound.
///
/// SPEC GAP: the draft types REQUANT as i32 → i8 only; the i64-input form
/// is required by RMSNORM-I as implemented here and needs a spec home.
pub fn requant_i64(acc: i64, q: QScale) -> i8 {
    let prod = acc as i128 * q.m as i128;
    let den = 1i128 << (31 + q.s as u32);
    rne_div(prod, den).clamp(-127, 127) as i8
}

/// One 2-bit weight code (existing A.L.I.C.E. packing, spec §1):
/// `00` = 0, `01` = +1, `10` = −1.
///
/// SPEC GAP: the undefined `11` code decodes to 0 here, mirroring the
/// production f32 kernels (`ops::UNPACK_LUT`) so both paths agree on any
/// byte stream. The spec must decide whether `11` is defined-as-zero or
/// makes the container non-conforming; the golden vectors below pin the
/// defined-as-zero behavior until then.
const fn wcode(code: u8) -> i32 {
    match code & 0b11 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

/// TMV (spec §2): `acc[j] = Σ_i w[j,i] · a[i]` in `i32`, exactly.
///
/// No rounding exists in this op; it is exact in ANY summation order — this
/// is the property f32 lacks and the only reason CIS-1 can exist. The loop
/// below runs ascending for readability, but ascending order is NOT part of
/// the semantics; the order-invariance test asserts exactly that.
///
/// Packing matches `ops::ternary_matvec`: row-major, 4 weights per byte,
/// low bit-pair first, `dim_in` a multiple of 4.
///
/// Headroom (spec §1): |acc| <= 127·dim_in, so i32 is exact for
/// dim_in <= 16.9M; the largest hidden dim in scope (6912) uses 0.04% of
/// the range.
/// The TMV contract, in one place.
///
/// Every implementation of this op must reject exactly the same inputs. When
/// these lived inline in the scalar body, the AVX2 kernel silently answered
/// four classes of illegal call that the reference rejects — and *which*
/// behaviour you got depended on whether the CPU had AVX2 and on whether the
/// shape happened to clear the kernel's blocking threshold. One binary, one
/// input, two observable behaviours, decided by the ISA: precisely what CIS-1
/// exists to make impossible. Enforced once here so a future implementation
/// cannot drift from it.
#[inline]
pub(crate) fn check_tmv_preconditions(
    output: &[i32],
    input: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
) {
    assert!(
        dim_in.is_multiple_of(4),
        "CIS-1 packing is 4 weights per byte"
    );
    assert!(
        dim_in <= (i32::MAX / 127) as usize,
        "CIS-1 spec §1 headroom: dim_in too large for an exact i32 dot"
    );
    assert!(output.len() >= dim_out, "output shorter than dim_out");
    assert!(input.len() >= dim_in, "input shorter than dim_in");
    assert!(
        weights_packed.len() >= dim_out * (dim_in / 4),
        "weights shorter than dim_out rows"
    );
}

pub fn ternary_matvec_i8(
    output: &mut [i32],
    input: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
) {
    check_tmv_preconditions(output, input, weights_packed, dim_out, dim_in);
    let packed_dim_in = dim_in / 4;

    let (input4, _) = input[..dim_in].as_chunks::<4>();
    for (row, out) in output.iter_mut().enumerate().take(dim_out) {
        let w_row = &weights_packed[row * packed_dim_in..(row + 1) * packed_dim_in];
        let mut acc: i32 = 0;
        for (&b, a4) in w_row.iter().zip(input4) {
            acc += wcode(b) * a4[0] as i32
                + wcode(b >> 2) * a4[1] as i32
                + wcode(b >> 4) * a4[2] as i32
                + wcode(b >> 6) * a4[3] as i32;
        }
        *out = acc;
    }
}

/// Dynamic per-token activation quantization onto the i8 grid, integer-only.
///
/// The integer analog of `ops::quantize_activations_int8` (per-token absmax,
/// symmetric, −128 forbidden — the grid the BitNet weights were trained
/// against). The scale is the exact rational 127/absmax — no float exists in
/// scale representation or application: `q_i = rne(x_i · 127 / absmax)`.
/// Returns the absmax denominator (as i64: an `i32::MIN` input has no i32
/// absolute value) so the caller can carry the scale forward exactly; an
/// all-zero tensor quantizes to zeros with denominator 0.
///
/// |q_i| <= 127 holds by construction (|x_i| <= absmax); the clamp makes the
/// REQUANT-shared invariant explicit rather than implicit.
///
/// SPEC GAP (two decisions needing ratification):
/// 1. The draft ships static per-tensor `(M, S)` scales but is silent on the
///    runtime *dynamic* absmax quantization BitNet actually uses per token.
///    Exact-rational RNE division is the simplest bit-reproducible choice.
/// 2. Rounding mode: the production f32 path rounds half away from zero
///    (`libm::roundf`); RNE here is consistent with REQUANT but diverges
///    from the grid the weights saw in training. E2 (perplexity gate) must
///    arbitrate.
pub fn quantize_activations_i32(output: &mut [i8], input: &[i32]) -> i64 {
    assert!(output.len() >= input.len(), "output shorter than input");
    let mut absmax: i64 = 0;
    for &v in input {
        let a = (v as i64).abs();
        if a > absmax {
            absmax = a;
        }
    }
    if absmax == 0 {
        for o in output[..input.len()].iter_mut() {
            *o = 0;
        }
        return 0;
    }
    for (o, &v) in output.iter_mut().zip(input) {
        let q = rne_div(v as i128 * 127, absmax as i128);
        *o = q.clamp(-127, 127) as i8;
    }
    absmax
}

/// RMSNORM-I (spec §2) — stand-in reference semantics.
///
/// Exact sum of squares in i64, inverse RMS as ONE rne-rounded Q2.30 value,
/// per-channel gain multiply, REQUANT. Steps, in exact integer arithmetic:
///
/// 1. `s2 = Σ x_i²` (i64, exact; <= 127²·n)
/// 2. `t = max(isqrt(s2 · n), 1)` — floor square root, so `t ≈ n·rms`;
///    exact and parameter-free
/// 3. `inv_rms_q30 = rne(n · 2^30 / t)` — (1/rms) in Q2.30
/// 4. per element: `y = rne(x_i · inv_rms_q30 / 2^15)` (Q0.15-scaled x/rms),
///    `out_i = REQUANT_i64(y · w_i)`
///
/// The gain `w` is an integer vector shipped in the model container; its
/// fixed-point interpretation (e.g. Q1.14) is folded into `(M, S)` offline,
/// so this op never needs to know it.
///
/// SPEC GAP — RATIFIED in spec v1.0.1 §5.4, note retained as history: the
/// draft's LUT+Newton inverse sqrt was unimplementable (constants never
/// existed); this exact floor `isqrt` stand-in is now normative, and — the
/// part the v1.0 freeze missed until adversarial review caught it — so are
/// the intermediate grids this function actually computes through:
/// `t = max(isqrt(s2·n), 1)`, `inv_rms_q30 = rne(n·2^30 / t)` (Q2.30), and
/// the per-element `y = rne(x_i·inv_rms_q30 / 2^15)` downshift before the
/// gain multiply. Fold those roundings any other way and the i8 outputs
/// differ: the spec text must (and since v1.0.1 does) pin every one.
///
/// Overflow proof, worst cases: `s2·n <= 127²·n² < 2^63` for `n <= 2^23`
/// (asserted); `inv_rms_q30 <= n·2^30 <= 2^53`; `|x_i·inv_rms_q30| <= 2^60`;
/// `|y| <= 2^45`; `|y·w_i| <= 2^60` fits i64 and REQUANT's i128 product.
pub fn rmsnorm_i(output: &mut [i8], input: &[i8], weight: &[i16], q: QScale) {
    let n = input.len();
    assert!(n > 0, "rmsnorm_i: empty input");
    assert!(n <= 1 << 23, "rmsnorm_i: n too large for exact i64 s2·n");
    assert!(output.len() >= n, "output shorter than input");
    assert!(weight.len() >= n, "weight shorter than input");

    let mut s2: i64 = 0;
    for &x in input {
        let x = x as i64;
        s2 += x * x;
    }
    let t = ((s2 as u64 * n as u64).isqrt() as i64).max(1);
    let inv_rms_q30 = rne_div((n as i128) << 30, t as i128) as i64;

    for ((out, &x), &w) in output.iter_mut().zip(input).zip(weight) {
        let u = x as i64 * inv_rms_q30;
        let y = rne_div(u as i128, 1 << 15) as i64;
        *out = requant_i64(y * w as i64, q);
    }
}

/// ARGMAX (spec §2): ties in exact i32 equality break to the LOWEST index.
///
/// E1's token flips happened because f32 near-ties are decided by rounding
/// noise; in CIS-1 a tie is an exact, reproducible event with a specified
/// resolution. Strict `>` while scanning ascending keeps the first maximum,
/// which IS the lowest-index rule — no separate tie handling exists to get
/// wrong.
pub fn argmax_i32(logits: &[i32]) -> u32 {
    debug_assert!(!logits.is_empty(), "argmax_i32: empty logits");
    let mut best = i32::MIN;
    let mut best_idx = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Unit goldens + properties. The golden constants were computed by an
// INDEPENDENT bignum implementation (Python, arbitrary-precision ints,
// scripted from the same spec text — LCG 6364136223846793005 /
// 1442695040888963407, seeds below). Two implementations agreeing on the
// same bits is the CIS-1 conformance model in miniature; these constants are
// the unit-level golden corpus and live in-file by design (tests/golden/ is
// append-only evidence, not unit fixtures).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic 64-bit LCG (Knuth MMIX constants) — no std RNG, no
    /// external crate, bit-identical in the Python golden generator.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    /// Packed ternary bytes with codes in {00, 01, 10} (no undefined 11):
    /// four sequential draws per byte, code = (draw >> 60) % 3, low pair
    /// first.
    fn gen_packed_no11(state: &mut u64, n_bytes: usize, out: &mut [u8]) {
        for b in out.iter_mut().take(n_bytes) {
            let mut byte = 0u8;
            for k in 0..4 {
                let c = ((lcg_next(state) >> 60) % 3) as u8;
                byte |= c << (2 * k);
            }
            *b = byte;
        }
    }

    /// i8 activations in [−127, 127] (−128 forbidden, spec §1).
    fn gen_acts_i8(state: &mut u64, out: &mut [i8]) {
        for a in out.iter_mut() {
            *a = (((lcg_next(state) >> 40) % 255) as i32 - 127) as i8;
        }
    }

    const SEED_TMV: u64 = 0xA1CE_5EED_0000_0001;
    const SEED_QNT: u64 = 0xA1CE_5EED_0000_0002;
    const SEED_NRM: u64 = 0xA1CE_5EED_0000_0003;
    const SEED_PROP: u64 = 0xA1CE_5EED_0000_0004;

    #[test]
    fn golden_tmv_8x32() {
        const GOLDEN: [i32; 8] = [291, 193, 292, -86, -369, 160, -134, -214];
        let mut state = SEED_TMV;
        let mut w = [0u8; 64]; // 8 rows × 32/4 bytes
        gen_packed_no11(&mut state, 64, &mut w);
        let mut a = [0i8; 32];
        gen_acts_i8(&mut state, &mut a);

        let mut out = [0i32; 8];
        ternary_matvec_i8(&mut out, &a, &w, 8, 32);
        assert_eq!(out, GOLDEN);
    }

    #[test]
    fn tmv_undefined_code_11_decodes_to_zero() {
        // All-11 bytes: every weight decodes to 0 regardless of activations.
        let w = [0xFFu8; 8];
        let a = [127i8, -127, 5, -5, 99, -99, 1, -1];
        let mut out = [123i32; 4];
        ternary_matvec_i8(&mut out, &a, &w, 4, 8);
        assert_eq!(out, [0; 4]);

        // Mixed byte 0b11_01_10_00: col0=0, col1=−1, col2=+1, col3=(11)→0.
        let w = [0b1101_1000u8];
        let a = [10i8, 20, 30, 40];
        let mut out = [0i32; 1];
        ternary_matvec_i8(&mut out, &a, &w, 1, 4);
        assert_eq!(out[0], -20 + 30);
    }

    #[test]
    fn golden_requant_table() {
        // (acc, M, S, expected). RNE ties: 5.5→6, 6.5→6, −5.5→−6, ±0.5→0,
        // ±1.5→±2; clamp: 127.5 rounds to even 128 then clamps to 127.
        const CASES: [(i32, i32, u8, i8); 14] = [
            (10, 1 << 30, 0, 5),
            (11, 1 << 30, 0, 6),
            (13, 1 << 30, 0, 6),
            (-11, 1 << 30, 0, -6),
            (255, 1 << 30, 0, 127),
            (300, 1 << 30, 0, 127),
            (-300, 1 << 30, 0, -127),
            (3, 1 << 30, 1, 1),
            (1, 1 << 30, 1, 0),
            (2, 1 << 30, 1, 0),
            (6, 1 << 30, 1, 2),
            (-2, 1 << 30, 1, 0),
            (-6, 1 << 30, 1, -2),
            (1000, 1_717_986_918, 4, 50),
        ];
        for (acc, m, s, want) in CASES {
            assert_eq!(
                requant_i32(acc, QScale::new(m, s)),
                want,
                "requant({acc}, M={m}, S={s})"
            );
        }
    }

    #[test]
    fn golden_quantize_activations_16() {
        const GOLDEN: [i8; 16] = [
            122, 110, -56, -122, -107, -120, 63, -15, -79, 49, -107, 12, 98, -127, -56, -57,
        ];
        let mut state = SEED_QNT;
        let mut v = [0i32; 16];
        for x in v.iter_mut() {
            *x = ((lcg_next(&mut state) >> 24) % 2_000_001) as i32 - 1_000_000;
        }
        let mut q = [0i8; 16];
        let absmax = quantize_activations_i32(&mut q, &v);
        assert_eq!(absmax, 955_153);
        assert_eq!(q, GOLDEN);
    }

    #[test]
    fn quantize_activations_ties_and_zero() {
        // absmax = 254: 1·127/254 = 0.5 → 0 (even); 3·127/254 = 1.5 → 2.
        let v = [254i32, 1, -1, 3, -3];
        let mut q = [0i8; 5];
        assert_eq!(quantize_activations_i32(&mut q, &v), 254);
        assert_eq!(q, [127, 0, 0, 2, -2]);

        // All-zero tensor: zero outputs, denominator 0 by convention.
        let v = [0i32; 4];
        let mut q = [9i8; 4];
        assert_eq!(quantize_activations_i32(&mut q, &v), 0);
        assert_eq!(q, [0; 4]);
    }

    #[test]
    fn golden_rmsnorm_16() {
        const GOLDEN: [i8; 16] = [
            -2, 11, 13, -29, 27, -1, -9, 4, -9, 9, 7, -8, 12, -12, 11, -4,
        ];
        let mut state = SEED_NRM;
        let mut x = [0i8; 16];
        gen_acts_i8(&mut state, &mut x);
        let mut w = [0i16; 16];
        for g in w.iter_mut() {
            // Gains in [8192, 16384] ≈ [0.5, 1.0] in Q1.14.
            *g = (((lcg_next(&mut state) >> 40) % 8193) + 8192) as i16;
        }
        let mut out = [0i8; 16];
        rmsnorm_i(&mut out, &x, &w, QScale::new(1 << 30, 24));
        assert_eq!(out, GOLDEN);

        // All-zero input: defined (t clamps to 1), output is all zeros.
        let z = [0i8; 16];
        let mut out = [7i8; 16];
        rmsnorm_i(&mut out, &z, &w, QScale::new(1 << 30, 24));
        assert_eq!(out, [0; 16]);
    }

    #[test]
    fn argmax_ties_break_low() {
        assert_eq!(argmax_i32(&[5, 9, 9, 1]), 1);
        assert_eq!(argmax_i32(&[3, 3, 3]), 0);
        assert_eq!(argmax_i32(&[-7]), 0);
        assert_eq!(argmax_i32(&[i32::MIN, i32::MIN]), 0);
        assert_eq!(argmax_i32(&[0, -1, 2, 2, -5, 2]), 2);
    }

    #[test]
    fn from_ratio_goldens() {
        assert_eq!(
            QScale::from_ratio(1, 3),
            Some(QScale::new(1_431_655_765, 1))
        );
        assert_eq!(QScale::from_ratio(1, 2), Some(QScale::new(1 << 30, 0)));
        assert_eq!(
            QScale::from_ratio(127, 300),
            Some(QScale::new(1_818_202_822, 1))
        );
        assert_eq!(
            QScale::from_ratio(1, 127),
            Some(QScale::new(1_082_196_484, 6))
        );
        // A ratio of exactly 1 is rejected (num >= den), not clamped…
        assert_eq!(QScale::from_ratio(u64::MAX, u64::MAX), None);
        // …but a ratio within half an ULP below 1.0 clamps to the largest
        // representable multiplier.
        assert_eq!(
            QScale::from_ratio(u64::MAX - 1, u64::MAX),
            Some(QScale::new(i32::MAX, 0))
        );
        assert_eq!(QScale::from_ratio(0, 5), None);
        assert_eq!(QScale::from_ratio(5, 0), None);
        assert_eq!(QScale::from_ratio(5, 4), None);
        // Below the bounded-S range (1/(2^64−1) ≈ 2^-64 would need S = 63).
        assert_eq!(QScale::from_ratio(1, u64::MAX), None);
    }

    /// The axiom, tested: chunked and reversed accumulation must equal the
    /// canonical ascending loop EXACTLY. This is true by construction for
    /// integers (and false for f32 — that failure is banked as E1).
    #[test]
    fn accumulation_order_invariance() {
        const DIM_OUT: usize = 16;
        const DIM_IN: usize = 256;
        const PACKED: usize = DIM_IN / 4;

        let mut state = SEED_PROP;
        // Full-range bytes: includes undefined 11 codes on purpose — order
        // invariance must hold for any byte stream.
        let mut w = [0u8; DIM_OUT * PACKED];
        for b in w.iter_mut() {
            *b = (lcg_next(&mut state) >> 56) as u8;
        }
        let mut a = [0i8; DIM_IN];
        gen_acts_i8(&mut state, &mut a);

        let mut want = [0i32; DIM_OUT];
        ternary_matvec_i8(&mut want, &a, &w, DIM_OUT, DIM_IN);

        for row in 0..DIM_OUT {
            let w_row = &w[row * PACKED..(row + 1) * PACKED];

            // (a) strictly reversed column order.
            let mut rev: i32 = 0;
            for col in (0..DIM_IN).rev() {
                let code = w_row[col / 4] >> (2 * (col % 4));
                rev += wcode(code) * a[col] as i32;
            }
            assert_eq!(rev, want[row], "reversed order diverged at row {row}");

            // (b) 32-column chunk partials, folded last-chunk-first.
            let mut partials = [0i32; 8];
            for (c, p) in partials.iter_mut().enumerate() {
                for col in c * 32..(c + 1) * 32 {
                    let code = w_row[col / 4] >> (2 * (col % 4));
                    *p += wcode(code) * a[col] as i32;
                }
            }
            let chunked: i32 = partials.iter().rev().sum();
            assert_eq!(chunked, want[row], "chunked order diverged at row {row}");
        }
    }

    /// Cross-check against the PRODUCTION f32 kernel in the exact regime:
    /// with i8-valued f32 inputs, every product is an integer <= 127 and
    /// every partial sum is bounded by 127·256 = 32512 < 2^24, so f32
    /// arithmetic (scalar SSE2 or AVX2+FMA alike) is exact and must equal
    /// the integer semantics bit-for-bit. Validates that the CIS packing
    /// convention matches `ops::ternary_matvec` on the same bytes.
    #[test]
    #[allow(clippy::float_cmp)] // equality is the point: both sides are exact
    fn matches_production_f32_kernel_in_exact_regime() {
        const DIM_OUT: usize = 16;
        const DIM_IN: usize = 256;

        let mut state = SEED_PROP;
        let mut w = [0u8; DIM_OUT * DIM_IN / 4];
        for b in w.iter_mut() {
            *b = (lcg_next(&mut state) >> 56) as u8;
        }
        let mut a = [0i8; DIM_IN];
        gen_acts_i8(&mut state, &mut a);

        let mut cis_out = [0i32; DIM_OUT];
        ternary_matvec_i8(&mut cis_out, &a, &w, DIM_OUT, DIM_IN);

        let mut a_f32 = [0.0f32; DIM_IN];
        for (f, &v) in a_f32.iter_mut().zip(a.iter()) {
            *f = v as f32;
        }
        let mut f32_out = [0.0f32; DIM_OUT];
        crate::ops::ternary_matvec(&mut f32_out, &a_f32, &w, DIM_OUT, DIM_IN, 1.0);

        for (row, (&got, &want)) in f32_out.iter().zip(cis_out.iter()).enumerate() {
            assert_eq!(got, want as f32, "production kernel diverged at row {row}");
        }
    }

    #[test]
    #[should_panic(expected = "M must be in")]
    fn qscale_rejects_unnormalized_multiplier() {
        let _ = QScale::new((1 << 30) - 1, 0);
    }
}
