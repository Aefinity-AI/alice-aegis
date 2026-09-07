//! CIS-1 E2 — integer-dominant forward path over the production artifacts.
//!
//! Spec: `docs/CIS-1_SPEC_DRAFT_v0.1.md` (+ the "v0.2 E2 implementation
//! notes" appended by this experiment). Reference integer ops live in
//! `crate::cis`; this module is the glue that runs a whole M7-class
//! transformer step with integer arithmetic everywhere the E2 scope demands:
//! embedding lookup, every RMSNorm, activation quantization, all seven
//! ternary matvecs per layer, the LM head, and argmax.
//!
//! TWO MODES (`CisMode`), selected at engine construction.
//!
//! `Hybrid` (E2): two declared f32 stages remain — (1) the attention core:
//! RoPE rotation, score dot products, softmax, and the probability-weighted
//! V mix (between the k/q/v projections and the o_proj input quantization);
//! (2) the MLP elementwise stage: silu(gate) · up (between the up/gate
//! matvecs and the down_proj input quantization). Everything crossing back
//! from a hybrid stage re-enters integer land through the exact
//! `f32_to_fixed` conversion of the f32 bit pattern, so the integer side
//! never depends on float *accumulation* — only on the per-element float
//! values, which are themselves deterministic on a fixed kernel path.
//! Same-machine determinism only.
//!
//! `FullInt` (v0.3): both stages replaced by ROPE-I / SOFTMAX-I / ACT-I
//! (`crate::cis_attn`). The whole forward pass is scalar integer arithmetic
//! with normative constants; no float value exists anywhere in it, so
//! cross-kernel-path/cross-ISA bit-identity holds by construction.
//!
//! Fixed-point conventions (E2 choices, documented in the spec notes):
//!   - residual stream: `i64`, Q.20 (`F`)
//!   - norm gains: `i32`, Q.20 (`GQ`), converted once at load from the
//!     checkpoint's BF16/F32 bytes by exact RNE
//!   - weight scales: exact f32 → (sign, odd mantissa, exponent) rationals
//!   - scale application: `QScale64`, a 63-bit fixed-point multiplier built
//!     by exact long division with RNE — the wide sibling of `cis::QScale`

use alloc::{format, string::String, vec, vec::Vec};

/// Times `$body` as `$phase` and records it into `$counters` (a
/// `&mut PhaseCycles` lvalue) when the `phase-timers` feature is on; a
/// no-op passthrough otherwise. Byte-for-byte the same macro as
/// `crate::inference`'s `timed_phase!` — duplicated here rather than
/// imported because `macro_rules!` items are module-scoped by default and
/// this module has no need of `inference`'s other, engine-specific
/// internals; see `phase_timers.rs`'s module docs for the RDTSC/RDTSCP
/// fencing this expands to.
#[cfg(feature = "phase-timers")]
macro_rules! timed_phase {
    ($counters:expr, $phase:expr, $body:block) => {{
        let __ts = crate::phase_timers::tick_start();
        let __ret = $body;
        let __te = crate::phase_timers::tick_end();
        $counters.record($phase, __ts, __te);
        __ret
    }};
}
#[cfg(not(feature = "phase-timers"))]
macro_rules! timed_phase {
    ($counters:expr, $phase:expr, $body:block) => {
        $body
    };
}

use crate::attention::RopeCache;
use crate::cis::rne_div;
use crate::cis_attn::{
    ExpLut, RopeTableI, inv_sqrt_q30, relu2_q20, rope_apply_i, silu_q20, softmax_i,
};
use crate::kvcache::KVCache;
use crate::model::{Activation, FullBitNetPipeline, ModelConfig};

/// Residual-stream fixed point: Q.20.
pub const F: u32 = 20;
/// Norm-gain fixed point: Q.20.
pub const GQ: u32 = 20;
/// Full-integer attention: q/k/v fixed point, Q.16 (spec v0.3).
pub const QK_F: u32 = 16;
/// Full-integer attention: score fixed point, Q.24 (spec v0.3).
pub const SCORE_F: u32 = 24;
/// Full-integer attention: probability fixed point, Q0.15 (spec §2).
pub const PROB_F: u32 = 15;

/// Which forward path the CIS engine runs.
///
/// `Hybrid` is E2's integer-dominant path: attention core and the MLP
/// elementwise stage in f32, re-entering integer land via exact
/// `f32_to_fixed` (same-machine determinism only). `FullInt` replaces both
/// hybrid stages with ROPE-I / SOFTMAX-I / ACT-I (spec v0.3): the entire
/// forward pass is scalar integer arithmetic — no float value exists
/// anywhere in it, so cross-kernel-path/cross-ISA bit-identity holds by
/// construction. Hybrid stays intact for A/B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CisMode {
    Hybrid,
    FullInt,
}

// ---------------------------------------------------------------------------
// Exact float-bit conversions (integer in, integer out — no float arithmetic)
// ---------------------------------------------------------------------------

/// Exact RNE division of a nonnegative i64 by 2^k (1 ≤ k ≤ 62) in pure
/// shift/mask arithmetic — bit-identical to `rne_div(m, 1<<k)` (asserted
/// exhaustively in tests). `bf16_to_fixed` runs per element of every LM-head
/// row when tables are converted on the fly (2B scale: ~1.3·10^8 calls per
/// scored token), where `rne_div`'s i128 division is the whole cost.
#[inline]
fn rne_shr(m: i64, k: i32) -> i64 {
    debug_assert!(m >= 0 && (1..=62).contains(&k));
    let floor = m >> k;
    let rem = m & ((1i64 << k) - 1);
    let half = 1i64 << (k - 1);
    if rem > half || (rem == half && floor & 1 == 1) {
        floor + 1
    } else {
        floor
    }
}

/// Exact RNE division of a signed i128 by 2^k (1 ≤ k ≤ 126) in pure
/// shift/mask arithmetic — bit-identical to `rne_div(p, 1i128 << k)` for
/// ANY sign of `p` (asserted exhaustively/randomly in tests). Unlike
/// `rne_shr` (nonnegative i64 only), this covers the signed i128 products
/// that `QScale64::rescale`, `f32_to_fixed`, and `fix_q_vec` divide by a
/// runtime power of two — `rne_div`'s i128/i128 division there compiles to
/// `compiler_builtins::u128_div_rem` (~100 cycles) and dominates the FullInt
/// Act phase (~5 G ticks / 15 % of a 2B verify). An arithmetic right shift
/// is `div_euclid` by `2^k` (floor), and masking off the low `k` bits is
/// `rem_euclid` — so the exact quotient/remainder pair `rne_div` computes
/// via `i128::div_euclid`/`rem_euclid` is available here without a divide;
/// `rne_round` then applies the single normative rounding rule to it.
#[inline]
fn rne_shr_i128(p: i128, k: u32) -> i128 {
    debug_assert!((1..=126).contains(&k));
    let floor = p >> k; // arithmetic shift == div_euclid(p, 2^k) for any sign
    let rem = p & ((1i128 << k) - 1); // == rem_euclid(p, 2^k), 0 <= rem < 2^k
    rne_round(floor, rem, 1i128 << k)
}

/// Exact BF16 → signed fixed-point with `frac` fractional bits, RNE.
/// Works on the bit pattern only. Inf/NaN and magnitudes that would exceed
/// ~2^44 are load-time errors, not silent saturation.
pub fn bf16_to_fixed(bits: u16, frac: u32) -> i64 {
    let neg = (bits >> 15) & 1 == 1;
    let exp = ((bits >> 7) & 0xFF) as i32;
    let man = (bits & 0x7F) as i64;
    assert!(exp != 0xFF, "bf16_to_fixed: inf/nan in model bytes");
    let (m, e) = if exp == 0 {
        (man, 1 - 127 - 7)
    } else {
        (man | 0x80, exp - 127 - 7)
    };
    let sh = e + frac as i32;
    assert!(sh <= 36, "bf16_to_fixed: value too large for fixed-point");
    let v = if sh >= 0 {
        m << sh
    } else if -sh >= 63 {
        0
    } else {
        rne_shr(m, -sh)
    };
    if neg { -v } else { v }
}

/// Exact f32 → signed fixed-point with `frac` fractional bits, RNE.
/// The hybrid→integer boundary: converts a float VALUE (via its bits)
/// without any float arithmetic.
pub fn f32_to_fixed(bits: u32, frac: u32) -> i64 {
    let neg = (bits >> 31) & 1 == 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = (bits & 0x7F_FFFF) as i64;
    assert!(
        exp != 0xFF,
        "f32_to_fixed: inf/nan crossed the hybrid boundary"
    );
    let (m, e) = if exp == 0 {
        (man, 1 - 127 - 23)
    } else {
        (man | 0x80_0000, exp - 127 - 23)
    };
    let sh = e + frac as i32;
    // Headroom bound, recalibrated for BitNet-2B width (E2-7 spec note):
    // m < 2^24, so sh ≤ 26 keeps |v| < 2^50 — the normq/quantq residual
    // bound. The original sh ≤ 20 (|v| < 2^44, real |x| < 2^24) fired on
    // real BitNet-2B hybrid-stage activations (relu² MLP products exceed
    // 2^24 before ffn_sub_norm); M7-scale values are untouched, so this is
    // a pure guard relaxation — in-range conversions are bit-identical.
    assert!(sh <= 26, "f32_to_fixed: value too large for fixed-point");
    let v = if sh >= 0 {
        m << sh
    } else if -sh >= 63 {
        0
    } else {
        rne_shr_i128(m as i128, (-sh) as u32) as i64
    };
    if neg { -v } else { v }
}

/// Exact f32 decomposition: value = (−1)^neg · m · 2^e with m odd (0 → m=0).
/// No rounding exists in this conversion; it is the identity on the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FRatio {
    pub neg: bool,
    pub m: u64,
    pub e: i32,
}

pub fn f32_to_ratio(bits: u32) -> FRatio {
    let neg = (bits >> 31) & 1 == 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = (bits & 0x7F_FFFF) as u64;
    assert!(exp != 0xFF, "f32_to_ratio: inf/nan weight scale");
    let (mut m, mut e) = if exp == 0 {
        (man, 1 - 127 - 23)
    } else {
        (man | 0x80_0000, exp - 127 - 23)
    };
    if m == 0 {
        return FRatio {
            neg: false,
            m: 0,
            e: 0,
        };
    }
    while m & 1 == 0 {
        m >>= 1;
        e += 1;
    }
    FRatio { neg, m, e }
}

// ---------------------------------------------------------------------------
// QScale64 — wide fixed-point multiplier (the i64-domain sibling of
// cis::QScale). value = m · 2^e, m ∈ [2^62, 2^63), built from an exact
// u128/u128 rational by restoring long division with RNE at 63 significant
// bits — no intermediate ever overflows, any num/den ≤ 2^127.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QScale64 {
    m: u64,
    e: i32,
}

impl QScale64 {
    pub const fn m(self) -> u64 {
        self.m
    }
    pub const fn e(self) -> i32 {
        self.e
    }

    /// num/den (num ≥ 0, den > 0) rounded (RNE) to 63 significant bits.
    /// num == 0 → the exact zero multiplier.
    pub fn from_ratio(num: u128, den: u128) -> QScale64 {
        assert!(den > 0, "QScale64: zero denominator");
        if num == 0 {
            return QScale64 { m: 0, e: 0 };
        }
        let nb = 128 - num.leading_zeros() as i32;
        let db = 128 - den.leading_zeros() as i32;
        let diff = nb - db;
        // Align to equal bit lengths — den<<diff has bit length nb ≤ 128,
        // num<<(−diff) has bit length db ≤ 128: neither shift can overflow.
        let (mut rem, d) = if diff >= 0 {
            (num, den << diff)
        } else {
            (num << -diff, den)
        };
        // x = rem/d ∈ (1/2, 2). Normalize to [1, 2).
        let mut exp_adj = 0i32;
        if rem < d {
            rem <<= 1; // rem was < d ≤ 2^127: safe
            exp_adj = -1;
        }
        // 64 quotient bits MSB-first (63 mantissa + 1 round bit).
        let mut q: u128 = 0;
        for _ in 0..64 {
            q <<= 1;
            if rem >= d {
                rem -= d;
                q |= 1;
            }
            rem <<= 1; // rem < d ≤ 2^127 before the shift: safe
        }
        let sticky = rem != 0;
        let round = q & 1 == 1;
        let mut m = (q >> 1) as u64; // 63 bits, leading bit set
        let mut e = diff + exp_adj - 62;
        if round && (sticky || m & 1 == 1) {
            m += 1;
            if m == 1 << 63 {
                m = 1 << 62;
                e += 1;
            }
        }
        debug_assert!((1 << 62..1 << 63).contains(&m));
        QScale64 { m, e }
    }

    /// `rne(x · m · 2^e)`, exact in i128.
    pub fn rescale(self, x: i64) -> i64 {
        let p = x as i128 * self.m as i128; // |p| < 2^63 · 2^63 = 2^126
        let v = if self.e >= 0 {
            let s = p << self.e;
            assert!(
                (p == 0) || (s >> self.e == p),
                "QScale64::rescale: left-shift overflow"
            );
            s
        } else if -self.e >= 127 {
            0 // |p| < 2^126 ⇒ |p|/2^127 < 1/2 strictly: rounds to 0
        } else {
            rne_shr_i128(p, (-self.e) as u32)
        };
        assert!(
            v >= i64::MIN as i128 && v <= i64::MAX as i128,
            "QScale64::rescale: result exceeds i64"
        );
        v as i64
    }
}

// ---------------------------------------------------------------------------
// Fused integer RMSNorm + per-token absmax i8 quantization.
// ---------------------------------------------------------------------------

/// Exact rational per-code scale of a quantized activation vector:
/// real value of code `a` is `a · num / den`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActScale {
    pub num: u128,
    pub den: u128,
}

/// Round-half-to-even step given an exact floor quotient `q` and exact
/// remainder `r` (`0 <= r < den`) of `num / den`, i.e. the second half of
/// what `rne_div` computes — shared here so `normq` and `quantq` can supply
/// `(q, r)` from a division-free path (or a cheaper narrower division) and
/// still apply the identical, normative rounding rule.
#[inline]
fn rne_round(q: i128, r: i128, den: i128) -> i128 {
    debug_assert!(den > 0 && r >= 0 && r < den, "rne_round: r out of range");
    if 2 * r > den || (2 * r == den && q & 1 != 0) {
        q + 1
    } else {
        q
    }
}

/// RMSNorm (gain `g`, Q.GQ) followed by per-token absmax quantization onto
/// the i8 grid, all in exact integer arithmetic.
///
/// The i8 codes are `rne(u_i·127 / max|u|)` with `u_i = h_i·g_i` — the RMS
/// divides out of the absmax ratio, so the codes never see it. The RMS
/// enters only the carried scale, through `t = isqrt(s2·n) ≈ n·rms` (the
/// same exact-isqrt stand-in as `cis::rmsnorm_i`; the float path's epsilon
/// is replaced by `max(t,1)`, which only guards the all-zero vector):
///
///   normed_i = (h_i/2^F / rms_real)·(g_i/2^GQ) = u_i·n / (t·2^GQ)
///   scale    = max|normed|/127 = A·n / (127·t·2^GQ)
///
/// Headroom (asserted, recalibrated for BitNet-2B width — E2-7 spec note):
/// |h| < 2^50 and |g| < 2^31 keep every intermediate exact — u = h·g < 2^81
/// (i128), s2 ≤ n·2^100 ≤ 2^113, and `s2·n` ≤ 2^126 < u128 for n ≤ 8192;
/// num = a·n ≤ 2^94, den = 127·(t·2^GQ) ≤ 2^90.
pub fn normq(codes: &mut [i8], h: &[i64], g: &[i32]) -> ActScale {
    let n = h.len();
    assert!(n > 0 && n <= 8192, "normq: n out of range");
    assert!(codes.len() >= n && g.len() >= n, "normq: buffer too short");

    let mut a: i128 = 0;
    for (&hi, &gi) in h.iter().zip(g) {
        assert!(hi.unsigned_abs() < 1 << 50, "normq: residual out of range");
        let u = hi as i128 * gi as i128;
        if u.abs() > a {
            a = u.abs();
        }
    }
    if a == 0 {
        for c in codes[..n].iter_mut() {
            *c = 0;
        }
        return ActScale { num: 0, den: 1 };
    }
    // Division-free per-element quotient: one i128/u128 reciprocal division
    // per call (not per element), then a multiply + shift + exact remainder
    // correction per element, replacing the u128 `rne_div` slow-path division
    // that otherwise dominates this loop (profiled at ~12% of 2B verify wall
    // time, ~33M calls/receipt).
    //
    // k = bit_length(a) + 8 puts the reciprocal R = floor(2^k / a) in
    // [2^8, 2^9) (since 2^(k-1) <= 2^k/a's scale... precisely: a in
    // [2^(bits-1), 2^bits), so 2^k/a in (2^8, 2^9]). For each element,
    // num = u*127 with |num| <= 127*a < 2^88, so |num*R| < 2^88 * 2^9 =
    // 2^97, safely inside i128. q_est = (num*R) >> k (arithmetic shift =
    // floor) satisfies |q_est - floor(num/a)| < 1: R underestimates the
    // true 2^k/a by less than 1 (integer floor of the division), so the
    // error introduced is < num/a scaled by (1/2^k), i.e.
    // < |num| / 2^k <= 127*a/2^k < 127/2^8 < 1. Since the true floor and
    // q_est are both integers within 1 of each other, at most one
    // correction step in each direction restores the exact floor quotient
    // and its exact remainder — verified with a debug_assert below, and
    // exhaustively against `rne_div` in
    // aegis-core/tests/normq_divfree_equivalence.rs.
    let k = (128 - a.leading_zeros() as i32) + 8;
    let r_recip: i128 = ((1u128 << k) / a as u128) as i128;
    for ((c, &hi), &gi) in codes.iter_mut().zip(h).zip(g) {
        let u = hi as i128 * gi as i128;
        let num = u * 127;
        let mut q = (num * r_recip) >> k;
        let mut r = num - q * a;
        let mut steps: u32 = 0;
        while r < 0 {
            q -= 1;
            r += a;
            steps += 1;
        }
        while r >= a {
            q += 1;
            r -= a;
            steps += 1;
        }
        debug_assert!(steps <= 1, "normq: reciprocal estimate off by >1 ULP");
        *c = rne_round(q, r, a).clamp(-127, 127) as i8;
    }
    let mut s2: u128 = 0;
    for &hi in h {
        let x = hi as i128;
        s2 += (x * x) as u128;
    }
    let t = (s2 * n as u128).isqrt().max(1);
    ActScale {
        num: a as u128 * n as u128,
        den: 127u128 * (t << GQ),
    }
}

/// Per-token absmax i8 quantization of a Q.`frac` fixed-point vector, no
/// norm — the integer analog of `quant_act` without a preceding sub-norm.
/// scale = A / (127·2^frac). `frac` is the fractional width the vector was
/// fixed at (F for the residual stream; possibly less at a hybrid boundary
/// — see `fix_f32_vec`).
pub fn quantq(codes: &mut [i8], h: &[i64], frac: u32) -> ActScale {
    let n = h.len();
    assert!(n > 0, "quantq: empty input");
    assert!(codes.len() >= n, "quantq: buffer too short");
    let mut a: i64 = 0;
    for &hi in h {
        assert!(hi.unsigned_abs() < 1 << 50, "quantq: value out of range");
        if hi.abs() > a {
            a = hi.abs();
        }
    }
    if a == 0 {
        for c in codes[..n].iter_mut() {
            *c = 0;
        }
        return ActScale { num: 0, den: 1 };
    }
    // `hi*127` and `a` both fit i64 here (|hi| < 2^50 => |hi*127| < 2^57,
    // a < 2^50), so a hardware 64/64 division replaces the u128 slow-path
    // `rne_div` division; the RNE rule itself is unchanged (shared via
    // `rne_round`).
    for (c, &hi) in codes.iter_mut().zip(h) {
        let num = hi * 127;
        let q = num.div_euclid(a);
        let r = num.rem_euclid(a);
        *c = rne_round(q as i128, r as i128, a as i128).clamp(-127, 127) as i8;
    }
    ActScale {
        num: a as u128,
        den: 127u128 << frac,
    }
}

/// Hybrid→integer boundary conversion of a whole f32 vector, with a
/// per-vector dynamic fractional width (a block exponent): the largest
/// `G ≤ F` such that every element converts inside the 2^50 headroom bound
/// (`f32_to_fixed`'s `sh ≤ 26` guard), i.e. `G = min(F, 176 − max_exp)`.
/// Returns `G`. Deterministic and exact: `G` depends only on the f32 bit
/// patterns, and each element is the exact RNE fixing at `G` fractional
/// bits. M7-scale vectors always get `G = F`, reproducing the A19 bits;
/// BitNet-2B relu² MLP products (which exceed 2^30 real before
/// `ffn_sub_norm`) get a coarser grid instead of a panic (spec E2-7).
/// `normq`'s i8 codes are invariant in `G` (exact ratios of `h·g`); its
/// carried scale agrees across `G` choices up to the exact-isqrt floor
/// granularity (relative O(1/t)) — that floor is already part of the
/// normative arithmetic at Q.20. `quantq` carries `G` explicitly.
fn fix_f32_vec(out: &mut [i64], src: &[f32]) -> u32 {
    debug_assert_eq!(out.len(), src.len());
    let mut max_exp: i32 = 0;
    for &v in src {
        let exp = ((v.to_bits() >> 23) & 0xFF) as i32;
        assert!(exp != 0xFF, "fix_f32_vec: inf/nan at the hybrid boundary");
        if exp > max_exp {
            max_exp = exp;
        }
    }
    // sh = exp − 150 + G ≤ 26  ⇔  G ≤ 176 − max_exp.
    let g = (176 - max_exp).min(F as i32);
    assert!(
        g >= 0,
        "fix_f32_vec: hybrid value >= 2^50 — model divergence"
    );
    for (o, &v) in out.iter_mut().zip(src) {
        *o = f32_to_fixed(v.to_bits(), g as u32);
    }
    g as u32
}

/// FullInt's own hybrid-free boundary: the ACT-I MLP output (`relu2_q20` /
/// `silu_q20`, already exact Q.20 i64 values, no float ever touched) can
/// still exceed the normq/quantq `2^50` residual headroom at BitNet-2B scale
/// — the same overflow `fix_f32_vec` escapes for Hybrid (spec §5.10 gap; E1b
/// trace: `normq: residual out of range`). Mirrors `fix_f32_vec`'s
/// block-exponent contract exactly, but derives `G` from the integer
/// magnitude already in hand (`64 − leading_zeros`), never from a float
/// exponent: the largest `G ≤ F` such that every element, after an exact RNE
/// right-shift by `F − G` bits, is guaranteed `< 2^49` — one full bit under
/// the `2^50` headroom bound, which absorbs the ±1 that RNE rounding can add
/// at the shift boundary. Rescales `v` in place and returns `G`. M7-scale
/// products never need more than 49 bits, so `shift = 0` and `G` degenerates
/// to `F` with every element numerically untouched — bit-identical to the
/// prior unconditional-`F` behavior.
fn fix_q_vec(v: &mut [i64]) -> u32 {
    let mut max_abs: u64 = 0;
    for &x in v.iter() {
        let a = x.unsigned_abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs == 0 {
        return F;
    }
    let bits = u64::BITS - max_abs.leading_zeros(); // bit length of max_abs
    let shift = bits.saturating_sub(49);
    assert!(
        shift <= F,
        "fix_q_vec: FullInt MLP product exceeds Q0 headroom — model divergence"
    );
    if shift > 0 {
        for x in v.iter_mut() {
            *x = rne_shr_i128(*x as i128, shift) as i64;
        }
    }
    F - shift
}

/// Integer dot of i8 codes against a fixed-point table row (the LM head).
/// Exact in i64: |a|·|e|·n ≤ 127·2^31·4096 < 2^51.
pub fn dot_i8_i32(a: &[i8], e: &[i32]) -> i64 {
    debug_assert_eq!(a.len(), e.len());
    let mut acc: i64 = 0;
    for (&x, &w) in a.iter().zip(e) {
        acc += x as i64 * w as i64;
    }
    acc
}

/// Argmax over i64 logits, ties break to the LOWEST index (spec §2 ARGMAX).
pub fn argmax_i64(logits: &[i64]) -> u32 {
    debug_assert!(!logits.is_empty());
    let mut best = i64::MIN;
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
// FNV-1a 64 — the determinism exhibit digest (argmax sequence identity check
// across runs/machines). Public constants; goldens pin the published vectors.
// ---------------------------------------------------------------------------

pub const FNV1A64_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
pub const FNV1A64_PRIME: u64 = 0x100_0000_01B3;

pub fn fnv1a64(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV1A64_PRIME);
    }
    h
}

// ---------------------------------------------------------------------------
// Scale plumbing: weight-scale rational × activation scale → either an f64
// (into a hybrid stage) or a QScale64 (back onto the Q.F residual grid).
// ---------------------------------------------------------------------------

/// 2^e as f64, exact for normal-range e.
fn exp2_f64(e: i32) -> f64 {
    assert!(
        (-1022..=1023).contains(&e),
        "exp2_f64: exponent out of range"
    );
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// Real-valued scale `W · s` for dequantizing a matvec accumulator into a
/// hybrid f32 stage: sign · wm·2^we · num/den. (f64 evaluation — this value
/// only ever feeds a declared hybrid stage.)
fn dequant_scale_f64(w: &FRatio, s: &ActScale) -> f64 {
    if w.m == 0 || s.num == 0 {
        return 0.0;
    }
    let v = (w.m as f64) * (s.num as f64) / (s.den as f64) * exp2_f64(w.e);
    if w.neg { -v } else { v }
}

/// Exact multiplier taking a matvec accumulator onto a Q.`frac` fixed-point
/// grid: OUT = ±rne(acc · wm·2^we · num/den · 2^frac). Returns the sign
/// separately; the magnitude ratio feeds QScale64 exactly. `frac = F`
/// targets the residual grid; the full-integer attention path uses
/// `frac = QK_F` for its q/k/v grids.
fn fixed_qscale(w: &FRatio, s: &ActScale, frac: u32) -> (bool, QScale64) {
    if w.m == 0 || s.num == 0 {
        return (false, QScale64::from_ratio(0, 1));
    }
    let mut num = (w.m as u128)
        .checked_mul(s.num)
        .expect("fixed_qscale: numerator overflow");
    let mut den = s.den;
    let sh = frac as i32 + w.e;
    if sh >= 0 {
        num = num
            .checked_shl(sh as u32)
            .filter(|v| v >> sh == (w.m as u128) * s.num)
            .expect("fixed_qscale: numerator shift overflow");
    } else {
        den = den
            .checked_shl((-sh) as u32)
            .filter(|v| v >> -sh == s.den)
            .expect("fixed_qscale: denominator shift overflow");
    }
    (w.neg, QScale64::from_ratio(num, den))
}

/// The E2 name for the residual-grid case, kept so the hybrid path reads
/// unchanged.
fn residual_qscale(w: &FRatio, s: &ActScale) -> (bool, QScale64) {
    fixed_qscale(w, s, F)
}

// ---------------------------------------------------------------------------
// Model conversion: checkpoint bytes → integer tables, once at load.
// ---------------------------------------------------------------------------

struct CisLayer<'a> {
    ln1: Vec<i32>,
    ln2: Vec<i32>,
    attn_sub: Option<Vec<i32>>,
    ffn_sub: Option<Vec<i32>>,
    q_w: &'a [u8],
    k_w: &'a [u8],
    v_w: &'a [u8],
    o_w: &'a [u8],
    gate_w: &'a [u8],
    up_w: &'a [u8],
    down_w: &'a [u8],
    q_s: FRatio,
    k_s: FRatio,
    v_s: FRatio,
    o_s: FRatio,
    gate_s: FRatio,
    up_s: FRatio,
    down_s: FRatio,
}

pub struct CisModel<'a> {
    layers: Vec<CisLayer<'a>>,
    final_g: Vec<i32>,
    /// BF16 embedding table bytes (vocab × hidden), borrowed from the
    /// artifact. Rows are converted to Q.F by exact RNE **on the fly** at
    /// each lookup/dot — materializing the converted table is ~0.5 GB as
    /// i32 at BitNet-2B scale (50,256 × 2560), which does not fit the 6 GB
    /// dev box next to the float engine. The conversion is per-element
    /// `bf16_to_fixed`, so the produced bits are identical to a
    /// pre-converted table (M7 E2 digest 0x42E820C2A8A59CD6 reproduces).
    emb: &'a [u8],
    /// LM-head BF16 bytes: the untied `lm_head.weight` when the checkpoint
    /// has one, otherwise the tied embedding table (same on-the-fly rule).
    head: &'a [u8],
    /// One-time i16 hi/lo plane conversion of `head` (see `HeadPlanes`),
    /// built at load by default. `None` when preconversion was disabled
    /// (`CisModel::new_with_options(.., false)`), in which case every
    /// LM-head dot falls back to the on-the-fly BF16 path over `head`,
    /// unchanged from before this pre-conversion existed.
    head_planes: Option<HeadPlanes>,
    pub config: ModelConfig,
}

/// Norm-gain bytes (BF16 or F32, derived from the length ratio exactly like
/// `ops::rmsnorm`) → Q.GQ i32 gains.
fn gains_to_q(bytes: &[u8], n: usize, what: &str) -> Result<Vec<i32>, String> {
    if n == 0 || bytes.len() < n {
        return Err(format!("{what}: {} bytes for {} gains", bytes.len(), n));
    }
    let elem = bytes.len() / n;
    let mut out = Vec::with_capacity(n);
    match elem {
        2 => {
            for i in 0..n {
                let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                let v = bf16_to_fixed(bits, GQ);
                if v.unsigned_abs() >= 1 << 31 {
                    return Err(format!("{what}: gain {i} out of Q.{GQ} i32 range"));
                }
                out.push(v as i32);
            }
        }
        4 => {
            for i in 0..n {
                let bits = u32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]);
                let v = f32_to_fixed(bits, GQ);
                if v.unsigned_abs() >= 1 << 31 {
                    return Err(format!("{what}: gain {i} out of Q.{GQ} i32 range"));
                }
                out.push(v as i32);
            }
        }
        _ => return Err(format!("{what}: unsupported element size {elem}")),
    }
    Ok(out)
}

/// Check a BF16 table's byte length without converting it (conversion is
/// per-row, on the fly — see `CisModel::emb`).
fn check_bf16_table(bytes: &[u8], count: usize, what: &str) -> Result<(), String> {
    if bytes.len() != count * 2 {
        return Err(format!(
            "{what}: {} bytes, expected {} BF16 values",
            bytes.len(),
            count
        ));
    }
    Ok(())
}

/// One BF16 table row → Q.F i64 values, exact RNE — element-for-element the
/// conversion the former load-time table materialization applied, including
/// its i32-range check (the LM-head dot bound below needs |e| < 2^31).
fn bf16_row_to_q(row: &[u8], out: &mut [i64]) {
    debug_assert_eq!(row.len(), out.len() * 2);
    for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) {
        let v = bf16_to_fixed(u16::from_le_bytes([b[0], b[1]]), F);
        assert!(
            v.unsigned_abs() < 1 << 31,
            "bf16_row_to_q: value out of Q.{F} i32 range"
        );
        *o = v;
    }
}

/// Integer dot of i8 codes against one BF16 table row, converting each
/// element on the fly (exact `bf16_to_fixed`, i32-range asserted). Produces
/// the same accumulator, in the same order, as `dot_i8_i32` over a
/// pre-converted row: exact in i64, |a|·|e|·n ≤ 127·2^31·4096 < 2^51.
pub fn dot_i8_bf16q(a: &[i8], row: &[u8]) -> i64 {
    debug_assert_eq!(a.len() * 2, row.len());
    let mut acc: i64 = 0;
    for (&x, b) in a.iter().zip(row.chunks_exact(2)) {
        let w = bf16_to_fixed(u16::from_le_bytes([b[0], b[1]]), F);
        assert!(
            w.unsigned_abs() < 1 << 31,
            "dot_i8_bf16q: value out of Q.{F} i32 range"
        );
        acc += x as i64 * w;
    }
    acc
}

// ---------------------------------------------------------------------------
// LM-head pre-conversion: i16 hi/lo planes, built once at load instead of
// re-deriving a BF16->Q.F conversion on every one of the many dot calls a
// single loaded head serves (one full head pass per generated token).
// ---------------------------------------------------------------------------

/// Split a Q.F fixed-point value into two `i16` halves, `w = w_hi * 2^15 +
/// w_lo`, exactly as `cis_avx2::dot_i8_bf16q_avx2_wide` already does per
/// element on the fly (see that function's doc comment for the bound this
/// mirrors). `w_hi = w >> 15` is an arithmetic (floor) shift, so `w_lo = w -
/// (w_hi << 15)` is always in `[0, 2^15)` regardless of `w`'s sign — the
/// remainder of a floor division is never negative. Returns `None` when
/// `w_hi` does not fit `i16` (`|w| >= 2^30`, roughly): the caller records the
/// whole row as an exception and keeps the on-the-fly BF16 path for it,
/// rather than truncating silently.
#[inline]
pub fn split_i16_planes(w: i64) -> Option<(i16, i16)> {
    let w_hi = w >> 15;
    let w_lo = w - (w_hi << 15);
    if w_hi < i16::MIN as i64 || w_hi > i16::MAX as i64 {
        None
    } else {
        // `w_lo` is in `[0, 2^15)`, which always fits `i16` (max 32767).
        Some((w_hi as i16, w_lo as i16))
    }
}

/// Scalar reference dot over pre-split i16 hi/lo planes: `sum(a[i] *
/// (hi[i]*2^15 + lo[i]))`, exact in i64. Bit-identical to `dot_i8_bf16q`
/// whenever every element of the row was split by `split_i16_planes` from
/// the same `bf16_to_fixed` value `dot_i8_bf16q` itself would compute —
/// integer addition and multiplication are exact and associative here (same
/// bound `dot_i8_bf16q` documents: `127 * 2^31 * 4096 < 2^51`), so this is
/// just that sum computed via the split representation instead of the
/// original `i64` weight.
pub fn dot_i8_i16planes(a: &[i8], hi: &[i16], lo: &[i16]) -> i64 {
    debug_assert_eq!(a.len(), hi.len());
    debug_assert_eq!(a.len(), lo.len());
    let mut acc: i64 = 0;
    for ((&x, &h), &l) in a.iter().zip(hi.iter()).zip(lo.iter()) {
        let w = (h as i64) * 32768 + l as i64;
        acc += x as i64 * w;
    }
    acc
}

/// Convert one BF16 table row into i16 hi/lo planes, in place. Returns
/// `true` if every element converted cleanly (`hi`/`lo` are now valid for
/// this row), `false` if any element's magnitude falls outside what the
/// plane split (or `dot_i8_bf16q`'s own `|w| < 2^31` bound) can represent —
/// an exception row, left for the caller to route to the on-the-fly BF16
/// path at dot time instead.
///
/// Every OTHER condition `dot_i8_bf16q` itself would reject — inf/nan bytes,
/// `bf16_to_fixed`'s own `sh <= 36` assert — is NOT caught here: `bf16_to_fixed`
/// panics with the identical message `dot_i8_bf16q` would produce, just at
/// load time (when the head is pre-converted) instead of at first dot-time
/// use. This is a deliberate, documented change in *when* a malformed
/// checkpoint is rejected, not a change in whether or how (see
/// `CisModel::new`'s doc comment).
pub fn head_row_to_planes(row: &[u8], hi: &mut [i16], lo: &mut [i16]) -> bool {
    debug_assert_eq!(row.len(), hi.len() * 2);
    debug_assert_eq!(hi.len(), lo.len());
    let (row_pairs, _) = row.as_chunks::<2>();
    for (i, b) in row_pairs.iter().enumerate() {
        let w = bf16_to_fixed(u16::from_le_bytes(*b), F);
        if w.unsigned_abs() >= 1 << 31 {
            return false;
        }
        match split_i16_planes(w) {
            Some((h, l)) => {
                hi[i] = h;
                lo[i] = l;
            }
            None => return false,
        }
    }
    true
}

/// Pre-converted LM-head: every row of `head` split into i16 hi/lo planes
/// (`hi`/`lo`, each `vocab * hidden`) once at load, plus an `exceptions` bit
/// per row for the (expected-empty, for any real checkpoint) rows that
/// cannot take the split — see `head_row_to_planes`. Built once and reused
/// across every `logits_int` call on the owning `CisModel`, which is the
/// whole point: the former per-dot on-the-fly BF16->Q.F conversion
/// (`dot_i8_bf16q`) is now a load-time cost instead of a per-token one.
///
/// Memory: `+4` bytes per weight over the BF16 bytes already held in `head`
/// (2 bytes each for `hi`/`lo`, vs. BF16's 2) — `vocab * hidden * 4` bytes,
/// ~515 MB at BitNet-2B scale (50,256 × 2,560). Accepted; see
/// `AEGIS_HEAD_PRECONVERT=0` (`CisModel::new_with_options`) for hosts where
/// that is too much.
struct HeadPlanes {
    hi: Vec<i16>,
    lo: Vec<i16>,
    exceptions: Vec<bool>,
}

impl HeadPlanes {
    fn build(head: &[u8], vocab: usize, hidden: usize) -> HeadPlanes {
        let mut hi = vec![0i16; vocab * hidden];
        let mut lo = vec![0i16; vocab * hidden];
        let mut exceptions = vec![false; vocab];
        for j in 0..vocab {
            let row = &head[j * hidden * 2..(j + 1) * hidden * 2];
            let ok = head_row_to_planes(
                row,
                &mut hi[j * hidden..(j + 1) * hidden],
                &mut lo[j * hidden..(j + 1) * hidden],
            );
            if !ok {
                exceptions[j] = true;
            }
        }
        HeadPlanes { hi, lo, exceptions }
    }

    /// Planes for row `j`, or `None` if `j` is an exception row (caller
    /// falls back to the on-the-fly BF16 path).
    #[inline]
    fn row(&self, j: usize, hidden: usize) -> Option<(&[i16], &[i16])> {
        if self.exceptions[j] {
            None
        } else {
            let start = j * hidden;
            let end = start + hidden;
            Some((&self.hi[start..end], &self.lo[start..end]))
        }
    }
}

impl<'a> CisModel<'a> {
    /// Builds the FullInt-CIS model with LM-head pre-conversion ON (the
    /// default — see `new_with_options`). Every existing call site keeps
    /// this signature and behavior unchanged.
    pub fn new(
        pipeline: &FullBitNetPipeline<'a>,
        config: &ModelConfig,
    ) -> Result<CisModel<'a>, String> {
        Self::new_with_options(pipeline, config, true)
    }

    /// `preconvert_head`: when `true`, build the one-time i16 hi/lo plane
    /// representation of the LM head (`HeadPlanes::build`) so every
    /// `logits_int` call reuses it instead of re-deriving a BF16->Q.F
    /// conversion per element per dot (the LM head is dotted once per
    /// generated token, so a checkpoint's head is otherwise converted from
    /// scratch that many times over the life of one `CisModel`). `false`
    /// keeps the pre-existing on-the-fly BF16 path unconditionally — an
    /// A/B and tiny-RAM-host escape hatch (the plane representation costs
    /// `+4` bytes per weight; see `HeadPlanes`'s doc comment). Callers that
    /// want an environment-variable-driven toggle (e.g.
    /// `AEGIS_HEAD_PRECONVERT=0`) read it themselves and pass the resulting
    /// bool here — this crate does not read environment variables (it is
    /// `no_std` outside the `parallel` feature).
    pub fn new_with_options(
        pipeline: &FullBitNetPipeline<'a>,
        config: &ModelConfig,
        preconvert_head: bool,
    ) -> Result<CisModel<'a>, String> {
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        if !hidden.is_multiple_of(4) || !inter.is_multiple_of(4) {
            return Err(String::from("CIS packing requires dims divisible by 4"));
        }

        let mut layers = Vec::with_capacity(pipeline.layers.len());
        for l in &pipeline.layers {
            let i = l.layer_idx;
            layers.push(CisLayer {
                ln1: gains_to_q(
                    l.input_layernorm_weight.data(),
                    hidden,
                    &format!("layer {i} input_layernorm"),
                )?,
                ln2: gains_to_q(
                    l.post_attention_layernorm_weight.data(),
                    hidden,
                    &format!("layer {i} post_attention_layernorm"),
                )?,
                attn_sub: match &l.attn_sub_norm {
                    Some(t) => Some(gains_to_q(
                        t.data(),
                        hidden,
                        &format!("layer {i} attn_sub_norm"),
                    )?),
                    None => None,
                },
                ffn_sub: match &l.ffn_sub_norm {
                    Some(t) => Some(gains_to_q(
                        t.data(),
                        inter,
                        &format!("layer {i} ffn_sub_norm"),
                    )?),
                    None => None,
                },
                q_w: l.q_proj.data(),
                k_w: l.k_proj.data(),
                v_w: l.v_proj.data(),
                o_w: l.o_proj.data(),
                gate_w: l.gate_proj.data(),
                up_w: l.up_proj.data(),
                down_w: l.down_proj.data(),
                q_s: f32_to_ratio(l.q_proj_scale.to_bits()),
                k_s: f32_to_ratio(l.k_proj_scale.to_bits()),
                v_s: f32_to_ratio(l.v_proj_scale.to_bits()),
                o_s: f32_to_ratio(l.o_proj_scale.to_bits()),
                gate_s: f32_to_ratio(l.gate_proj_scale.to_bits()),
                up_s: f32_to_ratio(l.up_proj_scale.to_bits()),
                down_s: f32_to_ratio(l.down_proj_scale.to_bits()),
            });
        }

        let final_g = gains_to_q(pipeline.final_norm.data(), hidden, "final norm")?;
        check_bf16_table(pipeline.embeddings, vocab * hidden, "EMBED.BIN")?;
        let head = match &pipeline.lm_head {
            Some(t) => {
                check_bf16_table(t.data(), vocab * hidden, "lm_head")?;
                t.data()
            }
            None => pipeline.embeddings,
        };
        let head_planes = if preconvert_head {
            Some(HeadPlanes::build(head, vocab, hidden))
        } else {
            None
        };

        Ok(CisModel {
            layers,
            final_g,
            emb: pipeline.embeddings,
            head,
            head_planes,
            config: config.clone(),
        })
    }

    /// Embedding row of `tok` → Q.F residual init in `out`. Out-of-range
    /// ids leave a zero state, as the float path does.
    fn emb_row_to_q(&self, tok: u32, out: &mut [i64]) {
        let hidden = out.len();
        let start = tok as usize * hidden * 2;
        if start + hidden * 2 <= self.emb.len() {
            bf16_row_to_q(&self.emb[start..start + hidden * 2], out);
        } else {
            out.fill(0);
        }
    }

    fn head_row(&self, j: usize, hidden: usize) -> &[u8] {
        &self.head[j * hidden * 2..(j + 1) * hidden * 2]
    }

    /// Pre-converted i16 hi/lo planes for row `j`, when preconversion is on
    /// for this model AND row `j` is not an exception row. `None` routes the
    /// caller to the on-the-fly BF16 path over `head_row(j, hidden)`.
    fn head_planes_row(&self, j: usize, hidden: usize) -> Option<(&[i16], &[i16])> {
        self.head_planes.as_ref().and_then(|hp| hp.row(j, hidden))
    }
}

// ---------------------------------------------------------------------------
// The engine: owns its own KV cache and buffers; never touches the float
// path's state.
// ---------------------------------------------------------------------------

pub struct CisPplResult {
    pub ppl: f64,
    /// FNV-1a 64 over the per-step integer-argmax token ids (u32 LE). Two
    /// conforming runs of the same sample must produce the same digest.
    pub argmax_digest: u64,
    /// Number of scored (predicted) tokens = sample length − 1.
    pub scored: usize,
}

pub struct CisEngine<'m, 'a> {
    model: &'m CisModel<'a>,
    mode: CisMode,
    kv: KVCache,
    rope: RopeCache,
    // integer state
    h: Vec<i64>,
    codes: Vec<i8>,
    acc_a: Vec<i32>,
    acc_b: Vec<i32>,
    fixed: Vec<i64>,
    logits: Vec<i64>,
    // hybrid f32 state
    qf: Vec<f32>,
    kf: Vec<f32>,
    vf: Vec<f32>,
    attn_out: Vec<f32>,
    scores: Vec<f32>,
    head_out: Vec<f32>,
    upf: Vec<f32>,
    gatef: Vec<f32>,
    // full-integer attention state (empty/None in Hybrid mode; sized for the
    // model in FullInt — note the integer KV cache is layers·max_pos·kv_dim
    // i32 pairs, so large-context models pay it only when the mode is on)
    qi: Vec<i32>,
    ki: Vec<i32>,
    vi: Vec<i32>,
    k_icache: Vec<i32>,
    v_icache: Vec<i32>,
    iscores: Vec<i64>,
    iprobs: Vec<i32>,
    exp_lut: Option<ExpLut>,
    rope_i: Option<RopeTableI>,
    isq_q30: i64,
    /// Leg C1 (2026-08-29 pre-reg): per (step, layer) sorted list of nonzero
    /// indices in the down_proj input (`self.fixed[..inter]` post-activation,
    /// pre-quant) — the active-neuron set a column-skip kernel would consume.
    /// `active_set_digest` feature only. Push order is chronological
    /// (step-major, layer-minor), which the digest fold in the example
    /// binary relies on. Diagnostic only; callers drain between runs.
    #[cfg(feature = "active_set_digest")]
    pub active_sets: alloc::vec::Vec<(usize, alloc::vec::Vec<u32>)>,
    /// Per-phase decode cycle counters (Amdahl decomposition of the FullInt
    /// verify path). `phase-timers` feature only: the field does not exist
    /// and no call site touches it when the feature is off. See
    /// `crate::phase_timers` for the RDTSC/RDTSCP fencing and
    /// `forward_step_int`'s doc comment for the FullInt phase mapping.
    /// `CisMode::Hybrid` is deliberately left uninstrumented here: it exists
    /// only for x86_64 A/B and its f32 attention/activation stages already
    /// have cross-ISA-agnostic timers in `aegis_core::inference` (exercised
    /// by `amdahl_decode`); every `agent_trace verify` replay constructs
    /// `CisMode::FullInt`.
    #[cfg(feature = "phase-timers")]
    pub phase_cycles: crate::phase_timers::PhaseCycles,
}

impl<'m, 'a> CisEngine<'m, 'a> {
    pub fn new(model: &'m CisModel<'a>) -> Self {
        Self::new_with_mode(model, CisMode::Hybrid)
    }

    pub fn new_with_mode(model: &'m CisModel<'a>, mode: CisMode) -> Self {
        let c = &model.config;
        let hidden = c.hidden_size;
        let inter = c.intermediate_size;
        let head_dim = hidden / c.num_attention_heads;
        let kv_dim = c.num_key_value_heads * head_dim;
        let max_pos = c.max_position_embeddings;
        let widest = hidden.max(inter);
        let full = mode == CisMode::FullInt;
        let z = |n: usize| if full { n } else { 0 };
        CisEngine {
            mode,
            kv: KVCache::new(model.layers.len(), c.num_key_value_heads, head_dim, max_pos),
            rope: RopeCache::new(max_pos, head_dim, c.rope_theta),
            h: vec![0i64; hidden],
            codes: vec![0i8; widest],
            acc_a: vec![0i32; widest],
            acc_b: vec![0i32; widest],
            fixed: vec![0i64; widest],
            logits: vec![0i64; c.vocab_size],
            qf: vec![0.0; hidden],
            kf: vec![0.0; kv_dim],
            vf: vec![0.0; kv_dim],
            attn_out: vec![0.0; hidden],
            scores: vec![0.0; max_pos],
            head_out: vec![0.0; head_dim],
            upf: vec![0.0; inter],
            gatef: vec![0.0; inter],
            qi: vec![0i32; z(hidden)],
            ki: vec![0i32; z(kv_dim)],
            vi: vec![0i32; z(kv_dim)],
            k_icache: vec![0i32; z(model.layers.len() * max_pos * kv_dim)],
            v_icache: vec![0i32; z(model.layers.len() * max_pos * kv_dim)],
            iscores: vec![0i64; z(max_pos)],
            iprobs: vec![0i32; z(max_pos)],
            exp_lut: full.then(ExpLut::new),
            rope_i: full.then(|| RopeTableI::new(max_pos, head_dim, c.rope_theta.to_bits())),
            isq_q30: inv_sqrt_q30(head_dim as u64),
            model,
            #[cfg(feature = "active_set_digest")]
            active_sets: alloc::vec::Vec::new(),
            #[cfg(feature = "phase-timers")]
            phase_cycles: crate::phase_timers::PhaseCycles::zero(),
        }
    }

    /// Zero the phase-cycle accumulators. Mirrors
    /// `TernaryInferenceEngine::reset_phase_cycles` (aegis-core/src/inference.rs):
    /// call this before the span whose cycles a caller wants isolated.
    /// `phase-timers` feature only.
    #[cfg(feature = "phase-timers")]
    pub fn reset_phase_cycles(&mut self) {
        self.phase_cycles = crate::phase_timers::PhaseCycles::zero();
    }

    /// Leg C1: byte-identical-by-construction ternary matvec dispatch (see
    /// `cis_avx2`/`cis_neon` module docs — both are proven bit-identical to
    /// `cis::ternary_matvec_i8`). Unified 2026-09: the default (no-feature)
    /// build now takes this same path, not just the `active_set_digest`
    /// verify build — see state/reports/2026-09-06-VERIFY-TIMING-2B-BOX1.md
    /// (8/8 suite receipts PASS bit-for-bit at 5.2-5.8x with this dispatch)
    /// and ledger A44 (AVX2 kernel byte-identical to scalar). Both
    /// `ternary_matvec_i8_avx2` and `ternary_matvec_i8_neon` do their OWN
    /// runtime capability check (`ops::simd_on()` / the neon force-scalar
    /// toggle) and fall back to `cis::ternary_matvec_i8` internally when the
    /// CPU lacks the ISA, the shape is too small to block, or the `-128`
    /// hazard is hit — so this function only has to pick the per-arch entry
    /// point, not re-implement feature detection. Under `scalar_only` we
    /// skip `cis_avx2` entirely (not merely rely on its internal check) so
    /// the CIS-1 chain keeps the zero-x86-intrinsics property that feature
    /// promises. Only the entry point is unconditional now; the
    /// active-neuron-set recording used for digesting is still gated behind
    /// `active_set_digest`, unchanged.
    #[inline]
    fn tmv_dispatch(
        output: &mut [i32],
        input: &[i8],
        weights_packed: &[u8],
        dim_out: usize,
        dim_in: usize,
    ) {
        #[cfg(all(
            target_arch = "x86_64",
            not(target_os = "uefi"),
            not(feature = "scalar_only")
        ))]
        {
            crate::cis_avx2::ternary_matvec_i8_avx2(output, input, weights_packed, dim_out, dim_in);
        }
        #[cfg(target_arch = "aarch64")]
        {
            crate::cis_neon::ternary_matvec_i8_neon(output, input, weights_packed, dim_out, dim_in);
        }
        #[cfg(not(any(
            all(
                target_arch = "x86_64",
                not(target_os = "uefi"),
                not(feature = "scalar_only")
            ),
            target_arch = "aarch64"
        )))]
        {
            crate::cis::ternary_matvec_i8(output, input, weights_packed, dim_out, dim_in);
        }
    }

    /// LM-head dot dispatch, same discipline as `tmv_dispatch`:
    /// `dot_i8_bf16q_avx2` does its own runtime `simd_on()` check and falls
    /// back to `dot_i8_bf16q` internally (including a whole-row scalar
    /// recompute whenever an element would trip one of `dot_i8_bf16q`'s own
    /// asserts, so the panic is reproduced exactly), so this function only
    /// picks the per-arch entry point. No NEON kernel exists yet for this
    /// dot (aarch64 uses the portable scalar reference); `scalar_only` skips
    /// `cis_avx2` entirely for the same zero-x86-intrinsics reason
    /// `tmv_dispatch` does.
    #[inline]
    fn lmhead_dot_dispatch(a: &[i8], row: &[u8]) -> i64 {
        #[cfg(all(
            target_arch = "x86_64",
            not(target_os = "uefi"),
            not(feature = "scalar_only")
        ))]
        {
            crate::cis_avx2::dot_i8_bf16q_avx2(a, row)
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            not(target_os = "uefi"),
            not(feature = "scalar_only")
        )))]
        {
            dot_i8_bf16q(a, row)
        }
    }

    /// LM-head dot dispatch over the pre-converted i16 hi/lo planes, same
    /// per-arch-entry-point-only discipline as `lmhead_dot_dispatch`:
    /// `dot_i8_i16planes_avx2` does its own runtime `simd_on()` check and
    /// falls back to the scalar `dot_i8_i16planes` internally. Callers only
    /// ever reach this for rows `CisModel::head_planes_row` returned
    /// `Some(..)` for, i.e. rows `HeadPlanes::build` already validated —
    /// there is no exception path to preserve here (that happens once, at
    /// build time, not per dot).
    #[inline]
    fn lmhead_dot_dispatch_planes(a: &[i8], hi: &[i16], lo: &[i16]) -> i64 {
        #[cfg(all(
            target_arch = "x86_64",
            not(target_os = "uefi"),
            not(feature = "scalar_only")
        ))]
        {
            crate::cis_avx2::dot_i8_i16planes_avx2(a, hi, lo)
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            not(target_os = "uefi"),
            not(feature = "scalar_only")
        )))]
        {
            dot_i8_i16planes(a, hi, lo)
        }
    }

    /// One decode step: token embedding through all layers, integer residual
    /// stream in `self.h`. Mirrors `TernaryInferenceEngine::forward_step`
    /// stage for stage.
    /// Materialize the integer logits for the current residual state (final
    /// norm + integer LM head) and return them, vocab-sized, as exact i64.
    /// This is the greedy-decode surface: pair with `argmax_i64`. It runs the
    /// same `logits_int` the perplexity path uses but deliberately discards
    /// the f64 NLL scale — decode must never touch a float, because the
    /// integer logits alone carry the cross-ISA identity claim that
    /// `arm-digest.yml` pins at token level.
    pub fn decode_logits(&mut self) -> &[i64] {
        self.logits_int();
        &self.logits[..self.model.config.vocab_size]
    }

    pub fn forward_step_int(&mut self, current_tok: u32, seq_pos: usize) {
        // Amdahl total span: one `record_total` pair per call, i.e. per
        // token forward-passed through every layer (prefill AND generated
        // tokens both call this). `logits_int` (final norm + LM head, only
        // called for generated tokens) records its OWN pair into the same
        // `total_raw`/`total_pairs` accumulator — see its doc comment for
        // why sharing one counter across two call-site kinds is deliberate,
        // not an oversight. `phase-timers` feature only.
        #[cfg(feature = "phase-timers")]
        let __amdahl_total_start = crate::phase_timers::tick_start();

        let c = &self.model.config;
        let hidden = c.hidden_size;
        let inter = c.intermediate_size;
        let num_heads = c.num_attention_heads;
        let num_kv_heads = c.num_key_value_heads;
        let head_dim = hidden / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // Embedding lookup: BF16 row converted to Q.F on the fly (exact),
        // integer residual. Out-of-range ids leave a zero state, as the
        // float path does.
        self.model.emb_row_to_q(current_tok, &mut self.h[..hidden]);

        let max_pos = c.max_position_embeddings;
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            // --- attention block -------------------------------------------
            let s_in = timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Norm, {
                normq(&mut self.codes[..hidden], &self.h[..hidden], &layer.ln1)
            });

            // Both modes end this stage with the attention output as fixed
            // point in self.fixed[..hidden] and yield the vector's fractional
            // width: Hybrid via the E2-7 per-vector block exponent (G ≤ F),
            // FullInt exactly F by construction. Everything after (o_proj,
            // residual, MLP norm) is shared.
            let g_attn = match self.mode {
                // Hybrid deliberately does not exist off x86: its attention
                // and activation legs run the f32 `ops` kernels, whose results
                // carry no cross-ISA identity claim. A non-x86 build gets
                // FullInt — the mode whose whole point is that it ports.
                #[cfg(not(target_arch = "x86_64"))]
                CisMode::Hybrid => unreachable!(
                    "CisMode::Hybrid requires the x86_64 f32 ops path; use CisMode::FullInt"
                ),
                #[cfg(target_arch = "x86_64")]
                CisMode::Hybrid => {
                    Self::tmv_dispatch(
                        &mut self.acc_a[..hidden],
                        &self.codes[..hidden],
                        layer.q_w,
                        hidden,
                        hidden,
                    );
                    let sq = dequant_scale_f64(&layer.q_s, &s_in);
                    for (o, &a) in self.qf.iter_mut().zip(&self.acc_a[..hidden]) {
                        *o = (a as f64 * sq) as f32;
                    }
                    Self::tmv_dispatch(
                        &mut self.acc_a[..kv_dim],
                        &self.codes[..hidden],
                        layer.k_w,
                        kv_dim,
                        hidden,
                    );
                    let sk = dequant_scale_f64(&layer.k_s, &s_in);
                    for (o, &a) in self.kf.iter_mut().zip(&self.acc_a[..kv_dim]) {
                        *o = (a as f64 * sk) as f32;
                    }
                    Self::tmv_dispatch(
                        &mut self.acc_a[..kv_dim],
                        &self.codes[..hidden],
                        layer.v_w,
                        kv_dim,
                        hidden,
                    );
                    let sv = dequant_scale_f64(&layer.v_s, &s_in);
                    for (o, &a) in self.vf.iter_mut().zip(&self.acc_a[..kv_dim]) {
                        *o = (a as f64 * sv) as f32;
                    }

                    // HYBRID STAGE 1: RoPE + scores + softmax + V mix (f32).
                    crate::attention::apply_rope(
                        &mut self.qf,
                        &mut self.kf,
                        seq_pos,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        &self.rope,
                    );
                    let (k_slot, v_slot) = self.kv.get_layer_mut(layer_idx, seq_pos);
                    k_slot.copy_from_slice(&self.kf);
                    v_slot.copy_from_slice(&self.vf);

                    self.attn_out.fill(0.0);
                    for h_idx in 0..num_heads {
                        let kv_h = h_idx / (num_heads / num_kv_heads);
                        for t in 0..=seq_pos {
                            let (k_cache, _) = self.kv.get_layer_mut(layer_idx, t);
                            let score = crate::ops::attn_dot(
                                &self.qf[h_idx * head_dim..(h_idx + 1) * head_dim],
                                &k_cache[kv_h * head_dim..(kv_h + 1) * head_dim],
                            );
                            self.scores[t] = score / libm::sqrtf(head_dim as f32);
                        }
                        crate::ops::softmax(&mut self.scores[0..=seq_pos]);
                        self.head_out.fill(0.0);
                        for t in 0..=seq_pos {
                            let (_, v_cache) = self.kv.get_layer_mut(layer_idx, t);
                            let w = self.scores[t];
                            crate::ops::attn_madd(
                                &mut self.head_out[0..head_dim],
                                w,
                                &v_cache[kv_h * head_dim..(kv_h + 1) * head_dim],
                            );
                        }
                        self.attn_out[h_idx * head_dim..(h_idx + 1) * head_dim]
                            .copy_from_slice(&self.head_out[..head_dim]);
                    }

                    // Back to integer: per-vector block-exponent fixing
                    // (spec E2-7) — the arm's value is the chosen width.
                    fix_f32_vec(&mut self.fixed[..hidden], &self.attn_out)
                }
                CisMode::FullInt => {
                    // FULL-INTEGER STAGE 1 (spec v0.3): q/k/v land on the
                    // absolute Q.QK_F grid via exact rational rescale;
                    // ROPE-I rotates them with the normative Q1.30 tables;
                    // scores are exact i128 dots scaled by 1/sqrt(head_dim)
                    // onto Q.SCORE_F; SOFTMAX-I yields Q0.15 probabilities;
                    // the V mix is an exact integer dot requantized to Q.F.
                    let lut = self.exp_lut.as_ref().expect("FullInt: exp LUT");
                    let rt = self.rope_i.as_ref().expect("FullInt: RoPE-I table");
                    timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Gemv, {
                        Self::tmv_dispatch(
                            &mut self.acc_a[..hidden],
                            &self.codes[..hidden],
                            layer.q_w,
                            hidden,
                            hidden,
                        );
                    });
                    let (neg, qs) = fixed_qscale(&layer.q_s, &s_in, QK_F);
                    for (o, &a) in self.qi[..hidden].iter_mut().zip(&self.acc_a[..hidden]) {
                        let v = qs.rescale(a as i64);
                        let v = if neg { -v } else { v };
                        // < 2^29 so the sqrt(2) growth under rotation stays
                        // inside i32 with headroom (cis_attn contract).
                        assert!(v.unsigned_abs() < 1 << 29, "FullInt: q exceeds Q.16 range");
                        *o = v as i32;
                    }
                    timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Gemv, {
                        Self::tmv_dispatch(
                            &mut self.acc_a[..kv_dim],
                            &self.codes[..hidden],
                            layer.k_w,
                            kv_dim,
                            hidden,
                        );
                    });
                    let (neg, qs) = fixed_qscale(&layer.k_s, &s_in, QK_F);
                    for (o, &a) in self.ki[..kv_dim].iter_mut().zip(&self.acc_a[..kv_dim]) {
                        let v = qs.rescale(a as i64);
                        let v = if neg { -v } else { v };
                        assert!(v.unsigned_abs() < 1 << 29, "FullInt: k exceeds Q.16 range");
                        *o = v as i32;
                    }
                    timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Gemv, {
                        Self::tmv_dispatch(
                            &mut self.acc_a[..kv_dim],
                            &self.codes[..hidden],
                            layer.v_w,
                            kv_dim,
                            hidden,
                        );
                    });
                    let (neg, qs) = fixed_qscale(&layer.v_s, &s_in, QK_F);
                    for (o, &a) in self.vi[..kv_dim].iter_mut().zip(&self.acc_a[..kv_dim]) {
                        let v = qs.rescale(a as i64);
                        let v = if neg { -v } else { v };
                        assert!(v.unsigned_abs() < 1 << 29, "FullInt: v exceeds Q.16 range");
                        *o = v as i32;
                    }

                    // Attn span, part 1: ROPE-I. Phase::Attn doc comment
                    // covers "RoPE-I + i128 score dots + SOFTMAX-I + V mix";
                    // the KV-cache *write* immediately below is timed
                    // separately as Phase::Kv (its doc explains why the
                    // read side stays inside Attn instead).
                    timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Attn, {
                        rope_apply_i(
                            &mut self.qi[..hidden],
                            &mut self.ki[..kv_dim],
                            seq_pos,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            rt,
                        );
                    });
                    let slot = (layer_idx * max_pos + seq_pos) * kv_dim;
                    timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Kv, {
                        self.k_icache[slot..slot + kv_dim].copy_from_slice(&self.ki[..kv_dim]);
                        self.v_icache[slot..slot + kv_dim].copy_from_slice(&self.vi[..kv_dim]);
                    });

                    // Attn span, part 2: score dots + SOFTMAX-I + V mix
                    // (includes the KV-cache READ traffic — see
                    // Phase::Kv's doc comment for why that read side is not
                    // split out).
                    timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Attn, {
                        for h_idx in 0..num_heads {
                            let kv_h = h_idx / (num_heads / num_kv_heads);
                            let qh = &self.qi[h_idx * head_dim..(h_idx + 1) * head_dim];
                            for t in 0..=seq_pos {
                                let kb = (layer_idx * max_pos + t) * kv_dim + kv_h * head_dim;
                                // Exact dot on the Q.(2·QK_F) grid; i128 so no
                                // headroom argument is ever needed.
                                let mut acc: i128 = 0;
                                for (qv, kv) in qh.iter().zip(&self.k_icache[kb..kb + head_dim]) {
                                    acc += *qv as i128 * *kv as i128;
                                }
                                // · 1/sqrt(head_dim) (Q0.30), onto Q.SCORE_F:
                                // 2·QK_F + 30 − SCORE_F = 38 bits back down.
                                let sc = rne_div(
                                    acc * self.isq_q30 as i128,
                                    1 << (2 * QK_F + 30 - SCORE_F),
                                );
                                assert!(
                                    sc >= i64::MIN as i128 && sc <= i64::MAX as i128,
                                    "FullInt: score exceeds i64"
                                );
                                self.iscores[t] = sc as i64;
                            }
                            softmax_i(
                                &mut self.iscores[..=seq_pos],
                                &mut self.iprobs[..=seq_pos],
                                lut,
                            );
                            for d in 0..head_dim {
                                // Σ_t p_t·v_t exact in i64 (≤ T·2^15·2^30),
                                // then Q.(QK_F+PROB_F) → Q.F.
                                let mut mix: i64 = 0;
                                for t in 0..=seq_pos {
                                    let vb = (layer_idx * max_pos + t) * kv_dim + kv_h * head_dim;
                                    mix += self.iprobs[t] as i64 * self.v_icache[vb + d] as i64;
                                }
                                self.fixed[h_idx * head_dim + d] =
                                    rne_div(mix as i128, 1 << (QK_F + PROB_F - F)) as i64;
                            }
                        }
                    });
                    // FullInt lands exactly on the Q.F grid.
                    F
                }
            };
            let s_o = timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Norm, {
                match &layer.attn_sub {
                    Some(g) => normq(&mut self.codes[..hidden], &self.fixed[..hidden], g),
                    None => quantq(&mut self.codes[..hidden], &self.fixed[..hidden], g_attn),
                }
            });
            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Gemv, {
                Self::tmv_dispatch(
                    &mut self.acc_a[..hidden],
                    &self.codes[..hidden],
                    layer.o_w,
                    hidden,
                    hidden,
                );
            });
            let (neg, qs) = residual_qscale(&layer.o_s, &s_o);
            for (hi, &a) in self.h[..hidden].iter_mut().zip(&self.acc_a[..hidden]) {
                let d = qs.rescale(a as i64);
                *hi += if neg { -d } else { d };
            }

            // --- MLP block --------------------------------------------------
            let s_mlp = timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Norm, {
                normq(&mut self.codes[..hidden], &self.h[..hidden], &layer.ln2)
            });
            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Gemv, {
                Self::tmv_dispatch(
                    &mut self.acc_a[..inter],
                    &self.codes[..hidden],
                    layer.up_w,
                    inter,
                    hidden,
                );
                Self::tmv_dispatch(
                    &mut self.acc_b[..inter],
                    &self.codes[..hidden],
                    layer.gate_w,
                    inter,
                    hidden,
                );
            });

            // Both modes end this stage with act(gate)·up as fixed point in
            // self.fixed[..inter], yielding the vector's fractional width
            // (block exponent for Hybrid, exact F for FullInt).
            let g_mlp = match self.mode {
                // Hybrid deliberately does not exist off x86: its attention
                // and activation legs run the f32 `ops` kernels, whose results
                // carry no cross-ISA identity claim. A non-x86 build gets
                // FullInt — the mode whose whole point is that it ports.
                #[cfg(not(target_arch = "x86_64"))]
                CisMode::Hybrid => unreachable!(
                    "CisMode::Hybrid requires the x86_64 f32 ops path; use CisMode::FullInt"
                ),
                #[cfg(target_arch = "x86_64")]
                CisMode::Hybrid => {
                    // HYBRID STAGE 2: silu(gate) · up (f32).
                    let su = dequant_scale_f64(&layer.up_s, &s_mlp);
                    let sg = dequant_scale_f64(&layer.gate_s, &s_mlp);
                    for (o, &a) in self.upf.iter_mut().zip(&self.acc_a[..inter]) {
                        *o = (a as f64 * su) as f32;
                    }
                    for (o, &a) in self.gatef.iter_mut().zip(&self.acc_b[..inter]) {
                        *o = (a as f64 * sg) as f32;
                    }
                    match c.hidden_act {
                        Activation::Relu2 => crate::ops::relu2(&mut self.gatef),
                        Activation::Silu => crate::ops::silu(&mut self.gatef),
                    }
                    for (u, &g) in self.upf.iter_mut().zip(self.gatef.iter()) {
                        *u *= g;
                    }

                    // Back to integer for the down projection: per-vector
                    // block-exponent fixing (spec E2-7).
                    fix_f32_vec(&mut self.fixed[..inter], &self.upf)
                }
                CisMode::FullInt => {
                    // FULL-INTEGER STAGE 2 (spec v0.3, ACT-I): gate and up
                    // land on the Q.F grid via exact rational rescale, then
                    // integer relu²/silu with RNE requants.
                    // Act span: elementwise integer relu²/silu (ACT-I) plus
                    // its immediate re-quantization onto a residual-safe
                    // fixed-point width — the FullInt path's counterpart to
                    // the MLP elementwise stage, timed separately from the
                    // up/gate and down_proj Gemv spans on either side of it.
                    timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Act, {
                        let lut = self.exp_lut.as_ref().expect("FullInt: exp LUT");
                        let (ng, qg) = fixed_qscale(&layer.gate_s, &s_mlp, F);
                        let (nu, qu) = fixed_qscale(&layer.up_s, &s_mlp, F);
                        for i in 0..inter {
                            let g = qg.rescale(self.acc_b[i] as i64);
                            let g = if ng { -g } else { g };
                            let u = qu.rescale(self.acc_a[i] as i64);
                            let u = if nu { -u } else { u };
                            assert!(
                                g.unsigned_abs() < 1 << 40 && u.unsigned_abs() < 1 << 40,
                                "FullInt: MLP value exceeds Q.20 headroom"
                            );
                            self.fixed[i] = match c.hidden_act {
                                Activation::Relu2 => relu2_q20(g, u),
                                Activation::Silu => silu_q20(g, u, lut),
                            };
                        }
                        // FullInt escape valve (spec §5.10 gap): the ACT-I
                        // output above is exact Q.20 already, but at
                        // BitNet-2B scale its magnitude can exceed the
                        // normq/quantq 2^50 residual headroom. Re-fix onto a
                        // per-vector block exponent G ≤ F, mirroring
                        // `fix_f32_vec`'s hybrid-boundary contract but
                        // derived purely from the integer magnitude (never a
                        // float). M7-scale products stay under the
                        // headroom, so this degenerates to G = F, unshifted.
                        fix_q_vec(&mut self.fixed[..inter])
                    })
                }
            };
            // Leg C1: capture the active-neuron set — sorted nonzero indices
            // of the down_proj input, at the exact instant a column-skip
            // kernel would consume it (post-activation, pre-quant; zero is
            // preserved by quantization, so this is also the set the
            // quantized `self.codes[..inter]` would carry into the matvec).
            #[cfg(feature = "active_set_digest")]
            {
                let idxs: alloc::vec::Vec<u32> = self.fixed[..inter]
                    .iter()
                    .enumerate()
                    .filter(|&(_, &v)| v != 0)
                    .map(|(i, _)| i as u32)
                    .collect();
                self.active_sets.push((layer_idx, idxs));
            }
            let s_down = timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Norm, {
                match &layer.ffn_sub {
                    Some(g) => normq(&mut self.codes[..inter], &self.fixed[..inter], g),
                    None => quantq(&mut self.codes[..inter], &self.fixed[..inter], g_mlp),
                }
            });
            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Gemv, {
                Self::tmv_dispatch(
                    &mut self.acc_a[..hidden],
                    &self.codes[..inter],
                    layer.down_w,
                    hidden,
                    inter,
                );
            });
            let (neg, qs) = residual_qscale(&layer.down_s, &s_down);
            for (hi, &a) in self.h[..hidden].iter_mut().zip(&self.acc_a[..hidden]) {
                let d = qs.rescale(a as i64);
                *hi += if neg { -d } else { d };
            }
        }

        #[cfg(feature = "phase-timers")]
        {
            let __amdahl_total_end = crate::phase_timers::tick_end();
            self.phase_cycles
                .record_total(__amdahl_total_start, __amdahl_total_end);
        }
    }

    /// Final norm + integer LM head over the current residual state.
    /// Fills `self.logits` (exact i64) and returns the real-value scale of
    /// one logit unit (for NLL only; argmax and the witness digest use the
    /// integer logits directly).
    fn logits_int(&mut self) -> f64 {
        // Amdahl total span: one `record_total` pair per call, sharing
        // `self.phase_cycles.total_*` with `forward_step_int`'s own
        // per-token span — see that function's doc comment for why. A
        // single local `ret` (no early `return`) keeps this span's end
        // reachable from every code path. `phase-timers` feature only.
        #[cfg(feature = "phase-timers")]
        let __amdahl_total_start = crate::phase_timers::tick_start();

        let c = &self.model.config;
        let hidden = c.hidden_size;
        let vocab = c.vocab_size;
        let s = timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Norm, {
            normq(
                &mut self.codes[..hidden],
                &self.h[..hidden],
                &self.model.final_g,
            )
        });
        timed_phase!(self.phase_cycles, crate::phase_timers::Phase::LmHead, {
            for (j, l) in self.logits[..vocab].iter_mut().enumerate() {
                *l = match self.model.head_planes_row(j, hidden) {
                    Some((hi, lo)) => {
                        Self::lmhead_dot_dispatch_planes(&self.codes[..hidden], hi, lo)
                    }
                    None => Self::lmhead_dot_dispatch(
                        &self.codes[..hidden],
                        self.model.head_row(j, hidden),
                    ),
                };
            }
        });
        // logit_real = acc · s / 2^F  (the head table is Q.F).
        let ret = if s.num == 0 {
            0.0
        } else {
            (s.num as f64) / (s.den as f64) / exp2_f64(F as i32)
        };

        #[cfg(feature = "phase-timers")]
        {
            let __amdahl_total_end = crate::phase_timers::tick_end();
            self.phase_cycles
                .record_total(__amdahl_total_start, __amdahl_total_end);
        }
        ret
    }

    /// Teacher-forced perplexity over `tokens`, integer path, mirroring
    /// `TernaryInferenceEngine::calculate_perplexity` step for step.
    /// Also folds every step's integer argmax into an FNV-1a 64 digest —
    /// the determinism exhibit.
    pub fn calculate_perplexity_int(&mut self, tokens: &[u32]) -> CisPplResult {
        let window = self.model.config.max_position_embeddings;
        let tokens = if tokens.len() > window {
            &tokens[..window]
        } else {
            tokens
        };
        let mut digest = FNV1A64_OFFSET;
        if tokens.len() < 2 {
            return CisPplResult {
                ppl: 0.0,
                argmax_digest: digest,
                scored: 0,
            };
        }
        self.kv.reset_prefix(tokens.len());

        let mut total_nll = 0.0f64;
        let mut count = 0usize;

        self.forward_step_int(tokens[0], 0);
        for i in 0..tokens.len() - 1 {
            let target = tokens[i + 1] as usize;
            let scale = self.logits_int();
            let vocab = self.model.config.vocab_size;
            if target >= vocab {
                return CisPplResult {
                    ppl: f64::NAN,
                    argmax_digest: digest,
                    scored: count,
                };
            }
            let arg = argmax_i64(&self.logits[..vocab]);
            digest = fnv1a64(digest, &arg.to_le_bytes());

            let max = self.logits[..vocab].iter().copied().max().unwrap_or(0);
            let mut sum_exp = 0.0f64;
            for &l in &self.logits[..vocab] {
                sum_exp += libm::exp((l - max) as f64 * scale);
            }
            let nll = -((self.logits[target] - max) as f64 * scale - libm::log(sum_exp));
            total_nll += nll;
            count += 1;

            self.forward_step_int(tokens[i + 1], i + 1);
        }

        CisPplResult {
            ppl: libm::exp(total_nll / count as f64),
            argmax_digest: digest,
            scored: count,
        }
    }
}

// ---------------------------------------------------------------------------
// Unit goldens. Constants computed by the INDEPENDENT big-int generator
// `scripts/cis_e2_golden_gen.py` (Python arbitrary-precision integers,
// scripted from the same spec text) — never by the Rust under test.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    #[test]
    fn golden_bf16_to_fixed() {
        const CASES: [(u16, i64); 11] = [
            (0x3F80, 1048576),
            (0xBF80, -1048576),
            (0x0000, 0),
            (0x8000, 0),
            (0x3FC0, 1572864),
            (0x4248, 52428800),
            (0x0001, 0),
            (0x8001, 0),
            (0x3881, 64), // tie x.5 → even (down)
            (0x3883, 66), // tie x.5 → even (up)
            (0x2E00, 0),
        ];
        for (bits, want) in CASES {
            assert_eq!(bf16_to_fixed(bits, 20), want, "bf16 {bits:#06X}");
        }
    }

    #[test]
    fn golden_f32_to_fixed() {
        const CASES: [(u32, i64); 11] = [
            (0x3F800000, 1048576),
            (0xBF800000, -1048576),
            (0x00000000, 0),
            (0x80000000, 0),
            (0x3F400000, 786432),
            (0x42F6E979, 129453000),
            (0x40800001, 4194304), // tie → even (down)
            (0x40800003, 4194306), // tie → even (up)
            (0xB48637BD, 0),
            // E2-7 recalibration goldens: BitNet-2B-scale hybrid values past
            // the old sh ≤ 20 guard, inside the new sh ≤ 26 / 2^50 headroom.
            (0x4D8F0D18, 314572800000000),   // 3.0e8
            (0xCE7F1B9E, -1121976320000000), // -1.07e9
        ];
        for (bits, want) in CASES {
            assert_eq!(f32_to_fixed(bits, 20), want, "f32 {bits:#010X}");
        }
    }

    #[test]
    fn golden_f32_to_ratio() {
        const CASES: [(u32, bool, u64, i32); 7] = [
            (0x3F800000, false, 1, 0),
            (0xBF800000, true, 1, 0),
            (0x00000000, false, 0, 0),
            (0x3D4CCCCD, false, 13421773, -28), // 0.05
            (0x40C80000, false, 25, -2),        // 6.25
            (0xBAB43958, true, 1476395, -30),   // -0.001375
            (0x40400000, false, 3, 0),          // 3.0
        ];
        for (bits, neg, m, e) in CASES {
            assert_eq!(f32_to_ratio(bits), FRatio { neg, m, e }, "f32 {bits:#010X}");
        }
    }

    #[test]
    fn golden_qscale64_from_ratio() {
        const CASES: [(u128, u128, u64, i32); 7] = [
            (1, 3, 6148914691236517205, -64),
            (5, 4, 5764607523034234880, -62),
            (7, 1, 8070450532247928832, -60),
            (1, 1, 4611686018427387904, -62),
            (38100, 1099511627776, 5362098306337996800, -87),
            (
                1237940039285380274899136569, // 2^90 + 12345
                205891132094649,              // 3^30
                6304663058712877703,
                -20,
            ),
            (0, 5, 0, 0),
        ];
        for (num, den, m, e) in CASES {
            let q = QScale64::from_ratio(num, den);
            assert_eq!((q.m(), q.e()), (m, e), "from_ratio({num}, {den})");
        }
        // Shift invariance: the ratio, not the representation, is normative.
        assert_eq!(
            QScale64::from_ratio(1 << 3, 3 << 3),
            QScale64::from_ratio(1, 3)
        );
    }

    #[test]
    fn golden_rescale() {
        const CASES: [(i64, u64, i32, i64); 6] = [
            (1000, 6148914691236517205, -64, 333),
            (-1000, 6148914691236517205, -64, -333),
            (123456789, 7244020073193463301, -61, 387850974),
            (-7, 4951760157141521100, -28, -129127208516),
            (3, 4611686018427387904, -63, 2), // tie 1.5 → 2
            (1, 4611686018427387904, -63, 0), // tie 0.5 → 0
        ];
        for (x, m, e, want) in CASES {
            let q = QScale64 { m, e };
            assert_eq!(q.rescale(x), want, "rescale({x}, m={m}, e={e})");
        }
    }

    #[test]
    fn golden_normq_16() {
        const GOLDEN_CODES: [i8; 16] = [
            82, 12, -12, -67, 73, -100, -106, 10, -54, -53, -40, -108, 69, -114, 70, 127,
        ];
        const GOLDEN_NUM: u128 = 17353612027868592;
        const GOLDEN_DEN: u128 = 1280903443508101120;
        let mut state = 0xA1CE_5EED_0000_0005u64;
        let mut h = [0i64; 16];
        for x in h.iter_mut() {
            *x = ((lcg_next(&mut state) >> 20) % (1 << 31)) as i64 - (1 << 30);
        }
        let mut g = [0i32; 16];
        for x in g.iter_mut() {
            *x = ((lcg_next(&mut state) >> 40) % (1 << 20)) as i32 + (1 << 19);
        }
        let mut codes = [0i8; 16];
        let s = normq(&mut codes, &h, &g);
        assert_eq!(codes, GOLDEN_CODES);
        assert_eq!(s.num, GOLDEN_NUM);
        assert_eq!(s.den, GOLDEN_DEN);
    }

    #[test]
    fn golden_quantq_16() {
        const GOLDEN_CODES: [i8; 16] = [
            -127, -66, -3, -5, -82, 91, -10, 39, 47, -54, -64, -73, 40, 44, 86, 97,
        ];
        let mut state = 0xA1CE_5EED_0000_0006u64;
        let mut h = [0i64; 16];
        for x in h.iter_mut() {
            *x = ((lcg_next(&mut state) >> 22) % (1 << 30)) as i64 - (1 << 29);
        }
        let mut codes = [0i8; 16];
        let s = quantq(&mut codes, &h, F);
        assert_eq!(codes, GOLDEN_CODES);
        assert_eq!(s.num, 505618677);
        assert_eq!(s.den, 133169152);
    }

    #[test]
    fn fix_f32_vec_block_exponent() {
        // Small vectors keep G = F and match per-element Q.20 fixing.
        let src = [1.0f32, -0.75, 123.456, 0.0];
        let mut out = [0i64; 4];
        let g = fix_f32_vec(&mut out, &src);
        assert_eq!(g, F);
        for (o, v) in out.iter().zip(src) {
            assert_eq!(*o, f32_to_fixed(v.to_bits(), F));
        }
        // A 2B-scale outlier (≥ 2^30 real) forces G < F: max_exp for 2^33
        // is 160, so G = 176 − 160 = 16; every element still < 2^50 and
        // exactly the RNE fixing at G bits.
        let big = [8.6e9f32, -1.0, 3.5];
        let mut out = [0i64; 3];
        let g = fix_f32_vec(&mut out, &big);
        assert_eq!(g, 16);
        for (o, v) in out.iter().zip(big) {
            assert_eq!(*o, f32_to_fixed(v.to_bits(), 16));
            assert!(o.unsigned_abs() < 1 << 50);
        }
        // normq G-invariance: the same real vector fixed at G and at G−2
        // (values ×4 exactly) must give identical i8 codes; the carried
        // scale agrees up to the exact-isqrt floor granularity (rel O(1/t)).
        let h16: [i64; 4] = [1 << 30, -(3 << 28), 5 << 27, 1 << 20];
        let h14: [i64; 4] = [1 << 32, -(3 << 30), 5 << 29, 1 << 22];
        let gains = [1i32 << 20; 4];
        let mut c16 = [0i8; 4];
        let mut c14 = [0i8; 4];
        let s16 = normq(&mut c16, &h16, &gains);
        let s14 = normq(&mut c14, &h14, &gains);
        assert_eq!(c16, c14);
        let r16 = s16.num as f64 / s16.den as f64;
        let r14 = s14.num as f64 / s14.den as f64;
        assert!(((r16 - r14) / r16).abs() < 1e-8, "{r16} vs {r14}");
    }

    #[test]
    fn golden_normq_16_wide_headroom() {
        // E2-7 recalibration golden: residuals near ±2^49 (inside the new
        // 2^50 bound, far past the old 2^45) must stay exact end to end.
        // Constants from scripts/cis_e2_golden_gen.py (seed …0009 variant).
        const GOLDEN_CODES: [i8; 16] = [
            65, -21, 124, 37, -65, 78, -81, 127, 103, 92, 88, -100, 74, 40, 20, 34,
        ];
        const GOLDEN_NUM: u128 = 5574890170215081393120;
        const GOLDEN_DEN: u128 = 508967061582749964959744;
        let mut state = 0xA1CE_5EED_0000_0009u64;
        let mut h = [0i64; 16];
        for x in h.iter_mut() {
            *x = ((lcg_next(&mut state) >> 14) % (1 << 50)) as i64 - (1 << 49);
        }
        let mut g = [0i32; 16];
        for x in g.iter_mut() {
            *x = ((lcg_next(&mut state) >> 40) % (1 << 20)) as i32 + (1 << 19);
        }
        let mut codes = [0i8; 16];
        let s = normq(&mut codes, &h, &g);
        assert_eq!(codes, GOLDEN_CODES);
        assert_eq!(s.num, GOLDEN_NUM);
        assert_eq!(s.den, GOLDEN_DEN);
    }

    #[test]
    fn normq_all_zero() {
        let h = [0i64; 8];
        let g = [1i32 << 20; 8];
        let mut codes = [9i8; 8];
        let s = normq(&mut codes, &h, &g);
        assert_eq!(codes, [0; 8]);
        assert_eq!((s.num, s.den), (0, 1));
    }

    #[test]
    fn golden_head_dot_32() {
        let mut state = 0xA1CE_5EED_0000_0007u64;
        let mut a = [0i8; 32];
        for x in a.iter_mut() {
            *x = (((lcg_next(&mut state) >> 40) % 255) as i32 - 127) as i8;
        }
        let mut e = [0i32; 32];
        for x in e.iter_mut() {
            *x = ((lcg_next(&mut state) >> 24) % (1 << 26)) as i32 - (1 << 25);
        }
        assert_eq!(dot_i8_i32(&a, &e), 18272237304);
    }

    #[test]
    fn golden_fnv1a64() {
        // Published FNV-1a 64 test vectors + the generator's sequence case.
        assert_eq!(fnv1a64(FNV1A64_OFFSET, b""), 0xCBF29CE484222325);
        assert_eq!(fnv1a64(FNV1A64_OFFSET, b"a"), 0xAF63DC4C8601EC8C);
        assert_eq!(fnv1a64(FNV1A64_OFFSET, b"foobar"), 0x85944171F73967E8);
        let mut d = FNV1A64_OFFSET;
        for t in [0u32, 1, 65535, 8191] {
            d = fnv1a64(d, &t.to_le_bytes());
        }
        assert_eq!(d, 0xB28EF838942854C0);
    }

    #[test]
    fn rne_shr_matches_rne_div_exhaustive() {
        // The shift/mask fast path must be bit-identical to the module's one
        // rounding primitive. Sweep every mantissa a BF16 can produce
        // (m ≤ 0xFF) plus larger values, across every shift bf16_to_fixed
        // can request.
        for m in 0..(1i64 << 10) {
            for k in 1..=62 {
                assert_eq!(
                    rne_shr(m, k),
                    rne_div(m as i128, 1i128 << k) as i64,
                    "rne_shr({m}, {k})"
                );
            }
        }
        for m in [i64::MAX, i64::MAX - 1, (1 << 62) + 12345, (1 << 45) - 3] {
            for k in 1..=62 {
                assert_eq!(rne_shr(m, k), rne_div(m as i128, 1i128 << k) as i64);
            }
        }
    }

    #[test]
    fn rne_shr_i128_matches_rne_div_small_sweep() {
        // (a) small signed values, every shift 1..=126.
        for p in -(1i128 << 12)..(1i128 << 12) {
            for k in 1u32..=126 {
                assert_eq!(
                    rne_shr_i128(p, k),
                    rne_div(p, 1i128 << k),
                    "rne_shr_i128({p}, {k})"
                );
            }
        }
    }

    #[test]
    fn rne_shr_i128_matches_rne_div_edge_values() {
        // (b) edge magnitudes (near the i128 headroom bound, near ties) at a
        // representative spread of shifts.
        for k in [1u32, 2, 20, 31, 62, 63, 64, 100, 126] {
            let half = 1i128 << (k - 1);
            let mut vals: Vec<i128> = vec![
                (1i128 << 126) - 1,
                (1i128 << 126) - (1i128 << k),
                1i128 << 125,
                (1i128 << 63) + 1,
                (1i128 << 63) - 1,
                half,
                3 * half,
                half + 1,
                half - 1,
            ];
            vals.extend(vals.clone().iter().map(|v| -v));
            for p in vals {
                assert_eq!(
                    rne_shr_i128(p, k),
                    rne_div(p, 1i128 << k),
                    "rne_shr_i128({p}, {k})"
                );
            }
        }
    }

    #[test]
    fn rne_shr_i128_matches_rne_div_random() {
        // (c) 10_000 pseudo-random signed i128 with |p| < 2^126, k in 1..=126
        // (aegis-core is no_std + libm only — reuse the module's seeded LCG,
        // not a rand crate).
        let mut state = 0xAC7_5EED_0000_00C1u64 ^ 0x1234_5678_9ABC_DEF0;
        for _ in 0..10_000 {
            let hi = lcg_next(&mut state) as i128;
            let lo = lcg_next(&mut state) as i128;
            let mut p = ((hi << 64) | (lo & 0xFFFF_FFFF_FFFF_FFFF)) % (1i128 << 126);
            if lcg_next(&mut state) & 1 == 1 {
                p = -p;
            }
            let k = 1 + (lcg_next(&mut state) % 126) as u32;
            assert_eq!(
                rne_shr_i128(p, k),
                rne_div(p, 1i128 << k),
                "rne_shr_i128({p}, {k})"
            );
        }
    }

    #[test]
    fn qscale64_rescale_matches_inline_rne_div() {
        // Round-trip: QScale64::rescale must equal the old formula (rne_div
        // against a runtime power of two) computed inline, for random
        // (m, e, x).
        let mut state = 0xAC7_5EED_0000_00C2u64;
        for _ in 0..2_000 {
            let m = (1u64 << 62) + (lcg_next(&mut state) % (1u64 << 62));
            let e = -1 - (lcg_next(&mut state) % 126) as i32; // -126..=-1
            let x = lcg_next(&mut state) as i64;
            let scale = QScale64 { m, e };
            let p = x as i128 * m as i128;
            let want = if -e >= 127 {
                0
            } else {
                rne_div(p, 1i128 << (-e))
            };
            assert_eq!(scale.rescale(x) as i128, want, "m={m} e={e} x={x}");
        }
    }

    #[test]
    fn dot_i8_bf16q_matches_preconverted_dot() {
        // The on-the-fly LM-head dot must equal dot_i8_i32 over a
        // pre-converted row — same values, same order, same accumulator.
        let mut state = 0xA1CE_5EED_0000_0008u64;
        let n = 64;
        let mut a = vec![0i8; n];
        for x in a.iter_mut() {
            *x = (((lcg_next(&mut state) >> 40) % 255) as i32 - 127) as i8;
        }
        let mut row = vec![0u8; n * 2];
        let mut conv = vec![0i32; n];
        for i in 0..n {
            // Finite BF16 patterns inside the Q.20 i32 range: exp in
            // [96,137] covers ~2^-31..~2^10, both signs, all mantissas.
            let sign = ((lcg_next(&mut state) >> 33) & 1) as u16;
            let exp = (96 + (lcg_next(&mut state) >> 40) % 42) as u16;
            let man = ((lcg_next(&mut state) >> 40) % 128) as u16;
            let bits = (sign << 15) | (exp << 7) | man;
            row[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            conv[i] = bf16_to_fixed(bits, F) as i32;
        }
        assert_eq!(dot_i8_bf16q(&a, &row), dot_i8_i32(&a, &conv));
    }

    #[test]
    fn argmax_i64_ties_break_low() {
        assert_eq!(argmax_i64(&[5, 9, 9, 1]), 1);
        assert_eq!(argmax_i64(&[3, 3, 3]), 0);
        assert_eq!(argmax_i64(&[i64::MIN, i64::MIN]), 0);
    }

    /// `TernaryEngine::tmv_dispatch` is now the single (no-cfg-split)
    /// dispatch used by every build. It must equal the scalar reference on
    /// every shape it might route to `cis_avx2`/`cis_neon`, including the
    /// tail/fallback/-128-hazard shapes those kernels themselves carve out.
    /// This is deliberately independent of `cis_avx2_equivalence.rs` /
    /// `cis_neon_equivalence.rs` (which test the kernels directly): this one
    /// pins the *dispatch function* the inference engine actually calls.
    #[test]
    fn tmv_dispatch_matches_scalar_reference() {
        let mut state = 0xC15_DEED_7C4E_00D1u64.wrapping_add(1);
        let shapes = [(1usize, 128usize), (3, 132), (5, 260), (7, 64), (2, 4)];
        for (dim_out, dim_in) in shapes {
            let n_bytes = dim_in / 4;
            let mut input = vec![0i8; dim_in];
            for x in input.iter_mut() {
                *x = (((lcg_next(&mut state) >> 40) % 255) as i32 - 127) as i8;
            }
            let mut weights = vec![0u8; dim_out * n_bytes];
            for w in weights.iter_mut() {
                *w = (lcg_next(&mut state) & 0xFF) as u8;
            }
            let mut want = vec![0i32; dim_out];
            let mut got = vec![0i32; dim_out];
            crate::cis::ternary_matvec_i8(&mut want, &input, &weights, dim_out, dim_in);
            CisEngine::tmv_dispatch(&mut got, &input, &weights, dim_out, dim_in);
            assert_eq!(
                want, got,
                "tmv_dispatch diverged from scalar reference at dim_out={dim_out} dim_in={dim_in}"
            );
        }
    }
}
