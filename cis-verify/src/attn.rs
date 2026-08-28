//! CIS-1 §5.7–§5.10 — exp machinery, SOFTMAX-I, ROPE-I, ACT-I.
//!
//! Transcribed per `docs/design/CIS_VERIFY_DESIGN.md` builder task 4
//! (§6.2 item 4), source of truth `docs/CIS-1_SPEC_v1.0.md` (v1.0.2)
//! §5.7–§5.10, from `aegis-core/src/cis_attn.rs` (whole file: constants,
//! `exp2_chain`/`exp2_neg_frac`/`ExpLut`/`exp_neg_q31`, `softmax_i`,
//! `log2_q32_f32`/`sincos_q62`/`RopeTableI`/`rope_apply_i`/`inv_sqrt_q30`,
//! `relu2_q20`/`silu_q20`) — every function below cites the reference
//! file:line it transcribes. Where the spec prose is ambiguous the
//! reference is definitive (spec §5).
//!
//! `core`-only integer arithmetic — no floats, no `libm`, no `unsafe`. All
//! rounding goes through `crate::ops::rne_div` (round-half-even), the same
//! single rounding primitive `ops.rs` uses — one rounding function for the
//! whole crate, mirroring `cis_attn.rs`'s reuse of `cis::rne_div`.

use alloc::vec::Vec;

use crate::ops::rne_div;

// ---------------------------------------------------------------------------
// Normative constants (spec §2: pinned integer literals, RNE of published
// 50-digit decimal expansions). Identical to `cis_attn.rs:30-37`.
// ---------------------------------------------------------------------------

/// 2π in Q2.62.
pub const TWO_PI_Q62: u128 = 28976077832308491370;
/// π in Q2.62.
pub const PI_Q62: u128 = 14488038916154245685;
/// π/2 in Q2.62, independently rounded.
pub const PI_2_Q62: u128 = 7244019458077122842;
/// log2(e) in Q32.32.
pub const LOG2E_Q32: i64 = 6196328019;

// ---------------------------------------------------------------------------
// §5.7 exp machinery (shared by SOFTMAX-I, ACT-I, ROPE-I generation) —
// cis_attn.rs:39-145
// ---------------------------------------------------------------------------

/// Chain constants `C[k] = 2^(-2^(-k))`, k = 1..32, Q0.62, by the
/// parameter-free floor-isqrt chain: `C[0] = 2^61`, `C[k] = isqrt(C[k-1]·2^62)`
/// (spec §5.7). Identical to `cis_attn.rs:47-54`.
pub fn exp2_chain() -> [u64; 33] {
    let mut c = [0u64; 33];
    c[0] = 1 << 61;
    for k in 1..=32 {
        c[k] = (((c[k - 1] as u128) << 62).isqrt()) as u64;
    }
    c
}

/// `2^(-f)` for `f` in Q0.32 ∈ [0, 2^32), result Q0.62 ∈ (2^61, 2^62].
/// Product over the set bits of `f` of the chain constants, RNE-requantized
/// after each multiply. Identical to `cis_attn.rs:62-71`.
pub fn exp2_neg_frac(f_q32: u64, chain: &[u64; 33]) -> u128 {
    debug_assert!(f_q32 < 1 << 32);
    let mut acc: u128 = 1 << 62;
    for (k, &c) in chain.iter().enumerate().take(33).skip(1) {
        if (f_q32 >> (32 - k)) & 1 == 1 {
            acc = rne_div((acc * c as u128) as i128, 1 << 62) as u128;
        }
    }
    acc
}

/// The normative exp LUT (spec §5.7): `E[i] = rne(2^(-i/1024)·2^31)` for
/// i = 0..1023 via the chain, plus the exact endpoint `E[1024] = 2^30`.
/// 1025 Q0.31 entries, strictly decreasing; FNV-1a 64 digest
/// `0x66C2A0EEB8C2DC43` (normative, pinned test below). Identical to
/// `cis_attn.rs:78-91`.
pub struct ExpLut {
    e: Vec<i64>,
}

impl ExpLut {
    pub fn new() -> ExpLut {
        let chain = exp2_chain();
        let mut e = Vec::with_capacity(1025);
        for i in 0..1024u64 {
            e.push(rne_div(exp2_neg_frac(i << 22, &chain) as i128, 1 << 31) as i64);
        }
        e.push(1 << 30);
        ExpLut { e }
    }

    /// Entry access for goldens/digests.
    pub fn entry(&self, i: usize) -> i64 {
        self.e[i]
    }

    /// `2^(-f)` for `f` in Q0.32 ∈ [0, 2^32), by linear interpolation
    /// between the two straddling entries (top 10 bits index, low 22 bits
    /// interpolate). Identical to `cis_attn.rs:103-110`.
    pub fn exp2_neg(&self, f_q32: u64) -> i64 {
        debug_assert!(f_q32 < 1 << 32);
        let i = (f_q32 >> 22) as usize;
        let r = (f_q32 & ((1 << 22) - 1)) as i128;
        let a = self.e[i];
        let b = self.e[i + 1];
        a - rne_div((a - b) as i128 * r, 1 << 22) as i64
    }
}

impl Default for ExpLut {
    fn default() -> Self {
        ExpLut::new()
    }
}

/// `e^(-z)` for `z ≥ 0` in Q32.32 (as u128 so wide-grid callers never
/// overflow), result Q0.31 ∈ [0, 2^31] (spec §5.7): route through base 2,
/// `y = z·log2(e)`, split into integer part `n` and RNE-rounded Q0.32
/// fraction `f`, then `E(f) / 2^n`; `n ≥ 31` underflows to exactly 0.
/// Identical to `cis_attn.rs:130-145`.
pub fn exp_neg_q31(z_q32: u128, lut: &ExpLut) -> i64 {
    let y = z_q32 * LOG2E_Q32 as u128; // Q.64; callers keep z < 2^68
    let mut n = (y >> 64) as u32;
    if n >= 31 {
        return 0;
    }
    let mut f = rne_div((y & ((1u128 << 64) - 1)) as i128, 1 << 32) as u64;
    if f == 1 << 32 {
        n += 1;
        f = 0;
        if n >= 31 {
            return 0;
        }
    }
    rne_div(lut.exp2_neg(f) as i128, 1 << n) as i64
}

// ---------------------------------------------------------------------------
// §5.8 SOFTMAX-I — cis_attn.rs:150-197
// ---------------------------------------------------------------------------

/// SOFTMAX-I (spec §5.8): input on the Q.24 score grid (`i64`). Max-subtract
/// exact; `e_t = exp_neg((m - s_t) << 8)`; sum in `i64`;
/// `p_t = rne(e_t·2^15 / Σe)` in Q0.15 by exact RNE division (**ratified**,
/// replacing the draft's Newton reciprocal). `scores` is overwritten with
/// the intermediate `e_t` values. Identical to `cis_attn.rs:173-197`.
pub fn softmax_i(scores: &mut [i64], probs: &mut [i32], lut: &ExpLut) {
    let n = scores.len();
    assert!(n > 0, "softmax_i: empty scores");
    assert!(
        n <= 1 << 20,
        "softmax_i: sequence too long for exact i64 sum"
    );
    assert!(probs.len() >= n, "softmax_i: probs buffer too short");
    let m = *scores.iter().max().unwrap();
    for s in scores.iter_mut() {
        let z = (m - *s) as u128; // >= 0, exact
        *s = exp_neg_q31(z << 8, lut); // Q.24 -> Q.32 argument, exact shift
    }
    let mut sum: i64 = 0;
    for &e in scores.iter() {
        sum += e;
    }
    debug_assert!(
        sum >= 1 << 31,
        "softmax_i: max element must contribute 2^31"
    );
    for (p, &e) in probs.iter_mut().zip(scores.iter()) {
        *p = rne_div((e as i128) << 15, sum as i128) as i32;
    }
}

// ---------------------------------------------------------------------------
// §5.9 ROPE-I: normative integer sin/cos tables + rotation —
// cis_attn.rs:203-372
// ---------------------------------------------------------------------------

/// `log2(value)` of a finite f32 ≥ 2, in Q32.32, by shift-and-square on the
/// bit pattern (spec §5.9 step 1). Identical to `cis_attn.rs:208-227`.
pub fn log2_q32_f32(bits: u32) -> u64 {
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;
    assert!(
        bits >> 31 == 0 && exp != 0xFF && exp >= 128,
        "log2_q32_f32: RoPE base must be finite and >= 2"
    );
    let e = (exp - 127) as u64;
    let mut m: u128 = ((man as u128) | (1 << 23)) << 39; // Q2.62 in [2^62, 2^63)
    let mut frac: u64 = 0;
    for _ in 0..32 {
        m = rne_div((m * m) as i128, 1 << 62) as u128;
        frac <<= 1;
        if m >= 1 << 63 {
            m = rne_div(m as i128, 2) as u128;
            frac |= 1;
        }
    }
    (e << 32) | frac
}

/// `(sin x, cos x)` for `x ∈ [0, 2π)` in Q2.62, results signed Q0.62 (spec
/// §5.9 step 4): quadrant reduction against the pinned π constants, then a
/// 9-term Taylor sum in Q0.62 with RNE requant after each multiply and
/// exact integer factorial divisors. Identical to `cis_attn.rs:238-275`.
pub fn sincos_q62(x: u128) -> (i64, i64) {
    debug_assert!(
        x < TWO_PI_Q62,
        "sincos_q62: argument must be reduced mod 2pi"
    );
    let mut x = x;
    let mut sign_s: i128 = 1;
    let mut sign_c: i128 = 1;
    if x >= PI_Q62 {
        x -= PI_Q62;
        sign_s = -1;
        sign_c = -1;
    }
    if x >= PI_2_Q62 {
        x = PI_Q62.saturating_sub(x).min(PI_2_Q62);
        sign_c = -sign_c;
    }
    let xi = x as i128; // <= pi/2 . 2^62 < 2^63
    let x2 = rne_div(xi * xi, 1 << 62); // Q0.62

    let mut t = xi;
    let mut s = xi;
    for k in 1..9i128 {
        t = rne_div(t * x2, 1 << 62);
        t = rne_div(t, (2 * k) * (2 * k + 1));
        s += if k & 1 == 1 { -t } else { t };
    }

    let mut t: i128 = 1 << 62;
    let mut c: i128 = 1 << 62;
    for k in 1..9i128 {
        t = rne_div(t * x2, 1 << 62);
        t = rne_div(t, (2 * k - 1) * (2 * k));
        c += if k & 1 == 1 { -t } else { t };
    }

    ((s * sign_s) as i64, (c * sign_c) as i64)
}

/// ROPE-I tables (spec §5.9): Q1.30 i32 sin/cos per (position, frequency),
/// generated at load by the normative integer-only procedure, reproducible
/// from `(max_seq, head_dim, base_bits)` alone — its FNV digest is
/// golden-tested (`0xD8345EBF01E990FA` for the M7 shape). Identical to
/// `cis_attn.rs:292-334`.
pub struct RopeTableI {
    pub cos: Vec<i32>,
    pub sin: Vec<i32>,
    pub max_seq: usize,
    pub half_dim: usize,
}

impl RopeTableI {
    pub fn new(max_seq: usize, head_dim: usize, base_bits: u32) -> RopeTableI {
        assert!(
            head_dim >= 2 && head_dim.is_multiple_of(2),
            "RoPE needs an even head_dim"
        );
        let half = head_dim / 2;
        let chain = exp2_chain();
        let l = log2_q32_f32(base_bits);
        let mut inv_freq: Vec<u128> = Vec::with_capacity(half); // Q0.62
        for d in 0..half {
            let a = rne_div(2 * d as i128 * l as i128, head_dim as i128) as u64;
            let n = a >> 32;
            let f = a & 0xFFFF_FFFF;
            assert!(n < 62, "RoPE-I: inverse frequency underflows Q0.62");
            inv_freq.push(rne_div(exp2_neg_frac(f, &chain) as i128, 1i128 << n) as u128);
        }
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        for pos in 0..max_seq {
            for &ivf in &inv_freq {
                let theta = pos as u128 * ivf; // <= max_seq . 2^62, exact
                let r = theta % TWO_PI_Q62;
                let (s, c) = sincos_q62(r);
                cos.push(rne_div(c as i128, 1 << 32).clamp(-(1 << 30), 1 << 30) as i32);
                sin.push(rne_div(s as i128, 1 << 32).clamp(-(1 << 30), 1 << 30) as i32);
            }
        }
        RopeTableI {
            cos,
            sin,
            max_seq,
            half_dim: half,
        }
    }
}

/// ROPE-I rotation over fixed-point q/k (the engine uses Q.16). `(d,
/// d+half)` pairing, `rne((q0·cos - q1·sin)/2^30)` in i64/i128
/// intermediates (spec §5.9). Identical to `cis_attn.rs:341-372`.
pub fn rope_apply_i(
    q: &mut [i32],
    k: &mut [i32],
    seq_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    tab: &RopeTableI,
) {
    if seq_pos >= tab.max_seq {
        return;
    }
    let half = head_dim / 2;
    debug_assert_eq!(half, tab.half_dim);
    let off = seq_pos * half;
    let rotate = |v: &mut [i32]| {
        for d in 0..half {
            let c = tab.cos[off + d] as i128;
            let s = tab.sin[off + d] as i128;
            let v0 = v[d] as i128;
            let v1 = v[d + half] as i128;
            v[d] = rne_div(v0 * c - v1 * s, 1 << 30) as i32;
            v[d + half] = rne_div(v0 * s + v1 * c, 1 << 30) as i32;
        }
    };
    for h in 0..num_heads {
        rotate(&mut q[h * head_dim..(h + 1) * head_dim]);
    }
    for h in 0..num_kv_heads {
        rotate(&mut k[h * head_dim..(h + 1) * head_dim]);
    }
}

/// `1/sqrt(n)` in Q0.30 by the parameter-free rule
/// `rne(2^60 / isqrt(n·2^60))` (spec §5.12, attention-score scale). Identical
/// to `cis_attn.rs:377-381`.
pub fn inv_sqrt_q30(n: u64) -> i64 {
    assert!(n > 0, "inv_sqrt_q30: n must be positive");
    let s = ((n as u128) << 60).isqrt(); // floor(sqrt(n) . 2^30)
    rne_div(1i128 << 60, s as i128) as i64
}

// ---------------------------------------------------------------------------
// §5.10 ACT-I: integer MLP elementwise stage — cis_attn.rs:384-432
// ---------------------------------------------------------------------------

/// relu²·up on the Q.20 grid (spec §5.10): exact `rne(max(0,g)²/2^20)·u`
/// with one more RNE requant to Q.20. Identical to `cis_attn.rs:391-402`.
pub fn relu2_q20(g: i64, u: i64) -> i64 {
    if g <= 0 {
        return 0;
    }
    let a = rne_div(g as i128 * g as i128, 1 << 20); // Q.20
    let r = rne_div(a * u as i128, 1 << 20);
    assert!(
        r >= i64::MIN as i128 && r <= i64::MAX as i128,
        "relu2_q20: result exceeds i64"
    );
    r as i64
}

/// silu(g)·up on the Q.20 grid (spec §5.10): `σ(g)` in Q0.31 from one
/// `exp_neg(|g| << 12)` evaluation, both sign branches evaluating a
/// non-negative argument so the seam at g=0 is continuous. Identical to
/// `cis_attn.rs:413-432`.
pub fn silu_q20(g: i64, u: i64, lut: &ExpLut) -> i64 {
    debug_assert!(
        g.unsigned_abs() < 1 << 40 && u.unsigned_abs() < 1 << 40,
        "silu_q20: inputs exceed the Q.20 headroom contract"
    );
    let sig_q31: i128 = if g >= 0 {
        let t = exp_neg_q31((g as u128) << 12, lut); // Q.20 -> Q.32 arg
        rne_div(1i128 << 62, (1i128 << 31) + t as i128)
    } else {
        let t = exp_neg_q31((g.unsigned_abs() as u128) << 12, lut);
        rne_div((t as i128) << 31, (1i128 << 31) + t as i128)
    };
    let s = rne_div(g as i128 * sig_q31, 1 << 31); // Q.20
    let r = rne_div(s * u as i128, 1 << 20);
    assert!(
        r >= i64::MIN as i128 && r <= i64::MAX as i128,
        "silu_q20: result exceeds i64"
    );
    r as i64
}

// ---------------------------------------------------------------------------
// Unit goldens + digests. Constants transcribed verbatim from
// `cis_attn.rs:439-707`'s own test module — computed by the INDEPENDENT
// big-int generator `scripts/cis_e2_golden_gen.py` per that file's doc
// comment, never by this crate's code under test. The two FNV digests here
// (`golden_exp_lut`, `golden_rope_table_m7`) are the pinned constants this
// task's instructions call out: `0x66C2A0EEB8C2DC43` (exp LUT, spec §5.7)
// and `0xD8345EBF01E990FA` (M7-shape RoPE table, spec §5.9) — computed the
// same way the reference does (`crate::fnv::{FNV1A64_OFFSET, fnv1a64}`,
// this crate's own phase-1 FNV-1a 64, identical algorithm to
// `cis_infer.rs`'s copy both digests were originally pinned against).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fnv::{FNV1A64_OFFSET, fnv1a64};
    use alloc::vec;

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    #[test]
    fn golden_exp2_chain() {
        let c = exp2_chain();
        assert_eq!(c[1], 3260954456333195553);
        assert_eq!(c[2], 3877950241171266237);
        assert_eq!(c[8], 4599216278082138664);
        assert_eq!(c[16], 4611637242788701949);
        assert_eq!(c[32], 4611686017683126785);
        let mut d = FNV1A64_OFFSET;
        for v in &c[1..=32] {
            d = fnv1a64(d, &v.to_le_bytes());
        }
        assert_eq!(d, 0x6CE6217F7A809D74);
    }

    /// The exp-LUT digest this task's acceptance criteria call out by name:
    /// pinned normative constant `0x66C2A0EEB8C2DC43` (spec §5.7), computed
    /// exactly as `cis_attn.rs:474-478` does — FNV-1a 64 over the 1025
    /// entries' LE u32 bytes.
    #[test]
    fn golden_exp_lut() {
        let lut = ExpLut::new();
        assert_eq!(lut.entry(0), 1 << 31);
        assert_eq!(lut.entry(1), 2146030505);
        assert_eq!(lut.entry(512), 1518500250);
        assert_eq!(lut.entry(1023), 1074468888);
        assert_eq!(lut.entry(1024), 1 << 30);
        let mut d = FNV1A64_OFFSET;
        for i in 0..=1024 {
            d = fnv1a64(d, &(lut.entry(i) as u32).to_le_bytes());
        }
        assert_eq!(d, 0x66C2A0EEB8C2DC43);
        for i in 0..1024 {
            assert!(lut.entry(i) > lut.entry(i + 1), "LUT not decreasing at {i}");
        }
    }

    #[test]
    fn golden_log2_q32() {
        assert_eq!(log2_q32_f32(2.0f32.to_bits()), 1 << 32);
        assert_eq!(log2_q32_f32(4.0f32.to_bits()), 2 << 32);
        assert_eq!(log2_q32_f32(10000.0f32.to_bits()), 57070290108);
        assert_eq!(log2_q32_f32(500000.0f32.to_bits()), 81310467867);
    }

    #[test]
    fn golden_exp_neg() {
        let lut = ExpLut::new();
        assert_eq!(exp_neg_q31(0, &lut), 1 << 31);
        assert_eq!(exp_neg_q31(1 << 32, &lut), 790015124);
        assert_eq!(exp_neg_q31(5 << 32, &lut), 14469632);
        assert_eq!(exp_neg_q31(2977044472, &lut), 1073741824);
        assert_eq!(exp_neg_q31(21 << 32, &lut), 2);
        assert_eq!(exp_neg_q31(22 << 32, &lut), 0);
        assert_eq!(exp_neg_q31(23 << 32, &lut), 0);
        assert_eq!(exp_neg_q31(1u128 << 80, &lut), 0);
    }

    #[test]
    fn exp_neg_monotone() {
        let lut = ExpLut::new();
        let mut prev = i64::MAX;
        let mut z: u128 = 0;
        let mut state = 0xA1CE_5EED_0000_0009u64;
        for _ in 0..50_000 {
            z += (lcg_next(&mut state) % (1 << 24)) as u128;
            let e = exp_neg_q31(z, &lut);
            assert!(e <= prev, "exp_neg not monotone at z = {z}");
            prev = e;
        }
    }

    #[test]
    fn golden_softmax_8() {
        const GOLDEN_P: [i32; 8] = [8860, 1178, 3948, 6664, 2926, 4387, 3224, 1579];
        let lut = ExpLut::new();
        let mut state = 0xA1CE_5EED_0000_0008u64;
        let mut s = [0i64; 8];
        for x in s.iter_mut() {
            *x = ((lcg_next(&mut state) >> 30) % (1 << 26)) as i64 - (1 << 25);
        }
        let mut p = [0i32; 8];
        softmax_i(&mut s, &mut p, &lut);
        assert_eq!(p, GOLDEN_P);
        let sum: i64 = p.iter().map(|&x| x as i64).sum();
        assert!((sum - (1 << 15)).abs() <= 4, "sum {sum} outside T/2 bound");
    }

    #[test]
    fn softmax_uniform_and_sum_bound() {
        let lut = ExpLut::new();
        let mut s = [7i64 << 24; 512];
        let mut p = [0i32; 512];
        softmax_i(&mut s, &mut p, &lut);
        assert!(p.iter().all(|&x| x == 64));
        assert_eq!(p.iter().map(|&x| x as i64).sum::<i64>(), 1 << 15);

        let mut state = 0xA1CE_5EED_0000_000Au64;
        for &n in &[2usize, 3, 17, 128, 512] {
            let mut s = vec![0i64; n];
            let mut p = vec![0i32; n];
            for x in s.iter_mut() {
                *x = ((lcg_next(&mut state) >> 28) % (1 << 28)) as i64 - (1 << 27);
            }
            let sref = s.clone();
            softmax_i(&mut s, &mut p, &lut);
            let sum: i64 = p.iter().map(|&x| x as i64).sum();
            assert!(
                (sum - (1 << 15)).unsigned_abs() <= n.div_ceil(2) as u64,
                "n={n}: sum {sum} outside the declared bound"
            );
            for i in 0..n {
                for j in 0..n {
                    if sref[i] >= sref[j] {
                        assert!(p[i] >= p[j], "monotonicity broken at ({i},{j})");
                    }
                }
            }
        }
    }

    #[test]
    fn golden_sincos() {
        assert_eq!(sincos_q62(0), (0, 1 << 62));
        assert_eq!(sincos_q62(PI_2_Q62), (4611686018427588579, -2425842));
        assert_eq!(sincos_q62(PI_Q62), (0, -4611686018427387904));
        assert_eq!(sincos_q62(TWO_PI_Q62 - 1), (-1, 4611686018427387904));
        assert_eq!(
            sincos_q62(1 << 62),
            (3880599975550901295, 2491704589696178674)
        );
        assert_eq!(
            sincos_q62(5 << 61),
            (2759965619382477058, -3694622810570160725)
        );
    }

    /// The RoPE-table digest this task's acceptance criteria call out by
    /// name: pinned normative constant `0xD8345EBF01E990FA` for the M7
    /// shape (head_dim 64, seq 512, base 10000.0; spec §5.9), computed
    /// exactly as `cis_attn.rs:618-624` does — FNV-1a 64 over (cos, sin)
    /// LE i32 pairs per (pos, d).
    #[test]
    fn golden_rope_table_m7() {
        let t = RopeTableI::new(512, 64, 10000.0f32.to_bits());
        for d in 0..32 {
            assert_eq!((t.cos[d], t.sin[d]), (1 << 30, 0), "pos 0 d {d}");
        }
        assert_eq!((t.cos[32], t.sin[32]), (580145183, 903522590));
        assert_eq!((t.cos[63], t.sin[63]), (1073741814, 143186));
        assert_eq!(
            (t.cos[100 * 32 + 7], t.sin[100 * 32 + 7]),
            (771714489, 746577693)
        );
        assert_eq!(
            (t.cos[511 * 32 + 31], t.sin[511 * 32 + 31]),
            (1071249849, 73111318)
        );
        let mut d = FNV1A64_OFFSET;
        for i in 0..512 * 32 {
            d = fnv1a64(d, &t.cos[i].to_le_bytes());
            d = fnv1a64(d, &t.sin[i].to_le_bytes());
        }
        assert_eq!(d, 0xD8345EBF01E990FA);
    }

    #[test]
    fn golden_rope_apply() {
        let t = RopeTableI::new(4, 4, 10000.0f32.to_bits());
        let mut q = [1000000i32, -2000000, 3000000, 4000000];
        let mut k = [70000i32, 80000, -90000, 100000];
        rope_apply_i(&mut q, &mut k, 1, 1, 1, 4, &t);
        assert_eq!(q, [-1984111, -2039899, 2462378, 3979800]);
        assert_eq!(k, [113554, 78996, 10276, 100795]);
    }

    #[test]
    fn rope_norm_preservation() {
        let t = RopeTableI::new(512, 64, 10000.0f32.to_bits());
        let mut state = 0xA1CE_5EED_0000_000Bu64;
        for &pos in &[1usize, 17, 255, 511] {
            let mut q = [0i32; 64];
            let mut k = [0i32; 64];
            for x in q.iter_mut().chain(k.iter_mut()) {
                *x = ((lcg_next(&mut state) >> 33) % (1 << 21)) as i32 - (1 << 20);
            }
            let s2_before: i128 = q.iter().map(|&v| v as i128 * v as i128).sum();
            let mut kk = [0i32; 64];
            rope_apply_i(&mut q, &mut kk, pos, 1, 0, 64, &t);
            let s2_after: i128 = q.iter().map(|&v| v as i128 * v as i128).sum();
            let diff = (s2_after - s2_before).unsigned_abs();
            assert!(
                diff <= (s2_before as u128 >> 13).max(1 << 12),
                "pos {pos}: norm drifted by {diff} of {s2_before}"
            );
        }
    }

    #[test]
    fn golden_inv_sqrt() {
        assert_eq!(inv_sqrt_q30(1), 1 << 30);
        assert_eq!(inv_sqrt_q30(4), 1 << 29);
        assert_eq!(inv_sqrt_q30(64), 1 << 27);
        assert_eq!(inv_sqrt_q30(128), 94906266);
    }

    #[test]
    fn golden_relu2() {
        assert_eq!(relu2_q20(-5 << 20, 3 << 20), 0);
        assert_eq!(relu2_q20(0, 3 << 20), 0);
        assert_eq!(relu2_q20(1 << 20, 1 << 20), 1 << 20);
        assert_eq!(relu2_q20(3 << 19, 1 << 21), 4718592);
        assert_eq!(relu2_q20(7, 1 << 20), 0);
        assert_eq!(relu2_q20(724, 1 << 20), 0);
    }

    #[test]
    fn golden_silu() {
        let lut = ExpLut::new();
        assert_eq!(silu_q20(0, 5 << 20, &lut), 0);
        assert_eq!(silu_q20(1 << 20, 1 << 20, &lut), 766570);
        assert_eq!(silu_q20(-(1i64 << 20), 1 << 20, &lut), -282006);
        assert_eq!(silu_q20(10 << 20, 1 << 20, &lut), 10485284);
        assert_eq!(silu_q20(-(10i64 << 20), 1 << 20, &lut), -476);
        assert_eq!(silu_q20(-(30i64 << 20), 1 << 20, &lut), 0);
        assert_eq!(silu_q20(3 << 19, -(1i64 << 20), &lut), -1285933);
    }

    #[test]
    fn silu_sign_seam() {
        let lut = ExpLut::new();
        let a = silu_q20(-1, 1 << 20, &lut);
        let b = silu_q20(0, 1 << 20, &lut);
        let c = silu_q20(1, 1 << 20, &lut);
        assert_eq!((a, b, c), (0, 0, 1));
        assert!(a <= b && b <= c, "silu seam not monotone: {a} {b} {c}");
    }
}
