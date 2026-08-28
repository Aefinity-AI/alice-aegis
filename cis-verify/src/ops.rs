//! CIS-1 §5.1–§5.6, §5.11 — TMV, QUANT-ACT, REQUANT/scale, RMSNORM-I,
//! NORMQ, container-boundary conversions, ARGMAX.
//!
//! Transcribed per `docs/design/CIS_VERIFY_DESIGN.md` builder task 3
//! (§6.2 item 3), source of truth `docs/CIS-1_SPEC_v1.0.md` (v1.0.2) §5.1–
//! §5.6 and §5.11. Where the spec prose is ambiguous the reference code is
//! the tie-breaker (spec §5: "The reference is definitive; optimized
//! kernels conform by matching its bits, not by copying its loops.") —
//! every function below cites the reference file:line it transcribes.
//!
//! Two source files:
//! - `aegis-core/src/cis.rs` — `rne_div`, `QScale`, `requant_i32`/
//!   `requant_i64`, `wcode`, `ternary_matvec_i8`, `quantize_activations_i32`,
//!   `rmsnorm_i`, `argmax_i32`.
//! - `aegis-core/src/cis_infer.rs` — the container-boundary conversions
//!   (`bf16_to_fixed`, `f32_to_fixed`, `f32_to_ratio`, `fix_f32_vec`),
//!   `QScale64`, `ActScale`, `normq` (§5.5 NORMQ), `quantq`, `argmax_i64`.
//!
//! `fix_q_vec` (the ACT-I block-exponent conversion) is a **pending
//! erratum**: it does not exist yet in this branch's `aegis-core` (checked —
//! `grep -n fix_q_vec aegis-core/src/cis_infer.rs` on this worktree returns
//! nothing). It is transcribed instead from
//! `../alice-aegis-cm-e1c/aegis-core/src/cis_infer.rs:421-445` (branch
//! `cm/e1c`, not merged as of this crate's authoring) per this task's
//! instruction to mirror it. At M7-scale magnitudes (every ACT-I product
//! `< 2^49`) it degenerates to the identity — `shift = 0`, `G = F`, every
//! element numerically untouched — so it changes nothing for the M7
//! conformance tiers this crate targets; it exists for BitNet-2B-scale
//! headroom, ahead of that scale actually landing on this branch.
//!
//! `core`-only integer arithmetic — no floats, no `libm`, no `unsafe`.

// ---------------------------------------------------------------------------
// §2 (normative rounding primitive) — cis.rs:22-43
// ---------------------------------------------------------------------------

/// Round-to-nearest-even division: `rne(num / den)` for `den > 0`, exact.
/// Identical to `aegis-core/src/cis.rs:32-43`. The single rounding
/// primitive every op below goes through.
pub fn rne_div(num: i128, den: i128) -> i128 {
    debug_assert!(den > 0, "rne_div: denominator must be positive");
    let q = num.div_euclid(den);
    let r = num.rem_euclid(den); // 0 <= r < den, exact
    if 2 * r > den || (2 * r == den && q & 1 != 0) {
        q + 1
    } else {
        q
    }
}

// ---------------------------------------------------------------------------
// §5.3 REQUANT and scale application — cis.rs:45-137
// ---------------------------------------------------------------------------

/// Fixed-point requantization multiplier `(M, S)`: value = M / 2^(31+S),
/// `M ∈ [2^30, 2^31)`, `S ≤ 62` (spec §5.3). Identical to
/// `aegis-core/src/cis.rs:49-117`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QScale {
    m: i32,
    s: u8,
}

impl QScale {
    /// Construct a validated multiplier. Panics on an out-of-range `M`
    /// rather than degrading — a non-normalized multiplier silently changes
    /// the produced bits. Identical to `cis.rs:65-69`.
    pub const fn new(m: i32, s: u8) -> Self {
        assert!(m >= 1 << 30, "CIS-1 spec §5.3: M must be in [2^30, 2^31)");
        assert!(s <= 62, "CIS-1 spec §5.3: S bounded to 62");
        QScale { m, s }
    }

    pub const fn m(self) -> i32 {
        self.m
    }

    pub const fn s(self) -> u8 {
        self.s
    }

    /// Offline generator (spec §5.3, "Offline (M,S) generation — normative
    /// procedure"): the RNE-nearest `(M, S)` for `num/den` in the open
    /// domain (0, 1). Identical to `cis.rs:90-116`.
    pub fn from_ratio(num: u64, den: u64) -> Option<QScale> {
        if num == 0 || den == 0 || num >= den {
            return None;
        }
        // Smallest e with floor(num·2^e / den) >= 2^30; M = rne at that e is
        // then in [2^30, 2^31]. num·2^e stays under 2^31·den < 2^95: exact
        // in u128.
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
                // representable multiplier (spec §5.3 rejection surface).
                return Some(QScale::new(i32::MAX, 0));
            }
            (1i128 << 30, e - 1)
        } else {
            (m, e)
        };
        Some(QScale::new(m as i32, (e - 31) as u8))
    }
}

/// REQUANT (spec §5.3): `y = clamp(rne((acc·M) / 2^(31+S)), -127, 127)`.
/// Identical to `cis.rs:123-125`.
pub fn requant_i32(acc: i32, q: QScale) -> i8 {
    requant_i64(acc as i64, q)
}

/// REQUANT over an `i64` accumulator (RMSNORM-I's weighted products exceed
/// i32) — spec §5.3, "The `i64`-input form ... is identical arithmetic at
/// `i128` width — ratified as part of this op". Identical to
/// `cis.rs:133-137`.
pub fn requant_i64(acc: i64, q: QScale) -> i8 {
    let prod = acc as i128 * q.m as i128;
    let den = 1i128 << (31 + q.s as u32);
    rne_div(prod, den).clamp(-127, 127) as i8
}

// ---------------------------------------------------------------------------
// §5.1 TMV — ternary matvec — cis.rs:139-224
// ---------------------------------------------------------------------------

/// One 2-bit weight code (spec §4): `00` = 0, `01` = +1, `10` = -1, and
/// **ratified** — `11` decodes to 0 (defined-as-zero). Identical to
/// `cis.rs:147-153`.
const fn wcode(code: u8) -> i32 {
    match code & 0b11 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

/// The TMV rejection surface (spec §5.1, "the rejection surface, itself
/// normative"): every implementation must reject exactly these five cases,
/// independent of ISA or internal blocking thresholds. Identical to
/// `cis.rs:179-200`.
#[inline]
fn check_tmv_preconditions(
    output: &[i32],
    input: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
) {
    assert!(
        dim_in.is_multiple_of(4),
        "CIS-1 spec §5.1: dim_in must be a multiple of 4"
    );
    assert!(
        dim_in <= (i32::MAX / 127) as usize,
        "CIS-1 spec §5.1: dim_in exceeds the exactness ceiling"
    );
    assert!(
        output.len() >= dim_out,
        "CIS-1 spec §5.1: output shorter than dim_out"
    );
    assert!(
        input.len() >= dim_in,
        "CIS-1 spec §5.1: input shorter than dim_in"
    );
    assert!(
        weights_packed.len() >= dim_out * (dim_in / 4),
        "CIS-1 spec §5.1: weights shorter than dim_out rows"
    );
}

/// TMV (spec §5.1): `acc[j] = Σ_i w[j,i]·a[i]` in `i32`, exact in any
/// summation order. Identical to `cis.rs:202-224`; packing is row-major,
/// 4 weights per byte, low bit-pair first (spec §4).
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

// ---------------------------------------------------------------------------
// §5.2 QUANT-ACT — dynamic per-token activation quantization — cis.rs:226-267
// ---------------------------------------------------------------------------

/// QUANT-ACT (spec §5.2): per-token symmetric absmax onto the i8 grid,
/// `q_i = rne(x_i·127 / absmax)`, the scale carried forward as the exact
/// rational `127/absmax`. Returns the absmax denominator (0 for an
/// all-zero tensor, which quantizes to zeros by convention). Identical to
/// `cis.rs:247-267`.
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

// ---------------------------------------------------------------------------
// §5.4 RMSNORM-I — cis.rs:269-318
// ---------------------------------------------------------------------------

/// RMSNORM-I (spec §5.4), the complete four-step normative procedure:
/// `s2 = Σx_i²` (i64, exact) → `t = max(isqrt(s2·n), 1)` (exact floor
/// integer square root) → `inv_rms_q30 = rne(n·2^30 / t)` (Q2.30) →
/// per-element `y = rne(x_i·inv_rms_q30 / 2^15)` then
/// `out_i = REQUANT_i64(y·w_i)`. Every intermediate grid here is
/// bit-determining (spec §5.4: "Folding these roundings any other way ...
/// produces different i8 outputs"). Identical to `cis.rs:298-318`.
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

// ---------------------------------------------------------------------------
// §5.11 ARGMAX — cis.rs:320-338, cis_infer.rs:418-430
// ---------------------------------------------------------------------------

/// ARGMAX over `i32` logits (spec §5.11): ties in exact equality break to
/// the LOWEST index. Identical to `cis.rs:327-338`. This is the form the
/// Tier-2 selftest exercises (`cis.rs`'s own reference-op scope).
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

/// ARGMAX over `i64` logits (spec §5.11): same lowest-index tie rule, at
/// the width the real forward pass's LM-head dot actually produces
/// (spec §5.12, "Logits ... `i64`, exact integer dot"). Identical to
/// `aegis-core/src/cis_infer.rs:459-470`. This selects `token_id` for both
/// a receipt's `token-ids` field and the witness chain fold.
pub fn argmax_i64(logits: &[i64]) -> u32 {
    debug_assert!(!logits.is_empty(), "argmax_i64: empty logits");
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
// §5.6 Container-boundary conversions — cis_infer.rs:76-405
// ---------------------------------------------------------------------------

/// Residual-stream fixed point: Q.20. Identical to
/// `aegis-core/src/cis_infer.rs:47` (`pub const F: u32 = 20`).
pub const F: u32 = 20;
/// Norm-gain fixed point: Q.20. Identical to `cis_infer.rs:49`
/// (`pub const GQ: u32 = 20`).
pub const GQ: u32 = 20;

/// Exact RNE division of a nonnegative i64 by 2^k (1 ≤ k ≤ 62) in pure
/// shift/mask arithmetic, bit-identical to `rne_div(m, 1<<k)`. Identical to
/// `cis_infer.rs:82-92`.
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

/// `bf16_to_fixed` (spec §5.6): BF16 bit pattern → signed fixed-point with
/// `frac` fractional bits, by exact RNE — depends only on the per-element
/// float *value* (its bits), never on float accumulation. Identical to
/// `cis_infer.rs:97-117`.
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

/// `f32_to_fixed` (spec §5.6): f32 bit pattern → signed fixed-point with
/// `frac` fractional bits, RNE, shift bound `sh ≤ 26` (guaranteeing
/// `|v| < 2^50`). Identical to `cis_infer.rs:122-151`.
pub fn f32_to_fixed(bits: u32, frac: u32) -> i64 {
    let neg = (bits >> 31) & 1 == 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = (bits & 0x7F_FFFF) as i64;
    assert!(exp != 0xFF, "f32_to_fixed: inf/nan crossed the boundary");
    let (m, e) = if exp == 0 {
        (man, 1 - 127 - 23)
    } else {
        (man | 0x80_0000, exp - 127 - 23)
    };
    let sh = e + frac as i32;
    assert!(sh <= 26, "f32_to_fixed: value too large for fixed-point");
    let v = if sh >= 0 {
        m << sh
    } else if -sh >= 63 {
        0
    } else {
        rne_div(m as i128, 1i128 << (-sh)) as i64
    };
    if neg { -v } else { v }
}

/// Exact f32 decomposition: value = (-1)^neg · m · 2^e with `m` odd (0 →
/// `m = 0`). No rounding — the identity on the value. Identical to
/// `cis_infer.rs:155-184`.
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

/// `fix_f32_vec` (spec §5.6, block exponent): the largest `G ≤ F` such that
/// every element converts inside the `2^50` headroom bound —
/// `G = min(F, 176 - max_exp)`. `G < 0` (v1.0.1 erratum: panic threshold is
/// 2^50, not the v1.0 draft's "2^54") MUST panic. Identical to
/// `cis_infer.rs:385-405`.
pub fn fix_f32_vec(out: &mut [i64], src: &[f32]) -> u32 {
    debug_assert_eq!(out.len(), src.len());
    let mut max_exp: i32 = 0;
    for &v in src {
        let exp = ((v.to_bits() >> 23) & 0xFF) as i32;
        assert!(exp != 0xFF, "fix_f32_vec: inf/nan at the f32 boundary");
        if exp > max_exp {
            max_exp = exp;
        }
    }
    // sh = exp - 150 + G <= 26  <=>  G <= 176 - max_exp.
    let g = (176 - max_exp).min(F as i32);
    assert!(
        g >= 0,
        "fix_f32_vec: value >= 2^50 — divergence, not headroom"
    );
    for (o, &v) in out.iter_mut().zip(src) {
        *o = f32_to_fixed(v.to_bits(), g as u32);
    }
    g as u32
}

/// `fix_q_vec` — ACT-I's integer-domain block exponent, **pending erratum**
/// (see module doc comment): FullInt's own hybrid-free equivalent of
/// `fix_f32_vec`, deriving `G` from the integer magnitude already in hand
/// (`64 - leading_zeros`) rather than a float exponent, so ACT-I's exact
/// Q.20 output (`relu2_q20`/`silu_q20`, `crate::attn`) can still be
/// rescaled onto the `2^50` normq/quantq residual headroom at scales where
/// it would otherwise overflow. Rescales `v` in place and returns `G`.
///
/// At M7-scale magnitudes (every element's bit length `< 49`) `shift = 0`
/// and `G` degenerates to `F` with every element numerically untouched —
/// bit-identical to unconditionally using `F`. Transcribed from
/// `../alice-aegis-cm-e1c/aegis-core/src/cis_infer.rs:421-445` (branch
/// `cm/e1c`; not present in this branch's `aegis-core` as of this writing —
/// cite this file, not `cis_infer.rs`, if reconciling against a checkout of
/// this branch).
pub fn fix_q_vec(v: &mut [i64]) -> u32 {
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
        "fix_q_vec: ACT-I product exceeds Q0 headroom — model divergence"
    );
    if shift > 0 {
        let d = 1i128 << shift;
        for x in v.iter_mut() {
            *x = rne_div(*x as i128, d) as i64;
        }
    }
    F - shift
}

// ---------------------------------------------------------------------------
// §5.3 QScale64 (runtime, i64/Q.20 domain) — cis_infer.rs:187-276
// ---------------------------------------------------------------------------

/// QScale64 (spec §5.3): a 63-bit fixed-point multiplier `m·2^e`,
/// `m ∈ [2^62, 2^63)`, built from an exact u128/u128 rational by restoring
/// long division with RNE at 63 significant bits. Identical to
/// `cis_infer.rs:194-276`.
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
    /// `num == 0` → the exact zero multiplier. Identical to
    /// `cis_infer.rs:209-253`.
    pub fn from_ratio(num: u128, den: u128) -> QScale64 {
        assert!(den > 0, "QScale64: zero denominator");
        if num == 0 {
            return QScale64 { m: 0, e: 0 };
        }
        let nb = 128 - num.leading_zeros() as i32;
        let db = 128 - den.leading_zeros() as i32;
        let diff = nb - db;
        let (mut rem, d) = if diff >= 0 {
            (num, den << diff)
        } else {
            (num << -diff, den)
        };
        let mut exp_adj = 0i32;
        if rem < d {
            rem <<= 1;
            exp_adj = -1;
        }
        let mut q: u128 = 0;
        for _ in 0..64 {
            q <<= 1;
            if rem >= d {
                rem -= d;
                q |= 1;
            }
            rem <<= 1;
        }
        let sticky = rem != 0;
        let round = q & 1 == 1;
        let mut m = (q >> 1) as u64;
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

    /// `rne(x · m · 2^e)`, exact in i128. Identical to `cis_infer.rs:256-275`.
    pub fn rescale(self, x: i64) -> i64 {
        let p = x as i128 * self.m as i128;
        let v = if self.e >= 0 {
            let s = p << self.e;
            assert!(
                (p == 0) || (s >> self.e == p),
                "QScale64::rescale: left-shift overflow"
            );
            s
        } else if -self.e >= 127 {
            0
        } else {
            rne_div(p, 1i128 << (-self.e))
        };
        assert!(
            v >= i64::MIN as i128 && v <= i64::MAX as i128,
            "QScale64::rescale: result exceeds i64"
        );
        v as i64
    }
}

// ---------------------------------------------------------------------------
// §5.5 NORMQ — fused norm + quantization (engine form) — cis_infer.rs:282-370
// ---------------------------------------------------------------------------

/// Exact rational per-code scale of a quantized activation vector: real
/// value of code `a` is `a · num / den`. Identical to `cis_infer.rs:284-288`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActScale {
    pub num: u128,
    pub den: u128,
}

/// NORMQ (spec §5.5): RMSNorm (gain `g`, Q.`GQ`) fused with per-token absmax
/// i8 quantization. The RMS is a common positive factor and divides out of
/// the codes entirely — `codes_i = rne(u_i·127 / max|u|)` with
/// `u_i = h_i·g_i` — surviving only in the carried exact rational scale via
/// `t = isqrt(s2·n)`: `scale = max|u|·n / (127·t·2^GQ)` (spec §5.5).
/// Identical to `cis_infer.rs:306-339`.
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
    for ((c, &hi), &gi) in codes.iter_mut().zip(h).zip(g) {
        let u = hi as i128 * gi as i128;
        *c = rne_div(u * 127, a).clamp(-127, 127) as i8;
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
/// preceding norm — `quantq`'s scale is `A / (127·2^frac)`. Identical to
/// `cis_infer.rs:346-370`.
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
    for (c, &hi) in codes.iter_mut().zip(h) {
        *c = rne_div(hi as i128 * 127, a as i128).clamp(-127, 127) as i8;
    }
    ActScale {
        num: a as u128,
        den: 127u128 << frac,
    }
}

// ---------------------------------------------------------------------------
// Unit goldens. Constants transcribed verbatim from the reference test
// modules (`cis.rs:349-628`, `cis_infer.rs`'s own QScale64/fix_f32_vec
// tests) — computed by an INDEPENDENT bignum implementation per those
// files' own doc comments, not by this crate's code under test.
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
        let mut w = [0u8; 64];
        gen_packed_no11(&mut state, 64, &mut w);
        let mut a = [0i8; 32];
        gen_acts_i8(&mut state, &mut a);

        let mut out = [0i32; 8];
        ternary_matvec_i8(&mut out, &a, &w, 8, 32);
        assert_eq!(out, GOLDEN);
    }

    #[test]
    fn tmv_undefined_code_11_decodes_to_zero() {
        let w = [0xFFu8; 8];
        let a = [127i8, -127, 5, -5, 99, -99, 1, -1];
        let mut out = [123i32; 4];
        ternary_matvec_i8(&mut out, &a, &w, 4, 8);
        assert_eq!(out, [0; 4]);

        let w = [0b1101_1000u8];
        let a = [10i8, 20, 30, 40];
        let mut out = [0i32; 1];
        ternary_matvec_i8(&mut out, &a, &w, 1, 4);
        assert_eq!(out[0], -20 + 30);
    }

    #[test]
    fn golden_requant_table() {
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
        let v = [254i32, 1, -1, 3, -3];
        let mut q = [0i8; 5];
        assert_eq!(quantize_activations_i32(&mut q, &v), 254);
        assert_eq!(q, [127, 0, 0, 2, -2]);

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
            *g = (((lcg_next(&mut state) >> 40) % 8193) + 8192) as i16;
        }
        let mut out = [0i8; 16];
        rmsnorm_i(&mut out, &x, &w, QScale::new(1 << 30, 24));
        assert_eq!(out, GOLDEN);

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

        assert_eq!(argmax_i64(&[5, 9, 9, 1]), 1);
        assert_eq!(argmax_i64(&[3, 3, 3]), 0);
        assert_eq!(argmax_i64(&[-7]), 0);
        assert_eq!(argmax_i64(&[i64::MIN, i64::MIN]), 0);
        assert_eq!(argmax_i64(&[0, -1, 2, 2, -5, 2]), 2);
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
        assert_eq!(QScale::from_ratio(u64::MAX, u64::MAX), None);
        assert_eq!(
            QScale::from_ratio(u64::MAX - 1, u64::MAX),
            Some(QScale::new(i32::MAX, 0))
        );
        assert_eq!(QScale::from_ratio(0, 5), None);
        assert_eq!(QScale::from_ratio(5, 0), None);
        assert_eq!(QScale::from_ratio(5, 4), None);
        assert_eq!(QScale::from_ratio(1, u64::MAX), None);
    }

    /// The axiom, tested (spec §1): chunked and reversed accumulation must
    /// equal the canonical ascending loop EXACTLY.
    #[test]
    fn accumulation_order_invariance() {
        const DIM_OUT: usize = 16;
        const DIM_IN: usize = 256;
        const PACKED: usize = DIM_IN / 4;

        let mut state = SEED_PROP;
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

            let mut rev: i32 = 0;
            for col in (0..DIM_IN).rev() {
                let code = w_row[col / 4] >> (2 * (col % 4));
                rev += wcode(code) * a[col] as i32;
            }
            assert_eq!(rev, want[row], "reversed order diverged at row {row}");

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

    #[test]
    #[should_panic(expected = "M must be in")]
    fn qscale_rejects_unnormalized_multiplier() {
        let _ = QScale::new((1 << 30) - 1, 0);
    }

    // -- container-boundary conversions ------------------------------------

    #[test]
    fn bf16_to_fixed_exact_values() {
        // 1.0 = 0x3F80, 0.5 = 0x3F00, -2.0 = 0xC000. Frac = F (Q.20).
        assert_eq!(bf16_to_fixed(0x3F80, F), 1 << F);
        assert_eq!(bf16_to_fixed(0x3F00, F), 1 << (F - 1));
        assert_eq!(bf16_to_fixed(0xC000, F), -(1 << (F + 1)));
        assert_eq!(bf16_to_fixed(0x0000, F), 0);
    }

    #[test]
    fn f32_to_fixed_exact_values() {
        assert_eq!(f32_to_fixed(1.0f32.to_bits(), F), 1 << F);
        assert_eq!(f32_to_fixed((-1.0f32).to_bits(), F), -(1 << F));
        assert_eq!(f32_to_fixed(0.0f32.to_bits(), F), 0);
    }

    #[test]
    fn f32_to_ratio_decomposition() {
        // 1.0 = 1 * 2^0 (mantissa forced odd by the trailing-zero strip).
        let r = f32_to_ratio(1.0f32.to_bits());
        assert_eq!((r.neg, r.m, r.e), (false, 1, 0));
        // -6.0 = -3 * 2^1.
        let r = f32_to_ratio((-6.0f32).to_bits());
        assert_eq!((r.neg, r.m, r.e), (true, 3, 1));
        // 0.0 -> (false, 0, 0) by convention.
        let r = f32_to_ratio(0.0f32.to_bits());
        assert_eq!((r.neg, r.m, r.e), (false, 0, 0));
    }

    #[test]
    fn fix_f32_vec_m7_scale_is_identity_at_g_eq_f() {
        let src = [1.0f32, -2.5, 0.0, 100.25];
        let mut out = [0i64; 4];
        let g = fix_f32_vec(&mut out, &src);
        assert_eq!(g, F); // small values: no coarsening needed
        for (o, &v) in out.iter().zip(src.iter()) {
            assert_eq!(*o, f32_to_fixed(v.to_bits(), F));
        }
    }

    #[test]
    #[should_panic(expected = "value >= 2^50")]
    fn fix_f32_vec_panics_past_headroom() {
        // 2^50 exactly: exp field 177 (176 in the spec's zero-based
        // reasoning), G would be negative.
        let huge = [2f32.powi(50)];
        let mut out = [0i64; 1];
        let _ = fix_f32_vec(&mut out, &huge);
    }

    /// `fix_q_vec` at M7-scale magnitudes (spec's stated degeneration to
    /// identity, pending-erratum module doc comment): every element well
    /// under `2^49` in magnitude, so `shift = 0`, `G = F`, output bytes
    /// numerically untouched.
    #[test]
    fn fix_q_vec_m7_scale_is_identity() {
        let mut v = [1i64 << 20, -(1i64 << 30), 0, 12345, -999];
        let orig = v;
        let g = fix_q_vec(&mut v);
        assert_eq!(g, F);
        assert_eq!(v, orig);
    }

    #[test]
    fn fix_q_vec_all_zero_returns_f() {
        let mut v = [0i64; 4];
        let g = fix_q_vec(&mut v);
        assert_eq!(g, F);
        assert_eq!(v, [0i64; 4]);
    }

    /// Above the `2^49` bit-length threshold, `fix_q_vec` rescales so the
    /// new max magnitude fits under `2^49`, and reports the coarser `G`.
    #[test]
    fn fix_q_vec_rescales_past_2_49() {
        let mut v = [1i64 << 55, -(1i64 << 54)];
        let g = fix_q_vec(&mut v);
        assert!(g < F, "expected coarsened G, got {g}");
        for &x in v.iter() {
            assert!(x.unsigned_abs() < 1 << 49, "rescaled element still >= 2^49");
        }
    }

    #[test]
    fn qscale64_from_ratio_and_rescale() {
        // 1/3, rescaled against a representative accumulator.
        let q = QScale64::from_ratio(1, 3);
        // m should be normalized into [2^62, 2^63).
        assert!((1u64 << 62..1u64 << 63).contains(&q.m()));
        // rne(1000 * 1/3) = 333.
        assert_eq!(q.rescale(1000), 333);
        // Zero numerator -> exact zero multiplier, rescale is always 0.
        let z = QScale64::from_ratio(0, 5);
        assert_eq!((z.m(), z.e()), (0, 0));
        assert_eq!(z.rescale(123456), 0);
        // Exact power-of-two ratio: 1/2.
        let half = QScale64::from_ratio(1, 2);
        assert_eq!(half.rescale(4), 2);
        assert_eq!(half.rescale(7), 4); // rne(3.5) -> 4 (ties to even)
    }

    #[test]
    fn normq_and_quantq_basic() {
        // normq: h all-zero -> codes all-zero, scale {0,1}.
        let h = [0i64; 8];
        let g = [1i32 << GQ; 8];
        let mut codes = [1i8; 8];
        let scale = normq(&mut codes, &h, &g);
        assert_eq!(codes, [0; 8]);
        assert_eq!((scale.num, scale.den), (0, 1));

        // normq: nonzero, uniform gain -> codes are the sign pattern of h
        // scaled by absmax (since g is uniform, u_i = h_i * g, absmax
        // divides out to the same ratio as quantizing h directly). The
        // -50/100 -> -63.5 tie resolves to -64 (RNE: -63 is odd, -64 is
        // even), verified against an independent Python rne_div/isqrt
        // reimplementation of this exact case.
        let h = [100i64, -50, 0, 25, -100];
        let g = [1i32 << GQ; 5];
        let mut codes = [0i8; 5];
        let scale = normq(&mut codes, &h, &g);
        assert_eq!(codes, [127, -64, 0, 32, -127]);
        assert_eq!(scale.num, 524_288_000);
        assert_eq!(scale.den, 45_277_511_680);

        // quantq: absmax quantization at a given fractional width. Same
        // sign/tie pattern as normq above (uniform gain divides out).
        let h = [100i64, -50, 0, 25, -100];
        let mut codes = [0i8; 5];
        let scale = quantq(&mut codes, &h, F);
        assert_eq!(codes, [127, -64, 0, 32, -127]);
        assert_eq!(scale.num, 100);
        assert_eq!(scale.den, 127u128 << F);

        // quantq: all-zero.
        let h = [0i64; 3];
        let mut codes = [9i8; 3];
        let scale = quantq(&mut codes, &h, F);
        assert_eq!(codes, [0; 3]);
        assert_eq!((scale.num, scale.den), (0, 1));
    }
}
