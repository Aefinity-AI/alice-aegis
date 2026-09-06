//! Exactness of the division-free `normq`/`quantq` inner loops against the
//! reference `rne_div`-based formula they replace (see cis_infer.rs for the
//! reciprocal + remainder-correction derivation). Every quotient produced by
//! `normq`/`quantq` in the shipped code equals the corresponding
//! `rne_div(num, den)`, over both random and adversarial inputs, so the
//! codes and `ActScale` are bit-identical to the old u128-division path.

use aegis_core::cis::rne_div;
use aegis_core::cis_infer::{normq, quantq, ActScale, GQ};

// ---------------------------------------------------------------------------
// Tiny deterministic xorshift64* PRNG (no new crate dependency).
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform i128 in `[-bound, bound]` (bound >= 0).
    fn i128_in(&mut self, bound: i128) -> i128 {
        if bound == 0 {
            return 0;
        }
        let hi = self.next_u64() as u128;
        let lo = self.next_u64() as u128;
        let raw = ((hi << 64) | lo) % (2 * bound as u128 + 1);
        raw as i128 - bound
    }
    /// Uniform u128 magnitude with a roughly log-uniform bit length in
    /// `[1, max_bits]`.
    fn magnitude(&mut self, max_bits: u32) -> i128 {
        let bits = 1 + (self.next_u64() as u32 % max_bits);
        let span: u128 = if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        let base: u128 = if bits >= 1 { 1u128 << (bits - 1) } else { 0 };
        let extra = if span > base {
            (self.next_u64() as u128) % (span - base + 1)
        } else {
            0
        };
        (base + extra) as i128
    }
}

/// Reference oracle: exactly the pre-change per-element formula, `rne_div`
/// straight from `cis::rne_div`. Used only in this test as ground truth.
fn oracle_quotient(num: i128, den: i128) -> i128 {
    rne_div(num, den)
}

/// Whichever quotient rule the production code computes: `rne_div(num, den)`
/// where `num = u*127` (normq) or `hi*127` (quantq), `den = a`. Both the
/// new normq and quantq internal quotient (before the -127..127 clamp)
/// must equal this for every element.
fn code_from_quotient(q: i128) -> i8 {
    q.clamp(-127, 127) as i8
}

// ---------------------------------------------------------------------------
// Property test: random (num, den) pairs over the real normq/quantq domain.
// ---------------------------------------------------------------------------

#[test]
fn property_normq_style_random_and_boundary() {
    let mut rng = Rng::new(0xA51CE_5EED);
    let mut cases = 0u64;

    // Crafted boundary / small denominators.
    let small_dens: &[i128] = &[1, 2, 3, 127, 128, 255, 1 << 31, (1i128 << 50) - 1];
    for &den in small_dens {
        check_den(den, &mut rng, &mut cases);
    }
    // A den right at the 2^80 + odd boundary (near normq's a < 2^81 bound).
    check_den((1i128 << 80) + 1, &mut rng, &mut cases);

    // Random denominators spanning 1 .. 2^81 - 1 in magnitude.
    while cases < 1_000_000 {
        let den = rng.magnitude(81).max(1);
        check_den(den, &mut rng, &mut cases);
    }
    assert!(cases >= 1_000_000);
}

fn check_den(den: i128, rng: &mut Rng, cases: &mut u64) {
    // u in [-den, den], including the endpoints and 0.
    for &u in &[-den, 0, den] {
        verify_one(u, den);
        *cases += 1;
    }
    // A handful of random u in [-den, den] for this den.
    for _ in 0..8 {
        let u = rng.i128_in(den);
        verify_one(u, den);
        *cases += 1;
    }
    // Construct an exact tie when possible: need den even and some u with
    // 2*(u*127 mod den) == den, i.e. num = u*127 sits exactly halfway
    // between two multiples of den. Search u in a small local window
    // around den/254 (since num ~ u*127 and we want num mod den == den/2).
    if den % 2 == 0 {
        let half = den / 2;
        // num = k*den + half for some k in [-1,1]; num must be expressible
        // as u*127 for integer u i.e. num % 127 == 0. Search nearby k, num.
        'outer: for k in -2i128..=2 {
            let target_num = k * den + half;
            // u = target_num / 127 if divisible, and |u| <= den (domain).
            if target_num % 127 == 0 {
                let u = target_num / 127;
                if u.unsigned_abs() <= den.unsigned_abs() {
                    verify_one(u, den);
                    *cases += 1;
                    break 'outer;
                }
            }
        }
    }
}

fn verify_one(u: i128, den: i128) {
    let num = u * 127;
    let expected = oracle_quotient(num, den);
    // Sign symmetry of the oracle itself (sanity on the test's own use of
    // rne_div, and documents the property normq/quantq rely on implicitly
    // since |u| <= a is symmetric).
    if u != 0 {
        assert_eq!(oracle_quotient(-num, den), -oracle_quotient(num, den));
    }
    let got = reciprocal_quotient(num, den);
    assert_eq!(
        got, expected,
        "mismatch: num={num} den={den} expected={expected} got={got}"
    );
    assert_eq!(code_from_quotient(got), code_from_quotient(expected));
}

/// Mirrors normq's inner-loop arithmetic exactly (reciprocal estimate +
/// exact remainder correction + shared RNE rule), so this test exercises
/// the identical bit pattern shipped in `cis_infer::normq`.
fn reciprocal_quotient(num: i128, den: i128) -> i128 {
    let a = den;
    debug_assert!(a > 0);
    let k = (128 - a.leading_zeros() as i32) + 8;
    let r_recip: i128 = ((1u128 << k) / a as u128) as i128;
    let mut q = (num * r_recip) >> k;
    let mut r = num - q * a;
    let mut steps = 0u32;
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
    debug_assert!(steps <= 1, "reciprocal estimate off by >1 ULP: {steps}");
    rne_round_test(q, r, a)
}

/// Local copy of the shared RNE rule (mirrors `cis_infer::rne_round`,
/// which is private) — kept in lockstep by the equivalence assertions
/// above against the true oracle.
fn rne_round_test(q: i128, r: i128, den: i128) -> i128 {
    if 2 * r > den || (2 * r == den && q & 1 != 0) {
        q + 1
    } else {
        q
    }
}

// Also drive quantq's i64 path through the same oracle comparison.
#[test]
fn property_quantq_style_i64_domain() {
    let mut rng = Rng::new(0x6117_51A5);
    let mut cases = 0u64;
    while cases < 1_000_000 {
        let bits = 1 + (rng.next_u64() as u32 % 50);
        let base: i128 = 1i128 << (bits - 1);
        let span: i128 = (1i128 << bits) - 1 - base;
        let extra = if span > 0 { rng.i128_in(span) } else { 0 };
        let a: i64 = (base + extra) as i64;
        let hi: i64 = {
            let v = rng.i128_in(a as i128);
            v as i64
        };
        let num = hi * 127;
        let q64 = num.div_euclid(a);
        let r64 = num.rem_euclid(a);
        let got = rne_round_test(q64 as i128, r64 as i128, a as i128);
        let expected = oracle_quotient(num as i128, a as i128);
        assert_eq!(got, expected, "quantq mismatch num={num} a={a}");
        cases += 1;
    }
}

// ---------------------------------------------------------------------------
// Whole-vector tests against a copy of the pre-change reference loops.
// ---------------------------------------------------------------------------

fn normq_oracle(codes: &mut [i8], h: &[i64], g: &[i32]) -> ActScale {
    let n = h.len();
    let mut a: i128 = 0;
    for (&hi, &gi) in h.iter().zip(g) {
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

fn quantq_oracle(codes: &mut [i8], h: &[i64], frac: u32) -> ActScale {
    let n = h.len();
    let mut a: i64 = 0;
    for &hi in h {
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

fn rng_h_g(rng: &mut Rng, n: usize) -> (Vec<i64>, Vec<i32>) {
    let mut h = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for _ in 0..n {
        let hv = rng.i128_in((1i128 << 50) - 1) as i64;
        let gv = rng.i128_in((1i128 << 31) - 1) as i32;
        h.push(hv);
        g.push(gv);
    }
    (h, g)
}

#[test]
fn whole_vector_normq_matches_oracle() {
    let mut rng = Rng::new(0xF00D_CAFE);
    for &n in &[1usize, 2, 3, 7, 64, 2560] {
        for _ in 0..20 {
            let (h, g) = rng_h_g(&mut rng, n);
            let mut codes_new = vec![0i8; n];
            let mut codes_ref = vec![0i8; n];
            let s_new = normq(&mut codes_new, &h, &g);
            let s_ref = normq_oracle(&mut codes_ref, &h, &g);
            assert_eq!(codes_new, codes_ref, "codes differ at n={n}");
            assert_eq!(s_new, s_ref, "scale differs at n={n}");
        }
    }
}

#[test]
fn whole_vector_quantq_matches_oracle() {
    let mut rng = Rng::new(0xBEEF_F00D);
    for &n in &[1usize, 2, 3, 7, 64, 2560] {
        for _ in 0..20 {
            let mut h = Vec::with_capacity(n);
            for _ in 0..n {
                h.push(rng.i128_in((1i128 << 50) - 1) as i64);
            }
            let frac = 20u32;
            let mut codes_new = vec![0i8; n];
            let mut codes_ref = vec![0i8; n];
            let s_new = quantq(&mut codes_new, &h, frac);
            let s_ref = quantq_oracle(&mut codes_ref, &h, frac);
            assert_eq!(codes_new, codes_ref, "codes differ at n={n}");
            assert_eq!(s_new, s_ref, "scale differs at n={n}");
        }
    }
}

#[test]
fn whole_vector_all_zero_inputs() {
    let n = 16;
    let h = vec![0i64; n];
    let g = vec![0i32; n];
    let mut codes_new = vec![1i8; n];
    let s = normq(&mut codes_new, &h, &g);
    assert_eq!(codes_new, vec![0i8; n]);
    assert_eq!(s, ActScale { num: 0, den: 1 });

    let mut codes_new_q = vec![1i8; n];
    let sq = quantq(&mut codes_new_q, &h, 20);
    assert_eq!(codes_new_q, vec![0i8; n]);
    assert_eq!(sq, ActScale { num: 0, den: 1 });
}
