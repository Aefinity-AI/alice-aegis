//! cis_selftest — cross-ISA identity artifact for the CIS-1 integer reference
//! ops (`aegis_core::cis`).
//!
//! Runs (a) the exact unit golden vectors from `cis.rs` (constants replicated
//! here — the in-crate versions are `#[cfg(test)]`) and (b) a deterministic
//! LCG-driven sweep (fixed seeds, >512 cases across `rne_div`,
//! `requant_i32`/`requant_i64`, `ternary_matvec_i8` with varied shapes
//! including non-multiple-of-8 input dims, `quantize_activations_i32`,
//! `rmsnorm_i`, `argmax_i32`), folds every produced output value's
//! little-endian bytes into one FNV-1a 64-bit digest in deterministic order,
//! and prints exactly one final line:
//!
//!   CIS_SELFTEST digest=<16 hex> ALL_PASS=<true|false>
//!
//! CIS-1's axiom is that its integer semantics are ISA-independent: the Dell
//! Inspiron 15 i5-5200U (AVX2) and the HP Stream N4020 (Gemini Lake,
//! SSE2-class) MUST print the same digest line. A cross-machine mismatch
//! falsifies the portability claim; a mismatch against the pinned constant in
//! `aegis-linux/tests/cis_selftest_digest.rs` means a refactor changed the
//! produced bits. Identity/correctness tool only; never a perf instrument.

use aegis_core::cis::{
    QScale, argmax_i32, quantize_activations_i32, requant_i32, requant_i64, rmsnorm_i, rne_div,
    ternary_matvec_i8,
};

// ---- FNV-1a 64 (offset basis / prime per the reference spec) ---------------

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn i8s(&mut self, v: &[i8]) {
        for &x in v {
            self.update(&x.to_le_bytes());
        }
    }
    fn i32s(&mut self, v: &[i32]) {
        for &x in v {
            self.update(&x.to_le_bytes());
        }
    }
}

// ---- deterministic generators (bit-identical to cis.rs unit tests) ---------

/// 64-bit LCG, Knuth MMIX constants — same generator as the cis.rs goldens
/// and their independent Python bignum cross-check.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// Packed ternary bytes with codes in {00, 01, 10}: four sequential draws per
/// byte, code = (draw >> 60) % 3, low pair first.
fn gen_packed_no11(state: &mut u64, out: &mut [u8]) {
    for b in out.iter_mut() {
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

/// Local mirror of the spec §1 weight decode (00→0, 01→+1, 10→−1, 11→0) for
/// the independent order-reversed re-accumulation check.
fn wcode(code: u8) -> i32 {
    match code & 0b11 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

// Seeds shared with the cis.rs golden corpus.
const SEED_TMV: u64 = 0xA1CE_5EED_0000_0001;
const SEED_QNT: u64 = 0xA1CE_5EED_0000_0002;
const SEED_NRM: u64 = 0xA1CE_5EED_0000_0003;
// Sweep seed, disjoint from the unit seeds by construction.
const SEED_SWEEP: u64 = 0xA1CE_5EED_0000_0010;

fn section(name: &str, ok: bool, all: &mut bool) {
    println!("[{name}] {}", if ok { "PASS" } else { "FAIL" });
    *all &= ok;
}

/// Runs every section, returns (digest, all_pass). Shared with the host test
/// that pins the digest constant.
pub fn run_selftest() -> (u64, bool) {
    let mut fnv = Fnv::new();
    let mut all = true;

    // ---- (a) golden vectors, replicated from cis.rs unit tests -------------

    // A1: ternary_matvec_i8 8x32 golden.
    {
        const GOLDEN: [i32; 8] = [291, 193, 292, -86, -369, 160, -134, -214];
        let mut state = SEED_TMV;
        let mut w = [0u8; 64];
        gen_packed_no11(&mut state, &mut w);
        let mut a = [0i8; 32];
        gen_acts_i8(&mut state, &mut a);
        let mut out = [0i32; 8];
        ternary_matvec_i8(&mut out, &a, &w, 8, 32);
        fnv.i32s(&out);
        section("A1 golden_tmv_8x32", out == GOLDEN, &mut all);
    }

    // A2: undefined weight code 11 decodes to zero.
    {
        let w = [0xFFu8; 8];
        let a = [127i8, -127, 5, -5, 99, -99, 1, -1];
        let mut out = [123i32; 4];
        ternary_matvec_i8(&mut out, &a, &w, 4, 8);
        fnv.i32s(&out);
        let ok_all11 = out == [0; 4];

        let w = [0b1101_1000u8];
        let a = [10i8, 20, 30, 40];
        let mut out1 = [0i32; 1];
        ternary_matvec_i8(&mut out1, &a, &w, 1, 4);
        fnv.i32s(&out1);
        section(
            "A2 tmv_undefined_code_11",
            ok_all11 && out1[0] == -20 + 30,
            &mut all,
        );
    }

    // A3: REQUANT golden table (RNE ties and clamps).
    {
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
        let mut ok = true;
        for (acc, m, s, want) in CASES {
            let got = requant_i32(acc, QScale::new(m, s));
            fnv.i8s(&[got]);
            ok &= got == want;
        }
        section("A3 golden_requant_table", ok, &mut all);
    }

    // A4: dynamic activation quantization golden.
    {
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
        fnv.i8s(&q);
        fnv.update(&absmax.to_le_bytes());
        section(
            "A4 golden_quantize_activations_16",
            absmax == 955_153 && q == GOLDEN,
            &mut all,
        );
    }

    // A5: quantization RNE ties and all-zero convention.
    {
        let v = [254i32, 1, -1, 3, -3];
        let mut q = [0i8; 5];
        let d1 = quantize_activations_i32(&mut q, &v);
        fnv.i8s(&q);
        fnv.update(&d1.to_le_bytes());
        let ok_ties = d1 == 254 && q == [127, 0, 0, 2, -2];

        let v = [0i32; 4];
        let mut qz = [9i8; 4];
        let d0 = quantize_activations_i32(&mut qz, &v);
        fnv.i8s(&qz);
        fnv.update(&d0.to_le_bytes());
        section(
            "A5 quantize_ties_and_zero",
            ok_ties && d0 == 0 && qz == [0; 4],
            &mut all,
        );
    }

    // A6: RMSNORM-I golden + all-zero input.
    {
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
        fnv.i8s(&out);
        let ok_golden = out == GOLDEN;

        let z = [0i8; 16];
        let mut outz = [7i8; 16];
        rmsnorm_i(&mut outz, &z, &w, QScale::new(1 << 30, 24));
        fnv.i8s(&outz);
        section(
            "A6 golden_rmsnorm_16",
            ok_golden && outz == [0; 16],
            &mut all,
        );
    }

    // A7: ARGMAX lowest-index tie rule.
    {
        let cases: [(&[i32], u32); 5] = [
            (&[5, 9, 9, 1], 1),
            (&[3, 3, 3], 0),
            (&[-7], 0),
            (&[i32::MIN, i32::MIN], 0),
            (&[0, -1, 2, 2, -5, 2], 2),
        ];
        let mut ok = true;
        for (logits, want) in cases {
            let got = argmax_i32(logits);
            fnv.update(&got.to_le_bytes());
            ok &= got == want;
        }
        section("A7 argmax_ties_break_low", ok, &mut all);
    }

    // A8: QScale::from_ratio goldens (offline multiplier generator).
    {
        // (num, den, expected (M, S) — None where from_ratio must reject).
        type RatioCase = (u64, u64, Option<(i32, u8)>);
        let cases: [RatioCase; 9] = [
            (1, 3, Some((1_431_655_765, 1))),
            (1, 2, Some((1 << 30, 0))),
            (127, 300, Some((1_818_202_822, 1))),
            (1, 127, Some((1_082_196_484, 6))),
            (u64::MAX, u64::MAX, None),
            (u64::MAX - 1, u64::MAX, Some((i32::MAX, 0))),
            (0, 5, None),
            (5, 0, None),
            (1, u64::MAX, None),
        ];
        let mut ok = true;
        for (num, den, want) in cases {
            let got = QScale::from_ratio(num, den);
            match got {
                Some(q) => {
                    fnv.update(&[1u8]);
                    fnv.update(&q.m().to_le_bytes());
                    fnv.update(&[q.s()]);
                }
                None => fnv.update(&[0u8]),
            }
            ok &= got.map(|q| (q.m(), q.s())) == want;
        }
        section("A8 from_ratio_goldens", ok, &mut all);
    }

    // ---- (b) deterministic LCG sweep ---------------------------------------

    let mut state = SEED_SWEEP;

    // B1: rne_div, 512 cases. Property: the result is the round-to-nearest
    // representable with ties to even — |2·rem| <= den, and equality only on
    // an even quotient.
    {
        let mut ok = true;
        for _ in 0..512 {
            let num = lcg_next(&mut state) as i64 as i128;
            let den = ((lcg_next(&mut state) % (1u64 << 40)) + 1) as i128;
            let q = rne_div(num, den);
            fnv.update(&q.to_le_bytes());
            let rem = num - q * den;
            let two_r = 2 * rem.abs();
            ok &= two_r <= den && (two_r != den || q & 1 == 0);
        }
        section("B1 rne_div sweep (512)", ok, &mut all);
    }

    // B2: requant_i32 / requant_i64, 512 cases each. Properties: output in
    // [−127, 127] and the i32 form agrees with the i64 form on i32 inputs.
    {
        let mut ok = true;
        for _ in 0..512 {
            let m = ((1u64 << 30) + (lcg_next(&mut state) % (1 << 30))) as i32;
            let s = (lcg_next(&mut state) % 63) as u8;
            let q = QScale::new(m, s);
            let acc32 = lcg_next(&mut state) as u32 as i32;
            let r32 = requant_i32(acc32, q);
            fnv.i8s(&[r32]);
            ok &= r32 == requant_i64(acc32 as i64, q) && (-127..=127).contains(&(r32 as i32));
            let acc64 = lcg_next(&mut state) as i64;
            let r64 = requant_i64(acc64, q);
            fnv.i8s(&[r64]);
            ok &= (-127..=127).contains(&(r64 as i32));
        }
        section("B2 requant sweep (512+512)", ok, &mut all);
    }

    // B3: ternary_matvec_i8 over varied shapes, including dim_in values that
    // are multiples of 4 but NOT of 8 (tail handling), with full-range weight
    // bytes (undefined 11 codes included on purpose). Property: strict
    // order-reversed re-accumulation reproduces every row exactly (the CIS-1
    // axiom; false for f32 — that failure is banked as E1).
    {
        const SHAPES: [(usize, usize); 10] = [
            (1, 4),
            (3, 12),
            (5, 20),
            (7, 36),
            (2, 52),
            (9, 68),
            (4, 100),
            (6, 244),
            (8, 32),
            (16, 256),
        ];
        let mut ok = true;
        for (dim_out, dim_in) in SHAPES {
            let packed = dim_in / 4;
            let mut w = vec![0u8; dim_out * packed];
            for b in w.iter_mut() {
                *b = (lcg_next(&mut state) >> 56) as u8;
            }
            let mut a = vec![0i8; dim_in];
            gen_acts_i8(&mut state, &mut a);
            let mut out = vec![0i32; dim_out];
            ternary_matvec_i8(&mut out, &a, &w, dim_out, dim_in);
            fnv.i32s(&out);
            for (row, &want) in out.iter().enumerate() {
                let w_row = &w[row * packed..(row + 1) * packed];
                let mut rev: i32 = 0;
                for col in (0..dim_in).rev() {
                    let code = w_row[col / 4] >> (2 * (col % 4));
                    rev += wcode(code) * a[col] as i32;
                }
                ok &= rev == want;
            }
        }
        section("B3 tmv shapes sweep (10 shapes)", ok, &mut all);
    }

    // B4: quantize_activations_i32 over 8 length-33 vectors (not a multiple
    // of 8). Properties: returned denominator equals max |x|; every output is
    // in [−127, 127] with the sign of its input (or zero).
    {
        let mut ok = true;
        for _ in 0..8 {
            let mut v = [0i32; 33];
            for x in v.iter_mut() {
                *x = ((lcg_next(&mut state) >> 24) % 4_000_001) as i32 - 2_000_000;
            }
            let mut q = [0i8; 33];
            let absmax = quantize_activations_i32(&mut q, &v);
            fnv.i8s(&q);
            fnv.update(&absmax.to_le_bytes());
            let want_absmax = v.iter().map(|&x| (x as i64).abs()).max().unwrap();
            ok &= absmax == want_absmax;
            for (&qi, &xi) in q.iter().zip(v.iter()) {
                ok &= (-127..=127).contains(&(qi as i32));
                ok &= qi == 0 || (qi > 0) == (xi > 0);
            }
        }
        section("B4 quantize sweep (8x33)", ok, &mut all);
    }

    // B5: rmsnorm_i over 8 length-24 vectors (not a multiple of 8), random
    // normalized multipliers, S in [20, 30]. Property: re-running the same
    // inputs reproduces the same bytes (pure function), and an interleaved
    // all-zero input maps to all-zero output.
    {
        let mut ok = true;
        for _ in 0..8 {
            let mut x = [0i8; 24];
            gen_acts_i8(&mut state, &mut x);
            let mut w = [0i16; 24];
            for g in w.iter_mut() {
                *g = ((lcg_next(&mut state) % 16384) + 1) as i16;
            }
            let m = ((1u64 << 30) + (lcg_next(&mut state) % (1 << 30))) as i32;
            let s = (20 + (lcg_next(&mut state) % 11)) as u8;
            let q = QScale::new(m, s);
            let mut out = [0i8; 24];
            rmsnorm_i(&mut out, &x, &w, q);
            fnv.i8s(&out);
            let mut out2 = [0i8; 24];
            rmsnorm_i(&mut out2, &x, &w, q);
            ok &= out == out2;
            let z = [0i8; 24];
            let mut outz = [5i8; 24];
            rmsnorm_i(&mut outz, &z, &w, q);
            ok &= outz == [0; 24];
        }
        section("B5 rmsnorm sweep (8x24)", ok, &mut all);
    }

    // B6: argmax_i32 over 64 length-17 vectors. Property: the winner is
    // maximal and no earlier index attains it (lowest-index tie rule).
    {
        let mut ok = true;
        for _ in 0..64 {
            let mut v = [0i32; 17];
            for x in v.iter_mut() {
                *x = lcg_next(&mut state) as u32 as i32;
            }
            let got = argmax_i32(&v);
            fnv.update(&got.to_le_bytes());
            let best = v[got as usize];
            ok &= v.iter().all(|&x| x <= best);
            ok &= v[..got as usize].iter().all(|&x| x < best);
        }
        section("B6 argmax sweep (64x17)", ok, &mut all);
    }

    (fnv.0, all)
}

#[cfg_attr(test, allow(dead_code))]
fn main() {
    println!("CIS-1 integer semantics self-test (aegis_core::cis reference ops)");
    let (digest, all_pass) = run_selftest();
    println!("CIS_SELFTEST digest={digest:016x} ALL_PASS={all_pass}");
    if !all_pass {
        std::process::exit(1);
    }
}
