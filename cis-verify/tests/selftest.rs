//! Tier 2 conformance gate (spec `docs/CIS-1_SPEC_v1.0.md` §8): the
//! operation-level digest. Ported from the public conformance harness
//! `aegis-linux/examples/cis_selftest.rs` (and the constant it's pinned
//! against, `aegis-linux/tests/cis_selftest_digest.rs`), calling this
//! crate's own `cis_verify::ops` transcriptions instead of
//! `aegis_core::cis` — the whole point of this crate (design doc §3.1: "no
//! engine dependency beyond the spec ops"). Same 14 sections (golden
//! vectors A1-A8, deterministic sweeps B1-B6), same seeds, same FNV-1a 64
//! fold order, so a match here means this independent transcription of
//! `docs/CIS-1_SPEC_v1.0.md` §5's ops did not silently drift from the
//! reference on any individual op.
//!
//! MUST print exactly:
//!   CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true
//! (run with `cargo test --test selftest -- --nocapture` to see it; the
//! `println!` always executes, `--nocapture` only controls whether the
//! test harness shows it on success). The assertions below are the actual
//! acceptance gate — the digest and ALL_PASS value are checked, not just
//! printed.

use cis_verify::fnv::{FNV1A64_OFFSET, fnv1a64};
use cis_verify::ops::{
    QScale, argmax_i32, quantize_activations_i32, requant_i32, requant_i64, rmsnorm_i, rne_div,
    ternary_matvec_i8,
};

// ---- FNV-1a 64 fold helpers, identical shape to cis_selftest.rs's `Fnv` ---

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(FNV1A64_OFFSET)
    }
    fn update(&mut self, bytes: &[u8]) {
        self.0 = fnv1a64(self.0, bytes);
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

// ---- deterministic generators, bit-identical to cis_selftest.rs's --------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

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

fn gen_acts_i8(state: &mut u64, out: &mut [i8]) {
    for a in out.iter_mut() {
        *a = (((lcg_next(state) >> 40) % 255) as i32 - 127) as i8;
    }
}

/// Local mirror of the spec §4 weight decode, for the independent
/// order-reversed re-accumulation check (B3).
fn wcode(code: u8) -> i32 {
    match code & 0b11 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

const SEED_TMV: u64 = 0xA1CE_5EED_0000_0001;
const SEED_QNT: u64 = 0xA1CE_5EED_0000_0002;
const SEED_NRM: u64 = 0xA1CE_5EED_0000_0003;
const SEED_SWEEP: u64 = 0xA1CE_5EED_0000_0010;

fn section(name: &str, ok: bool, all: &mut bool) {
    println!("[{name}] {}", if ok { "PASS" } else { "FAIL" });
    *all &= ok;
}

/// Runs every section, returns (digest, all_pass) — identical structure to
/// `aegis-linux/examples/cis_selftest.rs`'s `run_selftest`.
fn run_selftest() -> (u64, bool) {
    let mut fnv = Fnv::new();
    let mut all = true;

    // ---- (a) golden vectors, replicated from cis.rs unit tests -----------

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

    // ---- (b) deterministic LCG sweep --------------------------------------

    let mut state = SEED_SWEEP;

    // B1: rne_div, 512 cases.
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

    // B2: requant_i32 / requant_i64, 512 cases each.
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

    // B3: ternary_matvec_i8 over varied shapes, including dim_in values
    // that are multiples of 4 but NOT of 8.
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

    // B4: quantize_activations_i32 over 8 length-33 vectors.
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

    // B5: rmsnorm_i over 8 length-24 vectors.
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

    // B6: argmax_i32 over 64 length-17 vectors.
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

#[test]
fn digest_matches_pinned_constant() {
    let (digest, all_pass) = run_selftest();
    println!("CIS_SELFTEST digest={digest:016x} ALL_PASS={all_pass}");
    assert!(all_pass, "cis-verify selftest reported a section FAIL");
    assert_eq!(
        digest, 0x7698_5613_c965_f643,
        "CIS-1 spec §8 Tier 2 digest mismatch — this crate's ops.rs \
         transcription diverged from aegis-core::cis on at least one \
         produced bit"
    );
}
