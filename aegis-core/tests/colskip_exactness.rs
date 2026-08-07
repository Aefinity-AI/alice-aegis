//! Rule D bit-exactness tests for the ReLU^2 column-skip candidate
//! (`ops_colskip`, task #27 / ledger A15). No timing here — these assert
//! bytes.
//!
//! Pinned properties, per shape:
//!   1. `pack_colmajor` round-trips: every 2-bit code (including the
//!      undefined 11) is preserved verbatim at its (row, column) position.
//!   2. ORDERED variant `colskip_matvec_avx2_ordered` is BYTE-IDENTICAL to
//!      the incumbent `ops::ternary_matvec_avx2` (f32 bits, not approx) —
//!      the gold outcome the task asked for.
//!   3. ORDERED variant is byte-identical to its NON-SKIPPING scalar mirror
//!      `colskip_matvec_scalar_ordered`. The mirror processes every column
//!      unconditionally, so this equality — on inputs that contain +0.0 AND
//!      -0.0 elements — is the end-to-end proof that skipping zero-input
//!      columns is exact.
//!   4. CHAIN variant `colskip_matvec_avx2_chain` is byte-identical to its
//!      non-skipping scalar mirror `colskip_matvec_scalar_chain`. (It is NOT
//!      expected to match the incumbent — different rounding order — and no
//!      such assertion is made.)
//!   5. `fma_zero_skip_identity` proves the arithmetic claim directly:
//!      `fma(+/-0.0, w, s) == s` bitwise for w in {-1, 0, +1} and finite
//!      s != -0.0; that accumulators starting at +0.0 can never become -0.0;
//!      that hardware `_mm256_fmadd_ps` and `libm::fmaf` agree bitwise (the
//!      mirror-equivalence foundation); and the boundary case showing WHY
//!      s = -0.0 must be (and is) unreachable.
//!
//! Shapes cover the real down_proj (2560x6912) at the measured z = 0.789
//! plus z = 0 (no zeros) and z = 1 (all zeros, -0.0 included), the 8-row
//! remainder path, the tail-quad path, and a tail-only shape with fewer than
//! 8 rows. Weights are LCG-generated ternary at the measured real 42.21%
//! weight-zero fraction (ledger A6 full-model scan); input zeros include
//! bitwise -0.0 wherever `negzero` is set, because the A15 probe's z counts
//! -0.0 as zero and the kernel must treat it identically.

use aegis_core::ops::ternary_matvec_avx2;
use aegis_core::ops_colskip::{
    colskip_col_bytes, colskip_covered_cols, colskip_matvec_avx2_chain,
    colskip_matvec_avx2_ordered, colskip_matvec_scalar_chain, colskip_matvec_scalar_ordered,
    pack_colmajor,
};

/// xorshift64 PRNG — deterministic across runs and machines.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Packed 2-bit ternary weights (00=0, 01=+1, 10=-1) at `zero_frac` zeros.
fn make_packed(dim_out: usize, dim_in: usize, zero_frac: f64, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let zt = (zero_frac * 1000.0) as u64;
    let mut w = vec![0u8; dim_out * (dim_in / 4)];
    for b in w.iter_mut() {
        let mut byte = 0u8;
        for lane in 0..4 {
            let r = rng.next() % 1000;
            let code: u8 = if r < zt {
                0
            } else if r.is_multiple_of(2) {
                1
            } else {
                2
            };
            byte |= code << (2 * lane);
        }
        *b = byte;
    }
    w
}

/// Deterministic input activations in about [-3.9, 3.9] with an exact-zero
/// fraction of `zero_frac`; when `negzero` is set, alternate zeros are
/// bitwise -0.0 (the A15 caveat: z counted -0.0 as zero).
fn make_input(dim_in: usize, zero_frac: f64, negzero: bool, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let zt = (zero_frac * 1000.0) as u64;
    let mut flip = false;
    (0..dim_in)
        .map(|_| {
            let r = rng.next();
            if r % 1000 < zt {
                flip = !flip;
                if negzero && flip { -0.0 } else { 0.0 }
            } else {
                ((r % 4001) as f32 - 2000.0) / 512.0
            }
        })
        .collect()
}

fn assert_bits_eq(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: row {i}: {x:?} ({:#010x}) vs {y:?} ({:#010x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

/// Property 1: every 2-bit code is preserved verbatim at (row, col).
fn check_roundtrip(packed: &[u8], colmajor: &[u8], dim_out: usize, dim_in: usize) {
    let pdi = dim_in / 4;
    let cb = colskip_col_bytes(dim_out);
    for row in 0..dim_out {
        for cp in 0..pdi {
            let byte = packed[row * pdi + cp];
            for lane in 0..4 {
                let col = cp * 4 + lane;
                let code = (byte >> (2 * lane)) & 3;
                let got = (colmajor[col * cb + row / 4] >> (2 * (row % 4))) & 3;
                assert_eq!(
                    got, code,
                    "round-trip: row {row} col {col}: code {code:02b} became {got:02b}"
                );
            }
        }
    }
}

/// `full_mirrors` also runs the ordered scalar mirror (the chain mirror is
/// the chain variant's only oracle, so it always runs). Mirrors are
/// unconditional-fma simulations and dominate test time on big shapes.
fn check_shape(
    dim_out: usize,
    dim_in: usize,
    input_z: f64,
    negzero: bool,
    seed: u64,
    inject_undefined_code: bool,
    full_mirrors: bool,
) {
    let mut packed = make_packed(dim_out, dim_in, 0.4221, seed);
    if inject_undefined_code && !packed.is_empty() {
        // Pin the 11-degrades-to-0 contract on a real byte in BOTH layouts.
        let mid = packed.len() / 2;
        packed[mid] = 0b11_11_11_11;
    }
    let input = make_input(dim_in, input_z, negzero, seed ^ 0x9E37_79B9_7F4A_7C15);
    let scale = 0.037_f32; // non-trivial scale so the final multiply is exercised

    let cb = colskip_col_bytes(dim_out);
    let covered = colskip_covered_cols(dim_in);
    let mut colmajor = vec![0u8; covered * cb];
    pack_colmajor(&packed, dim_out, dim_in, &mut colmajor);

    // Property 1: repack round-trip.
    check_roundtrip(&packed, &colmajor, dim_out, dim_in);

    // Chain mirror: the chain variant's only oracle (always run).
    let mut out_chain_ref = vec![0.0f32; dim_out];
    colskip_matvec_scalar_chain(
        &mut out_chain_ref,
        &input,
        &colmajor,
        dim_out,
        dim_in,
        scale,
    );

    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        eprintln!("AVX2/FMA not detected: repack round-trip checked, SIMD identities skipped");
        return;
    }

    let mut out_incumbent = vec![0.0f32; dim_out];
    let mut out_ordered = vec![0.0f32; dim_out];
    let mut out_chain = vec![0.0f32; dim_out];
    let mut scratch = vec![0.0f32; 8 * dim_out];
    // SAFETY: AVX2+FMA detected above; buffer sizes match the documented
    // contracts (packed holds dim_out*(dim_in/4) bytes, colmajor holds
    // covered*cb bytes, input.len() == dim_in, outputs == dim_out, scratch
    // == 8*dim_out).
    unsafe {
        ternary_matvec_avx2(&mut out_incumbent, &input, &packed, dim_out, dim_in, scale);
        colskip_matvec_avx2_ordered(
            &mut out_ordered,
            &input,
            &colmajor,
            dim_out,
            dim_in,
            scale,
            &mut scratch,
        );
        colskip_matvec_avx2_chain(&mut out_chain, &input, &colmajor, dim_out, dim_in, scale);
    }

    // Property 2: ordered variant is byte-identical to the incumbent.
    assert_bits_eq(&out_ordered, &out_incumbent, "ordered vs incumbent AVX2");
    // Property 4: chain variant is byte-identical to its non-skipping mirror.
    assert_bits_eq(&out_chain, &out_chain_ref, "chain vs scalar chain mirror");

    // Property 3: ordered variant vs its non-skipping mirror.
    if full_mirrors {
        let mut out_ordered_ref = vec![0.0f32; dim_out];
        colskip_matvec_scalar_ordered(
            &mut out_ordered_ref,
            &input,
            &colmajor,
            dim_out,
            dim_in,
            scale,
        );
        assert_bits_eq(
            &out_ordered,
            &out_ordered_ref,
            "ordered vs scalar ordered mirror",
        );
    }
}

/// The real down_proj shape at the measured decode-mean z, -0.0 included:
/// the case the whole candidate exists for. Full mirror coverage.
#[test]
fn colskip_2560x6912_real_z() {
    check_shape(2560, 6912, 0.789, true, 0x1234_5678_9ABC_DEF1, false, true);
}

/// z = 0: no zeros, nothing skipped — the kernels must still match.
#[test]
fn colskip_2560x6912_z0_dense() {
    check_shape(2560, 6912, 0.0, false, 0x0BAD_C0DE_1357_9BDF, false, false);
}

/// z = 1: every input element is +/-0.0; every column is skipped and every
/// output must equal the incumbent's bit pattern for an all-zero input.
#[test]
fn colskip_2560x6912_z1_all_zero() {
    check_shape(2560, 6912, 1.0, true, 0xFEED_FACE_2468_ACE0, false, false);
}

/// Mid shape, half zeros, full mirrors.
#[test]
fn colskip_1280x3456_z05() {
    check_shape(1280, 3456, 0.5, true, 0xDEAD_BEEF_0F0F_0F0F, false, true);
}

/// Odd shape: 37 rows exercise the 8-row remainder (37 % 8 = 5) in both
/// AVX2 kernels; 91 columns -> 88 covered, 64 vector-region, 24 tail-quad
/// columns. Injects an undefined 11 code to pin the degrade-to-zero
/// contract across the repack.
#[test]
fn colskip_37x91_odd_shape_and_undefined_code() {
    check_shape(37, 91, 0.3, true, 0x0123_4567_89AB_CDEF, true, true);
}

/// Tail-only shape: 12 columns pack to 3 bytes < 8, so the incumbent's
/// vector loop never runs (vec_cols = 0) and EVERYTHING is tail quads; 5
/// rows < 8 so the ordered/chain kernels run pure-scalar row remainders.
#[test]
fn colskip_5x12_tail_only() {
    check_shape(5, 12, 0.25, true, 0xA5A5_5A5A_C3C3_3C3C, true, true);
}

/// High sparsity beyond the measured band.
#[test]
fn colskip_40x96_z09() {
    check_shape(40, 96, 0.9, true, 0x7777_1111_3333_9999, false, true);
}

// ---------------------------------------------------------------------------
// Property 5: the arithmetic proof the task demanded. The A15 caveat says a
// bit-exact skip kernel must treat -0.0-input columns correctly, and the
// claim "skipping them IS exact" must be PROVED, not assumed. Proof
// obligations:
//   (a) fma(x, w, s) == s bitwise for x in {+0.0, -0.0}, w in {-1, 0, +1},
//       finite s != -0.0 — the per-contribution skip identity.
//   (b) An accumulator that starts at +0.0 can never BECOME -0.0 through
//       fma steps with w in {-1, 0, +1} — so the s != -0.0 precondition of
//       (a) is an invariant, not an assumption.
//   (c) Hardware _mm256_fmadd_ps and libm::fmaf agree bitwise — the mirror
//       equivalence in properties 3/4 rests on this.
//   (d) The boundary case: fma(+0.0, 1.0, -0.0) == +0.0 != -0.0, i.e. IF an
//       accumulator could be -0.0, skipping would be WRONG — (b) is load-
//       bearing, which is why it is proved here.
// ---------------------------------------------------------------------------

/// Lane 0 of a hardware 8-lane fused multiply-add.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn hw_fma(x: f32, w: f32, s: f32) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: caller guarantees AVX2+FMA (checked via is_x86_feature_detected).
    unsafe {
        let r = _mm256_fmadd_ps(_mm256_set1_ps(x), _mm256_set1_ps(w), _mm256_set1_ps(s));
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), r);
        out[0]
    }
}

#[test]
fn fma_zero_skip_identity() {
    let have_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    let weights = [-1.0f32, 0.0, 1.0];
    // Finite s samples, -0.0 deliberately EXCLUDED (proved unreachable in
    // (b)): zero, subnormals, small/large magnitudes of both signs, f32::MAX.
    let s_samples = [
        0.0f32,
        1.0e-40,
        -1.0e-40,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        0.1,
        -0.1,
        1.0,
        -1.0,
        2.75,
        -2.75,
        1.0e30,
        -1.0e30,
        f32::MAX,
        f32::MIN,
    ];

    // (a) skip identity, on the exact scalar op the mirrors use...
    for &s in &s_samples {
        for &x in &[0.0f32, -0.0] {
            for &w in &weights {
                let r = libm::fmaf(x, w, s);
                assert_eq!(
                    r.to_bits(),
                    s.to_bits(),
                    "fmaf({x:?}, {w}, {s:?}) = {r:?} != s: skipping would be inexact"
                );
                // ...and on the exact vector op the kernels use.
                if have_avx2 {
                    // SAFETY: AVX2+FMA detected above.
                    let h = unsafe { hw_fma(x, w, s) };
                    assert_eq!(h.to_bits(), s.to_bits(), "hw fma({x:?}, {w}, {s:?}) != s");
                }
            }
        }
    }

    // (b) -0.0 unreachability: from any finite s != -0.0 and any finite x,
    // one fma step with w in {-1, 0, +1} never yields -0.0. (Round-to-
    // nearest returns -0.0 only when the exact result is -0.0, which needs
    // s = -0.0 when the product is -0.0 or zero.) Includes exact
    // cancellations (x = -s / x = s with w = -+1) which give +0.0.
    let x_samples = [
        0.0f32, -0.0, 1.0e-40, -1.0e-40, 0.1, -0.1, 1.0, -1.0, 2.75, -2.75, 1.0e30, -1.0e30,
    ];
    let neg_zero_bits = (-0.0f32).to_bits();
    for &s in &s_samples {
        for &x in &x_samples {
            for &w in &weights {
                let r = libm::fmaf(x, w, s);
                assert_ne!(
                    r.to_bits(),
                    neg_zero_bits,
                    "fmaf({x:?}, {w}, {s:?}) produced -0.0: accumulator invariant broken"
                );
                if have_avx2 {
                    // SAFETY: AVX2+FMA detected above.
                    let h = unsafe { hw_fma(x, w, s) };
                    assert_ne!(h.to_bits(), neg_zero_bits, "hw fma produced -0.0");
                }
            }
        }
    }

    // (c) hardware/libm bitwise agreement across a general grid (not just
    // zeros): the scalar mirrors stand in for the AVX2 kernels only if the
    // two fma implementations round identically.
    if have_avx2 {
        for &s in &s_samples {
            for &x in &x_samples {
                for &w in &weights {
                    // SAFETY: AVX2+FMA detected above.
                    let h = unsafe { hw_fma(x, w, s) };
                    let l = libm::fmaf(x, w, s);
                    assert_eq!(
                        h.to_bits(),
                        l.to_bits(),
                        "hw fma vs libm::fmaf diverge at ({x:?}, {w}, {s:?})"
                    );
                }
            }
        }
    }

    // (d) the boundary that makes (b) load-bearing: with a -0.0 accumulator,
    // "skip" and "compute" WOULD differ — fma(+0, +1, -0.0) is +0.0, not
    // -0.0. Documented as an assertion so the exclusion is visible.
    assert_eq!(libm::fmaf(0.0, 1.0, -0.0).to_bits(), 0.0f32.to_bits());
    assert_ne!(libm::fmaf(0.0, 1.0, -0.0).to_bits(), neg_zero_bits);
}
