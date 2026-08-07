//! Rule D bit-exactness tests for the bitplane-dense candidate
//! (`ops_bitplane`). No timing here — these assert bytes.
//!
//! Pinned properties, per shape:
//!   1. `pack_bitplanes` round-trips against the 2-bit codes
//!      (01 <-> +plane, 10 <-> -plane, 00 and 11 <-> neither).
//!   2. Variant (i) `bitplane_matvec_avx2` is BYTE-IDENTICAL to the incumbent
//!      `ops::ternary_matvec_avx2` (f32 bits compared, not approx).
//!   3. Variant (i) is byte-identical to its scalar mirror
//!      `bitplane_matvec_scalar` (which is therefore also incumbent-identical
//!      — the SSE2-class path shares the AVX2 path's exact numerics).
//!   4. Variant (ii) `bitplane_matvec_avx2_dual` is byte-identical to its
//!      scalar mirror `bitplane_matvec_scalar_dual`. (It is NOT expected to
//!      match the incumbent — different rounding order — and no such
//!      assertion is made.)
//!
//! Shapes cover the 4-row-unroll remainder rows and the col_packed tail:
//! 2560x2560, 2560x6912, 6912x2560 (BitNet decode shapes), 384x1024, and
//! 37x91 (37 = 9*4+1 remainder row; 91 -> 22 packed bytes = 16 vector + 6
//! tail, with 88 covered columns). Weights are LCG-generated ternary at the
//! measured real 42.21% zero fraction (ledger A6 full-model scan).

use aegis_core::ops::ternary_matvec_avx2;
use aegis_core::ops_bitplane::{
    bitplane_matvec_avx2, bitplane_matvec_avx2_dual, bitplane_matvec_scalar,
    bitplane_matvec_scalar_dual, bitplane_words_per_row, pack_bitplanes,
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

/// Deterministic input activations in about [-3.9, 3.9], zeros included.
fn make_input(dim_in: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..dim_in)
        .map(|_| ((rng.next() % 4001) as f32 - 2000.0) / 512.0)
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

/// Property 1: every 2-bit code maps to exactly the right plane bits.
fn check_roundtrip(packed: &[u8], pos: &[u64], neg: &[u64], dim_out: usize, dim_in: usize) {
    let pdi = dim_in / 4;
    let words = bitplane_words_per_row(dim_in);
    for row in 0..dim_out {
        for cp in 0..pdi {
            let byte = packed[row * pdi + cp];
            for lane in 0..4 {
                let col = cp * 4 + lane;
                let code = (byte >> (2 * lane)) & 3;
                let p = (pos[row * words + col / 64] >> (col % 64)) & 1;
                let n = (neg[row * words + col / 64] >> (col % 64)) & 1;
                let (ep, en) = match code {
                    1 => (1, 0),
                    2 => (0, 1),
                    _ => (0, 0), // 00 zero; undefined 11 degrades to zero
                };
                assert_eq!(
                    (p, n),
                    (ep, en),
                    "round-trip: row {row} col {col} code {code:02b} -> pos={p} neg={n}"
                );
            }
        }
    }
}

fn check_shape(dim_out: usize, dim_in: usize, seed: u64, inject_undefined_code: bool) {
    let mut packed = make_packed(dim_out, dim_in, 0.4221, seed);
    if inject_undefined_code && !packed.is_empty() {
        // Pin the 11-degrades-to-0 contract on a real byte: both the
        // incumbent's UNPACK_LUT and pack_bitplanes must treat it as zero.
        let mid = packed.len() / 2;
        packed[mid] = 0b11_11_11_11;
    }
    let input = make_input(dim_in, seed ^ 0x9E3779B97F4A7C15);
    let scale = 0.037_f32; // non-trivial scale so the final multiply is exercised

    let words = bitplane_words_per_row(dim_in);
    let mut pos = vec![0u64; dim_out * words];
    let mut neg = vec![0u64; dim_out * words];
    pack_bitplanes(&packed, dim_out, dim_in, &mut pos, &mut neg);

    // Property 1: pack round-trip.
    check_roundtrip(&packed, &pos, &neg, dim_out, dim_in);

    // Scalar oracles (always run, any machine).
    let mut out_scalar = vec![0.0f32; dim_out];
    let mut out_scalar_dual = vec![0.0f32; dim_out];
    bitplane_matvec_scalar(&mut out_scalar, &input, &pos, &neg, dim_out, dim_in, scale);
    bitplane_matvec_scalar_dual(
        &mut out_scalar_dual,
        &input,
        &pos,
        &neg,
        dim_out,
        dim_in,
        scale,
    );

    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        eprintln!("AVX2/FMA not detected: scalar round-trip checked, SIMD identities skipped");
        return;
    }

    let mut out_incumbent = vec![0.0f32; dim_out];
    let mut out_i = vec![0.0f32; dim_out];
    let mut out_ii = vec![0.0f32; dim_out];
    // SAFETY: AVX2+FMA detected above; buffer sizes match the documented
    // contracts (packed holds dim_out*(dim_in/4) bytes, planes hold
    // dim_out*words words, input.len() == dim_in, outputs == dim_out).
    unsafe {
        ternary_matvec_avx2(&mut out_incumbent, &input, &packed, dim_out, dim_in, scale);
        bitplane_matvec_avx2(&mut out_i, &input, &pos, &neg, dim_out, dim_in, scale);
        bitplane_matvec_avx2_dual(&mut out_ii, &input, &pos, &neg, dim_out, dim_in, scale);
    }

    // Property 2: order-preserving variant is byte-identical to the incumbent.
    assert_bits_eq(&out_i, &out_incumbent, "variant(i) vs incumbent AVX2");
    // Property 3: and to its scalar mirror.
    assert_bits_eq(&out_i, &out_scalar, "variant(i) vs scalar mirror");
    // Property 4: dual variant is byte-identical to its scalar mirror.
    assert_bits_eq(
        &out_ii,
        &out_scalar_dual,
        "variant(ii) vs scalar dual mirror",
    );
}

#[test]
fn bitplane_2560x2560() {
    check_shape(2560, 2560, 0x1234_5678_9ABC_DEF1, false);
}

#[test]
fn bitplane_2560x6912() {
    check_shape(2560, 6912, 0x0BAD_C0DE_1357_9BDF, false);
}

#[test]
fn bitplane_6912x2560() {
    check_shape(6912, 2560, 0xFEED_FACE_2468_ACE0, false);
}

#[test]
fn bitplane_384x1024() {
    check_shape(384, 1024, 0xDEAD_BEEF_0F0F_0F0F, false);
}

/// Odd shape: 37 = 9*4 + 1 exercises the remainder-row path; 91 columns pack
/// to 22 bytes (16 vector-loop + 6 scalar-tail, 88 covered columns). Also
/// injects an undefined 11 code to pin the degrade-to-zero contract.
#[test]
fn bitplane_37x91_odd_shape_and_undefined_code() {
    check_shape(37, 91, 0x0123_4567_89AB_CDEF, true);
}
