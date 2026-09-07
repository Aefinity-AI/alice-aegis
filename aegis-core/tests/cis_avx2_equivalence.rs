//! Rule D: the AVX2 CIS-1 matvec must be BYTE-IDENTICAL to the scalar
//! reference, for every input, or it is worthless.
//!
//! CIS-1's entire proposition is bit-exactness. A kernel that is *usually*
//! identical buys nothing — it would silently break the one property the spec
//! exists to guarantee. So these are equality assertions, not tolerance
//! comparisons, and they deliberately include the shapes and values most
//! likely to break a blocked SIMD implementation:
//!
//!   - `dim_in` not a multiple of the 128-weight block (exercises the tail)
//!   - `dim_in` smaller than one block (exercises the fallback)
//!   - the `11` code, which is defined-as-zero
//!   - activation `-128`, which `vpsignb` cannot negate within i8 and which
//!     therefore MUST route to the scalar path
//!   - activation `+127` and `-127`, the live extremes
//!   - all-zero weights, all-`+1`, all-`-1`

use aegis_core::cis::ternary_matvec_i8;
use aegis_core::cis_avx2::{ternary_matmul_i8_avx2, ternary_matvec_i8_avx2};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Assert both kernels agree exactly, and report the first divergence usefully.
fn assert_identical(input: &[i8], weights: &[u8], dim_out: usize, dim_in: usize, what: &str) {
    let mut want = vec![0i32; dim_out];
    let mut got = vec![0i32; dim_out];
    ternary_matvec_i8(&mut want, input, weights, dim_out, dim_in);
    ternary_matvec_i8_avx2(&mut got, input, weights, dim_out, dim_in);
    for j in 0..dim_out {
        assert_eq!(
            want[j], got[j],
            "{what}: row {j} diverged (dim_out={dim_out}, dim_in={dim_in}): \
             scalar={} avx2={}",
            want[j], got[j]
        );
    }
}

fn packed_len(dim_out: usize, dim_in: usize) -> usize {
    dim_out * dim_in / 4
}

#[test]
fn random_shapes_are_bit_identical() {
    // Deliberately mixes block-aligned, tail-carrying, and sub-block shapes.
    let shapes = [
        (1, 128),   // exactly one block
        (3, 132),   // one block + 1 tail byte
        (5, 260),   // two blocks + tail
        (7, 64),    // below one block -> fallback path
        (2, 4),     // minimum legal dim_in
        (16, 6912), // real down_proj width
        (16, 2560), // real attn width
        (4, 2564),  // real-ish width, deliberately unaligned
    ];
    for (dim_out, dim_in) in shapes {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (dim_in as u64));
        let weights: Vec<u8> = (0..packed_len(dim_out, dim_in))
            .map(|_| (rng.next() & 0xFF) as u8)
            .collect();
        let input: Vec<i8> = (0..dim_in)
            .map(|_| {
                // full legal range, -128 excluded (covered by its own test)
                ((rng.next() % 255) as i32 - 127) as i8
            })
            .collect();
        assert_identical(
            &input,
            &weights,
            dim_out,
            dim_in,
            &format!("random {dim_out}x{dim_in}"),
        );
    }
}

#[test]
fn code_eleven_is_zero_in_both_paths() {
    // 0xFF = four `11` codes. The spec pins `11` as defined-as-zero; if the
    // pshufb LUT ever disagreed with `wcode`, this is where it shows.
    let (dim_out, dim_in) = (4, 512);
    let weights = vec![0xFFu8; packed_len(dim_out, dim_in)];
    let input: Vec<i8> = (0..dim_in)
        .map(|i| ((i % 255) as i32 - 127) as i8)
        .collect();
    let mut got = vec![0i32; dim_out];
    ternary_matvec_i8_avx2(&mut got, &input, &weights, dim_out, dim_in);
    assert!(
        got.iter().all(|&v| v == 0),
        "all-`11` weights must produce all-zero output, got {got:?}"
    );
    assert_identical(&input, &weights, dim_out, dim_in, "code 11");
}

#[test]
fn activation_minus_128_routes_to_scalar_and_stays_exact() {
    // vpsignb cannot negate -128 within i8. The kernel must detect this and
    // fall back, so the result stays identical to the reference.
    let (dim_out, dim_in) = (4, 512);
    let mut rng = Rng(0xDEAD_BEEF);
    let weights: Vec<u8> = (0..packed_len(dim_out, dim_in))
        .map(|_| (rng.next() & 0xFF) as u8)
        .collect();

    // -128 in a few positions, including the first and last element.
    let mut input: Vec<i8> = (0..dim_in)
        .map(|i| ((i % 200) as i32 - 100) as i8)
        .collect();
    input[0] = i8::MIN;
    input[dim_in / 2 + 1] = i8::MIN;
    input[dim_in - 1] = i8::MIN;

    assert_identical(&input, &weights, dim_out, dim_in, "activation -128");
}

#[test]
fn saturating_extremes_are_bit_identical() {
    // Every weight +1 against every activation +127 (and -127) drives the
    // accumulator to its widest legal magnitude for the shape.
    let (dim_out, dim_in) = (4, 6912);
    let all_plus = vec![0b01_01_01_01u8; packed_len(dim_out, dim_in)];
    let all_minus = vec![0b10_10_10_10u8; packed_len(dim_out, dim_in)];
    let all_zero = vec![0u8; packed_len(dim_out, dim_in)];

    for (w, name) in [
        (&all_plus, "all +1"),
        (&all_minus, "all -1"),
        (&all_zero, "all 0"),
    ] {
        for v in [127i8, -127, 1, 0] {
            let input = vec![v; dim_in];
            assert_identical(&input, w, dim_out, dim_in, &format!("{name} x {v}"));
        }
    }

    // The widest case must also be arithmetically what we expect, so a
    // mutually-consistent-but-wrong pair cannot pass.
    let input = vec![127i8; dim_in];
    let mut got = vec![0i32; dim_out];
    ternary_matvec_i8_avx2(&mut got, &input, &all_plus, dim_out, dim_in);
    assert!(
        got.iter().all(|&v| v == 127 * dim_in as i32),
        "all +1 weights x 127 must equal 127*dim_in = {}, got {got:?}",
        127 * dim_in as i32
    );
}

#[test]
fn v3_flush_boundary_shapes_are_bit_identical() {
    // v3 accumulates FLUSH_BLOCKS=64 i16 pair-sums per lane before widening
    // to i32. 8192 = 64*128 is exactly 64 blocks (one flush, no remainder);
    // 8320 = 65*128 is 65 blocks (one flush plus a one-block remainder
    // flush). Both must remain bit-identical to the scalar reference, and
    // in particular must not silently saturate/wrap the i16 accumulator.
    let dim_outs = [1usize, 7, 64, 2560];
    let dim_ins = [128usize, 2560, 6912, 8192, 8320, 8192 + 4, 8192 + 40];

    for &dim_out in &dim_outs {
        for &dim_in in &dim_ins {
            let mut rng = Rng(0x1234_5678_9ABC_DEF0 ^ (dim_out as u64) ^ ((dim_in as u64) << 20));
            let weights: Vec<u8> = (0..packed_len(dim_out, dim_in))
                .map(|_| (rng.next() & 0xFF) as u8)
                .collect();
            let input: Vec<i8> = (0..dim_in)
                .map(|_| ((rng.next() % 255) as i32 - 127) as i8)
                .collect();
            assert_identical(
                &input,
                &weights,
                dim_out,
                dim_in,
                &format!("flush-boundary {dim_out}x{dim_in}"),
            );
        }
    }
}

#[test]
fn v3_all_extremes_hit_the_i16_flush_bound_exactly() {
    // All-+-127 activations against all-+-1 weights on dim_in = 64*128 =
    // 8192 drives every i16 pair-sum accumulator to exactly 64 * 254 =
    // 16256 in magnitude right at the flush boundary the module doc's
    // compile-time assertion is sized against. This is the worst case the
    // bound must cover, not just a representative one.
    let dim_out = 3usize;
    let dim_in = 64 * 128; // == 8192, exactly FLUSH_BLOCKS blocks.
    let all_plus = vec![0b01_01_01_01u8; packed_len(dim_out, dim_in)];
    let all_minus = vec![0b10_10_10_10u8; packed_len(dim_out, dim_in)];

    for w in [&all_plus, &all_minus] {
        for v in [127i8, -127] {
            let input = vec![v; dim_in];
            assert_identical(&input, w, dim_out, dim_in, "flush-bound extreme");
        }
    }

    let input = vec![127i8; dim_in];
    let mut got = vec![0i32; dim_out];
    ternary_matvec_i8_avx2(&mut got, &input, &all_plus, dim_out, dim_in);
    assert!(
        got.iter().all(|&v| v == 127 * dim_in as i32),
        "all +1 weights x 127 at the flush bound must equal 127*dim_in = {}, got {got:?}",
        127 * dim_in as i32
    );
}

#[test]
fn v3_activation_minus_128_at_flush_boundary_widths_routes_to_scalar() {
    // The -128 hazard fallback must still work at shapes that exercise the
    // new flush logic (exactly at, and one block past, the flush boundary).
    for &dim_in in &[8192usize, 8320] {
        let dim_out = 2usize;
        let mut rng = Rng(0xF00D_F00D ^ dim_in as u64);
        let weights: Vec<u8> = (0..packed_len(dim_out, dim_in))
            .map(|_| (rng.next() & 0xFF) as u8)
            .collect();
        let mut input: Vec<i8> = (0..dim_in)
            .map(|i| ((i % 200) as i32 - 100) as i8)
            .collect();
        input[0] = i8::MIN;
        input[dim_in / 2] = i8::MIN;
        input[dim_in - 1] = i8::MIN;
        assert_identical(
            &input,
            &weights,
            dim_out,
            dim_in,
            &format!("-128 at flush width {dim_in}"),
        );
    }
}

#[test]
fn v3_two_hundred_random_cases_are_bit_identical() {
    // >= 200 random (shape, weights, activations) triples across a wide
    // range of dim_in (below/at/above the flush boundary) and dim_out.
    let mut rng = Rng(0x5EED_5EED_5EED_5EEDu64);
    // Every value here must be a multiple of 4: CIS-1 packs 4 weights/byte.
    let dim_in_choices = [4usize, 60, 128, 132, 512, 2560, 4096, 6912, 8192, 8320];
    let dim_out_choices = [1usize, 2, 3, 7, 16, 64];

    for i in 0..200u32 {
        let dim_in = dim_in_choices[(rng.next() as usize) % dim_in_choices.len()];
        let dim_out = dim_out_choices[(rng.next() as usize) % dim_out_choices.len()];
        let weights: Vec<u8> = (0..packed_len(dim_out, dim_in))
            .map(|_| (rng.next() & 0xFF) as u8)
            .collect();
        let input: Vec<i8> = (0..dim_in)
            .map(|_| ((rng.next() % 255) as i32 - 127) as i8)
            .collect();
        assert_identical(
            &input,
            &weights,
            dim_out,
            dim_in,
            &format!("random#{i} {dim_out}x{dim_in}"),
        );
    }
}

#[test]
fn row_independence_holds() {
    // Each output row must depend only on its own weight row. A blocked kernel
    // that mis-strides would leak neighbouring rows; comparing a multi-row call
    // against per-row single-row calls catches exactly that.
    let (dim_out, dim_in) = (9, 1028); // rows odd, dim_in tail-carrying
    let mut rng = Rng(0x0BAD_C0DE);
    let weights: Vec<u8> = (0..packed_len(dim_out, dim_in))
        .map(|_| (rng.next() & 0xFF) as u8)
        .collect();
    let input: Vec<i8> = (0..dim_in)
        .map(|_| ((rng.next() % 255) as i32 - 127) as i8)
        .collect();

    let mut all = vec![0i32; dim_out];
    ternary_matvec_i8_avx2(&mut all, &input, &weights, dim_out, dim_in);

    let row_bytes = dim_in / 4;
    for r in 0..dim_out {
        let mut one = vec![0i32; 1];
        ternary_matvec_i8_avx2(
            &mut one,
            &input,
            &weights[r * row_bytes..(r + 1) * row_bytes],
            1,
            dim_in,
        );
        assert_eq!(all[r], one[0], "row {r} is not independent");
    }
}

#[test]
fn v3b_offset_identity_extremes() {
    // v3b decodes u = w + 1 in {0, 1, 2} and computes acc - sum_a instead of
    // v3's signed vpsignb product. Every combination of the live activation
    // extremes (+127, -127, alternating +-127, and -128 which must still
    // route to scalar) against the live weight extremes (all +1, all -1, all
    // zero, all code 0b11, random) must remain bit-identical to the scalar
    // reference: the offset identity must hold at every corner, not just
    // typical inputs.
    let dim_out = 3usize;
    let dim_in = 8192usize; // == 64 * 128, exactly FLUSH_BLOCKS blocks.
    let all_plus = vec![0b01_01_01_01u8; packed_len(dim_out, dim_in)];
    let all_minus = vec![0b10_10_10_10u8; packed_len(dim_out, dim_in)];
    let all_zero = vec![0u8; packed_len(dim_out, dim_in)];
    let all_eleven = vec![0xFFu8; packed_len(dim_out, dim_in)];
    let mut rng = Rng(0xB16B_00B5_B16B_00B5);
    let random_weights: Vec<u8> = (0..packed_len(dim_out, dim_in))
        .map(|_| (rng.next() & 0xFF) as u8)
        .collect();

    let weight_cases: [(&[u8], &str); 5] = [
        (&all_plus, "all +1"),
        (&all_minus, "all -1"),
        (&all_zero, "all 0"),
        (&all_eleven, "all code 11"),
        (&random_weights, "random"),
    ];

    let alternating: Vec<i8> = (0..dim_in)
        .map(|i| if i % 2 == 0 { 127 } else { -127 })
        .collect();
    let all_min: Vec<i8> = vec![i8::MIN; dim_in];

    let input_cases: [(&[i8], &str); 4] = [
        (&vec![127i8; dim_in], "all +127"),
        (&vec![-127i8; dim_in], "all -127"),
        (&alternating, "alternating +-127"),
        (&all_min, "all -128 (must route to scalar)"),
    ];

    for (weights, wname) in weight_cases {
        for (input, iname) in input_cases {
            assert_identical(
                input,
                weights,
                dim_out,
                dim_in,
                &format!("v3b extremes: weights={wname} input={iname}"),
            );
        }
    }
}

#[test]
fn v3b_flush_bound_508_exact() {
    // All +127 (and all -127) activations against all +1 weights (u = 2) make
    // each of the two activations in a pair contribute u*a = 2*127 = 254 in
    // the same direction, so each pair sum is exactly +-508 and FLUSH_BLOCKS
    // = 64 of them accumulate to exactly 64 * 508 = 32512 (< i16::MAX =
    // 32767) right at the widths that are exact multiples of
    // FLUSH_BLOCKS*BLOCK_BYTES*4, and one block short/long of that boundary.
    // This is the worst-case magnitude the compile-time assert in
    // cis_avx2.rs is sized against.
    let dim_out = 2usize;
    const FLUSH_BLOCKS_BOUNDARY: usize = 64 * 32 * 4; // FLUSH_BLOCKS * BLOCK_BYTES * 4 == 8192
    for &dim_in in &[
        FLUSH_BLOCKS_BOUNDARY - 4,
        FLUSH_BLOCKS_BOUNDARY,
        FLUSH_BLOCKS_BOUNDARY + 4,
    ] {
        let all_plus = vec![0b01_01_01_01u8; packed_len(dim_out, dim_in)];
        for v in [127i8, -127] {
            let input = vec![v; dim_in];
            assert_identical(
                &input,
                &all_plus,
                dim_out,
                dim_in,
                &format!("v3b flush bound 508, dim_in={dim_in} v={v}"),
            );
        }
    }

    // The exact-boundary case must also match the expected arithmetic value,
    // so a mutually-consistent-but-wrong pair cannot pass.
    let dim_in = FLUSH_BLOCKS_BOUNDARY;
    let all_plus = vec![0b01_01_01_01u8; packed_len(dim_out, dim_in)];
    let input = vec![127i8; dim_in];
    let mut got = vec![0i32; dim_out];
    ternary_matvec_i8_avx2(&mut got, &input, &all_plus, dim_out, dim_in);
    assert!(
        got.iter().all(|&v| v == 127 * dim_in as i32),
        "all +1 weights x 127 at the flush bound must equal 127*dim_in = {}, got {got:?}",
        127 * dim_in as i32
    );
}

#[test]
fn v3b_two_hundred_random_cases_are_bit_identical() {
    // Same coverage style as `v3_two_hundred_random_cases_are_bit_identical`,
    // re-run for the v3b kernel: random shapes including dim_in below one
    // block and shapes with a tail.
    let mut rng = Rng(0x0FFB_0FFB_0FFB_0FFBu64);
    let dim_in_choices = [4usize, 60, 128, 132, 512, 2560, 4096, 6912, 8192, 8320];
    let dim_out_choices = [1usize, 2, 3, 7, 16, 64];

    for i in 0..200u32 {
        let dim_in = dim_in_choices[(rng.next() as usize) % dim_in_choices.len()];
        let dim_out = dim_out_choices[(rng.next() as usize) % dim_out_choices.len()];
        let weights: Vec<u8> = (0..packed_len(dim_out, dim_in))
            .map(|_| (rng.next() & 0xFF) as u8)
            .collect();
        let input: Vec<i8> = (0..dim_in)
            .map(|_| ((rng.next() % 255) as i32 - 127) as i8)
            .collect();
        assert_identical(
            &input,
            &weights,
            dim_out,
            dim_in,
            &format!("v3b random#{i} {dim_out}x{dim_in}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Batched prefill: `ternary_matmul_i8_avx2` (8-token tiles).
// ---------------------------------------------------------------------------

/// For every token independently, the batched kernel's output row must
/// equal `ternary_matvec_i8` (the scalar single-token reference) applied to
/// that token alone — the batched kernel's whole correctness claim is that
/// tiling tokens changes nothing about any individual output element.
#[test]
fn matmul_matches_matvec_per_token() {
    let mut rng = Rng(0x5EED_5EED_5EED_5EEDu64);
    // dim_in multiples of 128 (block-aligned) and odd tails; dim_out spans
    // 1..70; n_tok spans across, below, and straddling the 8-token tile.
    let dim_in_choices = [128usize, 256, 132, 260, 4, 6912, 8320];
    let dim_out_choices = [1usize, 2, 3, 7, 17, 32, 69];
    let n_tok_choices = [1usize, 2, 7, 8, 9, 16, 17, 33];

    for i in 0..120u32 {
        let dim_in = dim_in_choices[(rng.next() as usize) % dim_in_choices.len()];
        let dim_out = dim_out_choices[(rng.next() as usize) % dim_out_choices.len()];
        let n_tok = n_tok_choices[(rng.next() as usize) % n_tok_choices.len()];

        let weights: Vec<u8> = (0..packed_len(dim_out, dim_in))
            .map(|_| (rng.next() & 0xFF) as u8)
            .collect();
        let inputs: Vec<i8> = (0..n_tok * dim_in)
            .map(|_| ((rng.next() % 255) as i32 - 127) as i8)
            .collect();

        let mut got = vec![0i32; n_tok * dim_out];
        ternary_matmul_i8_avx2(&mut got, &inputs, &weights, dim_out, dim_in, n_tok);

        for t in 0..n_tok {
            let mut want = vec![0i32; dim_out];
            ternary_matvec_i8(
                &mut want,
                &inputs[t * dim_in..(t + 1) * dim_in],
                &weights,
                dim_out,
                dim_in,
            );
            assert_eq!(
                want,
                got[t * dim_out..(t + 1) * dim_out],
                "matmul#{i}: token {t} diverged from per-token matvec \
                 (dim_out={dim_out}, dim_in={dim_in}, n_tok={n_tok})"
            );
        }
    }
}

/// The matmul kernel's own i16 flush bound (`FLUSH_BLOCKS_MM * 2032 =
/// 32512 < i16::MAX`), exercised at widths that land exactly on 16 blocks
/// and one block short/long of that boundary, with all-+127 activations
/// against all-+1 weights (`u = 2`) — the same worst-case magnitude
/// argument `v3b_flush_bound_508_exact` makes for the single-token kernel,
/// scaled to this kernel's larger per-block-per-token bound (it sums all
/// four lane pair-sums into one accumulator per block, not four).
#[test]
fn matmul_flush_bound_2032_exact() {
    let dim_out = 2usize;
    const FLUSH_BOUNDARY: usize = 16 * 32 * 4; // FLUSH_BLOCKS_MM * BLOCK_BYTES * 4 == 2048
    let n_tok = 9usize; // spans one full 8-token tile plus a 1-token tail tile
    for &dim_in in &[FLUSH_BOUNDARY - 4, FLUSH_BOUNDARY, FLUSH_BOUNDARY + 4] {
        let all_plus = vec![0b01_01_01_01u8; packed_len(dim_out, dim_in)];
        for v in [127i8, -127] {
            let inputs = vec![v; n_tok * dim_in];
            let mut got = vec![0i32; n_tok * dim_out];
            ternary_matmul_i8_avx2(&mut got, &inputs, &all_plus, dim_out, dim_in, n_tok);
            let want_row: i32 = v as i32 * dim_in as i32;
            for t in 0..n_tok {
                for j in 0..dim_out {
                    assert_eq!(
                        got[t * dim_out + j],
                        want_row,
                        "matmul flush bound 2032, dim_in={dim_in} v={v} token={t} row={j}"
                    );
                }
            }
        }
    }
}

/// Any token carrying `i8::MIN` must route the WHOLE call to the scalar
/// reference (never a per-token mix), matching `ternary_matvec_i8_avx2`'s
/// own whole-call fallback discipline.
#[test]
fn matmul_minus_128_routes_to_scalar() {
    let dim_out = 3usize;
    let dim_in = 260usize; // two blocks + tail, exercises both loop bodies
    let n_tok = 9usize;
    let weights: Vec<u8> = (0..packed_len(dim_out, dim_in))
        .map(|i| (i as u8).wrapping_mul(37))
        .collect();

    for hazard_tok in [0usize, 4, 8] {
        let mut inputs = vec![0i8; n_tok * dim_in];
        let mut rng = Rng(0xA11A_A11A_A11A_A11Au64 ^ hazard_tok as u64);
        for x in inputs.iter_mut() {
            *x = ((rng.next() % 255) as i32 - 127) as i8;
        }
        inputs[hazard_tok * dim_in] = i8::MIN;

        let mut got = vec![0i32; n_tok * dim_out];
        ternary_matmul_i8_avx2(&mut got, &inputs, &weights, dim_out, dim_in, n_tok);

        for t in 0..n_tok {
            let mut want = vec![0i32; dim_out];
            ternary_matvec_i8(
                &mut want,
                &inputs[t * dim_in..(t + 1) * dim_in],
                &weights,
                dim_out,
                dim_in,
            );
            assert_eq!(
                want,
                got[t * dim_out..(t + 1) * dim_out],
                "matmul -128 hazard (token {hazard_tok} carries it): token {t} diverged"
            );
        }
    }
}

/// Saturating extremes (+-127, alternating, zero, the `11` code) across a
/// multi-tile token batch.
#[test]
fn matmul_extremes() {
    let dim_out = 4usize;
    let dim_in = 512usize;
    let n_tok = 17usize; // two full tiles + a 1-token tail tile
    let weight_patterns: [u8; 4] = [
        0b00_00_00_00, // all zero
        0b01_01_01_01, // all +1
        0b10_10_10_10, // all -1
        0b11_11_11_11, // undefined code, defined-as-zero
    ];
    let weights: Vec<u8> = (0..dim_out * (dim_in / 4))
        .map(|i| weight_patterns[i % weight_patterns.len()])
        .collect();

    let mut inputs = vec![0i8; n_tok * dim_in];
    for (i, x) in inputs.iter_mut().enumerate() {
        *x = match i % 4 {
            0 => 127,
            1 => -127,
            2 => 0,
            _ => {
                if (i / 4) % 2 == 0 {
                    63
                } else {
                    -63
                }
            }
        };
    }

    let mut got = vec![0i32; n_tok * dim_out];
    ternary_matmul_i8_avx2(&mut got, &inputs, &weights, dim_out, dim_in, n_tok);

    for t in 0..n_tok {
        let mut want = vec![0i32; dim_out];
        ternary_matvec_i8(
            &mut want,
            &inputs[t * dim_in..(t + 1) * dim_in],
            &weights,
            dim_out,
            dim_in,
        );
        assert_eq!(
            want,
            got[t * dim_out..(t + 1) * dim_out],
            "matmul extremes: token {t} diverged"
        );
    }
}

// ---------------------------------------------------------------------
// Kernel microbench: tmm_8tok vs 8x tmv (ignored by default; run with
// `cargo test --release ... -- --ignored --nocapture bench_tmm_vs_tmv`).
// Shape is one real MLP up-projection (dim_out=6912, dim_in=2560,
// BitNet-2B scale). Not a correctness test — the equality check inside
// it is a sanity guard, not the point; the point is the printed
// BENCH line, which is the raw log this PR's A/B claims are read from
// (see the PR body's finding section).
// ---------------------------------------------------------------------
#[test]
#[ignore]
fn bench_tmm_vs_tmv() {
    use std::time::Instant;

    const DIM_OUT: usize = 6912;
    const DIM_IN: usize = 2560;
    const N_TOK: usize = 8;
    const REPS: usize = 10;

    let mut rng = Rng(0xB1E2_C3A4_D5F6_9788);
    let n_bytes = packed_len(1, DIM_IN); // packed bytes per row (dim_out=1 trick)
    let weights: Vec<u8> = (0..DIM_OUT * n_bytes)
        .map(|_| (rng.next() & 0xFF) as u8)
        .collect();
    let inputs: Vec<i8> = (0..N_TOK * DIM_IN)
        .map(|_| {
            // Avoid i8::MIN so this measures the fast path, not the
            // whole-call scalar fallback.
            let v = (rng.next() & 0xFF) as i8;
            if v == i8::MIN { 0 } else { v }
        })
        .collect();

    // Correctness sanity: batched matmul must equal 8 independent matvecs.
    let mut got_mm = vec![0i32; N_TOK * DIM_OUT];
    ternary_matmul_i8_avx2(&mut got_mm, &inputs, &weights, DIM_OUT, DIM_IN, N_TOK);
    let mut got_mv = vec![0i32; N_TOK * DIM_OUT];
    for t in 0..N_TOK {
        ternary_matvec_i8_avx2(
            &mut got_mv[t * DIM_OUT..(t + 1) * DIM_OUT],
            &inputs[t * DIM_IN..(t + 1) * DIM_IN],
            &weights,
            DIM_OUT,
            DIM_IN,
        );
    }
    assert_eq!(got_mm, got_mv, "bench_tmm_vs_tmv: outputs diverged");

    // Time REPS reps of each, timing the SAME shape/data both ways.
    let mut out_mm = vec![0i32; N_TOK * DIM_OUT];
    let t0 = Instant::now();
    for _ in 0..REPS {
        ternary_matmul_i8_avx2(&mut out_mm, &inputs, &weights, DIM_OUT, DIM_IN, N_TOK);
    }
    let tmm_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut out_mv = vec![0i32; DIM_OUT];
    let t1 = Instant::now();
    for _ in 0..REPS {
        for t in 0..N_TOK {
            ternary_matvec_i8_avx2(
                &mut out_mv,
                &inputs[t * DIM_IN..(t + 1) * DIM_IN],
                &weights,
                DIM_OUT,
                DIM_IN,
            );
        }
    }
    let tmv_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Bytes of packed weight streamed: tmm decodes each block ONCE for
    // the whole 8-token tile (n_bytes*DIM_OUT total, once); tmv decodes
    // it fresh for every one of the 8 independent matvec calls
    // (n_bytes*DIM_OUT*N_TOK total) — this is the traffic difference the
    // kernel exists to remove.
    let weight_bytes = (DIM_OUT * n_bytes) as f64;
    let tmm_gbps = (weight_bytes * REPS as f64) / (tmm_ms / 1000.0) / 1e9;
    let tmv_gbps = (weight_bytes * N_TOK as f64 * REPS as f64) / (tmv_ms / 1000.0) / 1e9;
    let speedup = tmv_ms / tmm_ms;

    println!(
        "BENCH tmm_8tok_ms={tmm_ms:.3} tmv_8x_ms={tmv_ms:.3} speedup={speedup:.3} tmm_gbps={tmm_gbps:.2} tmv_gbps={tmv_gbps:.2}"
    );
}
