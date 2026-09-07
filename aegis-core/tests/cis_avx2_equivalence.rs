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
use aegis_core::cis_avx2::ternary_matvec_i8_avx2;

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
