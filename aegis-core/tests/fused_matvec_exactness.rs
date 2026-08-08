//! The fused dual/tri matvec must be BYTE-IDENTICAL to sequential incumbent
//! calls (Rule D: bit-exactness over benchmarks).
//!
//! This is not a tolerance test. The fused kernels execute the incumbent's
//! per-row arithmetic verbatim — same FMA chain over the same columns, same
//! store-then-fold horizontal sum, same scalar-tail expression — so full
//! `to_bits()` equality is REQUIRED and any deviation is a bug, not float
//! noise.
//!
//! Dim coverage mirrors the incumbent/GEMM equivalence tests plus the shapes
//! the fusion targets: M7 (hidden 384, inter 1024), BitNet-2B (hidden 2560,
//! inter 6912, GQA K/V 640), unequal dims, <4-row tails, and dim_in shapes
//! that exercise both the 8-byte vector loop and the packed scalar tail.

use aegis_core::ops::{ternary_matvec, ternary_matvec_fused2, ternary_matvec_fused3};

fn make_weights(dim_out: usize, dim_in: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut w = vec![0u8; dim_out * dim_in / 4];
    for b in w.iter_mut() {
        // xorshift; keep only valid ternary codes 00/01/10 in each 2-bit lane
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let mut byte = 0u8;
        for lane in 0..4 {
            let code = ((s >> (lane * 8)) % 3) as u8; // 0,1,2 -> 0,+1,-1
            byte |= code << (lane * 2);
        }
        *b = byte;
    }
    w
}

fn make_input(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 2000) as f32 - 1000.0) / 1000.0
        })
        .collect()
}

fn assert_bits_eq(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}: bit mismatch at row {i}: {g} vs {w}"
        );
    }
}

/// (dim_out_a, dim_out_b, dim_in) cases shared by the fused2 tests.
/// Every dim_in is a multiple of 4 (the packed format's granularity).
const FUSED2_CASES: &[(usize, usize, usize)] = &[
    // M7 SwiGLU dual: gate/up are inter x hidden
    (1024, 1024, 384),
    // M7 attention pair
    (384, 384, 384),
    // BitNet-2B SwiGLU dual
    (6912, 6912, 2560),
    // BitNet-2B Q + K (GQA: unequal dims, fused prefix + long incumbent rest)
    (2560, 640, 2560),
    // <4-row tails on both sides, tiny dim_in (vector loop never runs)
    (7, 13, 32),
    // one side smaller than a single block
    (5, 4, 128),
    // packed tail: 36/4 = 9 bytes -> one 8-byte vector pass + 1 tail byte
    (100, 52, 36),
    // all-tail columns: 20/4 = 5 bytes < 8 -> scalar tail only
    (12, 8, 20),
];

/// (dim_out_a, dim_out_b, dim_out_c, dim_in) cases for fused3.
const FUSED3_CASES: &[(usize, usize, usize, usize)] = &[
    // BitNet-2B Q/K/V under GQA
    (2560, 640, 640, 2560),
    // M7 Q/K/V
    (384, 384, 384, 384),
    // ragged tails everywhere
    (12, 7, 5, 36),
    (4, 4, 3, 20),
];

/// Safe fused2 wrapper vs two sequential safe `ternary_matvec` calls.
/// Exercises whichever path (AVX2 or scalar) this machine dispatches.
#[test]
fn fused2_matches_sequential_matvec() {
    for &(dim_out_a, dim_out_b, dim_in) in FUSED2_CASES {
        let wa = make_weights(dim_out_a, dim_in, 0x9E3779B97F4A7C15);
        let wb = make_weights(dim_out_b, dim_in, 0xD1B54A32D192ED03);
        let input = make_input(dim_in, 0xDEADBEEFCAFEF00D);
        let (sa, sb) = (0.0123_f32, 0.0456_f32);

        let mut want_a = vec![0.0f32; dim_out_a];
        let mut want_b = vec![0.0f32; dim_out_b];
        ternary_matvec(&mut want_a, &input, &wa, dim_out_a, dim_in, sa);
        ternary_matvec(&mut want_b, &input, &wb, dim_out_b, dim_in, sb);

        let mut got_a = vec![0.0f32; dim_out_a];
        let mut got_b = vec![0.0f32; dim_out_b];
        ternary_matvec_fused2(
            &mut got_a, &mut got_b, &input, &wa, &wb, dim_out_a, dim_out_b, dim_in, sa, sb,
        );

        let label = format!("fused2 a={dim_out_a} b={dim_out_b} in={dim_in}");
        assert_bits_eq(&got_a, &want_a, &format!("{label} [A]"));
        assert_bits_eq(&got_b, &want_b, &format!("{label} [B]"));
    }
}

/// Safe fused3 wrapper vs three sequential safe `ternary_matvec` calls.
#[test]
fn fused3_matches_sequential_matvec() {
    for &(da, db, dc, dim_in) in FUSED3_CASES {
        let wa = make_weights(da, dim_in, 0xA24BAED4963EE407);
        let wb = make_weights(db, dim_in, 0x9FB21C651E98DF25);
        let wc = make_weights(dc, dim_in, 0xC2B2AE3D27D4EB4F);
        let input = make_input(dim_in, 0x165667B19E3779F9);
        let (sa, sb, sc) = (0.0123_f32, 0.0456_f32, 0.0789_f32);

        let mut want_a = vec![0.0f32; da];
        let mut want_b = vec![0.0f32; db];
        let mut want_c = vec![0.0f32; dc];
        ternary_matvec(&mut want_a, &input, &wa, da, dim_in, sa);
        ternary_matvec(&mut want_b, &input, &wb, db, dim_in, sb);
        ternary_matvec(&mut want_c, &input, &wc, dc, dim_in, sc);

        let mut got_a = vec![0.0f32; da];
        let mut got_b = vec![0.0f32; db];
        let mut got_c = vec![0.0f32; dc];
        ternary_matvec_fused3(
            &mut got_a, &mut got_b, &mut got_c, &input, &wa, &wb, &wc, da, db, dc, dim_in, sa, sb,
            sc,
        );

        let label = format!("fused3 a={da} b={db} c={dc} in={dim_in}");
        assert_bits_eq(&got_a, &want_a, &format!("{label} [A]"));
        assert_bits_eq(&got_b, &want_b, &format!("{label} [B]"));
        assert_bits_eq(&got_c, &want_c, &format!("{label} [C]"));
    }
}

/// Direct unsafe kernel A/B: `ternary_matvec_fused2_avx2` vs two sequential
/// `ternary_matvec_avx2` calls — the exact pairing the bench times. Skips on
/// non-AVX2 silicon (the HP Stream scalar path has no fused kernel).
#[test]
fn fused2_avx2_kernel_matches_sequential_avx2() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        eprintln!("skipping: AVX2+FMA not available on this machine");
        return;
    }
    use aegis_core::ops::{ternary_matvec_avx2, ternary_matvec_fused2_avx2};

    for &(dim_out_a, dim_out_b, dim_in) in FUSED2_CASES {
        let wa = make_weights(dim_out_a, dim_in, 0x2545F4914F6CDD1D);
        let wb = make_weights(dim_out_b, dim_in, 0x5851F42D4C957F2D);
        let input = make_input(dim_in, 0x14057B7EF767814F);
        let (sa, sb) = (0.031_f32, 0.057_f32);

        let mut want_a = vec![0.0f32; dim_out_a];
        let mut want_b = vec![0.0f32; dim_out_b];
        let mut got_a = vec![0.0f32; dim_out_a];
        let mut got_b = vec![0.0f32; dim_out_b];

        // SAFETY: AVX2+FMA presence checked above; buffers sized to
        // dim_out * ceil(dim_in/4) packed bytes and dim_out/dim_in floats.
        unsafe {
            ternary_matvec_avx2(&mut want_a, &input, &wa, dim_out_a, dim_in, sa);
            ternary_matvec_avx2(&mut want_b, &input, &wb, dim_out_b, dim_in, sb);
            ternary_matvec_fused2_avx2(
                &mut got_a, &mut got_b, &input, &wa, &wb, dim_out_a, dim_out_b, dim_in, sa, sb,
            );
        }

        let label = format!("fused2_avx2 a={dim_out_a} b={dim_out_b} in={dim_in}");
        assert_bits_eq(&got_a, &want_a, &format!("{label} [A]"));
        assert_bits_eq(&got_b, &want_b, &format!("{label} [B]"));
    }
}

/// Direct unsafe kernel A/B for the tri fusion.
#[test]
fn fused3_avx2_kernel_matches_sequential_avx2() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        eprintln!("skipping: AVX2+FMA not available on this machine");
        return;
    }
    use aegis_core::ops::{ternary_matvec_avx2, ternary_matvec_fused3_avx2};

    for &(da, db, dc, dim_in) in FUSED3_CASES {
        let wa = make_weights(da, dim_in, 0x27220A95FE9D0EFF);
        let wb = make_weights(db, dim_in, 0x9E3779B185EBCA87);
        let wc = make_weights(dc, dim_in, 0xC13FA9A902A6328F);
        let input = make_input(dim_in, 0x91E10DA5C79E7B1D);
        let (sa, sb, sc) = (0.031_f32, 0.057_f32, 0.093_f32);

        let mut want_a = vec![0.0f32; da];
        let mut want_b = vec![0.0f32; db];
        let mut want_c = vec![0.0f32; dc];
        let mut got_a = vec![0.0f32; da];
        let mut got_b = vec![0.0f32; db];
        let mut got_c = vec![0.0f32; dc];

        // SAFETY: AVX2+FMA presence checked above; buffers sized to
        // dim_out * ceil(dim_in/4) packed bytes and dim_out/dim_in floats.
        unsafe {
            ternary_matvec_avx2(&mut want_a, &input, &wa, da, dim_in, sa);
            ternary_matvec_avx2(&mut want_b, &input, &wb, db, dim_in, sb);
            ternary_matvec_avx2(&mut want_c, &input, &wc, dc, dim_in, sc);
            ternary_matvec_fused3_avx2(
                &mut got_a, &mut got_b, &mut got_c, &input, &wa, &wb, &wc, da, db, dc, dim_in, sa,
                sb, sc,
            );
        }

        let label = format!("fused3_avx2 a={da} b={db} c={dc} in={dim_in}");
        assert_bits_eq(&got_a, &want_a, &format!("{label} [A]"));
        assert_bits_eq(&got_b, &want_b, &format!("{label} [B]"));
        assert_bits_eq(&got_c, &want_c, &format!("{label} [C]"));
    }
}

/// The safe wrappers must no-op (leave outputs untouched) on undersized
/// buffers, mirroring `ternary_matvec`'s guard.
#[test]
fn fused_wrappers_reject_undersized_buffers() {
    let (da, db, dim_in) = (64usize, 32usize, 128usize);
    let wa = make_weights(da, dim_in, 1);
    let wb = make_weights(db, dim_in, 2);
    let input = make_input(dim_in, 3);

    // out_b one element short: BOTH outputs must stay untouched.
    let mut out_a = vec![7.0f32; da];
    let mut out_b = vec![7.0f32; db - 1];
    ternary_matvec_fused2(
        &mut out_a, &mut out_b, &input, &wa, &wb, da, db, dim_in, 1.0, 1.0,
    );
    assert!(
        out_a.iter().chain(out_b.iter()).all(|&v| v == 7.0),
        "fused2 must no-op on undersized output"
    );

    // weights_c short for fused3.
    let wc = make_weights(db, dim_in, 4);
    let mut out_a = vec![7.0f32; da];
    let mut out_b = vec![7.0f32; db];
    let mut out_c = vec![7.0f32; db];
    ternary_matvec_fused3(
        &mut out_a,
        &mut out_b,
        &mut out_c,
        &input,
        &wa,
        &wb,
        &wc[..wc.len() - 1],
        da,
        db,
        db,
        dim_in,
        1.0,
        1.0,
        1.0,
    );
    assert!(
        out_a
            .iter()
            .chain(out_b.iter())
            .chain(out_c.iter())
            .all(|&v| v == 7.0),
        "fused3 must no-op on undersized weights"
    );
}

/// Rows beyond `dim_out` in an oversized output buffer must stay untouched —
/// same contract as the incumbent (it writes rows `0..dim_out` only).
#[test]
fn fused2_leaves_rows_beyond_dim_out_untouched() {
    let (da, db, dim_in) = (10usize, 6usize, 64usize);
    let wa = make_weights(da, dim_in, 11);
    let wb = make_weights(db, dim_in, 12);
    let input = make_input(dim_in, 13);

    let mut out_a = vec![7.0f32; da + 3];
    let mut out_b = vec![7.0f32; db + 5];
    ternary_matvec_fused2(
        &mut out_a, &mut out_b, &input, &wa, &wb, da, db, dim_in, 1.0, 1.0,
    );
    assert!(
        out_a[da..].iter().all(|&v| v == 7.0),
        "fused2 wrote past dim_out_a"
    );
    assert!(
        out_b[db..].iter().all(|&v| v == 7.0),
        "fused2 wrote past dim_out_b"
    );
    // And the covered rows must still match the sequential path.
    let mut want_a = vec![0.0f32; da];
    let mut want_b = vec![0.0f32; db];
    ternary_matvec(&mut want_a, &input, &wa, da, dim_in, 1.0);
    ternary_matvec(&mut want_b, &input, &wb, db, dim_in, 1.0);
    assert_bits_eq(&out_a[..da], &want_a, "oversized [A]");
    assert_bits_eq(&out_b[..db], &want_b, "oversized [B]");
}
