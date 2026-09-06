//! `dot_i8_i16planes_avx2` (the pre-converted-head kernel) must be
//! BYTE-IDENTICAL to the scalar reference `dot_i8_i16planes`, and the whole
//! plane representation (`split_i16_planes` / `head_row_to_planes`) must be
//! an exact split of the same value `dot_i8_bf16q` computes — see
//! `cis_infer::HeadPlanes`'s doc comment for the exactness argument this
//! file checks piece by piece.

use aegis_core::cis_avx2::dot_i8_i16planes_avx2;
use aegis_core::cis_infer::{
    F, bf16_to_fixed, dot_i8_bf16q, dot_i8_i16planes, head_row_to_planes, split_i16_planes,
};

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// A finite (non-inf/nan) BF16 bit pattern whose magnitude is inside
/// `dot_i8_bf16q`'s own Q.F i32-range assert (`|w| < 2^31`) — same generator
/// as `lmhead_avx2_equivalence.rs::random_finite_bf16_in_range`, duplicated
/// rather than imported (neither test file is a library).
fn random_finite_bf16_in_range(state: &mut u64) -> u16 {
    let sign = ((lcg_next(state) >> 33) & 1) as u16;
    let exp = ((lcg_next(state) >> 40) % 138) as u16;
    let man = ((lcg_next(state) >> 40) % 128) as u16;
    (sign << 15) | (exp << 7) | man
}

fn random_i8(state: &mut u64) -> i8 {
    (((lcg_next(state) >> 40) % 255) as i32 - 127) as i8
}

/// Build a BF16 row and its i16 hi/lo planes together, asserting the row
/// converted cleanly (no exception) — the common case this file's
/// equivalence tests exercise.
fn row_and_planes(state: &mut u64, n: usize) -> (Vec<i8>, Vec<u8>, Vec<i16>, Vec<i16>) {
    let a: Vec<i8> = (0..n).map(|_| random_i8(state)).collect();
    let mut row = vec![0u8; n * 2];
    for i in 0..n {
        let bits = random_finite_bf16_in_range(state);
        row[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
    }
    let mut hi = vec![0i16; n];
    let mut lo = vec![0i16; n];
    let ok = head_row_to_planes(&row, &mut hi, &mut lo);
    assert!(ok, "row_and_planes: generator produced an exception row");
    (a, row, hi, lo)
}

/// Assert both plane-dot paths agree with each other AND with the original
/// BF16 dot, for one (a, row) pair.
fn assert_planes_identical(a: &[i8], row: &[u8], hi: &[i16], lo: &[i16], what: &str) {
    let want = dot_i8_bf16q(a, row);
    let scalar_planes = dot_i8_i16planes(a, hi, lo);
    let avx2_planes = dot_i8_i16planes_avx2(a, hi, lo);
    assert_eq!(
        want, scalar_planes,
        "{what}: bf16={want} scalar_planes={scalar_planes}"
    );
    assert_eq!(
        want, avx2_planes,
        "{what}: bf16={want} avx2_planes={avx2_planes}"
    );
}

#[test]
fn random_lengths_are_bit_identical() {
    let lengths = [0usize, 1, 2, 3, 15, 16, 17, 255, 256, 257, 2560, 2561];
    for &n in &lengths {
        let mut state = 0xB0B0_5EED_0000_0100u64 ^ (n as u64);
        let (a, row, hi, lo) = row_and_planes(&mut state, n);
        assert_planes_identical(&a, &row, &hi, &lo, &format!("random n={n}"));
    }
}

#[test]
fn split_identity_holds_for_every_representable_value() {
    // Sweep the full exponent range `bf16_to_fixed` accepts (sh <= 36) and
    // every mantissa, both signs: whenever `split_i16_planes` returns
    // `Some`, the split must reconstruct exactly, with `w_lo` in the
    // documented `[0, 2^15)` range.
    for exp in 0..=137i32 {
        for man in 0..128u16 {
            for sign in [0u16, 1] {
                let bits: u16 = (sign << 15) | ((exp as u16) << 7) | man;
                let w = bf16_to_fixed(bits, F);
                if let Some((hi, lo)) = split_i16_planes(w) {
                    let reconstructed = (hi as i64) * 32768 + lo as i64;
                    assert_eq!(reconstructed, w, "bits={bits:#06x} exp={exp} man={man}");
                    assert!(lo >= 0, "w_lo out of [0, 2^15): bits={bits:#06x} lo={lo}");
                }
            }
        }
    }
}

#[test]
fn accumulation_bound_adversarial_rows_are_bit_identical() {
    // Same adversarial construction as
    // `lmhead_avx2_equivalence.rs::accumulation_bound_adversarial_rows_are_bit_identical`:
    // maximum-magnitude alternating-sign weights against `i8`'s own extremes,
    // across enough elements to force multiple `FLUSH_BLOCKS` (256-element)
    // flushes.
    let bits_pos: u16 = (135u16 << 7) | 0x7F; // sh=21, man=0x7F, +
    let bits_neg: u16 = (1u16 << 15) | (135u16 << 7) | 0x7F; // sh=21, man=0x7F, -
    for &n in &[256usize, 257, 511, 512, 2560, 2561] {
        let mut a = vec![0i8; n];
        let mut row = vec![0u8; n * 2];
        for i in 0..n {
            a[i] = if i % 2 == 0 { -128 } else { 127 };
            let bits = if i % 4 < 2 { bits_pos } else { bits_neg };
            row[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        let mut hi = vec![0i16; n];
        let mut lo = vec![0i16; n];
        assert!(head_row_to_planes(&row, &mut hi, &mut lo));
        assert_planes_identical(&a, &row, &hi, &lo, &format!("adversarial n={n}"));
    }
}

#[test]
fn exception_row_out_of_range_is_not_a_panic_at_build_time() {
    // sh == 36 (bf16_to_fixed's own max, legal there) with a nonzero
    // mantissa exceeds `dot_i8_bf16q`'s own `< 2^31` bound (same input as
    // `lmhead_avx2_equivalence.rs::dot_i32_range_assert_panics_identically`)
    // — `head_row_to_planes` must record this as an exception (return
    // `false`), NOT panic, since that bound is checked before any assert
    // fires.
    let bits: u16 = (150u16 << 7) | 0x01; // exp=150 -> sh=36, man=1 -> m=0x81
    let row = bits.to_le_bytes();
    let mut hi = [0i16; 1];
    let mut lo = [0i16; 1];
    let ok = head_row_to_planes(&row, &mut hi, &mut lo);
    assert!(
        !ok,
        "row exceeding dot_i8_bf16q's i32 bound must be an exception row"
    );

    // The row is still reproducible via the ORIGINAL on-the-fly path's own
    // panic — same message as before this change (see
    // `lmhead_avx2_equivalence.rs::dot_i32_range_assert_panics_identically`,
    // which pins the exact message and cross-checks it against the AVX2
    // on-the-fly kernel). Here we only need: the reference itself still
    // panics on this row, i.e. routing an exception row to it is safe.
    let a = [1i8];
    let result = std::panic::catch_unwind(|| dot_i8_bf16q(&a, &row));
    assert!(
        result.is_err(),
        "exception row must still panic via the bf16 fallback path"
    );
}

#[test]
fn exception_row_inf_nan_panics_at_build_time_with_the_reference_message() {
    // inf/nan bytes: `bf16_to_fixed` panics inside `head_row_to_planes`
    // itself (before any range check runs) — this is the documented,
    // deliberate change in WHEN a malformed checkpoint is rejected (load
    // time instead of first dot-time use), not a change in whether or how.
    // The message must be identical to what `dot_i8_bf16q` would have
    // produced on the same bytes.
    let bits: u16 = (0xFFu16 << 7) | 0x01; // exp=0xFF, nan
    let row = bits.to_le_bytes();
    let a = [1i8; 1];

    let mut hi = [0i16; 1];
    let mut lo = [0i16; 1];
    let build = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        head_row_to_planes(&row, &mut hi, &mut lo)
    }));
    let dotted = std::panic::catch_unwind(|| dot_i8_bf16q(&a, &row));

    assert!(
        build.is_err(),
        "head_row_to_planes must panic on inf/nan bytes"
    );
    assert!(dotted.is_err(), "dot_i8_bf16q must panic on inf/nan bytes");

    let msg = |e: Box<dyn std::any::Any + Send>| -> String {
        if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            panic!("panic payload was neither &str nor String")
        }
    };
    assert_eq!(msg(build.unwrap_err()), msg(dotted.unwrap_err()));
}

#[test]
fn dot_i8_bf16q_matches_preconverted_dot_via_planes() {
    // The existing identity `dot_i8_bf16q(a, row) == dot_i8_i32(a, conv)`
    // (aegis-core/src/cis_infer.rs `dot_i8_bf16q_matches_preconverted_dot`)
    // extended one hop further: the SAME preconverted values, split into i16
    // planes and dotted via `dot_i8_i16planes`, must still match.
    let n = 2560;
    let mut state = 0xF0F0_u64;
    let (a, row, hi, lo) = row_and_planes(&mut state, n);
    assert_planes_identical(&a, &row, &hi, &lo, "n=2560 (real LM-head hidden size)");
}
