//! `dot_i8_bf16q_avx2` must be BYTE-IDENTICAL to the scalar reference
//! `dot_i8_bf16q`, for every input, including inputs that make the scalar
//! reference itself panic (inf/nan bytes, `bf16_to_fixed`'s magnitude bound,
//! `dot_i8_bf16q`'s own Q.F i32-range assert) — see `cis_avx2`'s doc comment
//! on `dot_i8_bf16q_avx2` for why those cases fall back to a whole-row
//! scalar recompute rather than being approximated.

use aegis_core::cis_avx2::dot_i8_bf16q_avx2;
use aegis_core::cis_infer::{F, bf16_to_fixed, dot_i8_bf16q};

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// A finite (non-inf/nan) BF16 bit pattern whose magnitude is inside
/// `bf16_to_fixed`'s Q.F range (`sh <= 36`, i.e. `exp <= 150` for F=20),
/// covering both signs and every mantissa bit pattern.
fn random_finite_bf16_in_range(state: &mut u64) -> u16 {
    let sign = ((lcg_next(state) >> 33) & 1) as u16;
    // exp in [0, 150]: spans zero, subnormals, and every normal exponent
    // this dot path can legally see (sh = exp - 114 <= 36).
    let exp = ((lcg_next(state) >> 40) % 151) as u16;
    let man = ((lcg_next(state) >> 40) % 128) as u16;
    (sign << 15) | (exp << 7) | man
}

fn random_i8(state: &mut u64) -> i8 {
    (((lcg_next(state) >> 40) % 255) as i32 - 127) as i8
}

/// Assert both paths agree exactly on one (a, row) pair, reporting a useful
/// diff on divergence.
fn assert_dot_identical(a: &[i8], row: &[u8], what: &str) {
    let want = dot_i8_bf16q(a, row);
    let got = dot_i8_bf16q_avx2(a, row);
    assert_eq!(want, got, "{what}: scalar={want} avx2={got}");
}

#[test]
fn random_lengths_are_bit_identical() {
    // Deliberately mixes lengths that are not multiples of the AVX2 dot's
    // 4-element block, plus the block size itself and the real LM-head
    // hidden size.
    let lengths = [0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 64, 65, 2560];
    for &n in &lengths {
        let mut state = 0xA1CE_5EED_0000_0100u64 ^ (n as u64);
        let a: Vec<i8> = (0..n).map(|_| random_i8(&mut state)).collect();
        let mut row = vec![0u8; n * 2];
        for i in 0..n {
            let bits = random_finite_bf16_in_range(&mut state);
            row[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        assert_dot_identical(&a, &row, &format!("random n={n}"));
    }
}

#[test]
fn range_edges_are_bit_identical() {
    // Craft rows at the exact edges bf16_to_fixed's branches switch on:
    // sh == 23 (largest shift whose result still fits `dot_i8_bf16q`'s own
    // Q.F i32-range assert even at the maximum mantissa 0xFF -- sh == 36,
    // bf16_to_fixed's own legal max, ALWAYS trips that second assert for any
    // nonzero mantissa, so it belongs to the panic-equivalence test below,
    // not this one), sh == 0 (shift-by-zero boundary), sh == -1 (first
    // right-shift step), sh == -62 (last non-zero right shift), sh == -63
    // (forced to 0), plus subnormal (exp=0) and signed zero.
    // sh = exp - 127 - 7 + F = exp - 114 for normals (F = 20).
    let edge_exps: [i32; 7] = [
        137, // sh = 23 (largest shift still inside the dot's i32 bound)
        136, // sh = 22
        114, // sh = 0 (left/right boundary)
        113, // sh = -1 (first RNE right shift)
        52,  // sh = -62 (last non-zero right shift)
        51,  // sh = -63 (forced to 0)
        0,   // subnormal / signed zero
    ];
    let mans: [u16; 5] = [0, 1, 0x40, 0x7E, 0x7F];
    let mut a = Vec::new();
    let mut row = Vec::new();
    for &exp in &edge_exps {
        for &man in &mans {
            for sign in [0u16, 1] {
                a.push(if a.len() % 2 == 0 { 127i8 } else { -127i8 });
                let bits: u16 = (sign << 15) | ((exp as u16) << 7) | man;
                row.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    assert_eq!(a.len() * 2, row.len());
    assert_dot_identical(&a, &row, "range edges");

    // Same edge set, but exercised as the *tail* of a row (not a multiple of
    // the 4-element block), so the AVX2 kernel's scalar tail loop is hit too.
    let mut a_tail = vec![1i8; 6];
    a_tail.extend_from_slice(&a[..7]);
    let mut row_tail = vec![0u8; 12];
    let mut state = 0x7A11_u64;
    for i in 0..6 {
        let bits = random_finite_bf16_in_range(&mut state);
        row_tail[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
    }
    row_tail.extend_from_slice(&row[..14]);
    assert_dot_identical(&a_tail, &row_tail, "range edges as tail");
}

/// Extract a panic payload as a `String` regardless of whether it was a
/// plain `&'static str` (unformatted `assert!`) or a formatted `String`
/// (`assert!` with interpolation) — `dot_i8_bf16q`'s three asserts are a mix
/// of both.
fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        panic!("panic payload was neither &str nor String")
    }
}

#[test]
fn inf_nan_panics_identically() {
    // exp == 0xFF (inf/nan) must panic in both paths with the same message,
    // not silently diverge.
    let a = vec![1i8; 4];
    let mut row = vec![0u8; 8];
    let bits: u16 = (0xFFu16 << 7) | 0x01; // exp=0xFF, nan
    row[0..2].copy_from_slice(&bits.to_le_bytes());

    let scalar = std::panic::catch_unwind(|| dot_i8_bf16q(&a, &row));
    let avx2 = std::panic::catch_unwind(|| dot_i8_bf16q_avx2(&a, &row));
    assert!(scalar.is_err(), "scalar path must panic on inf/nan bytes");
    assert!(avx2.is_err(), "avx2 path must panic on inf/nan bytes");
    assert_eq!(
        panic_msg(scalar.unwrap_err()),
        panic_msg(avx2.unwrap_err()),
        "panic messages must match exactly"
    );
}

#[test]
fn bf16_to_fixed_range_assert_panics_identically() {
    // sh > 36 (magnitude too large for Q.F, bf16_to_fixed's own bound) must
    // panic identically too.
    let a = vec![1i8; 4];
    let mut row = vec![0u8; 8];
    let bits: u16 = (200u16 << 7) | 0x01; // exp=200 -> sh = 200-114 = 86 > 36
    row[0..2].copy_from_slice(&bits.to_le_bytes());

    let scalar = std::panic::catch_unwind(|| dot_i8_bf16q(&a, &row));
    let avx2 = std::panic::catch_unwind(|| dot_i8_bf16q_avx2(&a, &row));
    assert!(scalar.is_err());
    assert!(avx2.is_err());
    assert_eq!(panic_msg(scalar.unwrap_err()), panic_msg(avx2.unwrap_err()));
}

#[test]
fn dot_i32_range_assert_panics_identically() {
    // sh == 36 (bf16_to_fixed's own max, legal there) but with a nonzero
    // mantissa: `m << 36 >= 2^36` always exceeds `dot_i8_bf16q`'s tighter
    // `< 2^31` bound. bf16_to_fixed itself succeeds; the second, dot-local
    // assert is the one that must fire, identically in both paths.
    let a = vec![1i8; 4];
    let mut row = vec![0u8; 8];
    let bits: u16 = (150u16 << 7) | 0x01; // exp=150 -> sh=36, man=1 -> m=0x81
    row[0..2].copy_from_slice(&bits.to_le_bytes());

    // Sanity: bf16_to_fixed itself does not panic on this input.
    let _ = bf16_to_fixed(bits, F);

    let scalar = std::panic::catch_unwind(|| dot_i8_bf16q(&a, &row));
    let avx2 = std::panic::catch_unwind(|| dot_i8_bf16q_avx2(&a, &row));
    assert!(
        scalar.is_err(),
        "scalar path must hit the dot's i32-range assert"
    );
    assert!(
        avx2.is_err(),
        "avx2 path must hit the dot's i32-range assert"
    );
    assert_eq!(panic_msg(scalar.unwrap_err()), panic_msg(avx2.unwrap_err()));
}

#[test]
fn matches_bf16_to_fixed_reference_pointwise() {
    // Cross-check against the scalar bf16_to_fixed conversion directly
    // (not just the fused dot), on a single-element row swept over a dense
    // exponent range, to localize any conversion bug independent of the
    // dot's own accumulation.
    let mut state = 0xF00D_u64;
    for exp in 0..=150i32 {
        for _ in 0..8 {
            let man = (lcg_next(&mut state) % 128) as u16;
            let sign = (lcg_next(&mut state) & 1) as u16;
            let bits: u16 = (sign << 15) | ((exp as u16) << 7) | man;
            let want = bf16_to_fixed(bits, F);
            let a = [1i8];
            let row = bits.to_le_bytes();
            let got = dot_i8_bf16q_avx2(&a, &row);
            assert_eq!(want, got, "bits={bits:#06x} exp={exp} man={man}");
        }
    }
}
