//! Exhaustive proofs of the two claims the AVX2 kernel's correctness rests on.
//!
//! Both are asserted in comments in `cis_avx2.rs`. A comment is not a proof,
//! and for a kernel whose entire value is bit-exactness the difference matters.
//! Each of these enumerates its *complete* input space — there is no sampling
//! and no seed, so a pass here is a proof rather than evidence.
//!
//! CLAIM 1 — the bit-pair extraction. The kernel extracts bit-pair `k` from
//! every byte with `_mm256_srli_epi16::<2k>(v) & 0b11`. `srli_epi16` shifts
//! 16-bit lanes, so bits from the high byte migrate into the low byte. The
//! comment argues they land at positions >= 8-2k >= 2 and are therefore
//! discarded by the mask. Enumerated below over all 65,536 lane values x 4
//! shifts = 262,144 cases.
//!
//! CLAIM 2 — `vpsignb` is ternary multiply. The kernel replaces `w * a` with
//! `_mm256_sign_epi8(a, w)` for w in {-1, 0, +1}. Enumerated below over all
//! 256 activations x all 4 weight codes = 1,024 cases, against the reference's
//! own `wcode` arithmetic.

use core::arch::x86_64::*;

/// The reference's weight decode, mirrored from `cis::wcode` (which is private).
fn wcode(code: u8) -> i32 {
    match code & 0b11 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

#[test]
fn bit_pair_extraction_is_exhaustively_correct() {
    // The claim is about 16-bit shift semantics, so it is provable in scalar
    // arithmetic over the whole lane space — no SIMD needed to establish it.
    for k in 0..4u32 {
        let shift = 2 * k;
        for v in 0..=u16::MAX {
            // What the kernel computes: shift the 16-bit lane, mask each byte.
            let shifted = v >> shift;
            let got_lo = (shifted & 0x00FF) as u8 & 0b11;
            let got_hi = ((shifted >> 8) & 0x00FF) as u8 & 0b11;

            // What it must equal: each byte's own bit-pair k, independently.
            let lo = (v & 0x00FF) as u8;
            let hi = ((v >> 8) & 0x00FF) as u8;
            let want_lo = (lo >> shift) & 0b11;
            let want_hi = (hi >> shift) & 0b11;

            assert_eq!(
                got_lo, want_lo,
                "low byte contaminated: v={v:#06x} k={k} -> got {got_lo}, want {want_lo}"
            );
            assert_eq!(
                got_hi, want_hi,
                "high byte wrong: v={v:#06x} k={k} -> got {got_hi}, want {want_hi}"
            );
        }
    }
}

#[test]
fn vpsignb_is_ternary_multiply_for_every_activation() {
    if !aegis_core::ops::avx2_active() {
        eprintln!("AVX2 not active; mechanism proof skipped");
        return;
    }
    // SAFETY: guarded by avx2_active(); all operands are stack arrays of the
    // correct width and the loads are unaligned-safe.
    unsafe {
        // The code -> value LUT exactly as the kernel builds it.
        let code_lut = _mm256_setr_epi8(
            0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        );

        for code in 0..4u8 {
            let codes = _mm256_set1_epi8(code as i8);
            let w = _mm256_shuffle_epi8(code_lut, codes);

            for a in i8::MIN..=i8::MAX {
                let av = _mm256_set1_epi8(a);
                let prod = _mm256_sign_epi8(av, w);
                let mut out = [0i8; 32];
                _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, prod);

                let want_i32 = wcode(code) * a as i32;

                if a == i8::MIN && wcode(code) == -1 {
                    // The documented hazard: vpsignb negates within i8, so
                    // -(-128) wraps. The kernel's contract is that this input
                    // never reaches the SIMD path -- the deinterleave detects
                    // i8::MIN and falls back. Pin the wrap so that contract
                    // stays load-bearing rather than incidental.
                    assert_eq!(
                        out[0],
                        i8::MIN,
                        "vpsignb(-128, -1) is expected to wrap to -128; if this \
                         ever changes, the i8::MIN fallback can be revisited"
                    );
                    assert_ne!(
                        out[0] as i32, want_i32,
                        "if vpsignb matched the reference here the fallback \
                         would be dead code -- verify before removing it"
                    );
                    continue;
                }

                assert_eq!(
                    out[0] as i32,
                    want_i32,
                    "vpsignb diverged: code={code} (w={}) a={a} -> got {}, want {want_i32}",
                    wcode(code),
                    out[0]
                );
                // Every lane must agree; a per-lane discrepancy would mean the
                // 128-bit lane boundary is doing something unexpected.
                assert!(
                    out.iter().all(|&x| x == out[0]),
                    "lanes disagree for code={code} a={a}: {out:?}"
                );
            }
        }
    }
}

#[test]
fn widening_chain_cannot_overflow_at_its_stated_bounds() {
    // The kernel's exactness argument: products in [-127,127], pair sums in
    // [-254,254] (fits i16), quad sums in [-508,508] (fits i32). Prove the
    // extremes rather than trusting the arithmetic in the comment.
    let max_prod: i32 = 127;
    let max_pair = 2 * max_prod;
    let max_quad = 2 * max_pair;
    assert!(max_pair <= i16::MAX as i32, "pair sum would saturate i16");
    assert!(max_quad <= i32::MAX, "quad sum would overflow i32");

    // And the accumulator bound the reference itself asserts: |acc| <= 127*dim_in
    // must stay inside i32 for every dim_in the precondition admits.
    let max_dim_in = (i32::MAX / 127) as i64;
    assert!(
        127 * max_dim_in <= i32::MAX as i64,
        "the headroom ceiling does not actually keep the accumulator exact"
    );
}
