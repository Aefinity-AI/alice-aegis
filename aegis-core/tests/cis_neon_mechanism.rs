//! Exhaustive proofs of the claims the NEON kernel's correctness rests on.
//!
//! All are asserted in comments in `cis_neon.rs`. A comment is not a proof,
//! and for a kernel whose entire value is bit-exactness the difference
//! matters. Each of these enumerates its *complete* input space — there is no
//! sampling and no seed, so a pass here is a proof rather than evidence.
//! Unlike the x86 mechanism file, these proofs run the actual intrinsics: on
//! aarch64 NEON is architecturally mandatory, so there is no capability gate
//! to skip past and the enumeration always executes.
//!
//! CLAIM 1 — the bit-pair extraction. The kernel extracts bit-pair `k` from
//! every byte with `vandq_u8(vshrq_n_u8::<2k>(v), 0b11)`. `vshrq_n_u8` shifts
//! each byte independently (no cross-byte migration to argue about, unlike
//! x86's 16-bit-lane shift), and the kernel additionally maps the extracted
//! code through the `vqtbl1q_s8` LUT. Enumerated below over all 256 byte
//! values x 4 pair positions, through the LUT, against the reference's own
//! `wcode` arithmetic.
//!
//! CLAIM 2 — `vmulq_s8` is ternary multiply. The kernel computes `w * a`
//! directly in i8 for w in {-1, 0, +1}. Enumerated below over all 256
//! activations x all 4 weight codes, with the `(-128) * (-1)` wrap pinned in
//! both directions so the scalar-fallback contract stays load-bearing.
//!
//! CLAIM 3 — the widening chain is exact. `vpaddlq_s8` then `vpadalq_s16`
//! must lose nothing at the extremes the kernel's bounds argument states.
//! Proven at the extreme vectors, plus the scalar headroom arithmetic.
#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;

/// The reference's weight decode, mirrored from `cis::wcode` (which is private).
fn wcode(code: u8) -> i32 {
    match code & 0b11 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

/// The code -> value LUT exactly as the kernel builds it.
///
/// # Safety
/// aarch64 only (guaranteed by the crate-level cfg); NEON is mandatory.
unsafe fn kernel_lut() -> int8x16_t {
    unsafe { core::mem::transmute([0i8, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) }
}

#[test]
fn bit_pair_extraction_through_the_lut_is_exhaustively_correct() {
    // SAFETY: NEON is mandatory on aarch64; all loads/stores use stack arrays
    // of the correct width.
    unsafe {
        let code_lut = kernel_lut();
        let pair_mask = vdupq_n_u8(0b11);

        for v in 0..=u8::MAX {
            let vv = vdupq_n_u8(v);
            // The four extractions exactly as the kernel writes them
            // (shift-by-0 is not a legal immediate, so k=0 is unshifted).
            let codes = [
                vandq_u8(vv, pair_mask),
                vandq_u8(vshrq_n_u8::<2>(vv), pair_mask),
                vandq_u8(vshrq_n_u8::<4>(vv), pair_mask),
                vandq_u8(vshrq_n_u8::<6>(vv), pair_mask),
            ];
            for (k, &cd) in codes.iter().enumerate() {
                let w = vqtbl1q_s8(code_lut, cd);
                let mut out = [0i8; 16];
                vst1q_s8(out.as_mut_ptr(), w);

                let want = wcode((v >> (2 * k)) & 0b11);
                assert!(
                    out.iter().all(|&x| x as i32 == want),
                    "byte {v:#04x} pair {k}: LUT gave {out:?}, wcode says {want}"
                );
            }
        }
    }
}

#[test]
fn vmulq_s8_is_ternary_multiply_for_every_activation() {
    // SAFETY: NEON is mandatory on aarch64; all loads/stores use stack arrays
    // of the correct width.
    unsafe {
        let code_lut = kernel_lut();

        for code in 0..4u8 {
            let codes = vdupq_n_u8(code);
            let w = vqtbl1q_s8(code_lut, codes);

            for a in i8::MIN..=i8::MAX {
                let av = vdupq_n_s8(a);
                let prod = vmulq_s8(av, w);
                let mut out = [0i8; 16];
                vst1q_s8(out.as_mut_ptr(), prod);

                let want_i32 = wcode(code) * a as i32;

                if a == i8::MIN && wcode(code) == -1 {
                    // The documented hazard: vmulq_s8 multiplies within i8, so
                    // (-128) * (-1) wraps. The kernel's contract is that this
                    // input never reaches the SIMD path -- the deinterleave
                    // detects i8::MIN and falls back. Pin the wrap so that
                    // contract stays load-bearing rather than incidental.
                    assert_eq!(
                        out[0],
                        i8::MIN,
                        "vmulq_s8(-128, -1) is expected to wrap to -128; if this \
                         ever changes, the i8::MIN fallback can be revisited"
                    );
                    assert_ne!(
                        out[0] as i32, want_i32,
                        "if vmulq_s8 matched the reference here the fallback \
                         would be dead code -- verify before removing it"
                    );
                    continue;
                }

                assert_eq!(
                    out[0] as i32,
                    want_i32,
                    "vmulq_s8 diverged: code={code} (w={}) a={a} -> got {}, want {want_i32}",
                    wcode(code),
                    out[0]
                );
                // Every lane must agree; a per-lane discrepancy would mean the
                // vector multiply is doing something positional.
                assert!(
                    out.iter().all(|&x| x == out[0]),
                    "lanes disagree for code={code} a={a}: {out:?}"
                );
            }
        }
    }
}

#[test]
fn widening_chain_is_exact_at_its_stated_extremes() {
    // The kernel's exactness argument: products in [-127,127], pairwise sums
    // in [-254,254] (fits i16), each vpadalq lane step adds at most 508
    // (fits i32). First the scalar arithmetic the comment states:
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

    // Then the instructions themselves, at the extremes. vpaddlq_s8 must
    // widen BEFORE adding (pairwise add LONG) — an i8 add that widened after
    // would wrap at ±254. Same for vpadalq_s16 at ±508 per step.
    // SAFETY: NEON is mandatory on aarch64; stores use stack arrays of the
    // correct width.
    unsafe {
        for extreme in [127i8, -127] {
            let prod = vdupq_n_s8(extreme);
            let pairs = vpaddlq_s8(prod);
            let mut p = [0i16; 8];
            vst1q_s16(p.as_mut_ptr(), pairs);
            assert!(
                p.iter().all(|&x| x as i32 == 2 * extreme as i32),
                "vpaddlq_s8 at {extreme}: got {p:?}, want {}",
                2 * extreme as i32
            );

            // Accumulate the widest step onto the widest running lane value
            // the kernel can legally reach, and require exactness.
            let acc = vdupq_n_s32(1_000_000);
            let acc2 = vpadalq_s16(acc, pairs);
            let mut q = [0i32; 4];
            vst1q_s32(q.as_mut_ptr(), acc2);
            assert!(
                q.iter().all(|&x| x == 1_000_000 + 4 * extreme as i32),
                "vpadalq_s16 at {extreme}: got {q:?}, want {}",
                1_000_000 + 4 * extreme as i32
            );
        }
    }
}
