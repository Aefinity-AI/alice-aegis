//! NEON ternary matvec for CIS-1 integer semantics — bit-identical to
//! [`crate::cis::ternary_matvec_i8`].
//!
//! # Why this is allowed to exist
//!
//! `lib.rs` states the porting doctrine: a new architecture gets the portable
//! chain first and earns its fast kernels second. aarch64 earned it — the
//! selftest digest (A28) and the token-level decode digest (A29) both
//! reproduced on Neoverse silicon with zero changes to the portable chain.
//! This module is the second step, the aarch64 analog of `cis_avx2` (whose
//! x86 measurement, ledger A27, showed the vectorized integer path beating
//! the float AVX2 kernel). No ARM performance number exists yet and none is
//! claimed here: Rule A requires named physical hardware, and the only ARM
//! this programme currently touches is unnamed shared CI. Until an ARM box is
//! on the bench, this kernel's whole deliverable is bit-identity at NEON
//! width, proven by the same test discipline as `cis_avx2`.
//!
//! # Why vectorising is SAFE here and is not for f32
//!
//! Same argument as `cis_avx2`, and it is the reference's own contract: the
//! integer dot is *"exact in ANY summation order"*. Lane reassociation cannot
//! change a bit. The f32 kernels can never claim this; that is why CIS-1
//! exists.
//!
//! # This is NOT a re-proposal of a settled negative
//!
//! Identical container, identical traffic: this reads the existing 2-bit
//! packing byte for byte. The A6/A7/A16/A17 rejections concerned x86 kernel
//! *designs* (zero-skip branching, 4-bit LUT respecialisation, fusion,
//! bitplane re-encoding); none of those designs appears here.
//!
//! # Method
//!
//! AArch64 makes both x86 workarounds unnecessary:
//!
//! - **Per-byte shifts exist** (`vshrq_n_u8` shifts each byte independently),
//!   so bit-pair extraction is literally `(v >> 2k) & 0b11` per byte — no
//!   16-bit-lane contamination argument to defend. The 4-entry code→value map
//!   is one `vqtbl1q_s8`.
//! - **A real i8 multiply exists** (`vmulq_s8`), so the ternary product is
//!   the product instruction itself rather than `vpsignb`'s sign-select.
//!
//! The structure is otherwise `cis_avx2`'s: a packed byte holds 4 weights in
//! bit-pairs; extracting pair `k` from 16 consecutive bytes yields weights
//! for input elements `≡ k (mod 4)`, so activations are deinterleaved into
//! four stride-4 subsequences **once per matvec** — O(dim_in) against
//! O(dim_out × dim_in) of work — and reused across all rows.
//!
//! Widening is exact at every step: products in `[-127, 127]` (i8), pairwise
//! sums in `[-254, 254]` (`vpaddlq_s8` → i16), accumulated into i32 lanes by
//! `vpadalq_s16`; the precondition `dim_in <= i32::MAX/127` bounds every lane.
//!
//! # The `-128` hazard, same hazard, same handling
//!
//! `vmulq_s8` multiplies within i8, so `(-128) × (-1)` wraps to `-128`; the
//! scalar reference computes in i32 and yields `+128`. Exactly `vpsignb`'s
//! wrap, handled exactly the same way: the deinterleave pass — which already
//! touches every activation — detects any `-128` and routes the whole call to
//! the scalar reference. The bit-identity guarantee is therefore
//! unconditional for every possible input, not conditional on the quantizer
//! upholding its `|q| <= 127` invariant.

// Matches `cis_avx2`/`ops`: preconditions live in a `# Safety` section on the
// function rather than per-intrinsic blocks. Every unsafe operation below is
// a load or an intrinsic covered by that contract.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::cis::{check_tmv_preconditions, ternary_matvec_i8};
use alloc::vec;
use core::arch::aarch64::*;
use core::sync::atomic::{AtomicBool, Ordering};

/// Bytes of packed weights consumed per SIMD block (4 weights per byte).
const BLOCK_BYTES: usize = 16;

/// Scalar-forcing toggle, mirroring `ops::FORCE_SCALAR` (which is x86-only).
/// NEON is architecturally mandatory in AArch64, so unlike x86 there is no
/// capability half to this gate — but a future in-boot same-binary A/B on ARM
/// needs the race toggle to reach this path, same as the x86 A/Bs do.
static FORCE_SCALAR: AtomicBool = AtomicBool::new(false);

pub fn set_force_scalar(v: bool) {
    FORCE_SCALAR.store(v, Ordering::Relaxed);
}

/// The dispatch gate. Capability is unconditional on AArch64; only the race
/// toggle can turn the SIMD path off.
#[inline]
pub(crate) fn simd_on() -> bool {
    !FORCE_SCALAR.load(Ordering::Relaxed)
}

/// NEON TMV. Byte-identical to [`crate::cis::ternary_matvec_i8`] for every
/// input, falling back to it when scalar is forced, when the shape is too
/// small to block, or when the activations contain `-128`.
pub fn ternary_matvec_i8_neon(
    output: &mut [i32],
    input: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
) {
    // Enforced here, not only on the fallback: otherwise an illegal call is
    // rejected or silently answered depending on the shape (the exact drift
    // the shared precondition fn exists to prevent — see its doc).
    check_tmv_preconditions(output, input, weights_packed, dim_out, dim_in);
    let n_bytes = dim_in / 4;

    if !simd_on() || n_bytes < BLOCK_BYTES {
        ternary_matvec_i8(output, input, weights_packed, dim_out, dim_in);
        return;
    }

    // Deinterleave activations into the four stride-4 subsequences the
    // bit-pair extraction consumes, and detect the -128 hazard in the same
    // pass.
    let mut lanes = vec![0i8; 4 * n_bytes];
    let mut has_min = false;
    for j in 0..n_bytes {
        for k in 0..4 {
            let v = input[4 * j + k];
            has_min |= v == i8::MIN;
            lanes[k * n_bytes + j] = v;
        }
    }
    if has_min {
        ternary_matvec_i8(output, input, weights_packed, dim_out, dim_in);
        return;
    }

    // SAFETY: NEON is mandatory in AArch64, satisfying the target-feature
    // precondition unconditionally. `lanes` is exactly `4 * n_bytes` long,
    // `n_bytes >= BLOCK_BYTES` holds, and the callee bounds every weight and
    // output access against `dim_out` and `n_bytes` before dereferencing.
    unsafe { tmv_i8_neon(output, input, &lanes, weights_packed, dim_out, n_bytes) }
}

/// # Safety
/// `lanes` must be the stride-4 deinterleave of `input`, `4 * n_bytes` long,
/// and `input` must contain no `i8::MIN`.
#[target_feature(enable = "neon")]
unsafe fn tmv_i8_neon(
    output: &mut [i32],
    input: &[i8],
    lanes: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    n_bytes: usize,
) {
    // code -> value, applied with `vqtbl1q_s8`. Codes are 0..=3; `11` is
    // defined-as-zero, matching `cis::wcode` and its golden vectors. Indices
    // 4..=15 of the table are unreachable (codes are masked to 2 bits).
    let code_lut: int8x16_t =
        core::mem::transmute([0i8, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let pair_mask = vdupq_n_u8(0b11);

    let full_blocks = n_bytes / BLOCK_BYTES;
    let tail_start_b = full_blocks * BLOCK_BYTES;

    for (row, out) in output.iter_mut().enumerate().take(dim_out) {
        let w_row = &weights_packed[row * n_bytes..(row + 1) * n_bytes];
        let mut acc = vdupq_n_s32(0);

        for blk in 0..full_blocks {
            let b0 = blk * BLOCK_BYTES;
            let v = vld1q_u8(w_row.as_ptr().add(b0));

            // Bit-pair k of every byte. `vshrq_n_u8` shifts bytes
            // independently, so each byte's extracted pair is trivially its
            // own; the mask isolates bits 0..1. The shift amount is an
            // instruction immediate (and 0 is not a legal immediate), so the
            // four extractions are written out rather than looped.
            let codes = [
                vandq_u8(v, pair_mask),
                vandq_u8(vshrq_n_u8::<2>(v), pair_mask),
                vandq_u8(vshrq_n_u8::<4>(v), pair_mask),
                vandq_u8(vshrq_n_u8::<6>(v), pair_mask),
            ];

            for (k, &cd) in codes.iter().enumerate() {
                let w = vqtbl1q_s8(code_lut, cd);
                let a = vld1q_s8(lanes.as_ptr().add(k * n_bytes + b0));
                // vmulq_s8 IS ternary multiply for w in {-1, 0, +1} and
                // |a| <= 127 (the -128 case never reaches this path).
                let prod = vmulq_s8(a, w);
                // i8 -> i16 -> i32, exact: |prod| <= 127, |pair| <= 254,
                // each vpadalq lane step adds at most 508.
                let pairs = vpaddlq_s8(prod);
                acc = vpadalq_s16(acc, pairs);
            }
        }

        // Horizontal sum. Reassociation is exact for integers (module doc).
        let mut total = vaddvq_s32(acc);

        // Tail bytes, in the reference's own arithmetic.
        for (off, &b) in w_row[tail_start_b..n_bytes].iter().enumerate() {
            let base = 4 * (tail_start_b + off);
            for k in 0..4 {
                let code = (b >> (2 * k)) & 0b11;
                let wv: i32 = match code {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                total += wv * input[base + k] as i32;
            }
        }

        *out = total;
    }
}
