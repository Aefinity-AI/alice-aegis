//! AVX2 ternary matvec for CIS-1 integer semantics — bit-identical to
//! [`crate::cis::ternary_matvec_i8`].
//!
//! # Why this is allowed to exist
//!
//! CIS-1's reference TMV is a scalar loop. Measured on the Dell i5-5200U on
//! 2026-08-05 (`docs/hardware_logs/cis_vs_float_L_dell_BOOTLOG_2026-08-05.txt`,
//! 3 captures x 3 shapes): the integer path costs **1.248x** against *scalar*
//! float, but **5.75x** against the AVX2 float kernel. The gap is 4.61x of
//! absent SIMD, not integer semantics. This module closes that gap.
//!
//! # Why vectorising is SAFE here and is not for f32
//!
//! `cis::ternary_matvec_i8`'s own contract: *"No rounding exists in this op;
//! it is exact in ANY summation order — this is the property f32 lacks and the
//! only reason CIS-1 can exist."* Integer addition is associative, so lane
//! reassociation cannot change a single bit. The f32 kernels can never make
//! this claim, which is precisely why they need bit-exactness A/Bs and this
//! does not — it needs an equality assertion, which `tests/` carries.
//!
//! # This is NOT a re-proposal of a settled negative
//!
//! - not A6 (CTZ zero-skip): no data-dependent branching, no bit-scan.
//! - not A7 (T-MAC `pshufb` LUT-mpGEMM): that respecialised weights to
//!   **4 bits/weight** and was rejected on *memory traffic* (1.67x decode
//!   traffic on a bandwidth-bound path). This reads the **same 2-bit packing,
//!   byte for byte** — identical traffic to the incumbent. `pshufb` appears
//!   here only as a 4-entry code->value map, not as a product LUT.
//! - not A16 (fused dual/tri): one matvec, one output row at a time.
//! - not A17 (bitplane-dense): no bitplane re-encoding; the container is
//!   untouched.
//!
//! # Method
//!
//! `vpsignb` *is* ternary multiply: `sign_epi8(a, w)` yields `a` for `w=+1`,
//! `-a` for `w=-1`, and `0` for `w=0`. One instruction, no product table.
//!
//! A packed byte holds 4 weights in bit-pairs. Extracting bit-pair `k` from 32
//! consecutive bytes yields the weights for input elements `= k (mod 4)`, so
//! the activations are consumed in four stride-4 subsequences. Those are
//! deinterleaved **once per matvec** and reused across all `dim_out` rows, so
//! the cost is O(dim_in) against O(dim_out x dim_in) of work.
//!
//! Widening is exact at every step: products are in `[-127, 127]`, pair sums
//! in `[-254, 254]` (i16), quad sums in `[-508, 508]` (i32).
//!
//! # The `-128` hazard, handled rather than documented away
//!
//! `vpsignb` negates within i8, so `-(-128)` wraps to `-128`; the scalar
//! reference computes in i32 and yields `+128`. CIS-1 forbids `-128` by
//! construction (`quantize_activations_i32` clamps to `|q| <= 127`), but a
//! kernel whose entire value is bit-exactness must not be *conditionally*
//! bit-exact. The deinterleave pass — which already touches every activation —
//! detects any `-128` and routes the whole call to the scalar reference. The
//! guarantee is therefore unconditional for every possible input.

// Matches `ops.rs`: the AVX2 kernels in this crate carry their preconditions in
// a `# Safety` section on the function rather than per-intrinsic blocks. Every
// unsafe operation below is a load or an intrinsic covered by that contract.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::cis::{check_tmv_preconditions, ternary_matvec_i8};
use crate::cis_infer::{F, dot_i8_bf16q};
use alloc::vec;
use core::arch::x86_64::*;

/// Bytes of packed weights consumed per SIMD block (4 weights per byte).
const BLOCK_BYTES: usize = 32;

/// The loop-invariant broadcast vectors, hoisted once per call. Grouped so the
/// inner step takes a reference rather than four separate register arguments.
struct Consts {
    /// code -> value map applied with `pshufb`; `11` is defined-as-zero.
    code_lut: __m256i,
    /// isolates the low bit-pair of every byte.
    pair_mask: __m256i,
    /// unsigned 1s, the `maddubs` multiplicand that turns it into a pair-add.
    ones_u8: __m256i,
    /// signed 1s, the `madd` multiplicand that turns it into a quad-add.
    ones_i16: __m256i,
}

/// AVX2 TMV. Byte-identical to [`crate::cis::ternary_matvec_i8`] for every
/// input, falling back to it when AVX2 is unavailable, when the shape is too
/// small to block, or when the activations contain `-128`.
pub fn ternary_matvec_i8_avx2(
    output: &mut [i32],
    input: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
) {
    // Enforced here, not only on the fallback: otherwise an illegal call is
    // rejected or silently answered depending on the CPU and the shape.
    check_tmv_preconditions(output, input, weights_packed, dim_out, dim_in);
    let n_bytes = dim_in / 4;

    // `simd_on`, not `avx2_active`: every other dispatch site in the crate
    // honours the set_force_scalar race toggle, and the in-boot same-binary A/B
    // depends on that toggle reaching every path.
    if !crate::ops::simd_on() || n_bytes < BLOCK_BYTES {
        ternary_matvec_i8(output, input, weights_packed, dim_out, dim_in);
        return;
    }

    // Deinterleave activations into the four stride-4 subsequences the bit-pair
    // extraction consumes, and detect the -128 hazard in the same pass.
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

    // SAFETY: `simd_on()` confirmed AVX2 is supported and OS-enabled, which is
    // this function's only CPU precondition. Shapes are re-checked inside:
    // `n_bytes >= BLOCK_BYTES` holds, `lanes` is exactly `4 * n_bytes` long, and
    // the callee bounds every weight and output access against `dim_out` and
    // `n_bytes` before dereferencing.
    unsafe { tmv_i8_avx2(output, input, &lanes, weights_packed, dim_out, n_bytes) }
}

/// # Safety
/// AVX2 must be supported and OS-enabled (see [`crate::ops::avx2_active`]).
/// `lanes` must be the stride-4 deinterleave of `input`, `4 * n_bytes` long,
/// and `input` must contain no `i8::MIN`.
#[target_feature(enable = "avx2")]
unsafe fn tmv_i8_avx2(
    output: &mut [i32],
    input: &[i8],
    lanes: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    n_bytes: usize,
) {
    // code -> value, applied with `pshufb`. Codes are 0..=3; `11` is
    // defined-as-zero, matching `cis::wcode` and its golden vectors.
    let c = Consts {
        code_lut: _mm256_setr_epi8(
            0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // low 128-bit lane
            0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // high 128-bit lane
        ),
        pair_mask: _mm256_set1_epi8(0b11),
        ones_u8: _mm256_set1_epi8(1),
        ones_i16: _mm256_set1_epi16(1),
    };

    let full_blocks = n_bytes / BLOCK_BYTES;
    let tail_start_b = full_blocks * BLOCK_BYTES;

    for (row, out) in output.iter_mut().enumerate().take(dim_out) {
        let w_row = &weights_packed[row * n_bytes..(row + 1) * n_bytes];
        let mut acc = _mm256_setzero_si256();

        for blk in 0..full_blocks {
            let b0 = blk * BLOCK_BYTES;
            let v = _mm256_loadu_si256(w_row.as_ptr().add(b0) as *const __m256i);

            // Bit-pair k of every byte. `srli_epi16` shifts 16-bit lanes, so
            // bits from the high byte migrate into the low byte — but only into
            // positions >= 8 - 2k >= 2, which `pair_mask` (bits 0..1) discards.
            // Each byte's extracted pair is therefore its own.
            //
            // The shift is an instruction immediate, so k must be const: the
            // four steps are const-generic rather than a runtime loop.
            // A nested item does NOT inherit the outer `target_feature`. If this
            // were ever emitted out of line it would be a featureless fn taking
            // __m256i by value from an +avx2 caller — the classic AVX ABI
            // mismatch, silently wrong. It inlines today; after A14 this
            // programme does not rest a correctness guarantee on that.
            #[target_feature(enable = "avx2")]
            unsafe fn step<const SHIFT: i32, const K: usize>(
                v: __m256i,
                lanes: &[i8],
                n_bytes: usize,
                b0: usize,
                c: &Consts,
            ) -> __m256i {
                let codes = _mm256_and_si256(_mm256_srli_epi16::<SHIFT>(v), c.pair_mask);
                let w = _mm256_shuffle_epi8(c.code_lut, codes);
                let a = _mm256_loadu_si256(lanes.as_ptr().add(K * n_bytes + b0) as *const __m256i);
                // vpsignb IS ternary multiply: +1 -> a, -1 -> -a, 0 -> 0.
                let prod = _mm256_sign_epi8(a, w);
                // i8 -> i16 -> i32, exact: |prod| <= 127, |pair| <= 254.
                let pairs = _mm256_maddubs_epi16(c.ones_u8, prod);
                _mm256_madd_epi16(pairs, c.ones_i16)
            }

            let q0 = step::<0, 0>(v, lanes, n_bytes, b0, &c);
            let q1 = step::<2, 1>(v, lanes, n_bytes, b0, &c);
            let q2 = step::<4, 2>(v, lanes, n_bytes, b0, &c);
            let q3 = step::<6, 3>(v, lanes, n_bytes, b0, &c);
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(q0, q1));
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(q2, q3));
        }

        // Horizontal sum. Reassociation is exact for integers (see module doc).
        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256(acc, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        let mut total = _mm_cvtsi128_si32(s);

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

// ---------------------------------------------------------------------------
// LM-head dot: i8 activations against an on-the-fly BF16->Q.F weight row.
// ---------------------------------------------------------------------------

/// Elements (bf16 weights) consumed per SIMD block. Each block is processed
/// as 4 lanes of `i64` — see the module-level doc below for why 64-bit lanes
/// are load-bearing here, not just a width choice.
const DOT_BLOCK: usize = 4;

/// AVX2 dot of i8 activations against a BF16 table row, converting each BF16
/// element to Q.F on the fly. Byte-identical to
/// [`crate::cis_infer::dot_i8_bf16q`] for every input, falling back to it
/// when AVX2 is unavailable, when the row is too short to block, or when any
/// element hits a condition `dot_i8_bf16q` itself would reject (inf/nan,
/// `bf16_to_fixed`'s magnitude bound, or the Q.F i32-range assert) — in which
/// case the *entire row* is recomputed by the scalar reference so the panic
/// (message and occurrence) is identical, not merely "close".
///
/// # Method
///
/// `bf16_to_fixed(bits, F)` is:
/// 1. unpack sign / exp / man from the 16 bits;
/// 2. `(m, e)` = subnormal `(man, 1-127-7)` or normal `(man|0x80, exp-127-7)`;
/// 3. `sh = e + F`; if `sh >= 0`, `v = m << sh`; elif `-sh >= 63`, `v = 0`;
///    else `v = rne_shr(m, -sh)` (banker's-rounding right shift);
/// 4. negate if the sign bit was set.
///
/// This is reproduced exactly, lanewise, in 64-bit integer SIMD (4 lanes per
/// `__m256i`): `m` needs up to 8 bits and `sh` can be as large as 36 (the
/// `bf16_to_fixed` assert bound), so `m << sh` can reach ~44 bits — it does
/// NOT fit a 32-bit lane, which is why this kernel uses `epi64` lanes
/// throughout rather than the `epi32` lanes `ternary_matvec_i8_avx2` uses.
/// Variable per-lane shifts (`vpsllvq`/`vpsrlvq`) and per-lane compares
/// (`vpcmpeqq`/`vpcmpgtq`) plus `vpblendvb` (whose byte mask is uniform
/// within each 8-byte lane group here, since every predicate comes from a
/// 64-bit compare) implement the branches as selects instead of divergent
/// control flow — same value, same order of operations as the scalar
/// reference, just computed for 4 rows... no, 4 *elements of one row* at a
/// time (the dot is a single row's reduction, unlike the TMV kernels which
/// parallelize across output rows).
///
/// The final product `x as i64 * w` is computed with `_mm256_mul_epi32`
/// (signed 32x32->64, taking the low 32 bits of each 64-bit input lane) —
/// exact here because `x` is `i8` (fits in the low 32 bits of its
/// sign-extended 64-bit lane) and `w`'s magnitude is asserted `< 2^31`
/// before this point, so its low 32 bits are its exact two's-complement
/// value. Accumulation is `i64` addition, associative and exact (same bound
/// `dot_i8_bf16q` documents: `127 * 2^31 * 4096 < 2^51`).
pub fn dot_i8_bf16q_avx2(a: &[i8], row: &[u8]) -> i64 {
    debug_assert_eq!(a.len() * 2, row.len());
    let n = a.len();

    if !crate::ops::simd_on() || n < DOT_BLOCK {
        return dot_i8_bf16q(a, row);
    }

    // SAFETY: `simd_on()` confirmed AVX2 is supported and OS-enabled, which
    // is this function's only CPU precondition. `n >= DOT_BLOCK` holds, and
    // `dot_i8_bf16q_avx2_inner` bounds every access against `n` and `row`
    // before dereferencing, falling back to the scalar reference (which
    // re-validates from scratch) whenever any element fails a precondition
    // the scalar path would have rejected.
    unsafe { dot_i8_bf16q_avx2_inner(a, row, n) }
}

/// # Safety
/// AVX2 must be supported and OS-enabled (see [`crate::ops::avx2_active`]).
/// `row` must be `2 * n` bytes and `a` must be `n` elements, with `n >=
/// DOT_BLOCK`.
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_bf16q_avx2_inner(a: &[i8], row: &[u8], n: usize) -> i64 {
    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi64x(1);
    let c_7f = _mm256_set1_epi64x(0x7F);
    let c_80 = _mm256_set1_epi64x(0x80);
    let c_8000 = _mm256_set1_epi64x(0x8000);
    let c_ff = _mm256_set1_epi64x(0xFF);
    let c_134 = _mm256_set1_epi64x(134);
    let c_neg133 = _mm256_set1_epi64x(-133);
    let c_frac = _mm256_set1_epi64x(F as i64);
    let c_36 = _mm256_set1_epi64x(36);
    let c_62 = _mm256_set1_epi64x(62);
    let c_max_i32 = _mm256_set1_epi64x((1i64 << 31) - 1);

    let full_blocks = n / DOT_BLOCK;
    let tail_start = full_blocks * DOT_BLOCK;

    let mut acc = zero;
    for blk in 0..full_blocks {
        let base = blk * DOT_BLOCK;

        // Load 4 BF16 (u16) values, zero-extended to i64 lanes, and 4 i8
        // activations, sign-extended to i64 lanes.
        let bits16 = _mm_loadl_epi64(row.as_ptr().add(base * 2) as *const __m128i);
        let bits = _mm256_cvtepu16_epi64(bits16);
        let xs8 = _mm_cvtsi32_si128(i32::from_le_bytes([
            a[base] as u8,
            a[base + 1] as u8,
            a[base + 2] as u8,
            a[base + 3] as u8,
        ]));
        let xs = _mm256_cvtepi8_epi64(xs8);

        let sign_set = _mm256_cmpgt_epi64(_mm256_and_si256(bits, c_8000), zero);
        let exp = _mm256_and_si256(_mm256_srli_epi64::<7>(bits), c_ff);
        let man = _mm256_and_si256(bits, c_7f);
        let exp_is_zero = _mm256_cmpeq_epi64(exp, zero);
        let exp_is_ff = _mm256_cmpeq_epi64(exp, c_ff);

        let man_or_80 = _mm256_or_si256(man, c_80);
        let m = _mm256_blendv_epi8(man_or_80, man, exp_is_zero);
        let e_normal = _mm256_sub_epi64(exp, c_134);
        let e = _mm256_blendv_epi8(e_normal, c_neg133, exp_is_zero);
        let sh = _mm256_add_epi64(e, c_frac);
        let sh_gt36 = _mm256_cmpgt_epi64(sh, c_36);

        // Any inf/nan or out-of-range magnitude: bail to the scalar
        // reference for the WHOLE ROW so its panic (message + occurrence)
        // is reproduced exactly rather than approximated.
        let invalid_early = _mm256_or_si256(exp_is_ff, sh_gt36);
        if _mm256_movemask_epi8(invalid_early) != 0 {
            return dot_i8_bf16q(a, row);
        }

        let sh_ge0 = _mm256_cmpgt_epi64(sh, _mm256_set1_epi64x(-1));

        // Left-shift branch (sh >= 0): shift amount clamped to 0 when unused
        // so the variable shift never sees a synthesized negative count.
        let shl_amt = _mm256_blendv_epi8(zero, sh, sh_ge0);
        let left_val = _mm256_sllv_epi64(m, shl_amt);

        // Right-shift (RNE) branch (sh < 0): k = -sh.
        let k_raw = _mm256_sub_epi64(zero, sh);
        let k_ge63 = _mm256_cmpgt_epi64(k_raw, c_62);
        // Clamp the shift operand to 62 when it would be discarded anyway
        // (k>=63 case), keeping every shift amount in the valid 0..=62 range.
        let k = _mm256_blendv_epi8(k_raw, c_62, k_ge63);
        let floor = _mm256_srlv_epi64(m, k);
        let low_mask = _mm256_sub_epi64(_mm256_sllv_epi64(one, k), one);
        let rem = _mm256_and_si256(m, low_mask);
        let half = _mm256_sllv_epi64(one, _mm256_sub_epi64(k, one));
        let rem_gt_half = _mm256_cmpgt_epi64(rem, half);
        let rem_eq_half = _mm256_cmpeq_epi64(rem, half);
        let floor_odd = _mm256_cmpeq_epi64(_mm256_and_si256(floor, one), one);
        let round_up = _mm256_or_si256(rem_gt_half, _mm256_and_si256(rem_eq_half, floor_odd));
        let rne_val = _mm256_add_epi64(floor, _mm256_blendv_epi8(zero, one, round_up));
        let right_val = _mm256_blendv_epi8(rne_val, zero, k_ge63);

        let v_mag = _mm256_blendv_epi8(right_val, left_val, sh_ge0);
        let v_neg = _mm256_sub_epi64(zero, v_mag);
        let v = _mm256_blendv_epi8(v_mag, v_neg, sign_set);

        // `dot_i8_bf16q`'s own bound: |w| < 2^31. Same whole-row fallback
        // discipline as the inf/nan/magnitude checks above.
        let v_is_neg = _mm256_cmpgt_epi64(zero, v);
        let v_abs = _mm256_blendv_epi8(v, _mm256_sub_epi64(zero, v), v_is_neg);
        let out_of_range = _mm256_cmpgt_epi64(v_abs, c_max_i32);
        if _mm256_movemask_epi8(out_of_range) != 0 {
            return dot_i8_bf16q(a, row);
        }

        // Exact 32x32->64 signed multiply: both operands' low 32 bits are
        // their true two's-complement value (x is i8; |v| < 2^31).
        let prod = _mm256_mul_epi32(xs, v);
        acc = _mm256_add_epi64(acc, prod);
    }

    // Horizontal sum of the 4 i64 lanes.
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256(acc, 1);
    let s = _mm_add_epi64(lo, hi);
    let s_hi = _mm_unpackhi_epi64(s, s);
    let s = _mm_add_epi64(s, s_hi);
    let mut total = _mm_cvtsi128_si64(s);

    // Tail elements, in the reference's own arithmetic (including its own
    // asserts, so an out-of-range tail element panics exactly as the scalar
    // reference would).
    for i in tail_start..n {
        let bits = u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]);
        let w = crate::cis_infer::bf16_to_fixed(bits, F);
        assert!(
            w.unsigned_abs() < 1 << 31,
            "dot_i8_bf16q: value out of Q.{F} i32 range"
        );
        total += a[i] as i64 * w;
    }

    total
}
