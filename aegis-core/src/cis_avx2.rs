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
//! Decode is by **nibble**, not by bit-pair: a byte's low nibble packs
//! bit-pairs 0 and 1, its high nibble packs bit-pairs 2 and 3. Two `pshufb`
//! lookups against the same nibble (one LUT keyed on the nibble's low
//! bit-pair, one on its high bit-pair) decode both bit-pairs in that nibble,
//! so extracting the nibble once (an `and`, or a `srli_epi16` + `and` for the
//! high nibble) feeds two decodes instead of one. This halves the
//! shift/mask instruction count of extracting four bit-pairs individually
//! (3 instructions for both nibbles vs. 8 for four separate `srli_epi16` +
//! `and` bit-pair extractions) while issuing the same four `pshufb`s.
//!
//! Widening is exact at every step. **v3b** decodes weights offset by one,
//! `u = w + 1 in {0, 1, 2}` (u8), instead of `w in {-1, 0, +1}` (i8). Since
//! `sum_j u_j * a_j = sum_j w_j * a_j + sum_j a_j`, the dot product is
//! `acc - sum_a`, where `sum_a` (the sum of activations over the full-block
//! range) is computed once per call and subtracted once per row after the
//! horizontal sum. This lets `vpmaddubsw(u, a)` compute the pair sum directly
//! (`u` unsigned, `a` signed — exactly the operand types `vpmaddubsw`
//! requires) with **no `vpsignb` negate step**: on Broadwell `vpsignb`,
//! `vpmaddubsw`, and `vpsrlw` are all port-0-only, so removing `vpsignb` drops
//! 4 port-0 uops per block (9 -> 5 per 32-byte block; ~20 vs ~24 total).
//!
//! Products are now `u * a` with `u in [0, 2]`, `a in [-127, 127]`, so
//! `|u * a| <= 2 * 127 = 254` and a pair sum is in `[-508, 508]` (i16) — twice
//! the old bound (`vpmaddubsw` itself only saturates past `+-32767`, so no
//! individual pair sum can saturate). `tmv_i8_avx2` (v3b) accumulates up to
//! `FLUSH_BLOCKS` pair sums per i16 lane before widening to i32 —
//! `FLUSH_BLOCKS * 508 < i16::MAX` is asserted at compile time, so the i16
//! accumulator can never overflow.
//!
//! This offset identity is exact and order-independent (integer addition
//! associates), so `tmv_i8_avx2` remains bit-identical to the scalar
//! reference for every legal input — same guarantee as v3, cheaper kernel.
//!
//! # The `-128` hazard, handled rather than documented away
//!
//! v3b has no negate step, so `u * (-128) in {0, -128, -256}` is exact with no
//! wraparound hazard on its own. The `-128` guard is kept anyway (scope
//! discipline: this change is the offset-weight identity, not a re-audit of
//! the hazard path) — CIS-1 forbids `-128` by construction
//! (`quantize_activations_i32` clamps to `|q| <= 127`), and the deinterleave
//! pass, which already touches every activation, detects any `-128` and
//! routes the whole call to the scalar reference regardless. A follow-up
//! could drop this fallback now that it is provably unnecessary here.

// Matches `ops.rs`: the AVX2 kernels in this crate carry their preconditions in
// a `# Safety` section on the function rather than per-intrinsic blocks. Every
// unsafe operation below is a load or an intrinsic covered by that contract.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::cis::{check_tmv_preconditions, ternary_matmul_i8, ternary_matvec_i8};
use crate::cis_infer::{F, dot_i8_bf16q};
use alloc::vec;
use core::arch::x86_64::*;

/// Bytes of packed weights consumed per SIMD block (4 weights per byte).
const BLOCK_BYTES: usize = 32;

/// The loop-invariant broadcast vectors, hoisted once per call. Grouped so the
/// inner step takes a reference rather than four separate register arguments.
struct Consts {
    /// nibble (low 2 bits = bit-pair 0) -> value map applied with `pshufb`;
    /// `11` is defined-as-zero. Indexed by a whole nibble so one `pshufb`
    /// decodes bit-pair 0 (or 2) of every byte in a single instruction; see
    /// `code_lut_hi` for bit-pair 1 (or 3) of the same nibble.
    code_lut_lo: __m256i,
    /// nibble -> value map for the *high* bit-pair of the nibble (bit-pair 1
    /// within the low nibble, bit-pair 3 within the high nibble).
    code_lut_hi: __m256i,
    /// isolates the low nibble of every byte.
    nibble_mask: __m256i,
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
// See `dot_i8_bf16q_avx2_wide`'s own `#[allow(unused_assignments)]` doc
// (below): the `flush!()` macro's final call resets `blocks_since_flush` to 0
// even though nothing reads it afterward — correct (the reset is what makes
// it safe to call `flush!()` unconditionally at both the periodic point and
// after the loop), just spuriously flagged by `unused_assignments`.
#[allow(unused_assignments)]
#[target_feature(enable = "avx2")]
unsafe fn tmv_i8_avx2(
    output: &mut [i32],
    input: &[i8],
    lanes: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    n_bytes: usize,
) {
    // Nibble -> two values, applied with `pshufb`. A byte's low nibble packs
    // bit-pairs 0 and 1; its high nibble packs bit-pairs 2 and 3. Indexing a
    // 16-entry LUT by the whole nibble decodes one bit-pair per `pshufb`, the
    // same instruction count as decoding by 2-bit code (`code_lut` in the
    // prior version of this kernel), but the *nibble extraction* shared by
    // both LUT lookups costs 3 instructions total (one `and` for the low
    // nibble, one `srli_epi16` + `and` for the high nibble) versus the prior
    // per-bit-pair `srli_epi16` + `and` (2 instructions x 4 = 8). Codes are
    // 0..=3; `11` is defined-as-zero, matching `cis::wcode` and its golden
    // vectors.
    let c = Consts {
        // index n -> wcode(n & 0b11) + 1: period-4 repeat over the 4-bit
        // index, offset by one so the decoded value is the unsigned
        // `u = w + 1 in {0, 1, 2}` v3b's `vpmaddubsw` consumes directly (code
        // 1 -> 2, code 2 -> 0, codes 0 and 3 -> 1; `11` is still
        // defined-as-zero, i.e. `u = 1`).
        code_lut_lo: _mm256_setr_epi8(
            1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, // low 128-bit lane
            1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, // high 128-bit lane
        ),
        // index n -> wcode((n >> 2) & 0b11) + 1: four values, each held for 4
        // consecutive indices, offset by one exactly as `code_lut_lo` above.
        code_lut_hi: _mm256_setr_epi8(
            1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, // low 128-bit lane
            1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, // high 128-bit lane
        ),
        nibble_mask: _mm256_set1_epi8(0x0F),
        ones_i16: _mm256_set1_epi16(1),
    };

    let full_blocks = n_bytes / BLOCK_BYTES;
    let tail_start_b = full_blocks * BLOCK_BYTES;

    // v3b offset identity: sum_j u_j * a_j = sum_j w_j * a_j + sum_j a_j, so
    // dot = acc - sum_a. `sum_a` runs over exactly the activations the block
    // loop below covers (elements `0..4*tail_start_b`); the tail loop below
    // is unaffected (it uses `w` directly, not `u`).
    let sum_a: i32 = input[0..4 * tail_start_b].iter().map(|&x| x as i32).sum();

    // Decode all four bit-pairs of `v` at once: `nibble_mask` (bits 0..3)
    // isolates each byte's own low nibble directly (no cross-byte bleed,
    // since `and` doesn't shift). For the high nibble, `srli_epi16` shifts
    // 16-bit lanes right by 4: within one byte this brings bits 4..7 down to
    // 0..3 (what we want); the bits that shift in from the neighbouring byte
    // land at position >= 4, which `nibble_mask` discards. Each byte's high
    // nibble is therefore its own, by the same argument the prior version of
    // this kernel made for its 2-bit `pair_mask`.
    //
    // A nested item does NOT inherit the outer `target_feature`. If this were
    // ever emitted out of line it would be a featureless fn taking __m256i by
    // value from an +avx2 caller — the classic AVX ABI mismatch, silently
    // wrong. It inlines today; after A14 this programme does not rest a
    // correctness guarantee on that.
    #[target_feature(enable = "avx2")]
    unsafe fn decode(v: __m256i, c: &Consts) -> [__m256i; 4] {
        let lo_nib = _mm256_and_si256(v, c.nibble_mask);
        let hi_nib = _mm256_and_si256(_mm256_srli_epi16::<4>(v), c.nibble_mask);
        [
            _mm256_shuffle_epi8(c.code_lut_lo, lo_nib), // bit-pair 0
            _mm256_shuffle_epi8(c.code_lut_hi, lo_nib), // bit-pair 1
            _mm256_shuffle_epi8(c.code_lut_lo, hi_nib), // bit-pair 2
            _mm256_shuffle_epi8(c.code_lut_hi, hi_nib), // bit-pair 3
        ]
    }

    // v3b: widen once per row instead of once per block, same as v3, but with
    // no `vpsignb` negate step at all. `decode` now yields the *offset*
    // weight `u = w + 1 in {0, 1, 2}` (unsigned) directly, so `pair_sum` is a
    // single `vpmaddubsw(u, a)` — `u` unsigned, `a` signed, exactly the
    // operand types the instruction wants — computing
    // `u0*a0 + u1*a1 = (w0+1)*a0 + (w1+1)*a1` per adjacent pair. Per block
    // this is 1 activation load + 1 `vpmaddubsw` + 1 `vpaddw`
    // (`vpaddw` is port 1/5, not port 0): only `vpmaddubsw` is port-0-only per
    // k, so 4 port-0 uops per block from this loop (plus 1 for the decode's
    // `vpsrlw`) = 5 port-0 uops per block, vs v3's 9 (`vpsignb` +
    // `vpmaddubsw` per k, plus the decode `vpsrlw`).
    //
    // Since `sum_j u_j*a_j = sum_j w_j*a_j + sum_j a_j`, the true dot product
    // is `acc - sum_a` (see `sum_a` above) — computed once per row after the
    // horizontal sum, not per block.
    //
    // i16 accumulation cannot run forever: each pair sum is now in
    // `[-508, 508]` (`u in [0, 2]`, `a in [-127, 127]`, so `|u*a| <= 254` and
    // a pair sum is `<= 2 * 254 = 508`; `vpmaddubsw` itself only saturates
    // past `+-32767`, so no individual pair sum saturates). Summing
    // `FLUSH_BLOCKS` of them into one i16 lane must stay inside
    // `[i16::MIN, i16::MAX]`. `FLUSH_BLOCKS * 508 < i16::MAX` is checked at
    // compile time below (`64 * 508 = 32512 < 32767`), so no i16 accumulator
    // can ever overflow regardless of input.
    const FLUSH_BLOCKS: usize = 64;
    const _: () = assert!((FLUSH_BLOCKS * 508) < i16::MAX as usize);

    #[target_feature(enable = "avx2")]
    unsafe fn pair_sum(a: __m256i, u: __m256i, _c: &Consts) -> __m256i {
        // u8 (u, decoded offset weight) x i8 (a, activation) -> i16
        // adjacent-pair sum, exact: |u*a| <= 254, |pair| <= 508.
        _mm256_maddubs_epi16(u, a)
    }

    for (row, out) in output.iter_mut().enumerate().take(dim_out) {
        let w_row = &weights_packed[row * n_bytes..(row + 1) * n_bytes];
        let mut acc = _mm256_setzero_si256();
        let mut acc16 = [_mm256_setzero_si256(); 4];
        let mut blocks_since_flush = 0usize;

        // Widens all four i16 pair-sum accumulators into `acc` (i32) and
        // resets them. Safe to call on an empty run (adds zero) so it can
        // double as both the periodic flush and the final, possibly-partial
        // one after the loop.
        macro_rules! flush {
            () => {
                for k in 0..4 {
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(acc16[k], c.ones_i16));
                    acc16[k] = _mm256_setzero_si256();
                }
                blocks_since_flush = 0;
            };
        }

        for blk in 0..full_blocks {
            let b0 = blk * BLOCK_BYTES;
            let v = _mm256_loadu_si256(w_row.as_ptr().add(b0) as *const __m256i);
            let w = decode(v, &c);

            for k in 0..4 {
                let a = _mm256_loadu_si256(lanes.as_ptr().add(k * n_bytes + b0) as *const __m256i);
                let pair = pair_sum(a, w[k], &c);
                acc16[k] = _mm256_add_epi16(acc16[k], pair);
            }
            blocks_since_flush += 1;
            if blocks_since_flush == FLUSH_BLOCKS {
                flush!();
            }
        }
        flush!();

        // Horizontal sum. Reassociation is exact for integers (see module doc).
        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256(acc, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        // v3b offset identity: acc = sum_j (w_j+1)*a_j over the full-block
        // range = sum_j w_j*a_j + sum_a, so the true dot needs `- sum_a` once.
        let mut total = _mm_cvtsi128_si32(s) - sum_a;

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
// Batched prefill: ternary GEMM over an 8-token tile (v3b decode reused).
// ---------------------------------------------------------------------------

/// AVX2 ternary GEMM for batched prefill: `n_tok` tokens' activations
/// against ONE packed ternary weight matrix, reading each packed weight
/// block ONCE per `TOK_TILE`-token tile instead of once per token (as
/// `n_tok` independent calls to [`ternary_matvec_i8_avx2`] would cost).
/// Byte-identical to [`crate::cis::ternary_matmul_i8`] for every input —
/// that function IS this kernel's definition (a per-token loop over
/// [`ternary_matvec_i8`]); integer addition/multiplication associate and
/// distribute exactly, so decoding a weight block once and applying it to
/// several tokens instead of redecoding it per token changes nothing but
/// which order the same additions happen in. Falls back to the scalar
/// reference under the same three conditions [`ternary_matvec_i8_avx2`]
/// does: AVX2 unavailable/disabled, the shape too small to block, or any
/// activation hitting the `-128` hazard (checked across ALL tokens, so the
/// whole call — never a per-token mix of kernel and reference — routes to
/// the reference).
///
/// # Reuse argument
///
/// Per weight block, the single-token kernel issues 1 activation load + 1
/// `vpmaddubsw` (port-0-only) + 1 `vpaddw` per row, i.e. it reads the block
/// from memory once per output row it's needed for — but needs the SAME
/// block once per *token* it processes, since each token is a separate
/// call. Prefilling `TOK_TILE = 8` tokens the old way reads a block 8 times
/// (once per token's call); this kernel decodes the block once (4
/// `pshufb` + 2 mask ops) and issues one `vpmaddubsw` per token against the
/// SAME decoded weight vectors — 4 port-0 `vpmaddubsw` uops per
/// token-block (~4 cycles per token-block on Broadwell, where `vpmaddubsw`
/// is port-0-only) against the ~9.7 cycles per token-block the pre-v3b
/// single-token kernel needed, while the packed-weight BYTE traffic drops
/// `TOK_TILE`x: the lever this kernel exists for is memory bandwidth, and
/// this is where it comes from — decode cost is now amortized over 8
/// tokens instead of paid once per token.
pub fn ternary_matmul_i8_avx2(
    outputs: &mut [i32],
    inputs: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
    n_tok: usize,
) {
    assert!(
        outputs.len() >= n_tok * dim_out,
        "outputs shorter than n_tok*dim_out"
    );
    assert!(
        inputs.len() >= n_tok * dim_in,
        "inputs shorter than n_tok*dim_in"
    );
    if n_tok == 0 {
        return;
    }
    // Per-token preconditions are identical for every token (same dim_out,
    // dim_in, weights_packed): checking once against token 0's slices
    // reproduces the same panic (message and occurrence) the scalar
    // reference's per-token loop would hit on its own first iteration.
    check_tmv_preconditions(
        &outputs[..dim_out],
        &inputs[..dim_in],
        weights_packed,
        dim_out,
        dim_in,
    );
    let n_bytes = dim_in / 4;

    if !crate::ops::simd_on() || n_bytes < BLOCK_BYTES {
        ternary_matmul_i8(outputs, inputs, weights_packed, dim_out, dim_in, n_tok);
        return;
    }

    // Deinterleave every token's activations into the four stride-4
    // subsequences the bit-pair extraction consumes (same per-token layout
    // `ternary_matvec_i8_avx2` uses), and detect the -128 hazard across ALL
    // tokens in the same pass.
    let mut lanes = vec![0i8; n_tok * 4 * n_bytes];
    let mut has_min = false;
    for t in 0..n_tok {
        let inp = &inputs[t * dim_in..t * dim_in + dim_in];
        let out_lanes = &mut lanes[t * 4 * n_bytes..(t + 1) * 4 * n_bytes];
        for j in 0..n_bytes {
            for k in 0..4 {
                let v = inp[4 * j + k];
                has_min |= v == i8::MIN;
                out_lanes[k * n_bytes + j] = v;
            }
        }
    }
    if has_min {
        ternary_matmul_i8(outputs, inputs, weights_packed, dim_out, dim_in, n_tok);
        return;
    }

    // SAFETY: `simd_on()` confirmed AVX2 is supported and OS-enabled, which
    // is this function's only CPU precondition. Shapes are re-checked
    // inside: `n_bytes >= BLOCK_BYTES` holds, `lanes` is exactly
    // `n_tok * 4 * n_bytes` long, and the callee bounds every weight,
    // activation, and output access against `dim_out`/`n_bytes`/`n_tok`
    // before dereferencing.
    unsafe {
        tmm_i8_avx2(
            outputs,
            inputs,
            &lanes,
            weights_packed,
            dim_out,
            dim_in,
            n_bytes,
            n_tok,
        )
    }
}

/// Tokens processed per tile. Register budget: `TOK_TILE` per-token i16
/// accumulators (one ymm each) + 4 decoded weight vectors (`w[0..4]`,
/// decoded once per block and shared across the whole tile) + 1 constant
/// (`ones_i16`) = 13 ymm, inside Broadwell's 16 architectural ymm
/// registers with headroom for the compiler's own scheduling.
const TOK_TILE: usize = 8;

/// Blocks a token's i16 accumulator sums before widening to i32 (mirrors
/// `tmv_i8_avx2`'s `FLUSH_BLOCKS`, at a different bound because this
/// kernel's per-block per-token contribution is larger). A block's
/// contribution to one token's i16 accumulator is the sum of that token's
/// four lane pair-sums (`vpmaddubsw(u, a)`, `u in [0, 2]`, `a in
/// [-127, 127]`), each `<= 508` in magnitude (`tmv_i8_avx2`'s bound), so
/// `<= 4 * 508 = 2032` per block per token. `FLUSH_BLOCKS_MM * 2032` must
/// stay under `i16::MAX` for the i16 accumulator to never overflow:
/// `16 * 2032 = 32512 < 32767`.
const FLUSH_BLOCKS_MM: usize = 16;
const _: () = assert!((FLUSH_BLOCKS_MM * 2032) < i16::MAX as usize);

/// # Safety
/// AVX2 must be supported and OS-enabled (see [`crate::ops::avx2_active`]).
/// `lanes` must be the per-token stride-4 deinterleave of `inputs`
/// (`n_tok * 4 * n_bytes` long, token `t`'s deinterleave at
/// `lanes[t*4*n_bytes..]`), and `inputs` must contain no `i8::MIN`.
#[target_feature(enable = "avx2")]
unsafe fn tmm_i8_avx2(
    outputs: &mut [i32],
    inputs: &[i8],
    lanes: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
    n_bytes: usize,
    n_tok: usize,
) {
    // Same LUTs/constants as `tmv_i8_avx2` — see that function's doc for
    // the derivation; duplicated here (not shared via `Consts::new()`)
    // because `Consts` has no constructor and this keeps the two kernels
    // textually independent, which is deliberate: neither should have to
    // change because the other's internal layout changed.
    let c = Consts {
        code_lut_lo: _mm256_setr_epi8(
            1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, // low 128-bit lane
            1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, // high 128-bit lane
        ),
        code_lut_hi: _mm256_setr_epi8(
            1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, // low 128-bit lane
            1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, // high 128-bit lane
        ),
        nibble_mask: _mm256_set1_epi8(0x0F),
        ones_i16: _mm256_set1_epi16(1),
    };

    let full_blocks = n_bytes / BLOCK_BYTES;
    let tail_start_b = full_blocks * BLOCK_BYTES;

    // Per-token `sum_a` (v3b offset-identity correction, see
    // `tmv_i8_avx2`): computed once per call, over exactly the
    // block-covered activation range for that token.
    let mut sum_a = vec![0i32; n_tok];
    for (t, sa) in sum_a.iter_mut().enumerate() {
        let base = t * dim_in;
        *sa = inputs[base..base + 4 * tail_start_b]
            .iter()
            .map(|&x| x as i32)
            .sum();
    }

    // Dispatch each tile to a MONOMORPHIC `tmm_tile::<M>` instantiation
    // (`M` a compile-time constant, not the runtime `tile_len` the old
    // single generic-length body used). This matters: with a runtime
    // bound, `[__m256i; TOK_TILE]` accumulator arrays can't be proven to
    // stay in registers across the block loop (LLVM can't unroll a
    // runtime-trip-count inner loop over 8 array slots), so they spill to
    // the stack and every token-block does load+add+store instead of a
    // register add — silently defeating the whole point of decoding a
    // block once per tile. A `const M` per call site lets LLVM unroll
    // `for ti in 0..M` fully and keep all M accumulators in ymm registers
    // for the instantiation's whole lifetime.
    let mut tile_start = 0usize;
    while tile_start < n_tok {
        let tile_len = (n_tok - tile_start).min(TOK_TILE);
        match tile_len {
            8 => tmm_tile::<8>(
                outputs,
                inputs,
                lanes,
                weights_packed,
                dim_out,
                n_bytes,
                tile_start,
                &sum_a,
                &c,
                full_blocks,
                tail_start_b,
                dim_in,
            ),
            7 => tmm_tile::<7>(
                outputs,
                inputs,
                lanes,
                weights_packed,
                dim_out,
                n_bytes,
                tile_start,
                &sum_a,
                &c,
                full_blocks,
                tail_start_b,
                dim_in,
            ),
            6 => tmm_tile::<6>(
                outputs,
                inputs,
                lanes,
                weights_packed,
                dim_out,
                n_bytes,
                tile_start,
                &sum_a,
                &c,
                full_blocks,
                tail_start_b,
                dim_in,
            ),
            5 => tmm_tile::<5>(
                outputs,
                inputs,
                lanes,
                weights_packed,
                dim_out,
                n_bytes,
                tile_start,
                &sum_a,
                &c,
                full_blocks,
                tail_start_b,
                dim_in,
            ),
            4 => tmm_tile::<4>(
                outputs,
                inputs,
                lanes,
                weights_packed,
                dim_out,
                n_bytes,
                tile_start,
                &sum_a,
                &c,
                full_blocks,
                tail_start_b,
                dim_in,
            ),
            3 => tmm_tile::<3>(
                outputs,
                inputs,
                lanes,
                weights_packed,
                dim_out,
                n_bytes,
                tile_start,
                &sum_a,
                &c,
                full_blocks,
                tail_start_b,
                dim_in,
            ),
            2 => tmm_tile::<2>(
                outputs,
                inputs,
                lanes,
                weights_packed,
                dim_out,
                n_bytes,
                tile_start,
                &sum_a,
                &c,
                full_blocks,
                tail_start_b,
                dim_in,
            ),
            1 => tmm_tile::<1>(
                outputs,
                inputs,
                lanes,
                weights_packed,
                dim_out,
                n_bytes,
                tile_start,
                &sum_a,
                &c,
                full_blocks,
                tail_start_b,
                dim_in,
            ),
            _ => unreachable!("tile_len in 1..=TOK_TILE(8)"),
        }
        tile_start += tile_len;
    }
}

/// Nibble/pshufb decode of one 32-byte packed-weight block into its 4
/// offset-code (`u = w+1 in {0,1,2}`) lane vectors. Shared by
/// [`tmm_tile`]; textually independent from `tmv_i8_avx2`'s own copy on
/// purpose (see `tmm_i8_avx2`'s doc).
#[target_feature(enable = "avx2")]
unsafe fn decode(v: __m256i, c: &Consts) -> [__m256i; 4] {
    let lo_nib = _mm256_and_si256(v, c.nibble_mask);
    let hi_nib = _mm256_and_si256(_mm256_srli_epi16::<4>(v), c.nibble_mask);
    [
        _mm256_shuffle_epi8(c.code_lut_lo, lo_nib), // bit-pair 0
        _mm256_shuffle_epi8(c.code_lut_hi, lo_nib), // bit-pair 1
        _mm256_shuffle_epi8(c.code_lut_lo, hi_nib), // bit-pair 2
        _mm256_shuffle_epi8(c.code_lut_hi, hi_nib), // bit-pair 3
    ]
}

/// One `M`-token tile (`M` a compile-time constant in `1..=TOK_TILE`),
/// every `dim_out` row of it. `M` monomorphizes the accumulator arrays
/// (`[__m256i; M]`) and their loops (`for ti in 0..M`) so LLVM can keep
/// all `M` token accumulators live in ymm registers across the whole
/// block loop instead of spilling a runtime-length array to the stack —
/// see the dispatch comment in `tmm_i8_avx2`.
///
/// # Safety
/// Same contract as `tmm_i8_avx2`: AVX2 must be OS-enabled; `lanes` must
/// be `tmm_i8_avx2`'s per-token stride-4 deinterleave of `inputs`;
/// `inputs` must contain no `i8::MIN`; `tile_start + M <= n_tok` (the
/// token count implicit in `sum_a`'s length and `lanes`'s layout).
#[allow(clippy::too_many_arguments, unused_assignments)]
#[target_feature(enable = "avx2")]
unsafe fn tmm_tile<const M: usize>(
    outputs: &mut [i32],
    inputs: &[i8],
    lanes: &[i8],
    weights_packed: &[u8],
    dim_out: usize,
    n_bytes: usize,
    tile_start: usize,
    sum_a: &[i32],
    c: &Consts,
    full_blocks: usize,
    tail_start_b: usize,
    dim_in: usize,
) {
    for row in 0..dim_out {
        let w_row = &weights_packed[row * n_bytes..(row + 1) * n_bytes];
        let mut acc32 = [_mm256_setzero_si256(); M];
        let mut acc16 = [_mm256_setzero_si256(); M];
        let mut blocks_since_flush = 0usize;

        // Widens every tile slot's i16 accumulator into i32 and resets
        // it; safe on an empty run, so it doubles as the periodic flush
        // and the final one after the block loop.
        macro_rules! flush {
            () => {
                for ti in 0..M {
                    acc32[ti] =
                        _mm256_add_epi32(acc32[ti], _mm256_madd_epi16(acc16[ti], c.ones_i16));
                    acc16[ti] = _mm256_setzero_si256();
                }
                blocks_since_flush = 0;
            };
        }

        for blk in 0..full_blocks {
            let b0 = blk * BLOCK_BYTES;
            let v = _mm256_loadu_si256(w_row.as_ptr().add(b0) as *const __m256i);
            let w = decode(v, c);

            for ti in 0..M {
                let tok = tile_start + ti;
                let lane_base = tok * 4 * n_bytes;
                let mut pk = [_mm256_setzero_si256(); 4];
                for (k, pkk) in pk.iter_mut().enumerate() {
                    let a = _mm256_loadu_si256(
                        lanes.as_ptr().add(lane_base + k * n_bytes + b0) as *const __m256i
                    );
                    *pkk = _mm256_maddubs_epi16(w[k], a);
                }
                // 3 adds folding the 4 lane pair-sums into one, then 1
                // more into the per-token i16 accumulator.
                let s01 = _mm256_add_epi16(pk[0], pk[1]);
                let s23 = _mm256_add_epi16(pk[2], pk[3]);
                let pair = _mm256_add_epi16(s01, s23);
                acc16[ti] = _mm256_add_epi16(acc16[ti], pair);
            }
            blocks_since_flush += 1;
            if blocks_since_flush == FLUSH_BLOCKS_MM {
                flush!();
            }
        }
        flush!();

        for ti in 0..M {
            let tok = tile_start + ti;
            let lo = _mm256_castsi256_si128(acc32[ti]);
            let hi = _mm256_extracti128_si256(acc32[ti], 1);
            let s = _mm_add_epi32(lo, hi);
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
            let mut total = _mm_cvtsi128_si32(s) - sum_a[tok];

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
                    total += wv * inputs[tok * dim_in + base + k] as i32;
                }
            }

            outputs[tok * dim_out + row] = total;
        }
    }
}

// ---------------------------------------------------------------------------
// LM-head dot: i8 activations against an on-the-fly BF16->Q.F weight row.
// ---------------------------------------------------------------------------

/// Elements (bf16 weights) consumed per SIMD block. Each block is processed
/// as 4 lanes of `i64` — see the module-level doc below for why 64-bit lanes
/// are load-bearing here, not just a width choice.
const DOT_BLOCK: usize = 4;

/// Elements consumed per block of the wider, pure-integer LM-head kernel
/// (`dot_i8_bf16q_avx2_wide`, below `dot_i8_bf16q_avx2_inner`'s module doc).
const WIDE_BLOCK: usize = 16;

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

    if !crate::ops::simd_on() {
        return dot_i8_bf16q(a, row);
    }

    if n >= WIDE_BLOCK {
        // SAFETY: `simd_on()` confirmed AVX2 is supported and OS-enabled,
        // which is this function's only CPU precondition.
        // `dot_i8_bf16q_avx2_wide` bounds every access against `n` and
        // `row` before dereferencing, and falls back to the scalar
        // reference over the WHOLE original `(a, row)` (not just its own
        // slice of it) whenever any element fails a precondition the
        // scalar path would have rejected, so the returned value (and any
        // panic) is exactly what `dot_i8_bf16q(a, row)` would have
        // produced. Its own tail (`n % WIDE_BLOCK` elements) is handled by
        // recursing into this same function on the suffix, which routes to
        // `dot_i8_bf16q_avx2_inner` or the scalar reference as appropriate;
        // integer addition is associative, so splitting the sum at any
        // boundary is exact.
        return unsafe { dot_i8_bf16q_avx2_wide(a, row, n) };
    }

    if n >= DOT_BLOCK {
        // SAFETY: `simd_on()` confirmed AVX2 is supported and OS-enabled, which
        // is this function's only CPU precondition. `n >= DOT_BLOCK` holds, and
        // `dot_i8_bf16q_avx2_inner` bounds every access against `n` and `row`
        // before dereferencing, falling back to the scalar reference (which
        // re-validates from scratch) whenever any element fails a precondition
        // the scalar path would have rejected.
        return unsafe { dot_i8_bf16q_avx2_inner(a, row, n) };
    }

    dot_i8_bf16q(a, row)
}

/// # Safety
/// AVX2 must be supported and OS-enabled (see [`crate::ops::avx2_active`]).
/// `row` must be `2 * n` bytes and `a` must be `n` elements, with `n >=
/// WIDE_BLOCK`.
///
/// # Method — a 16-wide, pure-integer BF16->Q.F conversion
///
/// `dot_i8_bf16q_avx2_inner` (above) widens every BF16 value into a 64-bit
/// lane because the general conversion can reach ~44 bits (`sh` up to 36).
/// Real LM-head rows overwhelmingly do not need that headroom: this kernel
/// takes a narrower conversion (32-bit lanes, 8 BF16 values per `__m256i`)
/// that is exact whenever `-30 <= sh <= 21`, which keeps `|w_q| < 2^30`
/// (worst case mantissa `0xFF`: `0xFF << 21 < 2^30`; the right-shift branch
/// only ever shrinks `m <= 0xFF`, so it is always far inside the bound).
/// Any BF16 value whose `sh` falls outside that window — including every
/// inf/nan (`exp == 0xFF` forces `sh = 141`) and every magnitude
/// `dot_i8_bf16q` itself would reject — is caught by the same `sh` test
/// (`sh > 21` alone already exceeds `dot_i8_bf16q`'s own `sh <= 36` bound,
/// so anything that would panic there is routed to the scalar fallback
/// here, never computed), and the ENTIRE original `(a, row)` — not just
/// this block — is recomputed by `dot_i8_bf16q` so the returned value or
/// panic is identical to the scalar reference's.
///
/// Once `w_q` (`i32`, `|w_q| < 2^30`) is known for 16 elements (two 8-lane
/// halves), each is split exactly as `w_q = (w_hi << 15) + w_lo`:
/// `w_hi = w_q >> 15` (arithmetic, i.e. floor division) and
/// `w_lo = w_q - (w_hi << 15)`, which is therefore always in `[0, 2^15)`
/// regardless of `w_q`'s sign (the remainder of a floor division is
/// nonnegative). Both halves fit `i16` exactly: `|w_q| < 2^30` implies
/// `w_hi` in `[-2^15, 2^15 - 1]` and `w_lo` in `[0, 2^15 - 1]`.
///
/// `w_hi` and `w_lo` (each 2x8 `i32` lanes) are packed to 16 `i16` lanes
/// with `vpackssdw` (exact, no saturation, by the bound above) plus one
/// `vpermq` to undo `vpackssdw`'s cross-128-bit-lane interleave, restoring
/// sequential element order so they line up with the 16 activations loaded
/// by a single `vpmovsxbw` (`i8` -> `i16`, also sequential order, exact
/// since every `i8` fits `i16`). `vpmaddwd(w_half, a)` then computes, for
/// each output lane `k`, `w_half[2k]*a[2k] + w_half[2k+1]*a[2k+1]` exactly
/// (`i16 * i16 -> i32`, no overflow: `|w_hi| <= 32768`, `|a| <= 127`, one
/// product `<= 32768*127 = 4,161,536`, a pair sum `<= 8,323,072`, both far
/// inside `i32`) — this is already a partial dot-product reduction (a
/// 2-element inner sum), which is fine because the final answer is itself a
/// sum over all elements, and integer addition is associative.
///
/// `acc_lo`/`acc_hi` (`i32`, 8 lanes each) accumulate these per-block
/// results directly; every 256 elements (16 blocks — comfortably inside the
/// bound worked out above: 16 additions of a value `<= 8,323,072` per lane
/// is `<= 133,169,152`, far under `2^31`) they are horizontally summed
/// (safe in `i32`: summing 8 such lanes is `<= 1,065,353,216 < 2^31`) and
/// folded into the running `i64` total as `sum_lo + (sum_hi << 15)` — the
/// promotion to `i64` happens before the `<< 15`, so this step cannot
/// overflow either. No floating-point instruction appears anywhere in this
/// kernel (the FullInt property `dot_i8_bf16q_avx2_inner` documents above).
// The `flush!()` macro's final call (after the loop) resets `acc_lo`,
// `acc_hi`, and `blocks_since_flush` even though nothing reads them
// afterward — correct (it mirrors the mid-loop calls exactly, and the
// resets are what make it safe to call before every read), just spuriously
// flagged by `unused_assignments` at that one call site.
#[allow(unused_assignments)]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_bf16q_avx2_wide(a: &[i8], row: &[u8], n: usize) -> i64 {
    let zero = _mm256_setzero_si256();

    // Exact BF16 -> Q.F (i32) conversion, valid only when `-30 <= sh <= 21`
    // (see the fn-level doc above for the bound). Returns `(w_q,
    // out_of_range)`; `out_of_range` is an all-ones/all-zeros per-lane mask
    // (from a `vpcmpgtd`), true in any lane this fast path must not be
    // trusted for.
    #[target_feature(enable = "avx2")]
    unsafe fn wq_i32(bits: __m256i) -> (__m256i, __m256i) {
        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi32(1);
        let c_7f = _mm256_set1_epi32(0x7F);
        let c_80 = _mm256_set1_epi32(0x80);
        let c_8000 = _mm256_set1_epi32(0x8000);
        let c_ff = _mm256_set1_epi32(0xFF);
        let c_134 = _mm256_set1_epi32(134);
        let c_neg133 = _mm256_set1_epi32(-133);
        let c_frac = _mm256_set1_epi32(F as i32);
        let c_sh_hi = _mm256_set1_epi32(21);
        let c_sh_lo = _mm256_set1_epi32(-30);

        let sign_set = _mm256_cmpgt_epi32(_mm256_and_si256(bits, c_8000), zero);
        let exp = _mm256_and_si256(_mm256_srli_epi32::<7>(bits), c_ff);
        let man = _mm256_and_si256(bits, c_7f);
        let exp_is_zero = _mm256_cmpeq_epi32(exp, zero);

        let man_or_80 = _mm256_or_si256(man, c_80);
        let m = _mm256_blendv_epi8(man_or_80, man, exp_is_zero);
        let e_normal = _mm256_sub_epi32(exp, c_134);
        let e = _mm256_blendv_epi8(e_normal, c_neg133, exp_is_zero);
        let sh = _mm256_add_epi32(e, c_frac);

        let out_of_range = _mm256_or_si256(
            _mm256_cmpgt_epi32(sh, c_sh_hi),
            _mm256_cmpgt_epi32(c_sh_lo, sh),
        );

        let sh_ge0 = _mm256_cmpgt_epi32(sh, _mm256_set1_epi32(-1));

        // Left-shift branch (sh >= 0): amount clamped to 0 when unused.
        let shl_amt = _mm256_blendv_epi8(zero, sh, sh_ge0);
        let left_val = _mm256_sllv_epi32(m, shl_amt);

        // Right-shift (RNE) branch (sh < 0): k = -sh, in `[1, 30]` for any
        // lane this fast path trusts (the `out_of_range` guard above);
        // lanes where this branch is unused (sh >= 0, or genuinely
        // out-of-range) may compute a meaningless `k`, but AVX2's variable
        // shifts are defined for every count (>= 32 yields 0), so this is
        // never undefined behaviour — only ever a discarded value.
        let k = _mm256_sub_epi32(zero, sh);
        let floor = _mm256_srlv_epi32(m, k);
        let low_mask = _mm256_sub_epi32(_mm256_sllv_epi32(one, k), one);
        let rem = _mm256_and_si256(m, low_mask);
        let half = _mm256_sllv_epi32(one, _mm256_sub_epi32(k, one));
        let rem_gt_half = _mm256_cmpgt_epi32(rem, half);
        let rem_eq_half = _mm256_cmpeq_epi32(rem, half);
        let floor_odd = _mm256_cmpeq_epi32(_mm256_and_si256(floor, one), one);
        let round_up = _mm256_or_si256(rem_gt_half, _mm256_and_si256(rem_eq_half, floor_odd));
        let right_val = _mm256_add_epi32(floor, _mm256_blendv_epi8(zero, one, round_up));

        let v_mag = _mm256_blendv_epi8(right_val, left_val, sh_ge0);
        let v_neg = _mm256_sub_epi32(zero, v_mag);
        let v = _mm256_blendv_epi8(v_mag, v_neg, sign_set);
        (v, out_of_range)
    }

    // Undo `vpackssdw`'s cross-128-bit-lane interleave: `_mm256_packs_epi32(lo,
    // hi)` yields 64-bit-qword order `[lo.low, hi.low, lo.high, hi.high]`;
    // permuting qwords `(0, 2, 1, 3)` restores sequential element order.
    #[target_feature(enable = "avx2")]
    unsafe fn pack16(lo: __m256i, hi: __m256i) -> __m256i {
        _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0b11_01_10_00)
    }

    let full_blocks = n / WIDE_BLOCK;
    let tail_start = full_blocks * WIDE_BLOCK;

    let mut acc_lo = zero;
    let mut acc_hi = zero;
    let mut total: i64 = 0;
    let mut blocks_since_flush: u32 = 0;
    // Flush before 16 accumulated blocks (256 elements) — see the fn-level
    // doc for why that bound keeps every lane inside i32.
    const FLUSH_BLOCKS: u32 = 16;

    macro_rules! flush {
        () => {{
            let hsum = |v: __m256i| -> i32 {
                let lo = _mm256_castsi256_si128(v);
                let hi = _mm256_extracti128_si256(v, 1);
                let s = _mm_add_epi32(lo, hi);
                let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
                _mm_cvtsi128_si32(s)
            };
            let sum_lo = hsum(acc_lo) as i64;
            let sum_hi = hsum(acc_hi) as i64;
            total += sum_lo + (sum_hi << 15);
            acc_lo = zero;
            acc_hi = zero;
            blocks_since_flush = 0;
        }};
    }

    for blk in 0..full_blocks {
        let base = blk * WIDE_BLOCK;

        let bits0_16 = _mm_loadu_si128(row.as_ptr().add(base * 2) as *const __m128i);
        let bits0 = _mm256_cvtepu16_epi32(bits0_16);
        let bits1_16 = _mm_loadu_si128(row.as_ptr().add((base + 8) * 2) as *const __m128i);
        let bits1 = _mm256_cvtepu16_epi32(bits1_16);

        let (w0, oor0) = wq_i32(bits0);
        let (w1, oor1) = wq_i32(bits1);
        let oor = _mm256_or_si256(oor0, oor1);
        if _mm256_movemask_epi8(oor) != 0 {
            // Whole ORIGINAL row, not just this block: reproduces
            // `dot_i8_bf16q`'s value (or panic) exactly.
            return dot_i8_bf16q(a, row);
        }

        let w_hi0 = _mm256_srai_epi32::<15>(w0);
        let w_lo0 = _mm256_sub_epi32(w0, _mm256_slli_epi32::<15>(w_hi0));
        let w_hi1 = _mm256_srai_epi32::<15>(w1);
        let w_lo1 = _mm256_sub_epi32(w1, _mm256_slli_epi32::<15>(w_hi1));

        let w_hi = pack16(w_hi0, w_hi1);
        let w_lo = pack16(w_lo0, w_lo1);

        let act8 = _mm_loadu_si128(a.as_ptr().add(base) as *const __m128i);
        let act = _mm256_cvtepi8_epi16(act8);

        acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(w_lo, act));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(w_hi, act));
        blocks_since_flush += 1;
        if blocks_since_flush == FLUSH_BLOCKS {
            flush!();
        }
    }
    if blocks_since_flush > 0 {
        flush!();
    }

    // Tail (`n % WIDE_BLOCK` elements): exact by associativity. Recursing
    // into the public entry point routes it to `dot_i8_bf16q_avx2_inner` or
    // the scalar reference as its own length dictates, preserving both the
    // value and any panic (a panic in the tail is still a panic on the
    // ORIGINAL `a`/`row` bytes at that position, so its message is
    // identical regardless of which kernel raises it).
    if tail_start < n {
        total += dot_i8_bf16q_avx2(&a[tail_start..], &row[tail_start * 2..]);
    }

    total
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

        // Fast path: real LM-head weights are overwhelmingly `|w| >= 2^-13`ish
        // (sh = exp - 114 >= 0 for exp >= 114, i.e. bf16 magnitude >= ~2^-13),
        // so all 4 lanes take the left-shift branch far more often than not.
        // Computing the RNE right-shift chain (floor/rem/half/round_up, ~10
        // dependent vector ops) unconditionally on every block would cost as
        // much as the branch it replaces, defeating the point of vectorizing.
        // `movemask == -1` iff every byte of every lane's `sh_ge0` mask is
        // set, i.e. all 4 lanes are `sh >= 0` — skip the right-shift chain
        // entirely in that case; the result is unaffected either way, this is
        // purely a work-skip, not a different computation.
        let v_mag = if _mm256_movemask_epi8(sh_ge0) == -1 {
            _mm256_sllv_epi64(m, sh)
        } else {
            // Left-shift branch (sh >= 0): shift amount clamped to 0 when
            // unused so the variable shift never sees a negative count.
            let shl_amt = _mm256_blendv_epi8(zero, sh, sh_ge0);
            let left_val = _mm256_sllv_epi64(m, shl_amt);

            // Right-shift (RNE) branch (sh < 0): k = -sh.
            let k_raw = _mm256_sub_epi64(zero, sh);
            let k_ge63 = _mm256_cmpgt_epi64(k_raw, c_62);
            // Clamp the shift operand to 62 when it would be discarded
            // anyway (k>=63 case), keeping every shift amount in 0..=62.
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

            _mm256_blendv_epi8(right_val, left_val, sh_ge0)
        };
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

// ---------------------------------------------------------------------------
// LM-head dot over pre-converted i16 hi/lo planes (see
// `crate::cis_infer::HeadPlanes`). The BF16->Q.F conversion this kernel's
// on-the-fly siblings above perform per element per call has already
// happened once, at load, into `hi`/`lo`; this kernel is therefore a plain
// widen + `vpmaddwd` stream with no per-element conversion, no per-block
// out-of-range guard, and no scalar-fallback-on-panic path (every row this
// function is ever called on was already validated when `HeadPlanes::build`
// produced its `hi`/`lo` slices — a row that could not take the split was
// recorded as an exception there and never reaches here; see
// `CisModel::head_planes_row`).
// ---------------------------------------------------------------------------

/// AVX2 dot of i8 activations against pre-converted i16 hi/lo planes.
/// Bit-identical to [`crate::cis_infer::dot_i8_i16planes`] for every input
/// (falls back to it when AVX2 is unavailable or the row is too short to
/// block), which is itself bit-identical to `dot_i8_bf16q` over the ORIGINAL
/// BF16 row whenever `hi`/`lo` came from `HeadPlanes::build` on that same
/// row — see `head_row_to_planes`'s doc comment for that half of the
/// exactness argument (this function only needs to reproduce the plane dot
/// itself, not the conversion).
///
/// # Method
///
/// `hi[i]` and `lo[i]` are already `i16` in memory (unlike the on-the-fly
/// kernels, which have to derive and pack them from a wider intermediate
/// every call), so 16 elements load directly as two `__m256i` registers —
/// no `vpackssdw`/`vpermq` repacking is needed at all. 16 `i8` activations
/// widen to `i16` with `_mm256_cvtepi8_epi16` (sign-extending, sequential
/// order, exact since every `i8` fits `i16`). `vpmaddwd(lo, act)` and
/// `vpmaddwd(hi, act)` each compute, per output lane `k`, `x[2k]*a[2k] +
/// x[2k+1]*a[2k+1]` exactly (`i16 * i16 -> i32`, no overflow: `|hi| <=
/// 32768`, `|lo| < 32768`, `|a| <= 127`, one product `<= 32768*127 =
/// 4,161,536`, a pair sum `<= 8,323,072`, both far inside `i32` —
/// `split_i16_planes`'s own doc comment gives the same bound its callers
/// rely on). Accumulation and the periodic flush into a running `i64` total
/// are byte-for-byte the same discipline `dot_i8_bf16q_avx2_wide` documents
/// above (flush every 256 elements, `sum_lo + (sum_hi << 15)`), which is why
/// this kernel reuses that function's `WIDE_BLOCK`/`FLUSH_BLOCKS` constants
/// and hsum helper shape rather than inventing new ones.
pub fn dot_i8_i16planes_avx2(a: &[i8], hi: &[i16], lo: &[i16]) -> i64 {
    debug_assert_eq!(a.len(), hi.len());
    debug_assert_eq!(a.len(), lo.len());
    let n = a.len();

    if !crate::ops::simd_on() || n < WIDE_BLOCK {
        return crate::cis_infer::dot_i8_i16planes(a, hi, lo);
    }

    // SAFETY: `simd_on()` confirmed AVX2 is supported and OS-enabled, which
    // is this function's only CPU precondition. `n >= WIDE_BLOCK` holds;
    // `dot_i8_i16planes_avx2_wide` bounds every access against `n` and
    // recurses into this same function on its own tail (`n % WIDE_BLOCK`
    // elements), which routes to the scalar reference for a short enough
    // remainder — integer addition is associative, so splitting the sum at
    // any boundary is exact.
    unsafe { dot_i8_i16planes_avx2_wide(a, hi, lo, n) }
}

/// # Safety
/// AVX2 must be supported and OS-enabled (see [`crate::ops::avx2_active`]).
/// `hi`, `lo`, and `a` must all have length `n`, with `n >= WIDE_BLOCK`.
// See `dot_i8_bf16q_avx2_wide`'s own `#[allow(unused_assignments)]` doc
// comment: the same spurious `flush!()` lint applies to this macro's final
// call here.
#[allow(unused_assignments)]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i16planes_avx2_wide(a: &[i8], hi: &[i16], lo: &[i16], n: usize) -> i64 {
    let zero = _mm256_setzero_si256();

    let full_blocks = n / WIDE_BLOCK;
    let tail_start = full_blocks * WIDE_BLOCK;

    let mut acc_lo = zero;
    let mut acc_hi = zero;
    let mut total: i64 = 0;
    let mut blocks_since_flush: u32 = 0;
    // Flush before 16 accumulated blocks (256 elements) — same bound
    // `dot_i8_bf16q_avx2_wide` documents (this kernel's per-lane products
    // are bounded identically: `|hi| <= 32768`, `|lo| < 32768`, `|a| <=
    // 127`).
    const FLUSH_BLOCKS: u32 = 16;

    macro_rules! flush {
        () => {{
            let hsum = |v: __m256i| -> i32 {
                let lo = _mm256_castsi256_si128(v);
                let hi = _mm256_extracti128_si256(v, 1);
                let s = _mm_add_epi32(lo, hi);
                let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
                _mm_cvtsi128_si32(s)
            };
            let sum_lo = hsum(acc_lo) as i64;
            let sum_hi = hsum(acc_hi) as i64;
            total += sum_lo + (sum_hi << 15);
            acc_lo = zero;
            acc_hi = zero;
            blocks_since_flush = 0;
        }};
    }

    for blk in 0..full_blocks {
        let base = blk * WIDE_BLOCK;

        // SAFETY: `base + WIDE_BLOCK <= n == hi.len() == lo.len() ==
        // a.len()`, so all three loads stay in bounds.
        let lo_v = _mm256_loadu_si256(lo.as_ptr().add(base) as *const __m256i);
        let hi_v = _mm256_loadu_si256(hi.as_ptr().add(base) as *const __m256i);
        let act8 = _mm_loadu_si128(a.as_ptr().add(base) as *const __m128i);
        let act = _mm256_cvtepi8_epi16(act8);

        acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(lo_v, act));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(hi_v, act));
        blocks_since_flush += 1;
        if blocks_since_flush == FLUSH_BLOCKS {
            flush!();
        }
    }
    if blocks_since_flush > 0 {
        flush!();
    }

    // Tail (`n % WIDE_BLOCK` elements): exact by associativity, same
    // discipline as `dot_i8_bf16q_avx2_wide`'s own tail.
    if tail_start < n {
        total += dot_i8_i16planes_avx2(&a[tail_start..], &hi[tail_start..], &lo[tail_start..]);
    }

    total
}
