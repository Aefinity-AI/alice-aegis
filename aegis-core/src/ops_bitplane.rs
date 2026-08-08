// House style (matches ops.rs): the `unsafe fn` bodies below are wall-to-wall
// intrinsics whose single obligation — AVX2 present and OS-enabled, buffers
// sized per the documented contract — is stated once per fn in its `# Safety`
// section, not per intrinsic call.
#![allow(unsafe_op_in_unsafe_fn)]

//! Bitplane-dense ternary matvec — the "tri-matvec, reading (b)" candidate.
//!
//! Same 2 bits/weight as the incumbent `ops::ternary_matvec_avx2`, stored as
//! two per-row bitmaps (a +1 plane and a -1 plane) instead of one 2-bit code
//! stream. The AVX2 kernels consume the planes DENSELY with SIMD mask
//! expansion — broadcast 4 plane bytes (`vpbroadcastd`), `vpshufb` byte-splat,
//! `vpand` against a per-lane bit-select constant, `vpcmpeqd` to a full-lane
//! mask, then a masked add of the shared input vector for the +1 plane and a
//! masked subtract for the -1 plane.
//!
//! This is NOT the settled CTZ negative (ledger A6): no `trailing_zeros`, no
//! bit scan, no per-nonzero branch, no dependence on weight sparsity — every
//! column is processed unconditionally, 8 lanes at a time, exactly like the
//! incumbent. It also is not the settled T-MAC pshufb LUT-mpGEMM (A7): the
//! layout stays at 2 bits/weight, so decode memory traffic is identical to
//! the incumbent's.
//!
//! # Bit order (the format contract)
//!
//! For a `dim_out x dim_in` matrix, each plane stores
//! `bitplane_words_per_row(dim_in)` `u64` words per row, row-major. Column
//! `c` of a row lives at bit `c % 64` (LSB-first) of word `c / 64`. Viewed as
//! bytes on a little-endian machine (x86_64 is this crate's only
//! architecture), byte `j` of a row covers columns `8j..=8j+7` with bit `k` =
//! column `8j + k` — which is what lets the AVX2 kernel load 4 plane bytes as
//! one `u32` and splat them. Unused padding bits above the last column are
//! zero and are never read.
//!
//! Column coverage mirrors the incumbent exactly: only the first
//! `(dim_in / 4) * 4` columns exist (the 2-bit packed format holds 4 weights
//! per byte and truncates the remainder). 2-bit codes map 01 -> +1 plane,
//! 10 -> -1 plane, 00 -> neither; the undefined 11 code also maps to neither,
//! degrading to 0.0 exactly like the incumbent's `UNPACK_LUT`.
//!
//! # Numeric identity
//!
//! `bitplane_matvec_avx2` (order-preserving, single accumulator) performs,
//! per lane and per 8-column group, the same single rounded add/subtract the
//! incumbent's FMA performs (`x*(+1) + s = s + x`, `x*(-1) + s = s - x`,
//! `x*0 + s = s`), in the same group order, with the same 8-lane horizontal
//! reduction (`storeu` then `iter().sum()`) and the same scalar tail
//! expression — so its output is BYTE-IDENTICAL to `ternary_matvec_avx2` for
//! finite inputs. Two footnotes to that claim:
//! - A NaN or Inf input at a ZERO-weight column propagates through the
//!   incumbent's `x * 0.0` multiply but is masked to +0.0 here. Activations
//!   are finite, so this cannot occur on the real path.
//! - The masked no-op leg is an exact identity: accumulator lanes start at
//!   +0.0 and can never become -0.0 (IEEE-754 round-to-nearest only produces
//!   -0.0 from `(-0.0) + (-0.0)`), so adding/subtracting a masked-out +0.0
//!   changes no bits, matching the incumbent's `fma(x, 0.0, s) = s`.
//!
//! `bitplane_matvec_avx2_dual` (two accumulators, accP/accM) halves the
//! loop-carried dependence chain but sums the two planes separately, so its
//! rounding ORDER differs — it is NOT byte-identical to the incumbent and is
//! instead pinned bit-for-bit to `bitplane_matvec_scalar_dual`, its
//! op-identical scalar mirror.
//!
//! `bitplane_matvec_scalar` is the SSE2-class fallback (HP Stream path) AND
//! the test oracle: it simulates the 8-lane vector computation literally, so
//! it is bit-identical to `bitplane_matvec_avx2` — and therefore also to the
//! incumbent AVX2 kernel. (A property the incumbent pair does not have:
//! `ops::ternary_matvec_scalar` does NOT match `ops::ternary_matvec_avx2`
//! bitwise.)
//!
//! # UEFI build note (why the AVX2 kernels are cfg-gated off `target_os = "uefi"`)
//!
//! The nightly rustc LLVM backend aborts with "Do not know how to split the
//! result of this operator!" when codegenning these multi-row-unrolled
//! kernels for the soft-float `x86_64-unknown-uefi` target — the same target
//! the incumbent's kernels compile for. Bisected empirically (2026-08-01):
//! a single-row loop body compiles; a 2-row unroll with masked adds only
//! compiles; a 2-row unroll with both masked adds and subs fails; the
//! adds-only dual-accumulator 4-row form fails too. Disabling the SLP and
//! loop vectorizers (`-C no-vectorize-slp` / `-C no-vectorize-loops`) does
//! NOT fix it, so it is a deeper legalization limit on the number of
//! parallel vector chains per block under the soft-float ABI. Helper
//! functions passing `__m256` by value were already eliminated for the same
//! backend error (see `expand_mask!`). This candidate races the incumbent on
//! the HOSTED path (aegis-linux); the unikernel keeps the incumbent, so the
//! gate costs nothing until the family wins its A/B — at which point the
//! kernels would need reshaping (or a fixed LLVM) to boot bare-metal.
//! `pack_bitplanes` and both scalar references remain available on every
//! target, UEFI included.

#[cfg(not(target_os = "uefi"))]
use core::arch::x86_64::*;

/// `u64` words per row of one bitplane for a matrix with `dim_in` input
/// columns. Only the incumbent-covered `(dim_in / 4) * 4` columns are stored.
pub fn bitplane_words_per_row(dim_in: usize) -> usize {
    ((dim_in / 4) * 4).div_ceil(64)
}

/// Derive the (+1 plane, -1 plane) bitmaps from the incumbent's row-major
/// 2-bit packed weights (4 weights/byte, codes 00=0, 01=+1, 10=-1, 11 -> 0).
///
/// `plane_pos` / `plane_neg` must each hold at least
/// `dim_out * bitplane_words_per_row(dim_in)` words; that prefix is fully
/// overwritten (cleared, then set). Bit order is the module contract above.
pub fn pack_bitplanes(
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
    plane_pos: &mut [u64],
    plane_neg: &mut [u64],
) {
    let packed_dim_in = dim_in / 4;
    let words = bitplane_words_per_row(dim_in);
    assert!(
        weights_packed.len() >= dim_out * packed_dim_in,
        "packed weights too short: {} < {}",
        weights_packed.len(),
        dim_out * packed_dim_in
    );
    assert!(
        plane_pos.len() >= dim_out * words && plane_neg.len() >= dim_out * words,
        "plane buffers too short: need {} words per plane",
        dim_out * words
    );
    plane_pos[..dim_out * words].fill(0);
    plane_neg[..dim_out * words].fill(0);

    for row in 0..dim_out {
        for cp in 0..packed_dim_in {
            let byte = weights_packed[row * packed_dim_in + cp];
            for lane in 0..4 {
                let col = cp * 4 + lane;
                let idx = row * words + col / 64;
                let bit = 1u64 << (col % 64);
                match (byte >> (2 * lane)) & 3 {
                    1 => plane_pos[idx] |= bit,
                    2 => plane_neg[idx] |= bit,
                    _ => {} // 00 = zero; undefined 11 degrades to zero too
                }
            }
        }
    }
}

/// Bit `c` of `plane` for the row whose words start at `base`.
#[inline(always)]
fn plane_bit(plane: &[u64], base: usize, c: usize) -> u64 {
    (plane[base + c / 64] >> (c % 64)) & 1
}

/// The incumbent's scalar-tail quad for one packed byte's 4 columns, with the
/// weight reconstructed from the planes. Expression shape and association
/// order are EXACTLY `ternary_matvec_avx2`'s tail:
/// `((x0*w0 + x1*w1) + x2*w2) + x3*w3`, weights in {1.0, -1.0, 0.0}.
#[inline(always)]
fn tail_quad(input: &[f32], col: usize, pos: &[u64], neg: &[u64], base: usize) -> f32 {
    let w = |c: usize| -> f32 {
        if plane_bit(pos, base, c) == 1 {
            1.0
        } else if plane_bit(neg, base, c) == 1 {
            -1.0
        } else {
            0.0
        }
    };
    input[col] * w(col)
        + input[col + 1] * w(col + 1)
        + input[col + 2] * w(col + 2)
        + input[col + 3] * w(col + 3)
}

/// Single-plane tail quad for the dual-accumulator variants: weights in
/// {1.0, 0.0}, same association order as `tail_quad`.
#[inline(always)]
fn tail_quad_plane(input: &[f32], col: usize, plane: &[u64], base: usize) -> f32 {
    let w = |c: usize| -> f32 { plane_bit(plane, base, c) as f32 };
    input[col] * w(col)
        + input[col + 1] * w(col + 1)
        + input[col + 2] * w(col + 2)
        + input[col + 3] * w(col + 3)
}

/// Scalar reference, ORDER-PRESERVING: a literal simulation of the 8-lane
/// single-accumulator vector computation (masked no-ops included), so it is
/// bit-identical to `bitplane_matvec_avx2` and to the incumbent
/// `ternary_matvec_avx2`. This is the SSE2-class path for the HP Stream and
/// the test oracle for variant (i).
pub fn bitplane_matvec_scalar(
    output: &mut [f32],
    input: &[f32],
    plane_pos: &[u64],
    plane_neg: &[u64],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    let packed_dim_in = dim_in / 4;
    let words = bitplane_words_per_row(dim_in);

    for (row, out) in output.iter_mut().take(dim_out).enumerate() {
        let base = row * words;
        let mut lanes = [0.0f32; 8];
        let mut col_packed = 0;
        let mut col = 0;

        while col_packed + 8 <= packed_dim_in {
            for g in 0..4 {
                for (k, lane) in lanes.iter_mut().enumerate() {
                    let c = col + g * 8 + k;
                    let x = input[c];
                    // Masked add then masked sub, exactly like the vector
                    // kernel: the inactive leg adds/subtracts +0.0.
                    let xp = if plane_bit(plane_pos, base, c) == 1 {
                        x
                    } else {
                        0.0
                    };
                    *lane += xp;
                    let xm = if plane_bit(plane_neg, base, c) == 1 {
                        x
                    } else {
                        0.0
                    };
                    *lane -= xm;
                }
            }
            col_packed += 8;
            col += 32;
        }

        let mut fs = lanes.iter().sum::<f32>();
        while col_packed < packed_dim_in {
            fs += tail_quad(input, col, plane_pos, plane_neg, base);
            col_packed += 1;
            col += 4;
        }
        *out = fs * scale;
    }
}

/// Scalar mirror of the dual-accumulator variant: separate +plane and -plane
/// 8-lane accumulators, reduced independently, combined as `(fsP - fsM)`.
/// Test oracle for `bitplane_matvec_avx2_dual`. NOT byte-identical to the
/// incumbent (different rounding order).
pub fn bitplane_matvec_scalar_dual(
    output: &mut [f32],
    input: &[f32],
    plane_pos: &[u64],
    plane_neg: &[u64],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    let packed_dim_in = dim_in / 4;
    let words = bitplane_words_per_row(dim_in);

    for (row, out) in output.iter_mut().take(dim_out).enumerate() {
        let base = row * words;
        let mut lanes_p = [0.0f32; 8];
        let mut lanes_m = [0.0f32; 8];
        let mut col_packed = 0;
        let mut col = 0;

        while col_packed + 8 <= packed_dim_in {
            for g in 0..4 {
                for k in 0..8 {
                    let c = col + g * 8 + k;
                    let x = input[c];
                    let xp = if plane_bit(plane_pos, base, c) == 1 {
                        x
                    } else {
                        0.0
                    };
                    lanes_p[k] += xp;
                    let xm = if plane_bit(plane_neg, base, c) == 1 {
                        x
                    } else {
                        0.0
                    };
                    lanes_m[k] += xm;
                }
            }
            col_packed += 8;
            col += 32;
        }

        let mut fs_p = lanes_p.iter().sum::<f32>();
        let mut fs_m = lanes_m.iter().sum::<f32>();
        while col_packed < packed_dim_in {
            fs_p += tail_quad_plane(input, col, plane_pos, base);
            fs_m += tail_quad_plane(input, col, plane_neg, base);
            col_packed += 1;
            col += 4;
        }
        *out = (fs_p - fs_m) * scale;
    }
}

/// Expand bit `8g+k` of the broadcast plane dword into an all-ones/all-zeros
/// 32-bit lane mask for lane k. `$v` holds the 4 plane bytes replicated into
/// every dword (so byte position g of every dword is plane byte g); `$ctrl`
/// splats byte `g`, `$bitsel` selects bit k in lane k.
///
/// A macro, not a function, deliberately: the `x86_64-unknown-uefi` target
/// has a soft-float ABI, and LLVM cannot pass or return `__m256`/`__m256i`
/// by value across any function boundary there ("Do not know how to split
/// the result of this operator!"). Textual expansion keeps every vector
/// value inside its `#[target_feature(enable = "avx2")]` kernel.
#[cfg(not(target_os = "uefi"))]
macro_rules! expand_mask {
    ($v:expr, $ctrl:expr, $bitsel:expr) => {
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(
            _mm256_and_si256(_mm256_shuffle_epi8($v, $ctrl), $bitsel),
            $bitsel,
        ))
    };
}

/// Bitplane matvec, variant (i): ORDER-PRESERVING single accumulator.
/// Masked-add of the +1 plane then masked-sub of the -1 plane per 8-column
/// group — same per-lane rounding sequence as the incumbent's FMA, so output
/// is byte-identical to `ternary_matvec_avx2` (finite inputs; see module
/// docs). 4-row unroll mirroring the incumbent's structure.
///
/// # Safety
/// Caller must guarantee AVX2 is supported and OS-enabled (see
/// `ops::avx2_active()`), `plane_pos`/`plane_neg` each hold
/// `dim_out * bitplane_words_per_row(dim_in)` words in this module's bit
/// order, `input.len() >= dim_in`, and `output.len() >= dim_out`.
#[cfg(not(target_os = "uefi"))] // see module docs: UEFI build note
#[target_feature(enable = "avx2")]
pub unsafe fn bitplane_matvec_avx2(
    output: &mut [f32],
    input: &[f32],
    plane_pos: &[u64],
    plane_neg: &[u64],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    let packed_dim_in = dim_in / 4;
    let words = bitplane_words_per_row(dim_in);
    let row_bytes = words * 8;
    // SAFETY: reinterpreting &[u64] as bytes is always valid; x86_64 is
    // little-endian, so byte j of a row is columns 8j..=8j+7 (module
    // contract). All reads below stay inside `dim_out * row_bytes` bytes:
    // the vector loop reads 4 bytes at offset col_packed/2 <=
    // (packed_dim_in - 8)/2, and (packed_dim_in-8)/2 + 4 <= words*8.
    let pos_base = plane_pos.as_ptr() as *const u8;
    let neg_base = plane_neg.as_ptr() as *const u8;
    let bitsel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

    let mut row = 0;
    while row + 3 < dim_out {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let p0 = pos_base.add(row * row_bytes);
        let p1 = pos_base.add((row + 1) * row_bytes);
        let p2 = pos_base.add((row + 2) * row_bytes);
        let p3 = pos_base.add((row + 3) * row_bytes);
        let n0 = neg_base.add(row * row_bytes);
        let n1 = neg_base.add((row + 1) * row_bytes);
        let n2 = neg_base.add((row + 2) * row_bytes);
        let n3 = neg_base.add((row + 3) * row_bytes);

        let mut col_packed = 0;
        let mut col = 0;

        while col_packed + 8 <= packed_dim_in {
            let bo = col_packed / 2; // plane byte offset: 32 columns = 4 bytes
            let vp0 = _mm256_set1_epi32(core::ptr::read_unaligned(p0.add(bo) as *const u32) as i32);
            let vp1 = _mm256_set1_epi32(core::ptr::read_unaligned(p1.add(bo) as *const u32) as i32);
            let vp2 = _mm256_set1_epi32(core::ptr::read_unaligned(p2.add(bo) as *const u32) as i32);
            let vp3 = _mm256_set1_epi32(core::ptr::read_unaligned(p3.add(bo) as *const u32) as i32);
            let vn0 = _mm256_set1_epi32(core::ptr::read_unaligned(n0.add(bo) as *const u32) as i32);
            let vn1 = _mm256_set1_epi32(core::ptr::read_unaligned(n1.add(bo) as *const u32) as i32);
            let vn2 = _mm256_set1_epi32(core::ptr::read_unaligned(n2.add(bo) as *const u32) as i32);
            let vn3 = _mm256_set1_epi32(core::ptr::read_unaligned(n3.add(bo) as *const u32) as i32);

            for g in 0..4 {
                let in_avx = _mm256_loadu_ps(input.as_ptr().add(col + g * 8));
                let ctrl = _mm256_set1_epi8(g as i8);

                acc0 = _mm256_add_ps(acc0, _mm256_and_ps(expand_mask!(vp0, ctrl, bitsel), in_avx));
                acc0 = _mm256_sub_ps(acc0, _mm256_and_ps(expand_mask!(vn0, ctrl, bitsel), in_avx));
                acc1 = _mm256_add_ps(acc1, _mm256_and_ps(expand_mask!(vp1, ctrl, bitsel), in_avx));
                acc1 = _mm256_sub_ps(acc1, _mm256_and_ps(expand_mask!(vn1, ctrl, bitsel), in_avx));
                acc2 = _mm256_add_ps(acc2, _mm256_and_ps(expand_mask!(vp2, ctrl, bitsel), in_avx));
                acc2 = _mm256_sub_ps(acc2, _mm256_and_ps(expand_mask!(vn2, ctrl, bitsel), in_avx));
                acc3 = _mm256_add_ps(acc3, _mm256_and_ps(expand_mask!(vp3, ctrl, bitsel), in_avx));
                acc3 = _mm256_sub_ps(acc3, _mm256_and_ps(expand_mask!(vn3, ctrl, bitsel), in_avx));
            }
            col_packed += 8;
            col += 32;
        }

        let mut s0 = [0.0f32; 8];
        let mut s1 = [0.0f32; 8];
        let mut s2 = [0.0f32; 8];
        let mut s3 = [0.0f32; 8];
        _mm256_storeu_ps(s0.as_mut_ptr(), acc0);
        _mm256_storeu_ps(s1.as_mut_ptr(), acc1);
        _mm256_storeu_ps(s2.as_mut_ptr(), acc2);
        _mm256_storeu_ps(s3.as_mut_ptr(), acc3);
        let mut fs0 = s0.iter().sum::<f32>();
        let mut fs1 = s1.iter().sum::<f32>();
        let mut fs2 = s2.iter().sum::<f32>();
        let mut fs3 = s3.iter().sum::<f32>();

        while col_packed < packed_dim_in {
            fs0 += tail_quad(input, col, plane_pos, plane_neg, row * words);
            fs1 += tail_quad(input, col, plane_pos, plane_neg, (row + 1) * words);
            fs2 += tail_quad(input, col, plane_pos, plane_neg, (row + 2) * words);
            fs3 += tail_quad(input, col, plane_pos, plane_neg, (row + 3) * words);
            col_packed += 1;
            col += 4;
        }
        output[row] = fs0 * scale;
        output[row + 1] = fs1 * scale;
        output[row + 2] = fs2 * scale;
        output[row + 3] = fs3 * scale;
        row += 4;
    }

    while row < dim_out {
        let mut acc0 = _mm256_setzero_ps();
        let p0 = pos_base.add(row * row_bytes);
        let n0 = neg_base.add(row * row_bytes);
        let mut col_packed = 0;
        let mut col = 0;

        while col_packed + 8 <= packed_dim_in {
            let bo = col_packed / 2;
            let vp0 = _mm256_set1_epi32(core::ptr::read_unaligned(p0.add(bo) as *const u32) as i32);
            let vn0 = _mm256_set1_epi32(core::ptr::read_unaligned(n0.add(bo) as *const u32) as i32);
            for g in 0..4 {
                let in_avx = _mm256_loadu_ps(input.as_ptr().add(col + g * 8));
                let ctrl = _mm256_set1_epi8(g as i8);
                acc0 = _mm256_add_ps(acc0, _mm256_and_ps(expand_mask!(vp0, ctrl, bitsel), in_avx));
                acc0 = _mm256_sub_ps(acc0, _mm256_and_ps(expand_mask!(vn0, ctrl, bitsel), in_avx));
            }
            col_packed += 8;
            col += 32;
        }

        let mut s0 = [0.0f32; 8];
        _mm256_storeu_ps(s0.as_mut_ptr(), acc0);
        let mut fs0 = s0.iter().sum::<f32>();
        while col_packed < packed_dim_in {
            fs0 += tail_quad(input, col, plane_pos, plane_neg, row * words);
            col_packed += 1;
            col += 4;
        }
        output[row] = fs0 * scale;
        row += 1;
    }
}

/// Bitplane matvec, variant (ii): DUAL accumulators (accP/accM per row).
/// The two masked adds per group are independent, halving the loop-carried
/// latency chain, at the cost of a different rounding order: NOT
/// byte-identical to the incumbent. Pinned bit-for-bit to
/// `bitplane_matvec_scalar_dual` instead.
///
/// # Safety
/// Same contract as `bitplane_matvec_avx2`.
#[cfg(not(target_os = "uefi"))] // see module docs: UEFI build note
#[target_feature(enable = "avx2")]
pub unsafe fn bitplane_matvec_avx2_dual(
    output: &mut [f32],
    input: &[f32],
    plane_pos: &[u64],
    plane_neg: &[u64],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    let packed_dim_in = dim_in / 4;
    let words = bitplane_words_per_row(dim_in);
    let row_bytes = words * 8;
    // SAFETY: same byte-view bounds argument as `bitplane_matvec_avx2`.
    let pos_base = plane_pos.as_ptr() as *const u8;
    let neg_base = plane_neg.as_ptr() as *const u8;
    let bitsel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

    let mut row = 0;
    while row + 3 < dim_out {
        let mut acc_p0 = _mm256_setzero_ps();
        let mut acc_p1 = _mm256_setzero_ps();
        let mut acc_p2 = _mm256_setzero_ps();
        let mut acc_p3 = _mm256_setzero_ps();
        let mut acc_m0 = _mm256_setzero_ps();
        let mut acc_m1 = _mm256_setzero_ps();
        let mut acc_m2 = _mm256_setzero_ps();
        let mut acc_m3 = _mm256_setzero_ps();

        let p0 = pos_base.add(row * row_bytes);
        let p1 = pos_base.add((row + 1) * row_bytes);
        let p2 = pos_base.add((row + 2) * row_bytes);
        let p3 = pos_base.add((row + 3) * row_bytes);
        let n0 = neg_base.add(row * row_bytes);
        let n1 = neg_base.add((row + 1) * row_bytes);
        let n2 = neg_base.add((row + 2) * row_bytes);
        let n3 = neg_base.add((row + 3) * row_bytes);

        let mut col_packed = 0;
        let mut col = 0;

        while col_packed + 8 <= packed_dim_in {
            let bo = col_packed / 2;
            let vp0 = _mm256_set1_epi32(core::ptr::read_unaligned(p0.add(bo) as *const u32) as i32);
            let vp1 = _mm256_set1_epi32(core::ptr::read_unaligned(p1.add(bo) as *const u32) as i32);
            let vp2 = _mm256_set1_epi32(core::ptr::read_unaligned(p2.add(bo) as *const u32) as i32);
            let vp3 = _mm256_set1_epi32(core::ptr::read_unaligned(p3.add(bo) as *const u32) as i32);
            let vn0 = _mm256_set1_epi32(core::ptr::read_unaligned(n0.add(bo) as *const u32) as i32);
            let vn1 = _mm256_set1_epi32(core::ptr::read_unaligned(n1.add(bo) as *const u32) as i32);
            let vn2 = _mm256_set1_epi32(core::ptr::read_unaligned(n2.add(bo) as *const u32) as i32);
            let vn3 = _mm256_set1_epi32(core::ptr::read_unaligned(n3.add(bo) as *const u32) as i32);

            for g in 0..4 {
                let in_avx = _mm256_loadu_ps(input.as_ptr().add(col + g * 8));
                let ctrl = _mm256_set1_epi8(g as i8);

                acc_p0 = _mm256_add_ps(
                    acc_p0,
                    _mm256_and_ps(expand_mask!(vp0, ctrl, bitsel), in_avx),
                );
                acc_m0 = _mm256_add_ps(
                    acc_m0,
                    _mm256_and_ps(expand_mask!(vn0, ctrl, bitsel), in_avx),
                );
                acc_p1 = _mm256_add_ps(
                    acc_p1,
                    _mm256_and_ps(expand_mask!(vp1, ctrl, bitsel), in_avx),
                );
                acc_m1 = _mm256_add_ps(
                    acc_m1,
                    _mm256_and_ps(expand_mask!(vn1, ctrl, bitsel), in_avx),
                );
                acc_p2 = _mm256_add_ps(
                    acc_p2,
                    _mm256_and_ps(expand_mask!(vp2, ctrl, bitsel), in_avx),
                );
                acc_m2 = _mm256_add_ps(
                    acc_m2,
                    _mm256_and_ps(expand_mask!(vn2, ctrl, bitsel), in_avx),
                );
                acc_p3 = _mm256_add_ps(
                    acc_p3,
                    _mm256_and_ps(expand_mask!(vp3, ctrl, bitsel), in_avx),
                );
                acc_m3 = _mm256_add_ps(
                    acc_m3,
                    _mm256_and_ps(expand_mask!(vn3, ctrl, bitsel), in_avx),
                );
            }
            col_packed += 8;
            col += 32;
        }

        // Unrolled by hand, NOT an array of (__m256, __m256) tuples walked by
        // iterator: on the soft-float x86_64-unknown-uefi ABI, LLVM cannot
        // move vector values through aggregate/function boundaries (same
        // "split the result" backend error as the expand_mask! note).
        let mut sp = [0.0f32; 8];
        let mut sm = [0.0f32; 8];
        let mut fs_p = [0.0f32; 4];
        let mut fs_m = [0.0f32; 4];
        _mm256_storeu_ps(sp.as_mut_ptr(), acc_p0);
        _mm256_storeu_ps(sm.as_mut_ptr(), acc_m0);
        fs_p[0] = sp.iter().sum::<f32>();
        fs_m[0] = sm.iter().sum::<f32>();
        _mm256_storeu_ps(sp.as_mut_ptr(), acc_p1);
        _mm256_storeu_ps(sm.as_mut_ptr(), acc_m1);
        fs_p[1] = sp.iter().sum::<f32>();
        fs_m[1] = sm.iter().sum::<f32>();
        _mm256_storeu_ps(sp.as_mut_ptr(), acc_p2);
        _mm256_storeu_ps(sm.as_mut_ptr(), acc_m2);
        fs_p[2] = sp.iter().sum::<f32>();
        fs_m[2] = sm.iter().sum::<f32>();
        _mm256_storeu_ps(sp.as_mut_ptr(), acc_p3);
        _mm256_storeu_ps(sm.as_mut_ptr(), acc_m3);
        fs_p[3] = sp.iter().sum::<f32>();
        fs_m[3] = sm.iter().sum::<f32>();

        while col_packed < packed_dim_in {
            for r in 0..4 {
                fs_p[r] += tail_quad_plane(input, col, plane_pos, (row + r) * words);
                fs_m[r] += tail_quad_plane(input, col, plane_neg, (row + r) * words);
            }
            col_packed += 1;
            col += 4;
        }
        for r in 0..4 {
            output[row + r] = (fs_p[r] - fs_m[r]) * scale;
        }
        row += 4;
    }

    while row < dim_out {
        let mut acc_p0 = _mm256_setzero_ps();
        let mut acc_m0 = _mm256_setzero_ps();
        let p0 = pos_base.add(row * row_bytes);
        let n0 = neg_base.add(row * row_bytes);
        let mut col_packed = 0;
        let mut col = 0;

        while col_packed + 8 <= packed_dim_in {
            let bo = col_packed / 2;
            let vp0 = _mm256_set1_epi32(core::ptr::read_unaligned(p0.add(bo) as *const u32) as i32);
            let vn0 = _mm256_set1_epi32(core::ptr::read_unaligned(n0.add(bo) as *const u32) as i32);
            for g in 0..4 {
                let in_avx = _mm256_loadu_ps(input.as_ptr().add(col + g * 8));
                let ctrl = _mm256_set1_epi8(g as i8);
                acc_p0 = _mm256_add_ps(
                    acc_p0,
                    _mm256_and_ps(expand_mask!(vp0, ctrl, bitsel), in_avx),
                );
                acc_m0 = _mm256_add_ps(
                    acc_m0,
                    _mm256_and_ps(expand_mask!(vn0, ctrl, bitsel), in_avx),
                );
            }
            col_packed += 8;
            col += 32;
        }

        let mut sp = [0.0f32; 8];
        let mut sm = [0.0f32; 8];
        _mm256_storeu_ps(sp.as_mut_ptr(), acc_p0);
        _mm256_storeu_ps(sm.as_mut_ptr(), acc_m0);
        let mut fs_p = sp.iter().sum::<f32>();
        let mut fs_m = sm.iter().sum::<f32>();
        while col_packed < packed_dim_in {
            fs_p += tail_quad_plane(input, col, plane_pos, row * words);
            fs_m += tail_quad_plane(input, col, plane_neg, row * words);
            col_packed += 1;
            col += 4;
        }
        output[row] = (fs_p - fs_m) * scale;
        row += 1;
    }
}
