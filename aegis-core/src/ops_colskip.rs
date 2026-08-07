// House style (matches ops.rs): the `unsafe fn` bodies below are wall-to-wall
// intrinsics whose single obligation — AVX2+FMA present and OS-enabled,
// buffers sized per the documented contract — is stated once per fn in its
// `# Safety` section, not per intrinsic call.
#![allow(unsafe_op_in_unsafe_fn)]

//! ReLU^2 activation-sparsity column-skip matvec for down_proj (task #27,
//! ledger A15).
//!
//! BitNet-2B's squared-ReLU FFN makes the down_proj INPUT vector 78.9% exact
//! zeros at the moment the kernel consumes it (pooled decode mean z = 0.7888,
//! per-layer 0.667–0.908; raw log docs/hardware_logs/
//! relu2_act_sparsity_bitnet2b_2026-08-01.log). Each zero input element
//! kills an entire COLUMN of the down_proj weight matrix (2560x6912 — 25.5%
//! of per-layer weight bytes). Break-even at pessimistic kernel efficiency
//! eps = 1.0 is z = 0.50; the measured z clears it 2.4x. This module holds
//! the candidate kernels; the same-binary interleaved A/B is
//! `benches/colskip_vs_incumbent.rs`, and the verdict comes from quiet
//! physical hardware (Rules A/B), not this box.
//!
//! # Not a settled negative
//!
//! - NOT the CTZ zero-multiplication kernel (ledger A6): that skipped WEIGHT
//!   zeros (42.21%) with a per-nonzero `trailing_zeros` bit scan inside the
//!   row loop. This skips ACTIVATION zeros (78.9%) at whole-column
//!   granularity with one scalar compare per column, amortized over 2560
//!   rows — no bit scan, no per-nonzero branch in the inner loop.
//! - NOT T-MAC pshufb LUT-mpGEMM (A7): layout stays 2 bits/weight; decode
//!   weight traffic is the incumbent's times (1 - z), never more.
//! - NOT the fused dual/tri matvec (A16) or the bitplane-dense matvec (A17):
//!   different mechanism entirely (input-stationary skipping, not
//!   input-load amortization or mask expansion).
//!
//! # Layout contract: column-major 2-bit packing
//!
//! `pack_colmajor` re-packs the incumbent's row-major 2-bit stream (4
//! weights/byte, codes 00=0, 01=+1, 10=-1, undefined 11 -> 0.0 via the shared
//! `UNPACK_LUT`) into column-major: column `c` occupies
//! `colskip_col_bytes(dim_out)` contiguous bytes starting at
//! `c * colskip_col_bytes(dim_out)`; row `r` of that column is the 2-bit code
//! at bit `2*(r % 4)` of byte `r / 4`. Column coverage mirrors the incumbent
//! exactly: only the first `(dim_in / 4) * 4` columns exist. One column of
//! the real shape = 2560 weights = 640 contiguous bytes, so a skipped column
//! skips 640 contiguous bytes of memory traffic.
//!
//! # Numeric identity (the Rule D story)
//!
//! Skipping a column whose input element is bitwise +0.0 or -0.0 is EXACT
//! for the incumbent's fma accumulation: `fma(+/-0.0, w, s) == s` bitwise for
//! `w` in {-1, 0, +1} and finite `s != -0.0`, and the accumulators can never
//! become -0.0 (they start at +0.0, and under round-to-nearest an add/fma
//! only produces -0.0 when the exact result is -0.0, which needs a -0.0
//! accumulator to begin with). Both facts are PROVED, not assumed, in
//! `tests/colskip_exactness.rs::fma_zero_skip_identity`, and the kernels'
//! scalar mirrors process every column UNCONDITIONALLY — so the required
//! byte-equality between a skipping AVX2 kernel and its non-skipping mirror
//! re-proves the identity end-to-end on every test shape.
//!
//! Two variants, mirroring the bitplane (A17) methodology:
//!
//! - **ordered** (`colskip_matvec_avx2_ordered`): reproduces the incumbent
//!   `ops::ternary_matvec_avx2` per-row rounding sequence exactly — per
//!   output row, eight lane accumulators keyed by `column % 8` (the
//!   incumbent's ymm lanes), fma contributions in ascending column order
//!   within each lane, the same `[f32; 8]` store + `iter().sum()` horizontal
//!   fold, the same scalar-tail quad expression over the same tail columns,
//!   the same final `* scale`. Output is REQUIRED to be byte-identical to
//!   the incumbent; `tests/colskip_exactness.rs` asserts `to_bits()`
//!   equality. Physically it is column-outer: lane accumulators live in a
//!   caller-provided scratch of 8 planes x `dim_out` f32 (80 KB at the real
//!   shape — L2-resident), and each non-zero column updates one plane.
//! - **chain** (`colskip_matvec_avx2_chain`): one accumulator per output row
//!   (the output buffer itself, 10 KB — L1-resident), single fma chain over
//!   ascending non-zero columns. 8x less accumulator traffic than ordered,
//!   but a DIFFERENT rounding order than the incumbent — NOT byte-identical
//!   to it, and instead pinned bit-for-bit to its op-identical scalar
//!   mirror `colskip_matvec_scalar_chain` (exactly how bitplane variant (ii)
//!   was handled).
//!
//! # Wiring status
//!
//! NOT wired into inference.rs — offline repack, kernels, exactness tests
//! and bench only, pending an admissible interleaved A/B on quiet physical
//! hardware with REAL captured activation vectors (the next Dell L-stick).
//!
//! # UEFI build note
//!
//! The AVX2 kernels are cfg-gated off `target_os = "uefi"` like
//! `ops_bitplane`'s: the candidate races on the hosted path first, and the
//! soft-float stock UEFI target must never carry a losing kernel anyway.
//! `pack_colmajor` and both scalar mirrors build on every target.

#[cfg(not(target_os = "uefi"))]
use core::arch::x86_64::*;

use crate::ops::UNPACK_LUT;

/// Bytes per column of the column-major 2-bit layout (4 rows per byte).
pub fn colskip_col_bytes(dim_out: usize) -> usize {
    dim_out.div_ceil(4)
}

/// Columns covered by the packed format — identical to the incumbent's
/// coverage: the 2-bit row-major stream holds 4 weights per byte and
/// truncates the remainder, so only the first `(dim_in / 4) * 4` columns
/// exist in either layout.
pub fn colskip_covered_cols(dim_in: usize) -> usize {
    (dim_in / 4) * 4
}

/// Offline repack: row-major 2-bit packed weights -> column-major 2-bit.
/// Codes are copied VERBATIM (including the undefined 11 code), so the shared
/// `UNPACK_LUT` degrades 11 to 0.0 identically in both layouts.
///
/// `colmajor` must hold at least
/// `colskip_covered_cols(dim_in) * colskip_col_bytes(dim_out)` bytes; that
/// prefix is fully overwritten. One-time cost at model load, like
/// `ops_bitplane::pack_bitplanes`.
pub fn pack_colmajor(weights_packed: &[u8], dim_out: usize, dim_in: usize, colmajor: &mut [u8]) {
    let packed_dim_in = dim_in / 4;
    let cb = colskip_col_bytes(dim_out);
    let cols = colskip_covered_cols(dim_in);
    assert!(
        weights_packed.len() >= dim_out * packed_dim_in,
        "packed weights too short: {} < {}",
        weights_packed.len(),
        dim_out * packed_dim_in
    );
    assert!(
        colmajor.len() >= cols * cb,
        "colmajor buffer too short: {} < {}",
        colmajor.len(),
        cols * cb
    );
    colmajor[..cols * cb].fill(0);

    for row in 0..dim_out {
        for cp in 0..packed_dim_in {
            let byte = weights_packed[row * packed_dim_in + cp];
            for lane in 0..4 {
                let code = (byte >> (2 * lane)) & 3;
                let col = cp * 4 + lane;
                colmajor[col * cb + row / 4] |= code << (2 * (row % 4));
            }
        }
    }
}

/// Weight of row `r` in column `c` of the column-major layout, decoded
/// through the incumbent's `UNPACK_LUT` (so 11 -> 0.0 identically).
#[inline(always)]
fn col_weight(colmajor: &[u8], cb: usize, c: usize, r: usize) -> f32 {
    let b = colmajor[c * cb + r / 4] as usize;
    UNPACK_LUT[b * 4 + (r % 4)]
}

/// Columns consumed by the incumbent's VECTOR loop: 32 per iteration while
/// `col_packed + 8 <= dim_in / 4`. Columns from here to
/// `colskip_covered_cols(dim_in)` are the incumbent's scalar-tail quads.
#[inline(always)]
fn vector_cols(dim_in: usize) -> usize {
    32 * ((dim_in / 4) / 8)
}

/// The incumbent's scalar-tail quad expression for one packed quad starting
/// at `col`, association order verbatim:
/// `((x0*w0 + x1*w1) + x2*w2) + x3*w3` (plain mul/add, NOT fma — the
/// incumbent's tail is scalar Rust arithmetic, which never contracts).
#[inline(always)]
fn tail_quad(input: &[f32], colmajor: &[u8], cb: usize, col: usize, r: usize) -> f32 {
    input[col] * col_weight(colmajor, cb, col, r)
        + input[col + 1] * col_weight(colmajor, cb, col + 1, r)
        + input[col + 2] * col_weight(colmajor, cb, col + 2, r)
        + input[col + 3] * col_weight(colmajor, cb, col + 3, r)
}

/// Scalar mirror of the ORDERED variant — and, by construction, an
/// op-identical simulation of the incumbent `ternary_matvec_avx2` per-row
/// sequence: lane accumulator `c % 8` receives `fma(x[c], w, lane)` in
/// ascending column order (`libm::fmaf` is the same correctly-rounded fused
/// op as `_mm256_fmadd_ps`), then the incumbent's `[f32; 8]` `iter().sum()`
/// fold, tail quads, and `* scale`.
///
/// Processes every column UNCONDITIONALLY — no skipping. Byte-equality
/// between this mirror and the skipping AVX2 kernel is the end-to-end proof
/// that zero-column skipping is exact.
pub fn colskip_matvec_scalar_ordered(
    output: &mut [f32],
    input: &[f32],
    colmajor: &[u8],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    let cb = colskip_col_bytes(dim_out);
    let vec_cols = vector_cols(dim_in);
    let covered = colskip_covered_cols(dim_in);

    for (r, out) in output.iter_mut().take(dim_out).enumerate() {
        let mut lanes = [0.0f32; 8];
        for (c, &x) in input.iter().enumerate().take(vec_cols) {
            lanes[c % 8] = libm::fmaf(x, col_weight(colmajor, cb, c, r), lanes[c % 8]);
        }
        let mut fs = lanes.iter().sum::<f32>();
        let mut col = vec_cols;
        while col + 4 <= covered {
            fs += tail_quad(input, colmajor, cb, col, r);
            col += 4;
        }
        *out = fs * scale;
    }
}

/// Scalar mirror of the CHAIN variant: per output row, one fma chain over
/// ALL covered columns in ascending order (no vector/tail split — the
/// column-major layout treats every column uniformly), then `* scale`.
/// Processes every column unconditionally; the AVX2 chain kernel skips
/// zero-input columns and must match this mirror bit-for-bit.
pub fn colskip_matvec_scalar_chain(
    output: &mut [f32],
    input: &[f32],
    colmajor: &[u8],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    let cb = colskip_col_bytes(dim_out);
    let covered = colskip_covered_cols(dim_in);

    for (r, out) in output.iter_mut().take(dim_out).enumerate() {
        let mut s = 0.0f32;
        for (c, &x) in input.iter().enumerate().take(covered) {
            s = libm::fmaf(x, col_weight(colmajor, cb, c, r), s);
        }
        *out = s * scale;
    }
}

/// Column-skip matvec, ORDERED variant: byte-identical to the incumbent
/// `ops::ternary_matvec_avx2` (asserted in tests/colskip_exactness.rs).
///
/// Column-outer over the incumbent's vector-region columns, skipping any
/// column whose input element is bitwise +/-0.0 (`x == 0.0` compares both
/// true); a non-zero column `c` updates lane plane `c % 8` — 8 rows per
/// `_mm256_fmadd_ps`, scalar `libm::fmaf` for the `dim_out % 8` remainder
/// rows. The fold + tail + scale pass then reproduces the incumbent's
/// per-row sequence exactly (see module docs).
///
/// # Safety
/// Caller must guarantee AVX2+FMA are supported and OS-enabled (see
/// `ops::avx2_active()`), `colmajor` holds
/// `colskip_covered_cols(dim_in) * colskip_col_bytes(dim_out)` bytes in this
/// module's layout, `input.len() >= dim_in`, `output.len() >= dim_out`, and
/// `scratch.len() >= 8 * dim_out` (contents ignored; fully overwritten).
#[cfg(not(target_os = "uefi"))] // hosted-path candidate; see module docs
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn colskip_matvec_avx2_ordered(
    output: &mut [f32],
    input: &[f32],
    colmajor: &[u8],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
    scratch: &mut [f32],
) {
    let cb = colskip_col_bytes(dim_out);
    let vec_cols = vector_cols(dim_in);
    let covered = colskip_covered_cols(dim_in);
    let lut_ptr = UNPACK_LUT.as_ptr();

    scratch[..8 * dim_out].fill(0.0);
    let scratch_ptr = scratch.as_mut_ptr();

    for (c, &xc) in input.iter().enumerate().take(vec_cols) {
        // The skip: bitwise +0.0 and -0.0 both compare equal to 0.0, and
        // skipping either is exact (module docs; proved in tests).
        if xc == 0.0 {
            continue;
        }
        let xv = _mm256_set1_ps(xc);
        let col_ptr = colmajor.as_ptr().add(c * cb);
        let plane = scratch_ptr.add((c % 8) * dim_out);

        let mut r = 0;
        while r + 8 <= dim_out {
            let b0 = *col_ptr.add(r / 4) as usize;
            let b1 = *col_ptr.add(r / 4 + 1) as usize;
            let w = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(lut_ptr.add(b0 * 4))),
                _mm_loadu_ps(lut_ptr.add(b1 * 4)),
                1,
            );
            let acc = _mm256_loadu_ps(plane.add(r));
            _mm256_storeu_ps(plane.add(r), _mm256_fmadd_ps(xv, w, acc));
            r += 8;
        }
        while r < dim_out {
            let b = *col_ptr.add(r / 4) as usize;
            let w = UNPACK_LUT[b * 4 + (r % 4)];
            *plane.add(r) = libm::fmaf(xc, w, *plane.add(r));
            r += 1;
        }
    }

    // Fold + tail + scale, per row, in the incumbent's exact order.
    for (r, out) in output.iter_mut().take(dim_out).enumerate() {
        let mut lanes = [0.0f32; 8];
        for (k, lane) in lanes.iter_mut().enumerate() {
            *lane = *scratch_ptr.add(k * dim_out + r);
        }
        let mut fs = lanes.iter().sum::<f32>();
        let mut col = vec_cols;
        while col + 4 <= covered {
            fs += tail_quad(input, colmajor, cb, col, r);
            col += 4;
        }
        *out = fs * scale;
    }
}

/// Column-skip matvec, CHAIN variant: single accumulator per output row (the
/// output buffer itself), one fma chain over ascending non-zero columns.
/// NOT byte-identical to the incumbent (different rounding order); pinned
/// bit-for-bit to `colskip_matvec_scalar_chain` instead.
///
/// # Safety
/// Caller must guarantee AVX2+FMA are supported and OS-enabled, `colmajor`
/// holds `colskip_covered_cols(dim_in) * colskip_col_bytes(dim_out)` bytes
/// in this module's layout, `input.len() >= dim_in`, and
/// `output.len() >= dim_out` (fully overwritten).
#[cfg(not(target_os = "uefi"))] // hosted-path candidate; see module docs
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn colskip_matvec_avx2_chain(
    output: &mut [f32],
    input: &[f32],
    colmajor: &[u8],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    let cb = colskip_col_bytes(dim_out);
    let covered = colskip_covered_cols(dim_in);
    let lut_ptr = UNPACK_LUT.as_ptr();

    output[..dim_out].fill(0.0);
    let out_ptr = output.as_mut_ptr();

    for (c, &xc) in input.iter().enumerate().take(covered) {
        if xc == 0.0 {
            continue;
        }
        let xv = _mm256_set1_ps(xc);
        let col_ptr = colmajor.as_ptr().add(c * cb);

        let mut r = 0;
        while r + 8 <= dim_out {
            let b0 = *col_ptr.add(r / 4) as usize;
            let b1 = *col_ptr.add(r / 4 + 1) as usize;
            let w = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(lut_ptr.add(b0 * 4))),
                _mm_loadu_ps(lut_ptr.add(b1 * 4)),
                1,
            );
            let acc = _mm256_loadu_ps(out_ptr.add(r));
            _mm256_storeu_ps(out_ptr.add(r), _mm256_fmadd_ps(xv, w, acc));
            r += 8;
        }
        while r < dim_out {
            let b = *col_ptr.add(r / 4) as usize;
            let w = UNPACK_LUT[b * 4 + (r % 4)];
            *out_ptr.add(r) = libm::fmaf(xc, w, *out_ptr.add(r));
            r += 1;
        }
    }

    for out in output.iter_mut().take(dim_out) {
        *out *= scale;
    }
}
