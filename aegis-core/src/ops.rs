#![allow(unsafe_op_in_unsafe_fn)]
use core::arch::x86_64::*;

/// # Safety
/// Caller must guarantee AVX2+FMA are supported and OS-enabled (see
/// `avx2_active()`); `weight_bytes` must hold `input.len()` elements at the
/// engine's derived element size.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rmsnorm_avx2(input: &mut [f32], weight_bytes: &[u8], eps: f32) {
    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0;

    // Vectorized sum of squares
    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        sum_vec = _mm256_fmadd_ps(x, x, sum_vec);
        i += 8;
    }

    let mut s = [0.0f32; 8];
    _mm256_storeu_ps(s.as_mut_ptr(), sum_vec);
    let mut sum_sq = s.iter().sum::<f32>();

    while i < input.len() {
        sum_sq += input[i] * input[i];
        i += 1;
    }

    let rms = libm::sqrtf(sum_sq / input.len() as f32 + eps);
    let inv_rms = _mm256_set1_ps(1.0 / rms);

    let bytes_per_elem = weight_bytes.len() / input.len();
    i = 0;

    if bytes_per_elem == 2 {
        while i + 8 <= input.len() {
            let in_vec = _mm256_loadu_ps(input.as_ptr().add(i));
            let bf16_128 = _mm_loadu_si128(
                weight_bytes.as_ptr().add(i * 2) as *const core::arch::x86_64::__m128i
            );
            let w_vec = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_128), 16));

            let res = _mm256_mul_ps(_mm256_mul_ps(in_vec, inv_rms), w_vec);
            _mm256_storeu_ps(input.as_mut_ptr().add(i), res);
            i += 8;
        }
        while i < input.len() {
            let offset = i * 2;
            let w = f32_from_bytes(0, 0, weight_bytes[offset], weight_bytes[offset + 1]);
            input[i] = input[i] * (1.0 / rms) * w;
            i += 1;
        }
    } else {
        while i + 8 <= input.len() {
            let in_vec = _mm256_loadu_ps(input.as_ptr().add(i));
            let w_vec = _mm256_loadu_ps(weight_bytes.as_ptr().add(i * 4) as *const f32);
            let res = _mm256_mul_ps(_mm256_mul_ps(in_vec, inv_rms), w_vec);
            _mm256_storeu_ps(input.as_mut_ptr().add(i), res);
            i += 8;
        }
        while i < input.len() {
            let offset = i * 4;
            let w = f32_from_bytes(
                weight_bytes[offset],
                weight_bytes[offset + 1],
                weight_bytes[offset + 2],
                weight_bytes[offset + 3],
            );
            input[i] = input[i] * (1.0 / rms) * w;
            i += 1;
        }
    }
}

pub fn rmsnorm(input: &mut [f32], weight_bytes: &[u8], eps: f32) {
    if input.is_empty() || weight_bytes.len() < input.len() * (weight_bytes.len() / input.len()) {
        return;
    }
    if simd_on() {
        unsafe {
            rmsnorm_avx2(input, weight_bytes, eps);
        }
    } else {
        rmsnorm_scalar(input, weight_bytes, eps);
    }
}

pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_val = x[0];
    for &val in x.iter() {
        if val > max_val {
            max_val = val;
        }
    }

    let mut sum = 0.0;
    for val in x.iter_mut() {
        *val = libm::expf(*val - max_val);
        sum += *val;
    }

    for val in x.iter_mut() {
        *val /= sum;
    }
}

/// Per-token absmax int8 activation quantization, applied in place as a
/// quantize→dequantize round trip (values land on the int8 grid but stay f32).
///
/// This is what the BitNet reference does before every BitLinear: the weights
/// were trained with this quantization in the loop via straight-through
/// estimation, so they *expect* the resulting grid. Feeding full-precision f32
/// activations is the deviation from the reference, not the other way round.
///
/// Clamped to [-127, 127], never -128: `_mm256_sign_epi8` saturates -(-128) to
/// +127, so the asymmetric end of the int8 range would silently flip sign in a
/// future integer kernel. Staying symmetric keeps this simulation and that
/// kernel numerically identical.
pub fn quantize_activations_int8(x: &mut [f32]) {
    let mut absmax = 0.0f32;
    for &v in x.iter() {
        let a = if v < 0.0 { -v } else { v };
        if a > absmax {
            absmax = a;
        }
    }
    if absmax <= 0.0 {
        return;
    }

    let s = 127.0 / absmax;
    let inv_s = 1.0 / s;
    for v in x.iter_mut() {
        let q = libm::roundf(*v * s).clamp(-127.0, 127.0);
        *v = q * inv_s;
    }
}

pub fn relu2(x: &mut [f32]) {
    for val in x.iter_mut() {
        if *val > 0.0 {
            *val = (*val) * (*val);
        } else {
            *val = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime SIMD dispatch: detect AVX2+FMA (and that the OS/firmware actually
// enabled AVX state via XCR0) once at engine init. Every public op falls back
// to a portable scalar path (auto-vectorized to SSE2, the x86_64 baseline)
// when AVX2 is unavailable — so the same binary runs on any x86_64 machine.
// ---------------------------------------------------------------------------
use core::sync::atomic::{AtomicU8, Ordering};

/// 0 = not yet probed, 1 = AVX2 unavailable, 2 = AVX2+FMA usable.
/// Atomic rather than `static mut`: the probe is idempotent, so a benign race
/// between threads is fine, but reading/writing a plain `static mut` from more
/// than one thread is undefined behavior regardless of the values involved.
static SIMD_STATE: AtomicU8 = AtomicU8::new(0);

fn probe_simd() -> u8 {
    unsafe {
        let l1 = core::arch::x86_64::__cpuid(1);
        let osxsave_enabled = (l1.ecx & (1 << 27)) != 0; // OS has set CR4.OSXSAVE
        let avx = (l1.ecx & (1 << 28)) != 0;
        let fma = (l1.ecx & (1 << 12)) != 0;
        let l7 = core::arch::x86_64::__cpuid_count(7, 0);
        let avx2 = (l7.ebx & (1 << 5)) != 0;

        // xgetbv faults with #UD unless OSXSAVE is enabled — check first.
        let ymm_state_ok = if osxsave_enabled {
            let eax: u32;
            let edx: u32;
            core::arch::asm!("xgetbv", in("ecx") 0u32, out("eax") eax, out("edx") edx);
            let _ = edx;
            (eax & 0b110) == 0b110 // XMM + YMM state enabled in XCR0
        } else {
            false
        };

        if avx && avx2 && fma && ymm_state_ok {
            2
        } else {
            1
        }
    }
}

pub fn init_simd() {
    let _ = avx2_active();
}

pub fn avx2_active() -> bool {
    match SIMD_STATE.load(Ordering::Relaxed) {
        0 => {
            let state = probe_simd();
            SIMD_STATE.store(state, Ordering::Relaxed);
            state == 2
        }
        s => s == 2,
    }
}

// ---------------------------------------------------------------------------
// Runtime race toggles. All default OFF, so a normal run dispatches exactly as
// before (byte-identical, still guarded by the coherence/equivalence tests).
// They exist so the fleet binary can race the scalar path against the AVX2
// path, and the per-token prefill against the batched GEMM, IN ONE BOOT — the
// compile-time features can't be rebuilt on a borrowed laptop.
// ---------------------------------------------------------------------------
static FORCE_SCALAR: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static FORCE_LEGACY_PREFILL: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn set_force_scalar(v: bool) {
    FORCE_SCALAR.store(v, Ordering::Relaxed);
}
pub fn set_force_legacy_prefill(v: bool) {
    FORCE_LEGACY_PREFILL.store(v, Ordering::Relaxed);
}

/// The dispatch gate: AVX2 is used only if the CPU supports it AND scalar has
/// not been forced for a race. `avx2_active()` remains the pure capability query.
#[inline]
pub(crate) fn simd_on() -> bool {
    avx2_active() && !FORCE_SCALAR.load(Ordering::Relaxed)
}

/// Capability of the silicon, independent of the race toggle.
pub fn simd_level_name() -> &'static str {
    if avx2_active() {
        "AVX2+FMA"
    } else {
        "SSE2 (scalar fallback)"
    }
}

/// The path actually in use right now, honoring the force-scalar toggle.
pub fn active_path_name() -> &'static str {
    if simd_on() { "AVX2+FMA" } else { "scalar" }
}

#[inline]
fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_le_bytes([0, 0, lo, hi])
}

fn rmsnorm_scalar(input: &mut [f32], weight_bytes: &[u8], eps: f32) {
    let mut sum_sq = 0.0f32;
    for &x in input.iter() {
        sum_sq += x * x;
    }
    let rms = libm::sqrtf(sum_sq / input.len() as f32 + eps);
    let inv_rms = 1.0 / rms;

    let bytes_per_elem = weight_bytes.len() / input.len();
    if bytes_per_elem == 2 {
        for i in 0..input.len() {
            let w = bf16_to_f32(weight_bytes[i * 2], weight_bytes[i * 2 + 1]);
            input[i] = input[i] * inv_rms * w;
        }
    } else {
        for i in 0..input.len() {
            let o = i * 4;
            let w = f32::from_le_bytes([
                weight_bytes[o],
                weight_bytes[o + 1],
                weight_bytes[o + 2],
                weight_bytes[o + 3],
            ]);
            input[i] = input[i] * inv_rms * w;
        }
    }
}

fn ternary_matvec_scalar(
    output: &mut [f32],
    input: &[f32],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    unsafe {
        init_unpack_lut();
    }
    let lut = &UNPACK_LUT;
    let packed_dim_in = dim_in / 4;

    for row in 0..dim_out {
        let w_row = &weights_packed[row * packed_dim_in..(row + 1) * packed_dim_in];
        let mut sum = 0.0f32;
        let mut col = 0;
        for &b in w_row.iter() {
            let l = b as usize * 4;
            sum += input[col] * lut[l]
                + input[col + 1] * lut[l + 1]
                + input[col + 2] * lut[l + 2]
                + input[col + 3] * lut[l + 3];
            col += 4;
        }
        output[row] = sum * scale;
    }
}

fn f32_dot_scalar(
    output: &mut [f32],
    input: &[f32],
    embeddings: &[u8],
    vocab_size: usize,
    emb_dim: usize,
) -> u32 {
    let mut max_val = -f32::INFINITY;
    let mut max_idx = 0u32;
    for row in 0..vocab_size {
        let start = row * emb_dim * 2;
        let mut sum = 0.0f32;
        for col in 0..emb_dim {
            let o = start + col * 2;
            sum += input[col] * bf16_to_f32(embeddings[o], embeddings[o + 1]);
        }
        output[row] = sum;
        if sum > max_val {
            max_val = sum;
            max_idx = row as u32;
        }
    }
    max_idx
}

/// Attention dot product over one head. Dispatches to the AVX2 kernel when the
/// head dim is 128 (the BitNet case) and AVX2 is active; otherwise scalar.
pub fn attn_dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() == 128 && simd_on() {
        return unsafe { vector_dot_128(a, b) };
    }
    let mut sum = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        sum += a[i] * b[i];
    }
    sum
}

/// Attention weighted accumulate: out += w * v. Same dispatch rule as attn_dot.
pub fn attn_madd(out: &mut [f32], w: f32, v: &[f32]) {
    if out.len() == 128 && v.len() >= 128 && simd_on() {
        unsafe { vector_madd_128(out, w, v) };
        return;
    }
    for i in 0..out.len().min(v.len()) {
        out[i] += w * v[i];
    }
}

/// Byte -> four ternary weights, as f32. Computed at compile time: a `const`
/// has no initialization race and no per-call `LUT_INIT` check, which is what
/// makes the kernels safe to call from multiple threads.
///
/// Codes: 00 = 0, 01 = +1, 10 = -1. The 11 code is undefined by the format and
/// maps to 0.0, so a corrupt weight byte degrades gracefully rather than
/// injecting a +3 weight.
pub(crate) static UNPACK_LUT: [f32; 1024] = build_unpack_lut();

const fn build_unpack_lut() -> [f32; 1024] {
    const fn decode(w: usize) -> f32 {
        match w {
            1 => 1.0,
            2 => -1.0,
            _ => 0.0,
        }
    }
    let mut lut = [0.0f32; 1024];
    let mut b = 0;
    while b < 256 {
        lut[b * 4] = decode(b & 3);
        lut[b * 4 + 1] = decode((b >> 2) & 3);
        lut[b * 4 + 2] = decode((b >> 4) & 3);
        lut[b * 4 + 3] = decode((b >> 6) & 3);
        b += 1;
    }
    lut
}

/// Retained as a no-op so existing call sites keep compiling; the table is now
/// built at compile time.
/// # Safety
/// Trivially safe (the LUT is `const` now); kept `unsafe` for call-site
/// compatibility with the historical runtime-init API.
#[inline(always)]
pub unsafe fn init_unpack_lut() {}

/// # Safety
/// Caller must guarantee AVX2+FMA are supported and OS-enabled, and that
/// `weights_packed` holds `dim_out * ceil(dim_in/4)` packed bytes with
/// `input.len() >= dim_in` and `output.len() >= dim_out`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn ternary_matvec_avx2(
    output: &mut [f32],
    input: &[f32],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    init_unpack_lut();
    let lut_ptr = UNPACK_LUT.as_ptr();

    let mut row = 0;
    let packed_dim_in = dim_in / 4;

    while row + 3 < dim_out {
        let mut sum0 = core::arch::x86_64::_mm256_setzero_ps();
        let mut sum1 = core::arch::x86_64::_mm256_setzero_ps();
        let mut sum2 = core::arch::x86_64::_mm256_setzero_ps();
        let mut sum3 = core::arch::x86_64::_mm256_setzero_ps();

        let w_ptr0 = weights_packed.as_ptr().add(row * packed_dim_in);
        let w_ptr1 = weights_packed.as_ptr().add((row + 1) * packed_dim_in);
        let w_ptr2 = weights_packed.as_ptr().add((row + 2) * packed_dim_in);
        let w_ptr3 = weights_packed.as_ptr().add((row + 3) * packed_dim_in);

        let mut col_packed = 0;
        let mut col = 0;

        while col_packed + 8 <= packed_dim_in {
            let p0 = core::ptr::read_unaligned(w_ptr0.add(col_packed) as *const u64);
            let p1 = core::ptr::read_unaligned(w_ptr1.add(col_packed) as *const u64);
            let p2 = core::ptr::read_unaligned(w_ptr2.add(col_packed) as *const u64);
            let p3 = core::ptr::read_unaligned(w_ptr3.add(col_packed) as *const u64);

            for i in 0..4 {
                let shift = i * 16;
                let in_avx = core::arch::x86_64::_mm256_loadu_ps(input.as_ptr().add(col + i * 8));

                let b0_0 = ((p0 >> shift) & 0xFF) as usize;
                let b0_1 = ((p0 >> (shift + 8)) & 0xFF) as usize;
                let w_avx0 = core::arch::x86_64::_mm256_insertf128_ps(
                    core::arch::x86_64::_mm256_castps128_ps256(core::arch::x86_64::_mm_loadu_ps(
                        lut_ptr.add(b0_0 * 4),
                    )),
                    core::arch::x86_64::_mm_loadu_ps(lut_ptr.add(b0_1 * 4)),
                    1,
                );
                sum0 = core::arch::x86_64::_mm256_fmadd_ps(in_avx, w_avx0, sum0);

                let b1_0 = ((p1 >> shift) & 0xFF) as usize;
                let b1_1 = ((p1 >> (shift + 8)) & 0xFF) as usize;
                let w_avx1 = core::arch::x86_64::_mm256_insertf128_ps(
                    core::arch::x86_64::_mm256_castps128_ps256(core::arch::x86_64::_mm_loadu_ps(
                        lut_ptr.add(b1_0 * 4),
                    )),
                    core::arch::x86_64::_mm_loadu_ps(lut_ptr.add(b1_1 * 4)),
                    1,
                );
                sum1 = core::arch::x86_64::_mm256_fmadd_ps(in_avx, w_avx1, sum1);

                let b2_0 = ((p2 >> shift) & 0xFF) as usize;
                let b2_1 = ((p2 >> (shift + 8)) & 0xFF) as usize;
                let w_avx2 = core::arch::x86_64::_mm256_insertf128_ps(
                    core::arch::x86_64::_mm256_castps128_ps256(core::arch::x86_64::_mm_loadu_ps(
                        lut_ptr.add(b2_0 * 4),
                    )),
                    core::arch::x86_64::_mm_loadu_ps(lut_ptr.add(b2_1 * 4)),
                    1,
                );
                sum2 = core::arch::x86_64::_mm256_fmadd_ps(in_avx, w_avx2, sum2);

                let b3_0 = ((p3 >> shift) & 0xFF) as usize;
                let b3_1 = ((p3 >> (shift + 8)) & 0xFF) as usize;
                let w_avx3 = core::arch::x86_64::_mm256_insertf128_ps(
                    core::arch::x86_64::_mm256_castps128_ps256(core::arch::x86_64::_mm_loadu_ps(
                        lut_ptr.add(b3_0 * 4),
                    )),
                    core::arch::x86_64::_mm_loadu_ps(lut_ptr.add(b3_1 * 4)),
                    1,
                );
                sum3 = core::arch::x86_64::_mm256_fmadd_ps(in_avx, w_avx3, sum3);
            }
            col_packed += 8;
            col += 32;
        }

        let mut s0 = [0.0f32; 8];
        let mut s1 = [0.0f32; 8];
        let mut s2 = [0.0f32; 8];
        let mut s3 = [0.0f32; 8];
        core::arch::x86_64::_mm256_storeu_ps(s0.as_mut_ptr(), sum0);
        core::arch::x86_64::_mm256_storeu_ps(s1.as_mut_ptr(), sum1);
        core::arch::x86_64::_mm256_storeu_ps(s2.as_mut_ptr(), sum2);
        core::arch::x86_64::_mm256_storeu_ps(s3.as_mut_ptr(), sum3);
        let mut fs0 = s0.iter().sum::<f32>();
        let mut fs1 = s1.iter().sum::<f32>();
        let mut fs2 = s2.iter().sum::<f32>();
        let mut fs3 = s3.iter().sum::<f32>();

        while col_packed < packed_dim_in {
            let b0 = *w_ptr0.add(col_packed) as usize;
            let b1 = *w_ptr1.add(col_packed) as usize;
            let b2 = *w_ptr2.add(col_packed) as usize;
            let b3 = *w_ptr3.add(col_packed) as usize;

            fs0 += input[col] * UNPACK_LUT[b0 * 4]
                + input[col + 1] * UNPACK_LUT[b0 * 4 + 1]
                + input[col + 2] * UNPACK_LUT[b0 * 4 + 2]
                + input[col + 3] * UNPACK_LUT[b0 * 4 + 3];
            fs1 += input[col] * UNPACK_LUT[b1 * 4]
                + input[col + 1] * UNPACK_LUT[b1 * 4 + 1]
                + input[col + 2] * UNPACK_LUT[b1 * 4 + 2]
                + input[col + 3] * UNPACK_LUT[b1 * 4 + 3];
            fs2 += input[col] * UNPACK_LUT[b2 * 4]
                + input[col + 1] * UNPACK_LUT[b2 * 4 + 1]
                + input[col + 2] * UNPACK_LUT[b2 * 4 + 2]
                + input[col + 3] * UNPACK_LUT[b2 * 4 + 3];
            fs3 += input[col] * UNPACK_LUT[b3 * 4]
                + input[col + 1] * UNPACK_LUT[b3 * 4 + 1]
                + input[col + 2] * UNPACK_LUT[b3 * 4 + 2]
                + input[col + 3] * UNPACK_LUT[b3 * 4 + 3];

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
        let mut sum0 = core::arch::x86_64::_mm256_setzero_ps();
        let w_ptr0 = weights_packed.as_ptr().add(row * packed_dim_in);
        let mut col_packed = 0;
        let mut col = 0;

        while col_packed + 8 <= packed_dim_in {
            let p0 = core::ptr::read_unaligned(w_ptr0.add(col_packed) as *const u64);
            for i in 0..4 {
                let shift = i * 16;
                let in_avx = core::arch::x86_64::_mm256_loadu_ps(input.as_ptr().add(col + i * 8));
                let b0_0 = ((p0 >> shift) & 0xFF) as usize;
                let b0_1 = ((p0 >> (shift + 8)) & 0xFF) as usize;
                let w_avx0 = core::arch::x86_64::_mm256_insertf128_ps(
                    core::arch::x86_64::_mm256_castps128_ps256(core::arch::x86_64::_mm_loadu_ps(
                        lut_ptr.add(b0_0 * 4),
                    )),
                    core::arch::x86_64::_mm_loadu_ps(lut_ptr.add(b0_1 * 4)),
                    1,
                );
                sum0 = core::arch::x86_64::_mm256_fmadd_ps(in_avx, w_avx0, sum0);
            }
            col_packed += 8;
            col += 32;
        }

        let mut s0 = [0.0f32; 8];
        core::arch::x86_64::_mm256_storeu_ps(s0.as_mut_ptr(), sum0);
        let mut fs0 = s0.iter().sum::<f32>();

        while col_packed < packed_dim_in {
            let b0 = *w_ptr0.add(col_packed) as usize;
            fs0 += input[col] * UNPACK_LUT[b0 * 4]
                + input[col + 1] * UNPACK_LUT[b0 * 4 + 1]
                + input[col + 2] * UNPACK_LUT[b0 * 4 + 2]
                + input[col + 3] * UNPACK_LUT[b0 * 4 + 3];
            col_packed += 1;
            col += 4;
        }
        output[row] = fs0 * scale;
        row += 1;
    }
}

/// Single-threaded kernel dispatch for one contiguous block of output rows.
fn ternary_matvec_serial(
    output: &mut [f32],
    input: &[f32],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    if simd_on() {
        unsafe {
            ternary_matvec_avx2(output, input, weights_packed, dim_out, dim_in, scale);
        }
    } else {
        ternary_matvec_scalar(output, input, weights_packed, dim_out, dim_in, scale);
    }
}

/// Below this many MACs, thread dispatch costs more than it saves.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_MACS: usize = 1 << 21; // ~2M

/// SMT siblings per physical core, from CPUID leaf 0xB (x2APIC topology).
/// Currently unused — kept because thread-count policy may need it on CPUs
/// where SMT does hurt these kernels.
#[allow(dead_code)]
/// Returns 1 when the leaf is unsupported or reports nothing useful.
#[cfg(feature = "parallel")]
fn smt_threads_per_core() -> usize {
    unsafe {
        let max_leaf = core::arch::x86_64::__cpuid(0).eax;
        if max_leaf < 0xB {
            return 1;
        }
        // Subleaf 0 describes the SMT level; EBX[15:0] = logical procs at it.
        let l = core::arch::x86_64::__cpuid_count(0xB, 0);
        let n = (l.ebx & 0xFFFF) as usize;
        if n == 0 { 1 } else { n }
    }
}

/// Worker threads to use, cached after the first query.
///
/// Defaults to LOGICAL processors. Measured on this hardware (4 cores / 8
/// threads), 8 workers decode ~5% faster than 4 — the ternary matvec stalls
/// often enough on LUT loads that an SMT sibling has useful work to fill.
///
/// (An earlier measurement suggested the opposite. It was taken while a runaway
/// process held a core, which made 4 workers oversubscribe. Re-measured on an
/// idle machine, SMT is a small win. Kept as a warning: verify the machine is
/// quiet before trusting any scaling number.)
///
/// Override with `AEGIS_THREADS`.
#[cfg(feature = "parallel")]
pub fn worker_threads() -> usize {
    use core::sync::atomic::AtomicUsize;
    static N: AtomicUsize = AtomicUsize::new(0);
    match N.load(Ordering::Relaxed) {
        0 => {
            let n = std::env::var("AEGIS_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v > 0)
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|v| v.get())
                        .unwrap_or(1)
                });
            N.store(n, Ordering::Relaxed);
            n
        }
        n => n,
    }
}

/// Ternary matrix-vector product, row-parallel when it pays for itself.
///
/// Output rows partition into disjoint chunks, each chunk needs only its own
/// slice of the weight matrix, and the input vector is read-only and shared —
/// so the split needs no locking and no reduction. This is 94% of decode
/// compute, so it is the whole multicore story.
pub fn ternary_matvec(
    output: &mut [f32],
    input: &[f32],
    weights_packed: &[u8],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    if output.len() < dim_out
        || input.len() < dim_in
        || weights_packed.len() < (dim_out * dim_in) / 4
    {
        return;
    }

    #[cfg(feature = "parallel")]
    {
        let pool = crate::pool::global();
        if pool.workers() > 1 && dim_out * dim_in >= PARALLEL_MIN_MACS {
            let packed_dim_in = dim_in / 4;
            let out_base = OutPtr(output.as_mut_ptr());
            pool.broadcast(|id, n| {
                let (row0, rows) = row_split(dim_out, id, n);
                if rows == 0 {
                    return;
                }
                // SAFETY: row ranges are disjoint across workers, so no two
                // threads ever touch the same output element.
                let chunk =
                    unsafe { core::slice::from_raw_parts_mut(out_base.get().add(row0), rows) };
                let w = &weights_packed[row0 * packed_dim_in..(row0 + rows) * packed_dim_in];
                ternary_matvec_serial(chunk, input, w, rows, dim_in, scale);
            });
            return;
        }
    }

    ternary_matvec_serial(output, input, weights_packed, dim_out, dim_in, scale);
}

/// Disjoint row range for worker `id` of `n`, aligned to the kernel's 4-row
/// unroll so every worker hits the fast path.
#[cfg(feature = "parallel")]
#[inline]
fn row_split(dim_out: usize, id: usize, n: usize) -> (usize, usize) {
    let per = (dim_out.div_ceil(n) + 3) & !3;
    let per = per.max(4);
    let row0 = id * per;
    if row0 >= dim_out {
        return (dim_out, 0);
    }
    (row0, core::cmp::min(per, dim_out - row0))
}

/// Send-able base pointer for an output buffer that workers slice disjointly.
///
/// Access goes through `get()` rather than the tuple field: closures capture
/// individual fields, so touching `.0` would capture a bare `*mut f32` (not
/// `Sync`) instead of this wrapper.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
#[cfg(feature = "parallel")]
impl OutPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}
#[cfg(feature = "parallel")]
unsafe impl Send for OutPtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for OutPtr {}

/// Send-able base pointer for the per-worker argmax result slots.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct PairPtr(*mut (u32, f32));
#[cfg(feature = "parallel")]
impl PairPtr {
    #[inline(always)]
    fn get(self) -> *mut (u32, f32) {
        self.0
    }
}
#[cfg(feature = "parallel")]
unsafe impl Send for PairPtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for PairPtr {}

// ---------------------------------------------------------------------------
// Fused-projection matvec candidates (dual: SwiGLU gate+up; tri: Q/K/V).
//
// Motivation: gate_proj/up_proj — and attention's Q/K/V — consume the SAME
// quantized input vector. The sequential path issues one 8-lane input load
// per 4-row block per matrix; the fused kernel pairs a 4-row block from every
// matrix so each input load feeds M*4 FMA chains, cutting input-load issue
// count by M and amortizing loop overhead. Honest prior: the input vector is
// KB-scale and cache-resident while the weights dominate memory traffic
// (weight traffic is unchanged by construction), so the expected gain is
// small — this exists to give the recurring "dual matvec" question a measured
// answer rather than a shrug.
//
// Bit-exactness contract: every output row executes the incumbent
// `ternary_matvec_avx2` per-row arithmetic verbatim — same FMA chain over the
// same columns, same store-then-fold horizontal sum, same scalar-tail
// expression, same final scale — so outputs are REQUIRED to be byte-identical
// to sequential incumbent calls. Any deviation is a bug;
// tests/fused_matvec_exactness.rs asserts `to_bits()` equality.
//
// Known structural cost (qualitative, from reading the release disassembly,
// not a timing claim): the ymm side is fine — all M*4 accumulators stay
// register-resident with one shared input load per pass — but the GPR side is
// not. Eight (twelve for M=3) row pointers plus their packed-u64 values
// exceed the 16 GPRs x86-64 has, so the inner loop reloads base addresses
// from the stack; the incumbent's 4-row shape fits and does not. Fusing more
// rows per input load therefore intrinsically trades input-load amortization
// against GPR spill traffic on this ISA. Whether that nets out ahead is
// exactly what benches/fused_vs_sequential.rs measures.
//
// NOT wired into inference.rs — kernel + tests + bench only, pending an
// admissible interleaved A/B on quiet physical hardware (Rules A/B).
// ---------------------------------------------------------------------------

/// Expand two packed weight bytes (8 ternary weights) from `$p` at bit
/// `$shift` into 8 f32 lanes via the shared 256x4 LUT — the incumbent
/// kernel's unpack step, token for token.
macro_rules! lut8 {
    ($lut:expr, $p:expr, $shift:expr) => {
        core::arch::x86_64::_mm256_insertf128_ps(
            core::arch::x86_64::_mm256_castps128_ps256(core::arch::x86_64::_mm_loadu_ps(
                $lut.add(((($p >> $shift) & 0xFF) as usize) * 4),
            )),
            core::arch::x86_64::_mm_loadu_ps(
                $lut.add(((($p >> ($shift + 8)) & 0xFF) as usize) * 4),
            ),
            1,
        )
    };
}

/// Fused inner loop: processes `blocks * 4` leading rows of each of the `M`
/// matrices, interleaving one 4-row block from every matrix per pass so each
/// 8-lane input load feeds `M * 4` FMA chains.
///
/// # Safety
/// Caller must guarantee AVX2+FMA are supported and OS-enabled, and for every
/// `m`: `weights[m].len() >= blocks * 4 * (dim_in / 4)`,
/// `outs[m].len() >= blocks * 4`, and `input.len() >= dim_in`.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn ternary_matvec_fusedn_avx2<const M: usize>(
    outs: &mut [&mut [f32]; M],
    input: &[f32],
    weights: [&[u8]; M],
    scales: [f32; M],
    blocks: usize,
    dim_in: usize,
) {
    use core::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let lut_ptr = UNPACK_LUT.as_ptr();
    let packed_dim_in = dim_in / 4;

    for blk in 0..blocks {
        let row = blk * 4;

        // M * 4 independent accumulator chains; each row's chain receives
        // exactly the FMA sequence the incumbent would give it.
        let mut acc = [[_mm256_setzero_ps(); 4]; M];
        let mut w_ptr = [[core::ptr::null::<u8>(); 4]; M];
        for m in 0..M {
            for (r, wp) in w_ptr[m].iter_mut().enumerate() {
                *wp = weights[m].as_ptr().add((row + r) * packed_dim_in);
            }
        }

        let mut col_packed = 0;
        let mut col = 0;

        while col_packed + 8 <= packed_dim_in {
            let mut p = [[0u64; 4]; M];
            for m in 0..M {
                for r in 0..4 {
                    p[m][r] = core::ptr::read_unaligned(w_ptr[m][r].add(col_packed) as *const u64);
                }
            }
            for i in 0..4 {
                let shift = i * 16;
                // ONE input load feeds all M matrices' row blocks — the
                // sequential path re-issues this load per matrix. This line
                // is the entire mechanism under test.
                let in_avx = _mm256_loadu_ps(input.as_ptr().add(col + i * 8));
                for m in 0..M {
                    for r in 0..4 {
                        acc[m][r] =
                            _mm256_fmadd_ps(in_avx, lut8!(lut_ptr, p[m][r], shift), acc[m][r]);
                    }
                }
            }
            col_packed += 8;
            col += 32;
        }

        // Horizontal fold, scalar tail, scaled store — per row, in the
        // incumbent's exact order of operations.
        for m in 0..M {
            let mut fs = [0.0f32; 4];
            for r in 0..4 {
                let mut t = [0.0f32; 8];
                _mm256_storeu_ps(t.as_mut_ptr(), acc[m][r]);
                fs[r] = t.iter().sum::<f32>();
            }
            let mut cp = col_packed;
            let mut c = col;
            while cp < packed_dim_in {
                for r in 0..4 {
                    let b = *w_ptr[m][r].add(cp) as usize;
                    fs[r] += input[c] * UNPACK_LUT[b * 4]
                        + input[c + 1] * UNPACK_LUT[b * 4 + 1]
                        + input[c + 2] * UNPACK_LUT[b * 4 + 2]
                        + input[c + 3] * UNPACK_LUT[b * 4 + 3];
                }
                cp += 1;
                c += 4;
            }
            for r in 0..4 {
                outs[m][row + r] = fs[r] * scales[m];
            }
        }
    }
}

/// Fused dual matvec: two weight matrices, one shared input, two outputs.
///
/// The fused prefix covers `4 * min(dim_out_a/4, dim_out_b/4)` rows of each
/// matrix; remaining rows (unequal dims, <4-row tails) delegate to the
/// incumbent `ternary_matvec_avx2` starting at the same 4-aligned offset, so
/// block partitioning — and therefore the bit pattern — matches a standalone
/// sequential call exactly.
///
/// # Safety
/// Same contract as `ternary_matvec_avx2`, applied to both matrices: AVX2+FMA
/// supported and OS-enabled; `weights_a`/`weights_b` hold
/// `dim_out_{a,b} * ceil(dim_in/4)` packed bytes; `input.len() >= dim_in`;
/// `out_a.len() >= dim_out_a`; `out_b.len() >= dim_out_b`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors two full incumbent signatures; a params struct would obscure the A/B"
)]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn ternary_matvec_fused2_avx2(
    out_a: &mut [f32],
    out_b: &mut [f32],
    input: &[f32],
    weights_a: &[u8],
    weights_b: &[u8],
    dim_out_a: usize,
    dim_out_b: usize,
    dim_in: usize,
    scale_a: f32,
    scale_b: f32,
) {
    let packed_dim_in = dim_in / 4;
    let blocks = (dim_out_a / 4).min(dim_out_b / 4);

    ternary_matvec_fusedn_avx2::<2>(
        &mut [&mut *out_a, &mut *out_b],
        input,
        [weights_a, weights_b],
        [scale_a, scale_b],
        blocks,
        dim_in,
    );

    let done = blocks * 4;
    if done < dim_out_a {
        ternary_matvec_avx2(
            &mut out_a[done..],
            input,
            &weights_a[done * packed_dim_in..],
            dim_out_a - done,
            dim_in,
            scale_a,
        );
    }
    if done < dim_out_b {
        ternary_matvec_avx2(
            &mut out_b[done..],
            input,
            &weights_b[done * packed_dim_in..],
            dim_out_b - done,
            dim_in,
            scale_b,
        );
    }
}

/// Fused tri matvec: three weight matrices (Q/K/V), one shared input.
///
/// Same delegation rule as `ternary_matvec_fused2_avx2`. Structural honesty
/// for GQA: with BitNet-2B shapes (Q 2560, K/V 640 rows) the fused prefix
/// covers only the first 640 rows of Q; Q's remaining 1920 rows run on the
/// incumbent kernel unfused.
///
/// # Safety
/// Same contract as `ternary_matvec_avx2`, applied to all three matrices.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors three full incumbent signatures; a params struct would obscure the A/B"
)]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn ternary_matvec_fused3_avx2(
    out_a: &mut [f32],
    out_b: &mut [f32],
    out_c: &mut [f32],
    input: &[f32],
    weights_a: &[u8],
    weights_b: &[u8],
    weights_c: &[u8],
    dim_out_a: usize,
    dim_out_b: usize,
    dim_out_c: usize,
    dim_in: usize,
    scale_a: f32,
    scale_b: f32,
    scale_c: f32,
) {
    let packed_dim_in = dim_in / 4;
    let blocks = (dim_out_a / 4).min(dim_out_b / 4).min(dim_out_c / 4);

    ternary_matvec_fusedn_avx2::<3>(
        &mut [&mut *out_a, &mut *out_b, &mut *out_c],
        input,
        [weights_a, weights_b, weights_c],
        [scale_a, scale_b, scale_c],
        blocks,
        dim_in,
    );

    let done = blocks * 4;
    if done < dim_out_a {
        ternary_matvec_avx2(
            &mut out_a[done..],
            input,
            &weights_a[done * packed_dim_in..],
            dim_out_a - done,
            dim_in,
            scale_a,
        );
    }
    if done < dim_out_b {
        ternary_matvec_avx2(
            &mut out_b[done..],
            input,
            &weights_b[done * packed_dim_in..],
            dim_out_b - done,
            dim_in,
            scale_b,
        );
    }
    if done < dim_out_c {
        ternary_matvec_avx2(
            &mut out_c[done..],
            input,
            &weights_c[done * packed_dim_in..],
            dim_out_c - done,
            dim_in,
            scale_c,
        );
    }
}

/// Safe dispatch for the fused dual matvec. Serial only — the row-parallel
/// pool split stays a wiring decision deferred until the kernel earns it on
/// quiet hardware. No-ops on undersized buffers, mirroring `ternary_matvec`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors two full incumbent signatures; a params struct would obscure the A/B"
)]
pub fn ternary_matvec_fused2(
    out_a: &mut [f32],
    out_b: &mut [f32],
    input: &[f32],
    weights_a: &[u8],
    weights_b: &[u8],
    dim_out_a: usize,
    dim_out_b: usize,
    dim_in: usize,
    scale_a: f32,
    scale_b: f32,
) {
    if out_a.len() < dim_out_a
        || out_b.len() < dim_out_b
        || input.len() < dim_in
        || weights_a.len() < (dim_out_a * dim_in) / 4
        || weights_b.len() < (dim_out_b * dim_in) / 4
    {
        return;
    }
    if simd_on() {
        // SAFETY: simd_on() gates on runtime AVX2+FMA detection (probe_simd,
        // honoring the force-scalar race toggle); buffer bounds checked above.
        unsafe {
            ternary_matvec_fused2_avx2(
                out_a, out_b, input, weights_a, weights_b, dim_out_a, dim_out_b, dim_in, scale_a,
                scale_b,
            );
        }
    } else {
        // Scalar path has no shared-load fusion to test; sequential calls ARE
        // the definition of correct here.
        ternary_matvec_scalar(out_a, input, weights_a, dim_out_a, dim_in, scale_a);
        ternary_matvec_scalar(out_b, input, weights_b, dim_out_b, dim_in, scale_b);
    }
}

/// Safe dispatch for the fused tri matvec. Serial only; see
/// `ternary_matvec_fused2`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors three full incumbent signatures; a params struct would obscure the A/B"
)]
pub fn ternary_matvec_fused3(
    out_a: &mut [f32],
    out_b: &mut [f32],
    out_c: &mut [f32],
    input: &[f32],
    weights_a: &[u8],
    weights_b: &[u8],
    weights_c: &[u8],
    dim_out_a: usize,
    dim_out_b: usize,
    dim_out_c: usize,
    dim_in: usize,
    scale_a: f32,
    scale_b: f32,
    scale_c: f32,
) {
    if out_a.len() < dim_out_a
        || out_b.len() < dim_out_b
        || out_c.len() < dim_out_c
        || input.len() < dim_in
        || weights_a.len() < (dim_out_a * dim_in) / 4
        || weights_b.len() < (dim_out_b * dim_in) / 4
        || weights_c.len() < (dim_out_c * dim_in) / 4
    {
        return;
    }
    if simd_on() {
        // SAFETY: simd_on() gates on runtime AVX2+FMA detection (probe_simd,
        // honoring the force-scalar race toggle); buffer bounds checked above.
        unsafe {
            ternary_matvec_fused3_avx2(
                out_a, out_b, out_c, input, weights_a, weights_b, weights_c, dim_out_a, dim_out_b,
                dim_out_c, dim_in, scale_a, scale_b, scale_c,
            );
        }
    } else {
        ternary_matvec_scalar(out_a, input, weights_a, dim_out_a, dim_in, scale_a);
        ternary_matvec_scalar(out_b, input, weights_b, dim_out_b, dim_in, scale_b);
        ternary_matvec_scalar(out_c, input, weights_c, dim_out_c, dim_in, scale_c);
    }
}

pub fn f32_from_bytes(b0: u8, b1: u8, b2: u8, b3: u8) -> f32 {
    f32::from_le_bytes([b0, b1, b2, b3])
}

/// # Safety
/// Caller must guarantee AVX2+FMA are supported and OS-enabled; raw pointers
/// must be valid for `vocab_size` rows of `emb_dim` BF16 elements
/// (embeddings) and `emb_dim`/`vocab_size` f32 elements (input/output).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn f32_dot_avx2(
    output: *mut f32,
    input: *const f32,
    embeddings: *const u8,
    vocab_size: usize,
    emb_dim: usize,
) -> u32 {
    let mut row = 0;
    let mut max_val = -f32::INFINITY;
    let mut max_idx = 0;
    while row + 4 <= vocab_size {
        let mut sum0 = core::arch::x86_64::_mm256_setzero_ps();
        let mut sum1 = core::arch::x86_64::_mm256_setzero_ps();
        let mut sum2 = core::arch::x86_64::_mm256_setzero_ps();
        let mut sum3 = core::arch::x86_64::_mm256_setzero_ps();

        let start0 = row * emb_dim * 2;
        let start1 = (row + 1) * emb_dim * 2;
        let start2 = (row + 2) * emb_dim * 2;
        let start3 = (row + 3) * emb_dim * 2;

        let mut col = 0;

        while col + 8 <= emb_dim {
            let in_vec = core::arch::x86_64::_mm256_loadu_ps(input.add(col));

            // Row 0
            let bf16_0 = core::arch::x86_64::_mm_loadu_si128(
                embeddings.add(start0 + col * 2) as *const core::arch::x86_64::__m128i
            );
            let emb_vec0 =
                core::arch::x86_64::_mm256_castsi256_ps(core::arch::x86_64::_mm256_slli_epi32(
                    core::arch::x86_64::_mm256_cvtepu16_epi32(bf16_0),
                    16,
                ));
            sum0 = core::arch::x86_64::_mm256_fmadd_ps(in_vec, emb_vec0, sum0);

            // Row 1
            let bf16_1 = core::arch::x86_64::_mm_loadu_si128(
                embeddings.add(start1 + col * 2) as *const core::arch::x86_64::__m128i
            );
            let emb_vec1 =
                core::arch::x86_64::_mm256_castsi256_ps(core::arch::x86_64::_mm256_slli_epi32(
                    core::arch::x86_64::_mm256_cvtepu16_epi32(bf16_1),
                    16,
                ));
            sum1 = core::arch::x86_64::_mm256_fmadd_ps(in_vec, emb_vec1, sum1);

            // Row 2
            let bf16_2 = core::arch::x86_64::_mm_loadu_si128(
                embeddings.add(start2 + col * 2) as *const core::arch::x86_64::__m128i
            );
            let emb_vec2 =
                core::arch::x86_64::_mm256_castsi256_ps(core::arch::x86_64::_mm256_slli_epi32(
                    core::arch::x86_64::_mm256_cvtepu16_epi32(bf16_2),
                    16,
                ));
            sum2 = core::arch::x86_64::_mm256_fmadd_ps(in_vec, emb_vec2, sum2);

            // Row 3
            let bf16_3 = core::arch::x86_64::_mm_loadu_si128(
                embeddings.add(start3 + col * 2) as *const core::arch::x86_64::__m128i
            );
            let emb_vec3 =
                core::arch::x86_64::_mm256_castsi256_ps(core::arch::x86_64::_mm256_slli_epi32(
                    core::arch::x86_64::_mm256_cvtepu16_epi32(bf16_3),
                    16,
                ));
            sum3 = core::arch::x86_64::_mm256_fmadd_ps(in_vec, emb_vec3, sum3);

            col += 8;
        }

        let mut s0 = [0.0f32; 8];
        let mut s1 = [0.0f32; 8];
        let mut s2 = [0.0f32; 8];
        let mut s3 = [0.0f32; 8];
        core::arch::x86_64::_mm256_storeu_ps(s0.as_mut_ptr(), sum0);
        core::arch::x86_64::_mm256_storeu_ps(s1.as_mut_ptr(), sum1);
        core::arch::x86_64::_mm256_storeu_ps(s2.as_mut_ptr(), sum2);
        core::arch::x86_64::_mm256_storeu_ps(s3.as_mut_ptr(), sum3);

        let mut total0 = s0.iter().sum::<f32>();
        let mut total1 = s1.iter().sum::<f32>();
        let mut total2 = s2.iter().sum::<f32>();
        let mut total3 = s3.iter().sum::<f32>();

        while col < emb_dim {
            let f0 = f32::from_le_bytes([
                0,
                0,
                *embeddings.add(start0 + col * 2),
                *embeddings.add(start0 + col * 2 + 1),
            ]);
            let f1 = f32::from_le_bytes([
                0,
                0,
                *embeddings.add(start1 + col * 2),
                *embeddings.add(start1 + col * 2 + 1),
            ]);
            let f2 = f32::from_le_bytes([
                0,
                0,
                *embeddings.add(start2 + col * 2),
                *embeddings.add(start2 + col * 2 + 1),
            ]);
            let f3 = f32::from_le_bytes([
                0,
                0,
                *embeddings.add(start3 + col * 2),
                *embeddings.add(start3 + col * 2 + 1),
            ]);

            total0 += *input.add(col) * f0;
            total1 += *input.add(col) * f1;
            total2 += *input.add(col) * f2;
            total3 += *input.add(col) * f3;
            col += 1;
        }

        output.add(row).write(total0);
        output.add(row + 1).write(total1);
        output.add(row + 2).write(total2);
        output.add(row + 3).write(total3);

        if total0 > max_val {
            max_val = total0;
            max_idx = row;
        }
        if total1 > max_val {
            max_val = total1;
            max_idx = row + 1;
        }
        if total2 > max_val {
            max_val = total2;
            max_idx = row + 2;
        }
        if total3 > max_val {
            max_val = total3;
            max_idx = row + 3;
        }

        row += 4;
    }

    // Remaining rows
    while row < vocab_size {
        let mut sum0 = core::arch::x86_64::_mm256_setzero_ps();
        let start0 = row * emb_dim * 2;
        let mut col = 0;

        while col + 8 <= emb_dim {
            let in_vec = core::arch::x86_64::_mm256_loadu_ps(input.add(col));
            let bf16_0 = core::arch::x86_64::_mm_loadu_si128(
                embeddings.add(start0 + col * 2) as *const core::arch::x86_64::__m128i
            );
            let emb_vec0 =
                core::arch::x86_64::_mm256_castsi256_ps(core::arch::x86_64::_mm256_slli_epi32(
                    core::arch::x86_64::_mm256_cvtepu16_epi32(bf16_0),
                    16,
                ));
            sum0 = core::arch::x86_64::_mm256_fmadd_ps(in_vec, emb_vec0, sum0);
            col += 8;
        }

        let mut s0 = [0.0f32; 8];
        core::arch::x86_64::_mm256_storeu_ps(s0.as_mut_ptr(), sum0);
        let mut total0 = s0.iter().sum::<f32>();

        while col < emb_dim {
            let f0 = f32::from_le_bytes([
                0,
                0,
                *embeddings.add(start0 + col * 2),
                *embeddings.add(start0 + col * 2 + 1),
            ]);
            total0 += *input.add(col) * f0;
            col += 1;
        }

        output.add(row).write(total0);
        if total0 > max_val {
            max_val = total0;
            max_idx = row;
        }
        row += 1;
    }
    max_idx as u32
}

/// # Safety
/// Caller must guarantee AVX2+FMA are supported and OS-enabled and that both
/// slices hold at least 128 elements (fixed head_dim).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vector_dot_128(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = core::arch::x86_64::_mm256_setzero_ps();
    let mut sum1 = core::arch::x86_64::_mm256_setzero_ps();
    let mut sum2 = core::arch::x86_64::_mm256_setzero_ps();
    let mut sum3 = core::arch::x86_64::_mm256_setzero_ps();

    let mut col = 0;
    while col < 128 {
        let va0 = core::arch::x86_64::_mm256_loadu_ps(a.as_ptr().add(col));
        let vb0 = core::arch::x86_64::_mm256_loadu_ps(b.as_ptr().add(col));
        sum0 = core::arch::x86_64::_mm256_fmadd_ps(va0, vb0, sum0);

        let va1 = core::arch::x86_64::_mm256_loadu_ps(a.as_ptr().add(col + 8));
        let vb1 = core::arch::x86_64::_mm256_loadu_ps(b.as_ptr().add(col + 8));
        sum1 = core::arch::x86_64::_mm256_fmadd_ps(va1, vb1, sum1);

        let va2 = core::arch::x86_64::_mm256_loadu_ps(a.as_ptr().add(col + 16));
        let vb2 = core::arch::x86_64::_mm256_loadu_ps(b.as_ptr().add(col + 16));
        sum2 = core::arch::x86_64::_mm256_fmadd_ps(va2, vb2, sum2);

        let va3 = core::arch::x86_64::_mm256_loadu_ps(a.as_ptr().add(col + 24));
        let vb3 = core::arch::x86_64::_mm256_loadu_ps(b.as_ptr().add(col + 24));
        sum3 = core::arch::x86_64::_mm256_fmadd_ps(va3, vb3, sum3);

        col += 32;
    }

    let mut s0 = [0.0f32; 8];
    let mut s1 = [0.0f32; 8];
    let mut s2 = [0.0f32; 8];
    let mut s3 = [0.0f32; 8];
    core::arch::x86_64::_mm256_storeu_ps(s0.as_mut_ptr(), sum0);
    core::arch::x86_64::_mm256_storeu_ps(s1.as_mut_ptr(), sum1);
    core::arch::x86_64::_mm256_storeu_ps(s2.as_mut_ptr(), sum2);
    core::arch::x86_64::_mm256_storeu_ps(s3.as_mut_ptr(), sum3);

    s0.iter().sum::<f32>()
        + s1.iter().sum::<f32>()
        + s2.iter().sum::<f32>()
        + s3.iter().sum::<f32>()
}

/// # Safety
/// Caller must guarantee AVX2+FMA are supported and OS-enabled and that both
/// slices hold at least 128 elements (fixed head_dim).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vector_madd_128(out: &mut [f32], w: f32, v: &[f32]) {
    let w_vec = core::arch::x86_64::_mm256_set1_ps(w);
    let mut col = 0;
    while col < 128 {
        let out_vec0 = core::arch::x86_64::_mm256_loadu_ps(out.as_ptr().add(col));
        let v_vec0 = core::arch::x86_64::_mm256_loadu_ps(v.as_ptr().add(col));
        let res0 = core::arch::x86_64::_mm256_fmadd_ps(w_vec, v_vec0, out_vec0);
        core::arch::x86_64::_mm256_storeu_ps(out.as_mut_ptr().add(col), res0);

        let out_vec1 = core::arch::x86_64::_mm256_loadu_ps(out.as_ptr().add(col + 8));
        let v_vec1 = core::arch::x86_64::_mm256_loadu_ps(v.as_ptr().add(col + 8));
        let res1 = core::arch::x86_64::_mm256_fmadd_ps(w_vec, v_vec1, out_vec1);
        core::arch::x86_64::_mm256_storeu_ps(out.as_mut_ptr().add(col + 8), res1);

        let out_vec2 = core::arch::x86_64::_mm256_loadu_ps(out.as_ptr().add(col + 16));
        let v_vec2 = core::arch::x86_64::_mm256_loadu_ps(v.as_ptr().add(col + 16));
        let res2 = core::arch::x86_64::_mm256_fmadd_ps(w_vec, v_vec2, out_vec2);
        core::arch::x86_64::_mm256_storeu_ps(out.as_mut_ptr().add(col + 16), res2);

        let out_vec3 = core::arch::x86_64::_mm256_loadu_ps(out.as_ptr().add(col + 24));
        let v_vec3 = core::arch::x86_64::_mm256_loadu_ps(v.as_ptr().add(col + 24));
        let res3 = core::arch::x86_64::_mm256_fmadd_ps(w_vec, v_vec3, out_vec3);
        core::arch::x86_64::_mm256_storeu_ps(out.as_mut_ptr().add(col + 24), res3);

        col += 32;
    }
}

fn f32_dot_argmax_serial(
    output: &mut [f32],
    input: &[f32],
    embeddings: &[u8],
    vocab_size: usize,
    emb_dim: usize,
) -> u32 {
    if simd_on() {
        unsafe {
            f32_dot_avx2(
                output.as_mut_ptr(),
                input.as_ptr(),
                embeddings.as_ptr(),
                vocab_size,
                emb_dim,
            )
        }
    } else {
        f32_dot_scalar(output, input, embeddings, vocab_size, emb_dim)
    }
}

/// Fused LM head + argmax over the tied BF16 embedding table.
///
/// Row-parallel like the ternary matvec, but the argmax is a reduction: each
/// worker returns the best (logit, index) in its own row range, and we combine
/// them preferring the LOWEST index on ties, which is exactly what a single
/// sequential scan would have chosen.
pub fn f32_dot_argmax(
    output: &mut [f32],
    input: &[f32],
    embeddings: &[u8],
    vocab_size: usize,
    emb_dim: usize,
) -> u32 {
    // Embeddings are BF16: 2 bytes per element (matches the *2 strides in f32_dot_avx2)
    if output.len() < vocab_size || embeddings.len() < vocab_size * emb_dim * 2 {
        return 0;
    }

    #[cfg(feature = "parallel")]
    {
        let pool = crate::pool::global();
        let n = pool.workers();
        if n > 1 && vocab_size * emb_dim >= PARALLEL_MIN_MACS {
            // One slot per worker; no locking, combined after the barrier.
            let mut best: alloc::vec::Vec<(u32, f32)> = alloc::vec![(0u32, -f32::INFINITY); n];
            let best_base = PairPtr(best.as_mut_ptr());
            let out_base = OutPtr(output.as_mut_ptr());

            pool.broadcast(|id, n| {
                let (row0, rows) = row_split(vocab_size, id, n);
                if rows == 0 {
                    return;
                }
                // SAFETY: disjoint row bands; slot `id` is written by worker `id` only.
                let chunk =
                    unsafe { core::slice::from_raw_parts_mut(out_base.get().add(row0), rows) };
                let emb = &embeddings[row0 * emb_dim * 2..(row0 + rows) * emb_dim * 2];
                let local = f32_dot_argmax_serial(chunk, input, emb, rows, emb_dim);
                unsafe { *best_base.get().add(id) = (row0 as u32 + local, chunk[local as usize]) };
            });

            // Combine preferring the lowest index on ties: workers are visited in
            // ascending row order and we take a strict improvement only, which is
            // exactly what one sequential scan would have selected.
            let mut best_idx = 0u32;
            let mut best_val = -f32::INFINITY;
            for &(idx, val) in best.iter() {
                if val > best_val {
                    best_val = val;
                    best_idx = idx;
                }
            }
            return best_idx;
        }
    }

    f32_dot_argmax_serial(output, input, embeddings, vocab_size, emb_dim)
}

pub fn silu(x: &mut [f32]) {
    for val in x.iter_mut() {
        let v = *val;
        *val = v / (1.0 + libm::expf(-v));
    }
}

/// Batch tile width for the prefill GEMM: how many tokens share one pass over
/// the weights. Each tile needs TILE ymm accumulators plus 2 scratch registers,
/// and x86-64 has 16 — 8 keeps us clear of spills while cutting weight traffic
/// (and weight-unpack work) by 8x.
const GEMM_TILE: usize = 8;

/// Core of the batched GEMM: one output row, `TILE` tokens, weights unpacked
/// once and reused across the whole tile.
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)] // GEMM kernel: strides/tiles are the interface
unsafe fn ternary_gemm_row_avx2<const TILE: usize>(
    out: *mut f32,
    out_stride: usize,
    row: usize,
    input: &[f32],
    in_stride: usize,
    w_row: *const u8,
    packed_dim_in: usize,
    lut_ptr: *const f32,
    scale: f32,
) {
    use core::arch::x86_64::*;
    let mut acc = [_mm256_setzero_ps(); TILE];

    let mut col_packed = 0;
    let mut col = 0;
    while col_packed + 8 <= packed_dim_in {
        let p = core::ptr::read_unaligned(w_row.add(col_packed) as *const u64);
        for i in 0..4 {
            let shift = i * 16;
            // Unpack 8 ternary weights ONCE, then apply to every token in the tile.
            let b0 = ((p >> shift) & 0xFF) as usize;
            let b1 = ((p >> (shift + 8)) & 0xFF) as usize;
            let w_avx = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(lut_ptr.add(b0 * 4))),
                _mm_loadu_ps(lut_ptr.add(b1 * 4)),
                1,
            );
            let c = col + i * 8;
            for t in 0..TILE {
                let in_avx = _mm256_loadu_ps(input.as_ptr().add(t * in_stride + c));
                acc[t] = _mm256_fmadd_ps(in_avx, w_avx, acc[t]);
            }
        }
        col_packed += 8;
        col += 32;
    }

    let mut sums = [0.0f32; TILE];
    for t in 0..TILE {
        let mut s = [0.0f32; 8];
        _mm256_storeu_ps(s.as_mut_ptr(), acc[t]);
        sums[t] = s.iter().sum::<f32>();
    }

    // Tail: dim_in not a multiple of 32 (never hit by BitNet dims, kept correct).
    while col_packed < packed_dim_in {
        let b = *w_row.add(col_packed) as usize;
        let l = b * 4;
        for t in 0..TILE {
            let base = t * in_stride + col;
            sums[t] += *input.as_ptr().add(base) * *lut_ptr.add(l)
                + *input.as_ptr().add(base + 1) * *lut_ptr.add(l + 1)
                + *input.as_ptr().add(base + 2) * *lut_ptr.add(l + 2)
                + *input.as_ptr().add(base + 3) * *lut_ptr.add(l + 3);
        }
        col_packed += 1;
        col += 4;
    }

    for t in 0..TILE {
        *out.add(t * out_stride + row) = sums[t] * scale;
    }
}

/// Batched GEMM restricted to output rows `[row_start, row_start + row_count)`.
/// `out_base` points at row 0 of token 0; rows are addressed absolutely, so
/// disjoint row bands can be filled concurrently by different workers.
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)] // GEMM kernel: strides/tiles are the interface
unsafe fn ternary_matmul_avx2_rows(
    out_base: *mut f32,
    input: &[f32],
    weights_packed: &[u8],
    batch: usize,
    dim_out: usize,
    dim_in: usize,
    scale: f32,
    row_start: usize,
    row_count: usize,
) {
    let lut_ptr = UNPACK_LUT.as_ptr();
    let packed_dim_in = dim_in / 4;

    let mut b0 = 0;
    while b0 < batch {
        let tile = core::cmp::min(GEMM_TILE, batch - b0);
        let out_tile = out_base.add(b0 * dim_out);
        let in_tile = &input[b0 * dim_in..];

        for row in row_start..row_start + row_count {
            let w_row = weights_packed.as_ptr().add(row * packed_dim_in);
            // Monomorphized per tile width so the inner `for t` fully unrolls.
            macro_rules! dispatch {
                ($($n:literal),*) => {
                    match tile {
                        $($n => ternary_gemm_row_avx2::<$n>(out_tile, dim_out, row, in_tile, dim_in, w_row, packed_dim_in, lut_ptr, scale),)*
                        _ => ternary_gemm_row_avx2::<1>(out_tile, dim_out, row, in_tile, dim_in, w_row, packed_dim_in, lut_ptr, scale),
                    }
                };
            }
            dispatch!(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        }
        b0 += tile;
    }
}

/// Prefill matmul: `batch` tokens through the same weight matrix.
///
/// Weights are streamed once per tile of GEMM_TILE tokens rather than once per
/// token, so both the weight traffic and — more importantly on this hardware —
/// the per-weight LUT unpack work are amortized across the tile. Each tile's
/// activations (8 x 2560 x 4 B = 80 KB) stay resident in L2.
///
/// Measured ~1.8x over the per-token path. Not the 8x the traffic reduction
/// suggests. The previous explanation here — "~2 GB/s against a 17.3 GB/s
/// ceiling" — is RETRACTED/superseded (A13.bw: that ceiling had no source
/// anywhere in this repo).
///
/// Measured on this host instead (membw_2026-08-25_162849Z.log):
///   A13.bw.seq1t   10.57 GB/s  peak sequential read, 1 thread
///   A13.bw.seq8t   25.28 GB/s  peak sequential read, 8 threads
///   A13.bw.tern1t   0.80 GB/s  ternary weight-stream, scalar LUT walk
///
/// The weight-streaming roofline divides by the TERNARY figure, not a
/// sequential peak. The compute-bound-vs-bandwidth-bound conclusion therefore
/// needs re-deriving against A13.bw.tern1t (itself a scalar LOWER BOUND, not
/// the AVX2 kernel's rate) and is deliberately NOT restated here.
/// The win here is the amortized unpack.
pub fn ternary_matmul(
    output: &mut [f32],
    input: &[f32],
    weights_packed: &[u8],
    batch: usize,
    dim_out: usize,
    dim_in: usize,
    scale: f32,
) {
    if output.len() < batch * dim_out
        || input.len() < batch * dim_in
        || weights_packed.len() < (dim_out * dim_in) / 4
    {
        return;
    }
    if cfg!(feature = "legacy_matmul") || FORCE_LEGACY_PREFILL.load(Ordering::Relaxed) {
        // Pre-2026-07-09 path, kept for same-session A/B benchmarking.
        for b in 0..batch {
            let out_slice = &mut output[b * dim_out..(b + 1) * dim_out];
            let in_slice = &input[b * dim_in..(b + 1) * dim_in];
            ternary_matvec(out_slice, in_slice, weights_packed, dim_out, dim_in, scale);
        }
        return;
    }

    // Prefill is row-parallel exactly as the matvec is: each worker owns a
    // disjoint band of output rows, for every token in the batch.
    #[cfg(feature = "parallel")]
    {
        let pool = crate::pool::global();
        if pool.workers() > 1 && simd_on() && dim_out * dim_in >= PARALLEL_MIN_MACS {
            let out_base = OutPtr(output.as_mut_ptr());
            pool.broadcast(|id, n| {
                let (row0, rows) = row_split(dim_out, id, n);
                if rows == 0 {
                    return;
                }
                // Each worker runs the BATCHED kernel over its own row band, so
                // the tile-level weight reuse is preserved on top of the split.
                // SAFETY: row bands are disjoint across workers.
                unsafe {
                    ternary_matmul_avx2_rows(
                        out_base.get(),
                        input,
                        weights_packed,
                        batch,
                        dim_out,
                        dim_in,
                        scale,
                        row0,
                        rows,
                    )
                };
            });
            return;
        }
    }

    if simd_on() {
        unsafe {
            ternary_matmul_avx2_rows(
                output.as_mut_ptr(),
                input,
                weights_packed,
                batch,
                dim_out,
                dim_in,
                scale,
                0,
                dim_out,
            )
        };
    } else {
        for b in 0..batch {
            let out_slice = &mut output[b * dim_out..(b + 1) * dim_out];
            let in_slice = &input[b * dim_in..(b + 1) * dim_in];
            ternary_matvec_scalar(out_slice, in_slice, weights_packed, dim_out, dim_in, scale);
        }
    }
}
