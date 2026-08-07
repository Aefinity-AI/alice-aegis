import re

with open('/home/killboxincorporated/aegis-core/src/ops.rs', 'r') as f:
    content = f.read()

new_ternary = """
static mut UNPACK_LUT: [f32; 1024] = [0.0; 1024];
static mut LUT_INIT: bool = false;

pub unsafe fn init_unpack_lut() {
    if LUT_INIT { return; }
    for b in 0..256 {
        let w0 = b & 3;
        let w1 = (b >> 2) & 3;
        let w2 = (b >> 4) & 3;
        let w3 = (b >> 6) & 3;
        UNPACK_LUT[b * 4 + 0] = if w0 == 2 { -1.0 } else { w0 as f32 };
        UNPACK_LUT[b * 4 + 1] = if w1 == 2 { -1.0 } else { w1 as f32 };
        UNPACK_LUT[b * 4 + 2] = if w2 == 2 { -1.0 } else { w2 as f32 };
        UNPACK_LUT[b * 4 + 3] = if w3 == 2 { -1.0 } else { w3 as f32 };
    }
    LUT_INIT = true;
}

pub fn ternary_matvec(output: &mut [f32], input: &[f32], weights_packed: &[u8], dim_out: usize, dim_in: usize, scale: f32) {
    unsafe {
        init_unpack_lut();
        let lut_ptr = UNPACK_LUT.as_ptr();
        
        let mut row = 0;
        let packed_dim_in = dim_in / 4;
        
        while row + 3 < dim_out {
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();
            
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
                    let in_avx = _mm256_loadu_ps(input.as_ptr().add(col + i * 8));
                    
                    let b0_0 = ((p0 >> shift) & 0xFF) as usize; let b0_1 = ((p0 >> (shift + 8)) & 0xFF) as usize;
                    let w_avx0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(lut_ptr.add(b0_0 * 4))), _mm_loadu_ps(lut_ptr.add(b0_1 * 4)), 1);
                    sum0 = _mm256_fmadd_ps(in_avx, w_avx0, sum0);
                    
                    let b1_0 = ((p1 >> shift) & 0xFF) as usize; let b1_1 = ((p1 >> (shift + 8)) & 0xFF) as usize;
                    let w_avx1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(lut_ptr.add(b1_0 * 4))), _mm_loadu_ps(lut_ptr.add(b1_1 * 4)), 1);
                    sum1 = _mm256_fmadd_ps(in_avx, w_avx1, sum1);
                    
                    let b2_0 = ((p2 >> shift) & 0xFF) as usize; let b2_1 = ((p2 >> (shift + 8)) & 0xFF) as usize;
                    let w_avx2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(lut_ptr.add(b2_0 * 4))), _mm_loadu_ps(lut_ptr.add(b2_1 * 4)), 1);
                    sum2 = _mm256_fmadd_ps(in_avx, w_avx2, sum2);
                    
                    let b3_0 = ((p3 >> shift) & 0xFF) as usize; let b3_1 = ((p3 >> (shift + 8)) & 0xFF) as usize;
                    let w_avx3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(lut_ptr.add(b3_0 * 4))), _mm_loadu_ps(lut_ptr.add(b3_1 * 4)), 1);
                    sum3 = _mm256_fmadd_ps(in_avx, w_avx3, sum3);
                }
                col_packed += 8; col += 32;
            }
            
            let mut s0 = [0.0f32; 8]; let mut s1 = [0.0f32; 8]; let mut s2 = [0.0f32; 8]; let mut s3 = [0.0f32; 8];
            _mm256_storeu_ps(s0.as_mut_ptr(), sum0); _mm256_storeu_ps(s1.as_mut_ptr(), sum1);
            _mm256_storeu_ps(s2.as_mut_ptr(), sum2); _mm256_storeu_ps(s3.as_mut_ptr(), sum3);
            let mut fs0 = s0.iter().sum::<f32>(); let mut fs1 = s1.iter().sum::<f32>();
            let mut fs2 = s2.iter().sum::<f32>(); let mut fs3 = s3.iter().sum::<f32>();
            
            while col_packed < packed_dim_in {
                let b0 = *w_ptr0.add(col_packed) as usize;
                let b1 = *w_ptr1.add(col_packed) as usize;
                let b2 = *w_ptr2.add(col_packed) as usize;
                let b3 = *w_ptr3.add(col_packed) as usize;
                
                fs0 += input[col] * UNPACK_LUT[b0 * 4] + input[col+1] * UNPACK_LUT[b0 * 4 + 1] + input[col+2] * UNPACK_LUT[b0 * 4 + 2] + input[col+3] * UNPACK_LUT[b0 * 4 + 3];
                fs1 += input[col] * UNPACK_LUT[b1 * 4] + input[col+1] * UNPACK_LUT[b1 * 4 + 1] + input[col+2] * UNPACK_LUT[b1 * 4 + 2] + input[col+3] * UNPACK_LUT[b1 * 4 + 3];
                fs2 += input[col] * UNPACK_LUT[b2 * 4] + input[col+1] * UNPACK_LUT[b2 * 4 + 1] + input[col+2] * UNPACK_LUT[b2 * 4 + 2] + input[col+3] * UNPACK_LUT[b2 * 4 + 3];
                fs3 += input[col] * UNPACK_LUT[b3 * 4] + input[col+1] * UNPACK_LUT[b3 * 4 + 1] + input[col+2] * UNPACK_LUT[b3 * 4 + 2] + input[col+3] * UNPACK_LUT[b3 * 4 + 3];
                
                col_packed += 1; col += 4;
            }
            output[row] = fs0 * scale; output[row+1] = fs1 * scale; output[row+2] = fs2 * scale; output[row+3] = fs3 * scale;
            row += 4;
        }
        
        while row < dim_out {
            let mut sum0 = _mm256_setzero_ps();
            let w_ptr0 = weights_packed.as_ptr().add(row * packed_dim_in);
            let mut col_packed = 0; let mut col = 0;
            
            while col_packed + 8 <= packed_dim_in {
                let p0 = core::ptr::read_unaligned(w_ptr0.add(col_packed) as *const u64);
                for i in 0..4 {
                    let shift = i * 16;
                    let in_avx = _mm256_loadu_ps(input.as_ptr().add(col + i * 8));
                    let b0_0 = ((p0 >> shift) & 0xFF) as usize; let b0_1 = ((p0 >> (shift + 8)) & 0xFF) as usize;
                    let w_avx0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(lut_ptr.add(b0_0 * 4))), _mm_loadu_ps(lut_ptr.add(b0_1 * 4)), 1);
                    sum0 = _mm256_fmadd_ps(in_avx, w_avx0, sum0);
                }
                col_packed += 8; col += 32;
            }
            
            let mut s0 = [0.0f32; 8]; _mm256_storeu_ps(s0.as_mut_ptr(), sum0);
            let mut fs0 = s0.iter().sum::<f32>();
            
            while col_packed < packed_dim_in {
                let b0 = *w_ptr0.add(col_packed) as usize;
                fs0 += input[col] * UNPACK_LUT[b0 * 4] + input[col+1] * UNPACK_LUT[b0 * 4 + 1] + input[col+2] * UNPACK_LUT[b0 * 4 + 2] + input[col+3] * UNPACK_LUT[b0 * 4 + 3];
                col_packed += 1; col += 4;
            }
            output[row] = fs0 * scale;
            row += 1;
        }
    }
}

pub fn ternary_matmul(output: &mut [f32], input: &[f32], weights_packed: &[u8], batch: usize, dim_out: usize, dim_in: usize, scale: f32) {
    // For batching, we simply wrap matvec over the batch elements!
    // Since we are optimizing for low latency decoding, matvec is sufficient.
    for b in 0..batch {
        let out_slice = &mut output[b * dim_out..(b + 1) * dim_out];
        let in_slice = &input[b * dim_in..(b + 1) * dim_in];
        ternary_matvec(out_slice, in_slice, weights_packed, dim_out, dim_in, scale);
    }
}
"""

start_idx = content.find("pub fn ternary_matvec")
end_idx = content.find("pub fn softmax", start_idx)

new_content = content[:start_idx] + new_ternary + content[end_idx:]

with open('/home/killboxincorporated/aegis-core/src/ops.rs', 'w') as f:
    f.write(new_content)
