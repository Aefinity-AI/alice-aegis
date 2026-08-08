import re

with open('/home/killboxincorporated/aegis-core/src/ops.rs', 'r') as f:
    content = f.read()

new_code = """
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn ternary_matmul_avx2(output: &mut [f32], input: &[f32], weights_u8: &[u8], batch_size: usize, dim_out: usize, dim_in: usize, scale: f32) {
    // A simplified but effective batched matmul where we outer-loop over rows, 
    // and inner-loop over the batch, maximizing weight reuse in the L1 cache.
    for row in 0..dim_out {
        let weight_start = row * (dim_in / 4);
        
        for b in 0..batch_size {
            let mut sum_vec = _mm256_setzero_ps();
            let mut col = 0;
            let mut weight_idx = weight_start;
            let in_ptr = input.as_ptr().add(b * dim_in);
            
            while col + 8 <= dim_in {
                let byte0 = *weights_u8.as_ptr().add(weight_idx);
                let byte1 = *weights_u8.as_ptr().add(weight_idx + 1);
                
                let in_vec = _mm256_loadu_ps(in_ptr.add(col));
                
                let w0_3 = _mm_loadu_ps(TERNARY_LUT[byte0 as usize].as_ptr());
                let w4_7 = _mm_loadu_ps(TERNARY_LUT[byte1 as usize].as_ptr());
                let w_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(w0_3), w4_7, 1);
                
                sum_vec = _mm256_fmadd_ps(in_vec, w_vec, sum_vec);
                
                col += 8;
                weight_idx += 2;
            }
            
            let mut sum_array = [0.0f32; 8];
            _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
            let mut final_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + 
                                sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
            
            while col < dim_in {
                let byte = *weights_u8.as_ptr().add(weight_idx);
                let w0 = ((byte & 0b11) == 1) as i32 - ((byte & 0b11) == 2) as i32;
                let w1 = (((byte >> 2) & 0b11) == 1) as i32 - (((byte >> 2) & 0b11) == 2) as i32;
                let w2 = (((byte >> 4) & 0b11) == 1) as i32 - (((byte >> 4) & 0b11) == 2) as i32;
                let w3 = (((byte >> 6) & 0b11) == 1) as i32 - (((byte >> 6) & 0b11) == 2) as i32;
                
                if col < dim_in { final_sum += *in_ptr.add(col) * (w0 as f32); }
                if col + 1 < dim_in { final_sum += *in_ptr.add(col + 1) * (w1 as f32); }
                if col + 2 < dim_in { final_sum += *in_ptr.add(col + 2) * (w2 as f32); }
                if col + 3 < dim_in { final_sum += *in_ptr.add(col + 3) * (w3 as f32); }
                
                col += 4;
                weight_idx += 1;
            }
            
            output[b * dim_out + row] = final_sum * scale;
        }
    }
}

pub fn ternary_matmul(output: &mut [f32], input: &[f32], weights_u8: &[u8], batch_size: usize, dim_out: usize, dim_in: usize, scale: f32) {
    if output.len() < batch_size * dim_out || input.len() < batch_size * dim_in || weights_u8.len() < (dim_out * dim_in) / 4 {
        return; 
    }
    unsafe {
        ternary_matmul_avx2(output, input, weights_u8, batch_size, dim_out, dim_in, scale);
    }
}
"""

if "ternary_matmul_avx2" not in content:
    content += "\n" + new_code

with open('/home/killboxincorporated/aegis-core/src/ops.rs', 'w') as f:
    f.write(content)
