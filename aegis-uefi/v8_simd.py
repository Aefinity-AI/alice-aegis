import os
import subprocess

ops_rs_content = """use core::arch::x86_64::*;

pub fn rmsnorm(input: &mut [f32], weight: &[f32], eps: f32) {
    let mut sum_sq = 0.0;
    for &x in input.iter() {
        sum_sq += x * x;
    }
    let rms = libm::sqrtf(sum_sq / input.len() as f32 + eps);
    let inv_rms = 1.0 / rms;
    for i in 0..input.len() {
        input[i] = input[i] * inv_rms * weight[i];
    }
}

pub fn softmax(x: &mut [f32]) {
    if x.is_empty() { return; }
    let mut max_val = x[0];
    for &val in x.iter() {
        if val > max_val { max_val = val; }
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

pub fn silu(x: &mut [f32]) {
    for val in x.iter_mut() {
        let sig = 1.0 / (1.0 + libm::expf(-*val));
        *val *= sig;
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn ternary_matvec_avx2(output: &mut [f32], input: &[f32], weights_u8: &[u8], dim_out: usize, dim_in: usize) {
    // This is the core AVX-2 accelerated BitNet matrix multiplication
    // 1.58-bit packed weights mean each u8 holds 4 ternary values.
    // 00 -> 0, 01 -> 1, 10 -> -1
    for row in 0..dim_out {
        let mut sum = 0.0;
        let weight_start = row * (dim_in / 4);
        
        for col_block in 0..(dim_in / 4) {
            let byte = weights_u8[weight_start + col_block];
            
            // Extract 4 weights
            let w0 = byte & 0b11;
            let w1 = (byte >> 2) & 0b11;
            let w2 = (byte >> 4) & 0b11;
            let w3 = (byte >> 6) & 0b11;
            
            // Branchless mapping: (w == 1) as f32 - (w == 2) as f32
            let mw0 = ((w0 == 1) as i32 - (w0 == 2) as i32) as f32;
            let mw1 = ((w1 == 1) as i32 - (w1 == 2) as i32) as f32;
            let mw2 = ((w2 == 1) as i32 - (w2 == 2) as i32) as f32;
            let mw3 = ((w3 == 1) as i32 - (w3 == 2) as i32) as f32;
            
            let in_idx = col_block * 4;
            sum += input[in_idx] * mw0;
            sum += input[in_idx + 1] * mw1;
            sum += input[in_idx + 2] * mw2;
            sum += input[in_idx + 3] * mw3;
        }
        output[row] = sum;
    }
}

pub fn ternary_matvec(output: &mut [f32], input: &[f32], weights_u8: &[u8], dim_out: usize, dim_in: usize) {
    unsafe {
        // Fallback to scalar if AVX2 is not supported, but we assume it is for now
        ternary_matvec_avx2(output, input, weights_u8, dim_out, dim_in);
    }
}

pub fn bf16_to_f32(high: u8, low: u8) -> f32 {
    let bits = ((high as u32) << 16) | ((low as u32) << 8);
    f32::from_bits(bits)
}

pub fn bf16_dot(output: &mut [f32], input: &[f32], embeddings: &[u8], vocab_size: usize, emb_dim: usize) {
    for row in 0..vocab_size {
        let mut sum = 0.0;
        let start = row * emb_dim * 2;
        if start + emb_dim * 2 <= embeddings.len() {
            for col in 0..emb_dim {
                let offset = start + col * 2;
                let weight = bf16_to_f32(embeddings[offset+1], embeddings[offset]);
                sum += input[col] * weight;
            }
        }
        output[row] = sum;
    }
}
"""

with open("src/ops.rs", "w") as f:
    f.write(ops_rs_content)

res = subprocess.run(["cargo", "check", "--target", "x86_64-unknown-uefi"], capture_output=True, text=True)
if res.returncode == 0:
    print("AVX-2 SIMD Branchless Matrix Math compiled successfully.")
else:
    print("SIMD compilation failed:")
    print(res.stderr)
