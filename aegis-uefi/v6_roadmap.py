import os

# 1. ops.rs - Math and SIMD Iron
ops_rs = """use core::arch::x86_64::*;

pub fn rmsnorm(input: &mut [f32], weight: &[f32], eps: f32) {
    let mut sum_sq = 0.0;
    for &x in input.iter() {
        sum_sq += x * x;
    }
    let inv_rms = libm::sqrtf((sum_sq / input.len() as f32) + eps);
    let inv_rms = 1.0 / inv_rms;
    for i in 0..input.len() {
        input[i] = input[i] * inv_rms * weight[i];
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = libm::expf(*v - max_val);
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + libm::expf(-*v));
    }
}

pub fn ternary_matvec(output: &mut [f32], input: &[f32], weights_i8: &[i8]) {
    // Branchless formulation for Ternary weights (-1, 0, 1)
    let in_len = input.len();
    for (i, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        let w_row = &weights_i8[i * in_len..(i + 1) * in_len];
        for (j, &in_val) in input.iter().enumerate() {
            let w = w_row[j] as f32;
            sum += in_val * w;
        }
        *out = sum;
    }
}
"""
with open("/home/killboxincorporated/aegis-uefi/src/ops.rs", "w") as f:
    f.write(ops_rs)

# 2. kvcache.rs
kvcache_rs = """use alloc::vec::Vec;

pub struct KVCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let size = num_layers * max_seq_len * num_kv_heads * head_dim;
        Self {
            k_cache: alloc::vec![0.0; size],
            v_cache: alloc::vec![0.0; size],
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
        }
    }

    pub fn get_layer_mut(&mut self, layer_idx: usize, seq_pos: usize) -> (&mut [f32], &mut [f32]) {
        let layer_size = self.max_seq_len * self.num_kv_heads * self.head_dim;
        let layer_offset = layer_idx * layer_size;
        let seq_offset = seq_pos * self.num_kv_heads * self.head_dim;
        
        let start = layer_offset + seq_offset;
        let end = start + (self.num_kv_heads * self.head_dim);
        
        (&mut self.k_cache[start..end], &mut self.v_cache[start..end])
    }
}
"""
with open("/home/killboxincorporated/aegis-uefi/src/kvcache.rs", "w") as f:
    f.write(kvcache_rs)

# 3. attention.rs
attention_rs = """use alloc::vec::Vec;

pub struct AttentionLayer {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
}

impl AttentionLayer {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        Self {
            hidden_dim,
            num_heads,
            head_dim: hidden_dim / num_heads,
        }
    }

    pub fn apply_rope(&self, q: &mut [f32], k: &mut [f32], seq_pos: usize) {
        let dim = self.head_dim;
        for h in 0..self.num_heads {
            let q_head = &mut q[h * dim..(h + 1) * dim];
            for d in (0..dim).step_by(2) {
                let dim_ratio = (d as f32) / (dim as f32);
                let base = 10000.0_f32;
                let freq = 1.0 / libm::powf(base, dim_ratio);
                let theta = (seq_pos as f32) * freq;
                
                let cos_theta = libm::cosf(theta);
                let sin_theta = libm::sinf(theta);
                
                let q0 = q_head[d];
                let q1 = q_head[d + 1];
                q_head[d] = q0 * cos_theta - q1 * sin_theta;
                q_head[d + 1] = q0 * sin_theta + q1 * cos_theta;
            }
        }
        
        let num_kv_heads = self.num_heads; // Grouped query attention can override this
        for h in 0..num_kv_heads {
            let k_head = &mut k[h * dim..(h + 1) * dim];
            for d in (0..dim).step_by(2) {
                let dim_ratio = (d as f32) / (dim as f32);
                let base = 10000.0_f32;
                let freq = 1.0 / libm::powf(base, dim_ratio);
                let theta = (seq_pos as f32) * freq;
                
                let cos_theta = libm::cosf(theta);
                let sin_theta = libm::sinf(theta);
                
                let k0 = k_head[d];
                let k1 = k_head[d + 1];
                k_head[d] = k0 * cos_theta - k1 * sin_theta;
                k_head[d + 1] = k0 * sin_theta + k1 * cos_theta;
            }
        }
    }
}
"""
with open("/home/killboxincorporated/aegis-uefi/src/attention.rs", "w") as f:
    f.write(attention_rs)

print("ops, kvcache, attention updated for V6 roadmap.")
