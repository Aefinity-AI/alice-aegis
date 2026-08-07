pub struct RopeCache {
    pub cos: alloc::vec::Vec<f32>,
    pub sin: alloc::vec::Vec<f32>,
    pub max_seq_len: usize,
    pub half_dim: usize,
}

impl RopeCache {
    pub fn new(max_seq_len: usize, head_dim: usize, base: f32) -> Self {
        let half_dim = head_dim / 2;
        let size = max_seq_len * half_dim;
        let mut cos = alloc::vec::Vec::with_capacity(size);
        let mut sin = alloc::vec::Vec::with_capacity(size);
        cos.resize(size, 0.0);
        sin.resize(size, 0.0);

        for pos in 0..max_seq_len {
            for d in 0..half_dim {
                let dim_ratio = ((d * 2) as f32) / (head_dim as f32);
                let freq = 1.0 / libm::powf(base, dim_ratio);
                let theta = (pos as f32) * freq;

                let c = libm::cosf(theta);
                let s = libm::sinf(theta);

                let idx = pos * half_dim + d;
                cos[idx] = c;
                sin[idx] = s;
            }
        }

        Self {
            cos,
            sin,
            max_seq_len,
            half_dim,
        }
    }
}

pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    seq_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache: &RopeCache,
) {
    if seq_pos >= cache.max_seq_len {
        return;
    }

    let half_dim = head_dim / 2;
    let cache_offset = seq_pos * half_dim;

    for h in 0..num_heads {
        let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
        for d in 0..half_dim {
            let cos_theta = cache.cos[cache_offset + d];
            let sin_theta = cache.sin[cache_offset + d];

            let q0 = q_head[d];
            let q1 = q_head[d + half_dim];
            q_head[d] = q0 * cos_theta - q1 * sin_theta;
            q_head[d + half_dim] = q0 * sin_theta + q1 * cos_theta;
        }
    }

    for h in 0..num_kv_heads {
        let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
        for d in 0..half_dim {
            let cos_theta = cache.cos[cache_offset + d];
            let sin_theta = cache.sin[cache_offset + d];

            let k0 = k_head[d];
            let k1 = k_head[d + half_dim];
            k_head[d] = k0 * cos_theta - k1 * sin_theta;
            k_head[d + half_dim] = k0 * sin_theta + k1 * cos_theta;
        }
    }
}
