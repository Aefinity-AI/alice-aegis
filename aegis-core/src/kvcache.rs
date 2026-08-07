use alloc::vec::Vec;

pub struct KVCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl KVCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
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

    /// Zero every cached K/V entry. Sequence restarts (a fresh perplexity
    /// pass, a new conversation) must not depend on the write-before-read
    /// discipline of the forward paths holding forever.
    pub fn reset(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
    }

    /// Zero only positions `0..len` of every layer — everything a pass over
    /// `len` tokens can read. A full `reset()` memsets the whole window
    /// (hundreds of MB for a 2B model) even for a 100-token evaluation;
    /// per-chunk evaluators call this instead so the hygiene cost scales
    /// with the work.
    pub fn reset_prefix(&mut self, len: usize) {
        let slot = self.num_kv_heads * self.head_dim;
        let n = len.min(self.max_seq_len) * slot;
        let layer_size = self.max_seq_len * slot;
        for layer in 0..self.num_layers {
            let start = layer * layer_size;
            self.k_cache[start..start + n].fill(0.0);
            self.v_cache[start..start + n].fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_zeroes_every_slot() {
        let mut cache = KVCache::new(2, 3, 4, 5);
        for layer in 0..2 {
            for pos in 0..5 {
                let (k, v) = cache.get_layer_mut(layer, pos);
                k.fill(1.5);
                v.fill(-2.5);
            }
        }
        assert!(cache.k_cache.iter().any(|&x| x != 0.0));
        cache.reset();
        assert!(cache.k_cache.iter().all(|&x| x == 0.0));
        assert!(cache.v_cache.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn reset_prefix_zeroes_exactly_the_prefix_in_every_layer() {
        let mut cache = KVCache::new(2, 3, 4, 5);
        for layer in 0..2 {
            for pos in 0..5 {
                let (k, v) = cache.get_layer_mut(layer, pos);
                k.fill(1.0);
                v.fill(1.0);
            }
        }
        cache.reset_prefix(3);
        for layer in 0..2 {
            for pos in 0..5 {
                let (k, v) = cache.get_layer_mut(layer, pos);
                let expect = if pos < 3 { 0.0 } else { 1.0 };
                assert!(
                    k.iter().all(|&x| x == expect),
                    "layer {} pos {} k",
                    layer,
                    pos
                );
                assert!(
                    v.iter().all(|&x| x == expect),
                    "layer {} pos {} v",
                    layer,
                    pos
                );
            }
        }
        // longer than the window must not panic
        cache.reset_prefix(99);
        assert!(cache.k_cache.iter().all(|&x| x == 0.0));
    }
}
