use alloc::vec::Vec;

pub struct KVCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// One past the highest sequence position any forward pass has handed
    /// out a slot for. `truncate` uses it so rolling back a rejected
    /// speculative tail touches only the positions that were actually
    /// written, instead of memsetting the whole (hundreds of MB) window.
    high_water: usize,
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
            high_water: 0,
        }
    }

    /// One past the highest position handed out by `get_layer_mut` since the
    /// last `reset`/`truncate`. Read-only view for tests and callers that
    /// want to assert what a rollback will touch.
    pub fn high_water(&self) -> usize {
        self.high_water
    }

    pub fn get_layer_mut(&mut self, layer_idx: usize, seq_pos: usize) -> (&mut [f32], &mut [f32]) {
        let layer_size = self.max_seq_len * self.num_kv_heads * self.head_dim;
        let layer_offset = layer_idx * layer_size;
        let seq_offset = seq_pos * self.num_kv_heads * self.head_dim;

        if seq_pos + 1 > self.high_water {
            self.high_water = seq_pos + 1;
        }

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
        self.high_water = 0;
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

    /// Roll the cache back to `len` valid positions: zero every entry from
    /// `len` up to the high-water mark, in every layer, and drop the mark to
    /// `len`. Speculative decoding calls this after a rejected draft tail —
    /// positions `len..` were written by a batched verify pass for tokens
    /// that turned out not to be the greedy continuation, so they must not
    /// be readable by the next pass.
    ///
    /// Bounded by what was actually written, so the cost scales with the
    /// rejected tail (a handful of positions), not with `max_seq_len`.
    pub fn truncate(&mut self, len: usize) {
        let len = len.min(self.max_seq_len);
        if len >= self.high_water {
            // Nothing above `len` has ever been written; setting the mark
            // down is still correct because those slots are already zero.
            self.high_water = len;
            return;
        }
        let slot = self.num_kv_heads * self.head_dim;
        let layer_size = self.max_seq_len * slot;
        let from = len * slot;
        let to = self.high_water.min(self.max_seq_len) * slot;
        for layer in 0..self.num_layers {
            let base = layer * layer_size;
            self.k_cache[base + from..base + to].fill(0.0);
            self.v_cache[base + from..base + to].fill(0.0);
        }
        self.high_water = len;
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

    #[test]
    fn truncate_zeroes_only_positions_at_or_above_len() {
        let mut cache = KVCache::new(2, 3, 4, 5);
        for layer in 0..2 {
            for pos in 0..5 {
                let (k, v) = cache.get_layer_mut(layer, pos);
                k.fill(1.0);
                v.fill(2.0);
            }
        }
        assert_eq!(cache.high_water(), 5);

        cache.truncate(2);
        assert_eq!(cache.high_water(), 2);
        for layer in 0..2 {
            for pos in 0..5 {
                let (k, v) = cache.get_layer_mut(layer, pos);
                let (ek, ev) = if pos < 2 { (1.0, 2.0) } else { (0.0, 0.0) };
                assert!(k.iter().all(|&x| x == ek), "layer {} pos {} k", layer, pos);
                assert!(v.iter().all(|&x| x == ev), "layer {} pos {} v", layer, pos);
            }
        }
    }

    #[test]
    fn truncate_past_high_water_is_a_noop_and_never_panics() {
        let mut cache = KVCache::new(2, 3, 4, 5);
        for pos in 0..3 {
            let (k, _) = cache.get_layer_mut(0, pos);
            k.fill(7.0);
        }
        cache.truncate(99); // beyond the window
        assert_eq!(cache.high_water(), 5);
        let (k, _) = cache.get_layer_mut(0, 0);
        assert!(k.iter().all(|&x| x == 7.0), "truncate above the mark wrote");
    }

    #[test]
    fn truncate_to_zero_matches_reset_on_written_positions() {
        let mut cache = KVCache::new(3, 2, 4, 6);
        for layer in 0..3 {
            for pos in 0..4 {
                let (k, v) = cache.get_layer_mut(layer, pos);
                k.fill(-1.0);
                v.fill(3.0);
            }
        }
        cache.truncate(0);
        assert!(cache.k_cache.iter().all(|&x| x == 0.0));
        assert!(cache.v_cache.iter().all(|&x| x == 0.0));
        assert_eq!(cache.high_water(), 0);
    }
}
