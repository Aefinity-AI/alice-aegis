use alloc::vec;
use alloc::vec::Vec;

#[repr(C, align(32))]
pub struct WorkingMemoryArena {
    pub hidden_state: Vec<f32>,
    pub norm_state: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub o: Vec<f32>,
    pub up: Vec<f32>,
    pub gate: Vec<f32>,
    pub down: Vec<f32>,
    pub logits: Vec<f32>,
    pub scores: Vec<f32>,
    pub head_out: Vec<f32>,
    pub batch_hidden_state: Vec<f32>,
    pub batch_norm_state: Vec<f32>,
    pub batch_q: Vec<f32>,
    pub batch_k: Vec<f32>,
    pub batch_v: Vec<f32>,
    pub batch_attn_out: Vec<f32>,
    pub batch_o: Vec<f32>,
    pub batch_up: Vec<f32>,
    pub batch_gate: Vec<f32>,
    pub batch_down: Vec<f32>,
    pub penalized: Vec<bool>,
}

impl WorkingMemoryArena {
    pub fn new(
        emb_dim: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        intermediate_size: usize,
        vocab_size: usize,
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        Self {
            hidden_state: vec![0.0f32; emb_dim],
            norm_state: vec![0.0f32; emb_dim],
            q: vec![0.0f32; emb_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn_out: vec![0.0f32; emb_dim],
            o: vec![0.0f32; emb_dim],
            up: vec![0.0f32; intermediate_size],
            gate: vec![0.0f32; intermediate_size],
            down: vec![0.0f32; emb_dim],
            logits: vec![0.0f32; vocab_size],
            scores: vec![0.0f32; max_seq_len],
            head_out: vec![0.0f32; head_dim],

            batch_hidden_state: vec![0.0f32; max_seq_len * emb_dim],
            batch_norm_state: vec![0.0f32; max_seq_len * emb_dim],
            batch_q: vec![0.0f32; max_seq_len * emb_dim],
            batch_k: vec![0.0f32; max_seq_len * kv_dim],
            batch_v: vec![0.0f32; max_seq_len * kv_dim],
            batch_attn_out: vec![0.0f32; max_seq_len * emb_dim],
            batch_o: vec![0.0f32; max_seq_len * emb_dim],
            batch_up: vec![0.0f32; max_seq_len * intermediate_size],
            batch_gate: vec![0.0f32; max_seq_len * intermediate_size],
            batch_down: vec![0.0f32; max_seq_len * emb_dim],
            penalized: vec![false; vocab_size],
        }
    }
}
