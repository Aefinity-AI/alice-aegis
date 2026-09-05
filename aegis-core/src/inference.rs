use alloc::{string::String, vec::Vec};

use crate::arena::WorkingMemoryArena;
use crate::attention::RopeCache;
use crate::kvcache::KVCache;
use crate::model::{FullBitNetPipeline, SafeTensors};
use crate::sampler::Sampler;
use crate::tokenizer::AegisTokenizer;

/// Times `$body` as `$phase` and records it into `$counters`
/// (a `&mut PhaseCycles` lvalue) when the `phase-timers` feature is on.
/// When it is off, the two `#[cfg(...)]`-gated statements below are removed
/// before codegen sees them — nothing is branched around at runtime, the
/// tick calls and the `Phase` reference simply do not exist in the compiled
/// output. See phase_timers.rs module docs for the RDTSC/RDTSCP fencing.
#[cfg(feature = "phase-timers")]
macro_rules! timed_phase {
    ($counters:expr, $phase:expr, $body:block) => {{
        let __ts = crate::phase_timers::tick_start();
        let __ret = $body;
        let __te = crate::phase_timers::tick_end();
        $counters.record($phase, __ts, __te);
        __ret
    }};
}
#[cfg(not(feature = "phase-timers"))]
macro_rules! timed_phase {
    ($counters:expr, $phase:expr, $body:block) => {
        $body
    };
}

/// Quantize one token's activation vector onto the int8 grid, if the
/// `int8_act` feature is on. No-op otherwise, so the f32 path is unchanged.
#[inline]
fn quant_act(x: &mut [f32]) {
    if cfg!(feature = "int8_act") {
        crate::ops::quantize_activations_int8(x);
    }
}

/// Same, for a batch: BitNet's activation quantization is PER TOKEN, so each
/// row gets its own absmax scale. Quantizing the whole batch buffer at once
/// would share one scale across tokens and is wrong.
#[inline]
fn quant_act_batch(x: &mut [f32], batch: usize, dim: usize) {
    if cfg!(feature = "int8_act") {
        for b in 0..batch {
            crate::ops::quantize_activations_int8(&mut x[b * dim..(b + 1) * dim]);
        }
    }
}

/// One record per (token, layer): how many elements of the down_proj input
/// vector (`arena.up` after gate activation, optional ffn_sub_norm, and
/// activation quantization) were exactly 0.0 at the moment the down_proj
/// kernel consumed them. `act_stats` feature only.
#[cfg(feature = "act_stats")]
#[derive(Clone, Copy, Debug)]
pub struct ActZeroRecord {
    pub layer: usize,
    pub zeros: usize,
    pub len: usize,
    /// true = recorded in the batched prefill path, false = decode.
    pub prefill: bool,
}

/// One captured down_proj input vector (`act_stats` build + runtime opt-in
/// via `act_dump_enabled`): the exact f32 contents of the vector at the
/// moment the down_proj kernel consumed it (post-activation, post-sub_norm,
/// post-quant — same instant `ActZeroRecord` counts). Fuel for benches that
/// need REAL activation vectors (benches/colskip_vs_incumbent.rs) instead of
/// synthetic zero patterns, which under-model clustering.
#[cfg(feature = "act_stats")]
#[derive(Clone, Debug)]
pub struct ActDumpRecord {
    pub layer: usize,
    /// true = recorded in the batched prefill path, false = decode.
    pub prefill: bool,
    pub values: Vec<f32>,
}

pub struct TernaryInferenceEngine<'a> {
    pipeline: FullBitNetPipeline<'a>,
    pub config: crate::model::ModelConfig,
    pub tokenizer: AegisTokenizer<'a>,
    kv_cache: KVCache,
    sampler: Sampler,
    rope_cache: RopeCache,
    arena: WorkingMemoryArena,
    /// Cycles spent in the last prefill, and how many tokens it covered.
    pub last_prefill_cycles: u64,
    pub last_prefill_tokens: usize,
    /// Cycles spent in the last decode loop, and how many steps it ran.
    /// Callers use these to report decode-only rates: a fixed prefill cost
    /// amortized into a whole-run average fakes a speedup at longer outputs.
    pub last_decode_cycles: u64,
    pub last_decode_steps: u64,
    /// Down_proj-input zero counts, appended per (token, layer) as forward
    /// passes run. Diagnostic only (allocates); callers drain between runs.
    #[cfg(feature = "act_stats")]
    pub act_zero_counts: Vec<ActZeroRecord>,
    /// When true, every down_proj input vector is copied into `act_dump`
    /// (~27 KB per (token, layer) on BitNet-2B). OFF by default; callers
    /// drain between runs.
    #[cfg(feature = "act_stats")]
    pub act_dump_enabled: bool,
    #[cfg(feature = "act_stats")]
    pub act_dump: Vec<ActDumpRecord>,
    /// Amdahl-decomposition per-phase cycle accumulators. Fixed-size, zero
    /// allocation; callers reset between measurement windows with
    /// `reset_phase_cycles`. `phase-timers` feature only.
    #[cfg(feature = "phase-timers")]
    pub phase_cycles: crate::phase_timers::PhaseCycles,
}

impl<'a> TernaryInferenceEngine<'a> {
    pub fn new(
        embeddings_bytes: &'a [u8],
        model_bytes: &'a [u8],
        vocab_bytes: &'a [u8],
    ) -> Result<Self, String> {
        let tensors = SafeTensors::deserialize(model_bytes)?;
        let tokenizer = AegisTokenizer::new(vocab_bytes)?;

        // The model's own config travels inside MODEL.SAF (safetensors
        // __metadata__, written by the forge). Artifacts forged before that
        // convention carry none and get the baked BitNet-2B config. A
        // present-but-malformed config is a load error (metadata_field),
        // never a silent fallback to the wrong model family.
        let config = match tensors.metadata_field("aegis_config")? {
            Some(cfg_json) => {
                let cfg = crate::model::ModelConfig::from_json(&cfg_json)?;
                // A forge-authored config records the vocab it was packed
                // with; disagreement means mispaired artifacts.
                if cfg.vocab_size != tokenizer.vocab_len() {
                    return Err(alloc::format!(
                        "artifact mismatch: MODEL.SAF config says vocab_size {}, VOCAB.BIN carries {} tokens",
                        cfg.vocab_size,
                        tokenizer.vocab_len()
                    ));
                }
                cfg
            }
            None => {
                // Legacy artifacts predate configs carrying vocab_size for
                // the pruned vocab, so the tokenizer is authoritative here.
                let config_str = include_str!("../../aegis-forge/aegis_pruned_config.json");
                let mut cfg = crate::model::ModelConfig::from_json(config_str)?;
                cfg.vocab_size = tokenizer.vocab_len();
                cfg
            }
        };

        // Hard check, not debug_assert: release bare-metal builds must also
        // refuse a mispaired or truncated embedding table.
        let expect_embed = config.vocab_size * config.hidden_size * 2;
        if embeddings_bytes.len() != expect_embed {
            return Err(alloc::format!(
                "EMBED.BIN: {} bytes, expected {} ({} x {} BF16) — mispaired artifacts?",
                embeddings_bytes.len(),
                expect_embed,
                config.vocab_size,
                config.hidden_size
            ));
        }

        let pipeline = FullBitNetPipeline::new(&tensors, embeddings_bytes, &config)?;

        // Validate head config before any division by it.
        if config.num_attention_heads == 0 || config.num_key_value_heads == 0 {
            return Err(String::from(
                "config: num_attention_heads / num_key_value_heads must be nonzero",
            ));
        }
        if config.num_attention_heads % config.num_key_value_heads != 0 {
            return Err(alloc::format!(
                "config: num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
                config.num_attention_heads,
                config.num_key_value_heads
            ));
        }
        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(String::from(
                "config: hidden_size must be divisible by num_attention_heads",
            ));
        }

        let emb_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = emb_dim / num_heads;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_seq = config.max_position_embeddings;

        let kv_cache = KVCache::new(pipeline.layers.len(), num_kv_heads, head_dim, max_seq);
        let sampler = Sampler::new(0.0, 1.0);
        let rope_cache = RopeCache::new(max_seq, head_dim, config.rope_theta);

        // Zero-allocation arena protects us from OOM
        let arena = WorkingMemoryArena::new(
            emb_dim,
            num_kv_heads,
            head_dim,
            max_seq,
            intermediate_size,
            vocab_size,
        );

        Ok(Self {
            pipeline,
            config,
            tokenizer,
            kv_cache,
            sampler,
            rope_cache,
            arena,
            last_prefill_cycles: 0,
            last_prefill_tokens: 0,
            last_decode_cycles: 0,
            last_decode_steps: 0,
            #[cfg(feature = "act_stats")]
            act_zero_counts: Vec::new(),
            #[cfg(feature = "act_stats")]
            act_dump_enabled: false,
            #[cfg(feature = "act_stats")]
            act_dump: Vec::new(),
            #[cfg(feature = "phase-timers")]
            phase_cycles: crate::phase_timers::PhaseCycles::zero(),
        })
    }

    /// Zeroes the phase-cycle accumulators. Call between measurement windows
    /// (e.g. once per context-length run) so each AMDAHL line reports only
    /// the decode it names. `phase-timers` feature only.
    #[cfg(feature = "phase-timers")]
    pub fn reset_phase_cycles(&mut self) {
        self.phase_cycles = crate::phase_timers::PhaseCycles::zero();
    }

    /// Read-only access to the loaded weight views — the CIS-1 integer path
    /// (`crate::cis_infer`) converts its tables from the same artifacts the
    /// float path serves, so both measure the identical checkpoint.
    pub fn pipeline(&self) -> &FullBitNetPipeline<'a> {
        &self.pipeline
    }

    fn forward_batch(
        &mut self,
        batch_tokens: &[u32],
        seq_pos_start: usize,
        print_cb: impl FnMut(&str),
    ) {
        self.forward_batch_with_capture(batch_tokens, seq_pos_start, print_cb, None);
    }

    /// `forward_batch` with an optional per-layer probe: after every decoder
    /// layer, the residual-stream state of the whole batch (pre final-norm)
    /// is appended to `capture`. This is the reference-parity hook (T2b) —
    /// an HF dump of the same prompt is compared layer by layer, which
    /// localizes a stride/dtype/graph bug to the exact layer that diverges.
    /// Diagnostic path: allocates per layer; never call it from the hot loop.
    fn forward_batch_with_capture(
        &mut self,
        batch_tokens: &[u32],
        seq_pos_start: usize,
        mut print_cb: impl FnMut(&str),
        mut capture: Option<&mut Vec<Vec<f32>>>,
    ) {
        let msg = alloc::format!("[SYSTEM] Analyzing {} tokens...\r\n", batch_tokens.len());
        print_cb(&msg);
        let batch_size = batch_tokens.len();
        let emb_dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = emb_dim / num_heads;
        let intermediate_size = self.config.intermediate_size;
        let eps = self.config.rms_norm_eps;
        let act = self.config.hidden_act;

        self.arena.batch_hidden_state[0..batch_size * emb_dim].fill(0.0);

        for b in 0..batch_size {
            let current_tok = batch_tokens[b];
            let start = current_tok as usize * emb_dim * 2;
            let hidden_offset = b * emb_dim;

            // Guard as forward_step does: an out-of-range id must not panic.
            if start + emb_dim * 2 > self.pipeline.embeddings.len() {
                continue; // leaves this row zeroed
            }
            for i in 0..emb_dim {
                let o = start + i * 2;
                let f = f32::from_le_bytes([
                    0,
                    0,
                    self.pipeline.embeddings[o],
                    self.pipeline.embeddings[o + 1],
                ]);
                self.arena.batch_hidden_state[hidden_offset + i] = f;
            }
        }

        for (layer_idx, layer) in self.pipeline.layers.iter().enumerate() {
            self.arena.batch_norm_state[0..batch_size * emb_dim]
                .copy_from_slice(&self.arena.batch_hidden_state[0..batch_size * emb_dim]);
            for b in 0..batch_size {
                let offset = b * emb_dim;
                crate::ops::rmsnorm(
                    &mut self.arena.batch_norm_state[offset..offset + emb_dim],
                    layer.input_layernorm_weight.data(),
                    eps,
                );
            }

            quant_act_batch(
                &mut self.arena.batch_norm_state[0..batch_size * emb_dim],
                batch_size,
                emb_dim,
            );

            self.arena.batch_q[0..batch_size * emb_dim].fill(0.0);
            self.arena.batch_k[0..batch_size * (num_kv_heads * head_dim)].fill(0.0);
            self.arena.batch_v[0..batch_size * (num_kv_heads * head_dim)].fill(0.0);

            crate::ops::ternary_matmul(
                &mut self.arena.batch_q,
                &self.arena.batch_norm_state[0..batch_size * emb_dim],
                layer.q_proj.data(),
                batch_size,
                emb_dim,
                emb_dim,
                layer.q_proj_scale,
            );
            crate::ops::ternary_matmul(
                &mut self.arena.batch_k,
                &self.arena.batch_norm_state[0..batch_size * emb_dim],
                layer.k_proj.data(),
                batch_size,
                num_kv_heads * head_dim,
                emb_dim,
                layer.k_proj_scale,
            );
            crate::ops::ternary_matmul(
                &mut self.arena.batch_v,
                &self.arena.batch_norm_state[0..batch_size * emb_dim],
                layer.v_proj.data(),
                batch_size,
                num_kv_heads * head_dim,
                emb_dim,
                layer.v_proj_scale,
            );

            for b in 0..batch_size {
                let q_offset = b * emb_dim;
                let kv_offset = b * (num_kv_heads * head_dim);
                crate::attention::apply_rope(
                    &mut self.arena.batch_q[q_offset..q_offset + emb_dim],
                    &mut self.arena.batch_k[kv_offset..kv_offset + (num_kv_heads * head_dim)],
                    seq_pos_start + b,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    &self.rope_cache,
                );

                let (k_slot, v_slot) = self.kv_cache.get_layer_mut(layer_idx, seq_pos_start + b);
                k_slot.copy_from_slice(
                    &self.arena.batch_k[kv_offset..kv_offset + (num_kv_heads * head_dim)],
                );
                v_slot.copy_from_slice(
                    &self.arena.batch_v[kv_offset..kv_offset + (num_kv_heads * head_dim)],
                );
            }

            self.arena.batch_attn_out[0..batch_size * emb_dim].fill(0.0);

            for b in 0..batch_size {
                let q_offset = b * emb_dim;
                let attn_out_offset = b * emb_dim;
                let current_seq_pos = seq_pos_start + b;

                for h in 0..num_heads {
                    let kv_h = h / (num_heads / num_kv_heads);
                    for t in 0..=current_seq_pos {
                        let (k_cache, _) = self.kv_cache.get_layer_mut(layer_idx, t);
                        let score = crate::ops::attn_dot(
                            &self.arena.batch_q
                                [q_offset + h * head_dim..q_offset + h * head_dim + head_dim],
                            &k_cache[kv_h * head_dim..kv_h * head_dim + head_dim],
                        );
                        self.arena.scores[t] = score / libm::sqrtf(head_dim as f32);
                    }

                    crate::ops::softmax(&mut self.arena.scores[0..=current_seq_pos]);
                    self.arena.head_out.fill(0.0);

                    for t in 0..=current_seq_pos {
                        let (_, v_cache) = self.kv_cache.get_layer_mut(layer_idx, t);
                        let w = self.arena.scores[t];
                        crate::ops::attn_madd(
                            &mut self.arena.head_out[0..head_dim],
                            w,
                            &v_cache[kv_h * head_dim..kv_h * head_dim + head_dim],
                        );
                    }
                    for d in 0..head_dim {
                        self.arena.batch_attn_out[attn_out_offset + h * head_dim + d] =
                            self.arena.head_out[d];
                    }
                }
            }

            self.arena.batch_o[0..batch_size * emb_dim].fill(0.0);
            if let Some(sub_norm) = &layer.attn_sub_norm {
                for b in 0..batch_size {
                    let offset = b * emb_dim;
                    crate::ops::rmsnorm(
                        &mut self.arena.batch_attn_out[offset..offset + emb_dim],
                        sub_norm.data(),
                        eps,
                    );
                }
            }
            quant_act_batch(
                &mut self.arena.batch_attn_out[0..batch_size * emb_dim],
                batch_size,
                emb_dim,
            );
            crate::ops::ternary_matmul(
                &mut self.arena.batch_o,
                &self.arena.batch_attn_out[0..batch_size * emb_dim],
                layer.o_proj.data(),
                batch_size,
                emb_dim,
                emb_dim,
                layer.o_proj_scale,
            );

            for i in 0..(batch_size * emb_dim) {
                self.arena.batch_hidden_state[i] += self.arena.batch_o[i];
            }

            self.arena.batch_norm_state[0..batch_size * emb_dim]
                .copy_from_slice(&self.arena.batch_hidden_state[0..batch_size * emb_dim]);
            for b in 0..batch_size {
                let offset = b * emb_dim;
                crate::ops::rmsnorm(
                    &mut self.arena.batch_norm_state[offset..offset + emb_dim],
                    layer.post_attention_layernorm_weight.data(),
                    eps,
                );
            }

            quant_act_batch(
                &mut self.arena.batch_norm_state[0..batch_size * emb_dim],
                batch_size,
                emb_dim,
            );

            self.arena.batch_up[0..batch_size * intermediate_size].fill(0.0);
            self.arena.batch_gate[0..batch_size * intermediate_size].fill(0.0);

            crate::ops::ternary_matmul(
                &mut self.arena.batch_up,
                &self.arena.batch_norm_state[0..batch_size * emb_dim],
                layer.up_proj.data(),
                batch_size,
                intermediate_size,
                emb_dim,
                layer.up_proj_scale,
            );
            crate::ops::ternary_matmul(
                &mut self.arena.batch_gate,
                &self.arena.batch_norm_state[0..batch_size * emb_dim],
                layer.gate_proj.data(),
                batch_size,
                intermediate_size,
                emb_dim,
                layer.gate_proj_scale,
            );

            for b in 0..batch_size {
                let offset = b * intermediate_size;
                let gate = &mut self.arena.batch_gate[offset..offset + intermediate_size];
                match act {
                    crate::model::Activation::Relu2 => crate::ops::relu2(gate),
                    crate::model::Activation::Silu => crate::ops::silu(gate),
                }
            }
            for i in 0..(batch_size * intermediate_size) {
                self.arena.batch_up[i] *= self.arena.batch_gate[i];
            }

            if let Some(sub_norm) = &layer.ffn_sub_norm {
                for b in 0..batch_size {
                    let offset = b * intermediate_size;
                    crate::ops::rmsnorm(
                        &mut self.arena.batch_up[offset..offset + intermediate_size],
                        sub_norm.data(),
                        eps,
                    );
                }
            }

            quant_act_batch(
                &mut self.arena.batch_up[0..batch_size * intermediate_size],
                batch_size,
                intermediate_size,
            );

            // Count exact zeros in the down_proj input, exactly as the kernel
            // will consume it (post-activation, post-sub_norm, post-quant).
            #[cfg(feature = "act_stats")]
            for b in 0..batch_size {
                let row = &self.arena.batch_up[b * intermediate_size..(b + 1) * intermediate_size];
                self.act_zero_counts.push(ActZeroRecord {
                    layer: layer_idx,
                    zeros: row.iter().filter(|v| **v == 0.0).count(),
                    len: intermediate_size,
                    prefill: true,
                });
                if self.act_dump_enabled {
                    self.act_dump.push(ActDumpRecord {
                        layer: layer_idx,
                        prefill: true,
                        values: row.to_vec(),
                    });
                }
            }

            self.arena.batch_down[0..batch_size * emb_dim].fill(0.0);
            crate::ops::ternary_matmul(
                &mut self.arena.batch_down,
                &self.arena.batch_up[0..batch_size * intermediate_size],
                layer.down_proj.data(),
                batch_size,
                emb_dim,
                intermediate_size,
                layer.down_proj_scale,
            );

            for i in 0..(batch_size * emb_dim) {
                self.arena.batch_hidden_state[i] += self.arena.batch_down[i];
            }

            if let Some(cap) = capture.as_deref_mut() {
                cap.push(self.arena.batch_hidden_state[0..batch_size * emb_dim].to_vec());
            }
        }
    }

    /// Run `tokens` from position 0 and return the residual-stream state
    /// after every layer, `layers × (tokens × hidden)`, pre final-norm.
    /// Reference-parity harness only (see `tests/reference_parity.rs`).
    pub fn capture_layer_hidden_states(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
        assert!(
            tokens.len() <= self.config.max_position_embeddings,
            "capture prompt exceeds the KV/arena window"
        );
        let mut states = Vec::with_capacity(self.pipeline.layers.len());
        self.kv_cache.reset_prefix(tokens.len());
        self.forward_batch_with_capture(tokens, 0, |_| {}, Some(&mut states));
        states
    }

    pub fn calculate_perplexity(&mut self, tokens: &[u32]) -> f64 {
        // Positions beyond the KV/arena window would index out of range
        // (a fault on bare metal) or spill into the next layer's cache.
        // Callers are expected to pre-clamp and warn; this is the backstop.
        let window = self.config.max_position_embeddings;
        let tokens = if tokens.len() > window {
            &tokens[..window]
        } else {
            tokens
        };
        if tokens.len() < 2 {
            return 0.0;
        }

        // Every evaluation restarts the sequence at position 0: clear the
        // positions this pass can read, explicitly, rather than relying on
        // write-before-read ordering (a full reset would memset the whole
        // window on every chunk).
        self.kv_cache.reset_prefix(tokens.len());

        let mut total_nll = 0.0f64;
        let mut count = 0;
        let emb_dim = self.config.hidden_size;

        // Process the first token to seed the KV cache and get the hidden state
        self.forward_batch(&tokens[0..1], 0, |_| {});
        self.arena
            .hidden_state
            .copy_from_slice(&self.arena.batch_hidden_state[0..emb_dim]);

        for i in 0..tokens.len() - 1 {
            let target_tok = tokens[i + 1];

            crate::ops::rmsnorm(
                &mut self.arena.hidden_state,
                self.pipeline.final_norm.data(),
                self.config.rms_norm_eps,
            );
            crate::ops::f32_dot_argmax(
                &mut self.arena.logits,
                &self.arena.hidden_state,
                self.pipeline.lm_head_bytes(),
                self.config.vocab_size,
                emb_dim,
            );

            let mut max_logit = f32::NEG_INFINITY;
            for l in &self.arena.logits {
                if *l > max_logit {
                    max_logit = *l;
                }
            }

            let mut sum_exp = 0.0;
            for l in &self.arena.logits {
                sum_exp += libm::expf(*l - max_logit);
            }

            // encode() can only produce in-vocab ids, but this API now takes
            // raw ids from callers (reference fixtures, tests). An id past
            // the vocab must surface as NaN — loud in every report line —
            // not as an index panic (a fault on bare metal).
            if target_tok as usize >= self.arena.logits.len() {
                return f64::NAN;
            }
            let target_logit = self.arena.logits[target_tok as usize];
            let nll = -((target_logit - max_logit) - libm::logf(sum_exp));

            total_nll += nll as f64;
            count += 1;

            // Advance the state with the NEXT token so iteration i+1 predicts tokens[i+2]
            self.forward_step(tokens[i + 1], i + 1);
        }

        libm::exp(total_nll / count as f64)
    }

    fn forward_step(&mut self, current_tok: u32, seq_pos: usize) {
        let emb_dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = emb_dim / num_heads;
        let intermediate_size = self.config.intermediate_size;
        let eps = self.config.rms_norm_eps;
        let act = self.config.hidden_act;

        self.arena.hidden_state.fill(0.0);

        let start = current_tok as usize * emb_dim * 2;
        if start + emb_dim * 2 <= self.pipeline.embeddings.len() {
            for i in 0..emb_dim {
                let offset = start + i * 2;
                let f_bytes = [
                    0,
                    0,
                    self.pipeline.embeddings[offset],
                    self.pipeline.embeddings[offset + 1],
                ];
                self.arena.hidden_state[i] += f32::from_le_bytes(f_bytes);
            }
        }

        for (layer_idx, layer) in self.pipeline.layers.iter().enumerate() {
            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Norm, {
                self.arena
                    .norm_state
                    .copy_from_slice(&self.arena.hidden_state);
                crate::ops::rmsnorm(
                    &mut self.arena.norm_state,
                    layer.input_layernorm_weight.data(),
                    eps,
                );
            });
            quant_act(&mut self.arena.norm_state);

            self.arena.q.fill(0.0);
            self.arena.k.fill(0.0);
            self.arena.v.fill(0.0);

            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Gemv, {
                crate::ops::ternary_matvec(
                    &mut self.arena.q,
                    &self.arena.norm_state,
                    layer.q_proj.data(),
                    emb_dim,
                    emb_dim,
                    layer.q_proj_scale,
                );
                crate::ops::ternary_matvec(
                    &mut self.arena.k,
                    &self.arena.norm_state,
                    layer.k_proj.data(),
                    num_kv_heads * head_dim,
                    emb_dim,
                    layer.k_proj_scale,
                );
                crate::ops::ternary_matvec(
                    &mut self.arena.v,
                    &self.arena.norm_state,
                    layer.v_proj.data(),
                    num_kv_heads * head_dim,
                    emb_dim,
                    layer.v_proj_scale,
                );
            });

            crate::attention::apply_rope(
                &mut self.arena.q,
                &mut self.arena.k,
                seq_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                &self.rope_cache,
            );

            // KV-cache WRITE only. The read side lives inside the Attn span
            // below — see Phase::Kv's doc comment in phase_timers.rs for why.
            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Kv, {
                let (k_slot, v_slot) = self.kv_cache.get_layer_mut(layer_idx, seq_pos);
                k_slot.copy_from_slice(&self.arena.k);
                v_slot.copy_from_slice(&self.arena.v);
            });

            self.arena.attn_out.fill(0.0);

            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Attn, {
                for h in 0..num_heads {
                    let kv_h = h / (num_heads / num_kv_heads);
                    for t in 0..=seq_pos {
                        let (k_cache, _) = self.kv_cache.get_layer_mut(layer_idx, t);
                        let score = crate::ops::attn_dot(
                            &self.arena.q[h * head_dim..h * head_dim + head_dim],
                            &k_cache[kv_h * head_dim..kv_h * head_dim + head_dim],
                        );
                        self.arena.scores[t] = score / libm::sqrtf(head_dim as f32);
                    }

                    crate::ops::softmax(&mut self.arena.scores[0..=seq_pos]);
                    self.arena.head_out.fill(0.0);

                    for t in 0..=seq_pos {
                        let (_, v_cache) = self.kv_cache.get_layer_mut(layer_idx, t);
                        let w = self.arena.scores[t];
                        crate::ops::attn_madd(
                            &mut self.arena.head_out[0..head_dim],
                            w,
                            &v_cache[kv_h * head_dim..kv_h * head_dim + head_dim],
                        );
                    }
                    for d in 0..head_dim {
                        self.arena.attn_out[h * head_dim + d] = self.arena.head_out[d];
                    }
                }
            });

            self.arena.o.fill(0.0);
            if let Some(sub_norm) = &layer.attn_sub_norm {
                timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Norm, {
                    crate::ops::rmsnorm(&mut self.arena.attn_out, sub_norm.data(), eps);
                });
            }
            quant_act(&mut self.arena.attn_out);
            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Gemv, {
                crate::ops::ternary_matvec(
                    &mut self.arena.o,
                    &self.arena.attn_out,
                    layer.o_proj.data(),
                    emb_dim,
                    emb_dim,
                    layer.o_proj_scale,
                );
            });
            for i in 0..emb_dim {
                self.arena.hidden_state[i] += self.arena.o[i];
            }

            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Norm, {
                self.arena
                    .norm_state
                    .copy_from_slice(&self.arena.hidden_state);
                crate::ops::rmsnorm(
                    &mut self.arena.norm_state,
                    layer.post_attention_layernorm_weight.data(),
                    eps,
                );
            });
            quant_act(&mut self.arena.norm_state);

            self.arena.up.fill(0.0);
            self.arena.gate.fill(0.0);
            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Gemv, {
                crate::ops::ternary_matvec(
                    &mut self.arena.up,
                    &self.arena.norm_state,
                    layer.up_proj.data(),
                    intermediate_size,
                    emb_dim,
                    layer.up_proj_scale,
                );
                crate::ops::ternary_matvec(
                    &mut self.arena.gate,
                    &self.arena.norm_state,
                    layer.gate_proj.data(),
                    intermediate_size,
                    emb_dim,
                    layer.gate_proj_scale,
                );
            });

            match act {
                crate::model::Activation::Relu2 => crate::ops::relu2(&mut self.arena.gate),
                crate::model::Activation::Silu => crate::ops::silu(&mut self.arena.gate),
            }
            for i in 0..intermediate_size {
                self.arena.up[i] *= self.arena.gate[i];
            }

            if let Some(sub_norm) = &layer.ffn_sub_norm {
                timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Norm, {
                    crate::ops::rmsnorm(&mut self.arena.up, sub_norm.data(), eps);
                });
            }
            quant_act(&mut self.arena.up);

            // Count exact zeros in the down_proj input, exactly as the kernel
            // will consume it (post-activation, post-sub_norm, post-quant).
            #[cfg(feature = "act_stats")]
            {
                let v = &self.arena.up[..intermediate_size];
                self.act_zero_counts.push(ActZeroRecord {
                    layer: layer_idx,
                    zeros: v.iter().filter(|v| **v == 0.0).count(),
                    len: intermediate_size,
                    prefill: false,
                });
                if self.act_dump_enabled {
                    self.act_dump.push(ActDumpRecord {
                        layer: layer_idx,
                        prefill: false,
                        values: v.to_vec(),
                    });
                }
            }

            self.arena.down.fill(0.0);
            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Gemv, {
                crate::ops::ternary_matvec(
                    &mut self.arena.down,
                    &self.arena.up,
                    layer.down_proj.data(),
                    emb_dim,
                    intermediate_size,
                    layer.down_proj_scale,
                );
            });
            for i in 0..emb_dim {
                self.arena.hidden_state[i] += self.arena.down[i];
            }
        }
    }

    pub fn process_intent(
        &mut self,
        intent: &str,
        max_new_tokens: usize,
        mut print_cb: impl FnMut(&str),
    ) -> String {
        let mut tokens = Vec::new();

        let push_tok =
            |toks: &mut Vec<u32>, tokenizer: &crate::tokenizer::AegisTokenizer<'a>, sym: &str| {
                if let Some(id) = tokenizer.get_token_id(sym) {
                    toks.push(id);
                }
            };

        match self.config.chat_template {
            crate::model::ChatTemplate::Llama3 => {
                push_tok(&mut tokens, &self.tokenizer, "<|begin_of_text|>");
                push_tok(&mut tokens, &self.tokenizer, "<|start_header_id|>");
                tokens.extend(self.tokenizer.encode("user"));
                push_tok(&mut tokens, &self.tokenizer, "<|end_header_id|>");
                push_tok(&mut tokens, &self.tokenizer, "ĊĊ"); // "\n\n"; tokenizer falls back
                tokens.extend(self.tokenizer.encode(intent));
                push_tok(&mut tokens, &self.tokenizer, "<|eot_id|>");
                push_tok(&mut tokens, &self.tokenizer, "<|start_header_id|>");
                tokens.extend(self.tokenizer.encode("assistant"));
                push_tok(&mut tokens, &self.tokenizer, "<|end_header_id|>");
                push_tok(&mut tokens, &self.tokenizer, "ĊĊ");
            }
            crate::model::ChatTemplate::ChatMl => {
                // {bos}<|im_start|>user\n{intent}<|im_end|>\n<|im_start|>assistant\n
                push_tok(&mut tokens, &self.tokenizer, "<|begin_of_text|>");
                push_tok(&mut tokens, &self.tokenizer, "<|im_start|>");
                tokens.extend(self.tokenizer.encode("user\n"));
                tokens.extend(self.tokenizer.encode(intent));
                push_tok(&mut tokens, &self.tokenizer, "<|im_end|>");
                tokens.extend(self.tokenizer.encode("\n"));
                push_tok(&mut tokens, &self.tokenizer, "<|im_start|>");
                tokens.extend(self.tokenizer.encode("assistant\n"));
            }
            crate::model::ChatTemplate::None => {
                tokens.extend(self.tokenizer.encode(intent));
            }
        }

        if tokens.is_empty() {
            return String::from("No valid tokens.");
        }

        // Prevent context overflow: keep the prompt inside the KV/RoPE window
        // (config-driven — the old hardcoded 1948 assumed the BitNet 2048
        // window), leaving headroom to generate. On tiny windows the fixed
        // 100-token headroom would swallow the whole prompt, so keep at
        // least half the window for it.
        let window = self.config.max_position_embeddings;
        let prompt_cap = window.saturating_sub(100).max(window / 2).max(1);
        if tokens.len() > prompt_cap {
            let start = tokens.len() - prompt_cap;
            tokens = tokens[start..].to_vec();
        }

        let mut generated_tokens = Vec::new();
        let emb_dim = self.pipeline.hidden_dim;
        let mut next_token;

        // --- PREFILL PHASE ---
        let prefill_start = unsafe { core::arch::x86_64::_rdtsc() };
        self.forward_batch(&tokens, 0, &mut print_cb);
        let prefill_cycles = unsafe { core::arch::x86_64::_rdtsc() } - prefill_start;
        self.last_prefill_cycles = prefill_cycles;
        self.last_prefill_tokens = tokens.len();

        let final_token_offset = (tokens.len() - 1) * emb_dim;
        self.arena.hidden_state.copy_from_slice(
            &self.arena.batch_hidden_state[final_token_offset..final_token_offset + emb_dim],
        );

        crate::ops::rmsnorm(
            &mut self.arena.hidden_state,
            self.pipeline.final_norm.data(),
            self.config.rms_norm_eps,
        );
        next_token = crate::ops::f32_dot_argmax(
            &mut self.arena.logits,
            &self.arena.hidden_state,
            self.pipeline.lm_head_bytes(),
            self.config.vocab_size,
            emb_dim,
        );
        let decoded_word = self.tokenizer.decode(&[next_token]);
        print_cb(&decoded_word);
        generated_tokens.push(next_token);

        let mut total_cycles = 0;
        let mut steps = 0;

        // --- DECODE PHASE ---
        let mut step = tokens.len();
        let eos_token = self.tokenizer.get_token_id("<|end_of_text|>").unwrap_or(0);
        let eot_token = self.tokenizer.get_token_id("<|eot_id|>").unwrap_or(0);
        // ChatML turn terminator; Option-compared so vocabularies without
        // it (BitNet/Llama-3) never match a real id-0 token by accident.
        let imend_token = self.tokenizer.get_token_id("<|im_end|>");

        // Positions >= window index past the KV/RoPE caches — an index panic
        // here, a memory fault on bare metal. Hit for real at 2,027 generated
        // tokens in the 2026-07-14 energy run (the old bound was only
        // prompt + max_new_tokens). Stop at the window; callers get a clean
        // end of generation instead of a crash.
        let ctx_limit = (tokens.len() + max_new_tokens).min(window);
        while step < ctx_limit {
            if next_token == eos_token || next_token == eot_token || Some(next_token) == imend_token
            {
                break;
            }

            let t_start = unsafe { core::arch::x86_64::_rdtsc() };
            // Amdahl total span: fenced separately from `t_start`/`t_end`
            // above (which are the engine's pre-existing, unfenced
            // last_decode_cycles bookkeeping — untouched here) so the
            // phase-timers sum_check is computed against a boundary that
            // used the SAME fencing discipline as every phase span it is
            // compared to. `phase-timers` feature only.
            #[cfg(feature = "phase-timers")]
            let __amdahl_total_start = crate::phase_timers::tick_start();

            self.forward_step(next_token, step);

            timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Norm, {
                crate::ops::rmsnorm(
                    &mut self.arena.hidden_state,
                    self.pipeline.final_norm.data(),
                    self.config.rms_norm_eps,
                );
            });
            next_token = timed_phase!(self.phase_cycles, crate::phase_timers::Phase::LmHead, {
                crate::ops::f32_dot_argmax(
                    &mut self.arena.logits,
                    &self.arena.hidden_state,
                    self.pipeline.lm_head_bytes(),
                    self.config.vocab_size,
                    emb_dim,
                )
            });

            next_token = timed_phase!(self.phase_cycles, crate::phase_timers::Phase::Sample, {
                self.arena.penalized.fill(false);
                let mut argmax_penalized = false;
                for &tok in &generated_tokens {
                    let idx = tok as usize;
                    if idx < self.arena.logits.len() && !self.arena.penalized[idx] {
                        self.arena.penalized[idx] = true;
                        if idx as u32 == next_token {
                            argmax_penalized = true;
                        }
                        let logit = &mut self.arena.logits[idx];
                        if *logit > 0.0 {
                            *logit /= 1.2;
                        } else {
                            *logit *= 1.2;
                        }
                        *logit -= 2.0;
                    }
                }

                if argmax_penalized {
                    self.sampler.argmax(&self.arena.logits)
                } else {
                    next_token
                }
            });

            #[cfg(feature = "phase-timers")]
            {
                let __amdahl_total_end = crate::phase_timers::tick_end();
                self.phase_cycles
                    .record_total(__amdahl_total_start, __amdahl_total_end);
            }

            if next_token == eos_token || next_token == eot_token || Some(next_token) == imend_token
            {
                break;
            }
            let decoded_word = self.tokenizer.decode(&[next_token]);
            print_cb(&decoded_word);
            generated_tokens.push(next_token);

            let t_end = unsafe { core::arch::x86_64::_rdtsc() };
            total_cycles += t_end - t_start;
            steps += 1;
            step += 1;
        }

        self.last_decode_cycles = total_cycles;
        self.last_decode_steps = steps;

        let avg_cycles = total_cycles.checked_div(steps).unwrap_or(0);
        print_cb(&alloc::format!(
            "

[PERFORMANCE] Average Cycles/Token: {}
",
            avg_cycles
        ));

        self.tokenizer.decode(&generated_tokens)
    }

    /// Prefill/decode parity: run `tokens` through `forward_batch`, then replay
    /// the same tokens one position at a time through `forward_step`, comparing
    /// the final-layer hidden state (pre final-norm) at every position. Returns
    /// the maximum absolute elementwise difference — 0.0 means bit-identical.
    ///
    /// No KV reset is needed: both paths write a position's KV entry before any
    /// read of it, so the replay overwrites the batch pass's cache in place.
    /// Diagnostic path — allocates a copy of the batch hidden states; never
    /// call it from the hot loop.
    pub fn prefill_decode_parity(&mut self, tokens: &[u32]) -> f32 {
        let emb_dim = self.pipeline.hidden_dim;
        assert!(!tokens.is_empty(), "parity needs at least one token");
        assert!(
            tokens.len() * emb_dim <= self.arena.batch_hidden_state.len(),
            "parity token count exceeds arena batch capacity"
        );

        self.forward_batch(tokens, 0, |_s| {});
        let batch_hidden = self.arena.batch_hidden_state[..tokens.len() * emb_dim].to_vec();

        let mut max_diff = 0.0f32;
        for (pos, &tok) in tokens.iter().enumerate() {
            self.forward_step(tok, pos);
            for i in 0..emb_dim {
                let d = (self.arena.hidden_state[i] - batch_hidden[pos * emb_dim + i]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        max_diff
    }
}
