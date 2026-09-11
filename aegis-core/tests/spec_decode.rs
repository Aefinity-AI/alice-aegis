//! Leg E-A: prompt-lookup speculative decoding must be a pure ACCELERATION,
//! never a change of output. The contract under test is byte-identity of the
//! emitted token stream against `greedy_decode` (batched prefill + one
//! `forward_step` per token) for every draft length K, on a synthetic model
//! small enough to build in-test.
//!
//! Bit-identity is achievable — and therefore assertable with `assert_eq!`
//! rather than a tolerance — because `tests/gemm_equivalence.rs` already
//! pins `ternary_matmul` (the batched verify path) to `ternary_matvec` (the
//! sequential path) bit for bit, and both decode paths run the same rmsnorm,
//! RoPE, attention and LM-head ops in the same order.

use aegis_core::inference::{SpecDecodeConfig, TernaryInferenceEngine, prompt_lookup_draft};

const HIDDEN: usize = 32;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const INTERMEDIATE: usize = 64;
const LAYERS: usize = 2;
const VOCAB: usize = 32;
const MAX_SEQ: usize = 64;

fn bf16(v: f32) -> [u8; 2] {
    let b = v.to_bits().to_be_bytes();
    [b[1], b[0]]
}

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 24) as u8
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn bf16_ones(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| bf16(1.0)).collect()
}

fn build_safetensors(tensors: &[(String, Vec<u8>)], config_json: &str) -> Vec<u8> {
    let escaped = config_json.replace('\\', "\\\\").replace('"', "\\\"");
    let mut header = format!("{{\"__metadata__\":{{\"aegis_config\":\"{}\"}}", escaped);
    let mut offset = 0usize;
    for (name, bytes) in tensors {
        header.push_str(&format!(
            ",\"{}\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name,
            bytes.len(),
            offset,
            offset + bytes.len()
        ));
        offset += bytes.len();
    }
    header.push('}');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    for (_, bytes) in tensors {
        out.extend_from_slice(bytes);
    }
    out
}

/// The two highest ids are the turn terminators, so id 0 is an ordinary
/// token: `stop_ids()` falls back to 0 when a vocabulary has no
/// `<|end_of_text|>`, which would truncate every stream at the first zero
/// argmax and hide the behaviour under test.
fn build_vocab(n: usize) -> Vec<u8> {
    let mut out = 0x564F4341u32.to_le_bytes().to_vec();
    out.extend_from_slice(&(n as u32).to_le_bytes());
    for i in 0..n {
        let s = if i == n - 1 {
            "<|end_of_text|>".to_string()
        } else if i == n - 2 {
            "<|eot_id|>".to_string()
        } else {
            format!("t{}", i)
        };
        out.extend_from_slice(&(s.len() as u16).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

fn build_embeddings() -> Vec<u8> {
    (0..VOCAB * HIDDEN)
        .flat_map(|i| bf16((i % 61) as f32 * 0.01 - 0.3))
        .collect()
}

fn build_tensors() -> Vec<(String, Vec<u8>)> {
    let mut rng = Lcg(0x1234_5678);
    let kv_dim = KV_HEADS * (HIDDEN / HEADS);
    let mut t: Vec<(String, Vec<u8>)> = Vec::new();
    for i in 0..LAYERS {
        let p = format!("model.layers.{}", i);
        t.push((format!("{}.input_layernorm.weight", p), bf16_ones(HIDDEN)));
        t.push((
            format!("{}.post_attention_layernorm.weight", p),
            bf16_ones(HIDDEN),
        ));
        t.push((
            format!("{}.self_attn.attn_sub_norm.weight", p),
            bf16_ones(HIDDEN),
        ));
        t.push((
            format!("{}.mlp.ffn_sub_norm.weight", p),
            bf16_ones(INTERMEDIATE),
        ));
        for (proj, dim_out, dim_in) in [
            ("self_attn.q_proj", HIDDEN, HIDDEN),
            ("self_attn.k_proj", kv_dim, HIDDEN),
            ("self_attn.v_proj", kv_dim, HIDDEN),
            ("self_attn.o_proj", HIDDEN, HIDDEN),
            ("mlp.gate_proj", INTERMEDIATE, HIDDEN),
            ("mlp.up_proj", INTERMEDIATE, HIDDEN),
            ("mlp.down_proj", HIDDEN, INTERMEDIATE),
        ] {
            t.push((
                format!("{}.{}.weight", p, proj),
                rng.bytes(dim_out * dim_in / 4),
            ));
            t.push((
                format!("{}.{}.weight_scale", p, proj),
                0.05f32.to_le_bytes().to_vec(),
            ));
        }
    }
    t.push(("model.norm.weight".to_string(), bf16_ones(HIDDEN)));
    t
}

fn config_json() -> String {
    format!(
        concat!(
            "{{\"num_hidden_layers\":{},\"hidden_size\":{},\"num_attention_heads\":{},",
            "\"num_key_value_heads\":{},\"intermediate_size\":{},\"vocab_size\":{},",
            "\"max_position_embeddings\":{},\"hidden_act\":\"relu2\",\"rope_theta\":10000.0,",
            "\"rms_norm_eps\":1e-06,\"tie_word_embeddings\":true}}"
        ),
        LAYERS, HIDDEN, HEADS, KV_HEADS, INTERMEDIATE, VOCAB, MAX_SEQ
    )
}

fn engine() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        build_embeddings(),
        build_safetensors(&build_tensors(), &config_json()),
        build_vocab(VOCAB),
    )
}

/// Fixed prompt set: a couple of plain ones, plus deliberately self-repetitive
/// ones so the lookup drafter has something to find.
fn prompts() -> Vec<Vec<u32>> {
    vec![
        vec![1, 2, 3, 4],
        vec![7, 11, 3],
        vec![5],
        vec![1, 2, 3, 1, 2, 3, 1, 2, 3],
        vec![9, 8, 9, 8, 9, 8, 9, 8],
        vec![4, 4, 4, 4, 4, 4],
        vec![2, 13, 6, 2, 13, 6, 2, 13, 6, 2, 13, 6],
    ]
}

#[test]
fn speculative_stream_is_byte_identical_to_greedy_for_every_k() {
    let (embed, model, vocab) = engine();
    let mut eng = TernaryInferenceEngine::new(&embed, &model, &vocab).expect("synthetic model");

    let max_new = 24;
    for prompt in prompts() {
        let reference = eng.greedy_decode(&prompt, max_new);
        for k in [0usize, 1, 2, 4, 8] {
            let cfg = SpecDecodeConfig {
                k,
                ngram_max: 3,
                ngram_min: 1,
            };
            let (spec, stats) = eng.speculative_decode(&prompt, max_new, cfg);
            assert_eq!(
                spec, reference,
                "K={} diverged from sequential greedy on prompt {:?}",
                k, prompt
            );
            // Every pass yields row 0's token plus the drafts it confirmed.
            assert_eq!(stats.committed, stats.passes + stats.accepted);
            assert_eq!(stats.lm_head_evals, stats.committed);
            assert!(stats.accepted <= stats.drafted);
            assert!(stats.emitted <= stats.committed + 1);
            if k == 0 {
                assert_eq!(stats.drafted, 0, "K=0 must draft nothing");
                assert_eq!(stats.accepted, 0);
                assert!(
                    (stats.tokens_per_pass() - 1.0).abs() < 1e-12 || stats.passes == 0,
                    "K=0 must be exactly sequential: {:?}",
                    stats
                );
            }
        }
    }
}

#[test]
fn drafting_actually_fires_on_a_repetitive_context() {
    let (embed, model, vocab) = engine();
    let mut eng = TernaryInferenceEngine::new(&embed, &model, &vocab).expect("synthetic model");
    let (_, stats) = eng.speculative_decode(
        &[1, 2, 3, 1, 2, 3, 1, 2, 3],
        24,
        SpecDecodeConfig {
            k: 4,
            ngram_max: 3,
            ngram_min: 1,
        },
    );
    assert!(
        stats.drafted > 0,
        "lookup drafter proposed nothing on a repetitive context: {:?}",
        stats
    );
    assert!(stats.passes > 0);
}

#[test]
fn kv_rollback_leaves_the_cache_clean_for_the_next_run() {
    let (embed, model, vocab) = engine();
    let mut eng = TernaryInferenceEngine::new(&embed, &model, &vocab).expect("synthetic model");
    let prompt = vec![1, 2, 3, 1, 2, 3];

    // Reference taken on a virgin cache.
    let reference = eng.greedy_decode(&prompt, 20);
    // A speculative run writes (and rolls back) KV entries past the end of
    // its own stream; a greedy run afterwards must be unaffected.
    let _ = eng.speculative_decode(
        &prompt,
        20,
        SpecDecodeConfig {
            k: 8,
            ngram_max: 3,
            ngram_min: 1,
        },
    );
    let after = eng.greedy_decode(&prompt, 20);
    assert_eq!(
        after, reference,
        "speculative run contaminated the KV cache"
    );
}

#[test]
fn lookup_draft_takes_the_most_recent_earlier_occurrence() {
    // suffix [1,2] occurs at 0 and at 4; the drafter must continue the LATER
    // one (…1,2 -> 9) not the earlier one (…1,2 -> 3).
    let ctx = [1u32, 2, 3, 7, 1, 2, 9, 8, 1, 2];
    assert_eq!(prompt_lookup_draft(&ctx, 2, 1, 3), vec![9, 8]);
}

#[test]
fn lookup_draft_prefers_the_longest_matching_suffix() {
    let ctx = [5u32, 1, 2, 3, 4, 9, 1, 2, 3];
    // longest suffix that recurs is [1,2,3] at index 1 -> continuation [4,9]
    assert_eq!(prompt_lookup_draft(&ctx, 2, 1, 3), vec![4, 9]);
}

#[test]
fn lookup_draft_is_empty_when_nothing_recurs() {
    let ctx = [1u32, 2, 3, 4, 5];
    assert!(prompt_lookup_draft(&ctx, 4, 1, 3).is_empty());
    // ...and degenerate knobs never panic
    assert!(prompt_lookup_draft(&ctx, 0, 1, 3).is_empty());
    assert!(prompt_lookup_draft(&[], 4, 1, 3).is_empty());
    assert!(prompt_lookup_draft(&[7], 4, 1, 3).is_empty());
    assert!(prompt_lookup_draft(&ctx, 4, 0, 3).is_empty());
    assert!(prompt_lookup_draft(&ctx, 4, 4, 3).is_empty());
}

#[test]
fn lookup_draft_is_clipped_to_the_context_end_and_may_overlap_the_pattern() {
    let ctx = [1u32, 2, 3, 1, 2];
    // Suffix [1,2] recurs at index 0. The continuation runs to the end of the
    // context and is allowed to overlap the suffix itself — that overlap IS
    // the mechanism that unrolls a repeating cycle, and it is clipped at the
    // context end, never past it.
    assert_eq!(prompt_lookup_draft(&ctx, 8, 1, 3), vec![3, 1, 2]);
    // A shorter draft budget clips the same continuation.
    assert_eq!(prompt_lookup_draft(&ctx, 2, 1, 3), vec![3, 1]);
}

#[test]
fn streams_agree_at_the_context_window_edge() {
    // A prompt that leaves only a handful of positions before the KV/RoPE
    // window ends: both paths must stop at the same place, and the drafter
    // must never propose a row past the window. An off-by-one in the draft
    // budget shows up here and nowhere else.
    let (embed, model, vocab) = engine();
    let mut eng = TernaryInferenceEngine::new(&embed, &model, &vocab).expect("synthetic model");
    for tail in [1usize, 2, 5, 9] {
        let prompt: Vec<u32> = (0..(MAX_SEQ - tail) as u32).map(|i| i % 20).collect();
        let reference = eng.greedy_decode(&prompt, 64);
        for k in [1usize, 2, 4, 8] {
            let (spec, stats) = eng.speculative_decode(
                &prompt,
                64,
                SpecDecodeConfig {
                    k,
                    ngram_max: 3,
                    ngram_min: 1,
                },
            );
            assert_eq!(
                spec, reference,
                "window edge (tail {}) diverged at K={}",
                tail, k
            );
            // Positions consumed is at most the window. The final token is
            // predicted from the last occupied position and emitted without
            // ever being fed, so the TOKEN count may exceed the window by
            // exactly one; anything beyond that is an off-by-one.
            assert!(
                prompt.len() + spec.len() <= MAX_SEQ + 1,
                "generation ran past the window: {} + {}",
                prompt.len(),
                spec.len()
            );
            assert_eq!(stats.committed, stats.passes + stats.accepted);
        }
    }
}

#[test]
fn an_over_long_prompt_is_truncated_the_same_way_by_both_paths() {
    let (embed, model, vocab) = engine();
    let mut eng = TernaryInferenceEngine::new(&embed, &model, &vocab).expect("synthetic model");
    let prompt: Vec<u32> = (0..(MAX_SEQ as u32 + 17)).map(|i| i % 19).collect();
    let reference = eng.greedy_decode(&prompt, 8);
    let (spec, _) = eng.speculative_decode(
        &prompt,
        8,
        SpecDecodeConfig {
            k: 4,
            ngram_max: 3,
            ngram_min: 1,
        },
    );
    assert_eq!(spec, reference, "over-long prompt handled differently");
}
