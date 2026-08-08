//! T2c: the generalized graph — SwiGLU, optional SubLN, untied LM head,
//! config-from-metadata — exercised end-to-end on a synthetic model small
//! enough to build in-test. `prefill_decode_parity()` proves the batch and
//! per-token paths received every graph delta identically; the config
//! assertions prove the baked BitNet-2B path is untouched (T2a support).

use aegis_core::inference::TernaryInferenceEngine;
use aegis_core::model::{Activation, ModelConfig};

const HIDDEN: usize = 32;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const INTERMEDIATE: usize = 64;
const LAYERS: usize = 2;
const VOCAB: usize = 16;
const MAX_SEQ: usize = 8;

fn bf16(v: f32) -> [u8; 2] {
    let b = v.to_bits().to_be_bytes();
    [b[1], b[0]] // little-endian bf16 = upper two bytes of the f32, swapped
}

/// Deterministic byte stream — tests must not depend on RNG state.
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

fn packed_ternary(rng: &mut Lcg, dim_out: usize, dim_in: usize) -> Vec<u8> {
    rng.bytes(dim_out * dim_in / 4)
}

fn f32_scale(v: f32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// Serialize (name, bytes) pairs plus an `aegis_config` metadata entry into
/// a valid safetensors buffer the engine's loader accepts.
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

fn build_vocab(n: usize) -> Vec<u8> {
    let mut out = 0x564F4341u32.to_le_bytes().to_vec();
    out.extend_from_slice(&(n as u32).to_le_bytes());
    for i in 0..n {
        let s = format!("t{}", i);
        out.extend_from_slice(&(s.len() as u16).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // no merges
    out
}

fn build_embeddings() -> Vec<u8> {
    (0..VOCAB * HIDDEN)
        .flat_map(|i| bf16((i % 61) as f32 * 0.01 - 0.3))
        .collect()
}

/// All layer tensors for one synthetic model. `with_sub_norms` toggles the
/// BitNet SubLN pair; `with_lm_head` emits an untied BF16 head.
fn build_tensors(with_sub_norms: bool, with_lm_head: bool) -> Vec<(String, Vec<u8>)> {
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
        if with_sub_norms {
            t.push((
                format!("{}.self_attn.attn_sub_norm.weight", p),
                bf16_ones(HIDDEN),
            ));
            t.push((
                format!("{}.mlp.ffn_sub_norm.weight", p),
                bf16_ones(INTERMEDIATE),
            ));
        }
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
                packed_ternary(&mut rng, dim_out, dim_in),
            ));
            t.push((format!("{}.{}.weight_scale", p, proj), f32_scale(0.05)));
        }
    }
    t.push(("model.norm.weight".to_string(), bf16_ones(HIDDEN)));
    if with_lm_head {
        t.push((
            "lm_head.weight".to_string(),
            (0..VOCAB * HIDDEN)
                .flat_map(|i| bf16((i % 37) as f32 * 0.02 - 0.35))
                .collect(),
        ));
    }
    t
}

fn config_json(hidden_act: &str, tie: bool) -> String {
    format!(
        concat!(
            "{{\"num_hidden_layers\":{},\"hidden_size\":{},\"num_attention_heads\":{},",
            "\"num_key_value_heads\":{},\"intermediate_size\":{},\"vocab_size\":{},",
            "\"max_position_embeddings\":{},\"hidden_act\":\"{}\",\"rope_theta\":10000.0,",
            "\"rms_norm_eps\":1e-06,\"tie_word_embeddings\":{}}}"
        ),
        LAYERS, HIDDEN, HEADS, KV_HEADS, INTERMEDIATE, VOCAB, MAX_SEQ, hidden_act, tie
    )
}

#[test]
fn untied_silu_no_subln_model_runs_with_prefill_decode_parity() {
    let model = build_safetensors(&build_tensors(false, true), &config_json("silu", false));
    let embed = build_embeddings();
    let vocab = build_vocab(VOCAB);

    let mut engine = TernaryInferenceEngine::new(&embed, &model, &vocab)
        .expect("synthetic Falcon-style model must load");

    // The metadata config, not the baked BitNet config, must be in charge.
    assert_eq!(engine.config.hidden_act, Activation::Silu);
    assert_eq!(engine.config.rope_theta, 10000.0);
    assert_eq!(engine.config.rms_norm_eps, 1e-6);
    assert!(!engine.config.tie_word_embeddings);
    assert_eq!(engine.config.vocab_size, VOCAB);

    let tokens: Vec<u32> = (0..MAX_SEQ as u32).collect();
    let diff = engine.prefill_decode_parity(&tokens);
    assert_eq!(
        diff, 0.0,
        "batch and per-token paths diverged on the new graph: {}",
        diff
    );
}

#[test]
fn tied_relu2_subln_model_still_runs_the_bitnet_graph() {
    let model = build_safetensors(&build_tensors(true, false), &config_json("relu2", true));
    let embed = build_embeddings();
    let vocab = build_vocab(VOCAB);

    let mut engine = TernaryInferenceEngine::new(&embed, &model, &vocab)
        .expect("synthetic BitNet-style model must load");
    assert_eq!(engine.config.hidden_act, Activation::Relu2);
    assert!(engine.config.tie_word_embeddings);

    let tokens: Vec<u32> = (0..MAX_SEQ as u32).collect();
    let diff = engine.prefill_decode_parity(&tokens);
    assert_eq!(diff, 0.0, "BitNet-shaped graph lost parity: {}", diff);
}

#[test]
fn untied_config_without_head_tensor_fails_loudly() {
    let model = build_safetensors(&build_tensors(false, false), &config_json("silu", false));
    let embed = build_embeddings();
    let vocab = build_vocab(VOCAB);
    let err = match TernaryInferenceEngine::new(&embed, &model, &vocab) {
        Err(e) => e,
        Ok(_) => panic!("load unexpectedly succeeded"),
    };
    assert!(err.contains("lm_head.weight"), "unexpected error: {}", err);
}

#[test]
fn bias_tensors_are_refused() {
    let mut tensors = build_tensors(true, false);
    tensors.push((
        "model.layers.0.self_attn.q_proj.bias".to_string(),
        vec![0u8; HIDDEN * 4],
    ));
    let model = build_safetensors(&tensors, &config_json("relu2", true));
    let embed = build_embeddings();
    let vocab = build_vocab(VOCAB);
    let err = match TernaryInferenceEngine::new(&embed, &model, &vocab) {
        Err(e) => e,
        Ok(_) => panic!("load unexpectedly succeeded"),
    };
    assert!(err.contains("bias"), "unexpected error: {}", err);
}

#[test]
fn unknown_activation_is_an_error_not_a_fallback() {
    let cfg = config_json("gelu", true);
    let err = ModelConfig::from_json(&cfg).unwrap_err();
    assert!(err.contains("hidden_act"), "unexpected error: {}", err);
}

/// Single-character vocabulary ('a'..'p') so `encode` maps every prompt char
/// to a real token — the t0..t15 vocab drops everything (no single-char
/// entries, no unk token), which would make a generation test vacuous.
fn build_char_vocab(n: usize) -> Vec<u8> {
    let mut out = 0x564F4341u32.to_le_bytes().to_vec();
    out.extend_from_slice(&(n as u32).to_le_bytes());
    for i in 0..n {
        let s = ((b'a' + i as u8) as char).to_string();
        out.extend_from_slice(&(s.len() as u16).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // no merges
    out
}

/// Regression: generation asked for more tokens than the KV/RoPE window has
/// positions must stop at the window, not index past it (an index panic in
/// userspace, a memory fault on bare metal — hit for real at 2,027 generated
/// tokens in the 2026-07-14 energy run). MAX_SEQ here is 8, so demanding 64
/// new tokens walks straight into the old prompt+max_new_tokens bound.
#[test]
fn generation_stops_at_the_context_window_instead_of_panicking() {
    let cfg = format!(
        concat!(
            "{{\"num_hidden_layers\":{},\"hidden_size\":{},\"num_attention_heads\":{},",
            "\"num_key_value_heads\":{},\"intermediate_size\":{},\"vocab_size\":{},",
            "\"max_position_embeddings\":{},\"hidden_act\":\"silu\",\"rope_theta\":10000.0,",
            "\"rms_norm_eps\":1e-06,\"tie_word_embeddings\":false,\"chat_template\":\"none\"}}"
        ),
        LAYERS, HIDDEN, HEADS, KV_HEADS, INTERMEDIATE, VOCAB, MAX_SEQ
    );
    let model = build_safetensors(&build_tensors(false, true), &cfg);
    let embed = build_embeddings();
    let vocab = build_char_vocab(VOCAB);

    let mut engine =
        TernaryInferenceEngine::new(&embed, &model, &vocab).expect("synthetic model must load");
    assert_eq!(engine.config.max_position_embeddings, MAX_SEQ);

    // 4-token prompt in an 8-position window, 64 tokens demanded. The old
    // bound ran `step` to 68 and faulted at position 8; the fixed loop must
    // return cleanly, having generated at most window - prompt tokens.
    let reply = engine.process_intent("abcd", 64, |_| {});
    assert_ne!(
        reply, "No valid tokens.",
        "prompt failed to tokenize — test is vacuous"
    );
}

#[test]
fn baked_bitnet_config_parses_to_the_historic_constants() {
    // T2a anchor: the compiled-in config must keep producing exactly the
    // values that were hardcoded before this change-set.
    let baked = include_str!("../../aegis-forge/aegis_pruned_config.json");
    let cfg = ModelConfig::from_json(baked).unwrap();
    assert_eq!(cfg.hidden_act, Activation::Relu2);
    assert_eq!(cfg.rope_theta, 500000.0);
    assert_eq!(cfg.rms_norm_eps, 1e-5);
    assert!(cfg.tie_word_embeddings);
}

#[test]
fn sparse_config_defaults_to_bitnet_conventions() {
    // Configs forged before the graph-variant fields existed carry none of
    // them; they must keep meaning "BitNet-2B".
    let cfg = ModelConfig::from_json(
        r#"{"num_hidden_layers":1,"hidden_size":8,"num_attention_heads":2,
            "num_key_value_heads":1,"intermediate_size":16,"vocab_size":8,
            "max_position_embeddings":4}"#,
    )
    .unwrap();
    assert_eq!(cfg.hidden_act, Activation::Relu2);
    assert_eq!(cfg.rope_theta, 500000.0);
    assert_eq!(cfg.rms_norm_eps, 1e-5);
    assert!(cfg.tie_word_embeddings);
}
