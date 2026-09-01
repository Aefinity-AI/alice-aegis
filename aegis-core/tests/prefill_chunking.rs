//! Chunked prefill is bit-exact (CLAUDE.md Rule D).
//!
//! `process_intent` runs the prompt through prefill in chunks of
//! `PREFILL_CHUNK_TOKENS` instead of one pass, so that a caller running a
//! wall-clock budget can be heard part-way through a long prompt. The whole
//! safety argument for that change is that it cannot alter the result:
//! attention reads the KV the earlier chunks wrote, every other operation in a
//! pass is per-token, and the chunk is a whole number of the kernel's GEMM
//! tiles. This test is what holds that argument up — the same prompt, run at
//! several chunk sizes, must generate the *identical* token ids.
//!
//! The model is synthetic and built in-test (same shape as
//! `tests/graph_variants.rs`) so the assertion is about the graph, not about
//! any checkpoint.

use aegis_core::inference::TernaryInferenceEngine;

const HIDDEN: usize = 32;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const INTERMEDIATE: usize = 64;
const LAYERS: usize = 2;
/// One id per single-character vocab entry, plus filler.
const VOCAB: usize = 32;
/// Long enough that a 200-token prompt survives `process_intent`'s window trim.
const MAX_SEQ: usize = 384;
/// Prompt length in characters; the vocab is single-character, so also in
/// tokens. Comfortably more than three `PREFILL_CHUNK_TOKENS` chunks.
const PROMPT_CHARS: usize = 200;

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

fn packed_ternary(rng: &mut Lcg, dim_out: usize, dim_in: usize) -> Vec<u8> {
    rng.bytes(dim_out * dim_in / 4)
}

fn f32_scale(v: f32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
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

/// The vocab entries, in id order: the 26 lowercase letters, the byte-level
/// alphabet's space (`Ġ`), then filler. Single-character entries are what make
/// `encode` produce one token per prompt character with no merges.
fn vocab_strings() -> Vec<String> {
    let mut v: Vec<String> = (b'a'..=b'z').map(|c| (c as char).to_string()).collect();
    v.push("\u{0120}".to_string()); // byte_to_unicode(0x20), i.e. space
    while v.len() < VOCAB {
        v.push(format!("<pad{}>", v.len()));
    }
    v
}

fn build_vocab() -> Vec<u8> {
    let strings = vocab_strings();
    let mut out = 0x564F_4341u32.to_le_bytes().to_vec();
    out.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    for s in &strings {
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
    t
}

fn config_json() -> String {
    format!(
        concat!(
            "{{\"num_hidden_layers\":{},\"hidden_size\":{},\"num_attention_heads\":{},",
            "\"num_key_value_heads\":{},\"intermediate_size\":{},\"vocab_size\":{},",
            "\"max_position_embeddings\":{},\"hidden_act\":\"silu\",\"rope_theta\":10000.0,",
            "\"rms_norm_eps\":1e-06,\"tie_word_embeddings\":true,\"chat_template\":\"none\"}}"
        ),
        LAYERS, HIDDEN, HEADS, KV_HEADS, INTERMEDIATE, VOCAB, MAX_SEQ
    )
}

/// A deterministic prompt of single-character vocab entries.
fn prompt() -> String {
    let letters: Vec<char> = (b'a'..=b'z').map(|c| c as char).collect();
    (0..PROMPT_CHARS)
        .map(|i| if i % 7 == 6 { ' ' } else { letters[i % 26] })
        .collect()
}

#[test]
fn chunked_prefill_generates_identical_tokens() {
    let model = build_safetensors(&build_tensors(), &config_json());
    let embed = build_embeddings();
    let vocab = build_vocab();
    let text = prompt();

    let mut engine =
        TernaryInferenceEngine::new(&embed, &model, &vocab).expect("synthetic model must load");

    // Reference: one pass over the whole prompt, the pre-chunking behaviour.
    engine.prefill_chunk_tokens = 0;
    let reference_text = engine.process_intent(&text, 8, |_| {});
    let reference_ids = engine.last_generated_ids.clone();
    assert!(
        !reference_ids.is_empty(),
        "the synthetic model generated nothing — the test would assert nothing"
    );
    assert!(
        engine.last_prefill_tokens > 3 * aegis_core::inference::PREFILL_CHUNK_TOKENS,
        "prompt must span several chunks to exercise chunking, got {} tokens",
        engine.last_prefill_tokens
    );

    for chunk in [
        aegis_core::inference::PREFILL_CHUNK_TOKENS,
        8,
        1,
        PROMPT_CHARS * 4,
    ] {
        engine.prefill_chunk_tokens = chunk;
        let got_text = engine.process_intent(&text, 8, |_| {});
        assert_eq!(
            engine.last_generated_ids, reference_ids,
            "prefill chunk size {chunk} changed the generated token ids"
        );
        assert_eq!(
            got_text, reference_text,
            "prefill chunk size {chunk} changed the decoded response"
        );
    }
}
