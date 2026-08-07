//! cis_decode — token-level cross-ISA identity artifact for the CIS-1
//! full-integer engine (`aegis_core::cis_infer`, `CisMode::FullInt`).
//!
//! This is the load-bearing demo the kernel-level digest (A25/A28) is not:
//! a COMPLETE autoregressive decode — tokenizer, embeddings, every layer,
//! integer attention, logits, greedy argmax — whose output token sequence
//! must be byte-identical on every conforming host. One FNV-1a digest over
//! the little-endian bytes of every token id (prompt ids included, so the
//! tokenizer is pinned too), printed as exactly one line:
//!
//!   CIS_DECODE digest=<16 hex> prompt_toks=<n> gen_toks=<m> mode=fullint
//!
//! An x86 host and an aarch64 host printing different digests falsifies
//! token-level cross-ISA identity; `arm-digest.yml` pins both against the
//! same constant on every push.
//!
//! Determinism choices, deliberate and part of the pinned claim:
//!   - greedy argmax only (`argmax_i64`, ties break to the lowest index);
//!   - EOS is IGNORED — exactly `max_new` tokens are always generated, so
//!     the digest never depends on a stop condition;
//!   - `CisMode::FullInt` only. Hybrid mode holds f32 state and makes no
//!     cross-ISA promise.
//!
//! Identity/correctness artifact ONLY (Rule A): this binary prints no
//! timing and never may.
//!
//! Run: cis_decode <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> [max_new] ["prompt"]

use aegis_core::cis_infer::{CisEngine, CisMode, CisModel, argmax_i64, fnv1a64};
use aegis_core::model::{FullBitNetPipeline, ModelConfig, SafeTensors};
use aegis_core::tokenizer::AegisTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: cis_decode <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> [max_new] [prompt]");
        std::process::exit(2);
    }
    let max_new: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let prompt = args
        .get(5)
        .map(String::as_str)
        .unwrap_or("Once upon a time");

    let model_bytes = std::fs::read(&args[1]).expect("read MODEL.SAF");
    let embed_bytes = std::fs::read(&args[2]).expect("read EMBED.BIN");
    let vocab_bytes = std::fs::read(&args[3]).expect("read VOCAB.BIN");

    let tensors = SafeTensors::deserialize(&model_bytes).expect("parse MODEL.SAF");
    let cfg_json = tensors
        .metadata_field("aegis_config")
        .expect("read __metadata__")
        .expect("MODEL.SAF carries no aegis_config — repack in the forge");
    let config = ModelConfig::from_json(&cfg_json).expect("parse aegis_config");

    let pipeline =
        FullBitNetPipeline::new(&tensors, &embed_bytes, &config).expect("build pipeline");
    let cis_model = CisModel::new(&pipeline, &config).expect("CIS model conversion");
    let mut engine = CisEngine::new_with_mode(&cis_model, CisMode::FullInt);

    let tokenizer = AegisTokenizer::new(&vocab_bytes).expect("parse VOCAB.BIN");
    let prompt_ids = tokenizer.encode(prompt);
    assert!(
        !prompt_ids.is_empty(),
        "prompt tokenized to nothing — digest would pin an empty decode"
    );
    assert!(
        prompt_ids.len() + max_new <= config.max_position_embeddings,
        "prompt ({}) + max_new ({}) exceeds max_position_embeddings ({})",
        prompt_ids.len(),
        max_new,
        config.max_position_embeddings
    );

    println!(
        "model : {} layers, hidden {}, vocab {}",
        config.num_hidden_layers, config.hidden_size, config.vocab_size
    );
    println!("prompt: {prompt:?} -> {} tokens", prompt_ids.len());

    // FNV-1a offset basis, matching cis_infer::fnv1a64's own convention.
    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;

    // Prefill: feed every prompt token; fold each id into the digest so the
    // tokenizer's byte->id mapping is part of the pinned claim.
    let mut pos = 0usize;
    for &t in &prompt_ids {
        digest = fnv1a64(digest, &t.to_le_bytes());
        engine.forward_step_int(t, pos);
        pos += 1;
    }

    // Greedy decode: exactly max_new tokens, EOS ignored by design.
    let mut generated = Vec::with_capacity(max_new);
    let mut current = argmax_i64(engine.decode_logits());
    for _ in 0..max_new {
        digest = fnv1a64(digest, &current.to_le_bytes());
        generated.push(current);
        engine.forward_step_int(current, pos);
        pos += 1;
        current = argmax_i64(engine.decode_logits());
    }

    println!("token ids: {generated:?}");
    println!("text     : {:?}", tokenizer.decode(&generated));
    println!(
        "CIS_DECODE digest={digest:016x} prompt_toks={} gen_toks={} mode=fullint",
        prompt_ids.len(),
        generated.len()
    );
}
