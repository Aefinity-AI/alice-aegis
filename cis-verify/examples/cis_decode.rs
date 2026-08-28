//! `cis_decode` — this crate's independent reproduction of the Tier 3
//! token-level digest (`docs/CIS-1_SPEC_v1.0.md` §8), built from
//! `cis-verify`'s own `forward`/`vocab`/`safetensors`/`config` modules with
//! **zero** dependency on `aegis-core` — the acceptance check for builder
//! task 5 (`docs/design/CIS_VERIFY_DESIGN.md` §6.2 item 5).
//!
//! Mirrors `aegis-linux/examples/cis_decode.rs`'s output format exactly so
//! the two independent implementations' output is directly diffable:
//!
//!   CIS_DECODE digest=<16 hex> prompt_toks=<n> gen_toks=<m> mode=fullint
//!
//! Run: `cargo run --example cis_decode --features std -- <MODEL.SAF>
//! <EMBED.BIN> <VOCAB.BIN> [max_new] [prompt]`

use cis_verify::config::ModelConfig;
use cis_verify::forward::{CisModel, run_decode};
use cis_verify::safetensors::SafeTensors;
use cis_verify::vocab::Tokenizer;

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
        .expect("MODEL.SAF carries no aegis_config");
    let config = ModelConfig::from_json(&cfg_json).expect("parse aegis_config");
    let model = CisModel::new(&tensors, &embed_bytes, &config).expect("build CIS model");
    let tokenizer = Tokenizer::new(&vocab_bytes).expect("parse VOCAB.BIN");

    let report = run_decode(&model, &tokenizer, prompt, max_new, None);
    assert!(
        !report.prompt_ids.is_empty(),
        "prompt tokenized to nothing — digest would pin an empty decode"
    );
    assert!(
        report.prompt_ids.len() + max_new <= config.max_position_embeddings,
        "prompt ({}) + max_new ({}) exceeds max_position_embeddings ({})",
        report.prompt_ids.len(),
        max_new,
        config.max_position_embeddings
    );

    println!(
        "model : {} layers, hidden {}, vocab {}",
        config.num_hidden_layers, config.hidden_size, config.vocab_size
    );
    println!("prompt: {prompt:?} -> {} tokens", report.prompt_ids.len());
    println!("token ids: {:?}", report.generated_ids);
    println!("text     : {:?}", tokenizer.decode(&report.generated_ids));
    println!(
        "CIS_DECODE digest={:016x} prompt_toks={} gen_toks={} mode=fullint",
        report.fnv_digest,
        report.prompt_ids.len(),
        report.generated_ids.len()
    );
}
