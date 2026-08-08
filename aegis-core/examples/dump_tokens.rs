//! Dev tool: print the engine's token ids for a text file, one id per line —
//! same ASCII filter as the T2d parity test, so the output diffs directly
//! against `scripts/dump_reference_fixtures.py`'s tokens.txt to localize
//! pre-tokenizer divergence.
//!
//!   cargo run --release --example dump_tokens -- \
//!       MODEL.SAF EMBED.BIN VOCAB.BIN text.txt > engine_tokens.txt

use aegis_core::inference::TernaryInferenceEngine;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: {} MODEL.SAF EMBED.BIN VOCAB.BIN text.txt", args[0]);
        std::process::exit(2);
    }
    let model = fs::read(&args[1]).expect("MODEL.SAF unreadable");
    let embed = fs::read(&args[2]).expect("EMBED.BIN unreadable");
    let vocab = fs::read(&args[3]).expect("VOCAB.BIN unreadable");
    let engine = TernaryInferenceEngine::new(&embed, &model, &vocab).expect("engine init");

    let text = fs::read_to_string(&args[4]).expect("text unreadable");
    let ascii: String = text.chars().filter(|c| c.is_ascii()).collect();
    for id in engine.tokenizer.encode(&ascii) {
        println!("{}", id);
    }
}
