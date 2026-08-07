//! T3b cross-language check: artifacts produced by aegis-forge/repack_ternary.py
//! must load and run in this engine. Env-gated because it needs a forged
//! artifact directory (synthetic or real):
//!
//!   AEGIS_FORGE_DIR=/path/with/MODEL.SAF+EMBED.BIN+VOCAB.BIN \
//!   cargo test --test forge_artifacts -- --ignored

use aegis_core::inference::TernaryInferenceEngine;
use std::fs;
use std::path::Path;

#[test]
#[ignore = "needs a forged artifact directory (AEGIS_FORGE_DIR)"]
fn forged_artifacts_load_and_hold_parity() {
    let dir = std::env::var("AEGIS_FORGE_DIR").expect("set AEGIS_FORGE_DIR");
    let dir = Path::new(&dir);
    let model = fs::read(dir.join("MODEL.SAF")).expect("MODEL.SAF unreadable");
    let embed = fs::read(dir.join("EMBED.BIN")).expect("EMBED.BIN unreadable");
    let vocab = fs::read(dir.join("VOCAB.BIN")).expect("VOCAB.BIN unreadable");

    let mut engine =
        TernaryInferenceEngine::new(&embed, &model, &vocab).expect("forged artifacts must load");
    assert_eq!(
        embed.len(),
        engine.config.vocab_size * engine.config.hidden_size * 2,
        "EMBED.BIN size disagrees with the metadata config"
    );

    let n = engine.config.max_position_embeddings.min(8) as u32;
    let tokens: Vec<u32> = (0..n).collect();
    let diff = engine.prefill_decode_parity(&tokens);
    assert_eq!(
        diff, 0.0,
        "forged model lost prefill/decode parity: {}",
        diff
    );
}
