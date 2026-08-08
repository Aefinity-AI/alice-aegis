//! T2b/T2d: parity against a reference-stack dump (transformers on
//! Colab/local — see `scripts/dump_reference_fixtures.py`). These tests are
//! `#[ignore]` because they need real artifacts and fixtures; run them as
//!
//!   AEGIS_MODEL=... AEGIS_EMBED=... AEGIS_VOCAB=... \
//!   AEGIS_REF_TOKENS=fixtures/tokens.txt \
//!   AEGIS_REF_HIDDEN=fixtures/hidden.bin \
//!   cargo test --test reference_parity -- --ignored
//!
//! T2d (tokenization) gates T2b (hidden states): cross-stack numbers are
//! meaningless unless both stacks agree on the token ids first.

use aegis_core::inference::TernaryInferenceEngine;
use std::fs;

fn env(name: &str) -> String {
    std::env::var(name)
        .unwrap_or_else(|_| panic!("{} must point at a fixture (see module docs)", name))
}

fn load_artifacts() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        fs::read(env("AEGIS_EMBED")).expect("EMBED.BIN unreadable"),
        fs::read(env("AEGIS_MODEL")).expect("MODEL.SAF unreadable"),
        fs::read(env("AEGIS_VOCAB")).expect("VOCAB.BIN unreadable"),
    )
}

/// Reference token ids: one decimal id per line.
fn load_ref_tokens() -> Vec<u32> {
    fs::read_to_string(env("AEGIS_REF_TOKENS"))
        .expect("token fixture unreadable")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().expect("token fixture: non-integer line"))
        .collect()
}

#[test]
#[ignore = "needs real artifacts + reference fixtures (AEGIS_* env vars)"]
fn t2d_tokenization_matches_reference() {
    let (embed, model, vocab) = load_artifacts();
    let engine = TernaryInferenceEngine::new(&embed, &model, &vocab).expect("engine init");

    let text = fs::read_to_string(env("AEGIS_EVAL_TEXT")).expect("eval text unreadable");
    let ascii: String = text.chars().filter(|c| c.is_ascii()).collect();
    let ours = engine.tokenizer.encode(&ascii);
    let reference = load_ref_tokens();

    assert_eq!(
        ours.len(),
        reference.len(),
        "token count mismatch: engine {} vs reference {}",
        ours.len(),
        reference.len()
    );
    for (i, (a, b)) in ours.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            a, b,
            "token id diverges at position {}: engine {} vs reference {}",
            i, a, b
        );
    }
}

#[test]
#[ignore = "needs real artifacts + reference fixtures (AEGIS_* env vars)"]
fn t2b_per_layer_hidden_states_match_reference() {
    let (embed, model, vocab) = load_artifacts();
    let mut engine = TernaryInferenceEngine::new(&embed, &model, &vocab).expect("engine init");

    let tokens = load_ref_tokens();
    assert!(!tokens.is_empty(), "empty token fixture");
    let states = engine.capture_layer_hidden_states(&tokens);

    // Fixture layout: raw little-endian f32, (layers - 1) x (tokens x
    // hidden), written by scripts/dump_reference_fixtures.py. The last
    // decoder layer is absent BY DESIGN: transformers' final hidden_states
    // entry is post-final-norm while the engine captures pre-norm, so the
    // two stacks only define layers 0..N-2 identically. The last layer is
    // covered by the logit/PPL gates (G4a) instead.
    let raw = fs::read(env("AEGIS_REF_HIDDEN")).expect("hidden-state fixture unreadable");
    let per_layer = states[0].len();
    let fixture_layers = states.len() - 1;
    assert_eq!(
        raw.len(),
        fixture_layers * per_layer * 4,
        "fixture size mismatch: {} bytes vs {} layers x {} f32 (fixture must hold num_layers - 1 layers)",
        raw.len(),
        fixture_layers,
        per_layer
    );
    let states = &states[..fixture_layers];

    // The two stacks accumulate in different orders; exact equality is not
    // the claim. The claim is layer-localized agreement within a tolerance.
    let tol: f32 = std::env::var("AEGIS_PARITY_TOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-2);

    for (layer, ours) in states.iter().enumerate() {
        let mut max_diff = 0.0f32;
        for (i, &v) in ours.iter().enumerate() {
            let off = (layer * per_layer + i) * 4;
            let r = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            let d = (v - r).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        println!("layer {:2}: max |engine - reference| = {}", layer, max_diff);
        assert!(
            max_diff <= tol,
            "layer {} diverges: max diff {} > tol {} — first bad layer localizes the bug",
            layer,
            max_diff,
            tol
        );
    }
}
