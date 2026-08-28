//! Acceptance gate B (`docs/design/CIS_VERIFY_DESIGN.md` §5 items 1, 4):
//! `cis_verify::verify::verify` against the real in-repo M7 artifacts and
//! the golden receipt, plus the required negative tests — tampered token
//! id, tampered chain hex, wrong artifact hash, truncated receipt — each
//! expected to FAIL naming the correct field. Requires the `std` feature
//! (file I/O); an ordinary `std` integration test crate, so it needs no
//! `extern crate std` gymnastics the way an in-lib `#[cfg(test)]` module
//! under `#![no_std]` would.
//!
//! Never touches `tests/golden/` (Rule C) — only the read-only copy already
//! vendored at `cis-verify/tests/fixtures/witness_v1_m7_once64.receipt` and
//! the in-repo M7 artifacts under `model-lab/`.

use cis_verify::verify::{FailedField, VerifyOutcome, verify};

const GOLDEN: &str = include_str!("fixtures/witness_v1_m7_once64.receipt");

fn artifact_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../model-lab/tinybit/m7_final_gate_work/artifacts")
        .join(name)
}

/// `None` if the M7 artifacts aren't fetched into this checkout (mirrors
/// `tests/artifact_hash_golden.rs`'s existing skip convention).
fn load_m7() -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let m = artifact_path("MODEL.SAF");
    let e = artifact_path("EMBED.BIN");
    let v = artifact_path("VOCAB.BIN");
    if !m.exists() || !e.exists() || !v.exists() {
        return None;
    }
    Some((
        std::fs::read(m).unwrap(),
        std::fs::read(e).unwrap(),
        std::fs::read(v).unwrap(),
    ))
}

#[test]
fn gate_b_golden_receipt_verifies_pass() {
    let Some((m, e, v)) = load_m7() else {
        eprintln!("M7 artifacts not present — skipping (not fetched in this checkout)");
        return;
    };
    let outcome = verify(GOLDEN, &m, &e, &v);
    assert!(
        matches!(outcome, VerifyOutcome::Pass { steps: 64 }),
        "expected VERIFY PASS with 64 steps, got {outcome:?}"
    );
}

#[test]
fn negative_tampered_token_id_fails_naming_token_id() {
    let Some((m, e, v)) = load_m7() else {
        eprintln!("M7 artifacts not present — skipping");
        return;
    };
    // "token-ids 12,...": flip the first generated id (12 -> 13) leaving
    // every other field (including cis-digest and chain) untouched.
    let tampered = GOLDEN.replacen("token-ids 12,", "token-ids 13,", 1);
    assert_ne!(
        tampered, GOLDEN,
        "replacement did not apply — fixture drifted"
    );
    match verify(&tampered, &m, &e, &v) {
        VerifyOutcome::Fail { field, .. } => {
            assert_eq!(field, FailedField::TokenId { step: 0 });
        }
        other => panic!("expected TokenId failure, got {other:?}"),
    }
}

#[test]
fn negative_tampered_chain_hex_fails_naming_chain() {
    let Some((m, e, v)) = load_m7() else {
        eprintln!("M7 artifacts not present — skipping");
        return;
    };
    let zeros = "0".repeat(64);
    let tampered = GOLDEN.replacen(
        "chain aee25b770bd7b22eea2ea8476bbd949881d58a98d6dc3085c7cc94d322b1961b",
        &format!("chain {zeros}"),
        1,
    );
    assert_ne!(
        tampered, GOLDEN,
        "replacement did not apply — fixture drifted"
    );
    match verify(&tampered, &m, &e, &v) {
        VerifyOutcome::Fail { field, .. } => assert_eq!(field, FailedField::Chain),
        other => panic!("expected Chain failure, got {other:?}"),
    }
}

#[test]
fn negative_wrong_artifact_hash_fails_naming_the_artifact() {
    let Some((m, e, v)) = load_m7() else {
        eprintln!("M7 artifacts not present — skipping");
        return;
    };
    let mut bad_vocab = v.clone();
    bad_vocab[0] ^= 0xFF;
    match verify(GOLDEN, &m, &e, &bad_vocab) {
        VerifyOutcome::Fail { field, .. } => {
            assert_eq!(field, FailedField::ArtifactHash { which: "vocab" });
        }
        other => panic!("expected ArtifactHash failure, got {other:?}"),
    }

    let mut bad_model = m.clone();
    bad_model[100] ^= 0xFF;
    match verify(GOLDEN, &bad_model, &e, &v) {
        VerifyOutcome::Fail { field, .. } => {
            assert_eq!(field, FailedField::ArtifactHash { which: "model" });
        }
        other => panic!("expected ArtifactHash failure, got {other:?}"),
    }
}

#[test]
fn negative_truncated_receipt_fails_naming_receipt_parse() {
    let Some((m, e, v)) = load_m7() else {
        eprintln!("M7 artifacts not present — skipping");
        return;
    };
    let cut = &GOLDEN[..GOLDEN.len() / 2];
    match verify(cut, &m, &e, &v) {
        VerifyOutcome::Fail { field, .. } => {
            assert!(
                matches!(field, FailedField::ReceiptParse { .. }),
                "{field:?}"
            );
        }
        other => panic!("expected ReceiptParse failure, got {other:?}"),
    }
}
