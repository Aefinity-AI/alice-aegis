//! Cross-check `artifact::check_artifact_hashes` against the real in-repo
//! M7 artifacts and the golden receipt's `model`/`embed`/`vocab` fields
//! (`docs/design/CIS_VERIFY_DESIGN.md` §5 item 1's artifacts, at
//! `model-lab/tinybit/m7_final_gate_work/artifacts/`). This is an ordinary
//! integration test (std, reads files) — it does not modify anything under
//! `tests/golden/` or `docs/hardware_logs/` (Rule C); it only reads the
//! existing M7 artifacts and the copy of the golden receipt already vendored
//! into `cis-verify/tests/fixtures/`.

use cis_verify::artifact::check_artifact_hashes;
use cis_verify::receipt::Receipt;

const GOLDEN: &str = include_str!("fixtures/witness_v1_m7_once64.receipt");

fn artifact_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../model-lab/tinybit/m7_final_gate_work/artifacts")
        .join(name)
}

#[test]
fn m7_artifacts_hash_match_the_golden_receipt() {
    let receipt = Receipt::parse(GOLDEN).expect("golden receipt parses");

    let model_path = artifact_path("MODEL.SAF");
    let embed_path = artifact_path("EMBED.BIN");
    let vocab_path = artifact_path("VOCAB.BIN");
    if !model_path.exists() || !embed_path.exists() || !vocab_path.exists() {
        eprintln!(
            "M7 artifacts not present at {model_path:?} — skipping (not fetched in this checkout)"
        );
        return;
    }

    let model_bytes = std::fs::read(&model_path).expect("read MODEL.SAF");
    let embed_bytes = std::fs::read(&embed_path).expect("read EMBED.BIN");
    let vocab_bytes = std::fs::read(&vocab_path).expect("read VOCAB.BIN");

    let mismatches = check_artifact_hashes(
        &model_bytes,
        &embed_bytes,
        &vocab_bytes,
        &receipt.model_sha,
        &receipt.embed_sha,
        &receipt.vocab_sha,
    );
    assert!(
        mismatches.is_empty(),
        "artifact hash mismatch(es) against golden receipt: {mismatches:?}"
    );
}

#[test]
fn tampering_one_byte_of_a_real_artifact_is_caught() {
    let receipt = Receipt::parse(GOLDEN).expect("golden receipt parses");
    let vocab_path = artifact_path("VOCAB.BIN");
    if !vocab_path.exists() {
        eprintln!("M7 artifacts not present — skipping");
        return;
    }
    let model_bytes = std::fs::read(artifact_path("MODEL.SAF")).unwrap();
    let embed_bytes = std::fs::read(artifact_path("EMBED.BIN")).unwrap();
    let mut vocab_bytes = std::fs::read(&vocab_path).unwrap();
    vocab_bytes[0] ^= 0x01;

    let mismatches = check_artifact_hashes(
        &model_bytes,
        &embed_bytes,
        &vocab_bytes,
        &receipt.model_sha,
        &receipt.embed_sha,
        &receipt.vocab_sha,
    );
    assert_eq!(mismatches.len(), 1);
    assert_eq!(
        mismatches[0].which,
        cis_verify::artifact::ArtifactKind::Vocab
    );
}
