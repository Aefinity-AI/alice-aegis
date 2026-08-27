//! Artifact hashing: `sha256(MODEL.SAF bytes)` / `sha256(EMBED.BIN bytes)` /
//! `sha256(VOCAB.BIN bytes)`, whole-file, no parsing — design doc §2.1 item
//! 11: "no normative parsing needed for this specific check (compare
//! against the receipt's `model`/`embed`/`vocab` fields before anything
//! else — `cis_witness.rs:195-212` fails fast here)."
//!
//! This module only computes and compares the three digests; it does not
//! parse SafeTensors, VOCAB.BIN, or any tensor format (design task 2's
//! `safetensors.rs`/`vocab.rs` work is out of scope for this phase — see
//! `docs/design/CIS_VERIFY_DESIGN.md` §3.3 for that module layout).

use crate::sha256::sha256;

/// Which of the three artifacts a hash mismatch was found in — mirrors the
/// `"MODEL" | "EMBED" | "VOCAB"` tags `cis_witness.rs:200-212` prints, and
/// the design doc's `FailedField::ArtifactHash { which }` shape (§3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    Model,
    Embed,
    Vocab,
}

/// One artifact hash mismatch: which artifact, and the two digests that
/// disagreed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactMismatch {
    pub which: ArtifactKind,
    pub receipt: [u8; 32],
    pub local: [u8; 32],
}

/// Hash `model_bytes`/`embed_bytes`/`vocab_bytes` and compare each against
/// the corresponding receipt field. Returns every mismatch found (not just
/// the first), so a caller can report all three at once — the field-order
/// doc (design §3.4) still says "fail fast, cheapest first" for the
/// overall verify pipeline, but hashing all three up front costs nothing
/// extra and gives a more useful report.
pub fn check_artifact_hashes(
    model_bytes: &[u8],
    embed_bytes: &[u8],
    vocab_bytes: &[u8],
    receipt_model_sha: &[u8; 32],
    receipt_embed_sha: &[u8; 32],
    receipt_vocab_sha: &[u8; 32],
) -> alloc::vec::Vec<ArtifactMismatch> {
    let mut mismatches = alloc::vec::Vec::new();
    let checks = [
        (ArtifactKind::Model, model_bytes, receipt_model_sha),
        (ArtifactKind::Embed, embed_bytes, receipt_embed_sha),
        (ArtifactKind::Vocab, vocab_bytes, receipt_vocab_sha),
    ];
    for (which, bytes, receipt_sha) in checks {
        let local = sha256(bytes);
        if &local != receipt_sha {
            mismatches.push(ArtifactMismatch {
                which,
                receipt: *receipt_sha,
                local,
            });
        }
    }
    mismatches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_bytes_produce_no_mismatch() {
        let model = b"pretend MODEL.SAF bytes";
        let embed = b"pretend EMBED.BIN bytes";
        let vocab = b"pretend VOCAB.BIN bytes";
        let ms = sha256(model);
        let es = sha256(embed);
        let vs = sha256(vocab);
        let out = check_artifact_hashes(model, embed, vocab, &ms, &es, &vs);
        assert!(out.is_empty());
    }

    #[test]
    fn tampered_model_bytes_are_caught() {
        let model = b"pretend MODEL.SAF bytes";
        let embed = b"pretend EMBED.BIN bytes";
        let vocab = b"pretend VOCAB.BIN bytes";
        let ms = sha256(model);
        let es = sha256(embed);
        let vs = sha256(vocab);
        let tampered_model = b"a different MODEL.SAF entirely";
        let out = check_artifact_hashes(tampered_model, embed, vocab, &ms, &es, &vs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].which, ArtifactKind::Model);
        assert_eq!(out[0].receipt, ms);
        assert_eq!(out[0].local, sha256(tampered_model));
    }

    #[test]
    fn all_three_can_mismatch_at_once() {
        let ms = sha256(b"m");
        let es = sha256(b"e");
        let vs = sha256(b"v");
        let out = check_artifact_hashes(b"not-m", b"not-e", b"not-v", &ms, &es, &vs);
        assert_eq!(out.len(), 3);
    }

    /// Cross-check against the real golden receipt's pinned artifact
    /// hashes (design doc §1.1, "Field values on that golden file"): the
    /// hex strings there are exactly `sha256` of the M7 MODEL.SAF/
    /// EMBED.BIN/VOCAB.BIN — this test just confirms `sha256()` produces
    /// hex of the right *shape* (32 bytes) that would compare correctly
    /// against those fields once real artifact bytes are hashed; it does
    /// not have the M7 binary artifacts available to hash directly (those
    /// live under `model-lab/`, not fetched here).
    #[test]
    fn digest_is_32_bytes_matching_receipt_field_width() {
        let d = sha256(b"anything");
        assert_eq!(d.len(), 32);
    }
}
