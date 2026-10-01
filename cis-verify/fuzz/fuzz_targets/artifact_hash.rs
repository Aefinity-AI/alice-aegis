//! E19 fuzz target: `cis_verify::artifact::check_artifact_hashes` against
//! arbitrary artifact bytes and receipt-side digests. Whole-file SHA-256 +
//! comparison, no parsing (src/artifact.rs) — the property under test is
//! simply "never panics regardless of input length or content", including
//! zero-length artifacts and digests that happen to collide byte-for-byte
//! with a prefix of the artifact bytes.

#![no_main]

use cis_verify::artifact::check_artifact_hashes;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 96 {
        return;
    }
    let mut model_sha = [0u8; 32];
    let mut embed_sha = [0u8; 32];
    let mut vocab_sha = [0u8; 32];
    model_sha.copy_from_slice(&data[0..32]);
    embed_sha.copy_from_slice(&data[32..64]);
    vocab_sha.copy_from_slice(&data[64..96]);

    // Split whatever's left three ways (uneven splits included) to cover
    // empty / tiny / large artifact byte slices.
    let rest = &data[96..];
    let third = rest.len() / 3;
    let (model_bytes, tail) = rest.split_at(third);
    let (embed_bytes, vocab_bytes) = tail.split_at(third.min(tail.len()));

    let _ = check_artifact_hashes(
        model_bytes,
        embed_bytes,
        vocab_bytes,
        &model_sha,
        &embed_sha,
        &vocab_sha,
    );
});
