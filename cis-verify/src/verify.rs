//! Verify orchestration (builder task 6, `docs/design/CIS_VERIFY_DESIGN.md`
//! §3.4/§6.2 item 6): load 3 artifacts → hash check → parse receipt → replay
//! → compare fields → [`VerifyOutcome`]. This is the "no engine dependency
//! beyond the spec ops... someone else's machine verifies without us" claim
//! made checkable end to end — everything below composes only this crate's
//! own `artifact`/`receipt`/`safetensors`/`config`/`vocab`/`forward`/
//! `witness` modules.
//!
//! Field order (design doc §3.4, "fail fast, cheapest first" — the same
//! order `cis_witness.rs:195-238` already uses): artifact hashes → prompt
//! tokenization → full replay → token-id sequence → `cis-digest` → chain.

use alloc::format;
use alloc::string::String;

use crate::artifact::check_artifact_hashes;
use crate::config::ModelConfig;
use crate::forward::{CisModel, run_decode};
use crate::receipt::{Receipt, ReceiptParseError, cis_digest_of};
use crate::safetensors::SafeTensors;
use crate::vocab::Tokenizer;
use crate::witness::{WitnessChain, WitnessHeader};

/// Which field a verification failed on — the design doc's illustrative
/// shape (§3.4), plus two variants for failure modes that shape doesn't
/// name but a verifier facing untrusted/malformed artifacts must still
/// report without panicking: `ArtifactLoad` (MODEL.SAF/VOCAB.BIN bytes that
/// hash-match the receipt but fail to parse as the declared container
/// format) and `PromptUtf8` (the receipt's `prompt-hex` decodes to bytes
/// that are not valid UTF-8, so it cannot be tokenized at all).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailedField {
    ArtifactHash { which: &'static str },
    ArtifactLoad,
    PromptUtf8,
    PromptTokenize,
    TokenId { step: usize },
    CisDigest,
    Chain,
    ReceiptParse { line: usize },
}

impl FailedField {
    /// The name printed in `VERIFY FAIL (<field>)` — grep-stable, one word
    /// or hyphenated-word per field.
    pub fn name(&self) -> &'static str {
        match self {
            FailedField::ArtifactHash { which } => which,
            FailedField::ArtifactLoad => "artifact-load",
            FailedField::PromptUtf8 => "prompt-utf8",
            FailedField::PromptTokenize => "prompt-tokenize",
            FailedField::TokenId { .. } => "token-id",
            FailedField::CisDigest => "cis-digest",
            FailedField::Chain => "chain",
            FailedField::ReceiptParse { .. } => "receipt-parse",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyOutcome {
    Pass { steps: u64 },
    Fail { field: FailedField, detail: String },
}

impl VerifyOutcome {
    pub fn is_pass(&self) -> bool {
        matches!(self, VerifyOutcome::Pass { .. })
    }
}

fn fail(field: FailedField, detail: String) -> VerifyOutcome {
    VerifyOutcome::Fail { field, detail }
}

/// Verify one receipt against the three artifacts it claims. Never panics
/// on malformed/tampered input — every failure mode returns
/// `VerifyOutcome::Fail` naming the field.
pub fn verify(
    receipt_text: &str,
    model_bytes: &[u8],
    embed_bytes: &[u8],
    vocab_bytes: &[u8],
) -> VerifyOutcome {
    // 1. Parse the receipt.
    let receipt = match Receipt::parse(receipt_text) {
        Ok(r) => r,
        Err(ReceiptParseError::BadMagic) => {
            return fail(
                FailedField::ReceiptParse { line: 1 },
                String::from("bad magic line"),
            );
        }
        Err(ReceiptParseError::MissingField(f)) => {
            return fail(
                FailedField::ReceiptParse { line: 0 },
                format!("missing field '{f}'"),
            );
        }
        Err(ReceiptParseError::BadField { line, field }) => {
            return fail(
                FailedField::ReceiptParse { line },
                format!("bad field '{field}' at line {line}"),
            );
        }
    };

    // 2. Artifact hashes — cheapest check, and the one that lets a verifier
    // refuse a substituted model/embed/vocab file before spending a full
    // decode replay on it.
    let mismatches = check_artifact_hashes(
        model_bytes,
        embed_bytes,
        vocab_bytes,
        &receipt.model_sha,
        &receipt.embed_sha,
        &receipt.vocab_sha,
    );
    if let Some(m) = mismatches.first() {
        let which = match m.which {
            crate::artifact::ArtifactKind::Model => "model",
            crate::artifact::ArtifactKind::Embed => "embed",
            crate::artifact::ArtifactKind::Vocab => "vocab",
        };
        return fail(
            FailedField::ArtifactHash { which },
            format!(
                "receipt {} local {}",
                crate::hex::hex_lower_string(&m.receipt),
                crate::hex::hex_lower_string(&m.local)
            ),
        );
    }

    // 3. Load the artifacts as their declared containers.
    let tensors = match SafeTensors::deserialize(model_bytes) {
        Ok(t) => t,
        Err(e) => return fail(FailedField::ArtifactLoad, format!("MODEL.SAF: {e}")),
    };
    let cfg_json = match tensors.metadata_field("aegis_config") {
        Ok(Some(s)) => s,
        Ok(None) => {
            return fail(
                FailedField::ArtifactLoad,
                String::from("MODEL.SAF carries no aegis_config metadata"),
            );
        }
        Err(e) => return fail(FailedField::ArtifactLoad, format!("aegis_config: {e}")),
    };
    let config = match ModelConfig::from_json(&cfg_json) {
        Ok(c) => c,
        Err(e) => {
            return fail(
                FailedField::ArtifactLoad,
                format!("aegis_config parse: {e}"),
            );
        }
    };
    let model = match CisModel::new(&tensors, embed_bytes, &config) {
        Ok(m) => m,
        Err(e) => return fail(FailedField::ArtifactLoad, format!("CIS model build: {e}")),
    };
    let tokenizer = match Tokenizer::new(vocab_bytes) {
        Ok(t) => t,
        Err(e) => return fail(FailedField::ArtifactLoad, format!("VOCAB.BIN: {e}")),
    };

    // 4. Prompt must be valid UTF-8 to tokenize at all.
    let prompt = match core::str::from_utf8(&receipt.prompt) {
        Ok(s) => s,
        Err(_) => {
            return fail(
                FailedField::PromptUtf8,
                String::from("prompt-hex is not UTF-8"),
            );
        }
    };

    // 5. Cheap tokenize-length pre-check before spending a full decode.
    let prompt_ids_probe = tokenizer.encode(prompt);
    if prompt_ids_probe.len() as u64 != receipt.prompt_toks {
        return fail(
            FailedField::PromptTokenize,
            format!(
                "encode(prompt) produced {} ids, receipt claims prompt-toks {}",
                prompt_ids_probe.len(),
                receipt.prompt_toks
            ),
        );
    }

    if receipt.gen_toks as usize > config.max_position_embeddings
        || prompt_ids_probe.len() + receipt.gen_toks as usize > config.max_position_embeddings
    {
        return fail(
            FailedField::ArtifactLoad,
            String::from("prompt + maxtok exceeds max_position_embeddings"),
        );
    }

    // 6. Full replay, folding the witness chain as we go (design doc §1.2:
    // the per-step logit vectors are never stored in the receipt — a
    // verifier must recompute every one of them).
    let header = WitnessHeader {
        model_sha: &receipt.model_sha,
        embed_sha: &receipt.embed_sha,
        vocab_sha: &receipt.vocab_sha,
        max_new: receipt.maxtok,
        prompt: &receipt.prompt,
    };
    let mut chain = WitnessChain::from_header(&header);
    let report = run_decode(
        &model,
        &tokenizer,
        prompt,
        receipt.maxtok as usize,
        Some(&mut chain),
    );

    // 7. Token-id sequence, first divergence named.
    if let Some(step) = first_divergence(&report.generated_ids, &receipt.token_ids) {
        return fail(
            FailedField::TokenId { step },
            format!(
                "local[{step}]={} receipt[{step}]={}",
                report.generated_ids.get(step).copied().unwrap_or(u32::MAX),
                receipt.token_ids.get(step).copied().unwrap_or(u32::MAX)
            ),
        );
    }

    // 8. cis-digest (FNV-1a 64 over prompt ids then generated ids).
    let local_digest = cis_digest_of(&report.prompt_ids, &report.generated_ids);
    if local_digest != receipt.cis_digest {
        return fail(
            FailedField::CisDigest,
            format!(
                "local {local_digest:016x} receipt {:016x}",
                receipt.cis_digest
            ),
        );
    }

    // 9. Witness chain — the full logit-vector claim.
    let local_chain = chain.digest();
    if local_chain != receipt.chain {
        return fail(
            FailedField::Chain,
            format!(
                "local {} receipt {}",
                crate::hex::hex_lower_string(&local_chain),
                crate::hex::hex_lower_string(&receipt.chain)
            ),
        );
    }

    VerifyOutcome::Pass {
        steps: report.generated_ids.len() as u64,
    }
}

fn first_divergence(local: &[u32], receipt: &[u32]) -> Option<usize> {
    if local.len() != receipt.len() {
        return Some(local.len().min(receipt.len()));
    }
    local.iter().zip(receipt).position(|(a, b)| a != b)
}

// Integration tests that load the real M7 artifacts and the golden receipt
// (Gate B, negative-field tests 4(a)-(e)/5) live in `tests/verify_golden.rs`
// — an ordinary `std` integration test crate, not this no_std+alloc module,
// since they need `std::fs` file I/O that `#![no_std]` (the default build,
// no `std` feature) cannot provide.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_names_are_stable() {
        assert_eq!(FailedField::ArtifactHash { which: "model" }.name(), "model");
        assert_eq!(FailedField::CisDigest.name(), "cis-digest");
        assert_eq!(FailedField::Chain.name(), "chain");
        assert_eq!(FailedField::TokenId { step: 3 }.name(), "token-id");
        assert_eq!(
            FailedField::ReceiptParse { line: 2 }.name(),
            "receipt-parse"
        );
        assert_eq!(FailedField::ArtifactLoad.name(), "artifact-load");
        assert_eq!(FailedField::PromptUtf8.name(), "prompt-utf8");
        assert_eq!(FailedField::PromptTokenize.name(), "prompt-tokenize");
    }

    #[test]
    fn garbage_receipt_text_fails_as_receipt_parse() {
        let outcome = verify("not a receipt at all", b"m", b"e", b"v");
        match outcome {
            VerifyOutcome::Fail { field, .. } => {
                assert!(
                    matches!(field, FailedField::ReceiptParse { .. }),
                    "{field:?}"
                );
            }
            other => panic!("expected ReceiptParse failure, got {other:?}"),
        }
    }
}
