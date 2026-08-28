//! The witness v1 chain construction: header hash (genesis) + per-step
//! fold. Reimplemented from `aegis-core/src/witness.rs:141-223`
//! (`WITNESS_DOMAIN_V1`, `WitnessHeader`, `WitnessChain`, `hex_lower` — the
//! last of those now lives in this crate's `hex` module instead, since it's
//! a general hex utility, not chain-specific; the cryptographic
//! construction itself — domain string, field order, endianness — is
//! reproduced byte-for-byte below).
//!
//! Per `docs/design/CIS_VERIFY_DESIGN.md` §1.2, the construction is:
//!
//! - **Header hash** (genesis value):
//!   `SHA256(WITNESS_DOMAIN_V1 ‖ model_sha ‖ embed_sha ‖ vocab_sha ‖
//!   BE64(max_new) ‖ BE64(len(prompt)) ‖ prompt)`
//! - **Per-step fold**:
//!   `chain' = SHA256(chain ‖ b"STEP" ‖ BE64(step_index) ‖ LE32(token_id) ‖
//!   BE64(len(logits)) ‖ LE64(logits[0]) ‖ … ‖ LE64(logits[n-1]))`
//!
//! The mixed endianness is load-bearing and reproduced exactly: lengths and
//! the step index are big-endian; `token_id` and each `i64` logit are
//! little-endian. Getting this wrong reproduces nothing — a verifier that
//! "picks one consistent endianness" fails every chain silently.

use crate::sha256::Sha256;

/// Domain-separation prefix for v1 headers. Identical byte string to
/// `aegis-core/src/witness.rs:143`. A version bump to the receipt format
/// means changing this literal, which changes every downstream digest by
/// design — so this constant is deliberately not derived from anything.
pub const WITNESS_DOMAIN_V1: &[u8] = b"AEGIS-WITNESS v1-CIS\n";

/// The inputs a witness commits to, hashed into the chain's genesis value.
/// Field-for-field identical to `aegis-core/src/witness.rs:146-152`.
pub struct WitnessHeader<'a> {
    pub model_sha: &'a [u8; 32],
    pub embed_sha: &'a [u8; 32],
    pub vocab_sha: &'a [u8; 32],
    pub max_new: u64,
    pub prompt: &'a [u8],
}

impl WitnessHeader<'_> {
    /// `SHA256(domain ‖ model_sha ‖ embed_sha ‖ vocab_sha ‖ BE64(max_new) ‖
    /// BE64(len(prompt)) ‖ prompt)`. The variable-width `prompt` field gets
    /// an explicit length prefix so no two distinct headers can collide by
    /// field-boundary ambiguity; the other fields are fixed-width and need
    /// none. Identical construction to `aegis-core/src/witness.rs:158-168`.
    pub fn hash(&self) -> [u8; 32] {
        let mut s = Sha256::new();
        s.update(WITNESS_DOMAIN_V1);
        s.update(self.model_sha);
        s.update(self.embed_sha);
        s.update(self.vocab_sha);
        s.update(&self.max_new.to_be_bytes());
        s.update(&(self.prompt.len() as u64).to_be_bytes());
        s.update(self.prompt);
        s.finalize()
    }
}

/// The chain itself: genesis = header hash; each decode step absorbs the
/// chosen token id and the complete i64 logit vector that produced it.
/// Identical shape to `aegis-core/src/witness.rs:173-210`.
pub struct WitnessChain {
    chain: [u8; 32],
    steps: u64,
}

impl WitnessChain {
    pub fn from_header(header: &WitnessHeader<'_>) -> Self {
        WitnessChain {
            chain: header.hash(),
            steps: 0,
        }
    }

    /// Absorb one decode step. `logits` is the full LM-head output the
    /// argmax ran over; `token_id` is the id it selected. Streaming absorb —
    /// no allocation regardless of vocabulary size. Identical byte layout
    /// to `aegis-core/src/witness.rs:189-201`:
    /// `SHA256(chain ‖ b"STEP" ‖ BE64(step) ‖ LE32(token_id) ‖
    /// BE64(len(logits)) ‖ LE64(logits[0..]))`.
    pub fn fold_step(&mut self, token_id: u32, logits: &[i64]) {
        let mut s = Sha256::new();
        s.update(&self.chain);
        s.update(b"STEP");
        s.update(&self.steps.to_be_bytes());
        s.update(&token_id.to_le_bytes());
        s.update(&(logits.len() as u64).to_be_bytes());
        for &l in logits {
            s.update(&l.to_le_bytes());
        }
        self.chain = s.finalize();
        self.steps += 1;
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn digest(&self) -> [u8; 32] {
        self.chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::sha256;
    use alloc::vec;

    fn test_header<'a>(
        m: &'a [u8; 32],
        e: &'a [u8; 32],
        v: &'a [u8; 32],
        max_new: u64,
        prompt: &'a [u8],
    ) -> WitnessHeader<'a> {
        WitnessHeader {
            model_sha: m,
            embed_sha: e,
            vocab_sha: v,
            max_new,
            prompt,
        }
    }

    // The per-step logit vectors that mint a real receipt are NOT stored in
    // the receipt text (design doc §1.2, "critical fact for verifier
    // design"), so this crate cannot reproduce
    // tests/golden/witness_v1_m7_once64.receipt's `chain` field directly —
    // doing so needs the M7 artifacts fed through a full CIS-1 decode
    // (design task 5, out of scope for tasks 1-2). What CAN be pinned here,
    // with zero engine dependency, is that this transcription reproduces
    // the *construction* exactly: same digest for the same synthetic
    // inputs as `aegis_core::witness` itself computes, checked via a
    // throwaway harness run against the engine crate (not committed to
    // aegis-core; the expected constant below is that harness's output,
    // copied in as a golden literal like any other pinned digest).
    //
    // Synthetic sequence: 3 steps, tiny logit vectors, arbitrary but fixed
    // model/embed/vocab/prompt bytes.
    #[test]
    fn chain_matches_engine_reference_on_synthetic_sequence() {
        let model_sha = sha256(b"synthetic-model");
        let embed_sha = sha256(b"synthetic-embed");
        let vocab_sha = sha256(b"synthetic-vocab");
        let prompt = b"hi";
        let header = test_header(&model_sha, &embed_sha, &vocab_sha, 3, prompt);

        let mut chain = WitnessChain::from_header(&header);
        chain.fold_step(7, &[10, -20, 30]);
        chain.fold_step(42, &[1, 2, 3, 4]);
        chain.fold_step(0, &[-1]);

        assert_eq!(chain.steps(), 3);
        // Computed by aegis-core's own `witness::{WitnessHeader,
        // WitnessChain}` against these exact inputs, via a throwaway test
        // harness (`aegis-core/tests/scratch_cisverify_harness.rs`, added,
        // run once for this constant, then deleted — never committed to
        // aegis-core). Pinned here as an ordinary golden constant, same
        // discipline as the FIPS and FNV vectors above it in this crate.
        // Hex: ac97cd828323bf18d881d604672377e6d2ef4ea9a047ecde108bb1c1d62e3738
        let expected = [
            0xac, 0x97, 0xcd, 0x82, 0x83, 0x23, 0xbf, 0x18, 0xd8, 0x81, 0xd6, 0x04, 0x67, 0x23,
            0x77, 0xe6, 0xd2, 0xef, 0x4e, 0xa9, 0xa0, 0x47, 0xec, 0xde, 0x10, 0x8b, 0xb1, 0xc1,
            0xd6, 0x2e, 0x37, 0x38,
        ];
        assert_eq!(
            chain.digest(),
            expected,
            "chain construction diverged from aegis_core::witness on the synthetic sequence"
        );
    }

    #[test]
    fn header_binds_every_field() {
        let m = sha256(b"m");
        let e = sha256(b"e");
        let v = sha256(b"v");
        let base = test_header(&m, &e, &v, 5, b"prompt");
        let h0 = base.hash();

        let m2 = sha256(b"m2");
        assert_ne!(test_header(&m2, &e, &v, 5, b"prompt").hash(), h0);
        assert_ne!(test_header(&m, &e, &v, 6, b"prompt").hash(), h0);
        assert_ne!(test_header(&m, &e, &v, 5, b"prompt2").hash(), h0);
    }

    #[test]
    fn single_logit_flip_changes_final_digest() {
        let m = sha256(b"m");
        let e = sha256(b"e");
        let v = sha256(b"v");
        let header = test_header(&m, &e, &v, 1, b"p");

        let mut a = WitnessChain::from_header(&header);
        a.fold_step(1, &[100, 200, 300]);

        let mut b = WitnessChain::from_header(&header);
        b.fold_step(1, &[100, 200, 301]); // one-bit-ish flip in the last logit

        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn token_id_substitution_changes_final_digest() {
        let m = sha256(b"m");
        let e = sha256(b"e");
        let v = sha256(b"v");
        let header = test_header(&m, &e, &v, 1, b"p");

        let mut a = WitnessChain::from_header(&header);
        a.fold_step(1, &[100, 200, 300]);

        let mut b = WitnessChain::from_header(&header);
        b.fold_step(2, &[100, 200, 300]);

        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn step_reordering_changes_final_digest() {
        let m = sha256(b"m");
        let e = sha256(b"e");
        let v = sha256(b"v");
        let header = test_header(&m, &e, &v, 2, b"p");

        let mut a = WitnessChain::from_header(&header);
        a.fold_step(1, &[10]);
        a.fold_step(2, &[20]);

        let mut b = WitnessChain::from_header(&header);
        b.fold_step(2, &[20]);
        b.fold_step(1, &[10]);

        assert_ne!(a.digest(), b.digest());
        let _ = vec![0u8; 0]; // keep `alloc::vec` import exercised if unused elsewhere
    }
}
