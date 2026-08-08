//! Witness v1 — chained cryptographic receipts for CIS-1 decode transcripts.
//!
//! A witness binds (model, embeddings, vocabulary, prompt, length budget) to
//! the COMPLETE integer state trajectory of a greedy CIS decode: at every
//! decode step the chain absorbs the chosen token id and the full i64 logit
//! vector that produced it. Any conforming implementation replaying the same
//! inputs reproduces the same chain; any divergence — one logit, one bit,
//! anywhere — changes the final digest.
//!
//! v0 (`aegis-linux/examples/witness.rs`) chained token *text* under the f32
//! engine and provably FAILS across arithmetic paths (E1). v1 inverts that:
//! built on the full-integer engine, the chain is the portable claim — it
//! must verify across paths, ISAs, and machines, or CIS-1 is falsified.
//!
//! This module is `no_std` + no-alloc so the UEFI unikernel verifies receipts
//! with the same bytes of code as the Linux host. Identity/correctness tool
//! only; never a perf instrument.

/// SHA-256 (FIPS 180-4). Streaming, fixed 64-byte block buffer, no alloc.
/// Correctness is pinned to the FIPS test vectors in
/// `tests/witness_contract.rs`.
pub struct Sha256 {
    h: [u32; 8],
    block: [u8; 64],
    blen: usize,
    total: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Sha256 {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0u8; 64],
            blen: 0,
            total: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        while !data.is_empty() {
            let take = core::cmp::min(64 - self.blen, data.len());
            self.block[self.blen..self.blen + take].copy_from_slice(&data[..take]);
            self.blen += take;
            data = &data[take..];
            if self.blen == 64 {
                let block = self.block;
                self.compress(&block);
                self.blen = 0;
            }
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, h].iter().enumerate() {
            self.h[i] = self.h[i].wrapping_add(*v);
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bitlen = self.total.wrapping_mul(8);
        self.update(&[0x80]);
        while self.blen != 56 {
            self.update(&[0]);
        }
        self.update(&bitlen.to_be_bytes());
        debug_assert_eq!(self.blen, 0, "padding must end block-aligned");
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[i * 4..i * 4 + 4].copy_from_slice(&self.h[i].to_be_bytes());
        }
        out
    }
}

/// One-shot convenience over the streaming hasher.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(data);
    s.finalize()
}

/// Domain-separation prefix for v1 headers. Version-bumping the format means
/// changing this string, which changes every digest downstream — deliberate.
pub const WITNESS_DOMAIN_V1: &[u8] = b"AEGIS-WITNESS v1-CIS\n";

/// The inputs a witness commits to, hashed into the chain's genesis value.
pub struct WitnessHeader<'a> {
    pub model_sha: &'a [u8; 32],
    pub embed_sha: &'a [u8; 32],
    pub vocab_sha: &'a [u8; 32],
    pub max_new: u64,
    pub prompt: &'a [u8],
}

impl WitnessHeader<'_> {
    /// Length-prefixed field encoding — no two distinct headers can collide
    /// by field-boundary ambiguity (the fixed-width fields need no prefix;
    /// the variable-width prompt gets one).
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
    /// no allocation regardless of vocabulary size.
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

/// Lowercase-hex encode `src` into `dst` (2 bytes out per byte in).
/// Returns the number of bytes written. `no_std`-printable digests for the
/// unikernel without a formatter.
pub fn hex_lower(src: &[u8], dst: &mut [u8]) -> usize {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    assert!(dst.len() >= src.len() * 2, "hex output buffer too small");
    for (i, &b) in src.iter().enumerate() {
        dst[i * 2] = HEX[(b >> 4) as usize];
        dst[i * 2 + 1] = HEX[(b & 0x0f) as usize];
    }
    src.len() * 2
}
