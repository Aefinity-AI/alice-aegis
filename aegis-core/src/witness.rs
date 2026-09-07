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

/// Compute message schedule word `i` into the 16-word rolling buffer `w`
/// (indexed mod 16). For `i < 16` the word was already loaded from the
/// block and is returned unchanged; for `i >= 16` the extension formula
/// overwrites the slot that held `w[i-16]` (no longer needed) with the new
/// word for `w[i]`, so the buffer never grows past 16 entries.
#[inline(always)]
fn sched_word(w: &mut [u32; 16], i: usize) -> u32 {
    let idx = i & 15;
    if i >= 16 {
        let x15 = w[(i + 1) & 15]; // w[i-15] == w[i+1 mod 16]
        let x2 = w[(i + 14) & 15]; // w[i-2]  == w[i+14 mod 16]
        let x16 = w[idx]; // w[i-16] == w[i mod 16], about to be overwritten
        let x7 = w[(i + 9) & 15]; // w[i-7]  == w[i+9 mod 16]
        let s0 = x15.rotate_right(7) ^ x15.rotate_right(18) ^ (x15 >> 3);
        let s1 = x2.rotate_right(17) ^ x2.rotate_right(19) ^ (x2 >> 10);
        w[idx] = x16.wrapping_add(s0).wrapping_add(x7).wrapping_add(s1);
    }
    w[idx]
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

    /// Absorb `data`. Whole 64-byte blocks are compressed straight out of
    /// the input slice via `chunks_exact` — no copy into `self.block`.
    /// Only a leftover tail (less than 64 bytes, carried across calls) ever
    /// touches `self.block`.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);

        // Finish a partial block carried from a previous call, if any.
        if self.blen > 0 {
            let need = 64 - self.blen;
            let take = core::cmp::min(need, data.len());
            self.block[self.blen..self.blen + take].copy_from_slice(&data[..take]);
            self.blen += take;
            data = &data[take..];
            if self.blen == 64 {
                let block = self.block;
                self.compress(&block);
                self.blen = 0;
            }
        }

        // Compress whole blocks directly from `data`, no intermediate copy.
        let mut chunks = data.chunks_exact(64);
        for chunk in &mut chunks {
            let block: &[u8; 64] = chunk
                .try_into()
                .expect("chunks_exact(64) yields 64-byte slices");
            self.compress(block);
        }

        // Carry any remaining tail (< 64 bytes) into self.block.
        let rem = chunks.remainder();
        if !rem.is_empty() {
            self.block[..rem.len()].copy_from_slice(rem);
            self.blen = rem.len();
        }
    }

    #[inline]
    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 16];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;

        // Fully unrolled 64 rounds. Each `round!` invocation names the 8
        // working variables in their current role order (a..h); the roles
        // rotate by one position every round (see the generated call list
        // below), so the "shuffle" is a compile-time renaming of locals —
        // no runtime move of h=g; g=f; ... — and every round's message word
        // comes from the 16-word rolling schedule via `sched_word`, never a
        // materialized `[u32; 64]`.
        macro_rules! round {
            ($va:ident, $vb:ident, $vc:ident, $vd:ident, $ve:ident, $vf:ident, $vg:ident, $vh:ident, $i:expr) => {{
                let wi = sched_word(&mut w, $i);
                let s1 = $ve.rotate_right(6) ^ $ve.rotate_right(11) ^ $ve.rotate_right(25);
                let ch = ($ve & $vf) ^ (!$ve & $vg);
                let t1 = $vh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[$i])
                    .wrapping_add(wi);
                let s0 = $va.rotate_right(2) ^ $va.rotate_right(13) ^ $va.rotate_right(22);
                let maj = ($va & $vb) ^ ($va & $vc) ^ ($vb & $vc);
                let t2 = s0.wrapping_add(maj);
                $vd = $vd.wrapping_add(t1);
                $vh = t1.wrapping_add(t2);
            }};
        }

        round!(a, b, c, d, e, f, g, h, 0);
        round!(h, a, b, c, d, e, f, g, 1);
        round!(g, h, a, b, c, d, e, f, 2);
        round!(f, g, h, a, b, c, d, e, 3);
        round!(e, f, g, h, a, b, c, d, 4);
        round!(d, e, f, g, h, a, b, c, 5);
        round!(c, d, e, f, g, h, a, b, 6);
        round!(b, c, d, e, f, g, h, a, 7);
        round!(a, b, c, d, e, f, g, h, 8);
        round!(h, a, b, c, d, e, f, g, 9);
        round!(g, h, a, b, c, d, e, f, 10);
        round!(f, g, h, a, b, c, d, e, 11);
        round!(e, f, g, h, a, b, c, d, 12);
        round!(d, e, f, g, h, a, b, c, 13);
        round!(c, d, e, f, g, h, a, b, 14);
        round!(b, c, d, e, f, g, h, a, 15);
        round!(a, b, c, d, e, f, g, h, 16);
        round!(h, a, b, c, d, e, f, g, 17);
        round!(g, h, a, b, c, d, e, f, 18);
        round!(f, g, h, a, b, c, d, e, 19);
        round!(e, f, g, h, a, b, c, d, 20);
        round!(d, e, f, g, h, a, b, c, 21);
        round!(c, d, e, f, g, h, a, b, 22);
        round!(b, c, d, e, f, g, h, a, 23);
        round!(a, b, c, d, e, f, g, h, 24);
        round!(h, a, b, c, d, e, f, g, 25);
        round!(g, h, a, b, c, d, e, f, 26);
        round!(f, g, h, a, b, c, d, e, 27);
        round!(e, f, g, h, a, b, c, d, 28);
        round!(d, e, f, g, h, a, b, c, 29);
        round!(c, d, e, f, g, h, a, b, 30);
        round!(b, c, d, e, f, g, h, a, 31);
        round!(a, b, c, d, e, f, g, h, 32);
        round!(h, a, b, c, d, e, f, g, 33);
        round!(g, h, a, b, c, d, e, f, 34);
        round!(f, g, h, a, b, c, d, e, 35);
        round!(e, f, g, h, a, b, c, d, 36);
        round!(d, e, f, g, h, a, b, c, 37);
        round!(c, d, e, f, g, h, a, b, 38);
        round!(b, c, d, e, f, g, h, a, 39);
        round!(a, b, c, d, e, f, g, h, 40);
        round!(h, a, b, c, d, e, f, g, 41);
        round!(g, h, a, b, c, d, e, f, 42);
        round!(f, g, h, a, b, c, d, e, 43);
        round!(e, f, g, h, a, b, c, d, 44);
        round!(d, e, f, g, h, a, b, c, 45);
        round!(c, d, e, f, g, h, a, b, 46);
        round!(b, c, d, e, f, g, h, a, 47);
        round!(a, b, c, d, e, f, g, h, 48);
        round!(h, a, b, c, d, e, f, g, 49);
        round!(g, h, a, b, c, d, e, f, 50);
        round!(f, g, h, a, b, c, d, e, 51);
        round!(e, f, g, h, a, b, c, d, 52);
        round!(d, e, f, g, h, a, b, c, 53);
        round!(c, d, e, f, g, h, a, b, 54);
        round!(b, c, d, e, f, g, h, a, 55);
        round!(a, b, c, d, e, f, g, h, 56);
        round!(h, a, b, c, d, e, f, g, 57);
        round!(g, h, a, b, c, d, e, f, 58);
        round!(f, g, h, a, b, c, d, e, 59);
        round!(e, f, g, h, a, b, c, d, 60);
        round!(d, e, f, g, h, a, b, c, 61);
        round!(c, d, e, f, g, h, a, b, 62);
        round!(b, c, d, e, f, g, h, a, 63);

        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(h);
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
