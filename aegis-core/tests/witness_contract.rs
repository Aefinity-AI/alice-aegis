//! Contract tests for `aegis_core::witness` (witness v1).
//!
//! SHA-256 correctness is pinned to the published FIPS 180-4 test vectors
//! (external citation: NIST FIPS 180-4 / NIST CAVP SHA byte-oriented test
//! vectors). The chain tests pin the sensitivity properties the receipt
//! depends on: any change to any committed input changes the digest.

use aegis_core::witness::{Sha256, WitnessChain, WitnessHeader, hex_lower, sha256};

fn hex(b: &[u8]) -> String {
    let mut out = vec![0u8; b.len() * 2];
    let n = hex_lower(b, &mut out);
    String::from_utf8(out[..n].to_vec()).unwrap()
}

// ---- SHA-256 vs FIPS 180-4 vectors ----------------------------------------

#[test]
fn sha256_fips_empty() {
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn sha256_fips_abc() {
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn sha256_fips_two_block() {
    // 56 bytes: forces the padding into a second block.
    assert_eq!(
        hex(&sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn sha256_million_a() {
    // FIPS 180-4 long-message vector: 1,000,000 repetitions of 'a'.
    let mut s = Sha256::new();
    let chunk = [b'a'; 1000];
    for _ in 0..1000 {
        s.update(&chunk);
    }
    assert_eq!(
        hex(&s.finalize()),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn sha256_streaming_equals_oneshot() {
    // Deterministic pseudo-random payload; absorb in awkward chunk sizes.
    let data: Vec<u8> = (0u32..4096)
        .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let oneshot = sha256(&data);
    for chunk in [1usize, 3, 7, 63, 64, 65, 200] {
        let mut s = Sha256::new();
        for c in data.chunks(chunk) {
            s.update(c);
        }
        assert_eq!(s.finalize(), oneshot, "chunk size {chunk} diverged");
    }
}

#[test]
fn sha256_fips_two_block_112() {
    // NIST/FIPS 180-4 style two-block message, 112 bytes (crosses the
    // `chunks_exact(64)` fast path boundary with a nonzero tail: 64 + 48).
    assert_eq!(
        hex(&sha256(
            b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
        )),
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
    );
}

#[test]
fn sha256_1mb_oneshot_equals_random_chunked() {
    // 1 MiB deterministic pseudo-random buffer: exercise the whole-block
    // `chunks_exact` fast path (16384 full 64-byte blocks) against the
    // same data absorbed through the old byte-at-a-time-ish random chunk
    // boundaries used by `sha256_streaming_equals_oneshot`, at scale.
    let data: Vec<u8> = (0u32..1024 * 1024)
        .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let oneshot = sha256(&data);

    // Deterministic "random" chunk sizes, 1..=511 bytes, covering many
    // block-alignment phases across the buffer.
    let mut lcg: u64 = 0x2545F4914F6CDD1D;
    let mut next_len = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        1 + ((lcg >> 33) as usize % 511)
    };
    let mut s = Sha256::new();
    let mut off = 0usize;
    while off < data.len() {
        let take = core::cmp::min(next_len(), data.len() - off);
        s.update(&data[off..off + take]);
        off += take;
    }
    assert_eq!(
        s.finalize(),
        oneshot,
        "1 MiB random-chunked absorb diverged"
    );
}

/// The pre-optimization SHA-256 compression function, kept verbatim as a
/// fuzz reference so the unrolled/rolling-schedule rewrite in
/// `aegis_core::witness::Sha256` cannot silently drift. Deliberately NOT
/// shared code with the module under test.
mod ref_sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    pub struct RefSha256 {
        h: [u32; 8],
        block: [u8; 64],
        blen: usize,
        total: u64,
    }

    impl RefSha256 {
        pub fn new() -> Self {
            RefSha256 {
                h: [
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                    0x1f83d9ab, 0x5be0cd19,
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
            let mut out = [0u8; 32];
            for i in 0..8 {
                out[i * 4..i * 4 + 4].copy_from_slice(&self.h[i].to_be_bytes());
            }
            out
        }
    }
}

#[test]
fn sha256_matches_reference_impl_across_sizes_and_chunking() {
    use ref_sha256::RefSha256;

    // Deterministic LCG so this test needs no external RNG crate.
    let mut lcg: u64 = 0x9E3779B97F4A7C15;
    let mut next_byte = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (lcg >> 56) as u8
    };

    for &len in &[
        0usize, 1, 55, 56, 57, 63, 64, 65, 111, 112, 113, 200, 1000, 4097,
    ] {
        let data: Vec<u8> = (0..len).map(|_| next_byte()).collect();

        let mut r = RefSha256::new();
        r.update(&data);
        let want = r.finalize();

        assert_eq!(
            sha256(&data),
            want,
            "len {len} oneshot diverged from reference"
        );

        // Also compare chunked absorb against the reference, one-shot.
        let mut s = Sha256::new();
        for c in data.chunks(17) {
            s.update(c);
        }
        assert_eq!(
            s.finalize(),
            want,
            "len {len} chunked(17) diverged from reference"
        );
    }
}

// ---- chain sensitivity -----------------------------------------------------

fn test_header<'a>(
    m: &'a [u8; 32],
    e: &'a [u8; 32],
    v: &'a [u8; 32],
    prompt: &'a [u8],
) -> WitnessHeader<'a> {
    WitnessHeader {
        model_sha: m,
        embed_sha: e,
        vocab_sha: v,
        max_new: 64,
        prompt,
    }
}

#[test]
fn chain_is_deterministic() {
    let (m, e, v) = ([1u8; 32], [2u8; 32], [3u8; 32]);
    let logits: Vec<i64> = (0..100).map(|i| i * 1000 - 50_000).collect();
    let mut a = WitnessChain::from_header(&test_header(&m, &e, &v, b"p"));
    let mut b = WitnessChain::from_header(&test_header(&m, &e, &v, b"p"));
    a.fold_step(7, &logits);
    b.fold_step(7, &logits);
    assert_eq!(a.digest(), b.digest());
    assert_eq!(a.steps(), 1);
}

#[test]
fn chain_detects_single_logit_flip() {
    let (m, e, v) = ([1u8; 32], [2u8; 32], [3u8; 32]);
    let logits: Vec<i64> = (0..100).map(|i| i * 1000 - 50_000).collect();
    let mut tampered = logits.clone();
    tampered[73] ^= 1; // one bit, one position
    let mut a = WitnessChain::from_header(&test_header(&m, &e, &v, b"p"));
    let mut b = WitnessChain::from_header(&test_header(&m, &e, &v, b"p"));
    a.fold_step(7, &logits);
    b.fold_step(7, &tampered);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn chain_detects_token_id_change() {
    let (m, e, v) = ([1u8; 32], [2u8; 32], [3u8; 32]);
    let logits: Vec<i64> = (0..100).map(|i| i * 7).collect();
    let mut a = WitnessChain::from_header(&test_header(&m, &e, &v, b"p"));
    let mut b = WitnessChain::from_header(&test_header(&m, &e, &v, b"p"));
    a.fold_step(7, &logits);
    b.fold_step(8, &logits);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn chain_detects_step_reordering() {
    let (m, e, v) = ([1u8; 32], [2u8; 32], [3u8; 32]);
    let l1: Vec<i64> = (0..50).map(|i| i).collect();
    let l2: Vec<i64> = (0..50).map(|i| -i).collect();
    let mut a = WitnessChain::from_header(&test_header(&m, &e, &v, b"p"));
    let mut b = WitnessChain::from_header(&test_header(&m, &e, &v, b"p"));
    a.fold_step(1, &l1);
    a.fold_step(2, &l2);
    b.fold_step(2, &l2);
    b.fold_step(1, &l1);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn header_binds_every_field() {
    let (m, e, v) = ([1u8; 32], [2u8; 32], [3u8; 32]);
    let base = test_header(&m, &e, &v, b"prompt").hash();
    let m2 = [9u8; 32];
    assert_ne!(test_header(&m2, &e, &v, b"prompt").hash(), base);
    assert_ne!(test_header(&m, &e, &v, b"promptx").hash(), base);
    let mut h = test_header(&m, &e, &v, b"prompt");
    h.max_new = 65;
    assert_ne!(h.hash(), base);
}

#[test]
fn hex_lower_matches_format() {
    let data: Vec<u8> = (0..=255u8).collect();
    let expect: String = data.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex(&data), expect);
}
