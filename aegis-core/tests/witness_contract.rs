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
