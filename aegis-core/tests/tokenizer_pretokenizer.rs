//! Pre-tokenizer split rules and the merge-table id-0 guard.
//!
//! The reference tokenizers (Falcon-E, Llama-3) pre-tokenize whitespace runs
//! that end in newlines as ONE unit (`\s*[\r\n]+`), so BPE merges ' \n \n'
//! into a single token; and their merge tables include pairs involving token
//! id 0 ('!' in Llama-3). The engine historically split words at every
//! mapped space (breaking newline groups — a measured 15-token T2d
//! divergence on a 7,325-char sample) and dropped any merge touching id 0
//! (117 real Llama-3 merges).

use aegis_core::tokenizer::AegisTokenizer;

/// vocab.bin bytes: magic, token strings in id order, then ranked merges.
fn build_vocab(tokens: &[&str], merges: &[(u32, u32, u32)]) -> Vec<u8> {
    let mut out = 0x564F4341u32.to_le_bytes().to_vec();
    out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for t in tokens {
        out.extend_from_slice(&(t.len() as u16).to_le_bytes());
        out.extend_from_slice(t.as_bytes());
    }
    out.extend_from_slice(&(merges.len() as u32).to_le_bytes());
    for &(a, b, m) in merges {
        out.extend_from_slice(&a.to_le_bytes());
        out.extend_from_slice(&b.to_le_bytes());
        out.extend_from_slice(&m.to_le_bytes());
    }
    out
}

// Mapped alphabet: 'Ġ' = space, 'Ċ' = newline.
const V: &[&str] = &[
    "!",    // 0 — deliberately id 0: real vocabularies put '!' here
    "Ġ",    // 1
    "Ċ",    // 2
    "ĠĊ",   // 3
    "ĠĊĠĊ", // 4
    "a",    // 5
    "b",    // 6
    "Ġa",   // 7
    "ab",   // 8
    "!!",   // 9
];
const MERGES: &[(u32, u32, u32)] = &[
    (1, 2, 3), // Ġ + Ċ   -> ĠĊ
    (3, 3, 4), // ĠĊ + ĠĊ -> ĠĊĠĊ
    (1, 5, 7), // Ġ + a   -> Ġa
    (5, 6, 8), // a + b   -> ab
    (0, 0, 9), // ! + !   -> !!  (id-0 pair: the old guard dropped this)
];

fn tok() -> (Vec<u8>,) {
    (build_vocab(V, MERGES),)
}

#[test]
fn newline_runs_merge_as_one_pretoken() {
    let (bytes,) = tok();
    let t = AegisTokenizer::new(&bytes).expect("vocab loads");
    // " \n \n" must become the single merged token, not two ' \n' halves —
    // the exact shape of every measured T2d divergence site.
    assert_eq!(t.encode(" \n \n"), vec![4]);
    // A run ending in a space before a word: run flushes WITHOUT its final
    // space, which prefixes the word (`\s+(?!\S)` + `Ġ?\S+`).
    assert_eq!(t.encode(" \n \n ab"), vec![4, 7, 6]); // ' \n \n', 'Ġa', 'b'
}

#[test]
fn plain_words_and_double_spaces_keep_gpt2_shape() {
    let (bytes,) = tok();
    let t = AegisTokenizer::new(&bytes).expect("vocab loads");
    assert_eq!(t.encode("ab ab"), vec![8, 7, 6]); // 'ab', 'Ġa'+'b'
    // "ab  ab": lone space is its own pre-token, second prefixes the word.
    assert_eq!(t.encode("ab  ab"), vec![8, 1, 7, 6]);
    // Newline directly after a word splits the word (reference `\s*[\r\n]+`
    // starts at the newline) — merges must not cross it.
    assert_eq!(t.encode("ab\nab"), vec![8, 2, 8]);
}

#[test]
fn id0_merges_are_honored() {
    let (bytes,) = tok();
    let t = AegisTokenizer::new(&bytes).expect("vocab loads");
    // '!' is id 0; the (0,0)->9 merge must apply. The old any-id-zero guard
    // dropped it, yielding [0, 0].
    assert_eq!(t.encode("!!"), vec![9]);
}
