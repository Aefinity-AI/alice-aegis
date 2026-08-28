//! VOCAB.BIN parse (magic `0x564F4341`, id_to_string, merges) + byte-level
//! BPE `encode()`/`decode()`. Transcribed from `aegis-core/src/tokenizer.rs`
//! (258 lines) per `docs/design/CIS_VERIFY_DESIGN.md` builder task 5 —
//! independent copy, no `path` dependency on `aegis-core`. This is the
//! tokenizer needed to reproduce prompt tokenization exactly: same magic,
//! same string table layout, same BPE merge-rank loop, same byte<->unicode
//! mapping, same pre-tokenizer word-splitting rules.

use alloc::{string::String, vec::Vec};

pub struct Tokenizer<'a> {
    pub id_to_string: Vec<&'a str>,
    pub string_to_id: Vec<(&'a str, u32)>,
    pub merges: alloc::collections::BTreeMap<(u32, u32), (u32, usize)>,
}

impl<'a> Tokenizer<'a> {
    /// Parse VOCAB.BIN. Identical byte layout to `tokenizer.rs:10-103`:
    /// magic `u32` LE `0x564F4341`, `num_tokens` `u32` LE, then that many
    /// `(u16 len, utf8 bytes)` records, then an optional merges table
    /// (`u32` count, then `(id1, id2, id_merged)` triples, each `u32` LE;
    /// an all-zero triple is null padding, skipped).
    pub fn new(vocab_bytes: &'a [u8]) -> Result<Self, String> {
        if vocab_bytes.len() < 8 {
            return Err("Vocab too small".into());
        }
        let magic = u32::from_le_bytes([
            vocab_bytes[0],
            vocab_bytes[1],
            vocab_bytes[2],
            vocab_bytes[3],
        ]);
        if magic != 0x564F_4341 {
            return Err("Invalid Vocab Magic".into());
        }

        let num_tokens = u32::from_le_bytes([
            vocab_bytes[4],
            vocab_bytes[5],
            vocab_bytes[6],
            vocab_bytes[7],
        ]);

        let mut id_to_string = Vec::with_capacity(num_tokens as usize);
        let mut string_to_id = Vec::with_capacity(num_tokens as usize);

        let mut offset = 8;
        for id in 0..num_tokens {
            if offset + 2 > vocab_bytes.len() {
                return Err("Vocab EOF".into());
            }
            let len = u16::from_le_bytes([vocab_bytes[offset], vocab_bytes[offset + 1]]) as usize;
            offset += 2;

            if offset + len > vocab_bytes.len() {
                return Err("Vocab EOF".into());
            }
            let bytes = &vocab_bytes[offset..offset + len];
            offset += len;

            let s = core::str::from_utf8(bytes).map_err(|_| "Invalid UTF8")?;
            id_to_string.push(s);
            string_to_id.push((s, id));
        }

        string_to_id.sort_by_key(|&(s, _)| s);

        let mut merges = alloc::collections::BTreeMap::new();
        if offset + 4 <= vocab_bytes.len() {
            let num_merges = u32::from_le_bytes([
                vocab_bytes[offset],
                vocab_bytes[offset + 1],
                vocab_bytes[offset + 2],
                vocab_bytes[offset + 3],
            ]);
            offset += 4;
            for rank in 0..num_merges {
                if offset + 12 > vocab_bytes.len() {
                    break;
                }
                let id1 = u32::from_le_bytes([
                    vocab_bytes[offset],
                    vocab_bytes[offset + 1],
                    vocab_bytes[offset + 2],
                    vocab_bytes[offset + 3],
                ]);
                offset += 4;
                let id2 = u32::from_le_bytes([
                    vocab_bytes[offset],
                    vocab_bytes[offset + 1],
                    vocab_bytes[offset + 2],
                    vocab_bytes[offset + 3],
                ]);
                offset += 4;
                let id_merged = u32::from_le_bytes([
                    vocab_bytes[offset],
                    vocab_bytes[offset + 1],
                    vocab_bytes[offset + 2],
                    vocab_bytes[offset + 3],
                ]);
                offset += 4;
                // Skip only the all-zero triple (null padding). Id 0 is a
                // real token ('!' in Llama-3) that participates in real
                // merges — an any-id-zero guard would silently drop them.
                if id1 != 0 || id2 != 0 || id_merged != 0 {
                    merges.insert((id1, id2), (id_merged, rank as usize));
                }
            }
        }

        Ok(Self {
            id_to_string,
            string_to_id,
            merges,
        })
    }

    pub fn vocab_len(&self) -> usize {
        self.id_to_string.len()
    }

    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        if let Ok(idx) = self.string_to_id.binary_search_by_key(&token, |&(s, _)| s) {
            Some(self.string_to_id[idx].1)
        } else {
            None
        }
    }

    /// GPT-2/Llama byte→unicode map: the exact inverse of the table
    /// `decode` uses. Identical to `tokenizer.rs:121-129`.
    fn byte_to_unicode(b: u8) -> char {
        let u: u32 = match b {
            33..=126 | 161..=172 | 174..=255 => b as u32,
            0..=32 => 256 + b as u32,
            127..=160 => 256 + 33 + (b as u32 - 127),
            173 => 323,
        };
        char::from_u32(u).unwrap_or('?')
    }

    /// Merge one word's token sequence in place using BPE ranks. Identical
    /// to `tokenizer.rs:132-159`.
    fn merge_word(&self, word_tokens: &mut Vec<u32>) {
        loop {
            if word_tokens.len() < 2 {
                break;
            }
            let mut best_rank = usize::MAX;
            let mut best_pair = None;
            let mut best_idx = 0;

            for j in 0..word_tokens.len() - 1 {
                if let Some(&(merged, rank)) =
                    self.merges.get(&(word_tokens[j], word_tokens[j + 1]))
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_pair = Some(merged);
                    best_idx = j;
                }
            }

            if let Some(merged) = best_pair {
                word_tokens[best_idx] = merged;
                word_tokens.remove(best_idx + 1);
            } else {
                break;
            }
        }
    }

    /// Byte-level BPE encode. Identical to `tokenizer.rs:169-222`: text is
    /// mapped byte-by-byte through `byte_to_unicode`, split into words on
    /// whitespace<->non-whitespace transitions (a detached trailing space
    /// prefixes the following word, GPT-2 `Ġ?\S+` semantics), then each
    /// word's tokens are BPE-merged independently.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mapped: Vec<char> = text.bytes().map(Self::byte_to_unicode).collect();

        let unk = self.get_token_id("<|reserved_special_token_0|>");
        let mut out = Vec::new();
        let mut word: Vec<u32> = Vec::new();
        let mut word_all_ws = false;
        let mut prev_char = '\0';
        let mut buf = String::new();

        for &c in &mapped {
            let c_ws = Self::is_mapped_whitespace(c);
            if !word.is_empty() && word_all_ws != c_ws {
                if word_all_ws && prev_char == 'Ġ' {
                    let sp = word.pop();
                    if !word.is_empty() {
                        self.merge_word(&mut word);
                        out.append(&mut word);
                    }
                    word.extend(sp);
                } else {
                    self.merge_word(&mut word);
                    out.append(&mut word);
                }
            }
            buf.clear();
            buf.push(c);
            if let Some(id) = self.get_token_id(&buf) {
                word.push(id);
            } else if let Some(id) = unk {
                word.push(id);
            }
            word_all_ws = if word.len() <= 1 {
                c_ws
            } else {
                word_all_ws && c_ws
            };
            prev_char = c;
        }
        if !word.is_empty() {
            self.merge_word(&mut word);
            out.append(&mut word);
        }
        out
    }

    /// Identical to `tokenizer.rs:227-229`.
    fn is_mapped_whitespace(c: char) -> bool {
        matches!(c, 'Ġ' | 'ĉ' | 'Ċ' | 'ċ' | 'Č' | 'č')
    }

    /// Identical to `tokenizer.rs:231-257`.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut byte_buf = Vec::new();
        for &id in token_ids {
            if (id as usize) < self.id_to_string.len() {
                let word = self.id_to_string[id as usize];
                for ch in word.chars() {
                    let u = ch as u32;
                    let b = if u < 256 {
                        u as u8
                    } else {
                        let offset = u - 256;
                        if offset <= 32 {
                            offset as u8
                        } else if (33..=66).contains(&offset) {
                            (offset - 33 + 127) as u8
                        } else if offset == 67 {
                            173
                        } else {
                            b'?'
                        }
                    };
                    byte_buf.push(b);
                }
            }
        }
        String::from_utf8_lossy(&byte_buf).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal VOCAB.BIN in memory: magic, count, records, no merges.
    fn make_vocab(tokens: &[&str]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0x564F_4341u32.to_le_bytes());
        b.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for t in tokens {
            b.extend_from_slice(&(t.len() as u16).to_le_bytes());
            b.extend_from_slice(t.as_bytes());
        }
        b
    }

    #[test]
    fn parses_minimal_vocab_and_looks_up_ids() {
        let buf = make_vocab(&["a", "b", "c"]);
        let tok = Tokenizer::new(&buf).unwrap();
        assert_eq!(tok.vocab_len(), 3);
        assert_eq!(tok.get_token_id("a"), Some(0));
        assert_eq!(tok.get_token_id("b"), Some(1));
        assert_eq!(tok.get_token_id("z"), None);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = make_vocab(&["a"]);
        buf[0] ^= 0xFF;
        assert!(Tokenizer::new(&buf).is_err());
    }

    #[test]
    fn too_small_is_rejected_not_panicking() {
        assert!(Tokenizer::new(&[0u8; 4]).is_err());
    }
}
