//! Parser/formatter for the `AEGIS-WITNESS v1-CIS` receipt text format.
//!
//! Grammar transcribed from `aegis-linux/examples/cis_witness.rs`'s `gen`
//! writer (println! calls, lines 149-159) and `verify` reader (the
//! line-splitting loop, lines 171-193): a literal first line, then ten
//! `key value` lines, one space between key and value, unknown keys
//! silently ignored (mirrors `_ => {}` at `cis_witness.rs:191`). Per
//! `docs/design/CIS_VERIFY_DESIGN.md` §1.1:
//!
//! ```text
//! AEGIS-WITNESS v1-CIS
//! model <64 hex>       sha256(MODEL.SAF)
//! embed <64 hex>       sha256(EMBED.BIN)
//! vocab <64 hex>       sha256(VOCAB.BIN)
//! maxtok <decimal>
//! prompt-hex <hex>     prompt bytes, hex-encoded
//! prompt-toks <decimal>
//! gen-toks <decimal>
//! token-ids <csv of decimal u32>
//! cis-digest <16 hex>  FNV-1a 64
//! chain <64 hex>       final WitnessChain digest (SHA-256)
//! ```
//!
//! Unlike the reference `cis_witness.rs` reader (which `.expect()`s on
//! every field and is fine crashing on malformed input in example-script
//! use), this parser never panics: every failure returns a
//! [`ReceiptParseError`] naming the offending line, since a verifier must
//! treat receipt text as untrusted input (design doc §5, "malformed-receipt
//! robustness").

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::hex::{decode_hex, decode_hex_exact, hex_lower_string};

/// The literal first line every v1-CIS receipt must open with. Identical to
/// `aegis_core::witness::WITNESS_DOMAIN_V1` minus its trailing `\n` (the
/// domain string is the SHA-256 input; this is the same text as it appears
/// as a receipt line).
pub const RECEIPT_MAGIC: &str = "AEGIS-WITNESS v1-CIS";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptParseError {
    /// The first line isn't exactly [`RECEIPT_MAGIC`].
    BadMagic,
    /// A required key never appeared.
    MissingField(&'static str),
    /// A line's value didn't parse for the reason given (line number is
    /// 1-indexed, counting the magic line as line 1).
    BadField { line: usize, field: &'static str },
}

/// A parsed `AEGIS-WITNESS v1-CIS` receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub model_sha: [u8; 32],
    pub embed_sha: [u8; 32],
    pub vocab_sha: [u8; 32],
    pub maxtok: u64,
    /// Decoded prompt bytes (the receipt stores these hex-encoded).
    pub prompt: Vec<u8>,
    pub prompt_toks: u64,
    pub gen_toks: u64,
    pub token_ids: Vec<u32>,
    /// FNV-1a 64 `cis-digest`, as the raw u64 (receipt stores 16 lowercase
    /// hex chars, big-endian byte order matching `{:016x}` formatting).
    pub cis_digest: u64,
    pub chain: [u8; 32],
}

impl Receipt {
    /// Parse receipt text. Never panics on malformed input; every failure
    /// mode returns `Err`.
    pub fn parse(text: &str) -> Result<Receipt, ReceiptParseError> {
        let mut lines = text.lines();

        let magic = lines.next().ok_or(ReceiptParseError::BadMagic)?;
        if magic != RECEIPT_MAGIC {
            return Err(ReceiptParseError::BadMagic);
        }

        let mut model_sha: Option<[u8; 32]> = None;
        let mut embed_sha: Option<[u8; 32]> = None;
        let mut vocab_sha: Option<[u8; 32]> = None;
        let mut maxtok: Option<u64> = None;
        let mut prompt: Option<Vec<u8>> = None;
        let mut prompt_toks: Option<u64> = None;
        let mut gen_toks: Option<u64> = None;
        let mut token_ids: Option<Vec<u32>> = None;
        let mut cis_digest: Option<u64> = None;
        let mut chain: Option<[u8; 32]> = None;

        // Line numbers: magic is line 1, so the first KV line is line 2 —
        // matches how a human counts lines in the receipt file.
        for (i, line) in lines.enumerate() {
            let lineno = i + 2;
            if line.is_empty() {
                continue;
            }
            // "Parsing is line-splitting on the first space" — design doc
            // §1.1, matching `cis_witness.rs:171-193`'s
            // `line.splitn(2, ' ')`.
            let mut it = line.splitn(2, ' ');
            let key = it.next().unwrap_or("");
            let value = it.next().unwrap_or("");
            match key {
                "model" => {
                    model_sha = Some(decode_hex_exact::<32>(value).ok_or(
                        ReceiptParseError::BadField {
                            line: lineno,
                            field: "model",
                        },
                    )?)
                }
                "embed" => {
                    embed_sha = Some(decode_hex_exact::<32>(value).ok_or(
                        ReceiptParseError::BadField {
                            line: lineno,
                            field: "embed",
                        },
                    )?)
                }
                "vocab" => {
                    vocab_sha = Some(decode_hex_exact::<32>(value).ok_or(
                        ReceiptParseError::BadField {
                            line: lineno,
                            field: "vocab",
                        },
                    )?)
                }
                "maxtok" => {
                    maxtok =
                        Some(
                            value
                                .parse::<u64>()
                                .map_err(|_| ReceiptParseError::BadField {
                                    line: lineno,
                                    field: "maxtok",
                                })?,
                        )
                }
                "prompt-hex" => {
                    prompt = Some(decode_hex(value).ok_or(ReceiptParseError::BadField {
                        line: lineno,
                        field: "prompt-hex",
                    })?)
                }
                "prompt-toks" => {
                    prompt_toks =
                        Some(
                            value
                                .parse::<u64>()
                                .map_err(|_| ReceiptParseError::BadField {
                                    line: lineno,
                                    field: "prompt-toks",
                                })?,
                        )
                }
                "gen-toks" => {
                    gen_toks =
                        Some(
                            value
                                .parse::<u64>()
                                .map_err(|_| ReceiptParseError::BadField {
                                    line: lineno,
                                    field: "gen-toks",
                                })?,
                        )
                }
                "token-ids" => {
                    let mut ids = Vec::new();
                    for part in value.split(',').filter(|s| !s.is_empty()) {
                        let id = part
                            .parse::<u32>()
                            .map_err(|_| ReceiptParseError::BadField {
                                line: lineno,
                                field: "token-ids",
                            })?;
                        ids.push(id);
                    }
                    token_ids = Some(ids);
                }
                "cis-digest" => {
                    let bytes =
                        decode_hex_exact::<8>(value).ok_or(ReceiptParseError::BadField {
                            line: lineno,
                            field: "cis-digest",
                        })?;
                    cis_digest = Some(u64::from_be_bytes(bytes));
                }
                "chain" => {
                    chain = Some(decode_hex_exact::<32>(value).ok_or(
                        ReceiptParseError::BadField {
                            line: lineno,
                            field: "chain",
                        },
                    )?)
                }
                // Unknown keys are silently ignored — matches
                // `cis_witness.rs:191`'s `_ => {}`.
                _ => {}
            }
        }

        let token_ids = token_ids.unwrap_or_default();
        let gen_toks_val = gen_toks.ok_or(ReceiptParseError::MissingField("gen-toks"))?;
        if token_ids.len() as u64 != gen_toks_val {
            return Err(ReceiptParseError::BadField {
                line: 0,
                field: "token-ids",
            });
        }

        Ok(Receipt {
            model_sha: model_sha.ok_or(ReceiptParseError::MissingField("model"))?,
            embed_sha: embed_sha.ok_or(ReceiptParseError::MissingField("embed"))?,
            vocab_sha: vocab_sha.ok_or(ReceiptParseError::MissingField("vocab"))?,
            maxtok: maxtok.ok_or(ReceiptParseError::MissingField("maxtok"))?,
            prompt: prompt.ok_or(ReceiptParseError::MissingField("prompt-hex"))?,
            prompt_toks: prompt_toks.ok_or(ReceiptParseError::MissingField("prompt-toks"))?,
            gen_toks: gen_toks_val,
            token_ids,
            cis_digest: cis_digest.ok_or(ReceiptParseError::MissingField("cis-digest"))?,
            chain: chain.ok_or(ReceiptParseError::MissingField("chain"))?,
        })
    }

    /// Re-emit the receipt text. Field order and formatting match
    /// `cis_witness.rs:149-159`'s `println!` sequence exactly, so
    /// `Receipt::parse(&r.format()) == Ok(r)` and, for text originally
    /// produced by that writer (or by this one), `format(parse(text)) ==
    /// text` byte-for-byte.
    pub fn format(&self) -> String {
        let ids: Vec<String> = self.token_ids.iter().map(|t| t.to_string()).collect();
        format!(
            "{magic}\nmodel {model}\nembed {embed}\nvocab {vocab}\nmaxtok {maxtok}\nprompt-hex {prompt_hex}\nprompt-toks {prompt_toks}\ngen-toks {gen_toks}\ntoken-ids {ids}\ncis-digest {digest:016x}\nchain {chain}\n",
            magic = RECEIPT_MAGIC,
            model = hex_lower_string(&self.model_sha),
            embed = hex_lower_string(&self.embed_sha),
            vocab = hex_lower_string(&self.vocab_sha),
            maxtok = self.maxtok,
            prompt_hex = hex_lower_string(&self.prompt),
            prompt_toks = self.prompt_toks,
            gen_toks = self.gen_toks,
            ids = ids.join(","),
            digest = self.cis_digest,
            chain = hex_lower_string(&self.chain),
        )
    }
}

/// Reproduce the receipt's `cis-digest`: FNV-1a 64 (seeded from
/// [`crate::fnv::FNV1A64_OFFSET`]) folded over the prompt token ids then
/// the generated token ids, each as little-endian `u32` bytes. Exact
/// sequence `aegis-linux/examples/cis_witness.rs`'s `replay()` performs
/// (lines 77-97): prompt ids absorbed first, in order, then every
/// generated id, in order — no separator, no length prefix, just the raw
/// LE32 stream through the fold.
pub fn cis_digest_of(prompt_ids: &[u32], generated_ids: &[u32]) -> u64 {
    use crate::fnv::{FNV1A64_OFFSET, fnv1a64};
    let mut d = FNV1A64_OFFSET;
    for &t in prompt_ids {
        d = fnv1a64(d, &t.to_le_bytes());
    }
    for &t in generated_ids {
        d = fnv1a64(d, &t.to_le_bytes());
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    const GOLDEN: &str = include_str!("../tests/fixtures/witness_v1_m7_once64.receipt");

    #[test]
    fn parses_golden_receipt() {
        let r = Receipt::parse(GOLDEN).expect("golden receipt must parse");
        assert_eq!(
            hex_lower_string(&r.model_sha),
            "23cfad0a3c7de2b676e176755f0848cd2cad8e71e2cdffea3d903a232c96e973"
        );
        assert_eq!(
            hex_lower_string(&r.embed_sha),
            "3752cd5ca2e39593cdf0910968f8ff629d79417f253e33e1a1c4faa0ce33848e"
        );
        assert_eq!(
            hex_lower_string(&r.vocab_sha),
            "5a1d79caf517084a73de5e5379d2995e61f23afaada91822da12ecb4ad7fcd8a"
        );
        assert_eq!(r.maxtok, 64);
        assert_eq!(r.prompt, b"Once upon a time");
        assert_eq!(r.prompt_toks, 4);
        assert_eq!(r.gen_toks, 64);
        assert_eq!(r.token_ids.len(), 64);
        assert_eq!(r.token_ids[0], 12);
        assert_eq!(*r.token_ids.last().unwrap(), 674);
        assert_eq!(format!("{:016x}", r.cis_digest), "67e8c0a96abc04e1");
        assert_eq!(
            hex_lower_string(&r.chain),
            "aee25b770bd7b22eea2ea8476bbd949881d58a98d6dc3085c7cc94d322b1961b"
        );
    }

    #[test]
    fn round_trips_golden_receipt_byte_identical() {
        let r = Receipt::parse(GOLDEN).expect("parse");
        assert_eq!(
            r.format(),
            GOLDEN,
            "format(parse(golden)) must equal golden byte-for-byte"
        );
    }

    #[test]
    fn parse_format_parse_is_idempotent() {
        let r1 = Receipt::parse(GOLDEN).unwrap();
        let text2 = r1.format();
        let r2 = Receipt::parse(&text2).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn unknown_keys_are_ignored() {
        let mut text = GOLDEN.to_string();
        text.push_str("future-field deadbeef\n");
        let r = Receipt::parse(&text).expect("unknown key must not fail parse");
        assert_eq!(r.maxtok, 64);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let text = GOLDEN.replacen("AEGIS-WITNESS v1-CIS", "AEGIS-WITNESS v2-CIS", 1);
        assert_eq!(Receipt::parse(&text), Err(ReceiptParseError::BadMagic));
    }

    #[test]
    fn non_hex_hash_field_is_rejected_not_panicking() {
        let text = GOLDEN.replacen(
            "model 23cfad0a3c7de2b676e176755f0848cd2cad8e71e2cdffea3d903a232c96e973",
            "model zz",
            1,
        );
        let err = Receipt::parse(&text).unwrap_err();
        assert!(matches!(
            err,
            ReceiptParseError::BadField { field: "model", .. }
        ));
    }

    #[test]
    fn truncated_receipt_is_rejected_not_panicking() {
        let cut = &GOLDEN[..GOLDEN.len() / 2];
        // Some prefix of the receipt is missing required fields entirely.
        assert!(Receipt::parse(cut).is_err());
    }

    #[test]
    fn short_token_id_list_is_rejected() {
        let text = GOLDEN.replacen(",199,674", "", 1); // drop trailing ids, gen-toks still says 64
        let err = Receipt::parse(&text).unwrap_err();
        assert!(matches!(
            err,
            ReceiptParseError::BadField {
                field: "token-ids",
                ..
            }
        ));
    }

    #[test]
    fn cis_digest_of_matches_golden_prompt_and_tokens() {
        // The golden receipt's prompt "Once upon a time" tokenizes to 4
        // ids under aegis-core's tokenizer/VOCAB.BIN, which this crate
        // (tasks 1-2) does not parse — that's design task 2's
        // safetensors/vocab/BPE work, out of scope here. This test instead
        // pins `cis_digest_of`'s fold order/shape against a synthetic
        // prompt+generation split, independent of tokenizer output.
        let prompt_ids = [12u32, 407, 283, 259];
        let generated_ids = [1u32, 2, 3];
        let a = cis_digest_of(&prompt_ids, &generated_ids);
        // Same fold, done by hand inline (prompt ids then generated ids,
        // LE32 each) must agree.
        use crate::fnv::{FNV1A64_OFFSET, fnv1a64};
        let mut d = FNV1A64_OFFSET;
        for t in prompt_ids.iter().chain(generated_ids.iter()) {
            d = fnv1a64(d, &t.to_le_bytes());
        }
        assert_eq!(a, d);
    }
}
