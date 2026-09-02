//! Boot-time witness verifier — the Provable AI Kit's headline mode.
//!
//! If the boot volume carries `RECEIPT.TXT` (witness v1 format, produced by
//! `aegis-linux`'s `cis_witness gen`), the unikernel becomes a verifier: it
//! hashes the model artifacts it just loaded, replays the receipt's decode
//! through the CIS-1 full-integer engine — the same ISA-independent
//! `cis_infer` path CI pins on x86-64 and aarch64 — and recomputes the
//! chained SHA-256 commitment over every decode step's token id and full
//! i64 logit vector. PASS means this machine, with no operating system
//! under it, reproduced the receipt's entire integer state trajectory
//! bit-for-bit.
//!
//! Identity/correctness mode ONLY (Rule A): this path prints no timing.

use aegis_core::cis_infer::{CisEngine, CisMode, CisModel, argmax_i64, fnv1a64};
use aegis_core::model::{FullBitNetPipeline, ModelConfig, SafeTensors};
use aegis_core::tokenizer::AegisTokenizer;
use aegis_core::witness::{WitnessChain, WitnessHeader, hex_lower, sha256};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

pub struct Verdict {
    pub pass: bool,
    /// Human-readable transcript of the verification, console- and
    /// BOOTLOG-ready.
    pub detail: String,
    /// Decode steps actually replayed. `AEFINITY OS` renders it as
    /// `job.N.items` (design §3) — a `VERIFY` that refused the receipt before
    /// the replay reports `0`, which is the difference between "disagreed
    /// after 64 steps" and "would not start".
    pub items: u64,
    /// The full 64-hex chained SHA-256 this machine computed over the replay,
    /// when there was one. `None` when the receipt never reached the replay
    /// (bad UTF-8, unparseable, artifact mismatch, model that would not
    /// build), because there is no digest to report and a zero one would be a
    /// claim. Rendered as `job.N.digest`.
    pub digest: Option<String>,
}

impl Verdict {
    /// A refusal that never reached the replay: no steps, no digest.
    fn refused(detail: String) -> Verdict {
        Verdict {
            pass: false,
            detail,
            items: 0,
            digest: None,
        }
    }
}

fn hex32(d: &[u8; 32]) -> String {
    let mut buf = [0u8; 64];
    let n = hex_lower(d, &mut buf);
    // hex_lower emits ASCII by construction.
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

fn unhex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    let b = s.as_bytes();
    if !b.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(b.len() / 2);
    for pair in b.as_chunks::<2>().0 {
        out.push(unhex_nibble(pair[0])? << 4 | unhex_nibble(pair[1])?);
    }
    Some(out)
}

struct Receipt {
    model: String,
    embed: String,
    vocab: String,
    maxtok: usize,
    prompt: String,
    token_ids: Vec<u32>,
    cis_digest: String,
    chain: String,
}

fn parse_receipt(text: &str) -> Option<Receipt> {
    let mut r = Receipt {
        model: String::new(),
        embed: String::new(),
        vocab: String::new(),
        maxtok: 0,
        prompt: String::new(),
        token_ids: Vec::new(),
        cis_digest: String::new(),
        chain: String::new(),
    };
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        let mut it = line.splitn(2, ' ');
        let (k, v) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
        match k {
            "model" => r.model = v.into(),
            "embed" => r.embed = v.into(),
            "vocab" => r.vocab = v.into(),
            "maxtok" => r.maxtok = v.parse().ok()?,
            "prompt-hex" => {
                r.prompt = String::from_utf8(unhex(v)?).ok()?;
            }
            "token-ids" => {
                for s in v.split(',').filter(|s| !s.is_empty()) {
                    r.token_ids.push(s.parse().ok()?);
                }
            }
            "cis-digest" => r.cis_digest = v.into(),
            "chain" => r.chain = v.into(),
            _ => {}
        }
    }
    if r.model.len() == 64 && r.maxtok > 0 && !r.chain.is_empty() {
        Some(r)
    } else {
        None
    }
}

/// Verify `receipt_bytes` against the artifacts already loaded into RAM.
/// Pure computation — the caller owns all console/BOOTLOG I/O.
pub fn run(model: &[u8], embed: &[u8], vocab: &[u8], receipt_bytes: &[u8]) -> Verdict {
    let mut d = String::new();
    let text = match core::str::from_utf8(receipt_bytes) {
        Ok(t) => t,
        Err(_) => {
            return Verdict::refused(String::from("RECEIPT.TXT is not valid UTF-8"));
        }
    };
    let r = match parse_receipt(text) {
        Some(r) => r,
        None => {
            return Verdict::refused(String::from("RECEIPT.TXT does not parse as witness v1"));
        }
    };

    // 1. Artifact binding: are these the exact bytes the receipt names?
    let model_sha = sha256(model);
    let embed_sha = sha256(embed);
    let vocab_sha = sha256(vocab);
    for (name, local, claimed) in [
        ("MODEL", hex32(&model_sha), &r.model),
        ("EMBED", hex32(&embed_sha), &r.embed),
        ("VOCAB", hex32(&vocab_sha), &r.vocab),
    ] {
        if &local != claimed {
            let _ = write!(
                d,
                "FAIL artifact: {} hash mismatch (receipt {}.. vs local {}..)",
                name,
                &claimed[..16.min(claimed.len())],
                &local[..16]
            );
            return Verdict::refused(d);
        }
    }
    let _ = writeln!(d, "artifacts: 3/3 hashes match the receipt");

    // 2. Replay under CIS-1 FullInt — the same loop as cis_decode /
    //    cis_witness, so the fnv fold reproduces the CI-pinned constant.
    let tensors = match SafeTensors::deserialize(model) {
        Ok(t) => t,
        Err(_) => {
            d.push_str("FAIL: MODEL.SAF did not parse");
            return Verdict::refused(d);
        }
    };
    let config = match tensors
        .metadata_field("aegis_config")
        .ok()
        .flatten()
        .and_then(|j| ModelConfig::from_json(&j).ok())
    {
        Some(c) => c,
        None => {
            d.push_str("FAIL: MODEL.SAF carries no parseable aegis_config");
            return Verdict::refused(d);
        }
    };
    let pipeline = match FullBitNetPipeline::new(&tensors, embed, &config) {
        Ok(p) => p,
        Err(_) => {
            d.push_str("FAIL: pipeline build");
            return Verdict::refused(d);
        }
    };
    let cis_model = match CisModel::new(&pipeline, &config) {
        Ok(m) => m,
        Err(_) => {
            d.push_str("FAIL: CIS model conversion");
            return Verdict::refused(d);
        }
    };
    let mut engine = CisEngine::new_with_mode(&cis_model, CisMode::FullInt);

    let tokenizer = match AegisTokenizer::new(vocab) {
        Ok(t) => t,
        Err(_) => {
            d.push_str("FAIL: VOCAB.BIN did not parse");
            return Verdict::refused(d);
        }
    };
    let prompt_ids = tokenizer.encode(&r.prompt);
    if prompt_ids.is_empty() || prompt_ids.len() + r.maxtok > config.max_position_embeddings {
        d.push_str("FAIL: prompt empty or exceeds context");
        return Verdict::refused(d);
    }

    let header = WitnessHeader {
        model_sha: &model_sha,
        embed_sha: &embed_sha,
        vocab_sha: &vocab_sha,
        max_new: r.maxtok as u64,
        prompt: r.prompt.as_bytes(),
    };
    let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
    let mut chain = WitnessChain::from_header(&header);

    let mut pos = 0usize;
    for &t in &prompt_ids {
        fnv = fnv1a64(fnv, &t.to_le_bytes());
        engine.forward_step_int(t, pos);
        pos += 1;
    }
    let mut generated: Vec<u32> = Vec::with_capacity(r.maxtok);
    for _ in 0..r.maxtok {
        let tok = {
            let logits = engine.decode_logits();
            let t = argmax_i64(logits);
            chain.fold_step(t, logits);
            t
        };
        fnv = fnv1a64(fnv, &tok.to_le_bytes());
        generated.push(tok);
        engine.forward_step_int(tok, pos);
        pos += 1;
    }

    let local_fnv = format!("{fnv:016x}");
    let local_chain = hex32(&chain.digest());
    let _ = writeln!(
        d,
        "receipt cis-digest {} chain {}..",
        r.cis_digest,
        &r.chain[..16.min(r.chain.len())]
    );
    let _ = writeln!(
        d,
        "local   cis-digest {} chain {}..",
        local_fnv,
        &local_chain[..16]
    );

    if local_fnv == r.cis_digest && local_chain == r.chain && generated == r.token_ids {
        let _ = write!(
            d,
            "VERIFY PASS — this machine reproduced all {} decode steps' full logit vectors bit-for-bit, with no OS underneath",
            generated.len()
        );
        Verdict {
            pass: true,
            detail: d,
            items: generated.len() as u64,
            digest: Some(local_chain),
        }
    } else {
        if generated != r.token_ids {
            let first = generated.iter().zip(&r.token_ids).position(|(a, b)| a != b);
            let _ = writeln!(d, "token divergence at generated index {first:?}");
        }
        let _ = write!(d, "VERIFY FAIL — replay diverged from the receipt");
        Verdict {
            pass: false,
            detail: d,
            items: generated.len() as u64,
            digest: Some(local_chain),
        }
    }
}
