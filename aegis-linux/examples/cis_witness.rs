//! cis_witness — witness v1: generate and verify chained receipts for
//! CIS-1 full-integer greedy decodes (`aegis_core::witness`).
//!
//! gen re-runs the same deterministic decode as `cis_decode` (greedy argmax,
//! EOS ignored, `CisMode::FullInt`) and emits a receipt binding artifact
//! hashes + prompt + every decode step's token id AND full i64 logit vector
//! into one SHA-256 chain. verify replays the decode from the receipt's
//! inputs and compares: artifact hashes, token ids, the FNV token-sequence
//! digest (the same fold `cis_decode` pins in CI), and the chain.
//!
//! The claim under test is the inverse of witness v0's: v0 (float engine)
//! provably FAILS across arithmetic paths; a v1 receipt must VERIFY on any
//! conforming host — any path, any ISA, any machine — or CIS-1 is falsified.
//!
//! Identity/correctness artifact ONLY (Rule A): prints no timing, ever.
//!
//!   cis_witness gen    <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <max_new> ["prompt"] > receipt
//!   cis_witness verify <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <receipt-file>

use aegis_core::cis_infer::{CisEngine, CisMode, CisModel, argmax_i64, fnv1a64};
use aegis_core::model::{FullBitNetPipeline, ModelConfig, SafeTensors};
use aegis_core::tokenizer::AegisTokenizer;
use aegis_core::witness::{WitnessChain, WitnessHeader, hex_lower, sha256};

fn hex(b: &[u8]) -> String {
    let mut out = vec![0u8; b.len() * 2];
    let n = hex_lower(b, &mut out);
    String::from_utf8(out[..n].to_vec()).unwrap()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap_or(0))
        .collect()
}

struct Replay {
    prompt_toks: usize,
    generated: Vec<u32>,
    fnv_digest: u64,
    chain: [u8; 32],
}

/// The deterministic decode both modes share. Step order and digest fold are
/// EXACTLY `cis_decode`'s (prompt ids then each generated id, LE bytes), so
/// for the same inputs the `cis-digest` line reproduces the CI-pinned
/// constant. The chain additionally absorbs each step's full logit vector.
fn replay(
    model_bytes: &[u8],
    embed_bytes: &[u8],
    vocab_bytes: &[u8],
    prompt: &str,
    max_new: usize,
    header: &WitnessHeader<'_>,
) -> Replay {
    let tensors = SafeTensors::deserialize(model_bytes).expect("parse MODEL.SAF");
    let cfg_json = tensors
        .metadata_field("aegis_config")
        .expect("read __metadata__")
        .expect("MODEL.SAF carries no aegis_config — repack in the forge");
    let config = ModelConfig::from_json(&cfg_json).expect("parse aegis_config");
    let pipeline = FullBitNetPipeline::new(&tensors, embed_bytes, &config).expect("build pipeline");
    let cis_model = CisModel::new(&pipeline, &config).expect("CIS model conversion");
    let mut engine = CisEngine::new_with_mode(&cis_model, CisMode::FullInt);

    let tokenizer = AegisTokenizer::new(vocab_bytes).expect("parse VOCAB.BIN");
    let prompt_ids = tokenizer.encode(prompt);
    assert!(!prompt_ids.is_empty(), "prompt tokenized to nothing");
    assert!(
        prompt_ids.len() + max_new <= config.max_position_embeddings,
        "prompt ({}) + max_new ({}) exceeds max_position_embeddings ({})",
        prompt_ids.len(),
        max_new,
        config.max_position_embeddings
    );

    let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
    let mut chain = WitnessChain::from_header(header);

    let mut pos = 0usize;
    for &t in &prompt_ids {
        fnv = fnv1a64(fnv, &t.to_le_bytes());
        engine.forward_step_int(t, pos);
        pos += 1;
    }

    let mut generated = Vec::with_capacity(max_new);
    for _ in 0..max_new {
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

    Replay {
        prompt_toks: prompt_ids.len(),
        generated,
        fnv_digest: fnv,
        chain: chain.digest(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: cis_witness gen    <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <max_new> [prompt]"
        );
        eprintln!("       cis_witness verify <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <receipt-file>");
        std::process::exit(2);
    }
    let mode = args[1].as_str();
    let model_bytes = std::fs::read(&args[2]).expect("read MODEL.SAF");
    let embed_bytes = std::fs::read(&args[3]).expect("read EMBED.BIN");
    let vocab_bytes = std::fs::read(&args[4]).expect("read VOCAB.BIN");
    let model_sha = sha256(&model_bytes);
    let embed_sha = sha256(&embed_bytes);
    let vocab_sha = sha256(&vocab_bytes);

    match mode {
        "gen" => {
            let max_new: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
            let prompt = args
                .get(6)
                .map(String::as_str)
                .unwrap_or("Once upon a time");
            let header = WitnessHeader {
                model_sha: &model_sha,
                embed_sha: &embed_sha,
                vocab_sha: &vocab_sha,
                max_new: max_new as u64,
                prompt: prompt.as_bytes(),
            };
            let r = replay(
                &model_bytes,
                &embed_bytes,
                &vocab_bytes,
                prompt,
                max_new,
                &header,
            );
            let ids: Vec<String> = r.generated.iter().map(|t| t.to_string()).collect();
            println!("AEGIS-WITNESS v1-CIS");
            println!("model {}", hex(&model_sha));
            println!("embed {}", hex(&embed_sha));
            println!("vocab {}", hex(&vocab_sha));
            println!("maxtok {max_new}");
            println!("prompt-hex {}", hex(prompt.as_bytes()));
            println!("prompt-toks {}", r.prompt_toks);
            println!("gen-toks {}", r.generated.len());
            println!("token-ids {}", ids.join(","));
            println!("cis-digest {:016x}", r.fnv_digest);
            println!("chain {}", hex(&r.chain));
        }
        "verify" => {
            let wtext = std::fs::read_to_string(&args[5]).expect("read receipt");
            let mut w_model = String::new();
            let mut w_embed = String::new();
            let mut w_vocab = String::new();
            let mut w_maxtok = 0usize;
            let mut w_prompt = String::new();
            let mut w_ids: Vec<u32> = Vec::new();
            let mut w_fnv = String::new();
            let mut w_chain = String::new();
            for line in wtext.lines() {
                let mut it = line.splitn(2, ' ');
                let (k, v) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
                match k {
                    "model" => w_model = v.into(),
                    "embed" => w_embed = v.into(),
                    "vocab" => w_vocab = v.into(),
                    "maxtok" => w_maxtok = v.parse().expect("maxtok"),
                    "prompt-hex" => {
                        w_prompt = String::from_utf8(unhex(v)).expect("prompt utf8");
                    }
                    "token-ids" => {
                        w_ids = v
                            .split(',')
                            .filter(|s| !s.is_empty())
                            .map(|s| s.parse().expect("token id"))
                            .collect();
                    }
                    "cis-digest" => w_fnv = v.into(),
                    "chain" => w_chain = v.into(),
                    _ => {}
                }
            }

            let mut fail = false;
            for (name, local, claimed) in [
                ("MODEL", hex(&model_sha), &w_model),
                ("EMBED", hex(&embed_sha), &w_embed),
                ("VOCAB", hex(&vocab_sha), &w_vocab),
            ] {
                if &local != claimed {
                    println!(
                        "FAIL artifact: {name} hash mismatch (receipt {} vs local {})",
                        &claimed[..16.min(claimed.len())],
                        &local[..16]
                    );
                    fail = true;
                }
            }
            if fail {
                std::process::exit(1);
            }

            let header = WitnessHeader {
                model_sha: &model_sha,
                embed_sha: &embed_sha,
                vocab_sha: &vocab_sha,
                max_new: w_maxtok as u64,
                prompt: w_prompt.as_bytes(),
            };
            let r = replay(
                &model_bytes,
                &embed_bytes,
                &vocab_bytes,
                &w_prompt,
                w_maxtok,
                &header,
            );
            let local_fnv = format!("{:016x}", r.fnv_digest);
            let local_chain = hex(&r.chain);
            println!(
                "receipt cis-digest {w_fnv} chain {}",
                &w_chain[..16.min(w_chain.len())]
            );
            println!(
                "local   cis-digest {local_fnv} chain {}",
                &local_chain[..16]
            );
            if local_fnv == w_fnv && local_chain == w_chain && r.generated == w_ids {
                println!(
                    "VERIFY PASS — replay reproduced {} tokens, the token digest, and the full logit chain bit-for-bit",
                    r.generated.len()
                );
            } else {
                if r.generated != w_ids {
                    let first = r.generated.iter().zip(&w_ids).position(|(a, b)| a != b);
                    println!(
                        "token divergence at generated index {:?} (local len {}, receipt len {})",
                        first,
                        r.generated.len(),
                        w_ids.len()
                    );
                }
                println!("VERIFY FAIL — replay diverged from the receipt");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}
