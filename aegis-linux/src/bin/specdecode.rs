//! Leg E-A gate harness: prompt-lookup speculative decoding vs sequential
//! greedy decode, on real artifacts.
//!
//! For every prompt and every draft length K it runs BOTH paths and compares
//! the emitted token stream byte for byte, then reports the counted quantity
//! the leg is pre-registered on: committed tokens per batched full-model
//! pass.
//!
//! Rule A: this harness prints COUNTS ONLY. It does not time anything, and
//! no rate may be derived from its output.
//!
//! Usage:
//!   specdecode <model> <embed> <vocab> <prompts.txt> <max_new> <k,k,...> [tag]

use aegis_core::inference::{SpecDecodeConfig, TernaryInferenceEngine};
use std::env;
use std::fs;
use std::io::Write;

/// FNV-1a 64 over the little-endian token ids — a stream identity that fits
/// on one line of a log, so two runs can be compared without diffing
/// thousands of tokens.
fn stream_digest(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 7 {
        eprintln!(
            "usage: {} <model> <embed> <vocab> <prompts.txt> <max_new> <k,k,...> [tag]",
            args[0]
        );
        std::process::exit(2);
    }
    let (model_path, embed_path, vocab_path) = (&args[1], &args[2], &args[3]);
    let prompts_path = &args[4];
    let max_new: usize = args[5].parse()?;
    let ks: Vec<usize> = args[6]
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;
    let tag = args.get(7).cloned().unwrap_or_else(|| String::from("run"));

    let model_bytes = fs::read(model_path)?;
    let vocab_bytes = fs::read(vocab_path)?;
    let emb_bytes = fs::read(embed_path)?;

    let mut engine = TernaryInferenceEngine::new(&emb_bytes, &model_bytes, &vocab_bytes)?;
    println!("== E-A specdecode gate ==");
    println!("tag              : {}", tag);
    println!("model            : {}", model_path);
    println!(
        "config           : {} layers, hidden {}, heads {}/{}, inter {}, vocab {}, window {}, {:?}",
        engine.config.num_hidden_layers,
        engine.config.hidden_size,
        engine.config.num_attention_heads,
        engine.config.num_key_value_heads,
        engine.config.intermediate_size,
        engine.config.vocab_size,
        engine.config.max_position_embeddings,
        engine.config.hidden_act
    );
    println!("max_new_tokens   : {}", max_new);
    println!("K set            : {:?}", ks);
    println!("ngram_min/max    : 1/3");
    println!();

    let text = fs::read_to_string(prompts_path)?;
    // One prompt per line; `#` comments and blank lines are skipped. A
    // literal two-character `\n` inside a line becomes a real newline, so a
    // multi-line code or receipt prompt still occupies one corpus line.
    let prompts: Vec<String> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.replace("\\n", "\n"))
        .collect();
    println!("prompts          : {} from {}", prompts.len(), prompts_path);
    println!();

    // Per-K aggregates.
    let mut agg_passes = vec![0u64; ks.len()];
    let mut agg_committed = vec![0u64; ks.len()];
    let mut agg_accepted = vec![0u64; ks.len()];
    let mut agg_drafted = vec![0u64; ks.len()];
    let mut agg_heads = vec![0u64; ks.len()];
    let mut identical = vec![0usize; ks.len()];
    let mut diverged: Vec<(usize, usize)> = Vec::new(); // (prompt index, K)
    let mut ref_tokens_total = 0u64;

    for (pi, p) in prompts.iter().enumerate() {
        let toks = engine.tokenizer.encode(p.as_str());
        if toks.is_empty() {
            println!("prompt {:02}: EMPTY after tokenization — skipped", pi);
            continue;
        }
        let reference = engine.greedy_decode(&toks, max_new);
        ref_tokens_total += reference.len() as u64;
        println!(
            "prompt {:02} | in {:3} tok | ref out {:3} tok | ref digest {:016x}",
            pi,
            toks.len(),
            reference.len(),
            stream_digest(&reference)
        );
        for (ki, &k) in ks.iter().enumerate() {
            let cfg = SpecDecodeConfig {
                k,
                ngram_max: 3,
                ngram_min: 1,
            };
            let (spec, stats) = engine.speculative_decode(&toks, max_new, cfg);
            let same = spec == reference;
            if same {
                identical[ki] += 1;
            } else {
                diverged.push((pi, k));
            }
            agg_passes[ki] += stats.passes;
            agg_committed[ki] += stats.committed;
            agg_accepted[ki] += stats.accepted;
            agg_drafted[ki] += stats.drafted;
            agg_heads[ki] += stats.lm_head_evals;
            println!(
                "   K={:<2} {} | digest {:016x} | out {:3} | passes {:4} committed {:4} drafted {:4} accepted {:4} lm_head {:4} | tok/pass {:.4}",
                k,
                if same { "IDENTICAL" } else { "DIVERGED " },
                stream_digest(&spec),
                spec.len(),
                stats.passes,
                stats.committed,
                stats.drafted,
                stats.accepted,
                stats.lm_head_evals,
                stats.tokens_per_pass()
            );
            let _ = std::io::stdout().flush();
        }
    }

    println!();
    println!("== aggregate ==");
    println!("reference tokens generated (all prompts): {}", ref_tokens_total);
    for (ki, &k) in ks.iter().enumerate() {
        let tpp = if agg_passes[ki] == 0 {
            0.0
        } else {
            agg_committed[ki] as f64 / agg_passes[ki] as f64
        };
        let acc = if agg_drafted[ki] == 0 {
            0.0
        } else {
            agg_accepted[ki] as f64 / agg_drafted[ki] as f64
        };
        println!(
            "K={:<2} identical {:>3}/{:<3} | passes {:6} | committed {:6} | drafted {:6} | accepted {:6} | lm_head {:6} | MEAN TOKENS/PASS {:.4} | draft acceptance {:.4}",
            k,
            identical[ki],
            prompts.len(),
            agg_passes[ki],
            agg_committed[ki],
            agg_drafted[ki],
            agg_accepted[ki],
            agg_heads[ki],
            tpp,
            acc
        );
    }
    println!();
    if diverged.is_empty() {
        println!("BYTE-IDENTITY: PASS — every (prompt, K) stream equals sequential greedy decode.");
    } else {
        println!("BYTE-IDENTITY: FAIL — {} (prompt,K) pairs diverged:", diverged.len());
        for (p, k) in &diverged {
            println!("  prompt {} K={}", p, k);
        }
    }
    if !diverged.is_empty() {
        std::process::exit(1);
    }
    Ok(())
}
