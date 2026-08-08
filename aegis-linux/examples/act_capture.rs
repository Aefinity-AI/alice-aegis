//! Capture REAL down_proj input vectors from a live model run (task #27).
//!
//! The column-skip A/B (aegis-core/benches/colskip_vs_incumbent.rs) must be
//! driven by the model's actual activation vectors: synthetic uniform-random
//! zeros under-model the clustering of ReLU^2 zeros, so a bench on synthetic
//! inputs would flatter or slander the skip kernel unpredictably. This
//! example runs the real engine (act_stats build) with `act_dump_enabled`,
//! collects the DECODE-path down_proj input vectors — bit-exact f32, -0.0
//! preserved — and writes them to a flat binary file the bench loads.
//!
//! This is a one-time DATA capture (counts/values, no timing), so it is fine
//! under Rule A on any machine; the file records values, not performance.
//!
//! File format (little-endian):
//!   8 bytes  magic  b"AEGISAV1"
//!   u32      dim    (vector length = intermediate_size)
//!   u32      count  (number of records)
//!   count x { u32 layer, u32 token_ordinal_within_layer, dim x f32 }
//!
//! Usage:
//!   act_capture <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <out.bin> [max_new_tokens]
//!
//! Requires `--features act_stats`. Default max_new_tokens = 8, over the same
//! 4 prompts as the A15 act_sparsity probe -> ~32 decode tokens.

extern crate alloc;
use aegis_core::inference::TernaryInferenceEngine;

use std::env;
use std::fs::File;
use std::io::{Read, Write};

const PROMPTS: [&str; 4] = [
    "The capital of France is",
    "Water boils at a temperature of",
    "In 1969, the first humans landed on",
    "A ternary computer represents numbers using",
];

fn read_file(path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut f = File::open(path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "Usage: {} <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <out.bin> [max_new_tokens]",
            args.first().map(String::as_str).unwrap_or("act_capture")
        );
        std::process::exit(1);
    }
    let model_path = &args[1];
    let embed_path = &args[2];
    let vocab_path = &args[3];
    let out_path = &args[4];
    let max_new_tokens: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);

    let model_bytes = read_file(model_path)?;
    let emb_bytes = read_file(embed_path)?;
    let vocab_bytes = read_file(vocab_path)?;

    println!("=== act_capture: real down_proj input vectors ===");
    println!(
        "machine: {} (data capture only — values, not timing)",
        env::var("AEGIS_MACHINE").unwrap_or_else(|_| "UNNAMED — set AEGIS_MACHINE".into())
    );
    println!("model: {model_path} ({} bytes)", model_bytes.len());
    println!("embed: {embed_path} ({} bytes)", emb_bytes.len());
    println!("vocab: {vocab_path} ({} bytes)", vocab_bytes.len());

    let mut engine = TernaryInferenceEngine::new(&emb_bytes, &model_bytes, &vocab_bytes)?;
    let num_layers = engine.config.num_hidden_layers;
    let inter = engine.config.intermediate_size;
    println!(
        "config: layers={} hidden={} intermediate={} vocab={} hidden_act={:?}",
        num_layers,
        engine.config.hidden_size,
        inter,
        engine.config.vocab_size,
        engine.config.hidden_act,
    );
    println!(
        "prompts: {}, max_new_tokens: {max_new_tokens} (decode records = prompts x tokens x layers)",
        PROMPTS.len()
    );

    engine.act_dump_enabled = true;

    // (layer, token_ordinal, values)
    let mut records: Vec<(u32, u32, Vec<f32>)> = Vec::new();
    for (i, prompt) in PROMPTS.iter().enumerate() {
        engine.act_dump.clear();
        engine.act_zero_counts.clear();
        let response = engine.process_intent(prompt, max_new_tokens, |_tok| {});
        let dump = std::mem::take(&mut engine.act_dump);

        let mut per_layer_tok = vec![0u32; num_layers];
        let mut decode_records = 0usize;
        let mut zero_sum = 0.0f64;
        for rec in dump {
            if rec.prefill {
                continue; // decode-path vectors only: that is the A/B target
            }
            assert_eq!(
                rec.values.len(),
                inter,
                "record length != intermediate_size"
            );
            let z = rec.values.iter().filter(|v| **v == 0.0).count() as f64 / inter as f64;
            zero_sum += z;
            decode_records += 1;
            records.push((rec.layer as u32, per_layer_tok[rec.layer], rec.values));
            per_layer_tok[rec.layer] += 1;
        }
        println!(
            "prompt {}: {prompt:?} -> {decode_records} decode records, mean z = {:.4}; response: {:?}",
            i + 1,
            if decode_records == 0 {
                0.0
            } else {
                zero_sum / decode_records as f64
            },
            response.chars().take(60).collect::<String>()
        );
    }

    // Whole-capture summary (counts only).
    let total_z: f64 = records
        .iter()
        .map(|(_, _, v)| v.iter().filter(|x| **x == 0.0).count() as f64 / inter as f64)
        .sum::<f64>()
        / records.len().max(1) as f64;
    let neg_zeros: usize = records
        .iter()
        .map(|(_, _, v)| {
            v.iter()
                .filter(|x| x.to_bits() == (-0.0f32).to_bits())
                .count()
        })
        .sum();
    println!(
        "captured {} decode vectors, dim {}, pooled mean z = {:.4}, bitwise -0.0 elements = {}",
        records.len(),
        inter,
        total_z,
        neg_zeros
    );

    let mut out = Vec::with_capacity(16 + records.len() * (8 + inter * 4));
    out.extend_from_slice(b"AEGISAV1");
    out.extend_from_slice(&(inter as u32).to_le_bytes());
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for (layer, tok, values) in &records {
        out.extend_from_slice(&layer.to_le_bytes());
        out.extend_from_slice(&tok.to_le_bytes());
        for v in values {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    File::create(out_path)?.write_all(&out)?;
    println!("wrote {} bytes to {out_path}", out.len());

    Ok(())
}
