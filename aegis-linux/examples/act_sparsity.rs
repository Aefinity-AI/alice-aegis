//! ReLU^2 activation-sparsity probe (data-first, no kernel).
//!
//! Question: on real prompts, what fraction of the down_proj INPUT vector is
//! exactly 0.0, per layer, per token? BitNet-2B's squared-ReLU gate zeroes
//! every negative pre-activation, and `up[i] *= gate[i]` propagates those
//! zeros into the vector down_proj consumes — each zero kills a whole COLUMN
//! of down_proj. This probe measures that zero fraction; it changes no
//! kernel and records no timing.
//!
//! Run with a SwiGLU (silu) artifact set as the control: silu has no flat
//! zero region, so its exact-zero fraction should be near zero (only the
//! int8 quantization dead zone contributes).
//!
//! Usage:
//!   act_sparsity <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <label> [max_new_tokens]
//!
//! Requires `--features act_stats` (counting hooks in aegis-core).

extern crate alloc;
use aegis_core::inference::TernaryInferenceEngine;

use std::env;
use std::fs::File;
use std::io::Read;

const PROMPTS: [&str; 4] = [
    "The capital of France is",
    "Water boils at a temperature of",
    "In 1969, the first humans landed on",
    "A ternary computer represents numbers using",
];

#[derive(Clone, Copy)]
struct LayerAgg {
    tokens: usize,
    sum_z: f64,
    min_z: f64,
    max_z: f64,
}

impl LayerAgg {
    fn new() -> Self {
        Self {
            tokens: 0,
            sum_z: 0.0,
            min_z: f64::INFINITY,
            max_z: f64::NEG_INFINITY,
        }
    }
    fn push(&mut self, z: f64) {
        self.tokens += 1;
        self.sum_z += z;
        if z < self.min_z {
            self.min_z = z;
        }
        if z > self.max_z {
            self.max_z = z;
        }
    }
    fn mean(&self) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            self.sum_z / self.tokens as f64
        }
    }
}

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
            "Usage: {} <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <label> [max_new_tokens]",
            args.first().map(String::as_str).unwrap_or("act_sparsity")
        );
        std::process::exit(1);
    }
    let model_path = &args[1];
    let embed_path = &args[2];
    let vocab_path = &args[3];
    let label = &args[4];
    let max_new_tokens: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(32);

    let model_bytes = read_file(model_path)?;
    let emb_bytes = read_file(embed_path)?;
    let vocab_bytes = read_file(vocab_path)?;

    println!("=== act_sparsity probe ===");
    println!("label: {label}");
    println!("model: {model_path} ({} bytes)", model_bytes.len());
    println!("embed: {embed_path} ({} bytes)", emb_bytes.len());
    println!("vocab: {vocab_path} ({} bytes)", vocab_bytes.len());

    let mut engine = TernaryInferenceEngine::new(&emb_bytes, &model_bytes, &vocab_bytes)?;
    let num_layers = engine.config.num_hidden_layers;
    let inter = engine.config.intermediate_size;
    println!(
        "config: layers={} hidden={} intermediate={} vocab={} hidden_act={:?} template={:?}",
        num_layers,
        engine.config.hidden_size,
        inter,
        engine.config.vocab_size,
        engine.config.hidden_act,
        engine.config.chat_template,
    );
    println!(
        "engine int8_act: {}",
        if aegis_core::INT8_ACT_ENABLED {
            "on"
        } else {
            "off"
        }
    );
    println!(
        "prompts: {}, max_new_tokens: {max_new_tokens}",
        PROMPTS.len()
    );

    // Per-layer aggregates, prefill and decode kept apart.
    let mut prefill = vec![LayerAgg::new(); num_layers];
    let mut decode = vec![LayerAgg::new(); num_layers];

    for (i, prompt) in PROMPTS.iter().enumerate() {
        engine.act_zero_counts.clear();
        let response = engine.process_intent(prompt, max_new_tokens, |_tok| {});
        let records = std::mem::take(&mut engine.act_zero_counts);

        let n_prefill_tokens = records.iter().filter(|r| r.prefill && r.layer == 0).count();
        let n_decode_tokens = records
            .iter()
            .filter(|r| !r.prefill && r.layer == 0)
            .count();
        println!("--- prompt {}: {prompt:?} ---", i + 1);
        println!(
            "tokens: prefill={n_prefill_tokens} decode={n_decode_tokens}; response: {:?}",
            response.chars().take(80).collect::<String>()
        );

        for r in &records {
            assert_eq!(r.len, inter, "record length must equal intermediate_size");
            let z = r.zeros as f64 / r.len as f64;
            if r.prefill {
                prefill[r.layer].push(z);
            } else {
                decode[r.layer].push(z);
            }
        }
    }

    println!("=== per-layer down_proj-input exact-0.0 fraction (all prompts pooled) ===");
    println!(
        "layer  dec_toks  dec_mean  dec_min   dec_max   pre_toks  pre_mean  pre_min   pre_max"
    );
    for l in 0..num_layers {
        let d = &decode[l];
        let p = &prefill[l];
        println!(
            "L{:02}    {:>7}   {:.4}    {:.4}    {:.4}    {:>7}   {:.4}    {:.4}    {:.4}",
            l,
            d.tokens,
            d.mean(),
            d.min_z,
            d.max_z,
            p.tokens,
            p.mean(),
            p.min_z,
            p.max_z
        );
    }

    let overall = |aggs: &[LayerAgg]| -> (usize, f64) {
        let toks: usize = aggs.iter().map(|a| a.tokens).sum();
        let sum: f64 = aggs.iter().map(|a| a.sum_z).sum();
        (toks, if toks == 0 { 0.0 } else { sum / toks as f64 })
    };
    let (dec_n, dec_mean) = overall(&decode);
    let (pre_n, pre_mean) = overall(&prefill);
    println!("=== overall ===");
    println!("decode:  {dec_n} (token,layer) records, mean z = {dec_mean:.4}");
    println!("prefill: {pre_n} (token,layer) records, mean z = {pre_mean:.4}");

    Ok(())
}
