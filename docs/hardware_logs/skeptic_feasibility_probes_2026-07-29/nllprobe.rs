// Probe #3: how large is the reduction-order perturbation on a TEACHER-FORCED
// continuation NLL — the quantity ARC-Easy --mc actually scores?
//
// Teacher forcing removes the autoregressive feedback loop entirely (gold ids
// are fed, not argmax), so the perturbation should stay at forward-pass noise
// level. Compare against the measured ARC-Easy top1-top2 decision margins.
extern crate alloc;
use aegis_core::inference::TernaryInferenceEngine;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = std::env::args().nth(1).expect("artifact dir");
    let model = fs::read(format!("{base}/MODEL.SAF"))?;
    let embed = fs::read(format!("{base}/EMBED.BIN"))?;
    let vocab = fs::read(format!("{base}/VOCAB.BIN"))?;

    // ARC-Easy-shaped contexts: a question stem plus a short continuation.
    let texts = [
        "Question: Which of these is a source of light? Answer: the sun",
        "Question: What do plants need to make food? Answer: sunlight and water",
        "Question: Which material is a good conductor of heat? Answer: copper metal",
        "Question: What causes day and night on Earth? Answer: the rotation of Earth",
        "Question: Which is a physical change? Answer: melting ice into water",
        "Question: What is the main gas in air? Answer: nitrogen",
        "Question: Which animal is a mammal? Answer: a whale",
        "Question: What tool measures temperature? Answer: a thermometer",
    ];

    println!("artifacts={base}  capability={}", aegis_core::ops::simd_level_name());
    println!("teacher-forced total-NLL under two reduction orders (same machine, same weights)");
    println!("{:>4} {:>6} {:>18} {:>18} {:>14} {:>12}", "i", "ntok", "nll_avx2", "nll_scalar", "abs_delta", "rel");

    let mut worst_abs = 0.0f64;
    let mut worst_rel = 0.0f64;

    for (i, t) in texts.iter().enumerate() {
        aegis_core::ops::set_force_scalar(false);
        let mut e = TernaryInferenceEngine::new(&embed, &model, &vocab)?;
        let ids = e.tokenizer.encode(t);
        // calculate_perplexity returns exp(mean NLL); recover total NLL.
        let ppl_a = e.calculate_perplexity(&ids);
        let n = ids.len().saturating_sub(1) as f64;
        let nll_a = ppl_a.ln() * n;
        drop(e);

        aegis_core::ops::set_force_scalar(true);
        let mut e = TernaryInferenceEngine::new(&embed, &model, &vocab)?;
        let ppl_b = e.calculate_perplexity(&ids);
        let nll_b = ppl_b.ln() * n;
        drop(e);
        aegis_core::ops::set_force_scalar(false);

        let abs = (nll_a - nll_b).abs();
        let rel = if nll_a != 0.0 { abs / nll_a.abs() } else { 0.0 };
        if abs > worst_abs {
            worst_abs = abs;
        }
        if rel > worst_rel {
            worst_rel = rel;
        }
        println!("{i:>4} {:>6} {:>18.9} {:>18.9} {:>14.3e} {:>12.3e}", ids.len(), nll_a, nll_b, abs, rel);
    }

    println!("\nworst |delta total-NLL| = {worst_abs:.4e} nats   worst relative = {worst_rel:.3e}");
    println!("ARC-Easy n=570 measured decision margins (m3_1b_results_n570.jsonl):");
    println!("  acc      smallest top1-top2 margin = 1.5362e-2 nats");
    println!("  acc_norm smallest top1-top2 margin = 4.4956e-4 nats/byte");
    println!("=> flips are possible only if the perturbation exceeds the margin.");
    Ok(())
}
