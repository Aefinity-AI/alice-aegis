// Feasibility probe (2026-07-29): isolate the process-launch confound from the
// OS-noise confound. Runs the IDENTICAL generation N times INSIDE one process,
// printing per-iteration decode cycles/token, prefill cycles, and an output
// hash so work-identity is verifiable.
//
// Usage: inproc_variance <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <max_new> <prompt> <n_iters>
extern crate alloc;
use aegis_core::inference::TernaryInferenceEngine;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        eprintln!("usage: {} MODEL EMBED VOCAB max_new prompt n_iters", a[0]);
        std::process::exit(1);
    }
    let model = fs::read(&a[1])?;
    let embed = fs::read(&a[2])?;
    let vocab = fs::read(&a[3])?;
    let max_new: usize = a[4].parse()?;
    let prompt = a[5].clone();
    let iters: usize = a[6].parse()?;

    let mut engine = TernaryInferenceEngine::new(&embed, &model, &vocab)?;
    println!("iter,decode_cyc_per_tok,decode_total_cyc,decode_steps,prefill_cyc,out_hash");
    for i in 0..iters {
        let resp = engine.process_intent(&prompt, max_new, |_t| {});
        // cheap FNV-1a of the response so identical work is provable
        let mut h: u64 = 0xcbf29ce484222325;
        for b in resp.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let steps = engine.last_decode_steps.max(1);
        println!(
            "{},{},{},{},{},{:016x}",
            i,
            engine.last_decode_ticks / steps as u64,
            engine.last_decode_ticks,
            engine.last_decode_steps,
            engine.last_prefill_ticks,
            h
        );
    }
    Ok(())
}
