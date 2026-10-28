//! bench_decode — bb-1 ARM A wall-clock decode instrument (PREREG 2026-10-28 s.4).
//!
//! One fresh process = one run: load model (untimed), prefill + greedy decode of
//! N_gen tokens, print key=value lines. Greedy (process_intent is argmax),
//! ignore_eos=true so exactly N_gen tokens are produced.
//!
//! Seconds are CLOCK_MONOTONIC: decode_s = last_decode_ticks / tsc_hz, with
//! tsc_hz calibrated in-process against Instant (same technique as
//! amdahl_decode/clockstate). wall_gen_s is an independent Instant around the
//! whole process_intent call (prefill+decode+sampling), as a cross-check.
//!
//! NOTE: engine emits token 1 at the end of prefill; the timed decode loop makes
//! decode_steps = N_gen-1 forward steps. tok_s_decode = decode_steps/decode_s;
//! tok_s_ngen = N_gen/decode_s is also printed. Which one maps to the PREREG
//! "N_gen / decode seconds" must be fixed at review before boot 1.
//!
//! Threads: AEGIS_THREADS env (aegis-core ops.rs); --threads sets it before the
//! pool is first used.
//!
//! Usage: bench_decode <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> [--threads N] [--n-gen 64] [--prompt-file F]
//! Build: cargo build --release --features parallel --example bench_decode
use aegis_core::inference::TernaryInferenceEngine;
use std::time::Instant;

pub const PROMPT: &str = "Explain, in plain language, how a transformer language model \
turns a prompt into a reply, one token at a time.";

fn tsc_hz() -> f64 {
    use std::arch::x86_64::_rdtsc;
    let t0 = Instant::now();
    let c0 = unsafe { _rdtsc() };
    while t0.elapsed().as_millis() < 200 {
        std::hint::spin_loop();
    }
    let c1 = unsafe { _rdtsc() };
    (c1 - c0) as f64 / t0.elapsed().as_secs_f64()
}

fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: bench_decode MODEL EMBED VOCAB [--threads N] [--n-gen 64] [--prompt-file F]");
        std::process::exit(2);
    }
    let (mut threads, mut n_gen, mut prompt) = (0usize, 64usize, PROMPT.to_string());
    let mut i = 4;
    while i < a.len() {
        match a[i].as_str() {
            "--threads" => { threads = a[i + 1].parse().expect("threads"); i += 2; }
            "--n-gen" => { n_gen = a[i + 1].parse().expect("n-gen"); i += 2; }
            "--prompt-file" => {
                prompt = std::fs::read_to_string(&a[i + 1]).expect("prompt file").trim_end().to_string();
                i += 2;
            }
            o => { eprintln!("unknown arg {o}"); std::process::exit(2); }
        }
    }
    if threads > 0 {
        // SAFETY: single-threaded at this point; pool not yet created.
        unsafe { std::env::set_var("AEGIS_THREADS", threads.to_string()) };
    }
    let model = std::fs::read(&a[1]).expect("read MODEL");
    let emb = std::fs::read(&a[2]).expect("read EMBED");
    let vocab = std::fs::read(&a[3]).expect("read VOCAB");
    let hz = tsc_hz();
    let mut e = TernaryInferenceEngine::new(&emb, &model, &vocab).expect("engine init");
    e.ignore_eos = true;
    let t0 = Instant::now();
    let out = e.process_intent(&prompt, n_gen, |_| {});
    let wall = t0.elapsed().as_secs_f64();
    let decode_s = e.last_decode_ticks as f64 / hz;
    let steps = e.last_decode_steps;
    let prefill_s = e.last_prefill_ticks as f64 / hz;
    let mut h: u64 = 0xcbf29ce484222325;
    for b in out.as_bytes() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
    println!("arm=A");
    println!("threads={}", if threads > 0 { threads.to_string() } else { "default".into() });
    println!("prompt_tokens={}", e.last_prefill_tokens);
    println!("n_gen_requested={n_gen}");
    println!("decode_steps={steps}");
    println!("n_gen_emitted={}", steps + 1);
    println!("tsc_hz={hz:.0}");
    println!("decode_s={decode_s:.6}");
    println!("tok_s_decode={:.4}", steps as f64 / decode_s);
    println!("tok_s_ngen={:.4}", n_gen as f64 / decode_s);
    println!("prefill_s={prefill_s:.6}");
    println!("prefill_tok_s={:.4}", e.last_prefill_tokens as f64 / prefill_s);
    println!("wall_gen_s={wall:.6}");
    println!("peak_rss_kb={}", peak_rss_kb());
    println!("out_fnv={h:016x}");
}
