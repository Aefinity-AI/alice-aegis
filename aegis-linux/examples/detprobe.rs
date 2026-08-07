//! E1 — cross-path determinism probe (identity/correctness only, never perf).
//!
//! Question: with f32 activations, the scalar and AVX2 kernels accumulate in
//! different orders and diverge at ~1e-4 relative. Does that divergence ever
//! flip a greedy argmax on the real M7 model — i.e., is float inference
//! already trustworthy as a cross-path witness, or is an integer canonical
//! semantics REQUIRED for bit-exact replay?
//!
//! Method: for each prompt, run the full generate loop twice on the AVX2 path
//! and twice on the scalar path, each on a FRESH engine (clean KV). Compare:
//!   (a) within-path run-to-run — must be identical or nothing else matters;
//!   (b) cross-path — first divergent token index, if any.
//! Every number printed below is computed from the runs in this process.

use aegis_core::inference::TernaryInferenceEngine;
use aegis_core::ops::{active_path_name, set_force_scalar};
use std::fs;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn run_once(
    model: &[u8],
    emb: &[u8],
    vocab: &[u8],
    prompt: &str,
    maxtok: usize,
) -> (Vec<String>, u64) {
    let mut engine = TernaryInferenceEngine::new(emb, model, vocab).expect("engine init failed");
    let mut toks: Vec<String> = Vec::new();
    let _ = engine.process_intent(prompt, maxtok, |t| {
        if !t.starts_with("[SYSTEM]") && !t.contains("[PERFORMANCE]") {
            toks.push(t.to_string());
        }
    });
    let h = fnv1a(toks.concat().as_bytes());
    (toks, h)
}

fn first_divergence(a: &[String], b: &[String]) -> Option<usize> {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return Some(i);
        }
    }
    if a.len() != b.len() { Some(n) } else { None }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: detprobe <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN>");
        std::process::exit(2);
    }
    let model = fs::read(&args[1]).expect("model");
    let emb = fs::read(&args[2]).expect("embed");
    let vocab = fs::read(&args[3]).expect("vocab");

    let prompts = [
        ("P1", "hello alice", 128usize),
        ("P2", "how are you today?", 128),
        ("P3", "continue", 128),
        (
            "P4",
            "Write a comprehensive and detailed essay about the future of artificial intelligence in aerospace.",
            192,
        ),
    ];

    println!("==== E1 detprobe: scalar vs AVX2 greedy-identity, fresh engine per run ====");

    set_force_scalar(false);
    let vec_path = active_path_name();
    set_force_scalar(true);
    let sca_path = active_path_name();
    set_force_scalar(false);
    println!("vector path = {vec_path}, forced path = {sca_path}");
    if vec_path == sca_path {
        println!("NOTE: no SIMD path active on this host; cross-path comparison is vacuous here.");
    }

    let mut all_within_ok = true;
    let mut total_cross_flips = 0usize;
    let mut total_tokens_compared = 0usize;

    for (tag, prompt, maxtok) in prompts {
        set_force_scalar(false);
        let (v1, hv1) = run_once(&model, &emb, &vocab, prompt, maxtok);
        let (v2, hv2) = run_once(&model, &emb, &vocab, prompt, maxtok);
        set_force_scalar(true);
        let (s1, hs1) = run_once(&model, &emb, &vocab, prompt, maxtok);
        let (s2, hs2) = run_once(&model, &emb, &vocab, prompt, maxtok);
        set_force_scalar(false);

        let within_v = hv1 == hv2 && v1.len() == v2.len();
        let within_s = hs1 == hs2 && s1.len() == s2.len();
        all_within_ok &= within_v && within_s;

        println!(
            "{tag}: {vec_path} run1 {} tok fnv={hv1:016x} | run2 fnv={hv2:016x} -> within-path {}",
            v1.len(),
            if within_v {
                "IDENTICAL"
            } else {
                "DIVERGED (!!)"
            }
        );
        println!(
            "{tag}: {sca_path} run1 {} tok fnv={hs1:016x} | run2 fnv={hs2:016x} -> within-path {}",
            s1.len(),
            if within_s {
                "IDENTICAL"
            } else {
                "DIVERGED (!!)"
            }
        );

        total_tokens_compared += v1.len().min(s1.len());
        match first_divergence(&v1, &s1) {
            None => println!(
                "{tag}: cross-path {vec_path} vs {sca_path}: IDENTICAL ({} tokens)",
                v1.len()
            ),
            Some(i) => {
                total_cross_flips += 1;
                let a = v1.get(i).map(String::as_str).unwrap_or("<end>");
                let b = s1.get(i).map(String::as_str).unwrap_or("<end>");
                println!(
                    "{tag}: cross-path FIRST DIVERGENCE at token {i}: {vec_path}={a:?} vs {sca_path}={b:?}"
                );
                println!("{tag}: {vec_path} text: {}", v1.concat().escape_debug());
                println!("{tag}: {sca_path} text: {}", s1.concat().escape_debug());
            }
        }
    }

    println!("==== E1 verdict ====");
    println!(
        "within-path determinism: {}",
        if all_within_ok {
            "PASS (all 16 runs)"
        } else {
            "FAIL — engine is not even self-deterministic; stop here"
        }
    );
    println!(
        "cross-path: {total_cross_flips} of {} prompts diverged over {} compared tokens",
        prompts.len(),
        total_tokens_compared
    );
    println!(
        "reading: {}",
        if total_cross_flips == 0 {
            "no argmax flipped on THESE prompts — but f32 identity remains unguaranteed by construction; integer semantics still required for a portable witness"
        } else {
            "f32 accumulation order flips real tokens — float inference cannot be a cross-implementation witness; integer canonical semantics is REQUIRED"
        }
    );
}
