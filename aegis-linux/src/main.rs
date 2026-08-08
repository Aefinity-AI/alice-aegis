extern crate alloc;
use aegis_core::inference::TernaryInferenceEngine;

use std::env;
use std::fs::File;
use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================================================");
    println!(" A.L.I.C.E. Linux Evaluation Harness");
    println!("==================================================");

    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 6 {
        eprintln!(
            "Usage: {} <model_path> <embeddings_path> <vocab_path> [max_new_tokens] [prompt] | --parity",
            args.get(0).unwrap_or(&String::from("aegis-linux"))
        );
        std::process::exit(1);
    }

    let model_path = &args[1];
    let embeddings_path = &args[2];
    let vocab_path = &args[3];
    let max_new_tokens: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let prompt = args
        .get(5)
        .cloned()
        .unwrap_or_else(|| String::from("The capital of France is"));

    let mut f = File::open(model_path)?;
    let mut model_bytes = Vec::new();
    f.read_to_end(&mut model_bytes)?;

    let mut v = File::open(vocab_path)?;
    let mut vocab_bytes = Vec::new();
    v.read_to_end(&mut vocab_bytes)?;

    let mut e = File::open(embeddings_path)?;
    let mut emb_bytes = Vec::new();
    e.read_to_end(&mut emb_bytes)?;

    println!("Initializing Ternary Inference Engine...");
    let mut engine = TernaryInferenceEngine::new(&emb_bytes, &model_bytes, &vocab_bytes)?;

    // The model identity comes from the artifact's embedded config, not a
    // hardcoded banner — this harness also runs Falcon-E and future ports.
    println!(
        "Engine Online ({} layers, hidden {}, vocab {}, {:?} template). Evaluation Harness Ready.",
        engine.config.num_hidden_layers,
        engine.config.hidden_size,
        engine.config.vocab_size,
        engine.config.chat_template
    );

    // Parity harness: passing `--parity` anywhere runs the same tokens through
    // forward_batch (prefill) and forward_step (decode) and compares the
    // hidden state at every position. The two paths must agree: a stride or
    // KV-write divergence between them corrupts prompts while decode still
    // "works" — this project's worst historical bug class.
    if args.iter().any(|a| a == "--parity") {
        let text = "The quick brown fox jumps over the lazy dog. Paris is the capital of France.";
        let tokens = engine.tokenizer.encode(text);
        println!(
            "\n[PARITY] {} tokens through forward_batch vs forward_step...",
            tokens.len()
        );
        let max_diff = engine.prefill_decode_parity(&tokens);
        println!(
            "[PARITY] max |batch - step| over all positions/dims: {:e}",
            max_diff
        );
        if max_diff == 0.0 {
            println!("[PARITY] PASS — bit-identical.");
        } else if max_diff <= 1e-4 {
            println!("[PARITY] PASS — within FP reduction-order tolerance (1e-4).");
        } else {
            println!(
                "[PARITY] FAIL — prefill and decode disagree. Check embedding stride and KV write path first."
            );
            std::process::exit(1);
        }
        return Ok(());
    }

    println!("\nPrompt: {}", prompt);

    // The callback fires once per generated token, plus two status messages
    // the engine emits inline; exclude those to get an exact token count.
    // Timestamp the first real token so decode can be timed apart from prefill.
    let mut generated = 0usize;
    let mut first_token_at: Option<std::time::Instant> = None;
    let t_start = std::time::Instant::now();
    let response = engine.process_intent(&prompt, max_new_tokens, |token| {
        if !token.starts_with("[SYSTEM]") && !token.contains("[PERFORMANCE]") {
            if first_token_at.is_none() {
                first_token_at = Some(std::time::Instant::now());
            }
            generated += 1;
        }
        print!("{}", token);
        let _ = std::io::stdout().flush();
    });
    let elapsed = t_start.elapsed().as_secs_f64();
    let decode_secs = first_token_at
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);

    println!("\n\nFinal Full Response: {}", response);

    // Machine-readable lines consumed by measure_energy.sh
    println!("Generated {} tokens", generated);
    println!(
        "Wall time {:.3} s ({:.2} tok/s incl. prefill)",
        elapsed,
        generated as f64 / elapsed
    );
    println!(
        "Prefill {} tokens in {} cycles ({} cycles/token)",
        engine.last_prefill_tokens,
        engine.last_prefill_cycles,
        engine.last_prefill_cycles / engine.last_prefill_tokens.max(1) as u64
    );
    // Decode timed from the first emitted token, so prefill cost is excluded.
    if generated > 1 && decode_secs > 0.0 {
        println!(
            "Decode {} tokens in {:.3} s ({:.2} tok/s decode-only)",
            generated - 1,
            decode_secs,
            (generated - 1) as f64 / decode_secs
        );
    }

    Ok(())
}
