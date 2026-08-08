mod mc;

use aegis_core::inference::TernaryInferenceEngine;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

/// A.L.I.C.E. perplexity evaluator — real measurement, no mock values.
///
/// Teacher-forced negative-log-likelihood computed by the same engine that
/// serves inference. Four modes:
///   (default)    chunked full-text PPL (whole file, KV-window-sized chunks)
///   --sample     single contiguous prefix capped at max_tokens (the original
///                harness behavior; regression anchor = 10.348 on test.txt
///                sha256 d790b833…, --sample 1900. The older 12.80 anchor is
///                unreproducible — its dataset file was deleted 2026-07-11)
///   --selfcheck  KV-isolation tripwire: a short measurement must be bitwise
///                identical before and after an unrelated long measurement
///   --mc         multiple-choice eval on explicit token ids (see mc.rs):
///                `--mc <items.jsonl> --mc-out <results.jsonl>` replaces the
///                text-file argument entirely; the tokenizer is bypassed so
///                both engine and reference score byte-identical inputs
///
/// The chunked mode refuses to report garbage: a near-char-level token/char
/// ratio or an absurd first-chunk PPL aborts with an artifact-mismatch error
/// instead of printing a number a reviewer might quote. The 2026-07-11
/// "overnight" run produced PPL ~500 precisely because a wrong vocab/embed
/// pairing was fed to a correct harness and nothing complained.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================================================");
    println!(" A.L.I.C.E. Perplexity Evaluator (measured)");
    println!("==================================================");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "Usage: {} <model.safetensors> <embed.bin> <vocab.bin> <text_file> [max_tokens] [--sample|--selfcheck]\n       {} <model.safetensors> <embed.bin> <vocab.bin> --mc <items.jsonl> --mc-out <results.jsonl>",
            args.first().map(String::as_str).unwrap_or("aegis-eval"),
            args.first().map(String::as_str).unwrap_or("aegis-eval")
        );
        std::process::exit(1);
    }

    let mut model_bytes = Vec::new();
    File::open(&args[1])?.read_to_end(&mut model_bytes)?;
    let mut emb_bytes = Vec::new();
    File::open(&args[2])?.read_to_end(&mut emb_bytes)?;
    let mut vocab_bytes = Vec::new();
    File::open(&args[3])?.read_to_end(&mut vocab_bytes)?;

    let mut engine = TernaryInferenceEngine::new(&emb_bytes, &model_bytes, &vocab_bytes)
        .map_err(|e| format!("engine init: {e}"))?;
    println!(
        "Engine online. SIMD level: {}",
        aegis_core::ops::simd_level_name()
    );

    // --mc replaces the text-file argument entirely: items carry explicit
    // token ids and the tokenizer is bypassed (ids-based parity with the
    // transformers reference).
    if let Some(p) = args.iter().position(|a| a == "--mc") {
        let items = args
            .get(p + 1)
            .filter(|a| !a.starts_with("--"))
            .ok_or("--mc requires <items.jsonl>")?;
        let op = args
            .iter()
            .position(|a| a == "--mc-out")
            .ok_or("--mc requires --mc-out <results.jsonl>")?;
        let out = args
            .get(op + 1)
            .filter(|a| !a.starts_with("--"))
            .ok_or("--mc-out requires <results.jsonl>")?;
        return mc::run_mc(&mut engine, items, out);
    }

    let text = std::fs::read_to_string(&args[4])?;
    let max_tokens: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1900);
    let mode = args
        .iter()
        .find(|a| a.starts_with("--"))
        .map(String::as_str);

    // The tokenizer is ASCII-oriented after vocab pruning; strip non-ASCII.
    let ascii: String = text.chars().filter(|c| c.is_ascii()).collect();

    match mode {
        Some("--sample") => run_sample(&mut engine, &ascii, max_tokens, &args[4]),
        Some("--selfcheck") => run_selfcheck(&mut engine, &ascii, max_tokens),
        // --cis consumes the engine (owned) so the float pass can be dropped
        // before the integer pass allocates (E2-7 memory discipline);
        // --cis-full borrows, running all three passes in one process.
        Some("--cis") => run_cis(engine, &ascii, max_tokens, &args[4]),
        Some("--cis-full") => run_cis_full(&mut engine, &ascii, max_tokens, &args[4]),
        _ => run_chunked(&mut engine, &ascii, max_tokens, &args[4]),
    }
}

/// Derive the --sample token window: the exact logic `run_sample` has always
/// used, factored out so `--cis` scores the identical token sequence.
fn sample_tokens(engine: &TernaryInferenceEngine, ascii: &str, max_tokens: usize) -> Vec<u32> {
    let char_budget = (max_tokens * 4).min(ascii.len());
    let mut sample = &ascii[..char_budget];
    let mut tokens = engine.tokenizer.encode(sample);
    while tokens.len() > max_tokens {
        // shrink proportionally, re-tokenize
        let keep = sample.len() * max_tokens / tokens.len();
        sample = &sample[..keep.min(sample.len())];
        tokens = engine.tokenizer.encode(sample);
    }
    tokens
}

/// CIS-1 E2: teacher-forced perplexity, integer path vs float path, on the
/// same --sample token window, same artifacts, same run. Prints both PPLs,
/// the relative delta against the preregistered kill line, and the FNV-1a 64
/// digest of the integer path's per-step argmax sequence (the determinism
/// exhibit: identical inputs must reproduce the identical digest).
///
/// Takes the engine BY VALUE: the passes run sequentially and the float
/// engine's working state (KV cache + batch arena, ~0.6 GB at BitNet-2B
/// scale) is dropped before the integer engine allocates its own KV cache.
/// `CisModel` borrows only the artifact bytes, so it outlives the engine.
fn run_cis(
    mut engine: TernaryInferenceEngine,
    ascii: &str,
    max_tokens: usize,
    dataset: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    /// E2 preregistered kill line: > +5% relative PPL increase kills
    /// CIS-1-as-full-integer.
    const KILL_LINE_PCT: f64 = 5.0;

    let tokens = sample_tokens(&engine, ascii, max_tokens);
    println!("Dataset: {} | sample: {} tokens", dataset, tokens.len());
    if tokens.len() < 64 {
        return Err("sample too short for a meaningful perplexity".into());
    }

    let t0 = Instant::now();
    let float_ppl = engine.calculate_perplexity(&tokens);
    let t_float = t0.elapsed().as_secs_f64();

    let vocab_len = engine.tokenizer.vocab_len();
    let model = aegis_core::cis_infer::CisModel::new(engine.pipeline(), &engine.config)
        .map_err(|e| format!("CIS model conversion: {e}"))?;
    // Free the float path's KV cache and arena before the integer pass
    // allocates its own (6 GB box, two 2B-scale engines do not coexist).
    drop(engine);
    let mut cis = aegis_core::cis_infer::CisEngine::new(&model);
    let t1 = Instant::now();
    let int_res = cis.calculate_perplexity_int(&tokens);
    let t_int = t1.elapsed().as_secs_f64();

    let delta_pct = (int_res.ppl - float_ppl) / float_ppl * 100.0;

    println!("--------------------------------------------------");
    println!("CIS-1 E2 — integer vs float, teacher-forced");
    println!(
        "float   PPL ({} scored tokens): {:.6}   [{:.1}s]",
        tokens.len() - 1,
        float_ppl,
        t_float
    );
    println!(
        "integer PPL ({} scored tokens): {:.6}   [{:.1}s]",
        int_res.scored, int_res.ppl, t_int
    );
    println!(
        "relative delta: {:+.4}%  (kill line: +{:.1}%)",
        delta_pct, KILL_LINE_PCT
    );
    if delta_pct.is_nan() {
        println!("E2 VERDICT: INVALID (NaN delta — target id out of vocab?)");
    } else if delta_pct > KILL_LINE_PCT {
        println!("E2 VERDICT: KILL (integer PPL exceeds the +{KILL_LINE_PCT}% line)");
    } else {
        println!("E2 VERDICT: PASS (within the +{KILL_LINE_PCT}% line)");
    }
    println!(
        "integer argmax digest (FNV-1a 64): 0x{:016X}",
        int_res.argmax_digest
    );
    println!("--------------------------------------------------");
    print_caveat(vocab_len);
    Ok(())
}

/// CIS-1 v0.3: THREE teacher-forced perplexities on the identical --sample
/// token window, same artifacts, same run — float engine, integer-dominant
/// hybrid (E2), and the FULL-INTEGER path (ROPE-I / SOFTMAX-I / ACT-I).
/// Prints each PPL, both integer paths' relative deltas vs float, both
/// FNV-1a 64 argmax digests, and the E2 kill-line verdict applied to the
/// full-integer path. The full-integer forward pass is pure scalar integer
/// code (no SIMD dispatch exists in it), so its digest is the
/// cross-path-identity exhibit as well as the determinism exhibit.
fn run_cis_full(
    engine: &mut TernaryInferenceEngine,
    ascii: &str,
    max_tokens: usize,
    dataset: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    /// E2 preregistered kill line, applied to the full-integer path too:
    /// > +5% relative PPL increase vs float kills CIS-1-as-full-integer.
    const KILL_LINE_PCT: f64 = 5.0;

    let tokens = sample_tokens(engine, ascii, max_tokens);
    println!("Dataset: {} | sample: {} tokens", dataset, tokens.len());
    if tokens.len() < 64 {
        return Err("sample too short for a meaningful perplexity".into());
    }

    let t0 = Instant::now();
    let float_ppl = engine.calculate_perplexity(&tokens);
    let t_float = t0.elapsed().as_secs_f64();

    let model = aegis_core::cis_infer::CisModel::new(engine.pipeline(), &engine.config)
        .map_err(|e| format!("CIS model conversion: {e}"))?;

    let mut hybrid = aegis_core::cis_infer::CisEngine::new_with_mode(
        &model,
        aegis_core::cis_infer::CisMode::Hybrid,
    );
    let t1 = Instant::now();
    let hyb = hybrid.calculate_perplexity_int(&tokens);
    let t_hyb = t1.elapsed().as_secs_f64();
    drop(hybrid);

    let mut full = aegis_core::cis_infer::CisEngine::new_with_mode(
        &model,
        aegis_core::cis_infer::CisMode::FullInt,
    );
    let t2 = Instant::now();
    let fi = full.calculate_perplexity_int(&tokens);
    let t_full = t2.elapsed().as_secs_f64();

    let d_hyb = (hyb.ppl - float_ppl) / float_ppl * 100.0;
    let d_full = (fi.ppl - float_ppl) / float_ppl * 100.0;

    println!("--------------------------------------------------");
    println!("CIS-1 v0.3 — float vs hybrid-int vs FULL-INTEGER, teacher-forced");
    println!(
        "float      PPL ({} scored tokens): {:.6}   [{:.1}s]",
        tokens.len() - 1,
        float_ppl,
        t_float
    );
    println!(
        "hybrid-int PPL ({} scored tokens): {:.6}   [{:.1}s]  delta vs float: {:+.4}%",
        hyb.scored, hyb.ppl, t_hyb, d_hyb
    );
    println!(
        "full-int   PPL ({} scored tokens): {:.6}   [{:.1}s]  delta vs float: {:+.4}%",
        fi.scored, fi.ppl, t_full, d_full
    );
    println!("kill line (full-int vs float): +{KILL_LINE_PCT:.1}%");
    if d_full.is_nan() {
        println!("FULL-INT VERDICT: INVALID (NaN delta — target id out of vocab?)");
    } else if d_full > KILL_LINE_PCT {
        println!("FULL-INT VERDICT: KILL (full-integer PPL exceeds the +{KILL_LINE_PCT}% line)");
    } else {
        println!("FULL-INT VERDICT: PASS (within the +{KILL_LINE_PCT}% line)");
    }
    println!(
        "hybrid-int argmax digest (FNV-1a 64): 0x{:016X}",
        hyb.argmax_digest
    );
    println!(
        "full-int   argmax digest (FNV-1a 64): 0x{:016X}",
        fi.argmax_digest
    );
    println!("--------------------------------------------------");
    print_caveat(engine.tokenizer.vocab_len());
    Ok(())
}

/// Original single-shot behavior, preserved verbatim as the regression anchor.
///
/// IMPORTANT: never feed the full document to encode() — the engine's BPE is
/// O(n^2) in input length (one merge applied per full rescan), fine for chat
/// prompts but days on megabyte inputs. Pre-truncate by characters first
/// (~4 chars/token is a safe overestimate for English).
fn run_sample(
    engine: &mut TernaryInferenceEngine,
    ascii: &str,
    max_tokens: usize,
    dataset: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let char_budget = (max_tokens * 4).min(ascii.len());
    let mut sample = &ascii[..char_budget];
    let mut tokens = engine.tokenizer.encode(sample);
    while tokens.len() > max_tokens {
        // shrink proportionally, re-tokenize
        let keep = sample.len() * max_tokens / tokens.len();
        sample = &sample[..keep.min(sample.len())];
        tokens = engine.tokenizer.encode(sample);
    }
    println!(
        "Dataset: {} | sample: {} chars -> {} tokens",
        dataset,
        sample.len(),
        tokens.len()
    );
    if tokens.len() < 64 {
        return Err("sample too short for a meaningful perplexity".into());
    }

    let t0 = Instant::now();
    let ppl = engine.calculate_perplexity(&tokens);
    let dt = t0.elapsed().as_secs_f64();

    println!("--------------------------------------------------");
    println!(
        "Perplexity (teacher-forced, {} tokens): {:.3}",
        tokens.len(),
        ppl
    );
    println!(
        "Eval wall time: {:.1}s ({:.2} tok/s)",
        dt,
        tokens.len() as f64 / dt
    );
    println!("--------------------------------------------------");
    print_caveat(engine.tokenizer.vocab_len());
    Ok(())
}

/// Chunked full-text PPL with artifact-mismatch tripwires.
fn run_chunked(
    engine: &mut TernaryInferenceEngine,
    ascii: &str,
    max_tokens: usize,
    dataset: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let char_chunk_size = 5000;
    let mut total_nll = 0.0f64;
    let mut total_tokens: usize = 0;
    let mut offset = 0;
    let mut first_chunk = true;
    // Must be exactly "1": AEGIS_EVAL_FORCE=0 (or empty) must NOT disarm the tripwires.
    let force = std::env::var("AEGIS_EVAL_FORCE").as_deref() == Ok("1");

    println!("Dataset: {} | total chars: {}", dataset, ascii.len());
    let t0 = Instant::now();

    while offset < ascii.len() {
        let end = (offset + char_chunk_size).min(ascii.len());
        let chunk = &ascii[offset..end];

        // Evaluate at most max_tokens per chunk to stay inside the KV window.
        let tokens = engine.tokenizer.encode(chunk);
        let eval_len = tokens.len().min(max_tokens);

        // Tripwire 1: near-char-level tokenization means the vocab has no
        // merges (wrong/stale vocab.bin) — every number after this is garbage.
        let chars_per_token = chunk.len() as f64 / tokens.len().max(1) as f64;
        if chars_per_token < 2.0 && !force {
            return Err(format!(
                "ABORT: {:.2} chars/token on a {}-char chunk (expect ~4 for English BPE). \
                 vocab.bin almost certainly lacks merges or does not match embed.bin/model. \
                 Set AEGIS_EVAL_FORCE=1 to override.",
                chars_per_token,
                chunk.len()
            )
            .into());
        }

        if eval_len > 1 {
            let eval_tokens = &tokens[..eval_len];
            let ppl = engine.calculate_perplexity(eval_tokens);
            let count = (eval_tokens.len() - 1) as f64;

            // Tripwire 2: an absurd first-chunk PPL is an artifact mismatch
            // (wrong model/embed/vocab pairing), not a measurement.
            if first_chunk && ppl > 100.0 && !force {
                return Err(format!(
                    "ABORT: first-chunk PPL {:.1} (>100). This is the signature of a \
                     mismatched model/embed/vocab triple, not a real measurement \
                     (known-good triple scores ~8-13 here). Verify the artifact paths. \
                     Set AEGIS_EVAL_FORCE=1 to override.",
                    ppl
                )
                .into());
            }
            first_chunk = false;

            if ppl > 0.0 && count > 0.0 {
                total_nll += ppl.ln() * count;
                total_tokens += eval_tokens.len() - 1;
                println!(
                    "Progress: {}/{} chars | chunk: {} tok ({:.1} c/t) PPL {:.3} | overall: {} tok PPL {:.3}",
                    end,
                    ascii.len(),
                    eval_tokens.len(),
                    chars_per_token,
                    ppl,
                    total_tokens,
                    (total_nll / total_tokens as f64).exp()
                );
            }
        }
        offset += char_chunk_size;
    }

    let ppl = if total_tokens > 0 {
        (total_nll / total_tokens as f64).exp()
    } else {
        0.0
    };
    let dt = t0.elapsed().as_secs_f64();

    println!("--------------------------------------------------");
    println!(
        "Perplexity (teacher-forced, {} tokens): {:.3}",
        total_tokens, ppl
    );
    println!(
        "Eval wall time: {:.1}s ({:.2} tok/s)",
        dt,
        total_tokens as f64 / dt
    );
    println!("--------------------------------------------------");
    println!(
        "Chunked mode: each {}–char chunk is teacher-forced from a cold KV cache;",
        char_chunk_size
    );
    println!("cross-chunk context is not carried, which biases PPL slightly upward.");
    print_caveat(engine.tokenizer.vocab_len());
    Ok(())
}

/// KV-isolation tripwire: calculate_perplexity relies on the engine's
/// write-before-read KV invariant instead of an explicit reset. If that
/// invariant ever breaks, a short measurement taken after a long one will
/// differ from the same measurement taken first. Assert bitwise equality.
fn run_selfcheck(
    engine: &mut TernaryInferenceEngine,
    ascii: &str,
    max_tokens: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let short = &ascii[..600.min(ascii.len())];
    let long_end = 4000.min(ascii.len());
    let long = &ascii[..long_end];

    let short_tokens = engine.tokenizer.encode(short);
    let long_tokens_full = engine.tokenizer.encode(long);
    let long_tokens = &long_tokens_full[..long_tokens_full.len().min(max_tokens)];
    if short_tokens.len() < 16 || long_tokens.len() <= short_tokens.len() {
        return Err("selfcheck needs a text file with at least ~4000 ASCII chars".into());
    }

    let fresh = engine.calculate_perplexity(&short_tokens);
    let _long = engine.calculate_perplexity(long_tokens);
    let after_long = engine.calculate_perplexity(&short_tokens);

    println!("short-first : {:.12}", fresh);
    println!("short-after : {:.12}", after_long);
    if fresh.to_bits() == after_long.to_bits() {
        println!("SELFCHECK PASS: KV write-before-read isolation holds (bitwise identical).");
        Ok(())
    } else {
        Err(format!(
            "SELFCHECK FAIL: short-text PPL changed after a long measurement \
             ({fresh:.12} -> {after_long:.12}). The KV write-before-read invariant \
             is broken; calculate_perplexity now needs an explicit cache reset."
        )
        .into())
    }
}

fn print_caveat(vocab_len: usize) {
    // Caveat must describe the artifact actually loaded, not assume the
    // BitNet pruned set — Falcon-family artifacts carry their full vocab.
    if vocab_len == 50_256 {
        println!("NOTE: pruned-vocab model (50,256 of 128,256 tokens; ASCII-oriented).");
    } else {
        println!(
            "NOTE: vocab {} tokens (full source vocab; cross-tokenizer PPLs are not comparable).",
            vocab_len
        );
    }
    println!("Report this number only alongside that caveat.");
}
