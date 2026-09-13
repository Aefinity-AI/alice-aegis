//! moat-3: single-bit-flip weight perturbation vs three verifiers.
//!
//! Reuses the SAME engine path as `cis_decode`/`cis_witness` (CisEngine,
//! CisMode::FullInt, WitnessChain) — no engine re-derivation. For each
//! trial: flip ONE bit in ONE F32 weight tensor byte (candidate list
//! precomputed offline by tools/moat3/f32_tensors.csv, since SafeTensors
//! exposes no public tensor-iteration API), re-run the pinned prompt for
//! max_new tokens, restore the original byte, and score:
//!   A = generated token ids identical to the clean baseline (all K steps)
//!   B = top-5 logit index SET identical to baseline at every step
//!   C = witness chain (full i64-logit SHA-256 fold, §12.1 format) identical
//!       to baseline chain
//!
//! Identity/correctness artifact ONLY (Rule A): no timing is printed.
//!
//! Usage:
//!   cis_bitflip_moat3 <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <F32_CSV> \
//!       <max_new> <prompt> <n_trials> <n_control_noflip> <n_control_highbit> <seed>

use aegis_core::cis_infer::{CisEngine, CisMode, CisModel, argmax_i64, fnv1a64};
use aegis_core::model::{FullBitNetPipeline, ModelConfig, SafeTensors};
use aegis_core::tokenizer::AegisTokenizer;
use aegis_core::witness::{WitnessChain, WitnessHeader, hex_lower};

fn hex(b: &[u8]) -> String {
    let mut out = vec![0u8; b.len() * 2];
    let n = hex_lower(b, &mut out);
    String::from_utf8(out[..n].to_vec()).unwrap()
}

struct RunResult {
    generated: Vec<u32>,
    top5: Vec<[u32; 5]>,
    chain: [u8; 32],
    fnv_digest: u64,
}

/// Same decode loop as cis_witness::replay, plus per-step top-5 capture.
fn run(
    model_bytes: &[u8],
    embed_bytes: &[u8],
    vocab_bytes: &[u8],
    prompt: &str,
    max_new: usize,
) -> RunResult {
    let tensors = SafeTensors::deserialize(model_bytes).expect("parse MODEL.SAF");
    let cfg_json = tensors
        .metadata_field("aegis_config")
        .expect("read __metadata__")
        .expect("MODEL.SAF carries no aegis_config");
    let config = ModelConfig::from_json(&cfg_json).expect("parse aegis_config");
    let pipeline =
        FullBitNetPipeline::new(&tensors, embed_bytes, &config).expect("build pipeline");
    let cis_model = CisModel::new(&pipeline, &config).expect("CIS model conversion");
    let mut engine = CisEngine::new_with_mode(&cis_model, CisMode::FullInt);

    let tokenizer = AegisTokenizer::new(vocab_bytes).expect("parse VOCAB.BIN");
    let prompt_ids = tokenizer.encode(prompt);
    assert!(!prompt_ids.is_empty(), "prompt tokenized to nothing");

    // Header fields are irrelevant to the chain's per-step logit folding
    // beyond binding max_new/prompt; use placeholders consistent across
    // baseline and perturbed runs (never compared to a receipt file here).
    let zero32 = [0u8; 32];
    let header = WitnessHeader {
        model_sha: &zero32,
        embed_sha: &zero32,
        vocab_sha: &zero32,
        max_new: max_new as u64,
        prompt: prompt.as_bytes(),
    };
    let mut chain = WitnessChain::from_header(&header);
    let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;

    let mut pos = 0usize;
    for &t in &prompt_ids {
        fnv = fnv1a64(fnv, &t.to_le_bytes());
        engine.forward_step_int(t, pos);
        pos += 1;
    }

    let mut generated = Vec::with_capacity(max_new);
    let mut top5 = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let (tok, top5_ids) = {
            let logits = engine.decode_logits();
            let t = argmax_i64(logits);
            chain.fold_step(t, logits);
            let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
            idx.sort_unstable_by(|&a, &b| logits[b as usize].cmp(&logits[a as usize]));
            let mut top5_ids = [0u32; 5];
            top5_ids.copy_from_slice(&idx[..5]);
            (t, top5_ids)
        };
        fnv = fnv1a64(fnv, &tok.to_le_bytes());
        generated.push(tok);
        top5.push(top5_ids);
        engine.forward_step_int(tok, pos);
        pos += 1;
    }

    RunResult {
        generated,
        top5,
        chain: chain.digest(),
        fnv_digest: fnv,
    }
}

/// xorshift64* — small deterministic PRNG so the whole trial plan is
/// reproducible from one printed seed (no external RNG crate needed).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 11 {
        eprintln!(
            "usage: cis_bitflip_moat3 <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <F32_CSV> <max_new> <prompt> <n_trials> <n_control_noflip> <n_control_highbit> <seed>"
        );
        std::process::exit(2);
    }
    let model_path = &args[1];
    let embed_path = &args[2];
    let vocab_path = &args[3];
    let csv_path = &args[4];
    let max_new: usize = args[5].parse().expect("max_new");
    let prompt = &args[6];
    let n_trials: usize = args[7].parse().expect("n_trials");
    let n_ctrl_noflip: usize = args[8].parse().expect("n_control_noflip");
    let n_ctrl_highbit: usize = args[9].parse().expect("n_control_highbit");
    let seed: u64 = args[10].parse().expect("seed");

    let mut model_bytes = std::fs::read(model_path).expect("read MODEL.SAF");
    let embed_bytes = std::fs::read(embed_path).expect("read EMBED.BIN");
    let vocab_bytes = std::fs::read(vocab_path).expect("read VOCAB.BIN");

    // Load the F32 tensor candidate ranges (name, abs_start, abs_end).
    let csv = std::fs::read_to_string(csv_path).expect("read F32_CSV");
    let mut ranges: Vec<(String, usize, usize)> = Vec::new();
    for line in csv.lines().skip(1) {
        let mut it = line.splitn(3, ',');
        let name = it.next().unwrap().to_string();
        let s: usize = it.next().unwrap().parse().unwrap();
        let e: usize = it.next().unwrap().parse().unwrap();
        ranges.push((name, s, e));
    }
    let total_bytes: usize = ranges.iter().map(|(_, s, e)| e - s).sum();

    println!("moat3: {} F32 candidate tensors, {} candidate bytes, model {} bytes", ranges.len(), total_bytes, model_bytes.len());

    println!("== baseline (clean, no flip) ==");
    let baseline = run(&model_bytes, &embed_bytes, &vocab_bytes, prompt, max_new);
    println!(
        "baseline: generated={:?} fnv={:016x} chain={}",
        baseline.generated,
        baseline.fnv_digest,
        hex(&baseline.chain)
    );

    // Some perturbed weights (especially high-order bit flips near a norm
    // gain) can push an intermediate value outside the engine's fixed-point
    // range, which the engine reports as a `panic!` (an assert), not a
    // `Result`. That is itself a detection event -- score it as A/B FAIL
    // (no valid generation to compare) and C FAIL (no valid chain either).
    // Silence the default panic handler's stderr spam for these expected,
    // caught panics; restore the perturbed byte unconditionally.
    std::panic::set_hook(Box::new(|_| {}));

    let mut rng = Rng(seed ^ 0x9E3779B97F4A7C15);

    // trial_kind: "control-noflip" | "control-highbit" | "trial"
    let mut plan: Vec<&str> = Vec::new();
    for _ in 0..n_ctrl_noflip {
        plan.push("control-noflip");
    }
    for _ in 0..n_ctrl_highbit {
        plan.push("control-highbit");
    }
    for _ in 0..n_trials {
        plan.push("trial");
    }

    println!("== trials ==");
    println!("trial#,kind,tensor,byte_off_in_tensor,bit,old_byte,new_byte,A_pass,B_pass,C_pass");

    let mut a_pass_c_fail = 0usize;
    let mut b_pass_c_fail = 0usize;
    let mut n = 0usize;

    for kind in plan {
        n += 1;
        let (result, tensor_name, byte_off, bit, old_byte, new_byte) = if kind == "control-noflip"
        {
            let r = run(&model_bytes, &embed_bytes, &vocab_bytes, prompt, max_new);
            (Some(r), "-".to_string(), 0usize, 0u8, 0u8, 0u8)
        } else {
            // pick a random tensor weighted by byte length, then a random
            // byte within it, then a random bit (low-mantissa 0..=9 for
            // "trial", high-order 23 or 30 or 31 for "control-highbit").
            let target = rng.below(total_bytes as u64) as usize;
            let mut acc = 0usize;
            let mut chosen = &ranges[0];
            for r in &ranges {
                let len = r.2 - r.1;
                if target < acc + len {
                    chosen = r;
                    break;
                }
                acc += len;
            }
            let byte_in_tensor = target - acc;
            let abs_off = chosen.1 + byte_in_tensor;
            // f32 little-endian: byte 0..2 = mantissa low/mid, byte 3 bits
            // 0..6 = mantissa high + exponent low bit, bit 7 = exponent top
            // region continues into byte3/ high bits; sign is byte3 bit7.
            // Restrict to byte 0 (pure low mantissa, bits 0..7) for "trial";
            // use byte 3 bit 7 (sign) for "control-highbit".
            let (flip_byte_idx, bit) = if kind == "control-highbit" {
                (3usize, 7u8) // sign bit
            } else {
                (0usize, (rng.below(8) as u8)) // low mantissa byte, any bit within it
            };
            let flip_abs = abs_off - (abs_off % 4) + flip_byte_idx;
            let old = model_bytes[flip_abs];
            let new = old ^ (1u8 << bit);
            model_bytes[flip_abs] = new;
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(&model_bytes, &embed_bytes, &vocab_bytes, prompt, max_new)
            }));
            model_bytes[flip_abs] = old; // restore unconditionally, even on panic
            match caught {
                Ok(r) => (
                    Some(r),
                    chosen.0.clone(),
                    flip_abs - chosen.1,
                    bit,
                    old,
                    new,
                ),
                Err(_) => (None, chosen.0.clone(), flip_abs - chosen.1, bit, old, new),
            }
        };

        let (a_pass, b_pass, c_pass, panicked) = match &result {
            Some(r) => (
                r.generated == baseline.generated,
                r.top5 == baseline.top5,
                r.chain == baseline.chain,
                false,
            ),
            None => (false, false, false, true),
        };

        if kind != "control-noflip" {
            if a_pass && !c_pass {
                a_pass_c_fail += 1;
            }
            if b_pass && !c_pass {
                b_pass_c_fail += 1;
            }
        }

        println!(
            "{},{},{},{},{},{:02x},{:02x},{},{},{}{}",
            n,
            kind,
            tensor_name,
            byte_off,
            bit,
            old_byte,
            new_byte,
            if a_pass { "PASS" } else { "FAIL" },
            if b_pass { "PASS" } else { "FAIL" },
            if c_pass { "PASS" } else { "FAIL" },
            if panicked { " (engine panicked: fixed-point range assert, caught)" } else { "" },
        );
    }

    println!("== summary ==");
    println!("headline A_pass_AND_C_fail={}", a_pass_c_fail);
    println!("headline B_pass_AND_C_fail={}", b_pass_c_fail);
    println!(
        "model_bytes_unchanged_after_run={}",
        model_bytes == std::fs::read(model_path).expect("reread MODEL.SAF")
    );
}
