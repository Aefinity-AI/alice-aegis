//! moat-3b: stratified single-corruption experiment vs three verifiers.
//!
//! Re-scope of moat-3 (see state/QUEUE.md "moat-3b RE-SCOPED", 2026-09-13):
//! the checkpoint has tied embeddings (no distinct lm_head), no F32 attn/ffn
//! weight matrices (real weights are U8 ternary-packed; only 4-byte
//! `weight_scale` f32 scalars per attn/ffn tensor), and a bf16 embed.bin.
//! So the 5 classes actually present on this checkpoint are:
//!   EMBED-bf16   flip a low mantissa bit (bits 0-2, or swept 3-5) of one
//!                bf16 entry in EMBED.BIN.
//!   SCALE-attn   flip a low mantissa bit of one f32 weight_scale scalar
//!                belonging to a self_attn.{q,k,v,o}_proj tensor.
//!   SCALE-ffn    same, for a mlp.{gate,up,down}_proj tensor.
//!   TRIT-attn    change ONE packed ternary weight code to a neighbouring
//!                value in a self_attn.*.weight U8 tensor.
//!   TRIT-ffn     same, for a mlp.*.weight U8 tensor.
//!
//! Byte->trit packing (from aegis-core/src/ops.rs UNPACK_LUT / build_unpack_lut,
//! verified by reading that source before writing this file): each U8 byte
//! packs 4 ternary weights, 2 bits each, LSB-first: bits[1:0] = weight 0,
//! bits[3:2] = weight 1, bits[5:4] = weight 2, bits[7:6] = weight 3. Code
//! 00 = 0.0, 01 = +1.0, 10 = -1.0, 11 = undefined (also decodes to 0.0,
//! "degrades gracefully"). TRIT-attn/TRIT-ffn pick a random byte in the
//! chosen tensor's data range, a random 2-bit group within it, and remap
//! its code to a *neighbouring* value on the {-1, 0, +1} number line:
//! 00(0) -> 01(+1), 01(+1) -> 00(0), 10(-1) -> 00(0), 11(undef) -> 00(0).
//!
//! Scoring, per trial, exactly as moat-3:
//!   A = generated token ids identical to the clean baseline (all K steps)
//!   B = top-5 logit index SET identical to baseline at every step
//!   C = witness chain (full i64-logit SHA-256 fold, section 12.1 format)
//!       identical to baseline chain
//!
//! Identity/correctness artifact ONLY (Rule A): no timing is printed.
//!
//! Usage:
//!   cis_bitflip_moat3b <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <class> <ranges_csv|-> \
//!       <bit_lo> <bit_hi> <max_new> <prompt> <n_trials> <seed> [--sweep]
//!
//! class in {EMBED, SCALE_ATTN, SCALE_FFN, TRIT_ATTN, TRIT_FFN, CONTROL}.
//! ranges_csv is "-" for EMBED and CONTROL (no tensor pool needed).
//! With --sweep, each of bit_lo..=bit_hi is used for exactly n_trials
//! trials each (instead of one random bit per trial drawn from the range);
//! ignored for TRIT_* classes (no bit concept there).

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
}

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

    let zero32 = [0u8; 32];
    let header = WitnessHeader {
        model_sha: &zero32,
        embed_sha: &zero32,
        vocab_sha: &zero32,
        max_new: max_new as u64,
        prompt: prompt.as_bytes(),
    };
    let mut chain = WitnessChain::from_header(&header);

    let mut pos = 0usize;
    for &t in &prompt_ids {
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
        generated.push(tok);
        top5.push(top5_ids);
        engine.forward_step_int(tok, pos);
        pos += 1;
    }

    RunResult {
        generated,
        top5,
        chain: chain.digest(),
    }
}

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

fn load_ranges(path: &str) -> Vec<(String, usize, usize)> {
    let csv = std::fs::read_to_string(path).expect("read ranges csv");
    let mut ranges = Vec::new();
    for line in csv.lines().skip(1) {
        let mut it = line.splitn(3, ',');
        let name = it.next().unwrap().to_string();
        let s: usize = it.next().unwrap().parse().unwrap();
        let e: usize = it.next().unwrap().parse().unwrap();
        ranges.push((name, s, e));
    }
    ranges
}

fn pick_weighted<'a>(rng: &mut Rng, ranges: &'a [(String, usize, usize)]) -> (&'a (String, usize, usize), usize) {
    let total: usize = ranges.iter().map(|(_, s, e)| e - s).sum();
    let target = rng.below(total as u64) as usize;
    let mut acc = 0usize;
    for r in ranges {
        let len = r.2 - r.1;
        if target < acc + len {
            return (r, target - acc);
        }
        acc += len;
    }
    (&ranges[ranges.len() - 1], 0)
}

/// neighbour remap for a 2-bit ternary code: 00->01, 01->00, 10->00, 11->00
fn neighbour_code(c: u8) -> u8 {
    match c {
        0 => 1,
        1 => 0,
        2 => 0,
        _ => 0,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 12 {
        eprintln!(
            "usage: cis_bitflip_moat3b <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <class> <ranges_csv|-> <bit_lo> <bit_hi> <max_new> <prompt> <n_trials> <seed> [--sweep]"
        );
        std::process::exit(2);
    }
    let model_path = &args[1];
    let embed_path = &args[2];
    let vocab_path = &args[3];
    let class = args[4].as_str();
    let ranges_path = &args[5];
    let bit_lo: u8 = args[6].parse().unwrap();
    let bit_hi: u8 = args[7].parse().unwrap();
    let max_new: usize = args[8].parse().unwrap();
    let prompt = &args[9];
    let n_trials: usize = args[10].parse().unwrap();
    let seed: u64 = args[11].parse().unwrap();
    let sweep = args.get(12).map(|s| s == "--sweep").unwrap_or(false);

    let mut model_bytes = std::fs::read(model_path).expect("read MODEL.SAF");
    let mut embed_bytes = std::fs::read(embed_path).expect("read EMBED.BIN");
    let vocab_bytes = std::fs::read(vocab_path).expect("read VOCAB.BIN");

    let ranges = if ranges_path == "-" {
        Vec::new()
    } else {
        load_ranges(ranges_path)
    };

    println!("moat3b: class={} ranges={} bit_lo={} bit_hi={} n_trials={} sweep={}", class, ranges_path, bit_lo, bit_hi, n_trials, sweep);

    let baseline = run(&model_bytes, &embed_bytes, &vocab_bytes, prompt, max_new);
    println!("baseline: chain={} generated={:?}", hex(&baseline.chain), baseline.generated);

    std::panic::set_hook(Box::new(|_| {}));
    let mut rng = Rng(seed ^ 0x9E3779B97F4A7C15);

    // Build the plan: (bit-for-scale/embed, trial-index) tuples.
    let mut plan: Vec<u8> = Vec::new();
    if class == "CONTROL" {
        for _ in 0..n_trials {
            plan.push(0);
        }
    } else if sweep {
        for b in bit_lo..=bit_hi {
            for _ in 0..n_trials {
                plan.push(b);
            }
        }
    } else {
        for _ in 0..n_trials {
            plan.push(0); // bit chosen per-trial below within [bit_lo,bit_hi]
        }
    }

    println!("== trials ==");
    println!("trial#,class,tensor,byte_off,bit,old,new,A_pass,B_pass,C_pass");

    let mut a_pass_c_fail = 0usize;
    let mut b_pass_c_fail = 0usize;
    let mut n = 0usize;

    for planned_bit in &plan {
        n += 1;
        let (result, tensor_name, byte_off, bit, old_v, new_v) = match class {
            "CONTROL" => {
                let r = run(&model_bytes, &embed_bytes, &vocab_bytes, prompt, max_new);
                (Some(r), "-".to_string(), 0usize, 0u8, 0u16, 0u16)
            }
            "EMBED" => {
                let n_elems = embed_bytes.len() / 2;
                let elem = rng.below(n_elems as u64) as usize;
                let off = elem * 2;
                let bit = if sweep { *planned_bit } else { bit_lo + (rng.below((bit_hi - bit_lo + 1) as u64) as u8) };
                let old = u16::from_le_bytes([embed_bytes[off], embed_bytes[off + 1]]);
                let new = old ^ (1u16 << bit);
                let nb = new.to_le_bytes();
                embed_bytes[off] = nb[0];
                embed_bytes[off + 1] = nb[1];
                let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(&model_bytes, &embed_bytes, &vocab_bytes, prompt, max_new)
                }));
                embed_bytes[off] = (old & 0xff) as u8;
                embed_bytes[off + 1] = (old >> 8) as u8;
                match caught {
                    Ok(r) => (Some(r), "embed.bin".to_string(), off, bit, old, new),
                    Err(_) => (None, "embed.bin".to_string(), off, bit, old, new),
                }
            }
            "SCALE_ATTN" | "SCALE_FFN" => {
                let (chosen, _byte_in_tensor) = pick_weighted(&mut rng, &ranges);
                // Each weight_scale tensor IS exactly one f32 (4 bytes);
                // chosen.1 is its abs file offset, byte 0 = pure low
                // mantissa (little-endian). Do not re-align to the file's
                // own 4-byte grid -- the tensor need not start on it.
                let flip_abs = chosen.1;
                let bit = if sweep { *planned_bit } else { bit_lo + (rng.below((bit_hi - bit_lo + 1) as u64) as u8) };
                let old = model_bytes[flip_abs];
                let new = old ^ (1u8 << bit);
                model_bytes[flip_abs] = new;
                let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(&model_bytes, &embed_bytes, &vocab_bytes, prompt, max_new)
                }));
                model_bytes[flip_abs] = old;
                match caught {
                    Ok(r) => (Some(r), chosen.0.clone(), flip_abs - chosen.1, bit, old as u16, new as u16),
                    Err(_) => (None, chosen.0.clone(), flip_abs - chosen.1, bit, old as u16, new as u16),
                }
            }
            "TRIT_ATTN" | "TRIT_FFN" => {
                let (chosen, byte_in_tensor) = pick_weighted(&mut rng, &ranges);
                let abs_off = chosen.1 + byte_in_tensor;
                let group = rng.below(4) as u8; // which of the 4 packed weights
                let shift = group * 2;
                let old = model_bytes[abs_off];
                let code = (old >> shift) & 0b11;
                let new_code = neighbour_code(code);
                let new = (old & !(0b11 << shift)) | (new_code << shift);
                model_bytes[abs_off] = new;
                let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(&model_bytes, &embed_bytes, &vocab_bytes, prompt, max_new)
                }));
                model_bytes[abs_off] = old;
                match caught {
                    Ok(r) => (Some(r), chosen.0.clone(), abs_off - chosen.1, group, old as u16, new as u16),
                    Err(_) => (None, chosen.0.clone(), abs_off - chosen.1, group, old as u16, new as u16),
                }
            }
            other => panic!("unknown class {}", other),
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

        if class != "CONTROL" {
            if a_pass && !c_pass {
                a_pass_c_fail += 1;
            }
            if b_pass && !c_pass {
                b_pass_c_fail += 1;
            }
        }

        println!(
            "{},{},{},{},{},{:04x},{:04x},{},{},{}{}",
            n,
            class,
            tensor_name,
            byte_off,
            bit,
            old_v,
            new_v,
            if a_pass { "PASS" } else { "FAIL" },
            if b_pass { "PASS" } else { "FAIL" },
            if c_pass { "PASS" } else { "FAIL" },
            if panicked { " (engine panicked: fixed-point range assert, caught)" } else { "" },
        );
    }

    println!("== summary ==");
    println!("class={} n={} A_pass_AND_C_fail={} B_pass_AND_C_fail={}", class, n, a_pass_c_fail, b_pass_c_fail);
    println!(
        "model_bytes_unchanged_after_run={} embed_bytes_unchanged_after_run={}",
        model_bytes == std::fs::read(model_path).expect("reread MODEL.SAF"),
        embed_bytes == std::fs::read(embed_path).expect("reread EMBED.BIN"),
    );
}
