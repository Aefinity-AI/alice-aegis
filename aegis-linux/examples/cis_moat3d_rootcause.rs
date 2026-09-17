//! moat-3d: deterministic targeted re-run of the 4 exact B_pass&C_fail rows
//! found in moat-3c (state/legs/aefinity-box/pace-b1-5 raw log). Unlike
//! cis_bitflip_moat3b, this does NOT use the RNG to pick a corruption site --
//! it takes the exact (tensor_name, abs_byte_off_in_tensor, group) triple
//! for one of the 4 known miss rows and runs it 3x clean + 3x corrupted,
//! printing: witness chain (C), top-5-set digest fingerprint (B, via a
//! cheap hash of the top5 arrays so we can compare across runs), argmax
//! token ids (A), AND the decoded text.
//!
//! Usage:
//!   cis_moat3d_rootcause <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <ranges_csv> \
//!       <tensor_name> <byte_in_tensor> <group> <max_new> <prompt>
//!
//! ranges_csv is only used to resolve tensor_name -> its absolute file
//! start offset (same CSVs moat-3b/3c used), so byte_in_tensor (as printed
//! in the RESULT logs) can be turned back into an absolute file offset.

use aegis_core::cis_infer::{CisEngine, CisMode, CisModel, argmax_i64};
use aegis_core::model::{FullBitNetPipeline, ModelConfig, SafeTensors};
use aegis_core::tokenizer::AegisTokenizer;
use aegis_core::witness::{Sha256, WitnessChain, WitnessHeader, hex_lower};

fn hex(b: &[u8]) -> String {
    let mut out = vec![0u8; b.len() * 2];
    let n = hex_lower(b, &mut out);
    String::from_utf8(out[..n].to_vec()).unwrap()
}

fn run(
    model_bytes: &[u8],
    embed_bytes: &[u8],
    vocab_bytes: &[u8],
    prompt: &str,
    max_new: usize,
) -> (Vec<u32>, [u8; 32], String, String) {
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
    let mut top5_hasher = Sha256::new();
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
        for id in top5_ids {
            top5_hasher.update(&id.to_le_bytes());
        }
        generated.push(tok);
        engine.forward_step_int(tok, pos);
        pos += 1;
    }
    let top5_digest = hex(&top5_hasher.finalize());
    let text = tokenizer.decode(&generated);

    (generated, chain.digest(), top5_digest, text)
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
    if args.len() < 10 {
        eprintln!(
            "usage: cis_moat3d_rootcause <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <ranges_csv> <tensor_name> <byte_in_tensor> <group> <max_new> <prompt>"
        );
        std::process::exit(2);
    }
    let model_path = &args[1];
    let embed_path = &args[2];
    let vocab_path = &args[3];
    let ranges_path = &args[4];
    let tensor_name = &args[5];
    let byte_in_tensor: usize = args[6].parse().unwrap();
    let group: u8 = args[7].parse().unwrap();
    let max_new: usize = args[8].parse().unwrap();
    let prompt = &args[9];

    let mut model_bytes = std::fs::read(model_path).expect("read MODEL.SAF");
    let embed_bytes = std::fs::read(embed_path).expect("read EMBED.BIN");
    let vocab_bytes = std::fs::read(vocab_path).expect("read VOCAB.BIN");

    let ranges = load_ranges(ranges_path);
    let tensor_start = ranges
        .iter()
        .find(|(n, _, _)| n == tensor_name)
        .unwrap_or_else(|| panic!("tensor {} not found in {}", tensor_name, ranges_path))
        .1;
    let abs_off = tensor_start + byte_in_tensor;

    let shift = group * 2;
    let old = model_bytes[abs_off];
    let code = (old >> shift) & 0b11;
    let new_code = neighbour_code(code);
    let new = (old & !(0b11 << shift)) | (new_code << shift);

    println!(
        "target: tensor={} byte_in_tensor={} abs_off={} group={} old_byte={:02x} new_byte={:02x} old_code={} new_code={}",
        tensor_name, byte_in_tensor, abs_off, group, old, new, code, new_code
    );

    for i in 1..=3 {
        let (toks, chain, top5, text) = run(&model_bytes, &embed_bytes, &vocab_bytes, prompt, max_new);
        println!(
            "CLEAN[{}]: chain={} top5digest={} tokens={:?}",
            i, hex(&chain), top5, toks
        );
        println!("CLEAN[{}]: text={:?}", i, text);
    }

    model_bytes[abs_off] = new;
    for i in 1..=3 {
        let (toks, chain, top5, text) = run(&model_bytes, &embed_bytes, &vocab_bytes, prompt, max_new);
        println!(
            "CORRUPT[{}]: chain={} top5digest={} tokens={:?}",
            i, hex(&chain), top5, toks
        );
        println!("CORRUPT[{}]: text={:?}", i, text);
    }
    model_bytes[abs_off] = old;
    assert_eq!(
        model_bytes,
        std::fs::read(model_path).expect("reread MODEL.SAF"),
        "model bytes not restored!"
    );
    println!("model_bytes_unchanged_after_run=true");
}
