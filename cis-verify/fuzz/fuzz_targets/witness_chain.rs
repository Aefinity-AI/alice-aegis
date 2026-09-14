//! E19 fuzz target: `cis_verify::witness::{WitnessHeader, WitnessChain}`
//! against arbitrary byte streams. The construction (src/witness.rs) is
//! pure SHA-256 folding over caller-supplied fields with mixed endianness
//! that is load-bearing; this target exercises header hashing plus a
//! variable-length run of `fold_step` calls (variable prompt length,
//! variable per-step logit-vector length) looking for a panic, OOM, or
//! hang on adversarial field lengths/values.

#![no_main]

use cis_verify::witness::{WitnessChain, WitnessHeader};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Layout: [model_sha:32][embed_sha:32][vocab_sha:32][max_new:8][prompt_len:1][prompt][steps...]
    if data.len() < 32 + 32 + 32 + 8 + 1 {
        return;
    }
    let mut model_sha = [0u8; 32];
    let mut embed_sha = [0u8; 32];
    let mut vocab_sha = [0u8; 32];
    model_sha.copy_from_slice(&data[0..32]);
    embed_sha.copy_from_slice(&data[32..64]);
    vocab_sha.copy_from_slice(&data[64..96]);
    let max_new = u64::from_le_bytes(data[96..104].try_into().unwrap());
    let prompt_len_byte = data[104] as usize;
    let rest = &data[105..];

    let prompt_len = prompt_len_byte.min(rest.len());
    let (prompt, mut steps) = rest.split_at(prompt_len);

    let header = WitnessHeader {
        model_sha: &model_sha,
        embed_sha: &embed_sha,
        vocab_sha: &vocab_sha,
        max_new,
        prompt,
    };
    let mut chain = WitnessChain::from_header(&header);

    // Consume `steps` as a run of [token_id:4][n_logits:1][logits: n * 8 bytes].
    // Cap n_logits so one malformed input can't force unbounded work; a real
    // vocab-sized logit vector is thousands wide, but the fold construction
    // is length-agnostic so a small cap still exercises the same code path.
    let mut iterations = 0u32;
    while steps.len() >= 5 && iterations < 64 {
        iterations += 1;
        let token_id = u32::from_le_bytes(steps[0..4].try_into().unwrap());
        let n = (steps[4] as usize) % 33;
        steps = &steps[5..];
        if steps.len() < n * 8 {
            break;
        }
        let mut logits: std::vec::Vec<i64> = std::vec::Vec::with_capacity(n);
        for i in 0..n {
            let bytes: [u8; 8] = steps[i * 8..i * 8 + 8].try_into().unwrap();
            logits.push(i64::from_le_bytes(bytes));
        }
        chain.fold_step(token_id, &logits);
        steps = &steps[n * 8..];
    }

    let _ = chain.digest();
    let _ = chain.steps();
});
