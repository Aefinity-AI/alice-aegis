//! `active_set_digest` — Leg C1 (2026-08-29 pre-reg,
//! state/reports/2026-08-29-COMPACT-AXIS-PREREG.md, alice-aegis private
//! branch cm/c1-sparsity-determinism).
//!
//! For a fixed prompt set, runs the CIS-1 FullInt decode
//! (`aegis_core::cis_infer`, `CisMode::FullInt`) and, at every (step, layer),
//! records the sorted set of nonzero indices in the down_proj input (the
//! ReLU² active-neuron set a column-skip kernel would consume — see
//! `cis_infer::CisEngine::active_sets`, `active_set_digest` feature). All
//! index lists across every prompt/step/layer are folded, in chronological
//! push order, into one SHA-256 digest for the run.
//!
//! Also reproduces the A36 `CIS_DECODE` digest (prompt "Once upon a time",
//! max_new=64) inline, via the same fnv1a64 fold `cis_decode` uses, so gate
//! (c) — decode digest unchanged — is checked in the same binary/run as the
//! active-set digest.
//!
//! `--force-scalar` (or env `AEGIS_FORCE_SCALAR=1`) selects the scalar path
//! (`cis::ternary_matvec_i8` only, via each accelerated kernel's own race
//! toggle); default is the fast path (AVX2 on x86_64 / NEON on aarch64,
//! self-falling-back to scalar per-call where the kernel's own contract
//! requires it — see `cis_avx2`/`cis_neon`).
//!
//! Identity/correctness artifact ONLY (Rule A): no timing is printed.
//!
//! Run: active_set_digest <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <PROMPTS.txt> <steps> [--force-scalar]

use aegis_core::cis_infer::{CisEngine, CisMode, CisModel, argmax_i64, fnv1a64};
use aegis_core::model::{FullBitNetPipeline, ModelConfig, SafeTensors};
use aegis_core::tokenizer::AegisTokenizer;

// ---------------------------------------------------------------------------
// Minimal self-contained SHA-256 (FIPS 180-4). Kept out of aegis-core
// (no_std, dependency-minimal by house convention — see its Cargo.toml
// preamble) since this digest is a std-only diagnostic example.
// ---------------------------------------------------------------------------
struct Sha256 {
    state: [u32; 8],
    buf: Vec<u8>,
    len: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: Vec::new(),
            len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.len += data.len() as u64;
        self.buf.extend_from_slice(data);
        while self.buf.len() >= 64 {
            let block: [u8; 64] = self.buf[..64].try_into().unwrap();
            self.process(&block);
            self.buf.drain(..64);
        }
    }

    fn process(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = self.state;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for i in 0..8 {
            self.state[i] = self.state[i].wrapping_add(v[i]);
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.len * 8;
        self.buf.push(0x80);
        while self.buf.len() % 64 != 56 {
            self.buf.push(0);
        }
        self.buf.extend_from_slice(&bit_len.to_be_bytes());
        while self.buf.len() >= 64 {
            let block: [u8; 64] = self.buf[..64].try_into().unwrap();
            self.process(&block);
            self.buf.drain(..64);
        }
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[i * 4..i * 4 + 4].copy_from_slice(&self.state[i].to_be_bytes());
        }
        out
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[cfg(target_arch = "x86_64")]
fn set_force_scalar(v: bool) {
    aegis_core::ops::set_force_scalar(v);
}
#[cfg(target_arch = "aarch64")]
fn set_force_scalar(v: bool) {
    aegis_core::cis_neon::set_force_scalar(v);
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn set_force_scalar(_v: bool) {}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "usage: active_set_digest <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <PROMPTS.txt> <steps> [--force-scalar]"
        );
        std::process::exit(2);
    }
    let steps: usize = args[5].parse().expect("steps must be a number");
    let force_scalar = args.iter().any(|a| a == "--force-scalar")
        || std::env::var("AEGIS_FORCE_SCALAR").as_deref() == Ok("1");
    set_force_scalar(force_scalar);
    let path_label = if force_scalar { "scalar" } else { "fast" };

    let model_bytes = std::fs::read(&args[1]).expect("read MODEL.SAF");
    let embed_bytes = std::fs::read(&args[2]).expect("read EMBED.BIN");
    let vocab_bytes = std::fs::read(&args[3]).expect("read VOCAB.BIN");
    let prompts_raw = std::fs::read_to_string(&args[4]).expect("read PROMPTS.txt");
    let prompts: Vec<&str> = prompts_raw.lines().filter(|l| !l.is_empty()).collect();
    assert!(!prompts.is_empty(), "prompt file is empty");

    let tensors = SafeTensors::deserialize(&model_bytes).expect("parse MODEL.SAF");
    let cfg_json = tensors
        .metadata_field("aegis_config")
        .expect("read __metadata__")
        .expect("MODEL.SAF carries no aegis_config");
    let config = ModelConfig::from_json(&cfg_json).expect("parse aegis_config");
    let pipeline =
        FullBitNetPipeline::new(&tensors, &embed_bytes, &config).expect("build pipeline");
    let cis_model = CisModel::new(&pipeline, &config).expect("CIS model conversion");
    let tokenizer = AegisTokenizer::new(&vocab_bytes).expect("parse VOCAB.BIN");

    let mut hasher = Sha256::new();
    let mut total_steps: usize = 0;
    let n_layers = config.num_hidden_layers;

    for prompt in &prompts {
        let mut engine = CisEngine::new_with_mode(&cis_model, CisMode::FullInt);
        let prompt_ids = tokenizer.encode(prompt);
        assert!(
            !prompt_ids.is_empty(),
            "prompt tokenized to nothing: {prompt:?}"
        );
        let mut pos = 0usize;
        for &t in &prompt_ids {
            engine.forward_step_int(t, pos);
            pos += 1;
            total_steps += 1;
        }
        let mut current = argmax_i64(engine.decode_logits());
        for _ in 0..steps {
            engine.forward_step_int(current, pos);
            pos += 1;
            total_steps += 1;
            current = argmax_i64(engine.decode_logits());
        }
        // Fold every (step, layer) active set from this prompt's run, in
        // chronological push order, into the running per-run hasher.
        for (layer_idx, idxs) in engine.active_sets.drain(..) {
            let mut per_layer = Sha256::new();
            for i in &idxs {
                per_layer.update(&i.to_le_bytes());
            }
            hasher.update(&per_layer.finalize());
            debug_assert!(layer_idx < n_layers);
        }
    }

    let digest = hasher.finalize();
    println!(
        "ACTIVE_SET digest={} layers={} steps={} path={}",
        hex(&digest),
        n_layers,
        total_steps,
        path_label
    );

    // Gate (c): the A36 prompt's CIS_DECODE digest must be unchanged under
    // this run's path (fast or scalar).
    let mut engine = CisEngine::new_with_mode(&cis_model, CisMode::FullInt);
    let a36_prompt = "Once upon a time";
    let a36_max_new = 64usize;
    let prompt_ids = tokenizer.encode(a36_prompt);
    let mut cis_digest: u64 = 0xcbf2_9ce4_8422_2325;
    let mut pos = 0usize;
    for &t in &prompt_ids {
        cis_digest = fnv1a64(cis_digest, &t.to_le_bytes());
        engine.forward_step_int(t, pos);
        pos += 1;
    }
    let mut generated = Vec::with_capacity(a36_max_new);
    let mut current = argmax_i64(engine.decode_logits());
    for _ in 0..a36_max_new {
        cis_digest = fnv1a64(cis_digest, &current.to_le_bytes());
        generated.push(current);
        engine.forward_step_int(current, pos);
        pos += 1;
        current = argmax_i64(engine.decode_logits());
    }
    println!(
        "CIS_DECODE digest={:016x} prompt_toks={} gen_toks={} mode=fullint path={}",
        cis_digest,
        prompt_ids.len(),
        generated.len(),
        path_label
    );
}
