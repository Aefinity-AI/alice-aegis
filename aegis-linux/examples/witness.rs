//! v0 inference witness — generate and verify a portable transcript of a
//! generation run. Identity/correctness tool only; never a perf instrument.
//!
//! DEMO SEMANTICS (v0): the chain commits to artifact hashes + prompt + the
//! emitted token text under the CURRENT float engine. E1
//! (docs/hardware_logs/e1_detprobe_crosspath_m7_2026-07-31.log) proved f32
//! results are only stable within one arithmetic path, so a v0 witness
//! verifies ONLY against the same path — and provably FAILS across paths.
//! That failure is the point: it is the measured hole CIS-1's integer
//! semantics closes (docs/CIS-1_SPEC_DRAFT_v0.1.md).
//!
//!   witness gen    <MODEL> <EMBED> <VOCAB> <maxtok> <prompt> [--scalar] > W
//!   witness verify <MODEL> <EMBED> <VOCAB> <witness-file> [--scalar]
//!
//! verify re-executes the generation and recomputes the chain; PASS iff the
//! final chain hash matches bit-for-bit.

use aegis_core::inference::TernaryInferenceEngine;
use aegis_core::ops::{active_path_name, set_force_scalar};
use std::fs;

// ---- minimal SHA-256 (FIPS 180-4), no dependencies ------------------------

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

struct Sha256 {
    h: [u32; 8],
    buf: Vec<u8>,
    len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Sha256 {
            h: [
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
            self.compress(&block);
            self.buf.drain(..64);
        }
    }
    fn compress(&mut self, block: &[u8; 64]) {
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
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, h].iter().enumerate() {
            self.h[i] = self.h[i].wrapping_add(*v);
        }
    }
    fn finish(mut self) -> [u8; 32] {
        let bitlen = self.len * 8;
        self.update(&[0x80]);
        while (self.len % 64) != 56 {
            self.update(&[0]);
        }
        // update() mutated self.len; write the ORIGINAL bit length.
        let mut block = [0u8; 8];
        block.copy_from_slice(&bitlen.to_be_bytes());
        self.len = 0;
        self.buf.extend_from_slice(&block);
        let b: [u8; 64] = self.buf[..64].try_into().unwrap();
        self.compress(&b);
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[i * 4..i * 4 + 4].copy_from_slice(&self.h[i].to_be_bytes());
        }
        out
    }
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(data);
    s.finish()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap_or(0))
        .collect()
}

// ---- witness ---------------------------------------------------------------

struct Run {
    ntok: usize,
    chain: [u8; 32],
}

fn execute(
    model: &[u8],
    emb: &[u8],
    vocab: &[u8],
    prompt: &str,
    maxtok: usize,
    header_hash: &[u8; 32],
) -> Run {
    let mut engine = TernaryInferenceEngine::new(emb, model, vocab).expect("engine init");
    let mut chain = *header_hash;
    let mut ntok = 0usize;
    let _ = engine.process_intent(prompt, maxtok, |t| {
        if !t.starts_with("[SYSTEM]") && !t.contains("[PERFORMANCE]") {
            let mut s = Sha256::new();
            s.update(&chain);
            s.update(t.as_bytes());
            chain = s.finish();
            ntok += 1;
        }
    });
    Run { ntok, chain }
}

fn header(
    model_sha: &[u8; 32],
    emb_sha: &[u8; 32],
    vocab_sha: &[u8; 32],
    prompt: &str,
    maxtok: usize,
) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(b"AEGIS-WITNESS v0-float-DEMO\n");
    s.update(model_sha);
    s.update(emb_sha);
    s.update(vocab_sha);
    s.update(&(maxtok as u64).to_be_bytes());
    s.update(&(prompt.len() as u64).to_be_bytes());
    s.update(prompt.as_bytes());
    s.finish()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scalar = args.iter().any(|a| a == "--scalar");
    set_force_scalar(scalar);
    let pos: Vec<&String> = args[1..].iter().filter(|a| *a != "--scalar").collect();

    if pos.len() < 5 {
        eprintln!("Usage: witness gen    <MODEL> <EMBED> <VOCAB> <maxtok> <prompt> [--scalar]");
        eprintln!("       witness verify <MODEL> <EMBED> <VOCAB> <witness-file> [--scalar]");
        std::process::exit(2);
    }
    let mode = pos[0].as_str();
    let model = fs::read(pos[1]).expect("model");
    let emb = fs::read(pos[2]).expect("embed");
    let vocab = fs::read(pos[3]).expect("vocab");
    let model_sha = sha256(&model);
    let emb_sha = sha256(&emb);
    let vocab_sha = sha256(&vocab);

    match mode {
        "gen" => {
            let maxtok: usize = pos[4].parse().expect("maxtok");
            let prompt = pos.get(5).map(|s| s.as_str()).unwrap_or("hello alice");
            let hh = header(&model_sha, &emb_sha, &vocab_sha, prompt, maxtok);
            let run = execute(&model, &emb, &vocab, prompt, maxtok, &hh);
            println!("AEGIS-WITNESS v0-float-DEMO");
            println!("model {}", hex(&model_sha));
            println!("embed {}", hex(&emb_sha));
            println!("vocab {}", hex(&vocab_sha));
            println!("arith {}", active_path_name());
            println!("maxtok {maxtok}");
            println!("prompt-hex {}", hex(prompt.as_bytes()));
            println!("ntok {}", run.ntok);
            println!("chain {}", hex(&run.chain));
        }
        "verify" => {
            let wtext = fs::read_to_string(pos[4]).expect("witness file");
            let mut model_w = String::new();
            let mut emb_w = String::new();
            let mut vocab_w = String::new();
            let mut arith_w = String::new();
            let mut maxtok = 0usize;
            let mut prompt = String::new();
            let mut ntok_w = 0usize;
            let mut chain_w = String::new();
            for line in wtext.lines() {
                let mut it = line.splitn(2, ' ');
                let (k, v) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
                match k {
                    "model" => model_w = v.into(),
                    "embed" => emb_w = v.into(),
                    "vocab" => vocab_w = v.into(),
                    "arith" => arith_w = v.into(),
                    "maxtok" => maxtok = v.parse().unwrap_or(0),
                    "prompt-hex" => prompt = String::from_utf8(unhex(v)).unwrap_or_default(),
                    "ntok" => ntok_w = v.parse().unwrap_or(0),
                    "chain" => chain_w = v.into(),
                    _ => {}
                }
            }
            let mut fail = false;
            if hex(&model_sha) != model_w {
                println!(
                    "FAIL artifact: MODEL hash mismatch (witness {} vs local {})",
                    &model_w[..16],
                    &hex(&model_sha)[..16]
                );
                fail = true;
            }
            if hex(&emb_sha) != emb_w {
                println!("FAIL artifact: EMBED hash mismatch");
                fail = true;
            }
            if hex(&vocab_sha) != vocab_w {
                println!("FAIL artifact: VOCAB hash mismatch");
                fail = true;
            }
            if fail {
                std::process::exit(1);
            }
            let hh = header(&model_sha, &emb_sha, &vocab_sha, &prompt, maxtok);
            let run = execute(&model, &emb, &vocab, &prompt, maxtok, &hh);
            let local = hex(&run.chain);
            println!(
                "witness arith: {arith_w} | verifier arith: {}",
                active_path_name()
            );
            println!("witness ntok {ntok_w} chain {}", &chain_w[..16]);
            println!("local   ntok {} chain {}", run.ntok, &local[..16]);
            if local == chain_w && run.ntok == ntok_w {
                println!("VERIFY PASS — re-execution reproduced the chain bit-for-bit");
            } else {
                println!("VERIFY FAIL — re-execution diverged from the witness");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}
