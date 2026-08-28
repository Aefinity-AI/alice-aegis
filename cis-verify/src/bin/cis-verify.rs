//! `cis-verify` — the `std`-feature CLI binary (builder task 6,
//! `docs/design/CIS_VERIFY_DESIGN.md` §3.4/§6.2 item 6). File I/O, argv,
//! stdout only — every byte of actual verification logic lives in
//! `cis_verify::verify`, which is `no_std`+`alloc`.
//!
//! ```text
//! cis-verify <receipt> <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN>
//! ```
//!
//! Prints one line per check, then exactly one of:
//!   VERIFY PASS
//!   VERIFY FAIL (<field>)
//! Exit code 0 iff PASS, 1 on any FAIL, 2 on a usage/IO error.

use cis_verify::verify::{VerifyOutcome, verify};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: cis-verify <receipt> <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN>");
        std::process::exit(2);
    }
    let receipt_path = &args[1];
    let model_path = &args[2];
    let embed_path = &args[3];
    let vocab_path = &args[4];

    let receipt_text = match std::fs::read_to_string(receipt_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: reading {receipt_path}: {e}");
            std::process::exit(2);
        }
    };
    let model_bytes = match std::fs::read(model_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: reading {model_path}: {e}");
            std::process::exit(2);
        }
    };
    let embed_bytes = match std::fs::read(embed_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: reading {embed_path}: {e}");
            std::process::exit(2);
        }
    };
    let vocab_bytes = match std::fs::read(vocab_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: reading {vocab_path}: {e}");
            std::process::exit(2);
        }
    };

    println!("cis-verify: receipt={receipt_path}");
    println!(
        "cis-verify: MODEL.SAF={model_path} ({} bytes)",
        model_bytes.len()
    );
    println!(
        "cis-verify: EMBED.BIN={embed_path} ({} bytes)",
        embed_bytes.len()
    );
    println!(
        "cis-verify: VOCAB.BIN={vocab_path} ({} bytes)",
        vocab_bytes.len()
    );

    match verify(&receipt_text, &model_bytes, &embed_bytes, &vocab_bytes) {
        VerifyOutcome::Pass { steps } => {
            println!("check: receipt parse ......... ok");
            println!("check: artifact hashes ........ ok");
            println!("check: prompt tokenization ..... ok");
            println!("check: token-id sequence ({steps} steps) ok");
            println!("check: cis-digest (FNV-1a 64) .. ok");
            println!("check: witness chain (SHA-256) . ok");
            println!("VERIFY PASS");
            std::process::exit(0);
        }
        VerifyOutcome::Fail { field, detail } => {
            println!("check failed: {} ({detail})", field.name());
            println!("VERIFY FAIL ({})", field.name());
            std::process::exit(1);
        }
    }
}
