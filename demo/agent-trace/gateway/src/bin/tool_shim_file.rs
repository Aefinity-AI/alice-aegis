//! SAFE-5c tool shim: file-write.
//!
//! Usage:
//!   tool_shim_file --idx N --cap HEX --exp UNIX --action-hash HEX
//!       --key-file PATH [--consumed-file PATH] --path OUT [--data STRING]
//!
//! If `--data` is not given, the bytes to write are read from stdin. The
//! action bytes bound by the capability token are exactly those bytes
//! (NOT including `--path`) — this mirrors "the receipt's own last-step
//! tool-call input bytes" binding in the gateway's `decide()`. Refuses
//! (SHIM REFUSE: <reason>) before writing anything on any capability
//! failure.

use gateway::shim_common::{check_and_consume, default_consumed_path, parse_shim_args};
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let parsed = parse_shim_args(&argv);

    let mut key_file: Option<PathBuf> = None;
    let mut consumed_file: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut data: Option<String> = None;
    let rest = &parsed.rest;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--key-file" => {
                key_file = rest.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--consumed-file" => {
                consumed_file = rest.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--path" => {
                out_path = rest.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--data" => {
                data = rest.get(i + 1).cloned();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    let key_file = key_file.unwrap_or_else(|| {
        println!("SHIM REFUSE: no --key-file given");
        std::process::exit(1);
    });
    let key = fs::read(&key_file).unwrap_or_else(|e| {
        println!("SHIM REFUSE: could not read key file: {e}");
        std::process::exit(1);
    });
    let out_path = out_path.unwrap_or_else(|| {
        println!("SHIM REFUSE: no --path given");
        std::process::exit(1);
    });
    let consumed_path = consumed_file.unwrap_or_else(|| default_consumed_path("file"));

    let bytes: Vec<u8> = match data {
        Some(s) => s.into_bytes(),
        None => {
            let mut buf = Vec::new();
            std::io::stdin().read_to_end(&mut buf).unwrap_or_else(|e| {
                println!("SHIM REFUSE: could not read stdin: {e}");
                std::process::exit(1);
            });
            buf
        }
    };

    // SAFE-11: this shim's fixed, non-negotiable identity for capability
    // binding -- never derived from argv/caller input (see
    // shim_common::check_and_consume doc comment).
    check_and_consume(&key, &parsed, &bytes, &consumed_path, "file");

    // Capability accepted: only now do we write the real file.
    match fs::write(&out_path, &bytes) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tool_shim_file: write failed: {e}");
            ExitCode::FAILURE
        }
    }
}
