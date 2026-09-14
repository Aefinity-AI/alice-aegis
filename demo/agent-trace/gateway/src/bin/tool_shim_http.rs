//! SAFE-5c tool shim: HTTP egress.
//!
//! Usage:
//!   tool_shim_http --idx N --cap HEX --exp UNIX --action-hash HEX
//!       --key-file PATH [--consumed-file PATH]
//!       --method GET|POST --url URL [--data STRING]
//!
//! The action bytes bound by the capability token are the canonical
//! request line `METHOD SP URL SP DATA\n` (DATA empty for GET). Kept
//! simple deliberately: this shim shells out to `curl` rather than
//! implementing HTTP itself (no crates.io deps, matches the rest of this
//! offline-only crate). Refuses (SHIM REFUSE: <reason>) before invoking
//! curl on any capability failure.

use gateway::shim_common::{check_and_consume, default_consumed_path, parse_shim_args};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let parsed = parse_shim_args(&argv);

    let mut key_file: Option<PathBuf> = None;
    let mut consumed_file: Option<PathBuf> = None;
    let mut method = "GET".to_string();
    let mut url: Option<String> = None;
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
            "--method" => {
                method = rest.get(i + 1).cloned().unwrap_or_else(|| "GET".into());
                i += 2;
            }
            "--url" => {
                url = rest.get(i + 1).cloned();
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
    let url = url.unwrap_or_else(|| {
        println!("SHIM REFUSE: no --url given");
        std::process::exit(1);
    });
    let consumed_path = consumed_file.unwrap_or_else(|| default_consumed_path("http"));
    let method_upper = method.to_uppercase();
    let data_str = data.clone().unwrap_or_default();

    let mut action = Vec::new();
    action.extend_from_slice(method_upper.as_bytes());
    action.push(b' ');
    action.extend_from_slice(url.as_bytes());
    action.push(b' ');
    action.extend_from_slice(data_str.as_bytes());
    action.push(b'\n');

    // SAFE-11: this shim's fixed, non-negotiable identity for capability
    // binding -- never derived from argv/caller input (see
    // shim_common::check_and_consume doc comment).
    check_and_consume(&key, &parsed, &action, &consumed_path, "http");

    // Capability accepted: only now do we make the real request.
    let mut cmd = Command::new("curl");
    cmd.arg("-sS").arg("-X").arg(&method_upper).arg(&url);
    if let Some(d) = &data {
        cmd.arg("--data").arg(d);
    }
    match cmd.status() {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("tool_shim_http: curl exec failed: {e}");
            ExitCode::FAILURE
        }
    }
}
