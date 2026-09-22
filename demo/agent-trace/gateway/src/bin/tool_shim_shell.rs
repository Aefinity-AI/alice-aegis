//! SAFE-5c tool shim: shell-exec.
//!
//! Usage:
//!   tool_shim_shell --idx N --cap HEX --exp UNIX --action-hash HEX
//!       --key-file PATH [--consumed-file PATH] -- <command...>
//!
//! The action bytes this shim binds against are the exact command string
//! (all tokens after `--`, joined with a single space) — the gateway
//! daemon must have issued the capability token for the SHA-256 of that
//! same string. Refuses (SHIM REFUSE: <reason>) on: no token, bad/forged
//! token, expired token, replayed token, or an action-hash mismatch
//! (i.e. this is not the command the token was issued for). Only on
//! success does it exec the command via `sh -c`.

use gateway::shim_common::{check_and_consume, default_consumed_path, parse_shim_args};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let parsed = parse_shim_args(&argv);

    let mut key_file: Option<PathBuf> = None;
    let mut consumed_file: Option<PathBuf> = None;
    let mut command_tokens: Vec<String> = Vec::new();
    let mut i = 0;
    let rest = &parsed.rest;
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
            "--" => {
                command_tokens = rest[i + 1..].to_vec();
                break;
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
    let consumed_path = consumed_file.unwrap_or_else(|| default_consumed_path("shell"));

    let command = command_tokens.join(" ");
    // SAFE-11: this shim's fixed, non-negotiable identity for capability
    // binding -- never derived from argv/caller input (see
    // shim_common::check_and_consume doc comment).
    check_and_consume(&key, &parsed, command.as_bytes(), &consumed_path, "shell");

    // Capability accepted: only now do we run the real command.
    let status = Command::new("sh").arg("-c").arg(&command).status();
    match status {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("tool_shim_shell: exec failed: {e}");
            ExitCode::FAILURE
        }
    }
}
