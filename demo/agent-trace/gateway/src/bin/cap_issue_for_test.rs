//! Test-only helper: issue a capability token exactly the way
//! `gateway_daemon` would on a real ALLOW, reusing the SAME
//! `gateway::capability::issue` and `gateway::shim_common::*_action`
//! functions the daemon and the shims use (not a separate/parallel
//! implementation of the token format) -- so a token minted here is
//! indistinguishable from one a real daemon ALLOW would hand out. This
//! lets `tests/escape/04_replayed_token.sh` exercise the shim's replay
//! defense (a genuinely valid, single-use token, reused twice) without
//! needing a full gateway_daemon + real receipt fixture in the loop for
//! that specific test.
//!
//! usage: cap_issue_for_test <cap-key-file> <decision-index> <expiry-unix>
//!          shell <command...>
//!        | file <path> <content>
//!        | http <method> <url>
//! prints the encoded token to stdout.

use gateway::{capability, shim_common};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!("usage: cap_issue_for_test <cap-key-file> <decision-index> <expiry-unix> shell <command...> | file <path> <content> | http <method> <url>");
        std::process::exit(2);
    }
    let key = std::fs::read(&args[1]).unwrap_or_else(|e| {
        eprintln!("read key file: {e}");
        std::process::exit(2);
    });
    let idx: u64 = args[2].parse().unwrap_or_else(|_| {
        eprintln!("bad decision-index");
        std::process::exit(2);
    });
    let expiry: u64 = args[3].parse().unwrap_or_else(|_| {
        eprintln!("bad expiry-unix");
        std::process::exit(2);
    });
    let action_bytes = match args[4].as_str() {
        "shell" => shim_common::shell_action(&args[5..].join(" ")),
        "file" => shim_common::file_action(&args[5], &args[6]),
        "http" => shim_common::http_action(&args[5], &args[6]),
        other => {
            eprintln!("unknown action kind: {other}");
            std::process::exit(2);
        }
    };
    let action_hash = gateway::sha256_bytes(&action_bytes);
    let cap = capability::issue(&key, idx, action_hash, expiry);
    println!("{}", capability::encode(&cap));
}
