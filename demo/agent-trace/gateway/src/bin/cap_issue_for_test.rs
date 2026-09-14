//! Test-only helper: issue a capability token exactly the way
//! `gateway_daemon` would on a real ALLOW, reusing the SAME
//! `gateway::capability::issue` function and the same action-byte
//! encodings the shims themselves use (not a separate/parallel
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
//!        | http <method> <url> <data>
//! prints the hex-encoded token to stdout.

use gateway::capability;

fn shell_action(command: &str) -> Vec<u8> {
    command.as_bytes().to_vec()
}

fn file_action(_path: &str, content: &str) -> Vec<u8> {
    content.as_bytes().to_vec()
}

fn http_action(method: &str, url: &str, data: &str) -> Vec<u8> {
    let mut action = Vec::new();
    action.extend_from_slice(method.to_uppercase().as_bytes());
    action.push(b' ');
    action.extend_from_slice(url.as_bytes());
    action.push(b' ');
    action.extend_from_slice(data.as_bytes());
    action.push(b'\n');
    action
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "usage: cap_issue_for_test <cap-key-file> <decision-index> <expiry-unix> shell <command...> | file <path> <content> | http <method> <url> <data>"
        );
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
        "shell" => shell_action(&args[5..].join(" ")),
        "file" => file_action(&args[5], &args[6]),
        "http" => http_action(
            &args[5],
            &args[6],
            args.get(7).map(String::as_str).unwrap_or(""),
        ),
        other => {
            eprintln!("unknown action kind: {other}");
            std::process::exit(2);
        }
    };
    let hash = gateway::action_hash(&action_bytes);
    let token = capability::issue(&key, idx, hash, expiry);
    println!("{}", token);
}
