//! tool-2 (state/QUEUE.md, claudius-maximus repo): standalone, OFFLINE
//! receipt-log verifier.
//!
//! This is the "another box" step of the tool-2 demo (see
//! `demo/agent-trace/gateway/receipt-demo/README.md`): it has NO shared
//! in-memory state with the agent/gateway/shim steps that produced the
//! log. It only reads two files off disk -- the shared HMAC key (the same
//! symmetric key file the gateway/shims already use per the SAFE-5c
//! design; this crate has no asymmetric-signature scheme, so an
//! independent verifier here still needs that key, exactly as
//! `tool_shim_file` does) and a JSON-lines receipt log -- and
//! re-verifies each ALLOW entry using the REAL `gateway::capability::verify`
//! function (the exact same code the shim itself calls), plus
//! `gateway::action_hash` to recompute the action hash straight from the
//! logged bytes. No new crypto or verification logic is implemented here.
//!
//! What it checks, per logged entry:
//!   1. the logged `action_hash` actually matches sha256(logged `data`)
//!      -- catches a log/receipt that was edited after the fact even if
//!      the capability token string itself was left alone.
//!   2. the capability token verifies against (idx, action_hash, exp,
//!      tool) under the shared key, using a FRESH, empty consumed-index
//!      set (this process's own memory, not the shim's) -- catches a
//!      forged/tampered/expired token exactly the way the shim's own
//!      `capability::verify` call would, independently.
//!
//! usage: offline_verify_receipts <key-file> <receipts.log>
//! Exit code 0 iff every ALLOW-decision entry independently re-verifies.

use gateway::capability;
use std::collections::HashSet;
use std::fs;

/// Extract a bare (unescaped) string field's value from one JSON-ish log
/// line. This log format is produced ONLY by this demo's own logger
/// (see receipt-demo/demo.sh), which never emits quotes/backslashes
/// inside values, so a hand-rolled substring search is sufficient here
/// and does not pull in a JSON crate for a demo script.
fn get_str(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let start = line.find(&pat)? + pat.len();
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

fn get_num(line: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: offline_verify_receipts <key-file> <receipts.log>");
        std::process::exit(2);
    }
    let key = fs::read(&args[1]).unwrap_or_else(|e| {
        eprintln!("read key file: {e}");
        std::process::exit(2);
    });
    let log = fs::read_to_string(&args[2]).unwrap_or_else(|e| {
        eprintln!("read receipt log: {e}");
        std::process::exit(2);
    });

    let mut any_fail = false;
    let mut n = 0usize;
    for line in log.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        n += 1;
        let decision = get_str(line, "decision").unwrap_or_default();
        let idx = get_num(line, "idx");
        let exp = get_num(line, "exp");
        let cap = get_str(line, "cap");
        let tool = get_str(line, "tool");
        let data = get_str(line, "data");
        let logged_hash = get_str(line, "action_hash");

        if decision != "ALLOW" {
            // Still worth independently reproducing the refusal, so the
            // log's own DENY claim is not just taken on faith either.
            let (idx, exp, cap, tool, data, logged_hash) =
                match (idx, exp, &cap, &tool, &data, &logged_hash) {
                    (Some(i), Some(e), Some(c), Some(t), Some(d), Some(h)) => {
                        (i, e, c.clone(), t.clone(), d.clone(), h.clone())
                    }
                    _ => {
                        println!("OFFLINE VERIFY entry {n}: SKIP (DENY, incomplete fields)");
                        continue;
                    }
                };
            let real_hash = gateway::action_hash(data.as_bytes());
            let mut consumed = HashSet::new();
            match capability::verify(&key, idx, real_hash, exp, &cap, now(), &mut consumed, &tool)
            {
                Ok(()) => {
                    println!(
                        "OFFLINE VERIFY entry {n} (idx={idx}): UNEXPECTED -- logged as DENY but token verifies! (gateway/log disagreement)"
                    );
                    any_fail = true;
                }
                Err(e) => {
                    println!(
                        "OFFLINE VERIFY entry {n} (idx={idx}): DENY confirmed independently ({}) -- {} -- matches gateway's own refusal",
                        e.reason(),
                        logged_hash
                    );
                }
            }
            continue;
        }

        let (idx, exp, cap, tool, data, logged_hash) =
            match (idx, exp, cap, tool, data, logged_hash) {
                (Some(i), Some(e), Some(c), Some(t), Some(d), Some(h)) => (i, e, c, t, d, h),
                _ => {
                    println!("OFFLINE VERIFY entry {n}: FAIL (ALLOW entry missing required fields)");
                    any_fail = true;
                    continue;
                }
            };

        let real_hash = gateway::action_hash(data.as_bytes());
        let real_hash_hex = gateway::hex32(&real_hash);
        if !real_hash_hex.eq_ignore_ascii_case(&logged_hash) {
            println!(
                "OFFLINE VERIFY entry {n} (idx={idx}): FAIL -- logged action_hash {logged_hash} does not match sha256(logged data) {real_hash_hex} (log tampered after the fact)"
            );
            any_fail = true;
            continue;
        }

        let mut consumed = HashSet::new();
        match capability::verify(&key, idx, real_hash, exp, &cap, now(), &mut consumed, &tool) {
            Ok(()) => {
                println!("OFFLINE VERIFY entry {n} (idx={idx}): PASS -- receipt independently re-verified (tool={tool})");
            }
            Err(e) => {
                println!(
                    "OFFLINE VERIFY entry {n} (idx={idx}): FAIL -- capability does not verify ({})",
                    e.reason()
                );
                any_fail = true;
            }
        }
    }

    if any_fail {
        std::process::exit(1);
    }
}

fn now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
