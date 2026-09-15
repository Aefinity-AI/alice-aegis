#![no_main]
//! SAFE-7c cargo-fuzz target: the `gateway serve` request-line parser
//! (`gateway::protocol::parse_request_lines`, the pure core delegated to
//! by `main.rs::parse_serve_request` -- see that module for the exact
//! wire format: RECEIPT/ACTION/SESSION/COUNTER/TOOL lines, or a
//! `POLL ticket=<id>` line, terminated by a blank line or EOF).

use gateway::protocol::parse_request_lines;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Mirror BufRead::lines(): split on '\n', each line UTF-8 validated
    // (a non-UTF-8 line makes the whole read return an io::Error in the
    // real socket path, which `parse_serve_request`'s `.ok()?` turns
    // into an immediate `None` -- so we just bail the same way here).
    let mut lines: Vec<String> = Vec::new();
    for chunk in data.split(|&b| b == b'\n') {
        match std::str::from_utf8(chunk) {
            Ok(s) => lines.push(s.trim_end_matches('\r').to_string()),
            Err(_) => return,
        }
    }
    // Truncate at the first blank line after the first (the
    // blank-line/EOF terminator convention), matching the `for line in
    // lines { if blank { break } ... }` loop in `parse_serve_request`.
    let mut req_lines: Vec<String> = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if i > 0 && l.trim().is_empty() {
            break;
        }
        req_lines.push(l.clone());
    }
    // Must never panic on any byte input, malformed or not.
    let _ = parse_request_lines(&req_lines);
});
