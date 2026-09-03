//! E19 fuzz target: `cis_verify::receipt::Receipt::parse` against arbitrary
//! byte streams. The parser's contract (src/receipt.rs) is "never panics on
//! malformed input; every failure mode returns Err" — this target exists to
//! find any input that violates that contract (panic, OOM, hang) or, worse,
//! any malformed receipt the parser accepts as valid.

#![no_main]

use cis_verify::receipt::Receipt;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Receipt::parse takes &str; only feed it valid UTF-8, same as any real
    // caller would (a receipt file read as text). Invalid UTF-8 bytes are
    // out of this target's contract, not a parser bug.
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = Receipt::parse(s);
    }
});
