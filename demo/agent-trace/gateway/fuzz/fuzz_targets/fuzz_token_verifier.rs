#![no_main]
//! SAFE-7c cargo-fuzz target: the capability-token issue/verify pair from
//! `gateway::capability`, specifically the #106 tool-tag binding fix
//! (`signable_bytes` folds `tool_tag` into the HMAC input; `verify` must
//! reject a token whose issuing tag doesn't match the caller's tag).
//! Two invariants are checked on every input (a violation panics, which
//! libFuzzer reports as a crash -- exactly the "must never wrongly
//! ALLOW" property #106 exists to guarantee):
//!   1. a token issued for `tag1` must never `verify()` Ok under a
//!      different `tag2`.
//!   2. a single-bit-flipped token must never verify Ok.

use gateway::capability::verify;
use gateway::capability::issue;
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;

fn take_u64(data: &[u8], off: &mut usize) -> u64 {
    let mut b = [0u8; 8];
    let n = data.len().saturating_sub(*off).min(8);
    b[..n].copy_from_slice(&data[*off..*off + n]);
    *off += n;
    u64::from_le_bytes(b)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }
    let mut off = 0usize;
    let key = b"cap-fuzz-key-not-for-prod".to_vec();

    let decision_index = take_u64(data, &mut off);
    let mut hash = [0u8; 32];
    let n = data.len().saturating_sub(off).min(32);
    hash[..n].copy_from_slice(&data[off..off + n]);
    off += n;
    let expiry = take_u64(data, &mut off);
    let now = take_u64(data, &mut off);

    // Remaining bytes: two tool tags separated by a 0xFF byte, so both
    // short/empty/long/non-ASCII tags get exercised on both sides of the
    // binding check.
    let rest = &data[off.min(data.len())..];
    let mut parts = rest.splitn(2, |&b| b == 0xFF);
    let tag1_bytes = parts.next().unwrap_or(&[]);
    let tag2_bytes = parts.next().unwrap_or(tag1_bytes);
    let tag1 = String::from_utf8_lossy(tag1_bytes).to_string();
    let tag2 = String::from_utf8_lossy(tag2_bytes).to_string();

    let tok = issue(&key, decision_index, hash, expiry, &tag1);

    // Invariant 1: cross-tag replay must fail (SAFE-11 / #106).
    {
        let mut consumed = HashSet::new();
        let r = verify(
            &key, decision_index, hash, expiry, &tok, now, &mut consumed, &tag2,
        );
        if tag1 != tag2 {
            assert_ne!(
                r,
                Ok(()),
                "token issued for tool_tag {tag1:?} verified under different tag {tag2:?} -- cross-tool binding broken"
            );
        }
    }

    // Invariant 2: a single-bit-flipped token must never verify.
    if !tok.is_empty() {
        let mut corrupted = tok.clone().into_bytes();
        let idx = (decision_index as usize) % corrupted.len();
        corrupted[idx] ^= 0x01;
        if let Ok(corrupted_str) = String::from_utf8(corrupted) {
            if corrupted_str != tok {
                let mut consumed2 = HashSet::new();
                let r2 = verify(
                    &key,
                    decision_index,
                    hash,
                    expiry,
                    &corrupted_str,
                    now,
                    &mut consumed2,
                    &tag1,
                );
                assert_ne!(r2, Ok(()), "single-bit-flipped token verified");
            }
        }
    }
});
