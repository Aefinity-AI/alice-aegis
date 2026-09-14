//! SAFE-5c: capability tokens.
//!
//! `issue` is called by the gateway daemon on ALLOW; `verify` is called
//! independently by each tool shim (`src/bin/tool_shim_*.rs`) before it
//! will execute the real action. Per the design
//! (`state/reports/2026-09-13-SAFE5-ENFORCEMENT-BOUNDARY-DESIGN.md`,
//! claudius-maximus repo, section (c)):
//!
//!   token = hex(HMAC-SHA256(key,
//!       decision_index_be_u64 || 0x00 || action_hash[32] || 0x00 ||
//!       expiry_unix_be_u64 || 0x00 || sha256(tool_tag)[32]))
//!
//! SAFE-11 fix (cross-shim capability-token confusion, safe-10 MAJOR
//! finding): the original signable bytes carried NO identifier of which
//! TOOL/shim the token was approved for, and each `tool_shim_*` binary
//! kept its own independent consumed-index set — so a token minted for
//! e.g. a file-write ALLOW verified successfully (bad-mac check passed)
//! at a completely different shim (e.g. `tool_shim_shell`) as long as the
//! action bytes happened to hash-match at both. `tool_tag` is now folded
//! into the HMAC input (hashed to a fixed 32 bytes so a variable-length
//! tag cannot create parsing ambiguity with the surrounding fixed-width
//! fields), and both `issue` and `verify` take it as an explicit
//! caller-supplied parameter — `issue` is called by the gateway daemon
//! with the tag it actually approved (from the DECIDE request's `TOOL`
//! field), `verify` is called by each shim with ITS OWN fixed identity
//! (see `src/bin/tool_shim_*.rs`; a shim must never accept a
//! caller-supplied tag for this check, or the fix is void). A token
//! issued for one tag now fails `BadMac` at any other tag, at the
//! HMAC-verification stage — before expiry/replay are even considered,
//! consistent with the existing "forger must get BadMac, not
//! AlreadyConsumed/Expired" ordering rule below.
//!
//! `verify` distinguishes three failure reasons (bad mac, expired,
//! already consumed) plus a fourth caller-side reason (no token at all,
//! which is just "empty string" -> bad mac here) so a tool shim can print
//! an honest `SHIM REFUSE: <reason>` line.

use crate::{action_hash, hmac_sha256};
use std::collections::HashSet;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum VerifyError {
    BadMac,
    Expired,
    AlreadyConsumed,
}

impl VerifyError {
    pub fn reason(&self) -> &'static str {
        match self {
            VerifyError::BadMac => "bad mac",
            VerifyError::Expired => "expired",
            VerifyError::AlreadyConsumed => "already consumed",
        }
    }
}

fn signable_bytes(
    decision_index: u64,
    action_hash_val: &[u8; 32],
    expiry_unix: u64,
    tool_tag: &str,
) -> Vec<u8> {
    let tag_hash = action_hash(tool_tag.as_bytes());
    let mut msg = Vec::with_capacity(8 + 1 + 32 + 1 + 8 + 1 + 32);
    msg.extend_from_slice(&decision_index.to_be_bytes());
    msg.push(0);
    msg.extend_from_slice(action_hash_val);
    msg.push(0);
    msg.extend_from_slice(&expiry_unix.to_be_bytes());
    msg.push(0);
    msg.extend_from_slice(&tag_hash);
    msg
}

/// Issue a capability token for one (decision_index, action_hash, expiry,
/// tool_tag) quadruple. `decision_index` is the gateway's own monotonic
/// decision counter (NOT caller-supplied), so a client cannot mint its
/// own indices. `tool_tag` (SAFE-11) is the identifier of the specific
/// tool/shim this token is bound to (e.g. "shell", "file", "http") — the
/// gateway daemon must pass the tag it actually approved for this
/// decision, not let the client choose it after the fact.
pub fn issue(
    key: &[u8],
    decision_index: u64,
    action_hash: [u8; 32],
    expiry_unix: u64,
    tool_tag: &str,
) -> String {
    let mac = hmac_sha256(
        key,
        &signable_bytes(decision_index, &action_hash, expiry_unix, tool_tag),
    );
    crate::hex(&mac)
}

/// Constant-time-ish byte comparison (prototype-grade: correctness of the
/// check matters more than timing-side-channel hardening here, matching
/// the allowlist signature check's own comment in main.rs).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Verify a capability token against the claimed (decision_index,
/// action_hash, expiry, tool_tag), checking MAC validity first, then
/// expiry, then whether this decision_index has already been consumed.
/// On success the index is inserted into `consumed` (single-use). Order
/// matters: an attacker who forges a token must get `BadMac`, not
/// accidentally learn via `AlreadyConsumed`/`Expired` that a real token
/// for that index once existed.
///
/// `tool_tag` (SAFE-11) MUST be the caller's OWN fixed identity (e.g.
/// `tool_shim_shell` always passes `"shell"`), never a value read out of
/// the request/argv being checked — otherwise a caller could simply claim
/// whatever tag the token was actually issued for and the binding is
/// void. A token issued for a different tag now fails here as `BadMac`,
/// closing the safe-10 cross-shim replay finding (a token minted for one
/// tool's ALLOW no longer verifies at any other tool's shim).
pub fn verify(
    key: &[u8],
    decision_index: u64,
    action_hash: [u8; 32],
    expiry_unix: u64,
    token: &str,
    now: u64,
    consumed: &mut HashSet<u64>,
    tool_tag: &str,
) -> Result<(), VerifyError> {
    let expect = issue(key, decision_index, action_hash, expiry_unix, tool_tag);
    if token.len() != expect.len() || !ct_eq(token.as_bytes(), expect.as_bytes()) {
        return Err(VerifyError::BadMac);
    }
    if now > expiry_unix {
        return Err(VerifyError::Expired);
    }
    if consumed.contains(&decision_index) {
        return Err(VerifyError::AlreadyConsumed);
    }
    consumed.insert(decision_index);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k() -> Vec<u8> {
        b"cap-test-key-not-for-prod".to_vec()
    }

    fn ah() -> [u8; 32] {
        crate::action_hash(b"echo hello")
    }

    #[test]
    fn happy_path() {
        let key = k();
        let hash = ah();
        let tok = issue(&key, 1, hash, 1_000_100, "shell");
        let mut consumed = HashSet::new();
        let r = verify(&key, 1, hash, 1_000_100, &tok, 1_000_000, &mut consumed, "shell");
        assert!(r.is_ok(), "expected Ok, got {r:?}");
        assert!(consumed.contains(&1));
    }

    #[test]
    fn bad_mac_forged_token() {
        let key = k();
        let hash = ah();
        let mut consumed = HashSet::new();
        let forged = "0".repeat(64);
        let r = verify(
            &key, 1, hash, 1_000_100, &forged, 1_000_000, &mut consumed, "shell",
        );
        assert_eq!(r, Err(VerifyError::BadMac));
    }

    #[test]
    fn bad_mac_wrong_key() {
        let key = k();
        let other_key = b"a-different-key".to_vec();
        let hash = ah();
        let tok = issue(&other_key, 1, hash, 1_000_100, "shell");
        let mut consumed = HashSet::new();
        let r = verify(
            &key, 1, hash, 1_000_100, &tok, 1_000_000, &mut consumed, "shell",
        );
        assert_eq!(r, Err(VerifyError::BadMac));
    }

    #[test]
    fn bad_mac_tampered_action_hash() {
        let key = k();
        let hash = ah();
        let tok = issue(&key, 1, hash, 1_000_100, "shell");
        let mut other_hash = hash;
        other_hash[0] ^= 0x01;
        let mut consumed = HashSet::new();
        let r = verify(
            &key,
            1,
            other_hash,
            1_000_100,
            &tok,
            1_000_000,
            &mut consumed,
            "shell",
        );
        assert_eq!(r, Err(VerifyError::BadMac));
    }

    #[test]
    fn expired_token() {
        let key = k();
        let hash = ah();
        let tok = issue(&key, 1, hash, 1_000_100, "shell");
        let mut consumed = HashSet::new();
        // now > expiry
        let r = verify(
            &key, 1, hash, 1_000_100, &tok, 1_000_101, &mut consumed, "shell",
        );
        assert_eq!(r, Err(VerifyError::Expired));
    }

    #[test]
    fn already_consumed_replay() {
        let key = k();
        let hash = ah();
        let tok = issue(&key, 1, hash, 1_000_100, "shell");
        let mut consumed = HashSet::new();
        let first = verify(
            &key, 1, hash, 1_000_100, &tok, 1_000_000, &mut consumed, "shell",
        );
        assert!(first.is_ok());
        let second = verify(
            &key, 1, hash, 1_000_100, &tok, 1_000_000, &mut consumed, "shell",
        );
        assert_eq!(second, Err(VerifyError::AlreadyConsumed));
    }

    #[test]
    fn distinct_indices_do_not_collide() {
        let key = k();
        let hash = ah();
        let tok1 = issue(&key, 1, hash, 1_000_100, "shell");
        let tok2 = issue(&key, 2, hash, 1_000_100, "shell");
        assert_ne!(tok1, tok2);
        let mut consumed = HashSet::new();
        assert!(
            verify(&key, 1, hash, 1_000_100, &tok1, 1_000_000, &mut consumed, "shell").is_ok()
        );
        assert!(
            verify(&key, 2, hash, 1_000_100, &tok2, 1_000_000, &mut consumed, "shell").is_ok()
        );
    }

    // -------------------------------------------------------------
    // SAFE-11: a token issued for one tool tag must NOT verify against
    // another tag, even with an identical (decision_index, action_hash,
    // expiry) and the correct key -- this is the exact cross-shim
    // confusion the safe-10 PoC exploited (a file-write token replayed at
    // tool_shim_shell).
    // -------------------------------------------------------------
    #[test]
    fn bad_mac_wrong_tool_tag() {
        let key = k();
        let hash = ah();
        let tok = issue(&key, 1, hash, 1_000_100, "file");
        let mut consumed = HashSet::new();
        let r = verify(
            &key, 1, hash, 1_000_100, &tok, 1_000_000, &mut consumed, "shell",
        );
        assert_eq!(r, Err(VerifyError::BadMac));
    }
}
