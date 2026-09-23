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
//! Tool binding: the signable bytes identify which TOOL/shim the token
//! was approved for, and each `tool_shim_*` binary keeps its own
//! independent consumed-index set, so a token minted for e.g. a
//! file-write ALLOW must not verify at a different shim (e.g.
//! `tool_shim_shell`) even when the action bytes hash-match at both.
//! `tool_tag` is folded
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
    /// tool-1 GATEWAY HARDENING: the token's `expiry_unix` is further in
    /// the future than any legitimate capability could be (see
    /// `MAX_SKEW_SECS` below). A well-behaved gateway never mints a token
    /// whose expiry is more than `cap_ttl` (default 60s, operator-capped)
    /// past its own issue-time clock reading, so a token whose expiry is
    /// implausibly far ahead of the *verifier's* clock is not a normal
    /// slow-clock skew case -- it is either a forged/tampered `--exp`, or
    /// the verifying process's own clock has been rolled backward/frozen
    /// (accidentally or by an attacker with local control, e.g. `faketime`)
    /// so that `now` never catches up to `expiry_unix` and the naive
    /// `now > expiry_unix` check would then accept the token forever. This
    /// bounds that: skew tolerance is finite, not unbounded.
    NotYetValid,
}

impl VerifyError {
    pub fn reason(&self) -> &'static str {
        match self {
            VerifyError::BadMac => "bad mac",
            VerifyError::Expired => "expired",
            VerifyError::AlreadyConsumed => "already consumed",
            VerifyError::NotYetValid => "not yet valid (clock skew out of bounds)",
        }
    }
}

/// tool-1 GATEWAY HARDENING: the largest gap between a verifier's `now`
/// and a token's `expiry_unix` that is tolerated as ordinary clock skew
/// between the gateway daemon (which stamps `expiry_unix` at ALLOW time
/// using ITS clock) and a tool shim (which checks it later using ITS OWN
/// clock, possibly on a different host via the inter-box relay). Chosen
/// generously above the default `--cap-ttl` (60s) and any plausible NTP
/// drift/verify-latency, while still being a *bounded* value -- a token
/// cannot be treated as valid no matter how far `expiry_unix` sits beyond
/// `now`. Without this bound, a verifier whose own clock is stuck/rolled
/// back (so `now` never advances past `expiry_unix`) would accept a
/// token indefinitely, which is a fail-open clock-skew bug, not a
/// legitimate tolerance.
pub const MAX_SKEW_SECS: u64 = 3600;

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
/// closing cross-shim token replay (a token minted for one
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
    // tool-1 GATEWAY HARDENING: bounded clock-skew check. A genuine token
    // never has `expiry_unix` more than MAX_SKEW_SECS beyond a sane
    // verifier clock; if it does, the verifier's own `now` is not to be
    // trusted (frozen/rolled-back clock) and the token must be rejected
    // rather than treated as valid-until-the-clock-catches-up (which
    // would be an unbounded tolerance, i.e. fail-open).
    if expiry_unix > now.saturating_add(MAX_SKEW_SECS) {
        return Err(VerifyError::NotYetValid);
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
    // confusion (a file-write token replayed at
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

    // ===============================================================
    // tool-1 GATEWAY HARDENING: (a) receipt/capability replay after key
    // rotation, (b) oversized/malformed tool-call arguments, (c) clock
    // skew on expiry. See src/main.rs cap_key loading (read fresh from
    // --cap-key-file at daemon startup / by each tool_shim_* invocation)
    // for why "rotation" here means simply verifying with a different key
    // than the one a token was issued under.
    // ===============================================================

    // ---- (a) key rotation ----------------------------------------

    #[test]
    fn key_rotation_invalidates_old_token() {
        let old_key = k();
        let new_key = b"post-rotation-cap-key-not-for-prod".to_vec();
        let hash = ah();
        // Token minted under the pre-rotation key (this is exactly what a
        // captured/leaked receipt+token pair from before a rotation looks
        // like).
        let tok = issue(&old_key, 1, hash, 1_000_100, "shell");
        let mut consumed = HashSet::new();
        // Presented to a verifier that has since rotated to the new key
        // (e.g. gateway daemon restarted with a fresh --cap-key-file, or a
        // tool shim invocation that re-reads a rotated key file) -- must
        // be rejected, not silently accepted just because the rest of the
        // fields line up.
        let r = verify(
            &new_key, 1, hash, 1_000_100, &tok, 1_000_000, &mut consumed, "shell",
        );
        assert_eq!(r, Err(VerifyError::BadMac));
        // The decision index must NOT have been consumed by the rejected
        // attempt (a forger must not be able to burn a real index via a
        // bad-key replay).
        assert!(!consumed.contains(&1));
    }

    #[test]
    fn key_rotation_new_tokens_work_after_rotation() {
        // Sanity: rotation does not brick the system -- a token freshly
        // minted under the NEW key still verifies fine under the NEW key.
        let new_key = b"post-rotation-cap-key-not-for-prod".to_vec();
        let hash = ah();
        let tok = issue(&new_key, 1, hash, 1_000_100, "shell");
        let mut consumed = HashSet::new();
        let r = verify(
            &new_key, 1, hash, 1_000_100, &tok, 1_000_000, &mut consumed, "shell",
        );
        assert!(r.is_ok(), "expected Ok, got {r:?}");
    }

    #[test]
    fn key_rotation_old_token_rejected_at_every_stage_not_just_first_check() {
        // Even if an attacker also replays the (now-rejected) decision
        // index and re-presents an already-expired-looking combination
        // under the old token, rotation must reject it purely on BadMac,
        // before expiry/replay logic is ever consulted -- consistent with
        // the "forger gets BadMac, not AlreadyConsumed/Expired" ordering
        // rule documented on `verify` above.
        let old_key = k();
        let new_key = b"another-rotated-key".to_vec();
        let hash = ah();
        let tok = issue(&old_key, 5, hash, 100, "shell"); // already "expired" by expiry too
        let mut consumed = HashSet::new();
        let r = verify(&new_key, 5, hash, 100, &tok, 1_000_000, &mut consumed, "shell");
        assert_eq!(
            r,
            Err(VerifyError::BadMac),
            "old-key token must fail closed on the key check, not leak Expired/AlreadyConsumed"
        );
    }

    // ---- (b) oversized / malformed tool-call arguments -------------

    #[test]
    fn oversized_action_bytes_do_not_panic_and_verify_correctly() {
        // A 8 MiB "argument" (e.g. a huge shell command or file payload)
        // must hash and verify without panicking, and a token minted for
        // different (smaller) bytes must still be cleanly refused as a
        // mismatch rather than accepted or causing a panic anywhere in
        // the hashing/verification path.
        let key = k();
        let big = vec![0x41u8; 8 * 1024 * 1024];
        let big_hash = crate::action_hash(&big);
        let tok = issue(&key, 1, big_hash, 1_000_100, "shell");
        let mut consumed = HashSet::new();
        let r = verify(
            &key, 1, big_hash, 1_000_100, &tok, 1_000_000, &mut consumed, "shell",
        );
        assert!(r.is_ok(), "expected Ok for the exact oversized payload, got {r:?}");

        // A DIFFERENT oversized payload (one bit flipped near the end) must
        // not verify against that token's hash.
        let mut other_big = big.clone();
        *other_big.last_mut().unwrap() ^= 0x01;
        let other_hash = crate::action_hash(&other_big);
        let mut consumed2 = HashSet::new();
        let r2 = verify(
            &key, 1, other_hash, 1_000_100, &tok, 1_000_000, &mut consumed2, "shell",
        );
        assert_eq!(r2, Err(VerifyError::BadMac));
    }

    #[test]
    fn malformed_token_strings_are_rejected_cleanly_not_panicking() {
        let key = k();
        let hash = ah();
        let mut consumed = HashSet::new();
        for malformed in [
            "",
            "not-hex-at-all",
            "zz",
            &"a".repeat(63), // one short of a real 64-hex-char token
            &"a".repeat(65), // one too many
            &"f".repeat(10_000), // grossly oversized token string
            "\u{0}\u{0}\u{0}",
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\n", // trailing newline
        ] {
            let r = verify(
                &key, 1, hash, 1_000_100, malformed, 1_000_000, &mut consumed, "shell",
            );
            assert_eq!(
                r,
                Err(VerifyError::BadMac),
                "malformed token {malformed:?} must fail closed as BadMac, not panic or pass"
            );
            assert!(
                !consumed.contains(&1),
                "a rejected malformed token must never consume the real index"
            );
        }
    }

    #[test]
    fn parse_shim_args_rejects_malformed_numeric_fields_without_panicking() {
        // shim_common::parse_shim_args must never panic on malformed
        // --idx/--exp values (e.g. non-numeric, oversized, negative-looking)
        // -- it should simply leave the field as None, which
        // check_and_consume then treats as "no token" (fail closed).
        use crate::shim_common::parse_shim_args;
        let argv: Vec<String> = [
            "--idx",
            "not-a-number",
            "--cap",
            "deadbeef",
            "--exp",
            "99999999999999999999999999999999", // overflows u64
            "--action-hash",
            "zz-not-hex",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let parsed = parse_shim_args(&argv);
        assert_eq!(parsed.idx, None, "non-numeric --idx must parse to None, not panic");
        assert_eq!(
            parsed.exp, None,
            "overflowing --exp must parse to None, not panic or wrap"
        );
        // --cap and --action-hash are opaque strings at this layer (validated
        // later by verify()/the hash-mismatch check), so they pass through.
        assert_eq!(parsed.cap.as_deref(), Some("deadbeef"));
        assert_eq!(parsed.action_hash_hex.as_deref(), Some("zz-not-hex"));
    }

    // ---- (c) clock skew on expiry -----------------------------------

    #[test]
    fn slightly_past_expiry_is_rejected() {
        let key = k();
        let hash = ah();
        let tok = issue(&key, 1, hash, 1_000_100, "shell");
        let mut consumed = HashSet::new();
        // Presented just 1 second past expiry -- must still be rejected;
        // there is no leniency on the "past expiry" side.
        let r = verify(
            &key, 1, hash, 1_000_100, &tok, 1_000_101, &mut consumed, "shell",
        );
        assert_eq!(r, Err(VerifyError::Expired));
    }

    #[test]
    fn far_future_expiry_beyond_skew_bound_is_rejected_not_yet_valid() {
        // A token whose expiry sits further ahead of `now` than any
        // legitimate cap-ttl + clock-skew tolerance allows must be
        // rejected -- this is the case where the verifying clock is
        // stuck/rolled back (or `--exp` was forged far into the future),
        // which without a bound would let `now > expiry_unix` never fire
        // and the token verify "forever". MAX_SKEW_SECS bounds it.
        let key = k();
        let hash = ah();
        let now = 1_000_000u64;
        let far_future_expiry = now + MAX_SKEW_SECS + 1;
        let tok = issue(&key, 1, hash, far_future_expiry, "shell");
        let mut consumed = HashSet::new();
        let r = verify(&key, 1, hash, far_future_expiry, &tok, now, &mut consumed, "shell");
        assert_eq!(r, Err(VerifyError::NotYetValid));
        assert!(!consumed.contains(&1));
    }

    #[test]
    fn expiry_within_skew_bound_is_still_accepted() {
        // Sanity/no-regression: a normal token (expiry well within the
        // skew bound of "now", e.g. the default 60s cap-ttl) must still
        // verify -- the fix must not turn into an over-broad rejection.
        let key = k();
        let hash = ah();
        let now = 1_000_000u64;
        let normal_expiry = now + 60; // default --cap-ttl
        let tok = issue(&key, 1, hash, normal_expiry, "shell");
        let mut consumed = HashSet::new();
        let r = verify(&key, 1, hash, normal_expiry, &tok, now, &mut consumed, "shell");
        assert!(r.is_ok(), "expected Ok, got {r:?}");
    }

    #[test]
    fn expiry_exactly_at_skew_boundary_is_accepted() {
        let key = k();
        let hash = ah();
        let now = 1_000_000u64;
        let boundary_expiry = now + MAX_SKEW_SECS; // exactly at the bound
        let tok = issue(&key, 1, hash, boundary_expiry, "shell");
        let mut consumed = HashSet::new();
        let r = verify(&key, 1, hash, boundary_expiry, &tok, now, &mut consumed, "shell");
        assert!(r.is_ok(), "boundary value must still be accepted, got {r:?}");
    }
}
