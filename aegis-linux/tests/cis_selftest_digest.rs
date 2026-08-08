//! Pins the cis_selftest FNV-1a digest (Rule D: bit-exactness over
//! benchmarks). The digest is the cross-ISA identity artifact the L-stick
//! prints on the Dell i5-5200U and HP Stream N4020; if a refactor of
//! `aegis_core::cis` (or of the selftest's deterministic sweep) changes any
//! produced bit, this fails loudly instead of the mismatch surfacing on iron.
//!
//! The constant was computed once by running the example on the dev host
//! (arithmetic identity only — carries no performance meaning):
//!   cargo run --release --example cis_selftest

#[path = "../examples/cis_selftest.rs"]
mod cis_selftest;

#[test]
fn digest_matches_pinned_constant() {
    let (digest, all_pass) = cis_selftest::run_selftest();
    assert!(all_pass, "cis_selftest reported a section FAIL");
    assert_eq!(
        digest, 0x7698_5613_c965_f643,
        "CIS selftest digest drifted — a refactor changed produced bits"
    );
}
