//! `cis-verify` — standalone third-party verifier core for CIS-1 decode
//! receipts (`AEGIS-WITNESS v1-CIS`).
//!
//! Design: `docs/design/CIS_VERIFY_DESIGN.md`. This crate implements
//! builder tasks 1-2 only (§6.2): SHA-256, FNV-1a 64, the witness
//! hash-chain construction, receipt text parsing, and artifact hashing.
//! Tasks 3-6 (reference integer ops, attention/activation ops, the forward
//! pass, and the CLI/orchestration layer) are out of scope for this phase.
//!
//! `no_std` + `alloc`, zero runtime dependencies (§3.2 of the design doc):
//! no `serde`, no `sha2` crate, no `clap`. The `std` feature is reserved
//! for a future `bin/cis-verify.rs` CLI and changes nothing about this
//! library today.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod artifact;
pub mod fnv;
pub mod hex;
pub mod receipt;
pub mod sha256;
pub mod witness;
