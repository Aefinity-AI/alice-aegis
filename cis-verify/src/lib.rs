//! `cis-verify` — standalone third-party verifier core for CIS-1 decode
//! receipts (`AEGIS-WITNESS v1-CIS`).
//!
//! Design: `docs/design/CIS_VERIFY_DESIGN.md`. This crate implements
//! builder tasks 1-4 (§6.2): SHA-256, FNV-1a 64, the witness hash-chain
//! construction, receipt text parsing, artifact hashing (tasks 1-2), the
//! CIS-1 §5 reference integer ops — TMV, QUANT-ACT, REQUANT/scale,
//! RMSNORM-I, NORMQ, container-boundary conversions (`ops`), and the exp
//! machinery, SOFTMAX-I, ROPE-I, ACT-I, ARGMAX (`attn`) (tasks 3-4). Tasks
//! 5-6 (the forward-pass glue and the CLI/orchestration layer) are out of
//! scope for this phase.
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
pub mod attn;
pub mod fnv;
pub mod hex;
pub mod ops;
pub mod receipt;
pub mod sha256;
pub mod witness;
