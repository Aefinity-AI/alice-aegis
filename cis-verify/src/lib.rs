//! `cis-verify` — standalone third-party verifier core for CIS-1 decode
//! receipts (`AEGIS-WITNESS v1-CIS`).
//!
//! Design: `docs/design/CIS_VERIFY_DESIGN.md`. This crate implements all
//! six builder tasks (§6.2): SHA-256, FNV-1a 64, the witness hash-chain
//! construction, receipt text parsing, artifact hashing (tasks 1-2), the
//! CIS-1 §5 reference integer ops — TMV, QUANT-ACT, REQUANT/scale,
//! RMSNORM-I, NORMQ, container-boundary conversions (`ops`), and the exp
//! machinery, SOFTMAX-I, ROPE-I, ACT-I, ARGMAX (`attn`) (tasks 3-4); MODEL.SAF/
//! VOCAB.BIN parsing, the tokenizer, and the `FullInt` forward-pass glue
//! (`safetensors`, `json_min`, `config`, `vocab`, `forward`) (task 5); and
//! the verify orchestration plus `std`-feature CLI (`verify`,
//! `src/bin/cis-verify.rs`) (task 6).
//!
//! `no_std` + `alloc` for the verification core, zero runtime dependencies
//! (§3.2 of the design doc): no `serde`, no `sha2` crate, no `clap`. The
//! `std` feature gates only file I/O, argv, and stdout in the CLI binary —
//! every module below is `no_std`-capable either way.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod artifact;
pub mod attn;
pub mod config;
pub mod fnv;
pub mod forward;
pub mod hex;
pub mod json_min;
pub mod ops;
pub mod receipt;
pub mod safetensors;
pub mod sha256;
pub mod verify;
pub mod vocab;
pub mod witness;
