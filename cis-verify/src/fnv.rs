//! FNV-1a 64, exactly as the engine defines it.
//!
//! Transcribed from `aegis-core/src/cis_infer.rs:437-445`:
//!
//! ```text
//! pub const FNV1A64_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
//! pub const FNV1A64_PRIME: u64 = 0x100_0000_01B3;
//!
//! pub fn fnv1a64(mut h: u64, bytes: &[u8]) -> u64 {
//!     for &b in bytes {
//!         h ^= b as u64;
//!         h = h.wrapping_mul(FNV1A64_PRIME);
//!     }
//!     h
//! }
//! ```
//!
//! Same constants, same fold order (XOR-then-multiply, i.e. FNV-1a not
//! FNV-1), same offset basis / prime as the published FNV-1a 64 spec — the
//! offset/prime values themselves are also the canonical published ones
//! (offset basis `0xcbf29ce484222325`, prime `0x100000001b3`).
//!
//! The receipt's `cis-digest` field (`docs/design/CIS_VERIFY_DESIGN.md`
//! §1.1) is this fold applied over the prompt token ids then the generated
//! token ids, each as little-endian `u32` bytes, seeded from
//! `FNV1A64_OFFSET` — the exact sequence `aegis-linux/examples/
//! cis_witness.rs`'s `replay()` (lines 77-97) performs, which this crate's
//! `receipt::cis_digest_of` (`src/receipt.rs`) reproduces for artifact
//! comparison.

/// Offset basis. Identical to `aegis-core/src/cis_infer.rs:437`.
pub const FNV1A64_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
/// FNV prime. Identical to `aegis-core/src/cis_infer.rs:438`.
pub const FNV1A64_PRIME: u64 = 0x100_0000_01B3;

/// FNV-1a 64 fold. Identical algorithm and byte order to
/// `aegis-core/src/cis_infer.rs:440-445`.
pub fn fnv1a64(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV1A64_PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    // Published FNV-1a 64 test vectors — same set
    // `aegis-core/src/cis_infer.rs:1503-1507`'s `golden_fnv1a64` test pins
    // the engine's copy against.
    #[test]
    fn published_vectors() {
        assert_eq!(fnv1a64(FNV1A64_OFFSET, b""), 0xCBF29CE484222325);
        assert_eq!(fnv1a64(FNV1A64_OFFSET, b"a"), 0xAF63DC4C8601EC8C);
        assert_eq!(fnv1a64(FNV1A64_OFFSET, b"foobar"), 0x85944171F73967E8);
    }
}
