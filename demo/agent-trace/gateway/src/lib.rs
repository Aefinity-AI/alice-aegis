//! Shared crypto/encoding helpers for the SAFE-5 gateway and its tool
//! shims. Split out of `main.rs` (which used to define these privately)
//! so that `src/bin/tool_shim_*.rs` binaries can share the exact same
//! HMAC-SHA256 and capability-token logic the gateway daemon uses,
//! without depending on any crates.io HMAC crate (this repo is
//! offline-only, see `demo/agent-trace/gateway/Cargo.toml`).

use aegis_core::witness::{Sha256, sha256};

pub mod capability;
pub mod protocol;
pub mod shim_common;

const BLOCK: usize = 64;

/// HMAC-SHA256 built on aegis_core's Sha256 (no external crate).
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut key_block = [0u8; BLOCK];
    if key.len() > BLOCK {
        let k = sha256(key);
        key_block[..32].copy_from_slice(&k);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= key_block[i];
        opad[i] ^= key_block[i];
    }
    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(msg);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&inner_digest);
    outer.finalize()
}

pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

pub fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 || !s.bytes().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
        i += 2;
    }
    Some(out)
}

/// Hash a 32-byte digest to its hex string (convenience for action hashes).
pub fn hex32(digest: &[u8; 32]) -> String {
    hex(digest)
}

/// Re-export sha256 for callers (tool shims) that need to hash the actual
/// action bytes they are about to execute.
pub fn action_hash(bytes: &[u8]) -> [u8; 32] {
    sha256(bytes)
}
