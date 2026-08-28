//! Lowercase hex encode/decode helpers.
//!
//! `hex_lower` is transcribed from `aegis-core/src/witness.rs:215-223`
//! (byte-for-byte, same table, same panic-on-undersized-buffer contract).
//! `hex_lower_string` and `decode_hex` have no engine equivalent — the
//! engine's own hex round-trip lives only as ad-hoc helpers inside
//! `aegis-linux/examples/cis_witness.rs` (`hex`/`unhex`, lines 23-33), which
//! is example code, not part of `aegis-core::witness`; this crate needs the
//! same shape (`alloc`-returning encode, and a decode for receipt parsing)
//! so it is written fresh here rather than "cited" from a non-library file.

use alloc::string::String;
use alloc::vec::Vec;

/// Lowercase-hex encode `src` into `dst` (2 bytes out per byte in). Returns
/// the number of bytes written. Mirrors `aegis-core/src/witness.rs:215-223`
/// exactly, including the panic contract on an undersized `dst`.
pub fn hex_lower(src: &[u8], dst: &mut [u8]) -> usize {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    assert!(dst.len() >= src.len() * 2, "hex output buffer too small");
    for (i, &b) in src.iter().enumerate() {
        dst[i * 2] = HEX[(b >> 4) as usize];
        dst[i * 2 + 1] = HEX[(b & 0x0f) as usize];
    }
    src.len() * 2
}

/// `alloc`-convenience wrapper: hex-encode into a fresh owned `String`.
pub fn hex_lower_string(src: &[u8]) -> String {
    let mut out = alloc::vec![0u8; src.len() * 2];
    let n = hex_lower(src, &mut out);
    out.truncate(n);
    // SAFETY-free: `hex_lower` only ever writes ASCII hex digits.
    String::from_utf8(out).expect("hex_lower output is always ASCII")
}

/// Decode a lowercase (or uppercase) hex string into bytes. Returns `None`
/// on odd length or a non-hex-digit character rather than panicking —
/// receipt text is untrusted input (design doc §5, "malformed-receipt
/// robustness": never a panic).
pub fn decode_hex(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    fn nibble(c: u8) -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.as_chunks::<2>().0 {
        let hi = nibble(pair[0])?;
        let lo = nibble(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

/// Decode exactly `N` bytes from a hex string, e.g. a 64-hex-char SHA-256
/// digest into `[u8; 32]`. `None` on wrong length or a non-hex character.
pub fn decode_hex_exact<const N: usize>(s: &str) -> Option<[u8; N]> {
    let v = decode_hex(s)?;
    if v.len() != N {
        return None;
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&v);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let data = [0u8, 1, 2, 0xff, 0xab, 0x10];
        let s = hex_lower_string(&data);
        assert_eq!(s, "000102ffab10");
        assert_eq!(decode_hex(&s).unwrap(), &data[..]);
    }

    #[test]
    fn decode_hex_exact_checks_length() {
        assert_eq!(decode_hex_exact::<3>("000102"), Some([0u8, 1, 2]));
        assert_eq!(decode_hex_exact::<3>("0001"), None);
        assert_eq!(decode_hex_exact::<3>("00010z"), None);
    }

    #[test]
    fn decode_hex_rejects_odd_length() {
        assert_eq!(decode_hex("abc"), None);
    }
}
