//! MODEL.SAF container parse — SafeTensors's flat header JSON, plus the
//! one `__metadata__` string field the pipeline needs (`aegis_config`).
//!
//! Transcribed from `aegis-core/src/model.rs:130-234` (`SafeTensors`,
//! `TensorView`) per `docs/design/CIS_VERIFY_DESIGN.md` builder task 5 —
//! independent copy, no `path` dependency on `aegis-core`. Byte layout
//! (spec §4): 8-byte little-endian header length, then that many bytes of
//! UTF-8 JSON `{"tensor.name": {"dtype":.., "shape":[..],
//! "data_offsets":[start,end]}, ..., "__metadata__": {...}}`, then the raw
//! tensor bytes (ternary weights packed 4-per-byte per spec §4, norm gains
//! and scales as BF16/F32).

use alloc::collections::BTreeMap;
use alloc::{format, string::String, string::ToString};

use crate::json_min;

/// A borrowed view of one tensor's raw bytes.
pub struct TensorView<'a> {
    data_bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn data(&self) -> &'a [u8] {
        self.data_bytes
    }
}

pub struct SafeTensors<'a> {
    buffer: &'a [u8],
    tensor_start: usize,
    offsets: BTreeMap<String, (usize, usize)>,
    /// Raw `__metadata__` value slice from the header, if present.
    metadata: Option<&'a str>,
}

impl<'a> SafeTensors<'a> {
    /// Parse the header only; tensor bytes are sliced lazily by `tensor()`.
    /// Identical shape to `model.rs:139-177`.
    pub fn deserialize(buffer: &'a [u8]) -> Result<Self, String> {
        if buffer.len() < 8 {
            return Err("Buffer too small".into());
        }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buffer[0..8]);
        let n = u64::from_le_bytes(len_bytes) as usize;
        if buffer.len() < 8 + n {
            return Err("Buffer smaller than header".into());
        }

        let header_str = core::str::from_utf8(&buffer[8..8 + n])
            .map_err(|_| "Invalid UTF-8 in header".to_string())?;

        let mut offsets = BTreeMap::new();
        let mut metadata = None;
        for (name, entry) in json_min::members(header_str)? {
            if name == "__metadata__" {
                metadata = Some(entry);
                continue;
            }
            // Each entry is {"dtype":..,"shape":[..],"data_offsets":[start,end]}
            for (field, value) in json_min::members(entry)? {
                if field == "data_offsets" {
                    let (start, end) = json_min::as_u64_pair(value)
                        .ok_or_else(|| format!("Tensor {}: bad data_offsets {}", name, value))?;
                    offsets.insert(String::from(name), (start as usize, end as usize));
                    break;
                }
            }
        }

        Ok(Self {
            buffer,
            tensor_start: 8 + n,
            offsets,
            metadata,
        })
    }

    /// Look up one string value in the header's `__metadata__` map, unescaped.
    /// `Ok(None)` means genuinely absent; a key that is present but malformed
    /// is an error. Identical to `model.rs:185-199`.
    pub fn metadata_field(&self, key: &str) -> Result<Option<String>, String> {
        let Some(meta) = self.metadata else {
            return Ok(None);
        };
        for (k, v) in json_min::members(meta)? {
            if k == key {
                let raw = json_min::as_str(v)
                    .ok_or_else(|| format!("__metadata__.{}: not a string value", key))?;
                return json_min::unescape(raw)
                    .map(Some)
                    .map_err(|e| format!("__metadata__.{}: {}", key, e));
            }
        }
        Ok(None)
    }

    /// Whether the header names this tensor. Identical to `model.rs:203-205`.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.offsets.contains_key(name)
    }

    /// Identical to `model.rs:207-233`: a truncated or malformed file yields
    /// `Err`, never a slice panic.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'a>, String> {
        if let Some(&(start, end)) = self.offsets.get(name) {
            if start > end {
                return Err(format!(
                    "Tensor {}: inverted offsets {}..{}",
                    name, start, end
                ));
            }
            let abs_start = self.tensor_start + start;
            let abs_end = self.tensor_start + end;
            if abs_end > self.buffer.len() {
                return Err(format!(
                    "Tensor {}: offsets {}..{} exceed buffer ({} bytes) — model file truncated?",
                    name,
                    abs_start,
                    abs_end,
                    self.buffer.len()
                ));
            }
            return Ok(TensorView {
                data_bytes: &self.buffer[abs_start..abs_end],
            });
        }
        Err(format!("Tensor {} not found in dynamic map", name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Build a minimal well-formed MODEL.SAF-shaped buffer for tests.
    fn make_buf(header: &str, tensor_bytes: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(tensor_bytes);
        buf
    }

    #[test]
    fn parses_header_and_slices_tensor_bytes() {
        let header = r#"{"a.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"__metadata__":{"aegis_config":"{\"n\":1}"}}"#;
        let buf = make_buf(header, &[1, 2, 3, 4]);
        let st = SafeTensors::deserialize(&buf).unwrap();
        assert!(st.has_tensor("a.weight"));
        assert!(!st.has_tensor("nope"));
        assert_eq!(st.tensor("a.weight").unwrap().data(), &[1, 2, 3, 4]);
        assert_eq!(
            st.metadata_field("aegis_config").unwrap().as_deref(),
            Some(r#"{"n":1}"#)
        );
        assert_eq!(st.metadata_field("nope").unwrap(), None);
    }

    #[test]
    fn truncated_buffer_is_rejected() {
        assert!(SafeTensors::deserialize(&[0u8; 4]).is_err());
        let header = r#"{"a":{"dtype":"U8","shape":[100],"data_offsets":[0,100]}}"#;
        let buf = make_buf(header, &[1, 2, 3]); // way short of 100 bytes
        let st = SafeTensors::deserialize(&buf).unwrap();
        assert!(st.tensor("a").is_err());
    }

    #[test]
    fn inverted_offsets_are_rejected() {
        let header = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[5,2]}}"#;
        let buf = make_buf(header, &[0u8; 10]);
        let st = SafeTensors::deserialize(&buf).unwrap();
        assert!(st.tensor("a").is_err());
    }
}
