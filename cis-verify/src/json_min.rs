//! A minimal JSON reader, sized to exactly what `safetensors.rs`/`config.rs`
//! need — the narrow slice `docs/design/CIS_VERIFY_DESIGN.md` §3.3 describes
//! for `json_min.rs`: object members, string unescape, u64 pairs. Transcribed
//! from `aegis-core/src/json.rs` (414 lines total including comments/tests),
//! independent copy per this crate's "vendor, don't depend" policy (design
//! doc §3.1 Option A) — no `path` dependency on `aegis-core`.
//!
//! Two jobs, and no more, exactly as the original states:
//!   1. The safetensors header: `{"name": {"dtype":.., "shape":[..],
//!      "data_offsets":[start,end]}, ...}` — only the offsets are needed.
//!   2. `aegis_config`'s embedded config.json: a handful of scalar fields.
//!
//! Same deliberate limitations as the original: strings are raw slices, not
//! unescaped by default (an escaped `\"` is skipped correctly so parsing
//! never desynchronizes, but the returned slice keeps the backslashes);
//! numbers parse as `u64` or `f64`; no `serde` data model.

use alloc::{format, string::String, vec::Vec};

type R<T> = Result<T, String>;

struct P<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    #[inline]
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    #[inline]
    fn peek(&self) -> u8 {
        if self.i < self.b.len() {
            self.b[self.i]
        } else {
            0
        }
    }

    fn expect(&mut self, c: u8) -> R<()> {
        self.ws();
        if self.peek() == c {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("json: expected '{}' at byte {}", c as char, self.i))
        }
    }

    /// Raw (still-escaped) contents of a string literal.
    fn string(&mut self) -> R<&'a str> {
        self.ws();
        if self.peek() != b'"' {
            return Err(format!("json: expected string at byte {}", self.i));
        }
        self.i += 1;
        let start = self.i;
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'\\' => self.i += 2, // skip the escape pair; never desyncs on \"
                b'"' => {
                    let s = &self.b[start..self.i];
                    self.i += 1;
                    return core::str::from_utf8(s)
                        .map_err(|_| String::from("json: invalid utf-8"));
                }
                _ => self.i += 1,
            }
        }
        Err(String::from("json: unterminated string"))
    }

    /// Advance past exactly one value of any type.
    fn skip_value(&mut self) -> R<()> {
        self.ws();
        match self.peek() {
            b'"' => {
                self.string()?;
            }
            b'{' => {
                self.i += 1;
                loop {
                    self.ws();
                    if self.peek() == b'}' {
                        self.i += 1;
                        break;
                    }
                    self.string()?;
                    self.expect(b':')?;
                    self.skip_value()?;
                    self.ws();
                    match self.peek() {
                        b',' => self.i += 1,
                        b'}' => {}
                        _ => return Err(format!("json: bad object at byte {}", self.i)),
                    }
                }
            }
            b'[' => {
                self.i += 1;
                loop {
                    self.ws();
                    if self.peek() == b']' {
                        self.i += 1;
                        break;
                    }
                    self.skip_value()?;
                    self.ws();
                    match self.peek() {
                        b',' => self.i += 1,
                        b']' => {}
                        _ => return Err(format!("json: bad array at byte {}", self.i)),
                    }
                }
            }
            0 => return Err(String::from("json: unexpected end of input")),
            _ => {
                // number | true | false | null
                let start = self.i;
                while self.i < self.b.len()
                    && !matches!(
                        self.b[self.i],
                        b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r'
                    )
                {
                    self.i += 1;
                }
                if self.i == start {
                    return Err(format!("json: bad literal at byte {}", self.i));
                }
            }
        }
        Ok(())
    }
}

/// `(key, raw_value_slice)` for every member of a top-level JSON object.
/// The value slice is exact — suitable for re-parsing.
pub fn members(s: &str) -> R<Vec<(&str, &str)>> {
    let mut p = P {
        b: s.as_bytes(),
        i: 0,
    };
    p.expect(b'{')?;
    let mut out = Vec::new();
    loop {
        p.ws();
        if p.peek() == b'}' {
            break; // parse ends at the closing brace; nothing reads past it
        }
        let key = p.string()?;
        p.expect(b':')?;
        p.ws();
        let start = p.i;
        p.skip_value()?;
        out.push((key, &s[start..p.i]));
        p.ws();
        match p.peek() {
            b',' => p.i += 1,
            b'}' => break, // parse ends at the closing brace
            _ => return Err(format!("json: bad top-level object at byte {}", p.i)),
        }
    }
    Ok(out)
}

/// Parse a bare unsigned integer, e.g. the `2560` in `"hidden_size": 2560`.
pub fn as_u64(s: &str) -> Option<u64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let mut n: u64 = 0;
    for c in t.bytes() {
        if !c.is_ascii_digit() {
            return None;
        }
        n = n.checked_mul(10)?.checked_add((c - b'0') as u64)?;
    }
    Some(n)
}

/// Extract `[a, b]` from a two-element integer array such as `data_offsets`.
pub fn as_u64_pair(s: &str) -> Option<(u64, u64)> {
    let t = s.trim();
    let inner = t.strip_prefix('[')?.strip_suffix(']')?;
    let (a, b) = inner.split_once(',')?;
    Some((as_u64(a)?, as_u64(b)?))
}

/// Parse a JSON number as f64. Bare integers parse too.
pub fn as_f64(s: &str) -> Option<f64> {
    let t = s.trim();
    let first = *t.as_bytes().first()?;
    if !(first.is_ascii_digit() || first == b'-') {
        return None;
    }
    if !t
        .bytes()
        .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'))
    {
        return None; // rejects inf/NaN/true and any trailing junk
    }
    t.parse::<f64>().ok()
}

/// Strip the quotes off a raw string value slice. The content is NOT
/// unescaped — pass it through `unescape()` if it may contain escapes.
pub fn as_str(s: &str) -> Option<&str> {
    let t = s.trim();
    t.strip_prefix('"')?.strip_suffix('"')
}

/// Parse a JSON boolean literal.
pub fn as_bool(s: &str) -> Option<bool> {
    match s.trim() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Resolve JSON string escapes. Needed exactly once: a config.json stored as
/// a string value inside the safetensors `__metadata__` map arrives with its
/// quotes escaped. Handles `\"`, `\\`, `\/`, `\n`, `\t`, `\r`; anything else
/// is an error rather than silent corruption. `\uXXXX` is rejected:
/// config.json is ASCII.
pub fn unescape(s: &str) -> R<String> {
    let b = s.as_bytes();
    let mut out = String::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' {
            let esc = *b.get(i + 1).ok_or("json: dangling escape")?;
            out.push(match esc {
                b'"' => '"',
                b'\\' => '\\',
                b'/' => '/',
                b'n' => '\n',
                b't' => '\t',
                b'r' => '\r',
                _ => return Err(format!("json: unsupported escape '\\{}'", esc as char)),
            });
            i += 2;
        } else {
            // Multi-byte UTF-8 passes through untouched: '\\' never occurs
            // inside a UTF-8 continuation sequence.
            let ch_len = match b[i] {
                0x00..=0x7f => 1,
                0xc0..=0xdf => 2,
                0xe0..=0xef => 3,
                _ => 4,
            };
            out.push_str(&s[i..(i + ch_len).min(s.len())]);
            i += ch_len;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_safetensors_style_header() {
        let h = r#"{"__metadata__":{"format":"pt"},
                    "a.weight":{"dtype":"F32","shape":[2560],"data_offsets":[0,10240]},
                    "b.weight":{"dtype":"U8","shape":[16,4],"data_offsets":[10240,10304]}}"#;
        let m = members(h).unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].0, "__metadata__");

        let inner = members(m[1].1).unwrap();
        let off = inner.iter().find(|(k, _)| *k == "data_offsets").unwrap().1;
        assert_eq!(as_u64_pair(off), Some((0, 10240)));

        let inner = members(m[2].1).unwrap();
        let off = inner.iter().find(|(k, _)| *k == "data_offsets").unwrap().1;
        assert_eq!(as_u64_pair(off), Some((10240, 10304)));
    }

    #[test]
    fn skips_floats_bools_nulls_and_nested_junk() {
        let c =
            r#"{"eps":1e-05,"ok":true,"nil":null,"nest":{"a":[1,{"b":[]}]},"hidden_size":2560}"#;
        let m = members(c).unwrap();
        let v = m.iter().find(|(k, _)| *k == "hidden_size").unwrap().1;
        assert_eq!(as_u64(v), Some(2560));
        assert!(m.iter().any(|(k, _)| *k == "eps"));
    }

    #[test]
    fn escaped_quote_does_not_desynchronize() {
        let s = r#"{"a\"b":1,"c":2}"#;
        let m = members(s).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[1].0, "c");
        assert_eq!(as_u64(m[1].1), Some(2));
    }

    #[test]
    fn rejects_malformed_input_instead_of_panicking() {
        assert!(members("{").is_err());
        assert!(members(r#"{"a":}"#).is_err());
        assert!(members(r#"{"a":1"#).is_err());
        assert!(members(r#"{"a" 1}"#).is_err());
        assert!(members("").is_err());
    }

    #[test]
    fn empty_object() {
        assert_eq!(members("{}").unwrap().len(), 0);
    }

    #[test]
    fn parses_config_floats() {
        assert_eq!(as_f64("500000.0"), Some(500000.0));
        assert_eq!(as_f64("500000"), Some(500000.0));
        assert_eq!(as_f64("1e-05"), Some(1e-05));
        assert_eq!(as_f64("1E-05"), Some(1e-05));
        assert_eq!(as_f64("-2.5e3"), Some(-2500.0));
        assert_eq!(as_f64(""), None);
        assert_eq!(as_f64("true"), None);
    }

    #[test]
    fn unescape_round_trips_embedded_config() {
        let escaped = r#"{\"hidden_act\":\"silu\",\"n\":1}"#;
        let clean = unescape(escaped).unwrap();
        assert_eq!(clean, r#"{"hidden_act":"silu","n":1}"#);
        assert_eq!(unescape(r"a\nb").unwrap(), "a\nb");
        assert_eq!(unescape("no escapes").unwrap(), "no escapes");
        assert!(unescape(r"bad \x escape").is_err());
        assert!(unescape("dangling \\").is_err());
    }
}
