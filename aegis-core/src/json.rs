//! A minimal JSON reader, sized to exactly what this engine parses.
//!
//! It replaces `serde` + `serde_json` + `serde_core` + `memchr` — roughly 27,700
//! lines of dependency source — with ~150. That matters not for the shipped
//! binary (dead-code elimination strips most of a JSON library anyway) but for
//! the *auditable* trusted base: source you depend on is source someone must read
//! before they will certify this thing.
//!
//! Two jobs, and no more:
//!   1. The safetensors header: `{"name": {"dtype":.., "shape":[..],
//!      "data_offsets":[start,end]}, ...}` — we need only the offsets.
//!   2. `config.json`: a handful of unsigned integer fields.
//!
//! Deliberate limitations, stated so nobody mistakes this for a JSON library:
//!   - Strings are returned as raw slices and are **not unescaped** by default.
//!     No key this engine reads contains an escape sequence (tensor names are
//!     `model.layers.0.self_attn.q_proj.weight`; dtypes are `F32`, `U8`, `BF16`).
//!     An escaped `\"` is skipped correctly so parsing never desynchronizes, but
//!     the returned slice keeps the backslashes. The one place escaped content
//!     is legitimate — a config.json stored as a string value inside the
//!     safetensors `__metadata__` map — goes through `unescape()` explicitly.
//!   - Numbers parse as `u64` (`as_u64`) or as `f64` (`as_f64`, for config
//!     fields like `rope_theta: 500000.0` and `rms_norm_eps: 1e-05`).
//!   - No `serde` data model, no deserialization, no error recovery.

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

/// One find-and-parse loop for every typed field accessor: the four public
/// `*_field` functions differ only in the parser and the type name that
/// appears in the error.
fn typed_field<'a, T>(
    json: &'a str,
    name: &str,
    parse: impl Fn(&'a str) -> Option<T>,
    ty: &str,
) -> R<T> {
    for (k, v) in members(json)? {
        if k == name {
            return parse(v).ok_or_else(|| format!("json: field '{}' is not {}: {}", name, ty, v));
        }
    }
    Err(format!("json: missing field '{}'", name))
}

/// Find one top-level unsigned integer field by name.
pub fn u64_field(json: &str, name: &str) -> R<u64> {
    typed_field(json, name, as_u64, "a u64")
}

/// Parse a JSON number as f64. Bare integers parse too, so
/// `"rope_theta": 500000` and `500000.0` agree.
///
/// The heavy lifting is `core`'s correctly-rounded `FromStr for f64`
/// (available under no_std); the guards restrict it to the JSON number
/// grammar, since `parse` alone also accepts `inf`, `NaN`, `+1`, `.5`.
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
/// quotes escaped. Handles the escapes a JSON writer emits for such content
/// (`\"`, `\\`, `\/`, `\n`, `\t`, `\r`); anything else is an error rather
/// than silent corruption. `\uXXXX` is rejected: config.json is ASCII.
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

/// Find one top-level float field by name.
pub fn f64_field(json: &str, name: &str) -> R<f64> {
    typed_field(json, name, as_f64, "a number")
}

/// Find one top-level string field by name (raw, not unescaped).
pub fn str_field<'a>(json: &'a str, name: &str) -> R<&'a str> {
    typed_field(json, name, as_str, "a string")
}

/// Find one top-level boolean field by name.
pub fn bool_field(json: &str, name: &str) -> R<bool> {
    typed_field(json, name, as_bool, "a bool")
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
        assert_eq!(u64_field(c, "hidden_size").unwrap(), 2560);
        assert!(u64_field(c, "eps").is_err());
        assert!(u64_field(c, "absent").is_err());
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
        assert_eq!(as_f64("0.02"), Some(0.02));
        assert_eq!(as_f64("1e400"), Some(f64::INFINITY));
        assert_eq!(as_f64(""), None);
        assert_eq!(as_f64("-"), None);
        assert_eq!(as_f64("."), None);
        assert_eq!(as_f64("e5"), None);
        assert_eq!(as_f64("1.5x"), None);
        assert_eq!(as_f64("true"), None);
    }

    #[test]
    fn typed_field_lookups() {
        let c = r#"{"hidden_act":"silu","rope_theta":1000042.0,"rms_norm_eps":1e-06,
                    "tie_word_embeddings":false,"hidden_size":2048}"#;
        assert_eq!(str_field(c, "hidden_act").unwrap(), "silu");
        assert_eq!(f64_field(c, "rope_theta").unwrap(), 1000042.0);
        assert_eq!(f64_field(c, "rms_norm_eps").unwrap(), 1e-06);
        assert!(!bool_field(c, "tie_word_embeddings").unwrap());
        assert!(str_field(c, "rope_theta").is_err()); // wrong type
        assert!(f64_field(c, "hidden_act").is_err());
        assert!(bool_field(c, "hidden_act").is_err());
        assert!(str_field(c, "absent").is_err());
    }

    #[test]
    fn unescape_round_trips_embedded_config() {
        // A config.json as it appears when stored as a string value inside
        // the safetensors __metadata__ map: quotes escaped.
        let escaped = r#"{\"hidden_act\":\"silu\",\"n\":1}"#;
        let clean = unescape(escaped).unwrap();
        assert_eq!(clean, r#"{"hidden_act":"silu","n":1}"#);
        assert_eq!(str_field(&clean, "hidden_act").unwrap(), "silu");
        assert_eq!(unescape(r"a\nb").unwrap(), "a\nb");
        assert_eq!(unescape("no escapes").unwrap(), "no escapes");
        assert!(unescape(r"bad \x escape").is_err());
        assert!(unescape("\\u0041").is_err()); // config.json is ASCII; unicode escapes unsupported
        assert!(unescape("dangling \\").is_err());
    }
}
