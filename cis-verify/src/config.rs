//! `aegis_config` metadata string → `ModelConfig`. Transcribed from
//! `aegis-core/src/model.rs:16-126` per
//! `docs/design/CIS_VERIFY_DESIGN.md` builder task 5 — independent copy, no
//! `path` dependency on `aegis-core`.

use alloc::format;
use alloc::string::String;

use crate::json_min;

/// FFN activation applied to the gate projection — the two functions the
/// ternary checkpoint families actually use. Identical to `model.rs:16-20`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Relu2,
    Silu,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    /// Absent in config.json ⇒ `Relu2` (BitNet-2B).
    pub hidden_act: Activation,
    /// Absent ⇒ 500000.0 (BitNet-2B / Llama-3).
    pub rope_theta: f32,
    /// Tied ⇒ logits dot against the embedding table; untied ⇒ a
    /// `lm_head.weight` tensor is required. Absent ⇒ tied.
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    /// Read the fields the verifier needs out of `aegis_config`'s embedded
    /// config.json. Seven integer fields are required; the graph-variant
    /// fields default to the BitNet-2B conventions when absent (identical
    /// defaults to `model.rs:56-125`) but a present-and-malformed field is
    /// an error, never a silent fallback.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let members = json_min::members(s)?;
        let find = |name: &str| members.iter().find(|(k, _)| *k == name).map(|&(_, v)| v);
        let g = |name: &str| -> Result<usize, String> {
            find(name)
                .and_then(json_min::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| format!("config: missing or non-integer field '{}'", name))
        };

        let hidden_act = match find("hidden_act") {
            None => Activation::Relu2,
            Some(v) => match json_min::as_str(v) {
                Some("relu2") => Activation::Relu2,
                Some("silu") => Activation::Silu,
                Some(other) => return Err(format!("config: unsupported hidden_act '{}'", other)),
                None => return Err(format!("config: hidden_act is not a string: {}", v)),
            },
        };
        let rope_theta = match find("rope_theta") {
            None => 500000.0,
            Some(v) => json_min::as_f64(v)
                .map(|f| f as f32)
                .ok_or_else(|| format!("config: rope_theta is not a number: {}", v))?,
        };
        let tie_word_embeddings = match find("tie_word_embeddings") {
            None => true,
            Some(v) => json_min::as_bool(v)
                .ok_or_else(|| format!("config: tie_word_embeddings is not a bool: {}", v))?,
        };

        Ok(Self {
            num_hidden_layers: g("num_hidden_layers")?,
            hidden_size: g("hidden_size")?,
            num_attention_heads: g("num_attention_heads")?,
            num_key_value_heads: g("num_key_value_heads")?,
            intermediate_size: g("intermediate_size")?,
            vocab_size: g("vocab_size")?,
            max_position_embeddings: g("max_position_embeddings")?,
            hidden_act,
            rope_theta,
            tie_word_embeddings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M7: &str = r#"{"num_hidden_layers":7,"hidden_size":384,"num_attention_heads":6,"num_key_value_heads":2,"intermediate_size":1024,"vocab_size":8192,"max_position_embeddings":512,"hidden_act":"silu","rope_theta":10000.0,"rms_norm_eps":1e-05,"tie_word_embeddings":true,"chat_template":"none"}"#;

    #[test]
    fn parses_m7_config() {
        let c = ModelConfig::from_json(M7).unwrap();
        assert_eq!(c.num_hidden_layers, 7);
        assert_eq!(c.hidden_size, 384);
        assert_eq!(c.num_attention_heads, 6);
        assert_eq!(c.num_key_value_heads, 2);
        assert_eq!(c.intermediate_size, 1024);
        assert_eq!(c.vocab_size, 8192);
        assert_eq!(c.max_position_embeddings, 512);
        assert_eq!(c.hidden_act, Activation::Silu);
        assert_eq!(c.rope_theta, 10000.0f32);
        assert!(c.tie_word_embeddings);
    }

    #[test]
    fn defaults_apply_when_fields_absent() {
        let minimal = r#"{"num_hidden_layers":1,"hidden_size":4,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":4,"vocab_size":4,"max_position_embeddings":4}"#;
        let c = ModelConfig::from_json(minimal).unwrap();
        assert_eq!(c.hidden_act, Activation::Relu2);
        assert_eq!(c.rope_theta, 500000.0f32);
        assert!(c.tie_word_embeddings);
    }

    #[test]
    fn missing_required_field_errors() {
        assert!(ModelConfig::from_json(r#"{"hidden_size":4}"#).is_err());
    }

    #[test]
    fn unsupported_activation_errors() {
        let bad = r#"{"num_hidden_layers":1,"hidden_size":4,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":4,"vocab_size":4,"max_position_embeddings":4,"hidden_act":"gelu"}"#;
        assert!(ModelConfig::from_json(bad).is_err());
    }
}
