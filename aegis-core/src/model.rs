use alloc::{format, string::String, string::ToString, vec::Vec};

pub struct TensorView<'a> {
    data_bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn data(&self) -> &'a [u8] {
        self.data_bytes
    }
}

/// FFN activation applied to the gate projection. The engine supports the
/// two functions the ternary checkpoint families actually use: BitNet-2B's
/// squared ReLU and the Llama-family SwiGLU (silu on gate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Relu2,
    Silu,
}

/// Prompt wrapping convention for instruct checkpoints. Absent in the
/// config ⇒ `Llama3` (the BitNet-2B artifact chain's convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<|begin_of_text|><|start_header_id|>user…` (Llama-3 headers)
    Llama3,
    /// `<|im_start|>user\n…<|im_end|>` (ChatML — Falcon-E, Qwen)
    ChatMl,
    /// No wrapping: the prompt is encoded verbatim (base models).
    None,
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
    /// FFN gate activation. Absent in config.json ⇒ `Relu2` (BitNet-2B).
    pub hidden_act: Activation,
    /// RoPE base frequency. Absent ⇒ 500000.0 (BitNet-2B / Llama-3).
    pub rope_theta: f32,
    /// RMSNorm epsilon, used at every norm site. Absent ⇒ 1e-5.
    pub rms_norm_eps: f32,
    /// Tied ⇒ logits dot against the embedding table; untied ⇒ a
    /// `lm_head.weight` tensor is REQUIRED at load. Absent ⇒ tied.
    pub tie_word_embeddings: bool,
    /// Prompt wrapping for `process_intent`. Absent ⇒ `Llama3`.
    pub chat_template: ChatTemplate,
}

impl ModelConfig {
    /// Read the fields we need out of a `config.json`.
    ///
    /// The seven integer fields are required. The graph-variant fields
    /// (`hidden_act`, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`)
    /// default to the BitNet-2B conventions when absent — matching every
    /// artifact forged before configs carried them — but a present field
    /// that fails to parse, or an activation this engine has no kernel
    /// for, is an error rather than a silent fallback.
    pub fn from_json(s: &str) -> Result<Self, String> {
        // One members() pass serves every lookup below.
        let members = crate::json::members(s)?;
        let find = |name: &str| members.iter().find(|(k, _)| *k == name).map(|&(_, v)| v);
        let g = |name: &str| -> Result<usize, String> {
            find(name)
                .and_then(crate::json::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| format!("config: missing or non-integer field '{}'", name))
        };

        let hidden_act = match find("hidden_act") {
            None => Activation::Relu2,
            Some(v) => match crate::json::as_str(v) {
                Some("relu2") => Activation::Relu2,
                Some("silu") => Activation::Silu,
                Some(other) => return Err(format!("config: unsupported hidden_act '{}'", other)),
                None => return Err(format!("config: hidden_act is not a string: {}", v)),
            },
        };
        let f32_or = |name: &str, default: f32| -> Result<f32, String> {
            match find(name) {
                None => Ok(default),
                Some(v) => crate::json::as_f64(v)
                    .map(|f| f as f32)
                    .ok_or_else(|| format!("config: {} is not a number: {}", name, v)),
            }
        };
        let tie_word_embeddings = match find("tie_word_embeddings") {
            None => true,
            Some(v) => crate::json::as_bool(v)
                .ok_or_else(|| format!("config: tie_word_embeddings is not a bool: {}", v))?,
        };
        let chat_template = match find("chat_template") {
            None => ChatTemplate::Llama3,
            Some(v) => match crate::json::as_str(v) {
                Some("llama3") => ChatTemplate::Llama3,
                Some("chatml") => ChatTemplate::ChatMl,
                Some("none") => ChatTemplate::None,
                Some(other) => {
                    return Err(format!("config: unsupported chat_template '{}'", other));
                }
                None => return Err(format!("config: chat_template is not a string: {}", v)),
            },
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
            rope_theta: f32_or("rope_theta", 500000.0)?,
            rms_norm_eps: f32_or("rms_norm_eps", 1e-5)?,
            tie_word_embeddings,
            chat_template,
        })
    }
}

use alloc::collections::BTreeMap;

pub struct SafeTensors<'a> {
    buffer: &'a [u8],
    tensor_start: usize,
    offsets: BTreeMap<String, (usize, usize)>,
    /// Raw `__metadata__` value slice from the header, if present.
    metadata: Option<&'a str>,
}

impl<'a> SafeTensors<'a> {
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
        for (name, entry) in crate::json::members(header_str)? {
            if name == "__metadata__" {
                metadata = Some(entry);
                continue;
            }
            // Each entry is {"dtype":..,"shape":[..],"data_offsets":[start,end]}
            for (field, value) in crate::json::members(entry)? {
                if field == "data_offsets" {
                    let (start, end) = crate::json::as_u64_pair(value)
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

    /// Look up one string value in the header's `__metadata__` map. Values
    /// arrive JSON-escaped (a config.json stored there has every quote as
    /// `\"`), so they are unescaped here. `Ok(None)` means genuinely absent
    /// (no metadata or no such key); a key that is PRESENT but malformed is
    /// an error — a config that fails to decode must never silently look
    /// like "no config, use the baked default".
    pub fn metadata_field(&self, key: &str) -> Result<Option<String>, String> {
        let Some(meta) = self.metadata else {
            return Ok(None);
        };
        for (k, v) in crate::json::members(meta)? {
            if k == key {
                let raw = crate::json::as_str(v)
                    .ok_or_else(|| format!("__metadata__.{}: not a string value", key))?;
                return crate::json::unescape(raw)
                    .map(Some)
                    .map_err(|e| format!("__metadata__.{}: {}", key, e));
            }
        }
        Ok(None)
    }

    /// Whether the header names this tensor. Header presence only — the
    /// bytes are not validated until `tensor()` slices them.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.offsets.contains_key(name)
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'a>, String> {
        if let Some(&(start, end)) = self.offsets.get(name) {
            // A truncated or malformed file must yield Err, not a slice panic —
            // on bare metal a panic is an unrecoverable fault.
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

pub struct DecoderLayer<'a> {
    pub layer_idx: usize,
    pub input_layernorm_weight: TensorView<'a>,
    /// BitNet SubLN before the o_proj. Absent in checkpoints trained
    /// without it (Falcon-E); the forward paths skip the norm when `None`.
    pub attn_sub_norm: Option<TensorView<'a>>,
    pub post_attention_layernorm_weight: TensorView<'a>,
    /// BitNet SubLN before the down_proj. Same optionality as above.
    pub ffn_sub_norm: Option<TensorView<'a>>,
    pub q_proj: TensorView<'a>,
    pub q_proj_scale: f32,
    pub k_proj: TensorView<'a>,
    pub k_proj_scale: f32,
    pub v_proj: TensorView<'a>,
    pub v_proj_scale: f32,
    pub o_proj: TensorView<'a>,
    pub o_proj_scale: f32,
    pub gate_proj: TensorView<'a>,
    pub gate_proj_scale: f32,
    pub up_proj: TensorView<'a>,
    pub up_proj_scale: f32,
    pub down_proj: TensorView<'a>,
    pub down_proj_scale: f32,
}

impl<'a> DecoderLayer<'a> {
    fn read_f32(tensors: &SafeTensors<'a>, name: &str) -> Result<f32, String> {
        let view = tensors.tensor(name)?;
        let data = view.data();
        if data.len() < 4 {
            return Err(format!("{} too small for f32", name));
        }
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&data[0..4]);
        Ok(f32::from_le_bytes(bytes))
    }

    /// Load a sub-norm that is allowed to be absent — but never half-broken:
    /// a present tensor that fails to load is still an error.
    fn read_optional(
        tensors: &SafeTensors<'a>,
        name: &str,
    ) -> Result<Option<TensorView<'a>>, String> {
        if tensors.has_tensor(name) {
            Ok(Some(tensors.tensor(name)?))
        } else {
            Ok(None)
        }
    }

    pub fn new(
        tensors: &SafeTensors<'a>,
        layer_idx: usize,
        subln_required: bool,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        // This engine's projections are bias-free. A checkpoint that carries
        // biases (e.g. Qwen-family QKV) would silently lose them — refuse it.
        for proj in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            let bias = format!("{}.{}.bias", prefix, proj);
            if tensors.has_tensor(&bias) {
                return Err(format!(
                    "Tensor {} present: this engine has no bias path; refusing to drop weights",
                    bias
                ));
            }
        }

        let attn_sub_norm = Self::read_optional(
            tensors,
            &format!("{}.self_attn.attn_sub_norm.weight", prefix),
        )?;
        let ffn_sub_norm =
            Self::read_optional(tensors, &format!("{}.mlp.ffn_sub_norm.weight", prefix))?;
        // SubLN is optional only for families trained without it (silu /
        // Falcon-E). A BitNet-family (relu2) checkpoint missing its SubLN
        // pair is a truncated or mis-forged artifact — the old always-required
        // load check must not silently relax for the family that needs it.
        if subln_required && (attn_sub_norm.is_none() || ffn_sub_norm.is_none()) {
            return Err(format!(
                "{}: BitNet (relu2) checkpoints carry attn/ffn sub_norm tensors; missing here — truncated or mis-forged MODEL.SAF",
                prefix
            ));
        }

        Ok(Self {
            layer_idx,
            input_layernorm_weight: tensors
                .tensor(&format!("{}.input_layernorm.weight", prefix))?,
            attn_sub_norm,
            post_attention_layernorm_weight: tensors
                .tensor(&format!("{}.post_attention_layernorm.weight", prefix))?,
            ffn_sub_norm,
            q_proj: tensors.tensor(&format!("{}.self_attn.q_proj.weight", prefix))?,
            q_proj_scale: Self::read_f32(
                tensors,
                &format!("{}.self_attn.q_proj.weight_scale", prefix),
            )?,
            k_proj: tensors.tensor(&format!("{}.self_attn.k_proj.weight", prefix))?,
            k_proj_scale: Self::read_f32(
                tensors,
                &format!("{}.self_attn.k_proj.weight_scale", prefix),
            )?,
            v_proj: tensors.tensor(&format!("{}.self_attn.v_proj.weight", prefix))?,
            v_proj_scale: Self::read_f32(
                tensors,
                &format!("{}.self_attn.v_proj.weight_scale", prefix),
            )?,
            o_proj: tensors.tensor(&format!("{}.self_attn.o_proj.weight", prefix))?,
            o_proj_scale: Self::read_f32(
                tensors,
                &format!("{}.self_attn.o_proj.weight_scale", prefix),
            )?,
            gate_proj: tensors.tensor(&format!("{}.mlp.gate_proj.weight", prefix))?,
            gate_proj_scale: Self::read_f32(
                tensors,
                &format!("{}.mlp.gate_proj.weight_scale", prefix),
            )?,
            up_proj: tensors.tensor(&format!("{}.mlp.up_proj.weight", prefix))?,
            up_proj_scale: Self::read_f32(
                tensors,
                &format!("{}.mlp.up_proj.weight_scale", prefix),
            )?,
            down_proj: tensors.tensor(&format!("{}.mlp.down_proj.weight", prefix))?,
            down_proj_scale: Self::read_f32(
                tensors,
                &format!("{}.mlp.down_proj.weight_scale", prefix),
            )?,
        })
    }
}

pub struct FullBitNetPipeline<'a> {
    pub embeddings: &'a [u8],
    pub layers: Vec<DecoderLayer<'a>>,
    pub final_norm: TensorView<'a>,
    pub hidden_dim: usize,
    /// Untied LM head, present iff `tie_word_embeddings` is false. When
    /// `None`, logits are computed against the embedding table as before.
    pub lm_head: Option<TensorView<'a>>,
}

impl<'a> FullBitNetPipeline<'a> {
    pub fn new(
        tensors: &SafeTensors<'a>,
        embeddings_bytes: &'a [u8],
        config: &ModelConfig,
    ) -> Result<Self, String> {
        let mut layers = Vec::new();
        let subln_required = config.hidden_act == Activation::Relu2;
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(tensors, i, subln_required)?);
        }
        let final_norm = tensors.tensor("model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            let head = tensors.tensor("lm_head.weight")?;
            // One invariant, one check: the logit kernel is BF16-only, so
            // the head must be exactly vocab x hidden x 2 bytes; anything
            // else (wrong dtype, truncation) must be normalized by the
            // forge, not accepted here.
            let expect = config.vocab_size * config.hidden_size * 2;
            if head.data().len() != expect {
                return Err(format!(
                    "lm_head.weight: {} bytes, expected {} ({} x {} BF16) — repack in the forge",
                    head.data().len(),
                    expect,
                    config.vocab_size,
                    config.hidden_size
                ));
            }
            Some(head)
        };

        Ok(Self {
            embeddings: embeddings_bytes,
            layers,
            final_norm,
            hidden_dim: config.hidden_size,
            lm_head,
        })
    }

    /// Bytes the logit projection dots against: the untied head when the
    /// checkpoint has one, otherwise the tied embedding table.
    pub fn lm_head_bytes(&self) -> &'a [u8] {
        match &self.lm_head {
            Some(t) => t.data(),
            None => self.embeddings,
        }
    }
}
