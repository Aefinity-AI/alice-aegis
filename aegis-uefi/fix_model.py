import os

content = """#![no_std]
extern crate alloc;
use alloc::{vec::Vec, string::String, vec, format, string::ToString, sync::Arc, boxed::Box};
use serde_json::Value;

pub struct TensorView<'a> {
    data_bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn data(&self) -> &'a [u8] {
        self.data_bytes
    }
}

pub struct SafeTensors<'a> {
    metadata: Value,
    buffer: &'a [u8],
    tensor_start: usize,
}

impl<'a> SafeTensors<'a> {
    pub fn deserialize(buffer: &'a [u8]) -> Result<Self, String> {
        if buffer.len() < 8 { return Err("Buffer too small".into()); }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buffer[0..8]);
        let n = u64::from_le_bytes(len_bytes) as usize;
        if buffer.len() < 8 + n { return Err("Buffer smaller than header".into()); }
        let json_bytes = &buffer[8..8+n];
        let metadata: Value = serde_json::from_slice(json_bytes).map_err(|e| format!("Invalid JSON: {}", e))?;
        Ok(Self {
            metadata,
            buffer,
            tensor_start: 8 + n,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'a>, String> {
        let obj = self.metadata.get(name).ok_or_else(|| format!("Tensor {} not found", name))?;
        let offsets = obj.get("data_offsets").ok_or("No offsets")?.as_array().ok_or("Invalid offsets")?;
        let start = self.tensor_start + offsets[0].as_u64().unwrap() as usize;
        let end = self.tensor_start + offsets[1].as_u64().unwrap() as usize;
        Ok(TensorView { data_bytes: &self.buffer[start..end] })
    }
}

pub struct BitNetModel<'a> {
    pub tensors: SafeTensors<'a>,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

impl<'a> BitNetModel<'a> {
    pub fn new(tensors: SafeTensors<'a>) -> Self {
        Self {
            tensors,
            num_layers: 26,
            hidden_size: 2048,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            intermediate_size: 8192,
            vocab_size: 100352, 
            max_position_embeddings: 4096,
        }
    }

    pub fn get_tensor(&self, name: &str) -> Option<TensorView<'a>> {
        self.tensors.tensor(name).ok()
    }
}

pub struct DecoderLayer<'a> {
    pub layer_idx: usize,
    pub input_layernorm_weight: TensorView<'a>,
    pub post_attention_layernorm_weight: TensorView<'a>,
    pub q_proj: TensorView<'a>,
    pub k_proj: TensorView<'a>,
    pub v_proj: TensorView<'a>,
    pub o_proj: TensorView<'a>,
    pub gate_proj: TensorView<'a>,
    pub up_proj: TensorView<'a>,
    pub down_proj: TensorView<'a>,
}

impl<'a> DecoderLayer<'a> {
    pub fn new(tensors: &SafeTensors<'a>, layer_idx: usize) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);
        Ok(Self {
            layer_idx,
            input_layernorm_weight: tensors.tensor(&format!("{}.input_layernorm.weight", prefix))?,
            post_attention_layernorm_weight: tensors.tensor(&format!("{}.post_attention_layernorm.weight", prefix))?,
            q_proj: tensors.tensor(&format!("{}.self_attn.q_proj.weight", prefix))?,
            k_proj: tensors.tensor(&format!("{}.self_attn.k_proj.weight", prefix))?,
            v_proj: tensors.tensor(&format!("{}.self_attn.v_proj.weight", prefix))?,
            o_proj: tensors.tensor(&format!("{}.self_attn.o_proj.weight", prefix))?,
            gate_proj: tensors.tensor(&format!("{}.mlp.gate_proj.weight", prefix))?,
            up_proj: tensors.tensor(&format!("{}.mlp.up_proj.weight", prefix))?,
            down_proj: tensors.tensor(&format!("{}.mlp.down_proj.weight", prefix))?,
        })
    }
}

pub struct FullBitNetPipeline<'a> {
    pub embeddings: &'a [u8], 
    pub layers: Vec<DecoderLayer<'a>>,
    pub final_norm: TensorView<'a>,
    pub hidden_dim: usize,
}

impl<'a> FullBitNetPipeline<'a> {
    pub fn new(tensors: &SafeTensors<'a>, embeddings_bytes: &'a [u8]) -> Result<Self, String> {
        let mut layers = Vec::new();
        for i in 0..30 {
            layers.push(DecoderLayer::new(tensors, i)?);
        }
        let final_norm = tensors.tensor("model.norm.weight")?;
        Ok(Self {
            embeddings: embeddings_bytes,
            layers,
            final_norm,
            hidden_dim: 2560,
        })
    }
}
"""

with open("/home/killboxincorporated/aegis-uefi/src/model.rs", "w") as f:
    f.write(content)
print("Updated model.rs with custom SafeTensors parser.")
