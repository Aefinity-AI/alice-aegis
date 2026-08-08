import os
import re

model_rs_path = "/home/killboxincorporated/aegis-core/src/model.rs"
inference_rs_path = "/home/killboxincorporated/aegis-core/src/inference.rs"

# 1. Update model.rs
with open(model_rs_path, "r") as f:
    model_content = f.read()

# Add ModelConfig struct
struct_def = """
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}
"""

if "pub struct ModelConfig" not in model_content:
    model_content = model_content.replace("use alloc::collections::BTreeMap;", struct_def + "\nuse alloc::collections::BTreeMap;")

# Update BitNetModel::new
model_content = model_content.replace(
    "pub fn new(tensors: SafeTensors<'a>) -> Self {",
    "pub fn new(tensors: SafeTensors<'a>, vocab_size: usize) -> Self {"
)
model_content = model_content.replace(
    "vocab_size: 128256,",
    "vocab_size,"
)

with open(model_rs_path, "w") as f:
    f.write(model_content)

# 2. Update inference.rs
with open(inference_rs_path, "r") as f:
    inf_content = f.read()

if "pub config: crate::model::ModelConfig," not in inf_content:
    inf_content = inf_content.replace(
        "pipeline: FullBitNetPipeline<'a>,",
        "pipeline: FullBitNetPipeline<'a>,\n    pub config: crate::model::ModelConfig,"
    )

new_fn_setup = """
        let config_str = include_str!("../../aegis-forge/aegis_pruned_config.json");
        let mut config: crate::model::ModelConfig = serde_json::from_str(config_str).map_err(|e| alloc::format!("Config parse error: {}", e))?;
        config.vocab_size = tokenizer.vocab_len();

        let emb_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = emb_dim / num_heads;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let max_seq = config.max_position_embeddings;
"""

inf_content = re.sub(
    r"let emb_dim = pipeline.hidden_dim;[\s\S]*?let max_seq = 2048;",
    new_fn_setup.strip(),
    inf_content
)

inf_content = inf_content.replace(
    "Ok(Self { pipeline, tokenizer, kv_cache, sampler, rope_cache, arena })",
    "Ok(Self { pipeline, config, tokenizer, kv_cache, sampler, rope_cache, arena })"
)

# forward_batch update
inf_content = re.sub(
    r"let emb_dim = self\.pipeline\.hidden_dim;\s+let num_heads = 20;\s+let num_kv_heads = 5;\s+let head_dim = emb_dim / num_heads;\s+let intermediate_size = 6912;",
    """let emb_dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = emb_dim / num_heads;
        let intermediate_size = self.config.intermediate_size;""",
    inf_content
)

# forward_step update
inf_content = re.sub(
    r"let emb_dim = self\.pipeline\.hidden_dim;\s+let num_heads = 20;\s+let num_kv_heads = 5;\s+let head_dim = emb_dim / num_heads;\s+let intermediate_size = 6912;",
    """let emb_dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = emb_dim / num_heads;
        let intermediate_size = self.config.intermediate_size;""",
    inf_content
)


with open(inference_rs_path, "w") as f:
    f.write(inf_content)

print("Directive 2 done.")
