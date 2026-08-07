import os
import re

# Fix inference.rs
with open("/home/killboxincorporated/aegis-uefi/src/inference.rs", "r") as f:
    content = f.read()

content = content.replace("use std::arch::x86_64::*;\n", "")
content = content.replace("use std::fs::File;\n", "")

struct_decl = """pub struct TernaryInferenceEngine<'a> {
    _embeddings_mmap: Mmap,
    pipeline: FullBitNetPipeline<'a>,
    tokenizer: AegisTokenizer,
}"""
new_struct_decl = """use crate::tokenizer::AegisTokenizer;

pub struct TernaryInferenceEngine<'a> {
    pipeline: FullBitNetPipeline<'a>,
    tokenizer: AegisTokenizer,
    kv_cache: KVCache,
    sampler: Sampler,
}"""
content = content.replace(struct_decl, new_struct_decl)

init_block = """        Ok(Self {
            pipeline,
            tokenizer,
        })"""
new_init_block = """        let kv_cache = KVCache::new(26, 32, 64, 4096);
        let sampler = Sampler::new(0.0, 1.0);
        Ok(Self {
            pipeline,
            tokenizer,
            kv_cache,
            sampler,
        })"""
content = content.replace(init_block, new_init_block)

with open("/home/killboxincorporated/aegis-uefi/src/inference.rs", "w") as f:
    f.write(content)

# Fix ops.rs std::arch
with open("/home/killboxincorporated/aegis-uefi/src/ops.rs", "r") as f:
    content = f.read()
content = content.replace("use std::arch::x86_64::*;\n", "")
with open("/home/killboxincorporated/aegis-uefi/src/ops.rs", "w") as f:
    f.write(content)

# Fix tokenizer.rs HashMap
with open("/home/killboxincorporated/aegis-uefi/src/tokenizer.rs", "r") as f:
    content = f.read()
content = content.replace("HashMap<u32, String> = HashMap::new()", "BTreeMap<u32, String> = alloc::collections::BTreeMap::new()")
with open("/home/killboxincorporated/aegis-uefi/src/tokenizer.rs", "w") as f:
    f.write(content)

print("Final fixes applied")
