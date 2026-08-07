import os

def replace_in_file(path, replacements):
    with open(path, "r") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)

# inference.rs
replace_in_file("/home/killboxincorporated/aegis-uefi/src/inference.rs", [
    ("use memmap2::Mmap;", ""),
    ("use safetensors::SafeTensors;", "use crate::model::SafeTensors;"),
    ("pub fn new(embeddings_path: &str, model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {", "pub fn new(embeddings_bytes: &'a [u8], model_bytes: &'a [u8]) -> Result<Self, String> {"),
    ("let m_file = std::fs::File::open(model_path)?;", ""),
    ("let m_mmap = unsafe { Mmap::map(&m_file)? };", ""),
    ("let m_mmap_ptr: &[u8] = unsafe { std::slice::from_raw_parts(m_mmap.as_ptr(), m_mmap.len()) };", "let m_mmap_ptr = model_bytes;"),
    ("let e_file = std::fs::File::open(embeddings_path)?;", ""),
    ("let e_mmap = unsafe { Mmap::map(&e_file)? };", ""),
    ("let e_mmap_ptr: &[u8] = unsafe { std::slice::from_raw_parts(e_mmap.as_ptr(), e_mmap.len()) };", "let e_mmap_ptr = embeddings_bytes;"),
    ("pub struct TernaryInferenceEngine<'a>", "pub struct TernaryInferenceEngine<'a>"),
    ("SafeTensors::deserialize(m_mmap_ptr)?", "SafeTensors::deserialize(m_mmap_ptr)?"),
])

# tokenizer.rs
replace_in_file("/home/killboxincorporated/aegis-uefi/src/tokenizer.rs", [
    ("pub fn new(vocab_path: &str) -> Result<Self, Box<dyn std::error::Error>> {", "pub fn new(vocab_str: &str) -> Result<Self, String> {"),
    ("let vocab_str = fs::read_to_string(vocab_path)?;", ""),
    ("serde_json::from_str(&vocab_str)?", "serde_json::from_str(vocab_str).map_err(|e| format!(\"Tokenizer error: {}\", e))?"),
])

# Cargo.toml - enable uefi alloc
replace_in_file("/home/killboxincorporated/aegis-uefi/Cargo.toml", [
    ("uefi = \"0.38.0\"", "uefi = { version = \"0.38.0\", features = [\"alloc\"] }"),
])

# main.rs - add alloc setup
replace_in_file("/home/killboxincorporated/aegis-uefi/src/main.rs", [
    ("use core::panic::PanicInfo;", "#![no_std]\n#![no_main]\nextern crate alloc;\nuse core::panic::PanicInfo;"),
])

print("Fixed rust errors")
