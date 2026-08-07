#!/bin/bash
set -e

# Copy the fixed logic from UEFI to Linux
cp /home/killboxincorporated/aegis-uefi/src/attention.rs /home/killboxincorporated/aegis-linux/src/
cp /home/killboxincorporated/aegis-uefi/src/inference.rs /home/killboxincorporated/aegis-linux/src/
cp /home/killboxincorporated/aegis-uefi/src/kvcache.rs /home/killboxincorporated/aegis-linux/src/
cp /home/killboxincorporated/aegis-uefi/src/model.rs /home/killboxincorporated/aegis-linux/src/
cp /home/killboxincorporated/aegis-uefi/src/ops.rs /home/killboxincorporated/aegis-linux/src/
cp /home/killboxincorporated/aegis-uefi/src/sampler.rs /home/killboxincorporated/aegis-linux/src/
cp /home/killboxincorporated/aegis-uefi/src/tokenizer.rs /home/killboxincorporated/aegis-linux/src/
cp /home/killboxincorporated/aegis-uefi/src/arena.rs /home/killboxincorporated/aegis-linux/src/

# Remove tensor_map
rm -f /home/killboxincorporated/aegis-linux/src/tensor_map.rs

# Rewrite Linux main.rs to support the new Engine signature and streaming output
cat << 'MAINEOF' > /home/killboxincorporated/aegis-linux/src/main.rs
mod attention;
mod inference;
mod kvcache;
mod model;
mod ops;
mod sampler;
mod tokenizer;
mod arena;

use std::fs::File;
use std::io::{Read, Write};

fn main() {
    println!("==================================================");
    println!(" A.L.I.C.E. Linux Brain Jar Test");
    println!("==================================================");
    
    let mut f = File::open("/home/killboxincorporated/model.safetensors").unwrap_or_else(|_| File::open("/home/killboxincorporated/aegis_model.safetensors").expect("Could not open model"));
    let mut model_bytes = Vec::new();
    f.read_to_end(&mut model_bytes).expect("Failed to read model bytes");
    
    let mut v = File::open("/home/killboxincorporated/tokenizer.json").expect("Could not open tokenizer.json");
    let mut vocab_str = String::new();
    v.read_to_string(&mut vocab_str).expect("Failed to read vocab string");
    
    let mut e = File::open("/home/killboxincorporated/aegis_lobotomized_embeddings.bin").expect("Could not open embeddings");
    let mut emb_bytes = Vec::new();
    e.read_to_end(&mut emb_bytes).expect("Failed to read embeddings bytes");
    
    println!("Initializing Ternary Inference Engine (BitNet 1.58b 2B-4T)...");
    let mut engine = inference::TernaryInferenceEngine::new(&emb_bytes, &model_bytes, &vocab_str).expect("Failed to init engine");
    
    println!("Engine Online. Brain Jar Test Ready.");
    
    let test_prompt = "The capital of France is";
    println!("\nPrompt: {}", test_prompt);
    
    let response = engine.process_intent(test_prompt, |token| {
        print!("{}", token);
        std::io::stdout().flush().unwrap();
    });
    
    println!("\n\nFinal Full Response: {}", response);
}
MAINEOF

# Let's fix Cargo.toml in aegis-linux if needed (add serde_json if not present)
cd /home/killboxincorporated/aegis-linux
# If serde_json is not in Cargo.toml, add it
if ! grep -q "serde_json" Cargo.toml; then
    echo 'serde_json = { version = "1.0", features = ["alloc"] }' >> Cargo.toml
fi
if ! grep -q "serde =" Cargo.toml; then
    echo 'serde = { version = "1.0", features = ["derive", "alloc"] }' >> Cargo.toml
fi
# Disable no_std in aegis-linux/Cargo.toml if it exists (it's a standard linux bin)
# We can just build it!
cargo build --release
