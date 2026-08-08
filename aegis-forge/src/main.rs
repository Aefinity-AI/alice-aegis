use aegis_forge::vocab_stripper::strip_non_ascii_vocab;
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ------------------------------------------------------------------
    // DO NOT USE — this tool is out of sync with the shipped artifacts and
    // will silently produce a broken tokenizer. Three defects:
    //   1. It never writes the BPE merges trailer, so the engine loads an
    //      empty merge map and tokenizes character-by-character.
    //   2. It only reads `model.vocab`, missing the 256 Llama-3 special
    //      tokens (they live in tokenizer.json `added_tokens`), so the model
    //      can never emit <|eot_id|> and always runs to the token cap.
    //   3. Its id assignment interleaves kept "<"-prefixed tokens, so
    //      vocab.bin ids no longer line up with embed.bin row order.
    // Use `regen_vocab_embed.py` (same directory) until this is ported.
    // ------------------------------------------------------------------
    if std::env::var("AEGIS_FORGE_I_KNOW_ITS_BROKEN").is_err() {
        eprintln!("REFUSING TO RUN: aegis-forge is out of sync with the engine's");
        eprintln!("vocab.bin format (no merges, no added_tokens, divergent id space).");
        eprintln!("Regenerating artifacts with it WILL break inference.");
        eprintln!();
        eprintln!("Use instead:  python3 aegis-forge/regen_vocab_embed.py");
        eprintln!("Override (for porting work only): AEGIS_FORGE_I_KNOW_ITS_BROKEN=1");
        std::process::exit(1);
    }

    println!("Starting Aegis Vocabulary Pruning & Embedding Slicer Pipeline...");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("Usage: aegis-forge <tokenizer.json> <config.json> <model.safetensors>".into());
    }
    let tokenizer_path = &args[1];
    let config_path = &args[2];
    let model_path = &args[3];

    let tokenizer_str = fs::read_to_string(tokenizer_path)?;
    let tokenizer_json: Value = serde_json::from_str(&tokenizer_str)?;

    let original_vocab_map = tokenizer_json
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .ok_or("Missing model.vocab in tokenizer.json")?;

    let mut original_vocab = HashMap::new();
    for (k, v) in original_vocab_map {
        original_vocab.insert(k.clone(), v.as_u64().ok_or("Invalid vocab id")? as u32);
    }

    let original_size = original_vocab.len();
    println!("Original Vocabulary Size: {}", original_size);

    // 2. Strip Vocab
    let (new_vocab, keep_indices) = strip_non_ascii_vocab(&original_vocab);
    println!("Pruned Vocabulary Size: {}", new_vocab.len());
    println!("Removed {} tokens.", original_size - new_vocab.len());

    // Construct the proper JSON structure expected by the Unikernel (keeping it just in case)
    let mut new_tokenizer_json = tokenizer_json.clone();
    new_tokenizer_json["model"]["vocab"] = serde_json::to_value(&new_vocab)?;
    fs::write(
        "aegis_pruned_vocab.json",
        serde_json::to_string_pretty(&new_tokenizer_json)?,
    )?;
    println!("Saved aegis_pruned_vocab.json");

    // Construct the flat binary Vocab (The Phase 2 Fix for Heap Fragmentation)
    let mut vocab_bin = Vec::new();
    vocab_bin.extend_from_slice(&0x564F4341u32.to_le_bytes()); // Magic "VOCA"
    vocab_bin.extend_from_slice(&(new_vocab.len() as u32).to_le_bytes());

    let mut reverse_vocab: Vec<(&String, &u32)> = new_vocab.iter().collect();
    reverse_vocab.sort_by_key(|a| a.1);

    for (token, _) in reverse_vocab {
        let bytes = token.as_bytes();
        vocab_bin.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        vocab_bin.extend_from_slice(bytes);
    }
    fs::write("vocab.bin", &vocab_bin)?;
    println!("Saved vocab.bin ({} bytes)", vocab_bin.len());

    // Update config.json with new vocab_size
    let config_str = fs::read_to_string(config_path)?;
    let mut config_json: Value = serde_json::from_str(&config_str)?;
    config_json["vocab_size"] = serde_json::json!(new_vocab.len());
    fs::write(
        "aegis_pruned_config.json",
        serde_json::to_string_pretty(&config_json)?,
    )?;
    println!("Saved aegis_pruned_config.json");

    // 3. Open Safetensors Model
    let model_data = fs::read(model_path)?;
    let tensors = SafeTensors::deserialize(&model_data)?;

    // Example: Slice model.embed_tokens.weight if it exists
    if let Ok(embed_tensor) = tensors.tensor("model.embed_tokens.weight") {
        let shape = embed_tensor.shape();
        let hidden_dim = shape[1];

        println!("Found model.embed_tokens.weight with shape: {:?}", shape);

        // We assume bf16 or f16/f32. For prototype, just read raw bytes to slice them.
        // We'll treat it as f32 for the slicer function API (as per manual), but it's likely bf16/f16.
        // In a real pipeline, we'd cast properly. The manual specifies f32 slice:
        // slice_embeddings(raw_embeddings: &[f32], hidden_dim: usize, keep_indices: &[u32]) -> Vec<f32>

        let byte_data = embed_tensor.data();

        // If it's f16/bf16, each element is 2 bytes. If it's f32, 4 bytes.
        let bytes_per_element = byte_data.len() / (shape[0] * shape[1]);
        println!("Bytes per element in embedding: {}", bytes_per_element);

        // Instead of true f32 casting which depends on dtype, we'll just slice the raw bytes directly to be safe.
        let mut sliced_bytes = Vec::with_capacity(keep_indices.len() * hidden_dim * 2);

        for &old_id in &keep_indices {
            let start_byte = (old_id as usize) * hidden_dim * bytes_per_element;

            for i in 0..hidden_dim {
                let elem_start = start_byte + i * bytes_per_element;
                if bytes_per_element == 4 {
                    // Extract top 16 bits of F32 to convert to BF16 (little endian)
                    // F32 in memory (LE): [b0, b1, b2, b3]. Top 16 bits are [b2, b3]
                    sliced_bytes.push(byte_data[elem_start + 2]);
                    sliced_bytes.push(byte_data[elem_start + 3]);
                } else if bytes_per_element == 2 {
                    // Already BF16/F16
                    sliced_bytes.push(byte_data[elem_start]);
                    sliced_bytes.push(byte_data[elem_start + 1]);
                } else {
                    panic!(
                        "Unsupported embedding bytes_per_element: {}",
                        bytes_per_element
                    );
                }
            }
        }

        fs::write("embed.bin", &sliced_bytes)?;
        println!("Saved embed.bin ({} bytes)", sliced_bytes.len());
    } else {
        println!("Warning: model.embed_tokens.weight not found in safetensors.");
    }

    println!("Aegis Vocabulary Pruning & Embedding Slicer Pipeline Completed Successfully.");
    Ok(())
}
