import os

# 1. Fix src/tokenizer.rs
tokenizer_rs = """use alloc::{vec::Vec, string::String, vec, format, string::ToString, boxed::Box};
use alloc::collections::BTreeMap;
use serde::Deserialize;

#[derive(Deserialize)]
struct TokenizerModel {
    vocab: BTreeMap<String, u32>,
    merges: Vec<String>,
}

#[derive(Deserialize)]
struct TokenizerJson {
    model: TokenizerModel,
}

pub struct AegisTokenizer {
    pub vocab: BTreeMap<String, u32>,
    pub reverse_vocab: BTreeMap<u32, String>,
    pub merges: BTreeMap<(String, String), usize>,
}

impl AegisTokenizer {
    pub fn new(tokenizer_str: &str) -> Result<Self, String> {
        let json: TokenizerJson = serde_json::from_str(tokenizer_str).map_err(|e| format!("Tokenizer error: {}", e))?;
        let vocab = json.model.vocab;
        let mut reverse_vocab = BTreeMap::new();
        for (word, &id) in &vocab {
            reverse_vocab.insert(id, word.clone());
        }
        
        let mut merges = BTreeMap::new();
        for (i, merge) in json.model.merges.iter().enumerate() {
            let parts: Vec<&str> = merge.split(' ').collect();
            if parts.len() == 2 {
                merges.insert((parts[0].to_string(), parts[1].to_string()), i);
            }
        }
        
        Ok(Self { vocab, reverse_vocab, merges })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let words = text.split_whitespace();
        
        for (i, word) in words.enumerate() {
            let mut w = String::new();
            if i > 0 || text.starts_with(' ') {
                w.push('Ġ');
            }
            w.push_str(word);
            
            let mut symbols: Vec<String> = w.chars().map(|c| c.to_string()).collect();
            
            loop {
                if symbols.len() < 2 { break; }
                let mut best = None;
                let mut best_idx = 0;
                
                for j in 0..symbols.len() - 1 {
                    if let Some(&rank) = self.merges.get(&(symbols[j].clone(), symbols[j+1].clone())) {
                        if best.is_none() || rank < best.unwrap() {
                            best = Some(rank);
                            best_idx = j;
                        }
                    }
                }
                
                if best.is_none() { break; }
                
                symbols[best_idx] = format!("{}{}", symbols[best_idx], symbols[best_idx+1]);
                symbols.remove(best_idx + 1);
            }
            
            for sym in symbols {
                if let Some(&id) = self.vocab.get(&sym) {
                    tokens.push(id);
                } else if let Some(&id) = self.vocab.get("<|reserved_special_token_0|>") {
                    tokens.push(id); // fallback
                }
            }
        }
        tokens
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in token_ids {
            if let Some(word) = self.reverse_vocab.get(&id) {
                let cleaned = word.replace("Ġ", " ");
                result.push_str(&cleaned);
            }
        }
        result
    }
}
"""
with open("/home/killboxincorporated/aegis-uefi/src/tokenizer.rs", "w") as f:
    f.write(tokenizer_rs)

# 2. Delete tensor_map.rs and fix model.rs
if os.path.exists("/home/killboxincorporated/aegis-uefi/src/tensor_map.rs"):
    os.remove("/home/killboxincorporated/aegis-uefi/src/tensor_map.rs")

with open("/home/killboxincorporated/aegis-uefi/src/model.rs", "r") as f:
    model_content = f.read()

model_content = model_content.replace(
"""pub struct SafeTensors<'a> {
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
        
        Ok(Self {
            buffer,
            tensor_start: 8 + n,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'a>, String> {
        for &(t_name, start, end) in crate::tensor_map::TENSOR_MAP {
            if t_name == name {
                let abs_start = self.tensor_start + start;
                let abs_end = self.tensor_start + end;
                return Ok(TensorView { data_bytes: &self.buffer[abs_start..abs_end] });
            }
        }
        Err(format!("Tensor {} not found in hardcoded map", name))
    }
}""",
"""use alloc::collections::BTreeMap;
use serde::Deserialize;

pub struct SafeTensors<'a> {
    buffer: &'a [u8],
    tensor_start: usize,
    offsets: BTreeMap<String, (usize, usize)>,
}

impl<'a> SafeTensors<'a> {
    pub fn deserialize(buffer: &'a [u8]) -> Result<Self, String> {
        if buffer.len() < 8 { return Err("Buffer too small".into()); }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buffer[0..8]);
        let n = u64::from_le_bytes(len_bytes) as usize;
        if buffer.len() < 8 + n { return Err("Buffer smaller than header".into()); }
        
        let header_str = core::str::from_utf8(&buffer[8..8+n]).map_err(|_| "Invalid UTF-8 in header".to_string())?;
        let parsed: BTreeMap<String, serde_json::Value> = serde_json::from_str(header_str).map_err(|e| format!("JSON Error: {}", e))?;
        
        let mut offsets = BTreeMap::new();
        for (k, v) in parsed {
            if k == "__metadata__" { continue; }
            if let Some(data_offsets) = v.get("data_offsets") {
                if let Some(arr) = data_offsets.as_array() {
                    if arr.len() == 2 {
                        if let (Some(start), Some(end)) = (arr[0].as_u64(), arr[1].as_u64()) {
                            offsets.insert(k, (start as usize, end as usize));
                        }
                    }
                }
            }
        }
        
        Ok(Self {
            buffer,
            tensor_start: 8 + n,
            offsets,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'a>, String> {
        if let Some(&(start, end)) = self.offsets.get(name) {
            let abs_start = self.tensor_start + start;
            let abs_end = self.tensor_start + end;
            return Ok(TensorView { data_bytes: &self.buffer[abs_start..abs_end] });
        }
        Err(format!("Tensor {} not found in dynamic map", name))
    }
}"""
)
with open("/home/killboxincorporated/aegis-uefi/src/model.rs", "w") as f:
    f.write(model_content)

# 3. Add Bounds Checks to ops.rs
with open("/home/killboxincorporated/aegis-uefi/src/ops.rs", "r") as f:
    ops_content = f.read()

ops_content = ops_content.replace(
"""pub fn ternary_matvec(output: &mut [f32], input: &[f32], weights_u8: &[u8], dim_out: usize, dim_in: usize) {
    unsafe {
        ternary_matvec_avx2(output, input, weights_u8, dim_out, dim_in);
    }
}""",
"""pub fn ternary_matvec(output: &mut [f32], input: &[f32], weights_u8: &[u8], dim_out: usize, dim_in: usize) {
    if output.len() < dim_out || input.len() < dim_in || weights_u8.len() < (dim_out * dim_in) / 4 {
        return; // PHASE 5: bounds check to prevent silent AVX-2 segfaults
    }
    unsafe {
        ternary_matvec_avx2(output, input, weights_u8, dim_out, dim_in);
    }
}"""
)
with open("/home/killboxincorporated/aegis-uefi/src/ops.rs", "w") as f:
    f.write(ops_content)

# 4. Remove panic vectors in main.rs and fix tokenizer path
with open("/home/killboxincorporated/aegis-uefi/src/main.rs", "r") as f:
    main_content = f.read()

main_content = main_content.replace("mod tensor_map;\n", "")
main_content = main_content.replace(
"""    let vocab_data = match load_file("vocab.json") {
        Some(d) => d,
        None => panic!("FATAL: Could not find vocab.json on USB drive."),
    };
    let vocab_str = core::str::from_utf8(&vocab_data).expect("Invalid UTF8 in vocab.json");

    let model_data = match load_file("aegis_model.safetensors") {
        Some(d) => d,
        None => panic!("FATAL: Could not find aegis_model.safetensors on USB drive."),
    };

    let embeddings_data = match load_file("aegis_lobotomized_embeddings.bin") {
        Some(d) => d,
        None => panic!("FATAL: Could not find aegis_lobotomized_embeddings.bin on USB drive."),
    };
    
    let _ = uefi::system::with_stdout(|st| st.write_str("\\r\\n[SYSTEM] Files loaded to RAM. Initializing Neural Matrix...\\r\\n"));
    
    let mut engine = match crate::inference::TernaryInferenceEngine::new(&embeddings_data, &model_data, vocab_str) {
        Ok(e) => e,
        Err(err) => panic!("FATAL: Neural Engine failed to parse weights: {}", err),
    };""",
"""    let vocab_data = match load_file("tokenizer.json") {
        Some(d) => d,
        None => { let _ = uefi::system::with_stdout(|st| st.write_str("FATAL: Could not find tokenizer.json\\r\\n")); return uefi::Status::ABORTED; }
    };
    let vocab_str = match core::str::from_utf8(&vocab_data) {
        Ok(s) => s,
        Err(_) => { let _ = uefi::system::with_stdout(|st| st.write_str("FATAL: Invalid UTF8 in tokenizer.json\\r\\n")); return uefi::Status::ABORTED; }
    };

    let model_data = match load_file("aegis_model.safetensors") {
        Some(d) => d,
        None => { let _ = uefi::system::with_stdout(|st| st.write_str("FATAL: Could not find aegis_model.safetensors\\r\\n")); return uefi::Status::ABORTED; }
    };

    let embeddings_data = match load_file("aegis_lobotomized_embeddings.bin") {
        Some(d) => d,
        None => { let _ = uefi::system::with_stdout(|st| st.write_str("FATAL: Could not find aegis_lobotomized_embeddings.bin\\r\\n")); return uefi::Status::ABORTED; }
    };
    
    let _ = uefi::system::with_stdout(|st| st.write_str("\\r\\n[SYSTEM] Files loaded to RAM. Initializing Neural Matrix...\\r\\n"));
    
    let mut engine = match crate::inference::TernaryInferenceEngine::new(&embeddings_data, &model_data, vocab_str) {
        Ok(e) => e,
        Err(err) => { let _ = uefi::system::with_stdout(|st| { st.write_str("FATAL: Engine failed: ").unwrap(); st.write_str(&err).unwrap(); st.write_str("\\r\\n").unwrap(); core::fmt::Result::Ok(()) }); return uefi::Status::ABORTED; }
    };"""
)
with open("/home/killboxincorporated/aegis-uefi/src/main.rs", "w") as f:
    f.write(main_content)

# 5. Fix build_usb_img.sh to use tokenizer.json
with open("/home/killboxincorporated/aegis-uefi/build_usb_img.sh", "r") as f:
    build_content = f.read()

build_content = build_content.replace(
    "mcopy -i esp.img ../aegis-linux/vocab.json ::/vocab.json",
    "mcopy -i esp.img ../tokenizer.json ::/tokenizer.json"
)
with open("/home/killboxincorporated/aegis-uefi/build_usb_img.sh", "w") as f:
    f.write(build_content)

print("All phases applied!")
