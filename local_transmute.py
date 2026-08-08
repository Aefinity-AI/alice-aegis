import os
import glob
import numpy as np
import ml_dtypes
from safetensors import safe_open
from safetensors.numpy import save_file

LOCAL_PATH = "/home/killboxincorporated"
OUTPUT_FILE = "/home/killboxincorporated/aegis-linux/aegis_model.safetensors"

def quantize_and_pack(weight):
    weight = weight.astype(np.float32)
    # 1. Calculate absolute mean for scaling
    scale = np.mean(np.abs(weight))
    
    # 2. Scale, clamp, and round to -1, 0, 1
    weight_scaled = np.clip(weight / (scale + 1e-8), -1.0, 1.0)
    weight_q = np.round(weight_scaled).astype(np.int8)
    
    # 3. Pack 4 ternary values into 1 byte (uint8)
    # Rust AVX-2 expects: 0 -> 0b00, 1 -> 0b01, -1 -> 0b10
    flat_w = weight_q.flatten()
    
    # Ensure divisible by 4
    pad_len = (4 - (len(flat_w) % 4)) % 4
    if pad_len > 0:
        flat_w = np.append(flat_w, np.zeros(pad_len, dtype=np.int8))
        
    packed = np.zeros(len(flat_w) // 4, dtype=np.uint8)
    
    for i in range(4):
        val = flat_w[i::4]
        bits = np.zeros_like(val, dtype=np.uint8)
        bits[val == 1] = 1
        bits[val == -1] = 2
        packed |= (bits << (i * 2))
        
    return packed

def transmute():
    print(f"=== Aegis Local Transmutation Engine ===")
    
    file = os.path.join(LOCAL_PATH, "model.safetensors")
    if not os.path.exists(file):
        print("Error: model.safetensors not found in home directory.")
        return
        
    aegis_tensors = {}
    
    print(f"Processing {file}...")
    print("Beginning 1.58-bit Transmutation. This will take a few minutes as we compress 1.2 GB of tensors...")
    
    with safe_open(file, framework="np", device="cpu") as f:
        keys = f.keys()
        for i, key in enumerate(keys):
            print(f"  [{i+1}/{len(keys)}] Quantizing {key}...")
            tensor = f.get_tensor(key)
            
            if "embed_tokens" in key or "lm_head" in key:
                print(f"Skipping {key} (handled externally)")
                continue
                
            # Aggressively quantize the massive Linear layers
            if "weight_scale" not in key and ("proj" in key or ("weight" in key and "norm" not in key and "embed" not in key)):
                packed_tensor = quantize_and_pack(tensor)
                aegis_tensors[key] = packed_tensor
            else:
                # Keep layernorms as they are
                aegis_tensors[key] = tensor.astype(np.float32)

    print(f"\nWriting proprietary Aegis format to {OUTPUT_FILE}...")
    save_file(aegis_tensors, OUTPUT_FILE)
    print("Transmutation Complete! The Aegis model is ready in aegis-linux/")

if __name__ == "__main__":
    transmute()
