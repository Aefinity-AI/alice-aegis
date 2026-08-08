import os
import glob
from huggingface_hub import snapshot_download
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# Read from the environment — never hardcode. (A previous revision inlined a
# live token here; it must be treated as compromised and rotated.)
TOKEN = os.environ.get("HF_TOKEN") or exit("Set HF_TOKEN in the environment.")
REPO_ID = "microsoft/bitnet-b1.58-2B-4T"
OUTPUT_FILE = "aegis_model.safetensors"

def quantize_and_pack(weight):
    print(f"  Quantizing shape {weight.shape}...")
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
    print(f"=== Aegis Transmutation Engine ===")
    print(f"Downloading {REPO_ID} (This may take a while)...")
    
    path = snapshot_download(repo_id=REPO_ID, token=TOKEN, local_files_only=False)
    print(f"Download complete! Located at: {path}")
    
    safetensor_files = glob.glob(os.path.join(path, "*.safetensors"))
    if not safetensor_files:
        print("Error: No .safetensors found in downloaded model.")
        return
        
    aegis_tensors = {}
    
    print("Beginning 1.58-bit Transmutation...")
    for file in safetensor_files:
        print(f"Processing {os.path.basename(file)}...")
        with safe_open(file, framework="np", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # We only aggressively quantize the massive Linear layers (q_proj, k_proj, v_proj, o_proj, up_proj, down_proj)
                if "proj" in key or "weight" in key and "norm" not in key and "embed" not in key:
                    packed_tensor = quantize_and_pack(tensor)
                    aegis_tensors[key] = packed_tensor
                else:
                    # Keep embeddings and layernorms as fp32/fp16
                    aegis_tensors[key] = tensor

    print(f"Writing proprietary Aegis format to {OUTPUT_FILE}...")
    save_file(aegis_tensors, OUTPUT_FILE)
    print("Transmutation Complete! The Aegis model is ready for the Unikernel.")

if __name__ == "__main__":
    transmute()
