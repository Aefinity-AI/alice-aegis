import os
import math
import struct
import argparse
import torch
from safetensors import safe_open
from tqdm import tqdm

PHI = 1.618033988749895

def generate_fibonacci_set(max_val):
    fib_set = set()
    a, b = 0, 1
    while a <= max_val:
        fib_set.add(a)
        a, b = b, a + b
    return fib_set

def get_fibonacci_threshold(dist, fib_set, tau_base, is_qk_proj):
    if dist in fib_set:
        if is_qk_proj:
            return tau_base / (PHI * PHI * PHI)
        else:
            return tau_base / (PHI * PHI)
    else:
        return tau_base * PHI

def pack_to_aegis_blocks(pos_map_bits, neg_map_bits, num_blocks):
    """
    Converts binary arrays of 1s and 0s into packed 64-bit unsigned integers.
    pos_map_bits and neg_map_bits are flat boolean/int tensors of length (num_blocks * 64).
    """
    pos_blocks = []
    neg_blocks = []
    
    # We do this in Python for the MVP. Vectorized PyTorch bitwise shifts could be faster.
    pos_list = pos_map_bits.tolist()
    neg_list = neg_map_bits.tolist()
    
    for b in range(num_blocks):
        start = b * 64
        end = start + 64
        p_chunk = pos_list[start:end]
        n_chunk = neg_list[start:end]
        
        p_val = 0
        n_val = 0
        for i in range(64):
            if i < len(p_chunk) and p_chunk[i]:
                p_val |= (1 << i)
            if i < len(n_chunk) and n_chunk[i]:
                n_val |= (1 << i)
        
        pos_blocks.append(p_val)
        neg_blocks.append(n_val)
        
    return pos_blocks, neg_blocks

@torch.no_grad()
def quantize_layer_hessian(weight, inv_hessian, layer_name):
    """
    The core Aegis Intelligent Transmuter logic.
    weight: [rows, cols] torch.Tensor
    inv_hessian: [cols, cols] torch.Tensor
    """
    rows, cols = weight.shape
    num_elements = rows * cols
    num_blocks = (num_elements + 63) // 64
    
    fib_set = generate_fibonacci_set(max(rows, cols))
    
    mean_mag = weight.abs().mean().item()
    tau_base = mean_mag * PHI
    is_qk_proj = "q_proj" in layer_name or "k_proj" in layer_name
    
    pos_map_bits = torch.zeros(num_blocks * 64, dtype=torch.bool, device=weight.device)
    neg_map_bits = torch.zeros(num_blocks * 64, dtype=torch.bool, device=weight.device)
    
    active_count = 0
    active_sum = 0.0
    
    # This loop is Python-heavy, but necessary for the row-wise causality of SparseGPT.
    # We vectorize the column update within each row.
    for r in tqdm(range(rows), desc=f"Transmuting {layer_name}", leave=False):
        row_offset = r * cols
        
        for c in range(cols):
            idx = row_offset + c
            w_val = weight[r, c].item()
            dist = abs(r - c)
            
            tau_eff = get_fibonacci_threshold(dist, fib_set, tau_base, is_qk_proj)
            
            q_val = 0.0
            if abs(w_val) >= tau_eff:
                if w_val > 0.0:
                    pos_map_bits[idx] = True
                    q_val = 1.0
                else:
                    neg_map_bits[idx] = True
                    q_val = -1.0
                active_count += 1
                active_sum += abs(w_val)
                
            error = q_val - w_val
            
            h_diag = inv_hessian[c, c].item()
            if abs(h_diag) > 1e-9 and c + 1 < cols:
                # Vectorized backprop of error to remaining elements in the row
                weight_adjust = (error / h_diag) * inv_hessian[c, c+1:]
                weight[r, c+1:] -= weight_adjust
                
    scale = 1.0
    if active_count > 0:
        base_scale = active_sum / active_count
        scale = base_scale if is_qk_proj else base_scale * 1.41421356
        
    pos_blocks, neg_blocks = pack_to_aegis_blocks(pos_map_bits, neg_map_bits, num_blocks)
    return pos_blocks, neg_blocks, scale

def write_v9_header(f, num_blocks, scale):
    """
    Writes the strict V9 103-byte header + 64-byte padding.
    """
    magic = b"AEGIS_09"
    version = 0x0900
    
    # Placeholder header generation - we match the Rust struct layout
    # magic(8), version(2), blocks(8), scale(4), pad(81) = 103 bytes
    f.write(magic)
    f.write(struct.pack("<H", version))
    f.write(struct.pack("<Q", num_blocks))
    f.write(struct.pack("<f", scale))
    f.write(b'\x00' * 81) # pad to 103
    
    # 64-byte alignment padding required by bare-metal CPU
    f.write(b'\x00' * 64)

def run_transmuter(input_safetensors, output_aegis):
    print(f"Starting Out-Of-Core Transmutation: {input_safetensors} -> {output_aegis}")
    
    with safe_open(input_safetensors, framework="pt", device="cpu") as f:
        tensor_keys = f.keys()
        print(f"Found {len(tensor_keys)} tensors in model.")
        
        with open(output_aegis, "wb") as out_f:
            for key in tensor_keys:
                # 1. Stream the tensor out of core (layer by layer)
                weight = f.get_tensor(key)
                
                if len(weight.shape) != 2:
                    print(f"Skipping 1D tensor {key} (likely bias or layer norm)")
                    continue
                    
                rows, cols = weight.shape
                
                # 2. Setup Dummy Hessian (Identity matrix) for baseline quantization
                # In a full run, this would be computed via forward pass calibration.
                inv_hessian = torch.eye(cols, device="cpu", dtype=torch.float32)
                
                # 3. Transmute
                pos_blocks, neg_blocks, scale = quantize_layer_hessian(weight, inv_hessian, key)
                
                # 4. Stream to disk (V9 Conformant)
                write_v9_header(out_f, len(pos_blocks), scale)
                
                # Write dual bitmaps
                for p, n in zip(pos_blocks, neg_blocks):
                    out_f.write(struct.pack("<Q", p))
                    out_f.write(struct.pack("<Q", n))
                
                # Clear memory
                del weight
                del inv_hessian
                
    print("\nTransmutation Complete. Zero-Skip Blocks written successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aegis Intelligent Out-of-Core Transmuter")
    parser.add_argument("--input", type=str, required=True, help="Path to input .safetensors")
    parser.add_argument("--output", type=str, required=True, help="Path to output .aegis binary")
    args = parser.parse_args()
    run_transmuter(args.input, args.output)
