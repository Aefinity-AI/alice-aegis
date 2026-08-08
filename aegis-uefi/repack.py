import numpy as np
from safetensors.numpy import save_file, load_file

def repack():
    print("Loading 1.8GB model...")
    tensors = load_file("/home/killboxincorporated/test_usb_dir/aegis_model.safetensors")
    packed_tensors = {}
    
    print("Repacking into 4-per-byte...")
    for key, weight_q in tensors.items():
        if ("proj" in key or "weight" in key) and "norm" not in key and "embed" not in key and "scale" not in key:
            print(f"  Repacking {key} {weight_q.shape}")
            flat_w = weight_q.flatten()
            
            # Pad to multiple of 4
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
                
            packed_tensors[key] = packed
        else:
            packed_tensors[key] = weight_q
            
    print("Saving 385MB packed model...")
    save_file(packed_tensors, "/home/killboxincorporated/test_usb_dir/aegis_model.safetensors.packed")
    print("Done!")

if __name__ == "__main__":
    repack()
