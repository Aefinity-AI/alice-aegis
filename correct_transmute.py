import os
import numpy as np
import ml_dtypes
from safetensors import safe_open
from safetensors.numpy import save_file

def unpack_bitnet(packed_tensor, out_features, in_features):
    # packed_tensor is shape [out_features // 4, in_features]
    unpacked = np.zeros((out_features, in_features), dtype=np.int8)
    M = out_features // 4
    for i in range(4):
        bits = (packed_tensor >> (i * 2)) & 0b11
        vals = bits.astype(np.int8) - 1
        unpacked[i * M : (i + 1) * M, :] = vals
    return unpacked


def pack_aegis(unpacked_tensor):
    # unpacked_tensor is shape [out_features, in_features]
    # We pack it to [out_features, in_features // 4] (along in_features)
    out_features, in_features = unpacked_tensor.shape
    packed = np.zeros((out_features, in_features // 4), dtype=np.uint8)
    
    for i in range(4):
        val = unpacked_tensor[:, i::4]
        bits = np.zeros_like(val, dtype=np.uint8)
        bits[val == 1] = 1
        bits[val == -1] = 2
        packed |= (bits << (i * 2))
        
    return packed

def process():
    in_file = "/home/killboxincorporated/model.safetensors"
    out_file = "/home/killboxincorporated/aegis-linux/aegis_model.safetensors"
    
    aegis_tensors = {}
    
    with safe_open(in_file, framework="np", device="cpu") as f:
        keys = f.keys()
        for i, key in enumerate(keys):
            print(f"[{i+1}/{len(keys)}] Processing {key}...")
            tensor = f.get_tensor(key)
            
            if "weight" in key and "norm" not in key and "embed" not in key and tensor.dtype == np.uint8:
                # Linear layer weight that is quantized
                packed_shape = tensor.shape
                out_feat_div_4, in_feat = packed_shape
                out_feat = out_feat_div_4 * 4
                
                unpacked = unpack_bitnet(tensor, out_feat, in_feat)
                repacked = pack_aegis(unpacked)
                
                # Flatten for safetensors if needed, but keeping shape is fine
                aegis_tensors[key] = repacked.flatten()
            else:
                if tensor.dtype == np.float16 or str(tensor.dtype) == 'bfloat16':
                    pass
                aegis_tensors[key] = tensor.astype(np.float32)

    save_file(aegis_tensors, out_file)
    print("Done!")

if __name__ == "__main__":
    process()
