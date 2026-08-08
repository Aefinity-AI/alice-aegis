import numpy as np
from safetensors import safe_open

def check():
    with safe_open("/home/killboxincorporated/model.safetensors", framework="np", device="cpu") as f:
        tensor = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
        unique, counts = np.unique(tensor, return_counts=True)
        print("Unique bytes:", unique[:10], "...")
        
        # Check 2-bit chunk distribution
        chunks = np.zeros(4, dtype=np.int64)
        for i in range(4):
            bits = (tensor >> (i * 2)) & 0b11
            u, c = np.unique(bits, return_counts=True)
            for val, count in zip(u, c):
                chunks[val] += count
        
        print("2-bit chunk distribution:", chunks)

if __name__ == "__main__":
    check()
