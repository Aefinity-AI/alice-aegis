import sys
from safetensors import safe_open
from safetensors.numpy import save_file

def main():
    print("Loading safetensors...")
    tensors = {}
    with safe_open("/home/killboxincorporated/aegis-linux/aegis_model.safetensors", framework="numpy", device="cpu") as f:
        for k in f.keys():
            if k != "model.embed_tokens.weight":
                tensors[k] = f.get_tensor(k)
    
    print("Saving pruned safetensors...")
    save_file(tensors, "/home/killboxincorporated/aegis_pruned_model.safetensors")
    print("Done!")

if __name__ == "__main__":
    main()
